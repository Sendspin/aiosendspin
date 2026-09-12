"""Tests for SendspinConnection writer task behavior."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Never
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiosendspin.models import pack_binary_header_raw
from aiosendspin.models.core import (
    ServerTimeMessage,
    ServerTimePayload,
    StreamEndMessage,
    StreamEndPayload,
    StreamStartMessage,
    StreamStartPayload,
)
from aiosendspin.models.player import StreamStartPlayer
from aiosendspin.models.types import AudioCodec, BinaryMessageType
from aiosendspin.server.audio import BufferTracker
from aiosendspin.server.clock import LoopClock, ManualClock
from aiosendspin.server.connection import (
    MAX_PENDING_MSG,
    SendspinConnection,
    _BinaryData,
    _RoleQueueEntry,
)
from aiosendspin.server.roles.base import BinaryHandling
from aiosendspin.server.roles.player.v1 import PlayerV1Role


@dataclass(slots=True)
class _DummyServer:
    loop: asyncio.AbstractEventLoop
    clock: Any
    id: str = "srv"
    name: str = "server"

    def get_or_create_client(self, client_id: str) -> Never:
        raise AssertionError(f"unexpected get_or_create_client({client_id}) in this test")

    def is_external_player(self, client_id: str) -> bool:  # noqa: ARG002
        return False


def _make_player_client_stub() -> MagicMock:
    client = MagicMock()
    state_store: dict[str, object] = {}

    def get_or_create_role_state(family: str, cls: type[object]) -> object:
        state_store.setdefault(family, cls())
        return state_store[family]

    client.get_or_create_role_state.side_effect = get_or_create_role_state
    client.info = MagicMock()
    client.info.player_support = None
    client.group = MagicMock()
    client._server = MagicMock()  # noqa: SLF001
    client._logger = MagicMock()  # noqa: SLF001
    client.client_id = "test-player"
    client.connection = None
    client.send_role_message = MagicMock()
    return client


def test_binary_data_supports_buffer_registration_metadata() -> None:
    """_BinaryData should optionally carry buffer registration info."""
    simple = _BinaryData(data=b"test", message_type=4)
    assert simple.buffer_end_time_us is None
    assert simple.buffer_byte_count is None

    with_meta = _BinaryData(
        data=b"test",
        message_type=4,
        buffer_end_time_us=1_000_000,
        buffer_byte_count=1234,
    )
    assert with_meta.buffer_end_time_us == 1_000_000
    assert with_meta.buffer_byte_count == 1234

    entry = _RoleQueueEntry(epoch=1, timestamp_us=0, binary=with_meta)
    assert entry.binary is not None
    assert entry.binary.buffer_end_time_us == 1_000_000


@pytest.mark.asyncio
async def test_send_binary_accepts_buffer_metadata() -> None:
    """send_binary should accept optional buffer registration parameters."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))

    wsock = MagicMock()
    wsock.closed = False

    conn = SendspinConnection(server, wsock_client=wsock)
    conn._transport = wsock  # noqa: SLF001

    conn.send_binary(
        b"audio_data",
        role="player",
        timestamp_us=0,
        message_type=BinaryMessageType.AUDIO_CHUNK.value,
        buffer_end_time_us=1_000_000,
        buffer_byte_count=100,
    )

    # Access the per-role queue
    role_queue = conn._role_queues.get("player")  # noqa: SLF001
    assert role_queue is not None
    assert len(role_queue) == 1
    _, _, entry = role_queue[0]
    assert entry.binary is not None
    assert entry.binary.buffer_end_time_us == 1_000_000
    assert entry.binary.buffer_byte_count == 100


@pytest.mark.asyncio
async def test_writer_registers_buffer_after_send() -> None:
    """Writer should call role's buffer_tracker.register() after successful send_bytes."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))

    wsock = MagicMock()
    wsock.closed = False
    wsock.send_str = AsyncMock()
    wsock.send_bytes = AsyncMock()

    conn = SendspinConnection(server, wsock_client=wsock)
    conn._transport = wsock  # noqa: SLF001
    await conn._setup_connection()  # noqa: SLF001
    conn._writer_task = asyncio.create_task(conn._writer())  # noqa: SLF001

    # Mock a role that handles AUDIO_CHUNK with buffer tracking
    mock_role = MagicMock()
    mock_buffer_tracker = MagicMock()
    mock_buffer_tracker.time_until_duration_capacity.return_value = 0
    mock_buffer_tracker.time_until_ready.return_value = 0
    mock_role.get_buffer_tracker.return_value = mock_buffer_tracker
    mock_role._stream_start_time_us = None  # noqa: SLF001
    mock_role._last_late_log_s = 0.0  # noqa: SLF001
    mock_role._late_skips_since_log = 0  # noqa: SLF001
    mock_role.get_binary_handling.return_value = BinaryHandling(
        drop_late=False,
        buffer_track=True,
    )

    mock_client = MagicMock()
    binary_handling = BinaryHandling(drop_late=False, buffer_track=True)
    mock_client.get_binary_handling_cached.return_value = (binary_handling, mock_role)
    conn._client = mock_client  # noqa: SLF001

    payload = b"audio_data"
    message_type = BinaryMessageType.AUDIO_CHUNK.value
    packed = pack_binary_header_raw(message_type, 0) + payload
    conn.send_binary(
        packed,
        role="player",
        timestamp_us=0,
        message_type=message_type,
        buffer_end_time_us=1_000_000,
        buffer_byte_count=100,
        duration_us=50_000,
    )

    for _ in range(50):
        if wsock.send_bytes.called:
            break
        await asyncio.sleep(0)

    assert wsock.send_bytes.call_count == 1
    mock_buffer_tracker.time_until_ready.assert_called()
    mock_buffer_tracker.register.assert_called_once_with(1_000_000, 100, 50_000)

    await conn.disconnect(retry_connection=False)


@pytest.mark.asyncio
async def test_writer_does_not_register_without_metadata() -> None:
    """Writer should not call register() when metadata is None."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))

    wsock = MagicMock()
    wsock.closed = False
    wsock.send_str = AsyncMock()
    wsock.send_bytes = AsyncMock()

    conn = SendspinConnection(server, wsock_client=wsock)
    conn._transport = wsock  # noqa: SLF001
    await conn._setup_connection()  # noqa: SLF001
    conn._writer_task = asyncio.create_task(conn._writer())  # noqa: SLF001

    # Mock a role that handles AUDIO_CHUNK with buffer tracking
    mock_role = MagicMock()
    mock_buffer_tracker = MagicMock()
    mock_buffer_tracker.time_until_duration_capacity.return_value = 0
    mock_buffer_tracker.time_until_ready.return_value = 0
    mock_role.get_buffer_tracker.return_value = mock_buffer_tracker
    mock_role._stream_start_time_us = None  # noqa: SLF001
    mock_role._last_late_log_s = 0.0  # noqa: SLF001
    mock_role._late_skips_since_log = 0  # noqa: SLF001

    mock_client = MagicMock()
    binary_handling = BinaryHandling(drop_late=False, buffer_track=True)
    mock_client.get_binary_handling_cached.return_value = (binary_handling, mock_role)
    conn._client = mock_client  # noqa: SLF001

    payload = b"audio_data"
    message_type = BinaryMessageType.AUDIO_CHUNK.value
    packed = pack_binary_header_raw(message_type, 0) + payload
    conn.send_binary(
        packed, role="player", timestamp_us=0, message_type=message_type
    )  # No buffer metadata

    for _ in range(50):
        if wsock.send_bytes.called:
            break
        await asyncio.sleep(0)

    assert wsock.send_bytes.call_count == 1
    mock_buffer_tracker.register.assert_not_called()

    await conn.disconnect(retry_connection=False)


@pytest.mark.asyncio
async def test_writer_blocks_on_buffer_tracker_capacity() -> None:
    """Writer should defer sending when buffer tracker reports no capacity."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))

    wsock = MagicMock()
    wsock.closed = False
    wsock.send_str = AsyncMock()
    wsock.send_bytes = AsyncMock()

    conn = SendspinConnection(server, wsock_client=wsock)
    conn._transport = wsock  # noqa: SLF001
    await conn._setup_connection()  # noqa: SLF001
    conn._writer_task = asyncio.create_task(conn._writer())  # noqa: SLF001

    mock_role = MagicMock()
    mock_buffer_tracker = MagicMock()
    mock_buffer_tracker.time_until_ready.return_value = 1_000_000
    mock_role.get_buffer_tracker.return_value = mock_buffer_tracker
    mock_role._stream_start_time_us = None  # noqa: SLF001
    mock_role._last_late_log_s = 0.0  # noqa: SLF001
    mock_role._late_skips_since_log = 0  # noqa: SLF001

    mock_client = MagicMock()
    binary_handling = BinaryHandling(drop_late=False, buffer_track=True)
    mock_client.get_binary_handling_cached.return_value = (binary_handling, mock_role)
    conn._client = mock_client  # noqa: SLF001

    payload = b"audio_data"
    message_type = BinaryMessageType.AUDIO_CHUNK.value
    packed = pack_binary_header_raw(message_type, 0) + payload
    conn.send_binary(
        packed,
        role="player",
        timestamp_us=0,
        message_type=message_type,
        buffer_end_time_us=1_000_000,
        buffer_byte_count=100,
        duration_us=50_000,
    )

    # Give writer a chance to process and apply blocking.
    for _ in range(10):
        await asyncio.sleep(0)

    assert wsock.send_bytes.call_count == 0
    mock_buffer_tracker.time_until_ready.assert_called_with(
        100,
        50_000,
        end_time_us=1_000_000,
    )

    await conn.disconnect(retry_connection=False)


@pytest.mark.asyncio
async def test_drop_pending_binary_unblocks_backpressured_role() -> None:
    """drop_pending_binary() must immediately release a backpressured role.

    A stream boundary evicts queued audio whose backpressure deadline was
    computed against now-invalidated state; new-epoch work must be
    schedulable right away while the stale entry is epoch-discarded.
    """
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))

    wsock = MagicMock()
    wsock.closed = False
    wsock.send_str = AsyncMock()
    wsock.send_bytes = AsyncMock()

    conn = SendspinConnection(server, wsock_client=wsock)
    conn._transport = wsock  # noqa: SLF001
    await conn._setup_connection()  # noqa: SLF001
    conn._writer_task = asyncio.create_task(conn._writer())  # noqa: SLF001

    mock_role = MagicMock()
    mock_buffer_tracker = MagicMock()
    mock_buffer_tracker.time_until_ready.return_value = 1_000_000
    mock_role.get_buffer_tracker.return_value = mock_buffer_tracker
    mock_role._stream_start_time_us = None  # noqa: SLF001
    mock_role._last_late_log_s = 0.0  # noqa: SLF001
    mock_role._late_skips_since_log = 0  # noqa: SLF001

    mock_client = MagicMock()
    binary_handling = BinaryHandling(drop_late=False, buffer_track=True)
    mock_client.get_binary_handling_cached.return_value = (binary_handling, mock_role)
    conn._client = mock_client  # noqa: SLF001

    message_type = BinaryMessageType.AUDIO_CHUNK.value
    conn.send_binary(
        pack_binary_header_raw(message_type, 0) + b"stale",
        role="player",
        timestamp_us=0,
        message_type=message_type,
        buffer_end_time_us=1_000_000,
        buffer_byte_count=100,
        duration_us=50_000,
    )

    for _ in range(10):
        await asyncio.sleep(0)
    assert wsock.send_bytes.call_count == 0
    assert "player" in conn._blocked_until_us  # noqa: SLF001

    # Stream boundary: evict the queued binary and open capacity.
    conn.drop_pending_binary(["player"])
    mock_buffer_tracker.time_until_ready.return_value = 0

    conn.send_binary(
        pack_binary_header_raw(message_type, 0) + b"fresh",
        role="player",
        timestamp_us=0,
        message_type=message_type,
        buffer_end_time_us=2_000_000,
        buffer_byte_count=100,
        duration_us=50_000,
    )

    for _ in range(10):
        await asyncio.sleep(0)

    # The stale entry was epoch-discarded and the new-epoch frame went out
    # immediately instead of waiting out the old backpressure deadline.
    assert wsock.send_bytes.call_count == 1
    assert wsock.send_bytes.call_args[0][0].endswith(b"fresh")
    assert "player" not in conn._blocked_until_us  # noqa: SLF001

    await conn.disconnect(retry_connection=False)


def test_check_late_binary_uses_player_effective_timestamp() -> None:
    """Output delay should make late-drop compare against effective play time."""
    loop = asyncio.new_event_loop()
    try:
        clock = ManualClock(now_us_value=10_000_000)
        server = _DummyServer(loop=loop, clock=clock)
        wsock = MagicMock()
        wsock.closed = False
        conn = SendspinConnection(server, wsock_client=wsock)
        conn._transport = wsock  # noqa: SLF001

        role = PlayerV1Role(client=_make_player_client_stub())
        role.output_delay_ms = 5_000
        role._stream_start_time_us = 0  # noqa: SLF001

        handling = BinaryHandling(drop_late=True, grace_period_us=2_000_000)

        # Raw timestamp is still 4s in the future, but effective play time is 1s in the past.
        entry = _RoleQueueEntry(epoch=0, timestamp_us=14_000_000)
        assert conn._check_late_binary(handling, role, entry) is True  # noqa: SLF001
    finally:
        loop.close()


@pytest.mark.asyncio
async def test_server_initiated_connection_starts_writer_task() -> None:
    """Server-initiated connections must start a writer task so enqueued messages are sent."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))

    wsock = MagicMock()
    wsock.closed = False
    wsock.send_str = AsyncMock()
    wsock.send_bytes = AsyncMock()

    conn = SendspinConnection(server, wsock_client=wsock)
    conn._transport = wsock  # noqa: SLF001
    await conn._setup_connection()  # noqa: SLF001
    conn._writer_task = asyncio.create_task(conn._writer())  # noqa: SLF001
    assert conn._writer_task is not None  # noqa: SLF001

    conn.send_message(
        ServerTimeMessage(
            payload=ServerTimePayload(
                client_transmitted=1,
                server_received=2,
                server_transmitted=3,
            )
        )
    )

    for _ in range(50):
        if wsock.send_str.called:
            break
        await asyncio.sleep(0)

    assert wsock.send_str.call_count == 1

    await conn.disconnect(retry_connection=False)


@pytest.mark.asyncio
async def test_role_stream_start_is_sent_before_binary_for_same_role() -> None:
    """Role-scoped stream/start must not be overtaken by timed binary for that role."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))

    send_order: list[str] = []

    async def _record_json(_payload: str) -> None:
        send_order.append("json")

    async def _record_binary(_payload: bytes) -> None:
        send_order.append("binary")

    wsock = MagicMock()
    wsock.closed = False
    wsock.send_str = AsyncMock(side_effect=_record_json)
    wsock.send_bytes = AsyncMock(side_effect=_record_binary)

    conn = SendspinConnection(server, wsock_client=wsock)
    conn._transport = wsock  # noqa: SLF001
    await conn._setup_connection()  # noqa: SLF001
    conn._writer_task = asyncio.create_task(conn._writer())  # noqa: SLF001

    conn.send_role_message(
        "player",
        StreamStartMessage(
            payload=StreamStartPayload(
                player=StreamStartPlayer(
                    codec=AudioCodec.PCM,
                    sample_rate=44_100,
                    channels=2,
                    bit_depth=16,
                    codec_header=None,
                )
            )
        ),
    )
    conn.send_binary(
        pack_binary_header_raw(BinaryMessageType.AUDIO_CHUNK.value, 123_456) + b"audio",
        role="player",
        timestamp_us=123_456,
        message_type=BinaryMessageType.AUDIO_CHUNK.value,
    )

    for _ in range(50):
        if len(send_order) >= 2:
            break
        await asyncio.sleep(0)

    assert send_order[:2] == ["json", "binary"]

    await conn.disconnect(retry_connection=False)


@pytest.mark.asyncio
async def test_role_stream_lifecycle_json_is_sent_before_older_binary() -> None:
    """Binary with older playback ts must not overtake queued stream lifecycle JSON."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))

    send_order: list[str] = []

    async def _record_json(_payload: str) -> None:
        send_order.append("json")

    async def _record_binary(_payload: bytes) -> None:
        send_order.append("binary")

    wsock = MagicMock()
    wsock.closed = False
    wsock.send_str = AsyncMock(side_effect=_record_json)
    wsock.send_bytes = AsyncMock(side_effect=_record_binary)

    conn = SendspinConnection(server, wsock_client=wsock)
    conn._transport = wsock  # noqa: SLF001
    await conn._setup_connection()  # noqa: SLF001
    conn._writer_task = asyncio.create_task(conn._writer())  # noqa: SLF001

    conn.send_role_message("player", StreamEndMessage(payload=StreamEndPayload(roles=None)))
    conn.send_role_message(
        "player",
        StreamStartMessage(
            payload=StreamStartPayload(
                player=StreamStartPlayer(
                    codec=AudioCodec.PCM,
                    sample_rate=44_100,
                    channels=2,
                    bit_depth=16,
                    codec_header=None,
                )
            )
        ),
    )
    # Timestamp intentionally older than stream lifecycle sort timestamp.
    conn.send_binary(
        pack_binary_header_raw(BinaryMessageType.AUDIO_CHUNK.value, 1) + b"audio",
        role="player",
        timestamp_us=1,
        message_type=BinaryMessageType.AUDIO_CHUNK.value,
    )

    for _ in range(50):
        if len(send_order) >= 3:
            break
        await asyncio.sleep(0)

    assert send_order[:3] == ["json", "json", "binary"]

    await conn.disconnect(retry_connection=False)


@pytest.mark.asyncio
async def test_writer_rewrites_server_transmitted_at_send_time() -> None:
    """`server/time` must carry the clock value at actual send, not at enqueue."""
    loop = asyncio.get_running_loop()
    clock = ManualClock(now_us_value=1_000_000)
    server = _DummyServer(loop=loop, clock=clock)

    sent_json: list[str] = []

    async def _record_json(payload: str) -> None:
        sent_json.append(payload)

    wsock = MagicMock()
    wsock.closed = False
    wsock.send_str = AsyncMock(side_effect=_record_json)
    wsock.send_bytes = AsyncMock()

    conn = SendspinConnection(server, wsock_client=wsock)
    conn._transport = wsock  # noqa: SLF001
    await conn._setup_connection()  # noqa: SLF001
    conn._writer_task = asyncio.create_task(conn._writer())  # noqa: SLF001

    conn.send_message(
        ServerTimeMessage(
            payload=ServerTimePayload(
                client_transmitted=11,
                server_received=22,
                server_transmitted=0,
            )
        )
    )

    # Simulate enqueue-to-send latency before the writer drains the queue.
    clock.advance_us(750_000)

    for _ in range(50):
        if sent_json:
            break
        await asyncio.sleep(0)

    assert len(sent_json) == 1
    payload = json.loads(sent_json[0])["payload"]
    assert payload["client_transmitted"] == 11
    assert payload["server_received"] == 22
    assert payload["server_transmitted"] == 1_750_000

    await conn.disconnect(retry_connection=False)


@pytest.mark.asyncio
async def test_send_message_stamps_stream_end_server_transmitted() -> None:
    """stream/end carries the clock value at actual send, like server/time."""
    loop = asyncio.get_running_loop()
    clock = ManualClock(now_us_value=5_000_000)
    server = _DummyServer(loop=loop, clock=clock)

    sent: list[str] = []
    wsock = MagicMock()
    wsock.closed = False
    wsock.send_str = AsyncMock(side_effect=sent.append)

    conn = SendspinConnection(server, wsock_client=wsock)

    await conn._send_message(  # noqa: SLF001
        wsock, StreamEndMessage(payload=StreamEndPayload(roles=["player"]))
    )

    payload = json.loads(sent[0])["payload"]
    assert payload["server_transmitted"] == 5_000_000
    assert payload["roles"] == ["player"]


@pytest.mark.asyncio
async def test_send_binary_disconnects_on_per_role_queue_overflow() -> None:
    """Per-role queue overflow should trigger disconnect."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))

    wsock = MagicMock()
    wsock.closed = False

    conn = SendspinConnection(server, wsock_client=wsock)
    conn._transport = wsock  # noqa: SLF001
    conn._max_pending_msg_by_role["player"] = 1  # noqa: SLF001
    conn.disconnect = AsyncMock()  # type: ignore[method-assign]

    conn.send_binary(
        b"frame-1",
        role="player",
        timestamp_us=0,
        message_type=BinaryMessageType.AUDIO_CHUNK.value,
    )
    conn.send_binary(
        b"frame-2",
        role="player",
        timestamp_us=25_000,
        message_type=BinaryMessageType.AUDIO_CHUNK.value,
    )

    await asyncio.sleep(0)

    assert conn.disconnect.call_count == 1  # type: ignore[attr-defined]


def test_priority_message_queue_cap_uses_own_length() -> None:
    """Priority queue cap should ignore unrelated role-queue bytes in `_queue_size`."""
    loop = asyncio.new_event_loop()
    try:
        server = _DummyServer(loop=loop, clock=LoopClock(loop))
        wsock = MagicMock()
        wsock.closed = False
        conn = SendspinConnection(server, wsock_client=wsock)
        conn.disconnect = AsyncMock()  # type: ignore[method-assign]

        # Simulate saturated role queues: aggregate _queue_size above the priority cap
        # without putting anything into _priority_messages itself.
        conn._queue_size = MAX_PENDING_MSG * 2  # noqa: SLF001

        conn.send_priority_message(
            ServerTimeMessage(
                payload=ServerTimePayload(
                    client_transmitted=1,
                    server_received=2,
                    server_transmitted=0,
                )
            )
        )

        assert conn.disconnect.call_count == 0  # type: ignore[attr-defined]
        assert len(conn._priority_messages) == 1  # noqa: SLF001
    finally:
        loop.close()


def test_per_role_queue_limit_is_isolated_between_roles() -> None:
    """One saturated role queue should not block enqueueing another role."""
    loop = asyncio.new_event_loop()
    try:
        server = _DummyServer(loop=loop, clock=LoopClock(loop))
        wsock = MagicMock()
        wsock.closed = False
        conn = SendspinConnection(server, wsock_client=wsock)
        conn._transport = wsock  # noqa: SLF001
        conn.disconnect = AsyncMock()  # type: ignore[method-assign]
        conn._max_pending_msg_by_role["player"] = 1  # noqa: SLF001
        conn._max_pending_msg_by_role["visualizer"] = 1  # noqa: SLF001

        conn.send_binary(
            b"player-frame",
            role="player",
            timestamp_us=0,
            message_type=BinaryMessageType.AUDIO_CHUNK.value,
        )
        conn.send_binary(
            b"visualizer-frame",
            role="visualizer",
            timestamp_us=0,
            message_type=BinaryMessageType.AUDIO_CHUNK.value,
        )

        assert len(conn._role_queues["player"]) == 1  # noqa: SLF001
        assert len(conn._role_queues["visualizer"]) == 1  # noqa: SLF001
        assert conn.disconnect.call_count == 0  # type: ignore[attr-defined]
    finally:
        loop.close()


def _make_connection_with_droppable_client(
    clock: ManualClock, loop: asyncio.AbstractEventLoop, *, drop_late: bool = True
) -> SendspinConnection:
    server = _DummyServer(loop=loop, clock=clock)
    wsock = MagicMock()
    wsock.closed = False
    conn = SendspinConnection(server, wsock_client=wsock)
    conn._transport = wsock  # noqa: SLF001
    client = MagicMock()
    client.active_roles = []
    role = MagicMock()
    role.get_output_delay_us.return_value = 0
    client.get_binary_handling_cached.return_value = (
        BinaryHandling(drop_late=drop_late, grace_period_us=2_000_000),
        role,
    )
    conn._client = client  # noqa: SLF001
    return conn


def test_send_binary_warns_when_chunk_already_past_deadline(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Enqueuing a droppable chunk whose play window fully passed warns immediately."""
    loop = asyncio.new_event_loop()
    try:
        clock = ManualClock(now_us_value=10_000_000)
        conn = _make_connection_with_droppable_client(clock, loop)

        with caplog.at_level(logging.WARNING):
            conn.send_binary(
                b"audio",
                role="player",
                timestamp_us=8_000_000,
                message_type=BinaryMessageType.AUDIO_CHUNK.value,
                duration_us=25_000,
            )

        assert "Enqueued already-late binary" in caplog.text
        assert "behind_by_us=2000000" in caplog.text
    finally:
        loop.close()


def test_send_binary_no_doomed_warning_without_drop_late(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Past timestamps on non-droppable message types are not flagged."""
    loop = asyncio.new_event_loop()
    try:
        clock = ManualClock(now_us_value=10_000_000)
        conn = _make_connection_with_droppable_client(clock, loop, drop_late=False)

        with caplog.at_level(logging.WARNING):
            conn.send_binary(
                b"data",
                role="visualizer",
                timestamp_us=8_000_000,
                message_type=BinaryMessageType.AUDIO_CHUNK.value,
                duration_us=25_000,
            )

        assert "Enqueued already-late binary" not in caplog.text
    finally:
        loop.close()


def test_late_binary_warning_reports_regime_diagnostics(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The late-drop warning includes enqueue lead, queue age and stream elapsed time."""
    loop = asyncio.new_event_loop()
    try:
        clock = ManualClock(now_us_value=10_000_000)
        server = _DummyServer(loop=loop, clock=clock)
        wsock = MagicMock()
        wsock.closed = False
        conn = SendspinConnection(server, wsock_client=wsock)
        conn._transport = wsock  # noqa: SLF001

        role = PlayerV1Role(client=_make_player_client_stub())
        role._stream_start_time_us = 0  # noqa: SLF001
        handling = BinaryHandling(drop_late=True, grace_period_us=2_000_000)

        entry = _RoleQueueEntry(epoch=0, timestamp_us=9_000_000, enqueued_at_us=8_500_000)
        with caplog.at_level(logging.WARNING):
            assert conn._check_late_binary(handling, role, entry) is True  # noqa: SLF001

        assert "enq_lead_ms=500" in caplog.text
        assert "queue_age_ms=1500" in caplog.text
        assert "stream_elapsed_s=10.0" in caplog.text
        # This role has no buffer tracker, so the buffer fields are left out
        # rather than reported as placeholder values.
        assert "buf_ms" not in caplog.text
    finally:
        loop.close()


def test_late_binary_warning_reports_buffer_state_when_tracked(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """With a buffer tracker present the warning reports the device buffer state."""
    loop = asyncio.new_event_loop()
    try:
        clock = ManualClock(now_us_value=10_000_000)
        server = _DummyServer(loop=loop, clock=clock)
        wsock = MagicMock()
        wsock.closed = False
        conn = SendspinConnection(server, wsock_client=wsock)
        conn._transport = wsock  # noqa: SLF001

        role = PlayerV1Role(client=_make_player_client_stub())
        role._stream_start_time_us = 0  # noqa: SLF001
        tracker = BufferTracker(
            clock=clock, client_id="p", capacity_bytes=200_000, max_duration_us=30_000_000
        )
        tracker.register(12_000_000, 4_000, 25_000)
        role._state().buffer_tracker = tracker  # noqa: SLF001

        entry = _RoleQueueEntry(epoch=0, timestamp_us=9_000_000, enqueued_at_us=8_500_000)
        handling = BinaryHandling(drop_late=True, grace_period_us=2_000_000)
        with caplog.at_level(logging.WARNING):
            assert conn._check_late_binary(handling, role, entry) is True  # noqa: SLF001

        assert "buf_ms=2000" in caplog.text
        assert "buf_bytes=4000/200000" in caplog.text
    finally:
        loop.close()


def test_late_binary_warning_is_throttled_across_a_burst(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A burst of late chunks yields one warning that carries the suppressed count."""
    loop = asyncio.new_event_loop()
    try:
        clock = ManualClock(now_us_value=10_000_000)
        server = _DummyServer(loop=loop, clock=clock)
        wsock = MagicMock()
        wsock.closed = False
        conn = SendspinConnection(server, wsock_client=wsock)
        conn._transport = wsock  # noqa: SLF001

        role = PlayerV1Role(client=_make_player_client_stub())
        role._stream_start_time_us = 0  # noqa: SLF001
        handling = BinaryHandling(drop_late=True, grace_period_us=2_000_000)

        with caplog.at_level(logging.WARNING):
            for _ in range(5):
                entry = _RoleQueueEntry(epoch=0, timestamp_us=9_000_000, enqueued_at_us=8_500_000)
                assert conn._check_late_binary(handling, role, entry) is True  # noqa: SLF001

        assert caplog.text.count("Late binary") == 1
        # The suppressed drops still accumulate for the next warning to report.
        assert role._late_skips_since_log == 4  # noqa: SLF001
    finally:
        loop.close()


def test_late_binary_diagnostics_use_the_effective_play_time(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A output delay shifts the deadline, so enq_lead must agree with late_by_us."""
    loop = asyncio.new_event_loop()
    try:
        clock = ManualClock(now_us_value=10_000_000)
        server = _DummyServer(loop=loop, clock=clock)
        wsock = MagicMock()
        wsock.closed = False
        conn = SendspinConnection(server, wsock_client=wsock)
        conn._transport = wsock  # noqa: SLF001

        role = PlayerV1Role(client=_make_player_client_stub())
        role._stream_start_time_us = 0  # noqa: SLF001
        role.output_delay_ms = 5_000

        # Raw timestamp is 4s ahead, but the effective play time is 1s in the past.
        entry = _RoleQueueEntry(epoch=0, timestamp_us=14_000_000, enqueued_at_us=9_500_000)
        handling = BinaryHandling(drop_late=True, grace_period_us=2_000_000)
        with caplog.at_level(logging.WARNING):
            assert conn._check_late_binary(handling, role, entry) is True  # noqa: SLF001

        # Reported against the same deadline as late_by_us, not the raw timestamp
        # (which would have claimed a healthy +4500ms lead for a late chunk).
        assert "enq_lead_ms=-500" in caplog.text
        assert "late_by_us=1000000" in caplog.text
    finally:
        loop.close()
