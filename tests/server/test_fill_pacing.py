"""Tests for pacing audio that fills a player's buffer."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.models.types import AudioCodec, PlayerCommand, Roles
from aiosendspin.server.audio_transformers import TransformerPool
from aiosendspin.server.channels import MAIN_CHANNEL
from aiosendspin.server.client import SendspinClient
from aiosendspin.server.clock import ManualClock
from aiosendspin.server.connection import SendspinConnection
from aiosendspin.server.roles.base import AudioChunk
from aiosendspin.server.roles.player.v1 import PlayerV1Role

CHUNK_US = 100_000
CHUNK_BYTES = 10_000


@dataclass(slots=True)
class _DummyServer:
    loop: Any
    clock: Any
    id: str = "srv"
    name: str = "server"

    def is_external_player(self, client_id: str) -> bool:  # noqa: ARG002
        return False

    def _signal_client_connected(self, client_id: str) -> None:
        pass

    def _signal_client_disconnected(self, client_id: str, goodbye_reason: object = None) -> None:
        pass


class _DummyGroup:
    def __init__(self) -> None:
        self.clients: list[Any] = []
        self.group_id = "g1"
        self.transformer_pool = TransformerPool()

    group_name = "dummy group"

    def _publish_if_name_changed(self, previous: str) -> None:  # noqa: ARG002
        return

    def on_client_connected(self, client: Any) -> None:  # noqa: ARG002
        return

    def _register_client_events(self, client: Any) -> None:  # noqa: ARG002
        return

    def group_role(self, family: str) -> None:  # noqa: ARG002
        return None

    def get_channel_for_player(self, player_id: str) -> UUID:  # noqa: ARG002
        return MAIN_CHANNEL


async def _start_player(
    clock: ManualClock,
) -> tuple[SendspinConnection, PlayerV1Role, list[bytes]]:
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=clock)
    sent: list[bytes] = []

    async def record(data: bytes) -> None:
        sent.append(data)

    wsock = MagicMock()
    wsock.closed = False
    wsock.send_str = AsyncMock()
    wsock.send_bytes = AsyncMock(side_effect=record)

    conn = SendspinConnection(server, wsock_client=wsock)
    conn._transport = wsock  # noqa: SLF001
    await conn._setup_connection()  # noqa: SLF001
    conn._initial_state_received = True  # noqa: SLF001
    conn._writer_task = asyncio.create_task(conn._writer())  # noqa: SLF001

    group = _DummyGroup()
    client = SendspinClient(server, client_id="p1")
    client._group = group  # noqa: SLF001
    group.clients.append(client)

    hello = type("Hello", (), {})()
    hello.client_id = "p1"
    hello.name = "p1"
    hello.device_info = None
    hello.player_support = ClientHelloPlayerSupport(
        supported_formats=[
            SupportedAudioFormat(codec=AudioCodec.FLAC, channels=2, sample_rate=48000, bit_depth=16)
        ],
        buffer_capacity=800_000,
        supported_commands=[PlayerCommand.VOLUME],
    )
    hello.artwork_support = None
    hello.visualizer_support = None
    client.attach_connection(
        conn,
        client_info=hello,
        negotiated_roles=[Roles.PLAYER.value],
        active_roles=[Roles.PLAYER.value],
    )
    client.mark_connected()
    conn._client = client  # noqa: SLF001

    role = client.role("player@v1")
    assert isinstance(role, PlayerV1Role)
    role._stream_started = True  # noqa: SLF001
    return conn, role, sent


def _send_catch_up(role: PlayerV1Role, first_timestamp_us: int, duration_s: int) -> None:
    for index in range(duration_s * 1_000_000 // CHUNK_US):
        role.on_audio_chunk(
            AudioChunk(
                timestamp_us=first_timestamp_us + index * CHUNK_US,
                data=b"x" * CHUNK_BYTES,
                byte_count=CHUNK_BYTES,
                duration_us=CHUNK_US,
            )
        )


async def _settle(conn: SendspinConnection, iterations: int = 200) -> None:
    conn._wake_writer()  # noqa: SLF001
    for _ in range(iterations):
        await asyncio.sleep(0)


async def _run_for(conn: SendspinConnection, clock: ManualClock, duration_us: int) -> None:
    await _settle(conn)
    # Coarse steps make the writer wake up to 30 ms after a paced send is due.
    for _ in range(duration_us // 30_000):
        clock.advance_us(30_000)
        await _settle(conn, iterations=10)


def _sent_audio_end_us(role: PlayerV1Role) -> int:
    tracker = role.get_buffer_tracker()
    assert tracker is not None
    assert tracker.buffered_chunks
    return tracker.buffered_chunks[-1].end_time_us


@pytest.mark.asyncio
async def test_join_does_not_send_the_whole_buffer_at_once() -> None:
    """A joiner with a deep cache gets audio well below line rate in its first second."""
    clock = ManualClock(now_us_value=100_000_000)
    conn, role, sent = await _start_player(clock)
    try:
        _send_catch_up(role, clock.now_us() + 1_000_000, duration_s=30)
        await _run_for(conn, clock, 1_000_000)

        assert sum(len(data) for data in sent) <= 50 * CHUNK_BYTES
    finally:
        await conn.disconnect(retry_connection=False)


@pytest.mark.asyncio
async def test_joiner_buffer_keeps_growing_faster_than_playback() -> None:
    """Paced audio still builds the joiner's buffer well ahead of playback."""
    clock = ManualClock(now_us_value=100_000_000)
    conn, role, _sent = await _start_player(clock)
    try:
        _send_catch_up(role, clock.now_us() + 1_000_000, duration_s=30)
        await _run_for(conn, clock, 4_000_000)

        assert _sent_audio_end_us(role) - clock.now_us() >= 6_000_000
    finally:
        await conn.disconnect(retry_connection=False)
