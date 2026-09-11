"""The client ends the input stream when the server drops source from active_roles."""

from __future__ import annotations

import asyncio

import orjson
import pytest

from aiosendspin.client.connection import SendspinConnection
from aiosendspin.models.core import ServerActivatePayload, ServerTimePayload
from aiosendspin.models.types import Activity, AudioCodec, GoodbyeReason, Roles
from aiosendspin.noise.trust_store import PskCategory, ResolvedPsk


class _FakeWs:
    closed = False

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.sent_bytes: list[bytes] = []

    async def send_str(self, data: str) -> None:
        self.sent.append(data)

    async def send_bytes(self, data: bytes) -> None:
        self.sent_bytes.append(data)


class _FakeClient:
    class _Clock:
        @staticmethod
        def now_us() -> int:
            return 1_000_000

    clock = _Clock()

    async def note_playback_activity(self, _conn: object) -> None:
        pass


class _FakeTimeFilter:
    def __init__(self, *, synchronized: bool) -> None:
        self.is_synchronized = synchronized

    def update(self, _offset: int, _delay: int, _now_us: int) -> None:
        self.is_synchronized = True


def _connection(
    ws: _FakeWs,
    *,
    active_roles: list[str],
    stream_active: bool,
    category: PskCategory = PskCategory.LONG_TERM,
    unpaired_access: bool = False,
    synchronized: bool = True,
) -> SendspinConnection:
    conn = SendspinConnection.__new__(SendspinConnection)
    conn._client = _FakeClient()  # type: ignore[assignment]  # noqa: SLF001
    conn._ws = ws  # type: ignore[assignment]  # noqa: SLF001
    conn._connected = True  # noqa: SLF001
    conn._send_lock = asyncio.Lock()  # noqa: SLF001
    conn._exchange_in_progress = False  # noqa: SLF001
    conn._noise_psk = ResolvedPsk("id", b"\x00" * 32, category)  # noqa: SLF001
    conn._active_roles = active_roles  # noqa: SLF001
    conn._source_stream_active = stream_active  # noqa: SLF001
    conn._reported_available = True  # noqa: SLF001
    conn._reported_source_signal = None  # noqa: SLF001
    conn._time_filter = _FakeTimeFilter(synchronized=synchronized)  # type: ignore[assignment]  # noqa: SLF001
    conn._selected_pairing = None  # noqa: SLF001

    async def _unpaired_access_enabled() -> bool:
        return unpaired_access

    conn._unpaired_access_enabled = _unpaired_access_enabled  # type: ignore[method-assign]  # noqa: SLF001
    return conn


async def test_source_dropped_from_active_roles_ends_client_stream() -> None:
    """Removing source@v1 from active_roles auto-sends client-stream/end."""
    ws = _FakeWs()
    conn = _connection(ws, active_roles=[Roles.SOURCE.value], stream_active=True)
    await conn._apply_activation(  # noqa: SLF001
        ServerActivatePayload(activities=[Activity.PLAYBACK], active_roles=[])
    )
    assert any("client-stream/end" in m for m in ws.sent)
    assert conn._source_stream_active is False  # noqa: SLF001


async def test_no_client_stream_end_when_source_retained() -> None:
    """Keeping source@v1 active does not end the input stream."""
    ws = _FakeWs()
    conn = _connection(ws, active_roles=[Roles.SOURCE.value], stream_active=True)
    await conn._apply_activation(  # noqa: SLF001
        ServerActivatePayload(activities=[Activity.PLAYBACK], active_roles=[Roles.SOURCE.value])
    )
    assert not any("client-stream/end" in m for m in ws.sent)
    assert conn._source_stream_active is True  # noqa: SLF001


async def test_no_client_stream_end_when_no_stream_active() -> None:
    """Dropping source with no active input stream sends nothing."""
    ws = _FakeWs()
    conn = _connection(ws, active_roles=[Roles.SOURCE.value], stream_active=False)
    await conn._apply_activation(  # noqa: SLF001
        ServerActivatePayload(activities=[Activity.PLAYBACK], active_roles=[])
    )
    assert ws.sent == []


async def test_unpaired_activation_rejects_source_role() -> None:
    """Unpaired connections cannot activate the source role."""
    ws = _FakeWs()
    conn = _connection(
        ws,
        active_roles=[],
        stream_active=False,
        category=PskCategory.SENTINEL,
        unpaired_access=True,
    )

    reason = await conn._apply_activation(  # noqa: SLF001
        ServerActivatePayload(activities=[Activity.PLAYBACK], active_roles=[Roles.SOURCE.value])
    )

    assert reason is GoodbyeReason.UNAUTHORIZED


async def test_source_chunks_rejected_after_role_deactivation() -> None:
    """Deactivated source roles cannot send more audio chunks."""
    ws = _FakeWs()
    conn = _connection(ws, active_roles=[Roles.SOURCE.value], stream_active=True)
    await conn._apply_activation(  # noqa: SLF001
        ServerActivatePayload(activities=[Activity.PLAYBACK], active_roles=[])
    )

    with pytest.raises(RuntimeError, match="not active"):
        await conn.send_source_chunk(b"audio", timestamp_us=1)

    assert ws.sent_bytes == []


async def test_send_source_chunk_rechecks_connection_under_lock() -> None:
    """A binary send reports disconnect while waiting for the send lock."""
    ws = _FakeWs()
    conn = _connection(ws, active_roles=[Roles.SOURCE.value], stream_active=True)
    await conn._send_lock.acquire()  # noqa: SLF001
    send_task = asyncio.create_task(conn._send_bytes(b"audio"))  # noqa: SLF001
    await asyncio.sleep(0)
    conn._ws = None  # type: ignore[assignment]  # noqa: SLF001
    conn._send_lock.release()  # noqa: SLF001

    with pytest.raises(RuntimeError, match="not connected"):
        await send_task


async def test_stream_start_rejected_during_in_band_exchange() -> None:
    """A stream start during an in-band exchange fails instead of marking the stream active."""
    ws = _FakeWs()
    conn = _connection(ws, active_roles=[Roles.SOURCE.value], stream_active=False)
    conn._exchange_in_progress = True  # noqa: SLF001

    with pytest.raises(RuntimeError, match="in-band exchange"):
        await conn.send_client_stream_start(
            codec=AudioCodec.PCM, sample_rate=48000, channels=2, bit_depth=16, codec_header=None
        )

    assert not conn.is_source_stream_active()
    assert ws.sent == []


async def test_stream_end_rejected_during_in_band_exchange() -> None:
    """A stream end during an in-band exchange fails instead of marking the stream inactive."""
    ws = _FakeWs()
    conn = _connection(ws, active_roles=[Roles.SOURCE.value], stream_active=True)
    conn._exchange_in_progress = True  # noqa: SLF001

    with pytest.raises(RuntimeError, match="in-band exchange"):
        await conn.send_client_stream_end()

    assert conn.is_source_stream_active()
    assert ws.sent == []


async def test_source_unavailable_ends_stream_before_state() -> None:
    """Reporting unavailable ends capture before sending source state without player fields."""
    ws = _FakeWs()
    conn = _connection(ws, active_roles=[Roles.SOURCE.value], stream_active=True)

    await conn.send_available(available=False)

    messages = [orjson.loads(message) for message in ws.sent]
    assert [message["type"] for message in messages] == [
        "client-stream/end",
        "client/state",
    ]
    assert messages[-1]["payload"] == {"available": False, "source": {}}


async def test_source_chunk_cannot_pass_queued_stream_end() -> None:
    """A chunk queued after stream end begins is rejected before reaching the wire."""
    ws = _FakeWs()
    conn = _connection(ws, active_roles=[Roles.SOURCE.value], stream_active=True)
    await conn._send_lock.acquire()  # noqa: SLF001
    available_task = asyncio.create_task(conn.send_available(available=False))
    await asyncio.sleep(0)
    chunk_task = asyncio.create_task(conn.send_source_chunk(b"audio", timestamp_us=1))
    await asyncio.sleep(0)
    conn._send_lock.release()  # noqa: SLF001

    await available_task
    with pytest.raises(RuntimeError, match="not active"):
        await chunk_task

    assert ws.sent_bytes == []


async def test_source_available_sends_retained_source_state() -> None:
    """Reporting available sends source state without adding a player payload."""
    ws = _FakeWs()
    conn = _connection(ws, active_roles=[Roles.SOURCE.value], stream_active=False)
    conn._reported_available = False  # noqa: SLF001

    await conn.send_available(available=True)

    assert orjson.loads(ws.sent[-1])["payload"] == {"available": True, "source": {}}


async def test_source_state_sent_when_clock_first_synchronizes() -> None:
    """Clock convergence sends the retained availability and source state."""
    ws = _FakeWs()
    conn = _connection(
        ws,
        active_roles=[Roles.SOURCE.value],
        stream_active=False,
        synchronized=False,
    )

    await conn._handle_server_time(  # noqa: SLF001
        ServerTimePayload(
            client_transmitted=900_000,
            server_received=950_000,
            server_transmitted=960_000,
        )
    )

    assert orjson.loads(ws.sent[-1])["payload"] == {"available": True, "source": {}}
