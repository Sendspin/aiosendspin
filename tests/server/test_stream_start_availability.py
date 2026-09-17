"""Player stream/start follows the latest client/state: its availability and its timing."""

from __future__ import annotations

import asyncio
import dataclasses
from dataclasses import dataclass
from typing import Any

import pytest

from aiosendspin.models.core import ClientStatePayload, StreamStartMessage
from aiosendspin.models.types import Roles
from aiosendspin.server.audio import AudioFormat
from aiosendspin.server.client import SendspinClient
from aiosendspin.server.clock import LoopClock
from aiosendspin.server.connection import SendspinConnection
from aiosendspin.server.group import SendspinGroup
from aiosendspin.server.push_stream import PushStream
from tests.server.test_group_add_client import _DummyConnection, _DummyServer, _make_player
from tests.server.test_group_add_client import _hello as _owner_hello
from tests.server.test_role_activation import _PLAYER_STATE, _client, _connect, _hello

_FORMAT = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)


@dataclass(slots=True)
class _Server(_DummyServer):
    allow_noncompliant_clients: bool = True


class _RecordingConnection(_DummyConnection):
    def __init__(self) -> None:
        super().__init__()
        self.binary_count = 0

    def send_binary(self, data: bytes, **kwargs: object) -> bool:  # noqa: ARG002
        self.binary_count += 1
        return True


def _player(server: _Server, client_id: str) -> tuple[SendspinClient, _RecordingConnection]:
    client = _make_player(server, client_id)
    connection = _RecordingConnection()
    client._connection = connection  # type: ignore[assignment]  # noqa: SLF001
    return client, connection


def _stream_starts(connection: _DummyConnection) -> list[object]:
    return [msg for _, msg in connection.role_messages if isinstance(msg, StreamStartMessage)]


async def _commit(stream: PushStream) -> None:
    stream.prepare_audio(bytes(4800 * 4), _FORMAT)
    await stream.commit_audio()


@pytest.mark.asyncio
async def test_start_stream_holds_stream_start_until_available() -> None:
    """A stream started for an unavailable player sends nothing until it becomes available."""
    loop = asyncio.get_running_loop()
    player, connection = _player(_Server(loop=loop, clock=LoopClock(loop)), "player")
    await player.handle_availability_change(available=False)

    stream = player.group.start_stream()
    await _commit(stream)

    assert _stream_starts(connection) == []
    assert connection.binary_count == 0

    await player.handle_availability_change(available=True)
    await _commit(stream)

    assert len(_stream_starts(connection)) == 1
    assert connection.binary_count > 0
    stream.stop()


@pytest.mark.asyncio
async def test_add_client_sends_no_stream_start_to_unavailable_player() -> None:
    """Adding an unavailable player to a playing group sends it no stream/start."""
    loop = asyncio.get_running_loop()
    server = _Server(loop=loop, clock=LoopClock(loop))
    owner, owner_connection = _player(server, "owner")
    joiner, joiner_connection = _player(server, "joiner")
    stream = owner.group.start_stream()
    await _commit(stream)
    await joiner.handle_availability_change(available=False)

    await owner.group.add_client(joiner)
    await _commit(stream)

    assert len(_stream_starts(owner_connection)) == 1
    assert _stream_starts(joiner_connection) == []
    assert joiner_connection.binary_count == 0
    stream.stop()


async def _joiner_in_playing_group(
    monkeypatch: pytest.MonkeyPatch, audio_s: int
) -> tuple[SendspinConnection, PushStream]:
    """Return a connection awaiting its initial client/state, grouped with a playing owner."""
    conn, _fake = await _connect(_hello([Roles.PLAYER.value]), send_state=False)
    server = conn._server  # noqa: SLF001
    monkeypatch.setattr(
        server, "request_client_playback_connection", lambda _client_id: False, raising=False
    )
    joiner = _client(conn)
    owner = SendspinClient(server, client_id="owner")  # type: ignore[arg-type]
    SendspinGroup(server, owner)  # type: ignore[arg-type]
    owner.attach_connection(
        _RecordingConnection(),  # type: ignore[arg-type]
        client_info=_owner_hello("owner"),
        negotiated_roles=[Roles.PLAYER.value],
        active_roles=[Roles.PLAYER.value],
    )
    owner.mark_connected()
    await owner.group.add_client(joiner)
    stream = owner.group.start_stream()
    # The owner shares the joiner's format, so the join replays its cached chunks at once.
    stream.prepare_audio(bytes(48000 * audio_s * 4), _FORMAT)
    await stream.commit_audio()
    return conn, stream


def _queued_player_entries(conn: SendspinConnection) -> list[Any]:
    return [entry for _, _, entry in conn._role_queues.get("player", [])]  # noqa: SLF001


@pytest.mark.asyncio
@pytest.mark.parametrize("available", [True, False])
async def test_initial_state_availability_applies_before_the_stream_join(
    available: bool,  # noqa: FBT001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An initial client/state joins a playing group only when it reports available: true."""
    conn, stream = await _joiner_in_playing_group(monkeypatch, audio_s=2)

    await conn._handle_client_state(  # noqa: SLF001
        ClientStatePayload(available=available, player=_PLAYER_STATE)
    )

    queued = _queued_player_entries(conn)
    starts = [entry for entry in queued if isinstance(entry.json_message, StreamStartMessage)]
    assert len(starts) == (1 if available else 0)
    assert any(entry.binary is not None for entry in queued) is available
    stream.stop()


@pytest.mark.asyncio
async def test_initial_state_timing_applies_before_the_stream_join(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The late-join replay starts at the lead the initial client/state reports."""
    conn, stream = await _joiner_in_playing_group(monkeypatch, audio_s=4)
    lead_ms = 1_500
    state = dataclasses.replace(_PLAYER_STATE, required_lead_time_ms=lead_ms, min_buffer_ms=0)
    now_us = conn._server.clock.now_us()  # noqa: SLF001

    await conn._handle_client_state(  # noqa: SLF001
        ClientStatePayload(available=True, player=state)
    )

    binary = [entry for entry in _queued_player_entries(conn) if entry.binary is not None]
    assert binary
    assert min(entry.timestamp_us for entry in binary) >= now_us + lead_ms * 1_000
    stream.stop()
