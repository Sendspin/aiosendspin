"""Tests for the player format preference carried by client/state."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aiosendspin.models.core import ClientStatePayload, StreamStartMessage
from aiosendspin.models.player import (
    ClientHelloPlayerSupport,
    PlayerStatePayload,
    SupportedAudioFormat,
)
from aiosendspin.models.types import AudioCodec, Roles
from aiosendspin.server.audio import AudioFormat
from aiosendspin.server.client import SendspinClient
from aiosendspin.server.clock import LoopClock
from aiosendspin.server.compliance import ClientComplianceError
from aiosendspin.server.connection import SendspinConnection
from aiosendspin.server.group import SendspinGroup
from aiosendspin.server.roles.player.audio_transformers import (
    FlacEncoder,
    OpusEncoder,
    PcmPassthrough,
)
from aiosendspin.server.roles.player.v1 import PlayerPersistentState, PlayerV1Role

PCM_48K = SupportedAudioFormat(codec=AudioCodec.PCM, sample_rate=48000, bit_depth=16, channels=2)
FLAC_48K = SupportedAudioFormat(codec=AudioCodec.FLAC, sample_rate=48000, bit_depth=16, channels=2)
OPUS_48K = SupportedAudioFormat(codec=AudioCodec.OPUS, sample_rate=48000, bit_depth=16, channels=2)


class _FakeConnection:
    def __init__(self) -> None:
        self.sent: list[object] = []
        self.dropped_pending_binary: list[list[str] | None] = []

    async def disconnect(self, *, retry_connection: bool = True) -> None:  # noqa: ARG002
        return

    def send_message(self, message: object) -> None:
        self.sent.append(message)

    def drop_pending_binary(self, roles: list[str] | None) -> None:
        self.dropped_pending_binary.append(roles)

    def send_role_message(self, role: str, message: object) -> None:  # noqa: ARG002
        self.sent.append(message)


@pytest.fixture
def mock_server() -> MagicMock:
    """Mock server."""
    loop = MagicMock()
    loop.time.return_value = 1000.0
    server = MagicMock()
    server.loop = loop
    server.clock = LoopClock(loop)
    server.allow_noncompliant_clients = True
    return server


def _make_player(
    server: MagicMock,
    supported_formats: list[SupportedAudioFormat] | None = None,
) -> tuple[SendspinClient, PlayerV1Role, _FakeConnection]:
    client = SendspinClient(server, client_id="p1")
    SendspinGroup(server, client)

    conn = _FakeConnection()
    hello = MagicMock()
    hello.client_id = "p1"
    hello.name = "p1"
    hello.player_support = ClientHelloPlayerSupport(
        supported_formats=supported_formats or [PCM_48K, FLAC_48K],
        buffer_capacity=100_000,
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
    role = client.role("player@v1")
    assert isinstance(role, PlayerV1Role)
    return client, role, conn


def _state(fmt: SupportedAudioFormat | None) -> ClientStatePayload:
    return ClientStatePayload(available=True, player=PlayerStatePayload(volume=50, format=fmt))


def _transformer(role: PlayerV1Role) -> object:
    req = role.get_audio_requirements()
    assert req is not None
    return req.transformer


def test_idle_format_applies_to_next_stream_without_starting_one(
    mock_server: MagicMock,
) -> None:
    """With no active stream a preference sends nothing and is used by the next stream."""
    client, role, conn = _make_player(mock_server)
    conn.sent.clear()

    role.on_client_state(_state(FLAC_48K))

    assert not client.group.has_active_stream
    assert conn.sent == []
    assert conn.dropped_pending_binary == []
    assert role._pending_stream_start is False  # noqa: SLF001

    client.group.start_stream()
    role.on_stream_start()
    assert isinstance(_transformer(role), FlacEncoder)


def test_format_change_mid_stream_keeps_buffer_count(mock_server: MagicMock) -> None:
    """The in-place stream/start of a format change leaves already sent chunks counted."""
    client, role, _conn = _make_player(mock_server)
    client.group.start_stream()
    tracker = role.get_buffer_tracker()
    assert tracker is not None
    chunk = tracker.register(mock_server.clock.now_us() + 1_000_000, 5_000, 25_000)
    assert chunk is not None
    tracker.finish_transmission(chunk)

    role.on_client_state(_state(FLAC_48K))

    assert role._pending_stream_start is True  # noqa: SLF001
    assert tracker.buffered_bytes == 5_000
    assert tracker.buffered_chunks[0].duration_us == 25_000


def test_format_change_mid_stream_begins_transition(mock_server: MagicMock) -> None:
    """A changed preference during a stream restarts it in the new format."""
    client, role, conn = _make_player(mock_server)
    client.group.start_stream()

    role.on_client_state(_state(FLAC_48K))

    assert conn.dropped_pending_binary == [["player"]]
    assert role._pending_stream_start is True  # noqa: SLF001
    assert isinstance(_transformer(role), FlacEncoder)
    assert not any(isinstance(msg, StreamStartMessage) for msg in conn.sent)


def test_unchanged_format_runs_no_boundary(mock_server: MagicMock) -> None:
    """A preference for the format already in use neither drops audio nor restarts."""
    client, role, conn = _make_player(mock_server)
    client.group.start_stream()

    role.on_client_state(_state(PCM_48K))
    role.on_client_state(_state(PCM_48K))

    assert conn.dropped_pending_binary == []
    assert isinstance(_transformer(role), PcmPassthrough)


def test_player_object_without_format_clears_preference(mock_server: MagicMock) -> None:
    """A player object without format falls back to the supported_formats priority."""
    client, role, conn = _make_player(mock_server)
    client.group.start_stream()
    role.on_client_state(_state(FLAC_48K))

    role.on_client_state(_state(None))

    assert conn.dropped_pending_binary == [["player"], ["player"]]
    assert isinstance(_transformer(role), PcmPassthrough)


def test_client_state_without_player_object_keeps_preference(mock_server: MagicMock) -> None:
    """A client/state that carries no player object leaves the preference untouched."""
    client, role, conn = _make_player(mock_server)
    client.group.start_stream()
    role.on_client_state(_state(FLAC_48K))

    role.on_client_state(ClientStatePayload(available=True))

    assert conn.dropped_pending_binary == [["player"]]
    assert isinstance(_transformer(role), FlacEncoder)


def test_operator_override_wins_over_client_format(mock_server: MagicMock) -> None:
    """A client preference hidden by the operator override changes nothing."""
    client, role, conn = _make_player(mock_server)
    client.group.start_stream()
    pcm = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)
    assert role.set_preferred_format(pcm, AudioCodec.PCM)

    role.on_client_state(_state(FLAC_48K))
    role.on_client_state(_state(None))

    assert conn.dropped_pending_binary == []
    assert isinstance(_transformer(role), PcmPassthrough)
    state = client.get_or_create_role_state("player", PlayerPersistentState)
    assert state.preferred_format_override == pcm
    assert state.preferred_codec_override == AudioCodec.PCM


def test_client_format_applies_once_override_is_cleared(mock_server: MagicMock) -> None:
    """Clearing the operator override falls back to the client's preference."""
    client, role, conn = _make_player(mock_server)
    client.group.start_stream()
    assert role.set_preferred_format(
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2), AudioCodec.PCM
    )
    role.on_client_state(_state(FLAC_48K))

    assert role.set_preferred_format(None)

    assert conn.dropped_pending_binary == [["player"]]
    assert isinstance(_transformer(role), FlacEncoder)


def test_operator_override_does_not_write_client_preference(mock_server: MagicMock) -> None:
    """The operator override and the client preference are kept apart."""
    _, role, _ = _make_player(mock_server)

    assert role.set_preferred_format(
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2), AudioCodec.FLAC
    )
    assert isinstance(_transformer(role), FlacEncoder)
    assert role.set_preferred_format(None)

    assert isinstance(_transformer(role), PcmPassthrough)


@pytest.mark.parametrize("allow_noncompliant", [True, False])
@pytest.mark.asyncio
async def test_undeclared_format_is_flagged_and_ignored(
    mock_server: MagicMock,
    allow_noncompliant: bool,  # noqa: FBT001
) -> None:
    """A format missing from the hello supported_formats is flagged; the preference stays."""
    mock_server.allow_noncompliant_clients = allow_noncompliant
    client, role, conn = _make_player(mock_server)
    client.group.start_stream()
    role.on_client_state(_state(FLAC_48K))
    server_conn = SendspinConnection(mock_server, wsock_client=MagicMock())
    server_conn._client = client  # noqa: SLF001
    server_conn._initial_state_received = True  # noqa: SLF001
    payload = _state(SupportedAudioFormat(AudioCodec.FLAC, 2, 44100, 16))

    if allow_noncompliant:
        await server_conn._handle_client_state(payload)  # noqa: SLF001
    else:
        with pytest.raises(ClientComplianceError, match="declared supported_formats"):
            await server_conn._handle_client_state(payload)  # noqa: SLF001

    assert conn.dropped_pending_binary == [["player"]]
    assert isinstance(_transformer(role), FlacEncoder)


def test_declared_format_is_not_flagged(mock_server: MagicMock) -> None:
    """A format from the hello supported_formats reports no deviation."""
    _, role, _ = _make_player(mock_server)
    assert role.client_state_deviations(_state(FLAC_48K)) == []


def test_opus_format_ignores_bit_depth(mock_server: MagicMock) -> None:
    """An opus preference matches its supported entry whatever bit_depth it carries."""
    _, role, _ = _make_player(mock_server, [PCM_48K, OPUS_48K])
    payload = _state(SupportedAudioFormat(AudioCodec.OPUS, 2, 48000, 24))

    assert role.client_state_deviations(payload) == []
    role.on_client_state(payload)

    assert isinstance(_transformer(role), OpusEncoder)
    req = role.get_audio_requirements()
    assert req is not None
    assert req.bit_depth == 16


def test_unencodable_format_falls_back_to_priority(mock_server: MagicMock) -> None:
    """A declared format the server cannot encode leaves selection to the priority order."""
    unencodable = SupportedAudioFormat(AudioCodec.PCM, 2, 48000, 8)
    _, role, _ = _make_player(mock_server, [FLAC_48K, unencodable])

    role.on_client_state(_state(unencodable))

    assert isinstance(_transformer(role), FlacEncoder)


def test_preference_is_reset_on_reconnect(mock_server: MagicMock) -> None:
    """A new connection starts without the previous connection's preference."""
    _, role, _ = _make_player(mock_server)
    role.on_client_state(_state(FLAC_48K))

    role.on_disconnect()
    role.on_connect()

    assert isinstance(_transformer(role), PcmPassthrough)
