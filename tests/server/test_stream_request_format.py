"""Legacy acceptance tests for the pre-#195 stream/request-format player object."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiosendspin.models.core import (
    ClientStatePayload,
    StreamClearMessage,
    StreamRequestFormatPayload,
    StreamStartMessage,
)
from aiosendspin.models.player import (
    ClientHelloPlayerSupport,
    PlayerStatePayload,
    StreamRequestFormatPlayer,
    SupportedAudioFormat,
)
from aiosendspin.models.types import AudioCodec, Roles
from aiosendspin.server.audio import AudioFormat
from aiosendspin.server.client import SendspinClient
from aiosendspin.server.clock import LoopClock
from aiosendspin.server.group import SendspinGroup
from aiosendspin.server.roles.player.v1 import PlayerV1Role


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

    def send_binary(
        self,
        data: bytes,  # noqa: ARG002
        *,
        role: str,  # noqa: ARG002
        timestamp_us: int,  # noqa: ARG002
        message_type: int,  # noqa: ARG002
        buffer_end_time_us: int | None = None,  # noqa: ARG002
        buffer_byte_count: int | None = None,  # noqa: ARG002
    ) -> bool:
        return True


@pytest.fixture
def mock_loop() -> MagicMock:
    """Mock event loop for deterministic timestamps."""
    loop = MagicMock()
    loop.time.return_value = 1000.0
    return loop


@pytest.fixture
def mock_server(mock_loop: MagicMock) -> MagicMock:
    """Mock server."""
    server = MagicMock()
    server.loop = mock_loop
    server.clock = LoopClock(mock_loop)
    server.allow_noncompliant_clients = True
    return server


def _make_player_client(
    server: MagicMock,
    client_id: str,
    supported_formats: list[SupportedAudioFormat] | None = None,
) -> tuple[SendspinClient, _FakeConnection]:
    client = SendspinClient(server, client_id=client_id)
    SendspinGroup(server, client)

    conn = _FakeConnection()
    hello = MagicMock()
    hello.client_id = client_id
    hello.name = client_id
    if supported_formats is None:
        supported_formats = [
            SupportedAudioFormat(codec=AudioCodec.PCM, sample_rate=48000, bit_depth=16, channels=2),
            SupportedAudioFormat(
                codec=AudioCodec.FLAC,
                sample_rate=48000,
                bit_depth=16,
                channels=2,
            ),
        ]

    hello.player_support = ClientHelloPlayerSupport(
        supported_formats=supported_formats,
        buffer_capacity=100_000,
        supported_commands=[],
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
    return client, conn


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_player_format_request_defers_stream_start_when_stream_active(
    mock_server: MagicMock,
) -> None:
    """When a PushStream is active, stream/start is deferred (via _pending_stream_start).

    No immediate stream/start or stream/clear is sent. The requirements
    are rebuilt with the new format.
    """
    client, conn = _make_player_client(mock_server, "p1")
    client.group.start_stream()

    request = StreamRequestFormatPayload(
        player=StreamRequestFormatPlayer(
            codec=AudioCodec.FLAC, sample_rate=48000, channels=2, bit_depth=16
        )
    )

    for role in client.active_roles:
        role.on_stream_request_format(request)

    # No immediate stream/start or stream/clear should be sent.
    assert not any(isinstance(msg, StreamStartMessage) for msg in conn.sent)
    assert not any(isinstance(msg, StreamClearMessage) for msg in conn.sent)

    # _pending_stream_start should be set (deferred until first audio chunk).
    player_role = client.role("player@v1")
    assert isinstance(player_role, PlayerV1Role)
    assert player_role._pending_stream_start is True  # noqa: SLF001

    # AudioRequirements should be rebuilt with the new format.
    req = player_role.get_audio_requirements()
    assert req is not None
    assert req.sample_rate == 48000
    assert req.bit_depth == 16
    assert req.channels == 2


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_player_format_request_starts_nothing_when_no_stream_active(
    mock_server: MagicMock,
) -> None:
    """With no active PushStream the request is stored and starts nothing."""
    client, conn = _make_player_client(mock_server, "p1")
    request = StreamRequestFormatPayload(
        player=StreamRequestFormatPlayer(
            codec=AudioCodec.FLAC, sample_rate=48000, channels=2, bit_depth=16
        )
    )

    for role in client.active_roles:
        role.on_stream_request_format(request)

    assert not any(isinstance(msg, StreamStartMessage) for msg in conn.sent)
    player_role = client.role("player@v1")
    assert isinstance(player_role, PlayerV1Role)
    assert player_role._pending_stream_start is False  # noqa: SLF001
    assert player_role.preferred_codec == AudioCodec.FLAC


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_player_format_request_uses_client_priority_order_when_codec_missing(
    mock_server: MagicMock,
) -> None:
    """When request omits codec, base format should follow client order (not Opus-first)."""
    owner = MagicMock()
    owner.client_id = "owner"
    owner.name = "owner"
    owner.group = MagicMock()
    owner.group.stop = AsyncMock()
    SendspinGroup(mock_server, owner)

    client, _conn = _make_player_client(
        mock_server,
        "p1",
        supported_formats=[
            SupportedAudioFormat(
                codec=AudioCodec.FLAC, sample_rate=44100, bit_depth=24, channels=2
            ),
            SupportedAudioFormat(
                codec=AudioCodec.OPUS, sample_rate=48000, bit_depth=16, channels=2
            ),
            SupportedAudioFormat(codec=AudioCodec.PCM, sample_rate=48000, bit_depth=16, channels=2),
        ],
    )

    request = StreamRequestFormatPayload(
        player=StreamRequestFormatPlayer(
            sample_rate=32000,
            channels=1,
            bit_depth=16,
        )
    )

    for role in client.active_roles:
        role.on_stream_request_format(request)

    player_role = client.role("player@v1")
    assert isinstance(player_role, PlayerV1Role)

    # Requested codec falls back to client's first compatible format codec (FLAC).
    assert player_role.preferred_codec == AudioCodec.FLAC
    # Since FLAC@32kHz mono 16-bit was unsupported by client list, it falls back to base format.
    assert player_role.preferred_format is not None
    assert player_role.preferred_format.sample_rate == 44100
    assert player_role.preferred_format.channels == 2
    assert player_role.preferred_format.bit_depth == 24


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_player_partial_format_request_preserves_unchanged_fields(
    mock_server: MagicMock,
) -> None:
    """A partial stream/request-format keeps prior values for omitted fields."""
    client, _conn = _make_player_client(
        mock_server,
        "p1",
        supported_formats=[
            SupportedAudioFormat(
                codec=AudioCodec.FLAC, sample_rate=48000, bit_depth=16, channels=2
            ),
            SupportedAudioFormat(
                codec=AudioCodec.FLAC, sample_rate=44100, bit_depth=16, channels=2
            ),
        ],
    )

    initial = StreamRequestFormatPayload(
        player=StreamRequestFormatPlayer(
            codec=AudioCodec.FLAC, sample_rate=48000, channels=2, bit_depth=16
        )
    )
    for role in client.active_roles:
        role.on_stream_request_format(initial)

    partial = StreamRequestFormatPayload(player=StreamRequestFormatPlayer(sample_rate=44100))
    for role in client.active_roles:
        role.on_stream_request_format(partial)

    player_role = client.role("player@v1")
    assert isinstance(player_role, PlayerV1Role)
    assert player_role.preferred_codec == AudioCodec.FLAC
    assert player_role.preferred_format is not None
    assert player_role.preferred_format.sample_rate == 44100
    assert player_role.preferred_format.bit_depth == 16
    assert player_role.preferred_format.channels == 2


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_format_request_resets_buffer_tracker(mock_server: MagicMock) -> None:
    """A mid-stream format change resets buffer tracking and binary timing.

    The client flushes its buffer at the new stream/start, so the writer must
    not pace the replacement audio against the old-format audio still
    registered in the buffer tracker.
    """
    client, _conn = _make_player_client(mock_server, "p1")
    client.group.start_stream()

    player_role = client.role("player@v1")
    assert isinstance(player_role, PlayerV1Role)
    tracker = player_role._buffer_tracker  # noqa: SLF001
    assert tracker is not None

    clock = mock_server.clock
    now_us = clock.now_us()
    tracker.register(now_us + 10_000_000, 100_000, 10_000_000)
    tracker.prune_consumed(now_us)
    assert tracker.buffered_duration_us > 0

    # Request a genuinely different format (FLAC); an identical-format request
    # must be ignored entirely.
    player_role.on_stream_request_format(
        StreamRequestFormatPayload(
            player=StreamRequestFormatPlayer(
                codec=AudioCodec.FLAC,
                sample_rate=48000,
                channels=2,
                bit_depth=16,
            )
        )
    )

    assert tracker.buffered_duration_us == 0


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_noop_format_request_runs_no_boundary(mock_server: MagicMock) -> None:
    """A request for the format already in use must not run the transition.

    The stream/start identity guard would suppress the announcement, so the
    boundary would evict queued audio the client still holds with nothing
    replacing it.
    """
    client, conn = _make_player_client(mock_server, "p1")
    client.group.start_stream()

    player_role = client.role("player@v1")
    assert isinstance(player_role, PlayerV1Role)
    tracker = player_role._buffer_tracker  # noqa: SLF001
    assert tracker is not None
    clock = mock_server.clock
    now_us = clock.now_us()
    tracker.register(now_us + 10_000_000, 100_000, 10_000_000)
    tracker.prune_consumed(now_us)

    # PCM 48kHz stereo 16-bit is the default negotiated format for this client.
    player_role.on_stream_request_format(
        StreamRequestFormatPayload(
            player=StreamRequestFormatPlayer(
                codec=AudioCodec.PCM,
                sample_rate=48000,
                channels=2,
                bit_depth=16,
            )
        )
    )

    assert tracker.buffered_duration_us > 0
    assert conn.dropped_pending_binary == []


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_format_request_survives_client_state_without_format(
    mock_server: MagicMock,
) -> None:
    """A pre-#195 client's later player state, which never has format, keeps its request."""
    client, _conn = _make_player_client(mock_server, "p1")
    player_role = client.role("player@v1")
    assert isinstance(player_role, PlayerV1Role)
    player_role.on_stream_request_format(
        StreamRequestFormatPayload(player=StreamRequestFormatPlayer(codec=AudioCodec.FLAC))
    )

    player_role.on_client_state(ClientStatePayload(player=PlayerStatePayload(volume=20)))
    assert player_role.preferred_codec == AudioCodec.FLAC

    # Once the client sends format itself, an absent format clears the preference.
    flac = SupportedAudioFormat(codec=AudioCodec.FLAC, sample_rate=48000, bit_depth=16, channels=2)
    player_role.on_client_state(ClientStatePayload(player=PlayerStatePayload(format=flac)))
    player_role.on_client_state(ClientStatePayload(player=PlayerStatePayload(volume=20)))
    assert player_role.preferred_codec == AudioCodec.PCM


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_format_request_does_not_beat_operator_override(mock_server: MagicMock) -> None:
    """The operator override still wins over a pre-#195 request."""
    client, conn = _make_player_client(mock_server, "p1")
    client.group.start_stream()
    player_role = client.role("player@v1")
    assert isinstance(player_role, PlayerV1Role)
    assert player_role.set_preferred_format(
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2), AudioCodec.PCM
    )

    player_role.on_stream_request_format(
        StreamRequestFormatPayload(player=StreamRequestFormatPlayer(codec=AudioCodec.FLAC))
    )

    assert player_role.preferred_codec == AudioCodec.PCM
    assert conn.dropped_pending_binary == []


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_unsupported_format_request_is_ignored(mock_server: MagicMock) -> None:
    """A request matching no encodable format leaves the current preference in place."""
    client, conn = _make_player_client(mock_server, "p1")
    client.group.start_stream()
    player_role = client.role("player@v1")
    assert isinstance(player_role, PlayerV1Role)

    player_role.on_stream_request_format(
        StreamRequestFormatPayload(player=StreamRequestFormatPlayer(sample_rate=96000))
    )

    assert player_role.preferred_format == AudioFormat(sample_rate=48000, bit_depth=16, channels=2)
    assert conn.dropped_pending_binary == []


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_undeclared_client_state_format_ends_legacy_retention(mock_server: MagicMock) -> None:
    """An undeclared format is ignored, but still marks the client as sending format."""
    client, _conn = _make_player_client(mock_server, "p1")
    player_role = client.role("player@v1")
    assert isinstance(player_role, PlayerV1Role)
    player_role.on_stream_request_format(
        StreamRequestFormatPayload(player=StreamRequestFormatPlayer(codec=AudioCodec.FLAC))
    )
    undeclared = SupportedAudioFormat(
        codec=AudioCodec.PCM, sample_rate=96000, bit_depth=16, channels=2
    )

    player_role.on_client_state(ClientStatePayload(player=PlayerStatePayload(format=undeclared)))
    assert player_role.preferred_codec == AudioCodec.FLAC

    player_role.on_client_state(ClientStatePayload(player=PlayerStatePayload(volume=20)))
    assert player_role.preferred_codec == AudioCodec.PCM
    assert player_role.preferred_format == AudioFormat(sample_rate=48000, bit_depth=16, channels=2)
