"""Tests for PlayerV1Role stream lifecycle message payloads."""

from __future__ import annotations

from unittest.mock import MagicMock

from aiosendspin.models import AudioCodec
from aiosendspin.models.core import StreamClearMessage, StreamEndMessage, StreamStartMessage
from aiosendspin.models.types import BinaryMessageType
from aiosendspin.server.roles import AudioChunk, AudioRequirements, PlayerV1Role
from aiosendspin.server.roles.player.audio_transformers import PcmPassthrough


def test_player_role_on_stream_clear_uses_role_family() -> None:
    """PlayerV1Role.on_stream_clear() sends stream/clear with unversioned role family."""
    client = MagicMock()
    client.send_role_message = MagicMock()

    role = PlayerV1Role(client=client)
    role._client.connection = MagicMock()  # noqa: SLF001
    role._buffer_tracker = None  # noqa: SLF001
    role.on_stream_clear()

    _role, msg = client.send_role_message.call_args.args
    assert isinstance(msg, StreamClearMessage)
    assert msg.payload.roles == ["player"]


def test_player_role_on_stream_end_uses_role_family() -> None:
    """PlayerV1Role.on_stream_end() targets the player role family."""
    client = MagicMock()
    client.send_role_message = MagicMock()

    role = PlayerV1Role(client=client)
    role._client.connection = MagicMock()  # noqa: SLF001
    role._buffer_tracker = None  # noqa: SLF001
    role.on_stream_end()

    _role, msg = client.send_role_message.call_args.args
    assert isinstance(msg, StreamEndMessage)
    assert msg.payload.roles == ["player"]


def test_player_role_on_deactivate_ends_active_stream() -> None:
    """on_deactivate() sends stream/end for an active stream, and nothing otherwise."""
    client = MagicMock()
    client.send_role_message = MagicMock()

    role = PlayerV1Role(client=client)
    role._client.connection = MagicMock()  # noqa: SLF001
    role._buffer_tracker = None  # noqa: SLF001

    role._stream_started = True  # noqa: SLF001
    role.on_deactivate()
    _role, msg = client.send_role_message.call_args.args
    assert isinstance(msg, StreamEndMessage)
    assert msg.payload.roles == ["player"]

    # on_stream_end cleared the flag, so a second deactivate is a no-op.
    client.send_role_message.reset_mock()
    role.on_deactivate()
    client.send_role_message.assert_not_called()


def test_player_role_on_audio_chunk_defers_header_and_tracks_duration() -> None:
    """on_audio_chunk defers the header to the connection and tracks duration accurately."""

    class _Tracker:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []

        def register(self, timestamp_us: int, byte_count: int) -> None:
            self.calls.append((timestamp_us, byte_count))

        def reset(self) -> None:
            return

    tracker = _Tracker()

    sent: list[tuple[bytes, int, int, bool]] = []
    client = MagicMock()
    state_store: dict[str, object] = {}

    def get_or_create_role_state(family: str, cls: type[object]) -> object:
        state_store.setdefault(family, cls())
        return state_store[family]

    client.get_or_create_role_state.side_effect = get_or_create_role_state

    def _send_binary(
        data: bytes,
        *,
        role_family: str,  # noqa: ARG001
        timestamp_us: int,
        message_type: int,
        buffer_end_time_us: int | None = None,
        buffer_byte_count: int | None = None,
        duration_us: int | None = None,  # noqa: ARG001
        player_audio_header: bool = False,
    ) -> bool:
        sent.append((data, timestamp_us, message_type, player_audio_header))
        if buffer_end_time_us is not None and buffer_byte_count is not None:
            tracker.register(buffer_end_time_us, buffer_byte_count)
        return True

    client.send_binary = MagicMock(side_effect=_send_binary)
    client.send_role_message = MagicMock()

    role = PlayerV1Role(client=client)
    role._stream_started = True  # noqa: SLF001
    payload = b"\x01\x02\x03"
    timestamp_us = 123_000
    duration_us = 40_000
    byte_count = len(payload)

    chunk = AudioChunk(
        data=payload, timestamp_us=timestamp_us, duration_us=duration_us, byte_count=byte_count
    )
    assert role.on_audio_chunk(chunk) is None
    assert sent, "Expected a binary send"

    # The connection prepends the header at send time.
    assert sent == [(payload, timestamp_us, BinaryMessageType.AUDIO_CHUNK.value, True)]

    assert tracker.calls == [(timestamp_us + duration_us, 13 + byte_count)]


def test_player_role_on_stream_start_drops_without_transport() -> None:
    """on_stream_start() is a no-op when no transport attached."""
    client = MagicMock()
    client.send_role_message = MagicMock()

    role = PlayerV1Role(client=client)
    role._client.connection = None  # noqa: SLF001
    role._audio_requirements = AudioRequirements(  # noqa: SLF001
        sample_rate=48000,
        bit_depth=16,
        channels=2,
        transformer=PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2),
    )

    role.on_stream_start()

    client.send_role_message.assert_not_called()


def test_player_role_on_stream_clear_drops_without_transport() -> None:
    """on_stream_clear() is a no-op for JSON message when no transport attached."""
    client = MagicMock()
    client.send_role_message = MagicMock()

    role = PlayerV1Role(client=client)
    role._client.connection = None  # noqa: SLF001

    role.on_stream_clear()

    client.send_role_message.assert_not_called()


def test_player_role_on_stream_end_drops_without_transport() -> None:
    """on_stream_end() is a no-op for JSON message when no transport attached."""
    client = MagicMock()
    client.send_role_message = MagicMock()

    role = PlayerV1Role(client=client)
    role._client.connection = None  # noqa: SLF001

    role.on_stream_end()

    client.send_role_message.assert_not_called()


# --- Tests for hook-based streaming methods ---


def test_player_role_on_stream_start_sets_pending_flag() -> None:
    """on_stream_start() sets pending flag, message sent on first chunk."""
    client = MagicMock()
    client.send_role_message = MagicMock()
    client.send_binary = MagicMock(return_value=True)

    role = PlayerV1Role(client=client)
    role._client.connection = MagicMock()  # noqa: SLF001
    role._audio_requirements = AudioRequirements(  # noqa: SLF001
        sample_rate=48000,
        bit_depth=16,
        channels=2,
        transformer=PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2),
    )

    role.on_stream_start()

    # Message is deferred until first chunk
    client.send_role_message.assert_not_called()
    assert role._pending_stream_start is True  # noqa: SLF001

    # First chunk triggers the stream/start message
    chunk = AudioChunk(data=b"\x00" * 100, timestamp_us=0, duration_us=25000, byte_count=100)
    role.on_audio_chunk(chunk)

    client.send_role_message.assert_called_once()
    _role, msg = client.send_role_message.call_args.args
    assert isinstance(msg, StreamStartMessage)
    assert msg.payload.player.sample_rate == 48000
    assert msg.payload.player.codec == AudioCodec.PCM
    assert role._pending_stream_start is False  # noqa: SLF001


def test_player_role_on_audio_chunk_sends_binary() -> None:
    """on_audio_chunk() sends a chunk when connection is present."""
    client = MagicMock()
    client.send_binary.return_value = True

    role = PlayerV1Role(client=client)
    role._client.connection = MagicMock()  # noqa: SLF001
    role._stream_started = True  # noqa: SLF001

    chunk = AudioChunk(data=b"audio", timestamp_us=1000, duration_us=25000, byte_count=5)
    result = role.on_audio_chunk(chunk)

    assert result is None
    client.send_binary.assert_called_once()


def _stream_start_messages(client: MagicMock) -> list[StreamStartMessage]:
    return [
        call.args[1]
        for call in client.send_role_message.call_args_list
        if isinstance(call.args[1], StreamStartMessage)
    ]


def test_player_role_skips_redundant_stream_start_when_format_unchanged() -> None:
    """Second on_stream_start with unchanged format must not re-emit stream/start.

    Spec: stream/start during an active stream only updates configuration,
    so re-sending an identical config is wasted traffic. Successor
    PushStreams that preserve format should reach the client with no
    intervening stream/start.
    """
    client = MagicMock()
    client.send_role_message = MagicMock()
    client.send_binary = MagicMock(return_value=True)

    audio_req = AudioRequirements(
        sample_rate=48000,
        bit_depth=16,
        channels=2,
        transformer=PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2),
    )
    role = PlayerV1Role(client=client, audio_requirements=audio_req)
    role._client.connection = MagicMock()  # noqa: SLF001

    # First cycle: initial stream/start.
    role.on_stream_start()
    chunk = AudioChunk(data=b"\x00" * 100, timestamp_us=0, duration_us=25000, byte_count=100)
    role.on_audio_chunk(chunk)
    assert len(_stream_start_messages(client)) == 1

    # Simulate clear (per spec, leaves stream active) followed by a
    # successor PushStream that re-triggers on_stream_start.
    role.on_stream_clear()
    role.on_stream_start()
    role.on_audio_chunk(chunk)

    # No second stream/start should have been emitted.
    assert len(_stream_start_messages(client)) == 1


def test_player_role_resends_stream_start_when_format_changes() -> None:
    """Format change must emit a fresh stream/start carrying the new config."""
    client = MagicMock()
    client.send_role_message = MagicMock()
    client.send_binary = MagicMock(return_value=True)

    audio_req = AudioRequirements(
        sample_rate=48000,
        bit_depth=16,
        channels=2,
        transformer=PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2),
    )
    role = PlayerV1Role(client=client, audio_requirements=audio_req)
    role._client.connection = MagicMock()  # noqa: SLF001

    role.on_stream_start()
    chunk = AudioChunk(data=b"\x00" * 100, timestamp_us=0, duration_us=25000, byte_count=100)
    role.on_audio_chunk(chunk)
    assert len(_stream_start_messages(client)) == 1

    # Swap to a different format (sample rate change).
    role._audio_requirements = AudioRequirements(  # noqa: SLF001
        sample_rate=44100,
        bit_depth=16,
        channels=2,
        transformer=PcmPassthrough(sample_rate=44100, bit_depth=16, channels=2),
    )
    role.on_stream_start()
    role.on_audio_chunk(chunk)

    starts = _stream_start_messages(client)
    assert len(starts) == 2
    assert starts[1].payload.player.sample_rate == 44100


def test_player_role_resends_stream_start_after_stream_end() -> None:
    """stream/end clears last-sent format so the next stream/start fires again."""
    client = MagicMock()
    client.send_role_message = MagicMock()
    client.send_binary = MagicMock(return_value=True)

    audio_req = AudioRequirements(
        sample_rate=48000,
        bit_depth=16,
        channels=2,
        transformer=PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2),
    )
    role = PlayerV1Role(client=client, audio_requirements=audio_req)
    role._client.connection = MagicMock()  # noqa: SLF001

    role.on_stream_start()
    chunk = AudioChunk(data=b"\x00" * 100, timestamp_us=0, duration_us=25000, byte_count=100)
    role.on_audio_chunk(chunk)
    role.on_stream_end()

    role.on_stream_start()
    role.on_audio_chunk(chunk)

    assert len(_stream_start_messages(client)) == 2
