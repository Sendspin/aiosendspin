"""
Player messages for the Sendspin protocol.

This module contains messages specific to clients with the player role, which
handle audio output and synchronized playback. Player clients receive timestamped
audio data, manage their own volume and mute state, and can request different
audio formats based on their capabilities and current conditions.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any, NamedTuple

from .base import SendspinConfig, SendspinModel, split_enum_values
from .types import AudioCodec, BinaryMessageType, PlayerCommand

# Pre-rename delay key, superseded by `output_delay_ms`.
_LEGACY_DELAY_KEY = "static_delay_ms"


def _rewrite_legacy_delay_key(d: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of `d` with the pre-rename `static_delay_ms` key on `output_delay_ms`."""
    normalized = dict(d)
    if _LEGACY_DELAY_KEY not in normalized:
        return normalized
    value = normalized.pop(_LEGACY_DELAY_KEY)
    # Rewrite only when the sender didn't also send the current key.
    if "output_delay_ms" not in normalized:
        normalized["output_delay_ms"] = value
    return normalized


# Audio chunk header (big-endian): message_type(1) + timestamp_us(8) + send_ahead(4) = 13 bytes
PLAYER_AUDIO_HEADER_FORMAT = ">BqI"
_PLAYER_AUDIO_HEADER_STRUCT = struct.Struct(PLAYER_AUDIO_HEADER_FORMAT)
PLAYER_AUDIO_HEADER_SIZE = _PLAYER_AUDIO_HEADER_STRUCT.size
_SEND_AHEAD_STRUCT = struct.Struct(">I")
_SEND_AHEAD_OFFSET = PLAYER_AUDIO_HEADER_SIZE - _SEND_AHEAD_STRUCT.size
# Saturated send_ahead: the lead exceeds what the field can hold.
SEND_AHEAD_MAX = 0xFFFFFFFF


class PlayerAudioHeader(NamedTuple):
    """Header of a player audio chunk."""

    message_type: int
    timestamp_us: int
    """Server clock time in microseconds when the first sample should be output."""
    send_ahead: int
    """Microseconds from server transmit to `timestamp_us`; 0 and SEND_AHEAD_MAX are saturated."""


def compute_send_ahead(timestamp_us: int, now_us: int) -> int:
    """Return the `send_ahead` for a chunk sent at `now_us`, saturated to the uint32 range."""
    return max(0, min(timestamp_us - now_us, SEND_AHEAD_MAX))


def pack_player_audio_header(timestamp_us: int, send_ahead: int) -> bytes:
    """Return the 13-byte player audio chunk header."""
    return _PLAYER_AUDIO_HEADER_STRUCT.pack(
        BinaryMessageType.AUDIO_CHUNK.value, timestamp_us, send_ahead
    )


def pack_player_audio_frame(timestamp_us: int, payload: bytes) -> bytearray:
    """Return a player audio frame whose send_ahead is 0 until stamp_send_ahead() sets it."""
    frame = bytearray(PLAYER_AUDIO_HEADER_SIZE + len(payload))
    _PLAYER_AUDIO_HEADER_STRUCT.pack_into(
        frame, 0, BinaryMessageType.AUDIO_CHUNK.value, timestamp_us, 0
    )
    frame[PLAYER_AUDIO_HEADER_SIZE:] = payload
    return frame


def stamp_send_ahead(frame: bytearray, send_ahead: int) -> None:
    """Write `send_ahead` into a frame from pack_player_audio_frame()."""
    _SEND_AHEAD_STRUCT.pack_into(frame, _SEND_AHEAD_OFFSET, send_ahead)


def unpack_player_audio_header(data: bytes) -> PlayerAudioHeader:
    """
    Unpack the player audio chunk header from the start of `data`.

    Raises ValueError when `data` is shorter than PLAYER_AUDIO_HEADER_SIZE.
    """
    if len(data) < PLAYER_AUDIO_HEADER_SIZE:
        raise ValueError(f"Expected at least {PLAYER_AUDIO_HEADER_SIZE} bytes, got {len(data)}")
    return PlayerAudioHeader(*_PLAYER_AUDIO_HEADER_STRUCT.unpack_from(data))


# Client -> Server client/hello player support object
@dataclass
class SupportedAudioFormat(SendspinModel):
    """Supported audio format configuration."""

    codec: AudioCodec
    """Codec identifier."""
    channels: int
    """Supported number of channels (e.g., 1 = mono, 2 = stereo)."""
    sample_rate: int
    """Sample rate in Hz (e.g., 44100, 48000)."""
    bit_depth: int
    """Bit depth for this format (e.g., 16, 24)."""

    def __post_init__(self) -> None:
        """Validate field values."""
        if self.channels <= 0:
            raise ValueError(f"channels must be positive, got {self.channels}")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.bit_depth <= 0:
            raise ValueError(f"bit_depth must be positive, got {self.bit_depth}")

    def matches(self, other: SupportedAudioFormat) -> bool:
        """Return whether `other` names the same format; `bit_depth` is ignored for opus."""
        return (
            self.codec == other.codec
            and self.channels == other.channels
            and self.sample_rate == other.sample_rate
            and (self.codec == AudioCodec.OPUS or self.bit_depth == other.bit_depth)
        )


@dataclass
class ClientHelloPlayerSupport(SendspinModel):
    """Player support configuration - only if player role is set."""

    supported_formats: list[SupportedAudioFormat]
    """List of supported audio formats in priority order (first is preferred)."""
    buffer_capacity: int
    """Max size in bytes of compressed audio messages in the buffer that are yet to be played."""
    # DEPRECATED(spec-pr-177): remove in aiosendspin <version>
    supported_commands: list[PlayerCommand] | None = None
    """Pre-#177 hello-level commands, subset of: 'volume', 'mute'.

    Not part of the current wire schema: players declare commands in the
    client/state player object. Tolerated on input only; never sent.
    """

    def __post_init__(self) -> None:
        """Validate field values."""
        if self.buffer_capacity <= 0:
            raise ValueError(f"buffer_capacity must be positive, got {self.buffer_capacity}")

        if not self.supported_formats:
            raise ValueError("supported_formats cannot be empty")

        valid_hello_commands = {PlayerCommand.VOLUME, PlayerCommand.MUTE}
        if self.supported_commands:
            invalid = [c for c in self.supported_commands if c not in valid_hello_commands]
            if invalid:
                raise ValueError(f"Invalid hello supported_commands: {invalid}")

    class Config(SendspinConfig):
        """Config for parsing json messages."""

        omit_none = True


# Client -> Server: client/state player object
@dataclass
class PlayerStatePayload(SendspinModel):
    """Player object in client/state message."""

    # DEPRECATED(before-spec-pr-50): Remove once all clients send state at client level.
    # State should now be sent at the ClientStatePayload level, not in the player object.
    state: str | None = None
    """
    State of the player - should always be 'synchronized' unless there is
    an error preventing current or future playback (unable to keep up,
    issues keeping the clock in sync, etc).

    DEPRECATED: State should now be sent at the client/state level, not here.
    """
    volume: int | None = None
    """Volume range 0-100, only included if 'volume' in supported_commands."""
    muted: bool | None = None
    """Mute state, only included if 'mute' in supported_commands."""
    output_delay_ms: int | None = None
    """Output delay in milliseconds (0-5000). Required on the initial state message;
    omitted in incremental updates means unchanged."""
    required_lead_time_ms: int | None = None
    """Minimum startup lead time in milliseconds (non-negative). Required on the initial state
    message; omitted in incremental updates means unchanged.

    Measured from the server transmit time of the start/restart trigger (stream/start
    or stream/clear) to the timestamp of the first subsequent audio chunk. Covers codec
    init, decode warmup, audio backend buffering, and DAC latency. Excludes output_delay_ms.
    """
    min_buffer_ms: int | None = None
    """Requested minimum ongoing buffer duration in milliseconds (non-negative). Required on
    the initial state message; omitted in incremental updates means unchanged.

    Maintained during playback (primarily for live streams) to absorb network jitter and
    decode/playback timing variance. Excludes output_delay_ms.
    """
    supported_commands: list[PlayerCommand] | None = None
    """Commands the server may send, subset of: 'volume', 'mute', 'set_output_delay'.

    Required on the initial state message and empty when the player accepts no
    commands; omitted in incremental updates means unchanged.
    """
    legacy_delay_key: str | None = None
    """Pre-rename delay key the parser rewrote, recorded for the role to flag.
    Not part of the wire schema (omitted when None)."""
    format: SupportedAudioFormat | None = None
    """Format the player currently prefers, one of its hello `supported_formats`.

    Absent means no preference: the server selects by `supported_formats` priority.
    """
    ignored_commands: list[str] | None = None
    """Supported commands this implementation does not recognize, dropped during parse
    and recorded for the role to flag. Not part of the wire schema (omitted when None)."""

    @classmethod
    def __pre_deserialize__(cls, d: dict[str, Any]) -> dict[str, Any]:
        """Accept the pre-rename `static_delay_ms` spelling, recording that it was used.

        Supported commands this implementation does not recognize are dropped and recorded.
        """
        normalized = _rewrite_legacy_delay_key(d)
        # Always overwrite so a client cannot spoof the records via the wire.
        normalized["legacy_delay_key"] = _LEGACY_DELAY_KEY if _LEGACY_DELAY_KEY in d else None
        commands, ignored = split_enum_values(d.get("supported_commands"), PlayerCommand)
        if "supported_commands" in d:
            normalized["supported_commands"] = commands
        normalized["ignored_commands"] = ignored or None
        return normalized

    def __post_init__(self) -> None:
        """Validate field values."""
        if self.volume is not None and not 0 <= self.volume <= 100:
            raise ValueError(f"Volume must be in range 0-100, got {self.volume}")
        if self.output_delay_ms is not None and not 0 <= self.output_delay_ms <= 5000:
            raise ValueError(f"output_delay_ms must be in range 0-5000, got {self.output_delay_ms}")
        if self.required_lead_time_ms is not None and self.required_lead_time_ms < 0:
            raise ValueError(
                f"required_lead_time_ms must be non-negative, got {self.required_lead_time_ms}"
            )
        if self.min_buffer_ms is not None and self.min_buffer_ms < 0:
            raise ValueError(f"min_buffer_ms must be non-negative, got {self.min_buffer_ms}")

    class Config(SendspinConfig):
        """Config for parsing json messages."""

        omit_none = True


# Server -> Client: server/command player object
@dataclass
class PlayerCommandPayload(SendspinModel):
    """Player object in server/command message."""

    command: PlayerCommand
    """
    Command - must be one of the values listed in supported_commands from the
    latest client/state player object.
    """
    volume: int | None = None
    """Volume range 0-100, only set if command is volume."""
    mute: bool | None = None
    """True to mute, false to unmute, only set if command is mute."""
    output_delay_ms: int | None = None
    """Delay in milliseconds (0-5000), only set if command is set_output_delay."""

    @classmethod
    def __pre_deserialize__(cls, d: dict[str, Any]) -> dict[str, Any]:
        """Accept the pre-rename `static_delay_ms` spelling from a pre-rename server."""
        return _rewrite_legacy_delay_key(d)

    def __post_init__(self) -> None:
        """Validate field values and command consistency."""
        if self.command == PlayerCommand.VOLUME:
            if self.volume is None:
                raise ValueError("Volume must be provided when command is 'volume'")
            if not 0 <= self.volume <= 100:
                raise ValueError(f"Volume must be in range 0-100, got {self.volume}")
        elif self.volume is not None:
            raise ValueError(f"Volume should not be provided for command '{self.command.value}'")

        if self.command == PlayerCommand.MUTE:
            if self.mute is None:
                raise ValueError("Mute must be provided when command is 'mute'")
        elif self.mute is not None:
            raise ValueError(f"Mute should not be provided for command '{self.command.value}'")

        if self.command in (PlayerCommand.SET_OUTPUT_DELAY, PlayerCommand.SET_STATIC_DELAY):
            if self.output_delay_ms is None:
                raise ValueError(
                    f"output_delay_ms must be provided when command is '{self.command.value}'"
                )
            if not 0 <= self.output_delay_ms <= 5000:
                raise ValueError(
                    f"output_delay_ms must be in range 0-5000, got {self.output_delay_ms}"
                )
        elif self.output_delay_ms is not None:
            raise ValueError(
                f"output_delay_ms should not be provided for command '{self.command.value}'"
            )

    def __post_serialize__(self, d: dict[str, Any]) -> dict[str, Any]:
        """Serialize `output_delay_ms` under the wire key matching `command`.

        Lets a caller address a client that only declared the pre-rename
        `set_static_delay` command by constructing this payload with
        `command=PlayerCommand.SET_STATIC_DELAY` — the Python-side field stays
        `output_delay_ms` either way.
        """
        if self.command == PlayerCommand.SET_STATIC_DELAY and "output_delay_ms" in d:
            d["static_delay_ms"] = d.pop("output_delay_ms")
        return d

    class Config(SendspinConfig):
        """Config for parsing json messages."""

        omit_none = True


# Client -> Server stream/request-format player object
# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
@dataclass
class StreamRequestFormatPlayer(SendspinModel):
    """Request different player stream format (upgrade or downgrade)."""

    codec: AudioCodec | None = None
    """Requested codec."""
    sample_rate: int | None = None
    """Requested sample rate."""
    channels: int | None = None
    """Requested channels."""
    bit_depth: int | None = None
    """Requested bit depth."""

    class Config(SendspinConfig):
        """Config for parsing json messages."""

        omit_none = True


# Server -> Client stream/start player object
@dataclass
class StreamStartPlayer(SendspinModel):
    """Player object in stream/start message."""

    codec: AudioCodec
    """Codec to be used."""
    sample_rate: int
    """Sample rate to be used."""
    channels: int
    """Channels to be used."""
    bit_depth: int
    """Bit depth to be used."""
    codec_header: str | None = None
    """Base64 encoded codec header (if necessary; e.g., FLAC)."""

    class Config(SendspinConfig):
        """Config for parsing json messages."""

        omit_none = True
