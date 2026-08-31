"""Models for enum types used by Sendspin."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from mashumaro.types import Discriminator

from .base import SendspinConfig, SendspinModel


# Base message classes
@dataclass
class ClientMessage(SendspinModel):
    """Base class for client messages."""

    class Config(SendspinConfig):
        """Config for parsing json messages."""

        discriminator = Discriminator(field="type", include_subtypes=True)


@dataclass
class ServerMessage(SendspinModel):
    """Base class for server messages."""

    def merge(self, _other: ServerMessage) -> ServerMessage | None:
        """Merge two messages of the same type when safe, else return None."""
        return None

    class Config(SendspinConfig):
        """Config for parsing json messages."""

        discriminator = Discriminator(field="type", include_subtypes=True)


# Helpers for discerning between null and undefined fields in messages
@dataclass
class UndefinedField(SendspinModel):
    """Marker type to indicate undefined fields in messages."""


_UNDEFINED_SINGLETON = UndefinedField()


def undefined_field() -> UndefinedField:
    """Return the singleton UndefinedField instance."""
    return _UNDEFINED_SINGLETON


# Enums


class TrustLevel(Enum):
    """Trust a client extends to a server, governing allowed management operations."""

    NONE = "none"
    USER = "user"


class Roles(Enum):
    """Client roles with explicit versioning."""

    PLAYER = "player@v1"
    """
    Receives audio and plays it in sync.

    Has its own volume and mute state and preferred format settings.
    """
    CONTROLLER = "controller@v1"
    """Controls the Sendspin group this client is part of."""
    METADATA = "metadata@v1"
    """Displays text metadata describing the currently playing audio."""
    ARTWORK = "artwork@v1"
    """Displays artwork images. Has preferred format for images."""
    VISUALIZER = "visualizer@v1"
    """
    Visualizes music. Has preferred format for audio features (FFT spectrum,
    loudness, beats, peaks, pitch).
    """
    COLOR = "color@v1"
    """Receives colors derived from the current audio."""
    SOURCE = "source@v1"
    """Captures audio from a local input and streams it to the server."""


class BinaryMessageType(Enum):
    """Enum for Binary Message Types."""

    # Player role (bits 000001xx, IDs 4-7):
    AUDIO_CHUNK = 4
    """Audio chunks with timestamps (Player role, slot 0)."""

    # Artwork role (bits 000010xx, IDs 8-11):
    ARTWORK_CHANNEL_0 = 8
    """Artwork channel 0 (Artwork role, slot 0)."""
    ARTWORK_CHANNEL_1 = 9
    """Artwork channel 1 (Artwork role, slot 1)."""
    ARTWORK_CHANNEL_2 = 10
    """Artwork channel 2 (Artwork role, slot 2)."""
    ARTWORK_CHANNEL_3 = 11
    """Artwork channel 3 (Artwork role, slot 3)."""

    # Visualizer role (bits 00010xxx, IDs 16-23):
    VISUALIZATION_LOUDNESS = 16
    """Loudness frame (Visualizer role, slot 0). Also reused for the legacy
    `visualizer@_draft_r1` `VISUALIZATION_DATA` blob — same wire byte, the
    negotiated role's `get_binary_handling` selects the framing."""
    VISUALIZATION_DATA = 16  # noqa: PIE796
    """Alias of `VISUALIZATION_LOUDNESS` for the legacy draft_r1 wire."""
    VISUALIZATION_BEAT = 17
    """Musical beat event (Visualizer role, slot 1)."""
    VISUALIZATION_F_PEAK = 18
    """Dominant frequency + amplitude (Visualizer role, slot 2)."""
    VISUALIZATION_SPECTRUM = 19
    """Display-binned spectrum (Visualizer role, slot 3)."""
    VISUALIZATION_PEAK = 20
    """Energy onset event with strength (Visualizer role, slot 4)."""
    VISUALIZATION_PITCH = 21
    """Perceived pitch (MIDI 8.8 + confidence) (Visualizer role, slot 5)."""

    # Source role (bits 000011xx, IDs 12-15):
    SOURCE_AUDIO_CHUNK = 12
    """Encoded audio frame captured by a source client (Source role, slot 0)."""


class RepeatMode(Enum):
    """Enum for Repeat Modes."""

    OFF = "off"
    ONE = "one"
    ALL = "all"


class SignalState(Enum):
    """Line-sensing/signal presence reported by a source that supports it."""

    PRESENT = "present"
    ABSENT = "absent"


class PlaybackStateType(Enum):
    """Enum for Playback States."""

    PLAYING = "playing"
    PAUSED = "paused"
    STOPPED = "stopped"


class AudioCodec(Enum):
    """Enum for Audio Codecs."""

    OPUS = "opus"
    FLAC = "flac"
    PCM = "pcm"


class PlayerCommand(Enum):
    """Enum for Player Commands."""

    VOLUME = "volume"
    MUTE = "mute"
    SET_STATIC_DELAY = "set_static_delay"


class MediaCommand(Enum):
    """Enum for Media Commands."""

    PLAY = "play"
    PAUSE = "pause"
    STOP = "stop"
    NEXT = "next"
    PREVIOUS = "previous"
    VOLUME = "volume"
    MUTE = "mute"
    REPEAT_OFF = "repeat_off"
    REPEAT_ONE = "repeat_one"
    REPEAT_ALL = "repeat_all"
    SHUFFLE = "shuffle"
    UNSHUFFLE = "unshuffle"
    SWITCH = "switch"
    SEEK = "seek"
    SEEK_RELATIVE = "seek_relative"


class PictureFormat(Enum):
    """Supported image formats for artwork/media art."""

    JPEG = "jpeg"
    PNG = "png"


class ArtworkSource(Enum):
    """Artwork source type."""

    ALBUM = "album"
    """Album artwork."""
    ARTIST = "artist"
    """Artist artwork."""
    NONE = "none"
    """No artwork - channel disabled."""


class ConnectionReason(Enum):
    """Reason for server connection (multi-server support)."""

    DISCOVERY = "discovery"
    """Server is connecting for general availability (e.g., initial discovery, reconnection)."""
    PAIRING = "pairing"
    """Server is performing a pairing handshake."""
    PLAYBACK = "playback"
    """Server needs client for active or upcoming playback."""
    MANAGEMENT = "management"
    """Server is opening a dedicated management session."""


class Activity(Enum):
    """A currently-active purpose on a connection, declared in server/activate."""

    PLAYBACK = "playback"
    """Active or upcoming playback."""
    PAIRING = "pairing"
    """A pairing exchange."""
    MANAGEMENT = "management"
    """A dedicated management session."""


class GoodbyeReason(Enum):
    """Reason for client disconnect (multi-server support)."""

    ANOTHER_SERVER = "another_server"
    """Client is switching to a different Sendspin server."""
    SHUTDOWN = "shutdown"
    """Client is shutting down."""
    RESTART = "restart"
    """Client is restarting and will reconnect."""
    USER_REQUEST = "user_request"
    """User explicitly requested to disconnect from this server."""
    UNAUTHORIZED = "unauthorized"
    """Server requested an activity the client's trust level does not permit."""
    PAIRING_REQUIRED = "pairing_required"
    """Server requested playback but the client requires pairing first."""
    CONCURRENT_ATTEMPT = "concurrent_attempt"
    """Incoming connection rejected because another connection is already admitted."""
    UNPAIRED = "unpaired"
    """Client processed server/unpair from this server."""


class PairMethod(Enum):
    """A pairing method a client offers or a server selects."""

    DYNAMIC_PAIRING_CODE = "dynamic_pairing_code"
    PAIRING_PSK = "pairing_psk"
    STATIC_PAIRING_CODE = "static_pairing_code"


class PairingCodeFormat(Enum):
    """An emission format for the dynamic pairing code."""

    DIGITS = "digits"
    QR_CODE = "qr_code"


class PairAbortReason(Enum):
    """Reason a pairing attempt was aborted."""

    ATTEMPT_TIMEOUT = "attempt_timeout"
    CONCURRENT_ATTEMPT = "concurrent_attempt"
    METHOD_NOT_SUPPORTED = "method_not_supported"
    PAIRING_CODE_MISMATCH = "pairing_code_mismatch"
    USER_CANCELLED = "user_cancelled"


# The sender closes the connection after these abort reasons; every other reason keeps it open.
CLOSING_ABORT_REASONS: frozenset[PairAbortReason] = frozenset({PairAbortReason.CONCURRENT_ATTEMPT})


class ManagementResult(Enum):
    """Result code carried by management/result."""

    OK = "ok"
    PERMISSION_DENIED = "permission_denied"
    ALREADY_EXISTS = "already_exists"
    INVALID = "invalid"
    NOT_FOUND = "not_found"
    STORAGE_EXHAUSTED = "storage_exhausted"


# Role ID helpers for spec-compliant role negotiation
# Wire format uses versioned strings like "player@v1", not the Roles enum directly


def role_family(role_id: str) -> str:
    """Extract role family from a versioned role ID.

    Examples:
        role_family("player@v1") -> "player"
        role_family("controller@v2") -> "controller"
    """
    return role_id.split("@", 1)[0]


def has_role_family(role_family_name: str, supported_roles: list[str]) -> bool:
    """Check if a role family is present in the supported roles list."""
    return any(role_family(r) == role_family_name for r in supported_roles)


def has_role(role_id: str, supported_roles: list[str]) -> bool:
    """Check if a role family is present in the supported roles list.

    Checks by family name, so "player@v2" in supported_roles matches
    a check for "player@v1" family.

    Examples:
        has_role("player@v1", ["player@v1", "metadata@v1"]) -> True
        has_role("player@v1", ["controller@v1"]) -> False
    """
    return has_role_family(role_family(role_id), supported_roles)
