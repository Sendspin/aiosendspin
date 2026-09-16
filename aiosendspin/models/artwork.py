"""
Artwork messages for the Sendspin protocol.

This module contains messages specific to clients with the artwork role, which
handle display of artwork images. Artwork clients receive images in their
preferred format and resolution.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, NamedTuple

from .base import SendspinConfig, SendspinModel
from .types import ArtworkSource, BinaryMessageType, PictureFormat

# Artwork binary flags (byte 1); bits 2-7 are reserved and must be zero.
ARTWORK_FLAG_CANCEL = 0x01
ARTWORK_FLAG_ANNOUNCE = 0x02
ARTWORK_RESERVED_FLAGS = 0xFC
# Largest artwork message: one Noise transport message without fragmentation.
ARTWORK_MAX_MESSAGE_SIZE = 65519
# type(1) + flags(1), the whole of a cancel and the header of a part.
ARTWORK_PREFIX_SIZE = 2
ARTWORK_MAX_PART_DATA_SIZE = ARTWORK_MAX_MESSAGE_SIZE - ARTWORK_PREFIX_SIZE
# Announce (big-endian): type(1) + flags(1) + timestamp_us(8) + total_size(4) = 14 bytes
_ARTWORK_ANNOUNCE_STRUCT = struct.Struct(">BBqI")
ARTWORK_ANNOUNCE_SIZE = _ARTWORK_ANNOUNCE_STRUCT.size


class ArtworkAnnounce(NamedTuple):
    """Fields of an artwork announce message."""

    channel: int
    """Artwork channel number (0-3)."""
    timestamp_us: int
    """Server clock time in microseconds when the image should be displayed."""
    total_size: int
    """Size in bytes of the encoded image; 0 clears the channel."""


def artwork_message_type(channel: int) -> int:
    """Return the binary message type of artwork `channel` (0-3)."""
    return BinaryMessageType.ARTWORK_CHANNEL_0.value + channel


def pack_artwork_announce(channel: int, timestamp_us: int, total_size: int) -> bytes:
    """Return the 14-byte announce of an image of `total_size` bytes on `channel`."""
    return _ARTWORK_ANNOUNCE_STRUCT.pack(
        artwork_message_type(channel), ARTWORK_FLAG_ANNOUNCE, timestamp_us, total_size
    )


def pack_artwork_parts(channel: int, image: bytes) -> Iterator[bytes]:
    """Yield the part messages carrying `image` on `channel`, each within the size cap."""
    prefix = bytes((artwork_message_type(channel), 0))
    view = memoryview(image)
    for offset in range(0, len(image), ARTWORK_MAX_PART_DATA_SIZE):
        yield prefix + view[offset : offset + ARTWORK_MAX_PART_DATA_SIZE]


def pack_artwork_cancel(channel: int) -> bytes:
    """Return the cancel message for `channel`."""
    return bytes((artwork_message_type(channel), ARTWORK_FLAG_CANCEL))


def unpack_artwork_announce(data: bytes) -> ArtworkAnnounce:
    """
    Unpack an artwork announce message.

    Raises ValueError when `data` is not a 14-byte artwork announce.
    """
    if len(data) != ARTWORK_ANNOUNCE_SIZE:
        raise ValueError(f"Expected {ARTWORK_ANNOUNCE_SIZE} bytes, got {len(data)}")
    message_type, flags, timestamp_us, total_size = _ARTWORK_ANNOUNCE_STRUCT.unpack(data)
    channel = message_type - BinaryMessageType.ARTWORK_CHANNEL_0.value
    if not 0 <= channel <= 3 or flags != ARTWORK_FLAG_ANNOUNCE:
        raise ValueError(f"Not an artwork announce: type={message_type} flags={flags:#04x}")
    return ArtworkAnnounce(channel, timestamp_us, total_size)


# Pre-rename dimension keys, superseded by `width`/`height`.
_DIMENSION_ALIASES = {"media_width": "width", "media_height": "height"}


def _rewrite_legacy_dimensions(d: dict[str, Any]) -> dict[str, Any]:
    """Rewrite pre-rename dimension keys onto width/height, recording which were used."""
    normalized = dict(d)
    legacy_keys: list[str] = []
    for legacy_key, current_key in _DIMENSION_ALIASES.items():
        if legacy_key not in normalized:
            continue
        legacy_keys.append(legacy_key)
        value = normalized.pop(legacy_key)
        # Rewrite only when the client didn't also send the current key.
        if current_key not in normalized:
            normalized[current_key] = value
    # Always overwrite so a client cannot spoof the record via the wire.
    normalized["legacy_dimension_keys"] = legacy_keys or None
    return normalized


@dataclass
class ArtworkChannel(SendspinModel):
    """Configuration for a single artwork channel."""

    source: ArtworkSource
    """Artwork source type."""
    format: PictureFormat | None = None
    """Image format identifier. Required unless `source` is `none`."""
    width: int | None = None
    """Width in pixels of the delivered image. Required unless `source` is `none`."""
    height: int | None = None
    """Height in pixels of the delivered image. Required unless `source` is `none`."""
    legacy_dimension_keys: list[str] | None = None
    """Pre-rename dimension keys the parser rewrote, recorded for the server to flag.
    Not part of the wire schema (omitted when None)."""

    @classmethod
    def __pre_deserialize__(cls, d: dict[str, Any]) -> dict[str, Any]:
        """Accept the pre-rename `media_width`/`media_height` spelling."""
        return _rewrite_legacy_dimensions(d)

    def __post_init__(self) -> None:
        """Validate field values."""
        if self.source is not ArtworkSource.NONE and None in (self.format, self.width, self.height):
            raise ValueError(
                f"format, width and height are required for source {self.source.value}"
            )
        if self.width is not None and self.width <= 0:
            raise ValueError(f"width must be positive, got {self.width}")
        if self.height is not None and self.height <= 0:
            raise ValueError(f"height must be positive, got {self.height}")

    class Config(SendspinConfig):
        """Config for parsing json messages."""

        omit_none = True


# Client -> Server: client/state artwork object
@dataclass
class ClientStateArtwork(SendspinModel):
    """Artwork channel configuration the client wants - only if artwork role is active."""

    channels: list[ArtworkChannel]
    """Configuration for each artwork channel (length 1-4), array index is the channel number.

    An index the array does not cover is `source: none`.
    """

    def __post_init__(self) -> None:
        """Validate field values."""
        if not 1 <= len(self.channels) <= 4:
            raise ValueError(f"channels must have 1-4 elements, got {len(self.channels)}")


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
# Client -> Server: client/hello artwork support object
@dataclass
class ClientHelloArtworkSupport(SendspinModel):
    """Artwork support configuration declared by clients predating the client/state object."""

    channels: list[ArtworkChannel]
    """List of supported artwork channels (length 1-4), array index is the channel number."""

    def __post_init__(self) -> None:
        """Validate field values."""
        if not 1 <= len(self.channels) <= 4:
            raise ValueError(f"channels must have 1-4 elements, got {len(self.channels)}")


@dataclass
class StreamArtworkChannelConfig(SendspinModel):
    """Configuration for an artwork channel in stream/start."""

    source: ArtworkSource
    """Artwork source type."""
    format: PictureFormat | None = None
    """Format of the encoded image. Omitted when `source` is `none`."""
    width: int | None = None
    """Width in pixels of the encoded image. Omitted when `source` is `none`."""
    height: int | None = None
    """Height in pixels of the encoded image. Omitted when `source` is `none`."""

    class Config(SendspinConfig):
        """Config for parsing json messages."""

        omit_none = True


# Server -> Client: stream/start artwork object
@dataclass
class StreamStartArtwork(SendspinModel):
    """
    Artwork object in stream/start message.

    Sent to clients with the artwork role.
    """

    channels: list[StreamArtworkChannelConfig]
    """Configuration for each artwork channel (at most 4), array index is the channel number.

    An index the array does not cover is not streamed.
    """


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
# Client -> Server: stream/request-format artwork object
@dataclass
class StreamRequestFormatArtwork(SendspinModel):
    """Request the server to change artwork format for a specific channel."""

    channel: int
    """Channel number (0-3) corresponding to the channel index declared in artwork client/hello."""
    source: ArtworkSource | None = None
    """Artwork source type."""
    format: PictureFormat | None = None
    """Requested image format identifier."""
    width: int | None = None
    """Requested width in pixels."""
    height: int | None = None
    """Requested height in pixels."""
    legacy_dimension_keys: list[str] | None = None
    """Pre-rename dimension keys the parser rewrote, recorded for the role to flag.
    Not part of the wire schema (omitted when None)."""

    @classmethod
    def __pre_deserialize__(cls, d: dict[str, Any]) -> dict[str, Any]:
        """Accept the pre-rename `media_width`/`media_height` spelling."""
        return _rewrite_legacy_dimensions(d)

    def __post_init__(self) -> None:
        """Validate field values."""
        if not 0 <= self.channel <= 3:
            raise ValueError(f"channel must be 0-3, got {self.channel}")

    class Config(SendspinConfig):
        """Config for parsing json messages."""

        omit_none = True
