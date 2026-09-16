"""The client/state artwork object and the stream/start artwork echo."""

from __future__ import annotations

import pytest

from aiosendspin.models.artwork import (
    ARTWORK_MAX_MESSAGE_SIZE,
    ARTWORK_MAX_PART_DATA_SIZE,
    ArtworkAnnounce,
    ArtworkChannel,
    ClientStateArtwork,
    StreamArtworkChannelConfig,
    StreamStartArtwork,
    pack_artwork_announce,
    pack_artwork_cancel,
    pack_artwork_parts,
    unpack_artwork_announce,
)
from aiosendspin.models.core import ClientHelloPayload, ClientStatePayload
from aiosendspin.models.types import ArtworkSource, PictureFormat

_ALBUM = {"source": "album", "format": "jpeg", "width": 300, "height": 200}


def test_client_state_artwork_round_trips() -> None:
    """An artwork object parses and serializes with its positional channels."""
    wire = {"available": True, "artwork": {"channels": [{"source": "none"}, _ALBUM]}}

    payload = ClientStatePayload.from_dict(wire)

    assert payload.artwork is not None
    assert payload.artwork.channels == [
        ArtworkChannel(source=ArtworkSource.NONE),
        ArtworkChannel(
            source=ArtworkSource.ALBUM, format=PictureFormat.JPEG, width=300, height=200
        ),
    ]
    assert payload.to_dict() == wire


@pytest.mark.parametrize("missing", ["format", "width", "height"])
def test_client_state_artwork_channel_requires_fields_unless_none(missing: str) -> None:
    """A streamed channel must carry format, width and height."""
    channel = {key: value for key, value in _ALBUM.items() if key != missing}

    with pytest.raises(ValueError, match="required"):
        ArtworkChannel.from_dict(channel)


@pytest.mark.parametrize("count", [0, 5])
def test_client_state_artwork_rejects_channel_count(count: int) -> None:
    """An artwork object must declare 1-4 channels."""
    with pytest.raises(ValueError, match="1-4"):
        ClientStateArtwork.from_dict({"channels": [{"source": "none"}] * count})


def test_client_state_rejects_invalid_artwork_object() -> None:
    """An invalid artwork object fails the whole client/state parse."""
    with pytest.raises(ValueError, match="artwork"):
        ClientStatePayload.from_dict(
            {"available": True, "artwork": {"channels": [{"source": "none"}] * 5}}
        )


def test_client_state_artwork_rejects_nonpositive_dimensions() -> None:
    """Dimensions must be positive."""
    with pytest.raises(ValueError, match="width must be positive"):
        ArtworkChannel.from_dict({**_ALBUM, "width": 0})


def test_stream_start_artwork_none_channel_serializes_bare() -> None:
    """A none channel in stream/start carries only its source."""
    artwork = StreamStartArtwork(
        channels=[
            StreamArtworkChannelConfig(source=ArtworkSource.NONE),
            StreamArtworkChannelConfig(
                source=ArtworkSource.ALBUM, format=PictureFormat.JPEG, width=300, height=200
            ),
        ]
    )

    assert artwork.to_dict() == {"channels": [{"source": "none"}, _ALBUM]}


def test_client_hello_lists_artwork_without_support() -> None:
    """artwork@v1 no longer needs a support object in the hello."""
    hello = ClientHelloPayload.from_dict({"name": "c", "supported_roles": ["artwork@v1"]})

    assert hello.artwork_support is None


def test_artwork_announce_round_trips() -> None:
    """An announce packs to 14 bytes with flags 0x02 and unpacks to its fields."""
    announce = pack_artwork_announce(2, -5, 0x01020304)

    assert announce == bytes([10, 0x02]) + (-5).to_bytes(8, "big", signed=True) + bytes(
        [1, 2, 3, 4]
    )
    assert unpack_artwork_announce(announce) == ArtworkAnnounce(2, -5, 0x01020304)


@pytest.mark.parametrize(
    "data",
    [
        pack_artwork_announce(0, 1, 1)[:-1],
        pack_artwork_announce(0, 1, 1) + b"\x00",
        bytes([8, 0x03]) + bytes(12),
        bytes([7, 0x02]) + bytes(12),
    ],
)
def test_unpack_artwork_announce_rejects_other_messages(data: bytes) -> None:
    """Only a 14-byte message with the announce flag on an artwork type unpacks."""
    with pytest.raises(ValueError):  # noqa: PT011
        unpack_artwork_announce(data)


def test_artwork_parts_split_at_the_size_cap() -> None:
    """Parts carry the image in order, each at most 65519 bytes with flags 0x00."""
    image = bytes(range(256)) * 520

    parts = pack_artwork_parts(3, image)

    assert [len(part) for part in parts] == [
        ARTWORK_MAX_MESSAGE_SIZE,
        ARTWORK_MAX_MESSAGE_SIZE,
        len(image) - 2 * ARTWORK_MAX_PART_DATA_SIZE + 2,
    ]
    assert all(part[:2] == bytes([11, 0x00]) for part in parts)
    assert b"".join(part[2:] for part in parts) == image
    assert pack_artwork_parts(0, b"") == []


def test_artwork_cancel_is_two_bytes() -> None:
    """A cancel is the type byte and flags 0x01."""
    assert pack_artwork_cancel(1) == bytes([9, 0x01])
