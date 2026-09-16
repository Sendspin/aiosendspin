"""The client/state artwork object and the stream/start artwork echo."""

from __future__ import annotations

import pytest

from aiosendspin.models.artwork import (
    ArtworkChannel,
    ClientStateArtwork,
    StreamArtworkChannelConfig,
    StreamStartArtwork,
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
