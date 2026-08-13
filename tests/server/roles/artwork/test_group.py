"""Tests for ArtworkGroupRole events."""

from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock

import pytest
from PIL import Image

from aiosendspin.models.types import ArtworkSource, PictureFormat
from aiosendspin.server.roles.artwork.events import ArtworkClearedEvent, ArtworkUpdatedEvent
from aiosendspin.server.roles.artwork.group import ArtworkGroupRole
from aiosendspin.server.roles.artwork.types import ArtworkRoleProtocol


def _make_group_stub() -> MagicMock:
    group = MagicMock()
    group._server = MagicMock()  # noqa: SLF001
    group._server.clock.now_us.return_value = 123_456  # noqa: SLF001
    return group


@pytest.mark.asyncio
async def test_set_album_artwork_emits_updated_event() -> None:
    """Setting album artwork emits ArtworkUpdatedEvent."""
    group = _make_group_stub()
    agr = ArtworkGroupRole(group)

    image = Image.new("RGB", (320, 240), (255, 0, 0))
    await agr.set_album_artwork(image)

    group._signal_event.assert_called_once()  # noqa: SLF001
    event = group._signal_event.call_args.args[0]  # noqa: SLF001
    assert isinstance(event, ArtworkUpdatedEvent)
    assert event.source == ArtworkSource.ALBUM
    assert event.timestamp_us == 123_456
    assert event.width == 320
    assert event.height == 240


@pytest.mark.asyncio
async def test_clear_album_artwork_emits_cleared_event() -> None:
    """Clearing album artwork emits ArtworkClearedEvent."""
    group = _make_group_stub()
    agr = ArtworkGroupRole(group)
    image = Image.new("RGB", (100, 100), (0, 255, 0))
    await agr.set_album_artwork(image)
    group._signal_event.reset_mock()  # noqa: SLF001

    await agr.set_album_artwork(None)

    group._signal_event.assert_called_once()  # noqa: SLF001
    event = group._signal_event.call_args.args[0]  # noqa: SLF001
    assert isinstance(event, ArtworkClearedEvent)
    assert event.source == ArtworkSource.ALBUM
    assert event.timestamp_us == 123_456
    assert agr.get_album_artwork() is None


def test_encode_letterboxes_to_the_declared_dimensions() -> None:
    """An off-aspect source is padded to exactly the declared size, never cropped."""
    agr = ArtworkGroupRole(_make_group_stub())
    image = Image.new("RGB", (400, 400), (255, 0, 0))

    encoded = agr._process_and_encode_image(image, 800, 480, PictureFormat.PNG)  # noqa: SLF001

    with Image.open(BytesIO(encoded)) as decoded:
        assert decoded.size == (800, 480)
        assert decoded.convert("RGB").getpixel((0, 0)) == (0, 0, 0)


def test_encode_still_supports_bmp() -> None:
    """A client declaring the removed 'bmp' format still gets a BMP image."""
    agr = ArtworkGroupRole(_make_group_stub())
    image = Image.new("RGB", (400, 400), (255, 0, 0))

    encoded = agr._process_and_encode_image(image, 320, 320, PictureFormat.BMP)  # noqa: SLF001

    with Image.open(BytesIO(encoded)) as decoded:
        assert decoded.format == "BMP"
