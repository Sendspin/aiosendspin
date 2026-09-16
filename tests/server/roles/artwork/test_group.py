"""Tests for ArtworkGroupRole events."""

from __future__ import annotations

import asyncio
import threading
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock

import pytest
from PIL import Image

from aiosendspin.clock import ManualClock
from aiosendspin.models import BINARY_HEADER_SIZE
from aiosendspin.models.artwork import ArtworkChannel, ClientHelloArtworkSupport
from aiosendspin.models.types import ArtworkSource, PictureFormat
from aiosendspin.server.roles.artwork.events import ArtworkClearedEvent, ArtworkUpdatedEvent
from aiosendspin.server.roles.artwork.group import ArtworkGroupRole
from aiosendspin.server.roles.artwork.v1 import MAX_ANNOUNCE_LEAD_US, ArtworkV1Role


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


_CHANNEL = ArtworkChannel(source=ArtworkSource.ALBUM, format=PictureFormat.PNG, width=2, height=2)


class _Member:
    """Artwork role recording what it is sent, as (image size or None, timestamp)."""

    role_id = "artwork@v1"

    def __init__(self, *, can_cancel: bool = True) -> None:
        self.sent: list[tuple[int | None, int] | tuple[str]] = []
        self._can_cancel = can_cancel

    def get_channel_configs(self) -> dict[int, ArtworkChannel]:
        return {0: _CHANNEL}

    def send_artwork(self, _channel: int, image_data: bytes, timestamp_us: int) -> None:
        with Image.open(BytesIO(image_data)) as image:
            self.sent.append((image.getpixel((0, 0))[0], timestamp_us))

    def send_artwork_cleared(self, _channel: int, timestamp_us: int) -> None:
        self.sent.append((None, timestamp_us))

    def cancel_scheduled_artwork(self, _channel: int) -> bool:
        self.sent.append(("cancel",))
        return self._can_cancel

    def uses_single_message_framing(self) -> bool:
        return not self._can_cancel


def _make_scheduling_group() -> tuple[MagicMock, ManualClock]:
    clock = ManualClock(now_us_value=1_000_000)
    group = MagicMock()
    group._server.clock = clock  # noqa: SLF001
    return group, clock


def _image(red: int) -> Image.Image:
    return Image.new("RGB", (2, 2), (red, 0, 0))


def _gate_encoding(agr: ArtworkGroupRole, monkeypatch: pytest.MonkeyPatch) -> threading.Event:
    """Hold every image encode until the returned event is set."""
    release = threading.Event()
    encode = agr._process_and_encode_image  # noqa: SLF001

    def _gated(*args: Any) -> bytes:
        release.wait(5)
        return encode(*args)

    monkeypatch.setattr(agr, "_process_and_encode_image", _gated)
    return release


async def _settle() -> None:
    for _ in range(20):
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_scheduled_artwork_is_sent_and_takes_effect_later() -> None:
    """Future artwork is sent with its timestamp, reported now, and current once due."""
    group, clock = _make_scheduling_group()
    agr = ArtworkGroupRole(group)
    member = _Member()
    agr._members = [member]  # type: ignore[list-item]  # noqa: SLF001
    await agr.set_album_artwork(_image(1))

    await agr.set_album_artwork(_image(2), timestamp_us=1_500_000)

    assert member.sent == [(1, 1_000_000), (2, 1_500_000)]
    event = group._signal_event.call_args.args[0]  # noqa: SLF001
    assert isinstance(event, ArtworkUpdatedEvent)
    assert event.timestamp_us == 1_500_000
    current = agr.get_album_artwork()
    assert current is not None
    assert current.getpixel((0, 0)) == (1, 0, 0)
    clock.advance_us(500_000)
    current = agr.get_album_artwork()
    assert current is not None
    assert current.getpixel((0, 0)) == (2, 0, 0)


@pytest.mark.asyncio
async def test_late_join_gets_current_then_scheduled_artwork() -> None:
    """A joining member gets the current image as of now, then the scheduled clear."""
    group, clock = _make_scheduling_group()
    agr = ArtworkGroupRole(group)
    await agr.set_album_artwork(_image(1))
    await agr.set_album_artwork(None, timestamp_us=1_500_000)
    clock.advance_us(100_000)

    member = _Member()
    agr.subscribe(member)  # type: ignore[arg-type]
    await _settle()

    assert member.sent == [(1, 1_100_000), (None, 1_500_000)]


@pytest.mark.asyncio
async def test_late_join_after_scheduled_artwork_took_effect() -> None:
    """Once due, scheduled artwork is the joining member's current image."""
    group, clock = _make_scheduling_group()
    agr = ArtworkGroupRole(group)
    await agr.set_album_artwork(_image(2), timestamp_us=1_500_000)
    clock.advance_us(500_000)

    member = _Member()
    agr.subscribe(member)  # type: ignore[arg-type]
    await _settle()

    assert member.sent == [(2, 1_500_000)]


@pytest.mark.asyncio
async def test_artwork_set_during_replay_is_sent_after_it() -> None:
    """Artwork set while a replay is encoding reaches the member after the replay."""
    group, _clock = _make_scheduling_group()
    agr = ArtworkGroupRole(group)
    await agr.set_album_artwork(_image(1))
    await agr.set_album_artwork(_image(2), timestamp_us=1_500_000)
    member = _Member()

    agr.subscribe(member)  # type: ignore[arg-type]
    await agr.set_album_artwork(_image(3), timestamp_us=1_600_000)
    await _settle()

    assert member.sent == [(1, 1_000_000), (2, 1_500_000), (3, 1_600_000)]


@pytest.mark.asyncio
async def test_warm_reconnect_replay_waits_for_send_in_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A member rejoining while a send to it is encoding gets the replay after that send."""
    group, _clock = _make_scheduling_group()
    agr = ArtworkGroupRole(group)
    await agr.set_album_artwork(_image(1))
    member = _Member()
    agr.subscribe(member)  # type: ignore[arg-type]
    await _settle()
    release = _gate_encoding(agr, monkeypatch)

    send = asyncio.create_task(agr.set_album_artwork(_image(2), timestamp_us=1_500_000))
    lock = agr._send_lock(member, 0)  # type: ignore[arg-type]  # noqa: SLF001
    while not lock.locked():  # noqa: ASYNC110
        await asyncio.sleep(0)
    agr.unsubscribe(member)  # type: ignore[arg-type]
    agr.subscribe(member)  # type: ignore[arg-type]
    release.set()
    await send
    await _settle()

    assert member.sent == [(1, 1_000_000), (2, 1_500_000), (1, 1_000_000), (2, 1_500_000)]


@pytest.mark.asyncio
async def test_member_leave_stops_its_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    """A replay still encoding when the member leaves sends nothing."""
    group, _clock = _make_scheduling_group()
    agr = ArtworkGroupRole(group)
    await agr.set_album_artwork(_image(1))
    member = _Member()
    release = _gate_encoding(agr, monkeypatch)

    agr.subscribe(member)  # type: ignore[arg-type]
    agr.unsubscribe(member)  # type: ignore[arg-type]
    release.set()
    await _settle()

    assert member.sent == []


@pytest.mark.asyncio
async def test_cancel_scheduled_artwork() -> None:
    """Cancelling asks each member to drop the scheduled image and keeps the current one."""
    group, clock = _make_scheduling_group()
    agr = ArtworkGroupRole(group)
    member = _Member()
    legacy = _Member(can_cancel=False)
    agr._members = [member, legacy]  # type: ignore[list-item]  # noqa: SLF001
    await agr.set_album_artwork(_image(1))
    await agr.set_album_artwork(_image(2), timestamp_us=1_500_000)
    clock.advance_us(100_000)

    await agr.cancel_scheduled(ArtworkSource.ALBUM)
    await agr.cancel_scheduled(ArtworkSource.ALBUM)
    await agr.cancel_scheduled(ArtworkSource.ARTIST)

    assert member.sent[2:] == [("cancel",)]
    assert legacy.sent[2:] == [("cancel",), (1, 1_100_000)]
    clock.advance_us(500_000)
    current = agr.get_album_artwork()
    assert current is not None
    assert current.getpixel((0, 0)) == (1, 0, 0)


@pytest.mark.asyncio
async def test_replacing_scheduled_artwork_restates_current_for_single_message_clients() -> None:
    """A single-message client drops a sent scheduled image before a farther one is held."""
    group, clock = _make_scheduling_group()
    agr = ArtworkGroupRole(group)
    client = MagicMock()
    client.info.artwork_support = ClientHelloArtworkSupport(channels=[_CHANNEL])
    client._server.clock = clock  # noqa: SLF001
    client.group.group_role.return_value = agr
    legacy_sent: list[tuple[int | None, int]] = []

    def _record(data: bytes, *, timestamp_us: int, **_: Any) -> None:
        image = data[BINARY_HEADER_SIZE:]
        if not image:
            legacy_sent.append((None, timestamp_us))
            return
        with Image.open(BytesIO(image)) as decoded:
            legacy_sent.append((decoded.getpixel((0, 0))[0], timestamp_us))

    client.send_binary.side_effect = _record
    legacy = ArtworkV1Role(client=client)
    legacy.on_connect()
    member = _Member()
    agr.subscribe(member)  # type: ignore[arg-type]
    await agr.set_album_artwork(_image(1))
    await agr.set_album_artwork(_image(2), timestamp_us=2_000_000)
    far_us = 31_000_000

    await agr.set_album_artwork(_image(3), timestamp_us=far_us)
    await _settle()

    assert legacy_sent == [(1, 1_000_000), (2, 2_000_000), (1, 1_000_000)]
    assert member.sent == [(1, 1_000_000), (2, 2_000_000), (3, far_us)]
    clock.now_us_value = far_us - MAX_ANNOUNCE_LEAD_US
    legacy._queue_changed.set()  # noqa: SLF001
    await _settle()
    assert legacy_sent[3:] == [(3, far_us)]
    legacy.on_disconnect()
