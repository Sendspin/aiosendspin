"""Tests for ArtworkV1Role (v1) implementation."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from aiosendspin.models import pack_binary_header_raw
from aiosendspin.models.artwork import (
    ARTWORK_MAX_MESSAGE_SIZE,
    ARTWORK_MAX_PART_DATA_SIZE,
    ArtworkChannel,
    ClientHelloArtworkSupport,
    ClientStateArtwork,
    StreamRequestFormatArtwork,
    unpack_artwork_announce,
)
from aiosendspin.models.core import (
    ClientStatePayload,
    StreamEndMessage,
    StreamRequestFormatPayload,
    StreamStartMessage,
)
from aiosendspin.models.types import ArtworkSource, PictureFormat
from aiosendspin.server.clock import LoopClock, ManualClock
from aiosendspin.server.roles.artwork.group import ArtworkGroupRole
from aiosendspin.server.roles.artwork.v1 import MAX_ANNOUNCE_LEAD_US, ArtworkV1Role

_NOW_US = 1_000_000

_ALBUM = ArtworkChannel(
    source=ArtworkSource.ALBUM, format=PictureFormat.JPEG, width=300, height=300
)
_ARTIST = ArtworkChannel(
    source=ArtworkSource.ARTIST, format=PictureFormat.PNG, width=400, height=200
)
_NONE = ArtworkChannel(source=ArtworkSource.NONE)

_ALBUM_WIRE = {"source": "album", "format": "jpeg", "width": 300, "height": 300}
_ARTIST_WIRE = {"source": "artist", "format": "png", "width": 400, "height": 200}
_NONE_WIRE = {"source": "none"}


def _make_client_stub() -> MagicMock:
    """Create a mock client for testing."""
    client = MagicMock()
    client.group = MagicMock()
    client.group.group_role.return_value = None
    client.info = MagicMock()
    client.info.artwork_support = None
    client.send_message = MagicMock()
    client.send_role_message = MagicMock()
    client.send_binary = MagicMock(return_value=True)
    client.wait_role_drained = AsyncMock()
    client._server.clock = ManualClock(now_us_value=_NOW_US)  # noqa: SLF001
    client._logger = MagicMock()  # noqa: SLF001
    return client


def _decode(data: bytes) -> tuple[Any, ...]:
    """Describe a transfer message by its kind and fields."""
    channel = data[0] - 8
    if data[1] == 0x02:
        announce = unpack_artwork_announce(data)
        return ("announce", channel, announce.timestamp_us, announce.total_size)
    if data[1] == 0x01:
        assert len(data) == 2
        return ("cancel", channel)
    assert data[1] == 0x00
    return ("part", channel, data[2:])


def _gate_writes(client: MagicMock) -> asyncio.Semaphore:
    """Make each wait for the artwork queue to drain consume one released write."""
    written = asyncio.Semaphore(0)

    async def _wait(_role_family: str) -> None:
        await written.acquire()

    client.wait_role_drained.side_effect = _wait
    return written


async def _write(written: asyncio.Semaphore, count: int = 1) -> None:
    """Report `count` artwork messages as written and let the role react."""
    for _ in range(count):
        written.release()
        for _ in range(5):
            await asyncio.sleep(0)


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def _make_legacy_client_stub(*channels: ArtworkChannel) -> MagicMock:
    """Mock client whose hello declares artwork channels."""
    client = _make_client_stub()
    client.info.artwork_support = ClientHelloArtworkSupport(channels=list(channels or [_ALBUM]))
    return client


def _state(*channels: ArtworkChannel) -> ClientStatePayload:
    return ClientStatePayload(available=True, artwork=ClientStateArtwork(channels=list(channels)))


def _record(client: MagicMock, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Attach a group role with current images and record what the role sends, in order."""
    group = MagicMock()
    group._server.clock.now_us.return_value = _NOW_US  # noqa: SLF001
    group_role = ArtworkGroupRole(group)
    group_role._current_artwork = {  # noqa: SLF001
        ArtworkSource.ALBUM: Image.new("RGB", (10, 10)),
        ArtworkSource.ARTIST: Image.new("RGB", (10, 10)),
    }
    client.group.group_role.return_value = group_role

    events: list[Any] = []

    def _message(_role: str, message: object) -> None:
        if isinstance(message, StreamStartMessage):
            assert message.payload.artwork is not None
            events.append(("start", message.payload.artwork.to_dict()["channels"]))
        else:
            events.append(type(message).__name__)

    def _binary(data: bytes, **kwargs: Any) -> None:
        if kwargs.get("epoch_exempt"):
            events.append(("exempt", _decode(data)))
        elif client.info.artwork_support is None:
            events.append(_decode(data))
        else:
            events.append(("binary", kwargs["message_type"] - 8, len(data)))

    def _schedule(_role: object, _image: object, channel: int, _config: object) -> None:
        events.append(("image", channel))

    client.send_role_message.side_effect = _message
    client.send_binary.side_effect = _binary
    client.drop_pending_binary.side_effect = lambda roles: events.append(("drop", roles))
    monkeypatch.setattr(group_role, "_schedule_send_artwork", _schedule)
    return events


def test_artwork_role_has_role_id() -> None:
    """ArtworkV1Role has role_id of 'artwork@v1'."""
    client = _make_client_stub()
    role = ArtworkV1Role(client=client)
    assert role.role_id == "artwork@v1"


def test_artwork_role_has_role_family() -> None:
    """ArtworkV1Role has role_family of 'artwork'."""
    client = _make_client_stub()
    role = ArtworkV1Role(client=client)
    assert role.role_family == "artwork"


def test_artwork_role_requires_client() -> None:
    """ArtworkV1Role raises ValueError if no client provided."""
    with pytest.raises(ValueError, match="requires a client"):
        ArtworkV1Role(client=None)


def test_artwork_role_has_no_audio_requirements() -> None:
    """ArtworkV1Role does not receive audio."""
    client = _make_client_stub()
    role = ArtworkV1Role(client=client)
    assert role.get_audio_requirements() is None


def test_artwork_role_on_connect_subscribes_to_group_role() -> None:
    """on_connect() subscribes to ArtworkGroupRole."""
    client = _make_client_stub()
    group_role = MagicMock()
    client.group.group_role.return_value = group_role

    role = ArtworkV1Role(client=client)
    role.on_connect()

    client.group.group_role.assert_called_with("artwork")
    group_role.subscribe.assert_called_once_with(role)


def test_artwork_role_on_disconnect_unsubscribes_from_group_role() -> None:
    """on_disconnect() unsubscribes from ArtworkGroupRole."""
    client = _make_client_stub()
    group_role = MagicMock()
    client.group.group_role.return_value = group_role

    role = ArtworkV1Role(client=client)
    role.on_connect()
    role.on_disconnect()

    group_role.unsubscribe.assert_called_once_with(role)


def test_artwork_role_on_connect_waits_for_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the client/state artwork object, nothing is streamed."""
    client = _make_client_stub()
    events = _record(client, monkeypatch)
    role = ArtworkV1Role(client=client)

    role.on_connect()
    role.on_client_state(ClientStatePayload(available=True))

    assert events == []
    assert role.get_channel_configs() == {}


@pytest.mark.asyncio
async def test_artwork_update_before_state_sends_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A group artwork update reaches the client only once its channels are declared."""
    client = _make_client_stub()
    events = _record(client, monkeypatch)
    role = ArtworkV1Role(client=client)
    role.on_connect()
    group_role = client.group.group_role.return_value

    await group_role.set_album_artwork(Image.new("RGB", (10, 10)))
    assert events == []

    role.on_client_state(_state(_ALBUM))
    events.clear()
    await group_role.set_album_artwork(Image.new("RGB", (10, 10)))
    assert [event[:2] for event in events] == [("announce", 0), ("part", 0)]


@pytest.mark.asyncio
async def test_artwork_encoded_for_old_configuration_is_discarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An image whose encode finishes after its channel was reconfigured is not sent."""
    client = _make_client_stub()
    events = _record(client, monkeypatch)
    role = ArtworkV1Role(client=client)
    role.on_connect()
    group_role = client.group.group_role.return_value
    role.on_client_state(_state(_ALBUM))
    png_album = ArtworkChannel(
        source=ArtworkSource.ALBUM, format=PictureFormat.PNG, width=300, height=300
    )
    role.on_client_state(_state(png_album))
    events.clear()
    image = Image.new("RGB", (10, 10))

    await group_role._send_artwork_to_role_channel(role, image, 0, _ALBUM)  # noqa: SLF001
    assert events == []

    await group_role._send_artwork_to_role_channel(role, image, 0, png_album)  # noqa: SLF001
    assert [event[:2] for event in events] == [("announce", 0), ("part", 0)]


@pytest.mark.parametrize(
    ("channels", "expected"),
    [
        ([_ALBUM], [_ALBUM_WIRE]),
        ([_ALBUM, _NONE, _ARTIST, _NONE], [_ALBUM_WIRE, _NONE_WIRE, _ARTIST_WIRE]),
        ([_NONE, _ARTIST], [_NONE_WIRE, _ARTIST_WIRE]),
        ([_NONE], [_NONE_WIRE]),
        ([_NONE, _NONE, _NONE, _NONE], [_NONE_WIRE]),
    ],
)
def test_artwork_state_starts_truncated_stream(
    monkeypatch: pytest.MonkeyPatch,
    channels: list[ArtworkChannel],
    expected: list[dict[str, object]],
) -> None:
    """stream/start matches the state, truncated after the last streamed channel."""
    client = _make_client_stub()
    events = _record(client, monkeypatch)
    role = ArtworkV1Role(client=client)
    role.on_connect()

    role.on_client_state(_state(*channels))

    assert events[0] == ("start", expected)


def test_artwork_state_start_is_followed_by_current_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After stream/start, the current image is sent for each streamed channel."""
    client = _make_client_stub()
    events = _record(client, monkeypatch)
    role = ArtworkV1Role(client=client)
    role.on_connect()

    role.on_client_state(_state(_ALBUM, _NONE, _ARTIST))

    assert events == [
        ("start", [_ALBUM_WIRE, _NONE_WIRE, _ARTIST_WIRE]),
        ("image", 0),
        ("image", 2),
    ]
    assert role.get_channel_configs() == {0: _ALBUM, 2: _ARTIST}


def test_artwork_unchanged_state_sends_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A state repeating the current channels produces no messages."""
    client = _make_client_stub()
    events = _record(client, monkeypatch)
    role = ArtworkV1Role(client=client)
    role.on_connect()
    role.on_client_state(_state(_ALBUM, _NONE))
    events.clear()

    stray_none = ArtworkChannel(
        source=ArtworkSource.NONE, format=PictureFormat.PNG, width=1, height=1
    )
    role.on_client_state(_state(_ALBUM, stray_none))
    role.on_client_state(_state(_ALBUM))

    assert events == []


def test_artwork_state_change_drops_clears_restarts_and_resends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A changed state drops queued images, clears disabled channels, then re-sends images."""
    client = _make_client_stub()
    events = _record(client, monkeypatch)
    role = ArtworkV1Role(client=client)
    role.on_connect()
    role.on_client_state(_state(_ALBUM, _NONE, _ARTIST))
    events.clear()

    role.on_client_state(_state(_NONE, _ARTIST, _ARTIST))

    assert events == [
        ("drop", ["artwork"]),
        ("announce", 0, _NOW_US, 0),
        ("start", [_NONE_WIRE, _ARTIST_WIRE, _ARTIST_WIRE]),
        ("image", 1),
        ("image", 2),
    ]


def test_artwork_state_format_change_restarts_and_resends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing one channel's format re-announces the stream and re-sends every image."""
    client = _make_client_stub()
    events = _record(client, monkeypatch)
    role = ArtworkV1Role(client=client)
    role.on_connect()
    role.on_client_state(_state(_ALBUM, _ARTIST))
    events.clear()

    png_album = ArtworkChannel(
        source=ArtworkSource.ALBUM, format=PictureFormat.PNG, width=300, height=300
    )
    role.on_client_state(_state(png_album, _ARTIST))

    assert events == [
        ("drop", ["artwork"]),
        ("start", [{**_ALBUM_WIRE, "format": "png"}, _ARTIST_WIRE]),
        ("image", 0),
        ("image", 1),
    ]


def test_artwork_all_none_state_keeps_stream_active(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disabling every channel keeps the stream, so a later state can enable one again."""
    client = _make_client_stub()
    events = _record(client, monkeypatch)
    role = ArtworkV1Role(client=client)
    role.on_connect()
    role.on_client_state(_state(_ALBUM))
    events.clear()

    role.on_client_state(_state(_NONE))
    role.on_client_state(_state(_ALBUM))

    assert events == [
        ("drop", ["artwork"]),
        ("announce", 0, _NOW_US, 0),
        ("start", [_NONE_WIRE]),
        ("drop", ["artwork"]),
        ("start", [_ALBUM_WIRE]),
        ("image", 0),
    ]


def test_artwork_role_on_deactivate_sends_stream_end_when_started() -> None:
    """on_deactivate() ends the artwork stream and forgets its channels."""
    client = _make_client_stub()
    role = ArtworkV1Role(client=client)
    role.on_connect()
    role.on_client_state(_state(_ALBUM))
    client.send_role_message.reset_mock()

    role.on_deactivate()

    sent = [call.args[1] for call in client.send_role_message.call_args_list]
    assert any(isinstance(m, StreamEndMessage) and m.payload.roles == ["artwork"] for m in sent)
    assert role._stream_started is False  # noqa: SLF001
    assert role.get_channel_configs() == {}


def test_artwork_role_on_deactivate_noop_without_stream() -> None:
    """on_deactivate() sends nothing when no stream/start was ever sent."""
    client = _make_client_stub()
    role = ArtworkV1Role(client=client)
    role.on_connect()
    client.send_role_message.reset_mock()

    role.on_deactivate()

    client.send_role_message.assert_not_called()


def test_artwork_reconnect_waits_for_state_again(monkeypatch: pytest.MonkeyPatch) -> None:
    """After a reconnect the stream restarts only once that connection's state arrives."""
    client = _make_client_stub()
    events = _record(client, monkeypatch)
    role = ArtworkV1Role(client=client)
    role.on_connect()
    role.on_client_state(_state(_ALBUM))
    role.on_disconnect()
    events.clear()

    role.on_connect()
    assert events == []
    assert role.get_channel_configs() == {}

    role.on_client_state(_state(_ALBUM))
    assert events == [("start", [_ALBUM_WIRE]), ("image", 0)]


@pytest.mark.asyncio
async def test_artwork_image_is_announced_then_sent_in_capped_parts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An image goes out as a 14-byte announce, then parts of at most 65519 bytes."""
    client = _make_client_stub()
    events = _record(client, monkeypatch)
    role = ArtworkV1Role(client=client)
    role.on_connect()
    role.on_client_state(_state(_ALBUM, _ARTIST))
    events.clear()
    image = bytes(range(256)) * 600

    role.send_artwork(channel=1, image_data=image, timestamp_us=1_500_000)
    await asyncio.sleep(0)

    assert events[0] == ("announce", 1, 1_500_000, len(image))
    parts = events[1:]
    assert [part[:2] for part in parts] == [("part", 1)] * 3
    assert [len(part[2]) for part in parts] == [
        ARTWORK_MAX_PART_DATA_SIZE,
        ARTWORK_MAX_PART_DATA_SIZE,
        len(image) - 2 * ARTWORK_MAX_PART_DATA_SIZE,
    ]
    assert b"".join(part[2] for part in parts) == image
    sizes = [len(call.args[0]) for call in client.send_binary.call_args_list]
    assert sizes[0] == 14
    assert max(sizes) == ARTWORK_MAX_MESSAGE_SIZE
    assert {call.kwargs["timestamp_us"] for call in client.send_binary.call_args_list} == {0}


@pytest.mark.asyncio
async def test_artwork_parts_wait_for_the_previous_message_to_be_written(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each part is enqueued only once the artwork queue has been written."""
    client = _make_client_stub()
    events = _record(client, monkeypatch)
    written = _gate_writes(client)
    role = ArtworkV1Role(client=client)
    role.on_connect()
    role.on_client_state(_state(_ALBUM))
    events.clear()

    role.send_artwork(0, bytes(ARTWORK_MAX_PART_DATA_SIZE + 1), _NOW_US)
    assert [event[0] for event in events] == ["announce"]
    await _write(written)
    assert [event[0] for event in events] == ["announce", "part"]
    await _write(written)
    assert [event[0] for event in events] == ["announce", "part", "part"]


@pytest.mark.asyncio
async def test_artwork_cleared_sends_empty_announce(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clearing a channel announces an empty image with no parts."""
    client = _make_client_stub()
    events = _record(client, monkeypatch)
    role = ArtworkV1Role(client=client)
    role.on_connect()
    role.on_client_state(_state(_NONE, _ARTIST))
    events.clear()

    role.send_artwork_cleared(channel=1, timestamp_us=2000)
    await asyncio.sleep(0)

    assert events == [("announce", 1, 2000, 0)]


@pytest.mark.asyncio
async def test_artwork_new_image_for_in_flight_channel_cancels_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replacing an image in flight drops its queued parts, cancels it, then announces."""
    client = _make_client_stub()
    events = _record(client, monkeypatch)
    written = _gate_writes(client)
    role = ArtworkV1Role(client=client)
    role.on_connect()
    role.on_client_state(_state(_ALBUM))
    events.clear()

    role.send_artwork(0, b"old", _NOW_US)
    role.send_artwork(0, b"new", _NOW_US)
    await _write(written, 2)

    assert events == [
        ("announce", 0, _NOW_US, 3),
        ("drop", ["artwork"]),
        ("exempt", ("cancel", 0)),
        ("announce", 0, _NOW_US, 3),
        ("part", 0, b"new"),
    ]


@pytest.mark.asyncio
async def test_artwork_one_transfer_in_flight_across_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An image for another channel is announced only after the transfer completes."""
    client = _make_client_stub()
    events = _record(client, monkeypatch)
    written = _gate_writes(client)
    role = ArtworkV1Role(client=client)
    role.on_connect()
    role.on_client_state(_state(_ALBUM, _ARTIST))
    events.clear()

    role.send_artwork(0, b"album", _NOW_US)
    role.send_artwork(1, b"artist", _NOW_US)
    role.send_artwork(1, b"artist2", _NOW_US)
    assert events == [("announce", 0, _NOW_US, 5)]

    await _write(written)
    assert events[-1] == ("part", 0, b"album")
    await _write(written)
    assert events[-1] == ("announce", 1, _NOW_US, 7)
    await _write(written)

    assert events == [
        ("announce", 0, _NOW_US, 5),
        ("part", 0, b"album"),
        ("announce", 1, _NOW_US, 7),
        ("part", 1, b"artist2"),
    ]


@pytest.mark.asyncio
async def test_artwork_in_flight_transfer_cancelled_before_stream_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deactivating mid-transfer cancels the transfer ahead of stream/end."""
    client = _make_client_stub()
    events = _record(client, monkeypatch)
    _gate_writes(client)
    role = ArtworkV1Role(client=client)
    role.on_connect()
    role.on_client_state(_state(_ALBUM, _ARTIST))
    role.send_artwork(1, b"artist", _NOW_US)
    events.clear()

    role.on_deactivate()

    assert events == [("drop", ["artwork"]), ("exempt", ("cancel", 1)), "StreamEndMessage"]


@pytest.mark.asyncio
async def test_artwork_in_flight_transfer_cancelled_before_reconfiguring_stream_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reconfiguring state cancels the transfer, clears disabled channels, then restarts."""
    client = _make_client_stub()
    events = _record(client, monkeypatch)
    _gate_writes(client)
    role = ArtworkV1Role(client=client)
    role.on_connect()
    role.on_client_state(_state(_ALBUM, _ARTIST))
    role.send_artwork(1, b"artist", _NOW_US)
    events.clear()

    role.on_client_state(_state(_NONE, _ARTIST))

    assert events == [
        ("drop", ["artwork"]),
        ("exempt", ("cancel", 1)),
        ("announce", 0, _NOW_US, 0),
        ("start", [_NONE_WIRE, _ARTIST_WIRE]),
        ("image", 1),
    ]


@pytest.mark.asyncio
async def test_artwork_scheduled_image_is_announced_at_most_20s_ahead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A far-future image waits until 20 s before its timestamp, without blocking others."""
    loop = asyncio.get_running_loop()
    clock = LoopClock(loop)
    client = _make_client_stub()
    client._server.clock = clock  # noqa: SLF001
    events = _record(client, monkeypatch)
    role = ArtworkV1Role(client=client)
    role.on_connect()
    role.on_client_state(_state(_ALBUM, _ARTIST))
    events.clear()
    scheduled_us = clock.now_us() + MAX_ANNOUNCE_LEAD_US + 100_000
    announced_at: list[int] = []
    client.send_binary.side_effect = lambda data, **_: (
        events.append(_decode(data)),
        announced_at.append(clock.now_us()),
    )

    role.send_artwork(0, b"later", scheduled_us)
    role.send_artwork(1, b"now", clock.now_us())
    await asyncio.sleep(0)
    assert [event[:2] for event in events] == [("announce", 1), ("part", 1)]

    await asyncio.sleep(0.2)

    assert events[2] == ("announce", 0, scheduled_us, 5)
    assert announced_at[2] >= scheduled_us - MAX_ANNOUNCE_LEAD_US


@pytest.mark.asyncio
async def test_artwork_transfers_continue_for_unavailable_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A client reporting available: false still receives complete transfers."""
    client = _make_client_stub()
    client.available = False
    events = _record(client, monkeypatch)
    role = ArtworkV1Role(client=client)
    role.on_connect()
    role.on_client_state(ClientStatePayload(available=False, artwork=_state(_ALBUM).artwork))
    events.clear()

    role.send_artwork(0, b"image", _NOW_US)
    await asyncio.sleep(0)

    assert events == [("announce", 0, _NOW_US, 5), ("part", 0, b"image")]


@pytest.mark.asyncio
async def test_artwork_disconnect_stops_transfer_and_forgets_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a disconnect mid-transfer, the next stream starts with no transfer in flight."""
    client = _make_client_stub()
    events = _record(client, monkeypatch)
    written = _gate_writes(client)
    role = ArtworkV1Role(client=client)
    role.on_connect()
    role.on_client_state(_state(_ALBUM))
    role.send_artwork(0, b"old", _NOW_US)
    role.on_disconnect()
    for _ in range(5):
        await asyncio.sleep(0)
    written.release()
    for _ in range(5):
        await asyncio.sleep(0)
    # The stopped transfer left the released write unused.
    await asyncio.wait_for(written.acquire(), 1)
    assert ("part", 0, b"old") not in events
    assert ("exempt", ("cancel", 0)) not in events
    events.clear()

    role.on_connect()
    role.on_client_state(_state(_ALBUM))
    role.send_artwork(0, b"new", _NOW_US)
    assert events == [("start", [_ALBUM_WIRE]), ("image", 0), ("announce", 0, _NOW_US, 3)]
    await _write(written)

    assert events[-1] == ("part", 0, b"new")


def test_artwork_role_send_artwork_skips_unstreamed_channels() -> None:
    """No artwork binary goes out before stream/start or for a channel that is not streamed."""
    client = _make_client_stub()
    role = ArtworkV1Role(client=client)
    role._client.connection = MagicMock()  # noqa: SLF001

    role.send_artwork(channel=0, image_data=b"image", timestamp_us=1000)
    role.on_client_state(_state(_ALBUM, _NONE))
    role.send_artwork(channel=1, image_data=b"image", timestamp_us=1000)
    role.send_artwork_cleared(channel=2, timestamp_us=1000)

    client.send_binary.assert_not_called()


def test_artwork_role_send_artwork_noop_without_transport() -> None:
    """send_artwork() is a no-op when no transport."""
    client = _make_client_stub()
    role = ArtworkV1Role(client=client)
    role.on_client_state(_state(_ALBUM))
    role._client.connection = None  # noqa: SLF001

    role.send_artwork(channel=0, image_data=b"image", timestamp_us=1000)

    client.send_binary.assert_not_called()


def test_artwork_initial_state_without_artwork_is_a_deviation() -> None:
    """An initial client/state must carry the artwork object."""
    client = _make_client_stub()
    role = ArtworkV1Role(client=client)

    assert role.initial_state_deviations(ClientStatePayload(available=True)) == [
        "has an active artwork role but no artwork state"
    ]
    assert role.initial_state_deviations(_state(_ALBUM)) == []


def test_artwork_state_on_superseded_wire_is_a_deviation() -> None:
    """A state artwork object using 'bmp' or media_width/media_height is reported."""
    client = _make_client_stub()
    role = ArtworkV1Role(client=client)
    payload = ClientStatePayload.from_dict(
        {
            "available": True,
            "artwork": {
                "channels": [
                    {"source": "album", "format": "bmp", "media_width": 8, "media_height": 8}
                ]
            },
        }
    )

    assert role.client_state_deviations(payload) == [
        "artwork used pre-rename dimension keys: media_height, media_width",
        "artwork declared the removed 'bmp' format",
    ]
    assert role.client_state_deviations(_state(_ALBUM)) == []


# DEPRECATED(spec-pr-188): remove in aiosendspin <version>
def test_legacy_hello_client_gets_single_message_artwork() -> None:
    """A client configured by its hello gets `[type][ts][image]`, and a bare header to clear."""
    client = _make_legacy_client_stub(_ALBUM, _ARTIST)
    role = ArtworkV1Role(client=client)
    role.on_connect()
    client.send_binary.reset_mock()

    role.send_artwork(channel=0, image_data=b"image", timestamp_us=1000)
    role.send_artwork(channel=0, image_data=b"image2", timestamp_us=1001)
    role.send_artwork_cleared(channel=1, timestamp_us=2000)

    assert [call.args[0] for call in client.send_binary.call_args_list] == [
        pack_binary_header_raw(8, 1000) + b"image",
        pack_binary_header_raw(8, 1001) + b"image2",
        pack_binary_header_raw(9, 2000),
    ]
    assert [call.kwargs["timestamp_us"] for call in client.send_binary.call_args_list] == [
        1000,
        1001,
        2000,
    ]


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_legacy_initial_state_without_artwork_is_not_a_deviation() -> None:
    """A client configured by its hello needs no artwork object in the initial state."""
    client = _make_legacy_client_stub()
    role = ArtworkV1Role(client=client)

    assert role.initial_state_deviations(ClientStatePayload(available=True)) == []


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_legacy_hello_channels_start_stream_on_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hello declaring artwork channels starts the stream on connect, then sends images."""
    client = _make_legacy_client_stub(_ALBUM, _ARTIST)
    events = _record(client, monkeypatch)
    role = ArtworkV1Role(client=client)

    role.on_connect()

    assert events == [("start", [_ALBUM_WIRE, _ARTIST_WIRE]), ("image", 0), ("image", 1)]
    assert role.get_channel_configs() == {0: _ALBUM, 1: _ARTIST}


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
# DEPRECATED(spec-pr-188): remove in aiosendspin <version>
def test_legacy_state_replaces_hello_channels(monkeypatch: pytest.MonkeyPatch) -> None:
    """A client/state artwork object replaces the channels the hello declared."""
    client = _make_legacy_client_stub(_ALBUM)
    events = _record(client, monkeypatch)
    role = ArtworkV1Role(client=client)
    role.on_connect()
    events.clear()

    role.on_client_state(_state(_NONE, _ARTIST))

    assert events == [
        ("drop", ["artwork"]),
        ("binary", 0, 9),
        ("start", [_NONE_WIRE, _ARTIST_WIRE]),
        ("image", 1),
    ]
    assert role.get_channel_configs() == {1: _ARTIST}


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_legacy_request_format_restarts_stream_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stream/request-format re-announces the stream; repeating it sends nothing."""
    client = _make_legacy_client_stub(_ALBUM, _ARTIST)
    events = _record(client, monkeypatch)
    role = ArtworkV1Role(client=client)
    role.on_connect()
    events.clear()

    role.on_stream_request_format(
        StreamRequestFormatPayload(artwork=StreamRequestFormatArtwork(channel=1, width=800))
    )
    role.on_stream_request_format(
        StreamRequestFormatPayload(artwork=StreamRequestFormatArtwork(channel=1, width=800))
    )

    assert events == [
        ("drop", ["artwork"]),
        ("start", [_ALBUM_WIRE, {**_ARTIST_WIRE, "width": 800}]),
        ("image", 0),
        ("image", 1),
    ]


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_legacy_request_format_enables_declared_none_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request can enable a hello channel declared as none, keeping its declared size."""
    declared_none = ArtworkChannel(
        source=ArtworkSource.NONE, format=PictureFormat.PNG, width=64, height=64
    )
    client = _make_legacy_client_stub(_ALBUM, declared_none)
    events = _record(client, monkeypatch)
    role = ArtworkV1Role(client=client)
    role.on_connect()
    assert events[0] == ("start", [_ALBUM_WIRE])
    events.clear()

    role.on_stream_request_format(
        StreamRequestFormatPayload(
            artwork=StreamRequestFormatArtwork(channel=1, source=ArtworkSource.ARTIST)
        )
    )

    assert events == [
        ("drop", ["artwork"]),
        ("start", [_ALBUM_WIRE, {"source": "artist", "format": "png", "width": 64, "height": 64}]),
        ("image", 0),
        ("image", 1),
    ]


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_legacy_request_format_keeps_size_set_on_none_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A size requested for a none channel applies once a later request enables it."""
    declared_none = ArtworkChannel(
        source=ArtworkSource.NONE, format=PictureFormat.PNG, width=64, height=64
    )
    client = _make_legacy_client_stub(_ALBUM, declared_none)
    events = _record(client, monkeypatch)
    role = ArtworkV1Role(client=client)
    role.on_connect()
    events.clear()

    role.on_stream_request_format(
        StreamRequestFormatPayload(artwork=StreamRequestFormatArtwork(channel=1, width=128))
    )
    assert events == []
    role.on_stream_request_format(
        StreamRequestFormatPayload(
            artwork=StreamRequestFormatArtwork(channel=1, source=ArtworkSource.ARTIST)
        )
    )

    assert role.get_channel_configs()[1].width == 128


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_artwork_role_flags_nonpositive_request_dimensions() -> None:
    """A stream/request-format with non-positive dimensions is flagged, not applied."""
    client = _make_legacy_client_stub()
    role = ArtworkV1Role(client=client)
    role.on_connect()
    role.on_stream_request_format(
        StreamRequestFormatPayload(artwork=StreamRequestFormatArtwork(channel=0, width=-10))
    )
    client.flag_noncompliance.assert_called_once()


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_artwork_role_applies_and_flags_pre_rename_request_dimensions() -> None:
    """A stream/request-format phrased as media_width/media_height resizes, and is flagged."""
    client = _make_legacy_client_stub()
    role = ArtworkV1Role(client=client)
    role.on_connect()
    role.on_stream_request_format(
        StreamRequestFormatPayload(
            artwork=StreamRequestFormatArtwork.from_dict(
                {"channel": 0, "media_width": 800, "media_height": 480}
            )
        )
    )
    config = role.get_channel_configs()[0]
    assert (config.width, config.height) == (800, 480)
    assert "media_width" in client.flag_noncompliance.call_args.args[0]


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_artwork_role_flags_bmp_request_format() -> None:
    """A stream/request-format asking for the removed 'bmp' format is flagged, not rejected."""
    client = _make_legacy_client_stub()
    role = ArtworkV1Role(client=client)
    role.on_connect()
    role.on_stream_request_format(
        StreamRequestFormatPayload(
            artwork=StreamRequestFormatArtwork(channel=0, format=PictureFormat.BMP)
        )
    )
    assert "bmp" in client.flag_noncompliance.call_args.args[0]
    assert role.get_channel_configs()[0].format == PictureFormat.BMP


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_artwork_role_no_flag_for_valid_request_dimensions() -> None:
    """A stream/request-format with positive dimensions is not flagged."""
    client = _make_legacy_client_stub()
    role = ArtworkV1Role(client=client)
    role.on_connect()
    role.on_stream_request_format(
        StreamRequestFormatPayload(
            artwork=StreamRequestFormatArtwork(channel=0, width=100, height=100)
        )
    )
    client.flag_noncompliance.assert_not_called()


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_artwork_role_flags_unknown_request_channel() -> None:
    """A stream/request-format for a channel with no config is flagged."""
    client = _make_legacy_client_stub()
    role = ArtworkV1Role(client=client)
    role.on_connect()
    role.on_stream_request_format(
        StreamRequestFormatPayload(artwork=StreamRequestFormatArtwork(channel=1))
    )
    client.flag_noncompliance.assert_called_once()


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_artwork_partial_format_request_preserves_unchanged_fields() -> None:
    """A partial stream/request-format only overwrites fields the client included."""
    client = _make_legacy_client_stub()
    role = ArtworkV1Role(client=client)
    role.on_connect()

    payload = StreamRequestFormatPayload(
        artwork=StreamRequestFormatArtwork(channel=0, format=PictureFormat.PNG),
    )
    role.on_stream_request_format(payload)

    configs = role.get_channel_configs()
    assert configs[0].format == PictureFormat.PNG
    assert configs[0].source == ArtworkSource.ALBUM
    assert configs[0].width == 300
    assert configs[0].height == 300
