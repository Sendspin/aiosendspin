"""Tests for client-side artwork transfer reassembly and validation."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiosendspin.client.connection import SendspinConnection
from aiosendspin.clock import ManualClock
from aiosendspin.models.artwork import (
    ARTWORK_MAX_MESSAGE_SIZE,
    ArtworkChannel,
    StreamArtworkChannelConfig,
    StreamStartArtwork,
    pack_artwork_announce,
    pack_artwork_cancel,
    pack_artwork_parts,
)
from aiosendspin.models.core import (
    StreamEndMessage,
    StreamEndPayload,
    StreamStartMessage,
    StreamStartPayload,
)
from aiosendspin.models.types import ArtworkSource, PictureFormat, Roles

from .conftest import make_sdk_client

_NOW_US = 10_000_000
_ALBUM = StreamArtworkChannelConfig(
    source=ArtworkSource.ALBUM, format=PictureFormat.JPEG, width=64, height=64
)
_ARTIST = StreamArtworkChannelConfig(
    source=ArtworkSource.ARTIST, format=PictureFormat.PNG, width=32, height=32
)


def _stream_start(*channels: StreamArtworkChannelConfig) -> StreamStartMessage:
    return StreamStartMessage(
        payload=StreamStartPayload(artwork=StreamStartArtwork(channels=list(channels)))
    )


async def _artwork_connection(
    *, synced: bool = False
) -> tuple[SendspinConnection, list[tuple[int, bytes]]]:
    """Return a connection with an active two-channel artwork stream and its deliveries."""
    client = make_sdk_client(
        client_name="Test Client",
        roles=[Roles.ARTWORK],
        artwork_channels=[
            ArtworkChannel(
                source=ArtworkSource.ALBUM, format=PictureFormat.JPEG, width=64, height=64
            )
        ],
        clock=ManualClock(now_us_value=_NOW_US),
    )
    delivered: list[tuple[int, bytes]] = []
    client.add_artwork_listener(lambda channel, image: delivered.append((channel, image)))
    connection = SendspinConnection(client)
    connection.disconnect = AsyncMock()  # type: ignore[method-assign]
    if synced:
        # Server and client clocks agree.
        time_filter = MagicMock()
        time_filter.count = 2
        time_filter.compute_client_time.side_effect = lambda server_us: server_us
        connection._time_filter = time_filter  # noqa: SLF001
    await connection._handle_stream_start(_stream_start(_ALBUM, _ARTIST))  # noqa: SLF001
    return connection, delivered


def _receive(connection: SendspinConnection, *messages: bytes) -> None:
    for message in messages:
        connection._handle_binary_message(message)  # noqa: SLF001


def _transfer(channel: int, image: bytes, timestamp_us: int = _NOW_US) -> list[bytes]:
    return [
        pack_artwork_announce(channel, timestamp_us, len(image)),
        *pack_artwork_parts(channel, image),
    ]


async def _assert_closed(connection: SendspinConnection) -> None:
    await asyncio.sleep(0)
    connection.disconnect.assert_awaited_once()  # type: ignore[attr-defined]


async def _assert_open(connection: SendspinConnection) -> None:
    await asyncio.sleep(0)
    connection.disconnect.assert_not_awaited()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_parts_are_reassembled_and_delivered_once_complete() -> None:
    """An image split across parts is delivered whole, only after its last part."""
    connection, delivered = await _artwork_connection()
    image = bytes(range(256)) * 300
    announce, first, second = _transfer(1, image)

    _receive(connection, announce, first)
    assert delivered == []
    _receive(connection, second)

    assert delivered == [(1, image)]
    await _assert_open(connection)


@pytest.mark.asyncio
async def test_image_is_delivered_at_its_timestamp() -> None:
    """A complete image waits for its translated timestamp."""
    connection, delivered = await _artwork_connection(synced=True)

    _receive(connection, *_transfer(0, b"image", _NOW_US + 50_000))
    assert delivered == []
    await asyncio.sleep(0.1)

    assert delivered == [(0, b"image")]


@pytest.mark.asyncio
async def test_late_image_is_delivered_immediately() -> None:
    """An image whose timestamp has passed is shown at once, not dropped."""
    connection, delivered = await _artwork_connection(synced=True)

    _receive(connection, *_transfer(0, b"image", _NOW_US - 5_000_000))

    assert delivered == [(0, b"image")]


@pytest.mark.asyncio
async def test_empty_image_clears_without_parts() -> None:
    """An announce with total_size 0 completes at once and delivers empty bytes."""
    connection, delivered = await _artwork_connection()

    _receive(connection, pack_artwork_announce(0, _NOW_US, 0), *_transfer(1, b"next"))

    assert delivered == [(0, b""), (1, b"next")]
    await _assert_open(connection)


@pytest.mark.asyncio
async def test_cancel_discards_only_the_pending_image() -> None:
    """A cancel drops the image in flight or scheduled; the current image stays."""
    connection, delivered = await _artwork_connection(synced=True)
    _receive(connection, *_transfer(0, b"current"))
    in_flight, first, _ = _transfer(0, bytes(ARTWORK_MAX_MESSAGE_SIZE))

    _receive(connection, in_flight, first, pack_artwork_cancel(0))
    _receive(connection, *_transfer(0, b"scheduled", _NOW_US + 50_000), pack_artwork_cancel(0))
    await asyncio.sleep(0.1)

    assert delivered == [(0, b"current")]
    await _assert_open(connection)


@pytest.mark.asyncio
async def test_cancel_on_other_channel_keeps_transfer_in_flight() -> None:
    """A cancel for another channel leaves the transfer in flight."""
    connection, delivered = await _artwork_connection()
    announce, part = _transfer(0, b"image")

    _receive(connection, announce, pack_artwork_cancel(1), part)

    assert delivered == [(0, b"image")]


@pytest.mark.asyncio
async def test_new_announce_replaces_the_pending_image() -> None:
    """An announce discards a complete image still waiting for its timestamp."""
    connection, delivered = await _artwork_connection(synced=True)

    _receive(connection, *_transfer(0, b"old", _NOW_US + 50_000), *_transfer(0, b"new"))
    await asyncio.sleep(0.1)

    assert delivered == [(0, b"new")]


@pytest.mark.parametrize(
    "message",
    [
        pytest.param(b"\x08", id="shorter-than-2"),
        pytest.param(b"\x08\x00" + bytes(ARTWORK_MAX_MESSAGE_SIZE - 1), id="over-size-cap"),
        pytest.param(pack_artwork_announce(0, 1, 1)[:-1], id="short-announce"),
        pytest.param(pack_artwork_announce(0, 1, 1) + b"\x00", id="long-announce"),
        pytest.param(pack_artwork_cancel(0) + b"\x00", id="long-cancel"),
        pytest.param(b"\x08\x04data", id="reserved-bit"),
        pytest.param(b"\x08\x03", id="cancel-and-announce"),
    ],
)
@pytest.mark.asyncio
async def test_malformed_message_closes_connection(message: bytes) -> None:
    """Every malformed artwork message is a protocol error."""
    connection, delivered = await _artwork_connection()

    _receive(connection, message)

    await _assert_closed(connection)
    assert delivered == []


@pytest.mark.asyncio
async def test_malformed_message_outside_stream_closes_connection() -> None:
    """A malformed artwork message is a protocol error even with no stream active."""
    connection, _ = await _artwork_connection()
    connection._handle_stream_end(  # noqa: SLF001
        StreamEndMessage(payload=StreamEndPayload(roles=["artwork"]))
    )

    _receive(connection, b"\x08\x04")

    await _assert_closed(connection)


@pytest.mark.parametrize(
    "messages",
    [
        pytest.param(
            [pack_artwork_announce(0, _NOW_US, 5), pack_artwork_announce(1, _NOW_US, 5)],
            id="announce-in-flight",
        ),
        pytest.param([pack_artwork_parts(0, b"part")[0]], id="part-without-transfer"),
        pytest.param(
            [pack_artwork_announce(0, _NOW_US, 5), pack_artwork_parts(1, b"part")[0]],
            id="part-on-other-channel",
        ),
        pytest.param(
            [pack_artwork_announce(0, _NOW_US, 5), pack_artwork_parts(0, b"too long")[0]],
            id="part-past-total-size",
        ),
        pytest.param(
            [*_transfer(0, b"done"), pack_artwork_parts(0, b"more")[0]],
            id="part-after-completion",
        ),
    ],
)
@pytest.mark.asyncio
async def test_malformed_sequence_closes_connection(messages: list[bytes]) -> None:
    """Every malformed artwork sequence within an active stream is a protocol error."""
    connection, _ = await _artwork_connection()

    _receive(connection, *messages)

    await _assert_closed(connection)


@pytest.mark.asyncio
async def test_messages_outside_stream_are_ignored() -> None:
    """Well-formed artwork messages outside an active stream are neither applied nor errors."""
    connection, delivered = await _artwork_connection()
    connection._handle_stream_end(  # noqa: SLF001
        StreamEndMessage(payload=StreamEndPayload(roles=["artwork"]))
    )

    _receive(connection, pack_artwork_parts(0, b"part")[0], *_transfer(0, b"image"))

    assert delivered == []
    await _assert_open(connection)


@pytest.mark.asyncio
async def test_stream_end_clears_current_images_and_drops_pending() -> None:
    """stream/end clears channels showing an image and discards pending and in-flight state."""
    connection, delivered = await _artwork_connection(synced=True)
    _receive(connection, *_transfer(0, b"current"))
    _receive(connection, *_transfer(1, b"scheduled", _NOW_US + 50_000))
    _receive(connection, pack_artwork_announce(1, _NOW_US, 5))

    connection._handle_stream_end(  # noqa: SLF001
        StreamEndMessage(payload=StreamEndPayload(roles=["artwork"]))
    )
    await connection._handle_stream_start(_stream_start(_ALBUM, _ARTIST))  # noqa: SLF001
    _receive(connection, *_transfer(1, b"next"))
    await asyncio.sleep(0.1)

    assert delivered == [(0, b"current"), (0, b""), (1, b"next")]
    await _assert_open(connection)


@pytest.mark.asyncio
async def test_reconfiguring_stream_start_drops_that_channels_pending_image() -> None:
    """A stream/start changing a channel discards its pending image; others are kept."""
    connection, delivered = await _artwork_connection(synced=True)
    _receive(connection, *_transfer(0, b"album", _NOW_US + 50_000))
    _receive(connection, *_transfer(1, b"artist", _NOW_US + 50_000))

    await connection._handle_stream_start(  # noqa: SLF001
        _stream_start(_ALBUM, StreamArtworkChannelConfig(source=ArtworkSource.NONE))
    )
    await asyncio.sleep(0.1)

    assert delivered == [(0, b"album")]


@pytest.mark.asyncio
async def test_reconfiguring_stream_start_ends_that_channels_transfer() -> None:
    """A stream/start changing the in-flight channel ends the transfer."""
    connection, delivered = await _artwork_connection()
    _receive(connection, pack_artwork_announce(1, _NOW_US, 5))

    await connection._handle_stream_start(_stream_start(_ALBUM))  # noqa: SLF001
    _receive(connection, *_transfer(0, b"album"))

    assert delivered == [(0, b"album")]
    await _assert_open(connection)


@pytest.mark.asyncio
async def test_disconnect_resets_artwork_state() -> None:
    """Disconnecting discards pending images and any transfer in flight."""
    connection, delivered = await _artwork_connection(synced=True)
    _receive(connection, *_transfer(0, b"scheduled", _NOW_US + 50_000))
    _receive(connection, pack_artwork_announce(1, _NOW_US, 5))
    del connection.disconnect
    connection._connected = True  # noqa: SLF001

    await connection.disconnect()
    await asyncio.sleep(0.1)

    assert delivered == []
    assert connection._artwork_pending == {}  # noqa: SLF001
    assert connection._artwork_in_flight is None  # noqa: SLF001
    assert connection._artwork_config is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_unavailable_client_tracks_transfers_without_delivering() -> None:
    """While unavailable, transfers are followed and image data is discarded; clears apply."""
    connection, delivered = await _artwork_connection()
    connection._reported_available = False  # noqa: SLF001
    in_flight, first, _ = _transfer(0, bytes(ARTWORK_MAX_MESSAGE_SIZE))

    _receive(connection, *_transfer(0, b"image"), pack_artwork_announce(1, _NOW_US, 0))
    _receive(connection, in_flight, first, pack_artwork_cancel(0))
    assert delivered == [(1, b"")]

    connection._reported_available = True  # noqa: SLF001
    _receive(connection, *_transfer(1, b"visible"))

    assert delivered == [(1, b""), (1, b"visible")]
    await _assert_open(connection)


@pytest.mark.asyncio
async def test_unavailable_client_still_closes_on_overlong_part() -> None:
    """Part bytes are counted while unavailable, so an overrun is still detected."""
    connection, _ = await _artwork_connection()
    connection._reported_available = False  # noqa: SLF001

    _receive(connection, pack_artwork_announce(0, _NOW_US, 5), *pack_artwork_parts(0, b"toolong"))

    await _assert_closed(connection)
