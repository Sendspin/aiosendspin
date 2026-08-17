"""Tests for client-side artwork binary scheduling and stream cleanup."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from aiosendspin.client.connection import SendspinConnection
from aiosendspin.client.time_sync import SendspinTimeFilter
from aiosendspin.clock import RawMonotonicClock
from aiosendspin.models import pack_binary_header_raw
from aiosendspin.models.core import StreamEndMessage, StreamEndPayload
from aiosendspin.models.types import BinaryMessageType


def _make_synced_connection() -> tuple[SendspinConnection, MagicMock, RawMonotonicClock]:
    """Build a connection whose time filter maps server time 1:1 onto client time."""
    clock = RawMonotonicClock()
    conn = SendspinConnection.__new__(SendspinConnection)
    client = MagicMock()
    client.clock = clock
    conn._client = client  # noqa: SLF001
    conn._time_filter = SendspinTimeFilter()  # noqa: SLF001
    now = clock.now_us()
    conn._time_filter.update(0, 0, now)  # noqa: SLF001
    conn._time_filter.update(0, 0, now + 1)  # noqa: SLF001
    assert conn._time_filter.is_synchronized  # noqa: SLF001
    conn._init_state_trackers()  # noqa: SLF001
    conn._artwork_stream_active = True  # noqa: SLF001
    return conn, client, clock


def _artwork_binary(channel: int, timestamp_us: int, image_data: bytes) -> bytes:
    message_type = BinaryMessageType.ARTWORK_CHANNEL_0.value + channel
    return pack_binary_header_raw(message_type, timestamp_us) + image_data


async def test_past_artwork_retains_header_timestamp_and_applies_immediately() -> None:
    """A past-timestamped artwork chunk applies without becoming pending."""
    conn, client, clock = _make_synced_connection()
    timestamp_us = clock.now_us() - 1_000_000
    payload = _artwork_binary(0, timestamp_us, b"jpeg-bytes")

    conn._handle_binary_message(payload)  # noqa: SLF001

    client.notify_artwork.assert_called_once_with(0, b"jpeg-bytes")
    client.notify_scheduled_artwork.assert_not_called()


async def test_future_artwork_becomes_pending_then_applies_as_current() -> None:
    """A future artwork chunk is held pending and reflects as current once effective."""
    conn, client, clock = _make_synced_connection()
    timestamp_us = clock.now_us() + 100_000
    payload = _artwork_binary(1, timestamp_us, b"future-art")

    conn._handle_binary_message(payload)  # noqa: SLF001

    client.notify_artwork.assert_not_called()
    client.notify_scheduled_artwork.assert_called_once_with(1, b"future-art", timestamp_us)
    assert conn._artwork_channels[1].confirmed is None  # noqa: SLF001

    await asyncio.sleep(0.3)

    client.notify_artwork.assert_called_once_with(1, b"future-art")
    assert conn._artwork_channels[1].confirmed is not None  # noqa: SLF001
    assert conn._artwork_channels[1].display is not None  # noqa: SLF001


async def test_scheduled_empty_clear_applies_at_effective_time() -> None:
    """An empty-payload artwork chunk (a clear) participates in the same pending logic."""
    conn, client, clock = _make_synced_connection()
    timestamp_us = clock.now_us() + 100_000
    payload = _artwork_binary(0, timestamp_us, b"")

    conn._handle_binary_message(payload)  # noqa: SLF001

    client.notify_artwork.assert_not_called()
    client.notify_scheduled_artwork.assert_called_once_with(0, b"", timestamp_us)

    await asyncio.sleep(0.3)

    client.notify_artwork.assert_called_once_with(0, b"")


async def test_latest_artwork_arrival_wins_when_timestamp_goes_backwards() -> None:
    """The latest future image replaces pending independently of timestamp order."""
    conn, client, clock = _make_synced_connection()
    now = clock.now_us()

    pending_payload = _artwork_binary(0, now + 500_000, b"pending-art")
    conn._handle_binary_message(pending_payload)  # noqa: SLF001
    earlier_payload = _artwork_binary(0, now + 100_000, b"earlier-art")
    conn._handle_binary_message(earlier_payload)  # noqa: SLF001

    assert [item.args for item in client.notify_scheduled_artwork.call_args_list] == [
        (0, b"pending-art", now + 500_000),
        (0, b"earlier-art", now + 100_000),
    ]

    await asyncio.sleep(0.3)

    client.notify_artwork.assert_called_once_with(0, b"earlier-art")


async def test_immediate_artwork_discards_pending() -> None:
    """A present image applies immediately and cancels the held future image."""
    conn, client, clock = _make_synced_connection()
    now = clock.now_us()
    conn._handle_binary_message(  # noqa: SLF001
        _artwork_binary(0, now + 5_000_000, b"pending")
    )

    conn._handle_binary_message(_artwork_binary(0, now - 1, b"immediate"))  # noqa: SLF001

    client.notify_scheduled_artwork.assert_called_once_with(0, b"pending", now + 5_000_000)
    client.notify_artwork.assert_called_once_with(0, b"immediate")
    await asyncio.sleep(0.05)
    client.notify_artwork.assert_called_once()


async def test_artwork_pending_discarded_on_stream_end() -> None:
    """Pending artwork per channel is dropped, unfired, when the artwork stream ends."""
    conn, client, clock = _make_synced_connection()
    timestamp_us = clock.now_us() + 10_000_000
    payload = _artwork_binary(2, timestamp_us, b"never-applied")
    conn._handle_binary_message(payload)  # noqa: SLF001
    client.notify_scheduled_artwork.assert_called_once_with(2, b"never-applied", timestamp_us)
    client.notify_artwork.assert_not_called()

    conn._handle_stream_end(  # noqa: SLF001
        StreamEndMessage(payload=StreamEndPayload(roles=["artwork"]))
    )

    await asyncio.sleep(0.05)
    client.notify_artwork.assert_not_called()
    assert conn._artwork_channels[2].confirmed is None  # noqa: SLF001
    assert conn._artwork_channels[2].display is None  # noqa: SLF001


async def test_artwork_stream_end_keeps_applied_current_without_extra_callback() -> None:
    """Stream end discards only pending images and retains current artwork."""
    conn, client, clock = _make_synced_connection()
    base_timestamp = clock.now_us() - 1_000_000
    conn._handle_binary_message(_artwork_binary(3, base_timestamp, b"base"))  # noqa: SLF001
    timestamp_us = clock.now_us() + 100_000
    payload = _artwork_binary(3, timestamp_us, b"already-shown")
    conn._handle_binary_message(payload)  # noqa: SLF001
    await asyncio.sleep(0.3)
    assert client.notify_artwork.call_count == 2

    conn._handle_stream_end(  # noqa: SLF001
        StreamEndMessage(payload=StreamEndPayload(roles=["artwork"]))
    )

    assert client.notify_artwork.call_count == 2
    assert conn._artwork_channels[3].confirmed is not None  # noqa: SLF001
    assert conn._artwork_channels[3].confirmed.image_data == b"already-shown"  # noqa: SLF001
