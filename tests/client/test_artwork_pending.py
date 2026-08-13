"""Tests for client-side artwork binary handling.

Covers timestamp retention, per-channel pending/current reconciliation, rollback,
and stream/end cleanup.
"""

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
    """A past-timestamped artwork chunk applies at once, keeping the header timestamp."""
    conn, client, clock = _make_synced_connection()
    timestamp_us = clock.now_us() - 1_000_000
    payload = _artwork_binary(0, timestamp_us, b"jpeg-bytes")

    conn._handle_binary_message(payload)  # noqa: SLF001

    client.notify_artwork.assert_called_once_with(0, b"jpeg-bytes")
    client.notify_artwork_timestamp.assert_called_once_with(0, b"jpeg-bytes", timestamp_us)
    client.notify_effective_artwork.assert_called_once_with(0, b"jpeg-bytes", timestamp_us)


async def test_future_artwork_becomes_pending_then_applies_as_current() -> None:
    """A future artwork chunk is held pending and reflects as current once effective."""
    conn, client, clock = _make_synced_connection()
    timestamp_us = clock.now_us() + 100_000
    payload = _artwork_binary(1, timestamp_us, b"future-art")

    conn._handle_binary_message(payload)  # noqa: SLF001

    client.notify_artwork.assert_called_once_with(1, b"future-art")
    client.notify_artwork_timestamp.assert_called_once_with(1, b"future-art", timestamp_us)
    client.notify_effective_artwork.assert_not_called()
    assert conn._artwork_channels[1].confirmed is None  # noqa: SLF001

    await asyncio.sleep(0.3)

    client.notify_effective_artwork.assert_called_once_with(1, b"future-art", timestamp_us)
    # still logically pending, even though it has been displayed
    assert conn._artwork_channels[1].confirmed is None  # noqa: SLF001
    assert conn._artwork_channels[1].display is not None  # noqa: SLF001


async def test_scheduled_empty_clear_applies_at_effective_time() -> None:
    """An empty-payload artwork chunk (a clear) participates in the same pending logic."""
    conn, client, clock = _make_synced_connection()
    timestamp_us = clock.now_us() + 100_000
    payload = _artwork_binary(0, timestamp_us, b"")

    conn._handle_binary_message(payload)  # noqa: SLF001

    client.notify_artwork.assert_called_once_with(0, b"")
    client.notify_artwork_timestamp.assert_called_once_with(0, b"", timestamp_us)
    client.notify_effective_artwork.assert_not_called()

    await asyncio.sleep(0.3)

    client.notify_effective_artwork.assert_called_once_with(0, b"", timestamp_us)


async def test_artwork_rollback_after_pending_applied_reverts_to_confirmed_frame() -> None:
    """An already-displayed pending artwork frame rolls back to the confirmed frame."""
    conn, client, clock = _make_synced_connection()
    now = clock.now_us()

    base_payload = _artwork_binary(0, now - 1_000_000, b"base-art")
    conn._handle_binary_message(base_payload)  # noqa: SLF001
    assert client.notify_effective_artwork.call_count == 1
    client.notify_effective_artwork.assert_called_with(0, b"base-art", now - 1_000_000)

    pending_payload = _artwork_binary(0, now + 150_000, b"pending-art")
    conn._handle_binary_message(pending_payload)  # noqa: SLF001
    await asyncio.sleep(0.3)
    assert client.notify_effective_artwork.call_count == 2
    client.notify_effective_artwork.assert_called_with(0, b"pending-art", now + 150_000)

    earlier_payload = _artwork_binary(0, now + 100_000, b"earlier-art")
    conn._handle_binary_message(earlier_payload)  # noqa: SLF001

    # The rollback re-emits the confirmed frame ("base-art"); "pending-art" is
    # discarded entirely. "earlier-art" is itself already past its own effective
    # time by now, so it applies right after the rollback.
    assert client.notify_effective_artwork.call_args_list[2] == (
        (0, b"base-art", now - 1_000_000),
        {},
    )
    assert client.notify_effective_artwork.call_args_list[3] == (
        (0, b"earlier-art", now + 100_000),
        {},
    )


async def test_artwork_pending_discarded_on_stream_end() -> None:
    """Pending artwork per channel is dropped, unfired, when the artwork stream ends."""
    conn, client, clock = _make_synced_connection()
    timestamp_us = clock.now_us() + 10_000_000
    payload = _artwork_binary(2, timestamp_us, b"never-applied")
    conn._handle_binary_message(payload)  # noqa: SLF001
    client.notify_artwork.assert_called_once_with(2, b"never-applied")
    client.notify_effective_artwork.assert_not_called()

    conn._handle_stream_end(  # noqa: SLF001
        StreamEndMessage(payload=StreamEndPayload(roles=["artwork"]))
    )

    await asyncio.sleep(0.05)
    client.notify_effective_artwork.assert_not_called()
    assert conn._artwork_channels[2].confirmed is None  # noqa: SLF001
    assert conn._artwork_channels[2].display is None  # noqa: SLF001


async def test_artwork_stream_end_discards_applied_pending_without_extra_callback() -> None:
    """Stream end discards an already-displayed pending frame silently."""
    conn, client, clock = _make_synced_connection()
    base_timestamp = clock.now_us() - 1_000_000
    conn._handle_binary_message(_artwork_binary(3, base_timestamp, b"base"))  # noqa: SLF001
    timestamp_us = clock.now_us() + 100_000
    payload = _artwork_binary(3, timestamp_us, b"already-shown")
    conn._handle_binary_message(payload)  # noqa: SLF001
    await asyncio.sleep(0.3)
    assert client.notify_effective_artwork.call_count == 2

    conn._handle_stream_end(  # noqa: SLF001
        StreamEndMessage(payload=StreamEndPayload(roles=["artwork"]))
    )

    assert client.notify_effective_artwork.call_count == 2
    assert conn._artwork_channels[3].confirmed is not None  # noqa: SLF001
    assert conn._artwork_channels[3].confirmed.image_data == b"base"  # noqa: SLF001
