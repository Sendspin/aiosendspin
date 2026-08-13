"""Tests for metadata/color pending-update reconciliation in server/state handling."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from aiosendspin.client.connection import SendspinConnection
from aiosendspin.client.time_sync import SendspinTimeFilter
from aiosendspin.clock import RawMonotonicClock
from aiosendspin.models.color import SessionUpdateColor
from aiosendspin.models.core import ServerStatePayload
from aiosendspin.models.metadata import SessionUpdateMetadata


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
    return conn, client, clock


async def test_future_metadata_becomes_pending_then_applies() -> None:
    """A metadata update timestamped in the future is held pending, then delivered."""
    conn, client, clock = _make_synced_connection()
    future_update = SessionUpdateMetadata(timestamp=clock.now_us() + 100_000, title="Later")

    conn._handle_server_state(ServerStatePayload(metadata=future_update))  # noqa: SLF001

    client.notify_effective_metadata.assert_not_called()

    await asyncio.sleep(0.3)

    client.notify_effective_metadata.assert_called_once()
    (delivered,) = client.notify_effective_metadata.call_args[0]
    assert delivered.metadata.title == "Later"
    assert conn._metadata_state.confirmed is future_update  # noqa: SLF001


async def test_past_color_applies_immediately() -> None:
    """A color update timestamped now/past commits synchronously."""
    conn, client, clock = _make_synced_connection()
    update = SessionUpdateColor(timestamp=clock.now_us() - 1_000_000, primary=(1, 2, 3))

    conn._handle_server_state(ServerStatePayload(color=update))  # noqa: SLF001

    client.notify_effective_color.assert_called_once()
    (delivered,) = client.notify_effective_color.call_args[0]
    assert delivered.color is update
    assert conn._color_state.confirmed is update  # noqa: SLF001


async def test_latest_metadata_arrival_wins_when_timestamp_goes_backwards() -> None:
    """A future metadata arrival replaces pending without timestamp comparison."""
    conn, client, clock = _make_synced_connection()
    now = clock.now_us()
    pending = SessionUpdateMetadata(timestamp=now + 500_000, title="Pending")
    conn._handle_server_state(ServerStatePayload(metadata=pending))  # noqa: SLF001
    earlier = SessionUpdateMetadata(timestamp=now + 100_000, title="Earlier")
    conn._handle_server_state(ServerStatePayload(metadata=earlier))  # noqa: SLF001

    await asyncio.sleep(0.3)

    client.notify_effective_metadata.assert_called_once()
    (delivered,) = client.notify_effective_metadata.call_args[0]
    assert delivered.metadata.title == "Earlier"
    assert conn._metadata_state.confirmed is earlier  # noqa: SLF001


async def test_later_color_arrival_replaces_pending_without_applying_it() -> None:
    """Replacing pending never applies the displaced color update."""
    conn, client, clock = _make_synced_connection()
    now = clock.now_us()
    pending = SessionUpdateColor(timestamp=now + 100_000, primary=(9, 9, 9))
    conn._handle_server_state(ServerStatePayload(color=pending))  # noqa: SLF001
    later = SessionUpdateColor(timestamp=now + 500_000, primary=(1, 1, 1))
    conn._handle_server_state(ServerStatePayload(color=later))  # noqa: SLF001

    await asyncio.sleep(0.3)

    client.notify_effective_color.assert_not_called()
    assert conn._color_state.confirmed is None  # noqa: SLF001
    conn._color_state.discard_pending()  # noqa: SLF001


async def test_past_metadata_arrival_merges_into_applied_scheduled_state() -> None:
    """Once due, scheduled metadata is current regardless of later timestamp values."""
    conn, client, clock = _make_synced_connection()
    now = clock.now_us()

    base = SessionUpdateMetadata(timestamp=now - 1_000_000, title="A", artist="Artist A")
    conn._handle_server_state(ServerStatePayload(metadata=base))  # noqa: SLF001
    assert client.notify_effective_metadata.call_count == 1

    pending = SessionUpdateMetadata(timestamp=now + 150_000, title="B")
    conn._handle_server_state(ServerStatePayload(metadata=pending))  # noqa: SLF001
    await asyncio.sleep(0.3)
    assert client.notify_effective_metadata.call_count == 2
    (displayed,) = client.notify_effective_metadata.call_args_list[1][0]
    assert displayed.metadata.title == "B"
    assert displayed.metadata.artist == "Artist A"  # shallow-merged from confirmed

    earlier = SessionUpdateMetadata(timestamp=now + 100_000, artist="Artist C")
    conn._handle_server_state(ServerStatePayload(metadata=earlier))  # noqa: SLF001

    assert client.notify_effective_metadata.call_count == 3
    (applied,) = client.notify_effective_metadata.call_args_list[2][0]
    assert applied.metadata.title == "B"
    assert applied.metadata.artist == "Artist C"
    assert conn._metadata_state.confirmed is not None  # noqa: SLF001


async def test_past_color_arrival_merges_into_applied_scheduled_state() -> None:
    """A past arrival updates current color without rolling scheduled state back."""
    conn, client, clock = _make_synced_connection()
    now = clock.now_us()

    base = SessionUpdateColor(timestamp=now - 1_000_000, primary=(1, 1, 1), accent=(2, 2, 2))
    conn._handle_server_state(ServerStatePayload(color=base))  # noqa: SLF001
    assert client.notify_effective_color.call_count == 1

    pending = SessionUpdateColor(timestamp=now + 150_000, primary=(9, 9, 9))
    conn._handle_server_state(ServerStatePayload(color=pending))  # noqa: SLF001
    await asyncio.sleep(0.3)
    assert client.notify_effective_color.call_count == 2
    (displayed,) = client.notify_effective_color.call_args_list[1][0]
    assert displayed.color.primary == (9, 9, 9)
    assert displayed.color.accent == (2, 2, 2)  # shallow-merged from confirmed

    earlier = SessionUpdateColor(timestamp=now + 100_000, accent=(3, 3, 3))
    conn._handle_server_state(ServerStatePayload(color=earlier))  # noqa: SLF001

    assert client.notify_effective_color.call_count == 3
    (applied,) = client.notify_effective_color.call_args_list[2][0]
    assert applied.color.primary == (9, 9, 9)
    assert applied.color.accent == (3, 3, 3)


async def test_future_metadata_after_applied_update_waits_without_extra_callback() -> None:
    """A new future update waits while the applied scheduled state remains current."""
    conn, client, clock = _make_synced_connection()
    now = clock.now_us()

    pending = SessionUpdateMetadata(timestamp=now + 150_000, title="B")
    conn._handle_server_state(ServerStatePayload(metadata=pending))  # noqa: SLF001
    await asyncio.sleep(0.3)
    assert client.notify_effective_metadata.call_count == 1

    later = SessionUpdateMetadata(timestamp=now + 800_000, title="C")
    conn._handle_server_state(ServerStatePayload(metadata=later))  # noqa: SLF001

    assert client.notify_effective_metadata.call_count == 1
    confirmed = conn._metadata_state.confirmed  # noqa: SLF001
    assert confirmed is not None
    assert confirmed.title == "B"


async def test_whole_role_null_drops_active_pending_metadata() -> None:
    """A whole-role null clears immediately and the previously pending update never fires."""
    conn, client, clock = _make_synced_connection()
    pending = SessionUpdateMetadata(timestamp=clock.now_us() + 10_000_000, title="Pending")
    conn._handle_server_state(ServerStatePayload(metadata=pending))  # noqa: SLF001
    client.notify_effective_metadata.assert_not_called()

    conn._handle_server_state(ServerStatePayload(metadata=None))  # noqa: SLF001

    client.notify_effective_metadata.assert_called_once()
    (delivered,) = client.notify_effective_metadata.call_args[0]
    assert delivered.metadata is None
    assert conn._metadata_state.confirmed is None  # noqa: SLF001

    await asyncio.sleep(0.05)
    client.notify_effective_metadata.assert_called_once()
