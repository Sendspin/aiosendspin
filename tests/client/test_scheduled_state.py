"""Tests for the generic pending/current update scheduler.

Used by metadata, color, and artwork to schedule and reconcile timestamped updates.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass

from aiosendspin.client.scheduled_state import ScheduledStateUpdate


@dataclass(slots=True)
class _Update:
    timestamp: int
    label: str = ""


def _real_now_us() -> int:
    return time.monotonic_ns() // 1_000


def _make_state(
    *,
    map_to_client_time: Callable[[int], int] | None = None,
    poll_interval_s: float = 0.05,
) -> tuple[ScheduledStateUpdate[_Update], list[_Update | None]]:
    committed: list[_Update | None] = []
    state = ScheduledStateUpdate[_Update](
        map_to_client_time=map_to_client_time or (lambda ts: ts),
        now_us=_real_now_us,
        commit=committed.append,
        poll_interval_s=poll_interval_s,
    )
    return state, committed


async def test_past_timestamp_applies_immediately() -> None:
    """now/past applies: an update whose effective time has already passed commits at once."""
    state, committed = _make_state()
    update = _Update(timestamp=_real_now_us() - 1_000_000)

    state.handle_update(update)

    assert committed == [update]
    assert state.confirmed is update
    assert state.display is update


async def test_future_timestamp_becomes_pending_then_applies() -> None:
    """A future update is held as pending and applied once its effective time arrives."""
    state, committed = _make_state()
    update = _Update(timestamp=_real_now_us() + 100_000)

    state.handle_update(update)

    assert committed == []
    assert state.confirmed is None

    await asyncio.sleep(0.3)

    assert committed == [update]
    assert state.confirmed is None  # still logically pending, only displayed
    assert state.display is update


async def test_earlier_incoming_before_pending_effective_discards_silently() -> None:
    """An earlier incoming update, arriving before the pending one took effect, is silent."""
    state, committed = _make_state()
    now = _real_now_us()
    pending = _Update(timestamp=now + 10_000_000, label="pending")
    state.handle_update(pending)
    assert committed == []

    earlier = _Update(timestamp=now + 9_000_000, label="earlier")
    state.handle_update(earlier)

    # Neither the discarded pending nor the still-future replacement has fired,
    # since the discarded pending was never displayed in the first place.
    assert committed == []
    assert state.confirmed is None
    assert state.display is None

    await asyncio.sleep(0.1)
    assert committed == []  # earlier's own effective time is still far in the future


async def test_earlier_incoming_after_pending_applied_rolls_back_display() -> None:
    """An earlier incoming update after the pending one took effect rolls back the display."""
    state, committed = _make_state()
    now = _real_now_us()
    first = _Update(timestamp=now - 1_000_000, label="first")
    state.handle_update(first)
    assert committed == [first]

    pending = _Update(timestamp=now + 100_000, label="pending")
    state.handle_update(pending)
    await asyncio.sleep(0.3)
    assert committed == [first, pending]
    assert state.display is pending
    assert state.confirmed is first  # still logically pending despite being displayed

    # "earlier" is itself already past its own effective time by now, so once the
    # rollback resolves the stale pending, it applies immediately in turn.
    earlier = _Update(timestamp=now + 50_000, label="earlier")
    state.handle_update(earlier)

    assert committed == [first, pending, first, earlier]
    assert state.confirmed is earlier
    assert state.display is earlier


async def test_equal_or_later_incoming_commits_pending_before_effective_time() -> None:
    """An incoming update >= the pending timestamp force-commits the pending update."""
    state, committed = _make_state()
    now = _real_now_us()
    pending = _Update(timestamp=now + 10_000_000, label="pending")
    state.handle_update(pending)
    assert committed == []

    later = _Update(timestamp=now + 10_000_000, label="later")  # equal timestamp
    state.handle_update(later)

    # Committed synchronously even though "pending"'s own effective time (10s out)
    # never arrived - it was superseded instead.
    assert committed == [pending]
    assert state.confirmed is pending

    # "later" is itself still in the future, so it becomes the new pending update.
    await asyncio.sleep(0.1)
    assert committed == [pending]


async def test_equal_or_later_incoming_after_pending_applied_no_duplicate_callback() -> None:
    """Promoting an already-displayed pending update does not re-fire its callback."""
    state, committed = _make_state()
    now = _real_now_us()
    pending = _Update(timestamp=now + 150_000, label="pending")
    state.handle_update(pending)
    await asyncio.sleep(0.3)
    assert committed == [pending]
    assert state.display is pending

    later = _Update(timestamp=now + 800_000, label="later")
    state.handle_update(later)

    # Promoting "pending" into confirmed does not change what is displayed, so no
    # extra callback fires for it; "later" is itself still in the future.
    assert committed == [pending]
    assert state.confirmed is pending
    assert state.display is pending  # confirmed alone, until "later" applies


async def test_now_cancellation_of_pending_task() -> None:
    """A pending task that gets discarded does not fire its commit callback."""
    state, committed = _make_state()
    update = _Update(timestamp=_real_now_us() + 5_000_000)
    state.handle_update(update)

    state.discard_pending()

    await asyncio.sleep(0.1)
    assert committed == []
    assert state.confirmed is None
    assert state.display is None


async def test_discard_pending_after_applied_restores_confirmed_without_callback() -> None:
    """Discarding applied pending state restores confirmed without a callback."""
    first = _Update(timestamp=_real_now_us() - 1_000_000, label="first")
    state, committed = _make_state()
    state.handle_update(first)
    update = _Update(timestamp=_real_now_us() + 100_000)
    state.handle_update(update)
    await asyncio.sleep(0.3)
    assert committed == [first, update]

    state.discard_pending()

    assert committed == [first, update]
    assert state.confirmed is first
    assert state.display is first


async def test_clear_immediately_drops_pending_and_commits_none() -> None:
    """clear_immediately applies unconditionally, discarding any pending update."""
    state, committed = _make_state()
    update = _Update(timestamp=_real_now_us() + 5_000_000)
    state.handle_update(update)

    state.clear_immediately()

    assert committed == [None]
    assert state.confirmed is None
    assert state.display is None

    await asyncio.sleep(0.1)
    assert committed == [None]  # the discarded pending never fires afterwards


async def test_clock_correction_reschedules_pending_sooner() -> None:
    """Clock correction moves up the effective time instead of a one-time conversion.

    The mapping is re-evaluated on every poll, so an update scheduled far in the
    future applies immediately once the mapping shifts, rather than waiting out
    the delay that was computed when the update first became pending.
    """
    offset = {"value": 2_000_000}

    def map_to_client_time(ts: int) -> int:
        return ts + offset["value"]

    state, committed = _make_state(map_to_client_time=map_to_client_time, poll_interval_s=0.05)
    start = _real_now_us()
    update = _Update(timestamp=start)

    state.handle_update(update)
    assert committed == []

    await asyncio.sleep(0.1)
    assert committed == []  # still ~2s out per the original mapping

    # Simulate a clock correction: the update's effective time has now passed.
    offset["value"] = -1_000_000

    await asyncio.sleep(0.2)

    assert committed == [update]
    elapsed_us = _real_now_us() - start
    assert elapsed_us < 2_000_000  # applied well before the original 2s schedule
