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
    assert state.confirmed is update
    assert state.display is update


async def test_later_arrival_replaces_pending_when_timestamp_goes_backwards() -> None:
    """The latest future arrival wins without comparing update timestamps."""
    state, committed = _make_state()
    now = _real_now_us()
    first = _Update(timestamp=now + 500_000, label="first")
    replacement = _Update(timestamp=now + 100_000, label="replacement")

    state.handle_update(first)
    state.handle_update(replacement)

    await asyncio.sleep(0.3)

    assert committed == [replacement]
    assert state.confirmed is replacement


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


async def test_immediate_update_discards_pending_and_merges_into_current() -> None:
    """A present arrival applies immediately and cancels the held future update."""
    first = _Update(timestamp=_real_now_us() - 1_000_000, label="first")
    state, committed = _make_state()
    state.handle_update(first)
    pending = _Update(timestamp=_real_now_us() + 5_000_000, label="pending")
    state.handle_update(pending)
    immediate = _Update(timestamp=_real_now_us() - 1, label="immediate")

    state.handle_update(immediate)

    assert committed == [first, immediate]
    assert state.confirmed is immediate

    await asyncio.sleep(0.1)
    assert committed == [first, immediate]


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
