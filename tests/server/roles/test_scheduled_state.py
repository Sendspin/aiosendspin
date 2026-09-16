"""Tests for the server-side scheduled role state tracker."""

from __future__ import annotations

from aiosendspin.server.roles.scheduled_state import ScheduledRoleState


def test_current_promotes_due_pending() -> None:
    """Reading the current state promotes a pending state once it is due."""
    commits: list[tuple[str | None, int]] = []
    state: ScheduledRoleState[str] = ScheduledRoleState(
        lambda value, ts: commits.append((value, ts))
    )
    state.schedule("next", 1_000)

    assert state.current(999) is None
    assert state.pending_timestamp_us == 1_000
    assert state.current(1_000) == "next"
    assert state.pending is None
    assert state.pending_timestamp_us is None
    assert commits == [("next", 1_000)]


def test_scheduled_clear_promotes_to_none() -> None:
    """A pending None clears the current state once due."""
    state: ScheduledRoleState[str] = ScheduledRoleState()
    state.apply("now", 0)
    state.schedule(None, 1_000)

    assert state.current(999) == "now"
    assert state.current(1_000) is None


def test_apply_discards_pending() -> None:
    """An immediate apply replaces the current state and drops the pending one."""
    commits: list[tuple[str | None, int]] = []
    state: ScheduledRoleState[str] = ScheduledRoleState(
        lambda value, ts: commits.append((value, ts))
    )
    state.schedule("next", 1_000)

    state.apply("now", 500)

    assert state.pending_timestamp_us is None
    assert state.current(2_000) == "now"
    assert commits == [("now", 500)]


def test_schedule_replaces_prior_pending_by_arrival() -> None:
    """Only the latest scheduled state is kept, even with an earlier timestamp."""
    state: ScheduledRoleState[str] = ScheduledRoleState()
    state.schedule("later", 3_000)
    state.schedule("earlier", 2_000)

    assert state.pending == "earlier"
    assert state.pending_timestamp_us == 2_000
    assert state.current(2_500) == "earlier"
    assert state.current(3_500) == "earlier"
