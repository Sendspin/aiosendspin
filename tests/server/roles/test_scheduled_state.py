"""Tests for the server-side scheduled role state tracker."""

from __future__ import annotations

from aiosendspin.server.roles.scheduled_state import ScheduledRoleState


def test_current_promotes_due_pending() -> None:
    """Reading current state promotes a due pending value and commits it."""
    commits: list[tuple[str | None, int]] = []
    state: ScheduledRoleState[str, str] = ScheduledRoleState(
        lambda value, ts: commits.append((value, ts))
    )
    state.schedule("next", "update", 1_000, {"title"})

    assert state.current(999) is None
    assert state.has_pending

    assert state.current(1_000) == "next"
    assert not state.has_pending
    assert state.pending_update is None
    assert commits == [("next", 1_000)]


def test_promote_keeps_scheduled_fields() -> None:
    """Promotion retains the scheduled fields so the next diff restates them.

    The server clock can promote before a client's mapped clock does, and that
    client discards its pending copy on the next arrival (last arrival wins).
    """
    state: ScheduledRoleState[str, str] = ScheduledRoleState()
    state.schedule("next", "update", 1_000, {"title", "artist"})

    state.promote_due(1_000)

    assert state.scheduled_fields == {"title", "artist"}


def test_apply_drops_pending_and_flushes_scheduled_fields() -> None:
    """An immediate apply replaces pending outright and clears the field memory."""
    commits: list[tuple[str | None, int]] = []
    state: ScheduledRoleState[str, str] = ScheduledRoleState(
        lambda value, ts: commits.append((value, ts))
    )
    state.schedule("next", "update", 1_000, {"title"})

    state.apply("now", 500)

    assert state.current(2_000) == "now"
    assert not state.scheduled_fields
    assert commits == [("now", 500)]


def test_schedule_replaces_prior_pending() -> None:
    """Only the latest scheduled update is retained, regardless of timestamps."""
    state: ScheduledRoleState[str, str] = ScheduledRoleState()
    state.schedule("later", "update-a", 3_000, {"title"})
    state.schedule("earlier", "update-b", 2_000, {"artist"})

    assert state.pending == "earlier"
    assert state.pending_update == "update-b"
    assert state.pending_effective_us == 2_000
    assert state.scheduled_fields == {"artist"}
    assert state.current(2_500) == "earlier"
