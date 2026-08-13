"""Generic scheduling for role state updates that may take effect in the future.

Sendspin server/state (metadata, color) and artwork binary updates each carry a
server-clock timestamp for when they take effect. A client retains its confirmed
state plus at most one pending update: an update whose effective client time is
still in the future is held as that single pending update and displayed later;
one whose effective time has already passed is applied to confirmed right away.

A pending update stays logically pending, even once its effective time has
passed and it has been displayed, until a later incoming update resolves it:

* An incoming timestamp earlier than the pending one discards the pending
  update without ever committing it to confirmed. If the pending update had
  already taken effect (was being displayed), the display rolls back to
  confirmed.
* An incoming timestamp equal to or later than the pending one commits the
  pending update into confirmed immediately, even if its own effective time has
  not arrived yet. If the pending update was already being displayed, this
  commit does not change what is shown, so no extra callback fires for it.

Only after resolution is the incoming update itself classified as future
(scheduled as the new pending) or now/past (applied to confirmed directly).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Protocol

from aiosendspin.util import create_task

logger = logging.getLogger(__name__)

_DEFAULT_POLL_INTERVAL_S = 0.1


class _TimestampedUpdate(Protocol):
    """Structural type for updates scheduled by `ScheduledStateUpdate`."""

    timestamp: int


def _replace_merge[T](_confirmed: T | None, overlay: T) -> T:
    """Default merge: the overlay wholly replaces confirmed (used for artwork frames)."""
    return overlay


class ScheduledStateUpdate[T: _TimestampedUpdate]:
    """Track confirmed state plus at most one pending update, applied at effective time."""

    def __init__(
        self,
        *,
        map_to_client_time: Callable[[int], int],
        now_us: Callable[[], int],
        commit: Callable[[T | None], None],
        merge: Callable[[T | None, T], T] | None = None,
        poll_interval_s: float = _DEFAULT_POLL_INTERVAL_S,
    ) -> None:
        """Create a tracker that applies updates via `commit` at their effective time.

        Args:
            map_to_client_time: Maps a server-clock timestamp to the client's local
                clock, using the connection's time filter. Called again on each poll
                so a pending update's schedule follows clock correction rather than
                a one-time stale conversion.
            now_us: Returns the client's current local time in microseconds.
            commit: Invoked with the value that should now be displayed (confirmed,
                merged with an applied pending update, or None when the role is
                cleared wholesale). Not invoked when a state change would not alter
                what is displayed.
            merge: Combines confirmed with an overlay update to produce the value to
                display or commit to confirmed. Defaults to the overlay wholly
                replacing confirmed, which is correct for whole artwork frames.
                Metadata and color pass a shallow, field-by-field merge instead.
            poll_interval_s: Maximum interval between effective-time re-checks while
                waiting for a pending update. Kept small in tests for fast execution.
        """
        self._map_to_client_time = map_to_client_time
        self._now_us = now_us
        self._commit = commit
        self._merge = merge or _replace_merge
        self._poll_interval_s = poll_interval_s
        self.confirmed: T | None = None
        self._pending: T | None = None
        self._pending_applied = False
        self._pending_task: asyncio.Task[None] | None = None

    @property
    def display(self) -> T | None:
        """Return what should currently be shown: confirmed, merged with an applied pending."""
        if self._pending_applied and self._pending is not None:
            return self._merge(self.confirmed, self._pending)
        return self.confirmed

    def handle_update(self, update: T) -> None:
        """Resolve `update` against any pending update, then apply or schedule it."""
        if self._pending is not None:
            if update.timestamp >= self._pending.timestamp:
                self._promote_pending()
            else:
                self._resolve_pending()

        effective_time = self._map_to_client_time(update.timestamp)
        if effective_time > self._now_us():
            self._schedule_pending(update)
        else:
            self._apply_confirmed(update)

    def clear_immediately(self) -> None:
        """Clear confirmed and any pending update right away.

        Used when the whole role is nulled (e.g. on role deactivation), which
        takes effect unconditionally rather than being subject to the timestamp
        comparison used for partial updates.
        """
        self._cancel_pending_task()
        self._pending = None
        self._pending_applied = False
        self.confirmed = None
        self._commit(None)

    def discard_pending(self) -> None:
        """Discard pending state without notifying display callbacks."""
        self._cancel_pending_task()
        self._pending = None
        self._pending_applied = False

    def _apply_confirmed(self, update: T) -> None:
        self.confirmed = self._merge(self.confirmed, update)
        self._commit(self.confirmed)

    def _promote_pending(self) -> None:
        pending = self._pending
        was_applied = self._pending_applied
        self._cancel_pending_task()
        self._pending = None
        self._pending_applied = False
        if pending is None:
            return
        self.confirmed = self._merge(self.confirmed, pending)
        if not was_applied:
            self._commit(self.confirmed)

    def _resolve_pending(self) -> None:
        pending = self._pending
        was_applied = self._pending_applied
        self._cancel_pending_task()
        self._pending = None
        self._pending_applied = False
        if pending is None:
            return
        if was_applied:
            self._commit(self.confirmed)

    def _cancel_pending_task(self) -> None:
        if self._pending_task is not None:
            self._pending_task.cancel()
            self._pending_task = None

    def _schedule_pending(self, update: T) -> None:
        self._cancel_pending_task()
        self._pending = update
        self._pending_applied = False
        self._pending_task = create_task(self._wait_and_apply(update))

    async def _wait_and_apply(self, update: T) -> None:
        try:
            while True:
                effective_time = self._map_to_client_time(update.timestamp)
                remaining_us = effective_time - self._now_us()
                if remaining_us <= 0:
                    break
                await asyncio.sleep(min(remaining_us / 1_000_000, self._poll_interval_s))
        except asyncio.CancelledError:
            return
        self._pending_task = None
        self._pending_applied = True
        self._commit(self._merge(self.confirmed, update))
