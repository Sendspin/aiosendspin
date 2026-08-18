"""Schedule one future role update while retaining applied state."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Protocol

from aiosendspin.util import create_task


class _TimestampedUpdate(Protocol):
    """Structural type for updates scheduled by `ScheduledStateUpdate`."""

    timestamp: int


def _replace_merge[T](_confirmed: T | None, overlay: T) -> T:
    """Default merge: the overlay wholly replaces confirmed (used for artwork frames)."""
    return overlay


class ScheduledStateUpdate[T: _TimestampedUpdate]:
    """Internal helper tracking current state plus one future update.

    ``T`` is a timestamped wire update. Metadata and color merge it into current
    state, while artwork replaces the current frame.
    """

    def __init__(
        self,
        *,
        map_to_client_time: Callable[[int], int],
        now_us: Callable[[], int],
        commit: Callable[[T | None], None] | None = None,
        merge: Callable[[T | None, T], T] | None = None,
    ) -> None:
        """Create a tracker that applies updates via `commit` at their effective time.

        Args:
            map_to_client_time: Maps a server-clock timestamp to the client's local
                clock, using the connection's time filter. Called again on each poll
                so a pending update's schedule follows clock correction rather than
                a one-time stale conversion.
            now_us: Returns the client's current local time in microseconds.
            commit: Optionally invoked with the applied value or None for a role clear.
            merge: Combines confirmed with an overlay update to produce the value to
                display or commit to confirmed. Defaults to the overlay wholly
                replacing confirmed, which is correct for whole artwork frames.
                Metadata and color pass a shallow, field-by-field merge instead.
        """
        self._map_to_client_time = map_to_client_time
        self._now_us = now_us
        self._commit = commit
        self._merge = merge or _replace_merge
        self.confirmed: T | None = None
        self._pending: T | None = None
        self._pending_on_apply: Callable[[], None] | None = None
        self._pending_task: asyncio.Task[None] | None = None

    @property
    def display(self) -> T | None:
        """Return the applied state."""
        return self.confirmed

    def handle_update(self, update: T, on_apply: Callable[[], None] | None = None) -> bool:
        """Schedule a future update or apply it now, replacing any pending update."""
        effective_time = self._map_to_client_time(update.timestamp)
        if effective_time > self._now_us():
            self._schedule_pending(update, on_apply)
            return self._pending is update
        self.discard_pending()
        self._apply_confirmed(update)
        if on_apply is not None:
            on_apply()
        return False

    def clear_immediately(self, on_apply: Callable[[], None] | None = None) -> None:
        """Clear current and pending state when the whole role is nulled."""
        self._cancel_pending_task()
        self._pending = None
        self._pending_on_apply = None
        self.confirmed = None
        if self._commit is not None:
            self._commit(None)
        if on_apply is not None:
            on_apply()

    def discard_pending(self) -> None:
        """Discard pending state without notifying display callbacks."""
        self._cancel_pending_task()
        self._pending = None
        self._pending_on_apply = None

    def reschedule_pending(self) -> None:
        """Recalculate the pending timer after the clock mapping changes."""
        if self._pending is None:
            return
        self._cancel_pending_task()
        self._pending_task = create_task(self._wait_and_apply(self._pending))

    def _apply_confirmed(self, update: T) -> None:
        self.confirmed = self._merge(self.confirmed, update)
        if self._commit is not None:
            self._commit(self.confirmed)

    def _cancel_pending_task(self) -> None:
        if self._pending_task is not None:
            self._pending_task.cancel()
            self._pending_task = None

    def _schedule_pending(self, update: T, on_apply: Callable[[], None] | None) -> None:
        self._cancel_pending_task()
        self._pending = update
        self._pending_on_apply = on_apply
        self.reschedule_pending()

    async def _wait_and_apply(self, update: T) -> None:
        try:
            # A new time-filter sample reschedules this task with the updated mapping.
            effective_time = self._map_to_client_time(update.timestamp)
            remaining_us = effective_time - self._now_us()
            if remaining_us > 0:
                await asyncio.sleep(remaining_us / 1_000_000)
        except asyncio.CancelledError:
            return
        # Ignore a timer superseded by a later arrival.
        if self._pending is not update:
            return
        on_apply = self._pending_on_apply
        self._pending = None
        self._pending_on_apply = None
        self._pending_task = None
        self._apply_confirmed(update)
        if on_apply is not None:
            on_apply()
