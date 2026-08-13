"""Schedule one future role update while retaining applied state."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Protocol

from aiosendspin.util import create_task

_DEFAULT_POLL_INTERVAL_S = 0.1


class _TimestampedUpdate(Protocol):
    """Structural type for updates scheduled by `ScheduledStateUpdate`."""

    timestamp: int


def _replace_merge[T](_confirmed: T | None, overlay: T) -> T:
    """Default merge: the overlay wholly replaces confirmed (used for artwork frames)."""
    return overlay


class ScheduledStateUpdate[T: _TimestampedUpdate]:
    """Track current state plus the latest future update by arrival order."""

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
            commit: Invoked with the new applied value or None for a role clear.
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
        self._pending_task: asyncio.Task[None] | None = None

    @property
    def display(self) -> T | None:
        """Return the applied state."""
        return self.confirmed

    def handle_update(self, update: T) -> None:
        """Replace pending with a future update or apply an immediate update."""
        effective_time = self._map_to_client_time(update.timestamp)
        if effective_time > self._now_us():
            self._schedule_pending(update)
        else:
            self.discard_pending()
            self._apply_confirmed(update)

    def clear_immediately(self) -> None:
        """Clear current and pending state when the whole role is nulled."""
        self._cancel_pending_task()
        self._pending = None
        self.confirmed = None
        self._commit(None)

    def discard_pending(self) -> None:
        """Discard pending state without notifying display callbacks."""
        self._cancel_pending_task()
        self._pending = None

    def _apply_confirmed(self, update: T) -> None:
        self.confirmed = self._merge(self.confirmed, update)
        self._commit(self.confirmed)

    def _cancel_pending_task(self) -> None:
        if self._pending_task is not None:
            self._pending_task.cancel()
            self._pending_task = None

    def _schedule_pending(self, update: T) -> None:
        self._cancel_pending_task()
        self._pending = update
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
        if self._pending is not update:
            return
        self._pending = None
        self._pending_task = None
        self._apply_confirmed(update)
