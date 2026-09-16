"""Track current group-role state plus at most one scheduled update."""

from __future__ import annotations

from collections.abc import Callable


class ScheduledRoleState[S]:
    """Current role state plus at most one pending state that takes effect later.

    No timer runs: a due pending state becomes current when the state is next read
    or written, so read the current state through `current()`.
    """

    def __init__(self, on_commit: Callable[[S | None, int], None] | None = None) -> None:
        """Create an empty tracker.

        `on_commit` is called with (state, timestamp_us) whenever a state becomes
        current, whether applied directly or promoted from pending.
        """
        self._current: S | None = None
        self._pending: S | None = None
        self._pending_timestamp_us: int | None = None
        self._on_commit = on_commit

    @property
    def pending(self) -> S | None:
        """Return the pending state, also None while a clear is pending."""
        return self._pending

    @property
    def pending_timestamp_us(self) -> int | None:
        """Return the server time the pending state takes effect, or None without one."""
        return self._pending_timestamp_us

    def current(self, now_us: int) -> S | None:
        """Return the current state, first promoting a pending state that is due."""
        if self._pending_timestamp_us is not None and self._pending_timestamp_us <= now_us:
            timestamp_us = self._pending_timestamp_us
            self._pending_timestamp_us = None
            self._commit(self._pending, timestamp_us)
            self._pending = None
        return self._current

    def schedule(self, state: S | None, timestamp_us: int) -> None:
        """Hold `state` as the pending state from `timestamp_us`, replacing any held one."""
        self._pending = state
        self._pending_timestamp_us = timestamp_us

    def apply(self, state: S | None, timestamp_us: int) -> None:
        """Make `state` current at once, discarding any pending state."""
        self._pending = None
        self._pending_timestamp_us = None
        self._commit(state, timestamp_us)

    def _commit(self, state: S | None, timestamp_us: int) -> None:
        self._current = state
        if self._on_commit is not None:
            self._on_commit(state, timestamp_us)
