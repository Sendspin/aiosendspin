"""Track current group-role state plus at most one scheduled update."""

from __future__ import annotations

from collections.abc import Callable


class ScheduledRoleState[S, U]:
    """Current state plus at most one future-dated update, promoted lazily.

    The server keeps no timer. A due pending update is promoted whenever the
    state is next observed or written, so all reads go through :meth:`current`
    rather than a raw attribute.
    """

    def __init__(self, on_commit: Callable[[S | None, int], None] | None = None) -> None:
        """Create an empty tracker.

        :param on_commit: Invoked with (state, timestamp_us) whenever a value
            becomes current, applied directly or promoted from pending.
        """
        self._current: S | None = None
        self._pending: S | None = None
        self._pending_update: U | None = None
        self._pending_effective_us: int | None = None
        self._scheduled_fields: set[str] = set()
        self._on_commit = on_commit

    def current(self, now_us: int) -> S | None:
        """Return current state, promoting a due pending update first."""
        self.promote_due(now_us)
        return self._current

    @property
    def pending(self) -> S | None:
        """Return the pending state value, None also while a clear is pending."""
        return self._pending

    @property
    def pending_update(self) -> U | None:
        """Return the pending wire update, for late-joiner replay."""
        return self._pending_update

    @property
    def pending_effective_us(self) -> int | None:
        """Return when the pending update takes effect, or None without one."""
        return self._pending_effective_us

    @property
    def has_pending(self) -> bool:
        """Return whether an update is scheduled."""
        return self._pending_effective_us is not None

    @property
    def scheduled_fields(self) -> set[str]:
        """Fields a scheduled update touched, which later diffs must restate."""
        return self._scheduled_fields

    def promote_due(self, now_us: int) -> None:
        """Make the pending update current once its effective time has passed."""
        if self._pending_effective_us is None or self._pending_effective_us > now_us:
            return
        effective_us = self._pending_effective_us
        self._current = self._pending
        self._pending = None
        self._pending_update = None
        self._pending_effective_us = None
        # _scheduled_fields deliberately survives promotion. The server clock
        # can promote before a client's mapped clock does, and that client
        # discards its own pending copy when the next update arrives (last
        # arrival wins). The next diff must therefore restate every field the
        # scheduled update touched. Only an immediate apply() flushes the set.
        if self._on_commit is not None:
            self._on_commit(self._current, effective_us)

    def schedule(
        self,
        state: S | None,
        update: U | None,
        effective_us: int,
        fields: set[str] | None = None,
    ) -> None:
        """Hold state as the single pending update, replacing any prior pending."""
        self._pending = state
        self._pending_update = update
        self._pending_effective_us = effective_us
        self._scheduled_fields = fields if fields is not None else set()

    def apply(self, state: S | None, timestamp_us: int) -> None:
        """Make state current immediately, dropping any pending update."""
        self._pending = None
        self._pending_update = None
        self._pending_effective_us = None
        self._current = state
        self._scheduled_fields = set()
        if self._on_commit is not None:
            self._on_commit(state, timestamp_us)
