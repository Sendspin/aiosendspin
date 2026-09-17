"""Group-role state that can be scheduled to take effect later."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from aiosendspin.models.core import LegacyServerStateClearMessage
from aiosendspin.server.roles.base import GroupRole, Role

if TYPE_CHECKING:
    import asyncio

    from aiosendspin.models.core import ServerStateMessage
    from aiosendspin.server.group import SendspinGroup

# A scheduled update is sent at most this long before it takes effect.
MAX_SCHEDULED_LEAD_US = 20_000_000


class ScheduledRoleState[S]:
    """Current role state plus at most one pending state that takes effect later.

    No timer runs: a due pending state becomes current when the state is next read,
    so read the current state through `current()`.
    """

    def __init__(self) -> None:
        """Create an empty tracker."""
        self._current: S | None = None
        self._pending: S | None = None
        self._pending_timestamp_us: int | None = None

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
            self.apply(self._pending)
        return self._current

    def schedule(self, state: S | None, timestamp_us: int) -> None:
        """Hold `state` as the pending state from `timestamp_us`, replacing any held one."""
        self._pending = state
        self._pending_timestamp_us = timestamp_us

    def apply(self, state: S | None) -> None:
        """Make `state` current at once, discarding any pending state."""
        self._current = state
        self._pending = None
        self._pending_timestamp_us = None


class ScheduledStateGroupRole[S](GroupRole):
    """Group role whose server/state object can be scheduled to take effect later.

    Members get the current state, then the scheduled state once it is at most
    `MAX_SCHEDULED_LEAD_US` ahead.
    """

    def __init__(self, group: SendspinGroup) -> None:
        """Initialize the role with no state."""
        super().__init__(group)
        self._state: ScheduledRoleState[S] = ScheduledRoleState()
        self._scheduled_sent = False
        self._send_scheduled_handle: asyncio.TimerHandle | None = None

    def on_member_join(self, role: Role) -> None:
        """Send the current state, then any scheduled state already sent to members."""
        now_us = self._now_us()
        current = self._current_state(now_us)
        self._send_state(role, current, self._state_message(current, now_us))
        scheduled_us = self._state.pending_timestamp_us
        if scheduled_us is not None and self._scheduled_sent:
            scheduled = self._state.pending
            self._send_state(role, scheduled, self._state_message(scheduled, scheduled_us))

    def on_group_deleted(self) -> None:
        """Stop a deferred send of the scheduled state."""
        self._cancel_send_scheduled()

    def cancel_scheduled(self) -> None:
        """Cancel the scheduled state, if any, keeping the current one.

        No event fires for the cancellation.
        """
        now_us = self._now_us()
        current = self._current_state(now_us)
        if self._state.pending_timestamp_us is None:
            return
        if self._scheduled_sent:
            self._apply(current, now_us)
            return
        self._state.apply(current)
        self._cancel_send_scheduled()

    @abstractmethod
    def _state_message(self, state: S | None, timestamp_us: int) -> ServerStateMessage:
        """Return the server/state carrying `state` as of `timestamp_us`.

        None is carried as a timestamp-only object.
        """

    def _current_state(self, now_us: int) -> S | None:
        """Return the current state as it is restated at `now_us`."""
        return self._state.current(now_us)

    def _apply(self, state: S | None, timestamp_us: int) -> None:
        """Make `state` current and send it to all members."""
        message = self._state_message(state, timestamp_us)
        self._state.apply(state)
        self._cancel_send_scheduled()
        self._send_to_members(state, message)

    def _schedule(self, state: S, timestamp_us: int) -> None:
        """Hold `state` as scheduled and send it once it is within the lead limit."""
        # Building the message first rejects a state that cannot be sent.
        self._state_message(state, timestamp_us)
        replaced_sent = self._scheduled_sent and self._state.pending_timestamp_us is not None
        self._cancel_send_scheduled()
        self._state.schedule(state, timestamp_us)
        delay_us = timestamp_us - MAX_SCHEDULED_LEAD_US - self._now_us()
        if delay_us <= 0:
            self._send_scheduled()
            return
        self._send_scheduled_handle = self._group._server.loop.call_later(  # noqa: SLF001
            delay_us / 1_000_000, self._send_scheduled
        )
        if replaced_sent:
            # Clients still hold the replaced state; restating the current one discards it.
            now_us = self._now_us()
            current = self._current_state(now_us)
            self._send_to_members(current, self._state_message(current, now_us))

    def _send_scheduled(self) -> None:
        """Send the scheduled state to all members."""
        self._send_scheduled_handle = None
        scheduled_us = self._state.pending_timestamp_us
        if scheduled_us is not None:
            self._scheduled_sent = True
            scheduled = self._state.pending
            self._send_to_members(scheduled, self._state_message(scheduled, scheduled_us))

    def _cancel_send_scheduled(self) -> None:
        self._scheduled_sent = False
        if self._send_scheduled_handle is not None:
            self._send_scheduled_handle.cancel()
            self._send_scheduled_handle = None

    def _send_to_members(self, state: S | None, message: ServerStateMessage) -> None:
        for role in self._members:
            self._send_state(role, state, message)

    def _send_state(self, role: Role, state: S | None, message: ServerStateMessage) -> None:
        """Send `message`, which carries `state`, to a role."""
        # DEPRECATED(spec-pr-275): remove in aiosendspin <version>
        if state is None and role.clears_state_with_null():
            role.send_message(LegacyServerStateClearMessage(self.role_family))
            return
        role.send_message(message)
