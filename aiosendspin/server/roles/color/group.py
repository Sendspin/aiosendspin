"""ColorGroupRole - group-level color coordination."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiosendspin.models.color import SessionUpdateColor
from aiosendspin.models.core import (
    LegacyServerStateClearMessage,
    ServerStateMessage,
    ServerStatePayload,
)
from aiosendspin.server.roles.base import GroupRole, Role
from aiosendspin.server.roles.color.events import ColorClearedEvent, ColorUpdatedEvent
from aiosendspin.server.roles.color.state import Color

if TYPE_CHECKING:
    import asyncio

    from aiosendspin.server.group import SendspinGroup


class ColorGroupRole(GroupRole):
    """Coordinate color palette across a group.

    Stores current color state and pushes updates to subscribed ColorV1Roles.
    """

    role_family = "color"

    def __init__(self, group: SendspinGroup) -> None:
        """Initialize ColorGroupRole."""
        super().__init__(group)
        self._state: ScheduledRoleState[Color] = ScheduledRoleState()
        # Defers sending the scheduled palette until it is close enough to its timestamp.
        self._send_scheduled_handle: asyncio.TimerHandle | None = None

    @property
    def color(self) -> Color | None:
        """Return current color palette."""
        return self._state.current(self._now_us())

    def on_member_join(self, role: Role) -> None:
        """Send current color to newly joined member."""
        self._send_state_to_role(role)

    def _send_state_to_role(self, role: Role) -> None:
        """
        Send the complete current color state to a single role.

        Without a palette the state is a timestamp-only object.
        """
        timestamp = self._group._server.clock.now_us()  # noqa: SLF001
        if self._current_color is None:
            self._send_color(role, SessionUpdateColor(timestamp=timestamp), cleared=True)
            return

        self._send_color(role, self._current_color.snapshot_update(timestamp), cleared=False)

    def _send_color(self, role: Role, update: SessionUpdateColor, *, cleared: bool) -> None:
        """Send a color object to a role; `cleared` marks the object for no palette."""
        # DEPRECATED(spec-pr-275): remove in aiosendspin <version>
        if cleared and role.clears_state_with_null():
            role.send_message(LegacyServerStateClearMessage(self.role_family))
            return
        role.send_message(ServerStateMessage(ServerStatePayload(color=update)))

    def set_color(self, color: Color | None, *, timestamp_us: int | None = None) -> None:
        """Set color palette and push updates to all subscribed roles.

        A future `timestamp_us` schedules the palette to take effect then, replacing any
        palette already scheduled. It is sent to clients at most 20 seconds ahead, and
        `ColorUpdatedEvent` fires now, carrying that timestamp. Otherwise the palette
        applies at once and cancels a scheduled one. To show two palettes in sequence,
        schedule the second only after the first took effect.

        Raises ValueError when scheduling None; schedule `Color()` to blank the palette.
        """
        now_us = self._now_us()
        current = self._state.current(now_us)
        timestamp = now_us if timestamp_us is None else timestamp_us

        if not self._state.has_pending and not self._state.scheduled_fields and color == current:
            return

        last_color = self._current_color
        color_update = (
            SessionUpdateColor(timestamp=timestamp)
            if color is None
            else color.snapshot_update(timestamp)
        )

        if timestamp > now_us:
            self._warn_scheduled_lead(timestamp, now_us)
            self._state.schedule(
                color, color_update, timestamp, set(color_update.to_dict()) - {"timestamp"}
            )
        else:
            self._state.apply(color, timestamp)

        for role in self._members:
            self._send_color(role, color_update, cleared=color is None)

        if color is None:
            self.emit_group_event(
                ColorClearedEvent(previous_color=last_color, timestamp_us=timestamp)
            )
            return
        self.emit_group_event(
            ColorUpdatedEvent(
                color=color,
                previous_color=last_color,
                timestamp_us=timestamp,
            )
        )

    def cancel_scheduled(self) -> None:
        """Cancel the scheduled palette, if any, keeping the current one."""
        now_us = self._now_us()
        current = self._state.current(now_us)
        if self._state.pending_timestamp_us is None:
            return
        if self._send_scheduled_handle is not None:
            self._state.apply(current, now_us)
            self._cancel_send_scheduled()
            return
        self._apply(current, now_us)

    def clear(self) -> None:
        """Clear the color palette, and any scheduled palette, at once."""
        self.set_color(None)

    def _apply(self, color: Color | None, timestamp_us: int) -> None:
        """Make `color` current and send it to all members."""
        self._state.apply(color, timestamp_us)
        self._cancel_send_scheduled()
        self._send_to_members(_state_message(color, timestamp_us))

    def _schedule(self, color: Color, timestamp_us: int) -> None:
        """Hold `color` as the scheduled palette and send it once within the lead limit."""
        replaced_sent = (
            self._state.pending_timestamp_us is not None and self._send_scheduled_handle is None
        )
        self._state.schedule(color, timestamp_us)
        self._cancel_send_scheduled()
        self._send_scheduled_handle = self._call_before(timestamp_us, self._send_scheduled)
        if replaced_sent and self._send_scheduled_handle is not None:
            # Clients still hold the replaced palette; the current one discards it.
            now_us = self._now_us()
            self._send_to_members(_state_message(self._state.current(now_us), now_us))

    def _send_scheduled(self) -> None:
        """Send the scheduled palette to all members."""
        self._send_scheduled_handle = None
        scheduled_us = self._state.pending_timestamp_us
        if scheduled_us is not None:
            self._send_to_members(_state_message(self._state.pending, scheduled_us))

    def _cancel_send_scheduled(self) -> None:
        if self._send_scheduled_handle is not None:
            self._send_scheduled_handle.cancel()
            self._send_scheduled_handle = None

    def _send_to_members(self, message: ServerStateMessage) -> None:
        for role in self._members:
            role.send_message(message)


def _state_message(color: Color | None, timestamp_us: int) -> ServerStateMessage:
    """Return the server/state carrying `color` as of `timestamp_us`."""
    color_update = None if color is None else color.snapshot_update(timestamp_us)
    return ServerStateMessage(ServerStatePayload(color=color_update))
