"""ColorGroupRole - group-level color coordination."""

from __future__ import annotations

from dataclasses import replace
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
    from aiosendspin.server.group import SendspinGroup


class ColorGroupRole(GroupRole):
    """Coordinate color palette across a group.

    Stores current color state and pushes updates to subscribed ColorV1Roles.
    """

    role_family = "color"

    def __init__(self, group: SendspinGroup) -> None:
        """Initialize ColorGroupRole."""
        super().__init__(group)
        self._current_color: Color | None = None
        self._pending_color: Color | None = None
        self._pending_update: SessionUpdateColor | None = None
        self._scheduled_fields: set[str] = set()

    @property
    def color(self) -> Color | None:
        """Return current color palette."""
        self._promote_due_pending(self._group._server.clock.now_us())  # noqa: SLF001
        return self._current_color

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
        """Set or schedule a color palette and push it to subscribed roles."""
        now_us = self._group._server.clock.now_us()  # noqa: SLF001
        self._promote_due_pending(now_us)
        timestamp = now_us if timestamp_us is None else timestamp_us
        if color is not None:
            if timestamp_us is not None:
                color = replace(color, timestamp_us=timestamp_us)
            elif color.timestamp_us is None:
                color = replace(color, timestamp_us=timestamp)
            else:
                timestamp = color.timestamp_us

        had_pending = self._pending_update is not None
        if not had_pending and not self._scheduled_fields and color == self._current_color:
            return

        last_color = self._current_color
        color_update = (
            SessionUpdateColor(timestamp=timestamp)
            if color is None
            else color.snapshot_update(timestamp)
        )

        self._pending_color = None
        self._pending_update = None
        if timestamp > now_us:
            self._pending_color = color
            self._pending_update = color_update
            self._scheduled_fields = set(color_update.to_dict()) - {"timestamp"}
        else:
            self._current_color = color
            self._scheduled_fields.clear()

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

    def _promote_due_pending(self, now_us: int) -> None:
        pending_update = self._pending_update
        if pending_update is None or pending_update.timestamp > now_us:
            return
        self._current_color = self._pending_color
        self._pending_color = None
        self._pending_update = None

    def _include_scheduled_fields(
        self,
        update: SessionUpdateColor,
        color: Color | None,
    ) -> None:
        if not self._scheduled_fields:
            return
        snapshot = (
            SessionUpdateColor.cleared(update.timestamp)
            if color is None
            else color.snapshot_update(update.timestamp)
        )
        for field_name in self._scheduled_fields:
            setattr(update, field_name, getattr(snapshot, field_name))

    def clear(self, *, timestamp_us: int | None = None) -> None:
        """Clear the color palette."""
        self.set_color(None, timestamp_us=timestamp_us)
