"""ColorGroupRole - group-level color coordination."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiosendspin.models.core import ServerStateMessage, ServerStatePayload
from aiosendspin.server.roles.base import GroupRole, Role
from aiosendspin.server.roles.color.events import ColorClearedEvent, ColorUpdatedEvent
from aiosendspin.server.roles.color.state import Color
from aiosendspin.server.roles.color.types import ColorRoleProtocol

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

    @property
    def color(self) -> Color | None:
        """Return current color palette."""
        return self._current_color

    def on_member_join(self, role: Role) -> None:
        """Send current color to newly joined member."""
        self._send_state_to_role(role)

    def _send_state_to_role(self, role: ColorRoleProtocol) -> None:
        """Send current color state to a single role."""
        if self._current_color is None:
            role.send_message(ServerStateMessage(ServerStatePayload(color=None)))
            return

        timestamp = self._group._server.clock.now_us()  # noqa: SLF001
        color_update = self._current_color.snapshot_update(timestamp)
        role.send_message(ServerStateMessage(ServerStatePayload(color=color_update)))

    def set_color(self, color: Color | None) -> None:
        """Set color palette and push updates to all subscribed roles."""
        if color == self._current_color:
            return

        timestamp = self._group._server.clock.now_us()  # noqa: SLF001
        last_color = self._current_color
        color_update = None if color is None else color.snapshot_update(timestamp)

        self._current_color = color

        for role in self._members:
            state_message = ServerStateMessage(ServerStatePayload(color=color_update))
            role.send_message(state_message)

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

    def clear(self) -> None:
        """Clear the color palette."""
        self.set_color(None)
