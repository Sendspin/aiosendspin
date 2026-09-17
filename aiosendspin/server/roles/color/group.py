"""ColorGroupRole - group-level color coordination."""

from __future__ import annotations

from aiosendspin.models.color import SessionUpdateColor
from aiosendspin.models.core import ServerStateMessage, ServerStatePayload
from aiosendspin.server.roles.color.events import ColorClearedEvent, ColorUpdatedEvent
from aiosendspin.server.roles.color.state import Color
from aiosendspin.server.roles.scheduled_state import ScheduledStateGroupRole


class ColorGroupRole(ScheduledStateGroupRole[Color]):
    """Coordinate color palette across a group.

    Stores current color state and pushes updates to subscribed ColorV1Roles.
    """

    role_family = "color"

    @property
    def color(self) -> Color | None:
        """Return current color palette."""
        return self._state.current(self._now_us())

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
        last_color = self._state.current(now_us)
        if timestamp_us is not None and timestamp_us > now_us:
            if color is None:
                raise ValueError("a color clear cannot be scheduled; schedule Color() instead")
            self._schedule(color, timestamp_us)
            timestamp = timestamp_us
        else:
            if color == last_color and self._state.pending_timestamp_us is None:
                return
            timestamp = now_us if timestamp_us is None else timestamp_us
            self._apply(color, timestamp)

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
        """Clear the color palette, and any scheduled palette, at once."""
        self.set_color(None)

    def _state_message(self, state: Color | None, timestamp_us: int) -> ServerStateMessage:
        color_update = (
            SessionUpdateColor(timestamp=timestamp_us)
            if state is None
            else state.snapshot_update(timestamp_us)
        )
        return ServerStateMessage(ServerStatePayload(color=color_update))
