"""Controller command events.

Events emitted when controller clients send commands to the server.
"""

from __future__ import annotations

from dataclasses import dataclass

from aiosendspin.models.types import RepeatMode
from aiosendspin.server.events import GroupRoleEvent


class ControllerEvent(GroupRoleEvent):
    """Base event type for controller commands."""


@dataclass
class ControllerPlayEvent(ControllerEvent):
    """Play command received."""


@dataclass
class ControllerPauseEvent(ControllerEvent):
    """Pause command received."""


@dataclass
class ControllerStopEvent(ControllerEvent):
    """Stop command received."""


@dataclass
class ControllerNextEvent(ControllerEvent):
    """Next track command received."""


@dataclass
class ControllerPreviousEvent(ControllerEvent):
    """Previous track command received."""


@dataclass
class ControllerVolumeEvent(ControllerEvent):
    """Volume change command received."""

    volume: int
    """Target volume (0-100)."""


@dataclass
class ControllerMuteEvent(ControllerEvent):
    """Mute state change command received."""

    muted: bool
    """Target mute state."""


@dataclass
class ControllerSwitchEvent(ControllerEvent):
    """Switch groups command received."""


@dataclass
class ControllerRepeatEvent(ControllerEvent):
    """Repeat mode change command received."""

    mode: RepeatMode
    """Target repeat mode."""


@dataclass
class ControllerSeekEvent(ControllerEvent):
    """Absolute seek command received."""

    position_ms: int
    """Target absolute position in milliseconds."""


@dataclass
class ControllerSeekRelativeEvent(ControllerEvent):
    """Relative seek command received.

    The offset is clamped so that, from the group's current metadata position, it lands
    between 0 and `seek_max_ms` (with no upper bound while `seek_max_ms` is unset). Clamping
    never reverses the seek: from beyond `seek_max_ms` a forward offset becomes 0. When the
    current position is unknown the offset is passed through unchanged, and the handler owns
    clamping the result to the seekable range.
    """

    offset_ms: int
    """Signed offset in milliseconds from the current position."""


@dataclass
class ControllerShuffleEvent(ControllerEvent):
    """Shuffle state change command received."""

    shuffle: bool
    """Target shuffle state (True for shuffle, False for unshuffle)."""
