"""
Controller messages for the Sendspin protocol.

This module contains messages specific to clients with the controller role, which
enables the client to control the Sendspin group this client is part of, and switch
between groups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import SendspinConfig, SendspinModel, split_enum_values
from .types import MediaCommand, RepeatMode


# Client -> Server: client/command controller object
@dataclass
class ControllerCommandPayload(SendspinModel):
    """Control the group that's playing."""

    command: MediaCommand
    """
    Command must be one of the values listed in supported_commands from server/state controller
    object.
    """
    volume: int | None = None
    """Volume range 0-100, only set if command is volume."""
    mute: bool | None = None
    """True to mute, false to unmute, only set if command is mute."""
    position_ms: int | None = None
    """Absolute playback position in ms, only set if command is seek."""
    offset_ms: int | None = None
    """Signed offset in ms from current position, only set if command is seek_relative."""

    def __post_init__(self) -> None:
        """Validate field values and command consistency."""
        if self.command == MediaCommand.VOLUME:
            if self.volume is None:
                raise ValueError("Volume must be provided when command is 'volume'")
            if not 0 <= self.volume <= 100:
                raise ValueError(f"Volume must be in range 0-100, got {self.volume}")
        elif self.volume is not None:
            raise ValueError(f"Volume should not be provided for command '{self.command.value}'")

        if self.command == MediaCommand.MUTE:
            if self.mute is None:
                raise ValueError("Mute must be provided when command is 'mute'")
        elif self.mute is not None:
            raise ValueError(f"Mute should not be provided for command '{self.command.value}'")

        if self.command == MediaCommand.SEEK:
            if self.position_ms is None:
                raise ValueError("position_ms must be provided when command is 'seek'")
            if self.position_ms < 0:
                raise ValueError(f"position_ms must be non-negative, got {self.position_ms}")
            if self.offset_ms is not None:
                raise ValueError("offset_ms should not be provided for command 'seek'")
        elif self.command == MediaCommand.SEEK_RELATIVE:
            if self.offset_ms is None:
                raise ValueError("offset_ms must be provided when command is 'seek_relative'")
            if self.position_ms is not None:
                raise ValueError("position_ms should not be provided for command 'seek_relative'")
        else:
            if self.position_ms is not None:
                raise ValueError(
                    f"position_ms should not be provided for command '{self.command.value}'"
                )
            if self.offset_ms is not None:
                raise ValueError(
                    f"offset_ms should not be provided for command '{self.command.value}'"
                )

    class Config(SendspinConfig):
        """Config for parsing json messages."""

        omit_none = True


# Server -> Client: server/state controller object
@dataclass
class ControllerStatePayload(SendspinModel):
    """Controller state object in server/state message."""

    supported_commands: list[MediaCommand]
    """
    Subset of: play, pause, stop, next, previous, volume, mute, repeat_off, repeat_one,
    repeat_all, shuffle, unshuffle, switch, seek, seek_relative.
    """
    volume: int
    """Volume of the whole group, range 0-100."""
    muted: bool
    """Mute state of the whole group."""
    repeat: RepeatMode
    """Repeat mode: 'off' = no repeat, 'one' = repeat current track, 'all' = repeat all."""
    shuffle: bool
    """Whether shuffle is enabled."""
    seek_max_ms: int | None = None
    """Max absolute position (ms) a 'seek' may target. Set only when 'seek' is supported."""

    @classmethod
    def __pre_deserialize__(cls, d: dict[str, Any]) -> dict[str, Any]:
        """Drop supported commands this implementation does not recognize."""
        if "supported_commands" not in d:
            return d
        commands, _ = split_enum_values(d["supported_commands"], MediaCommand)
        return d | {"supported_commands": commands}

    def __post_init__(self) -> None:
        """Validate field values."""
        if not 0 <= self.volume <= 100:
            raise ValueError(f"Volume must be in range 0-100, got {self.volume}")

    class Config(SendspinConfig):
        """Config for serializing state messages."""

        omit_none = True
