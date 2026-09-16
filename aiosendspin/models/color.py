"""
Color messages for the Sendspin protocol.

This module contains messages specific to clients with the color role, which
receive color palettes derived from the current audio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .base import SendspinConfig, SendspinModel

_RGB = tuple[int, int, int]
_RGB_LEN = 3


def _validate_rgb(name: str, value: _RGB) -> None:
    if len(value) != _RGB_LEN:
        raise ValueError(f"{name} must be (R, G, B) (length 3), got length {len(value)}")
    for component in value:
        if not (0 <= component <= 255):
            raise ValueError(f"{name} values must be 0-255, got {component}")


# Server -> Client: server/state color object
@dataclass
class SessionUpdateColor(SendspinModel):
    """Color object in server/state message."""

    _RGB_FIELDS: ClassVar[tuple[str, ...]] = (
        "background_dark",
        "background_light",
        "primary",
        "accent",
        "on_dark",
        "on_light",
    )

    timestamp: int
    """Server clock time in microseconds for when these colors are valid."""
    background_dark: _RGB | None = None
    """Background color for dark mode as `(R, G, B)`."""
    background_light: _RGB | None = None
    """Background color for light mode as `(R, G, B)`."""
    primary: _RGB | None = None
    """Dominant color as `(R, G, B)`."""
    accent: _RGB | None = None
    """Secondary or complementary color as `(R, G, B)`."""
    on_dark: _RGB | None = None
    """Light color for use on dark backgrounds as `(R, G, B)`."""
    on_light: _RGB | None = None
    """Dark color for use on light backgrounds as `(R, G, B)`."""

    def __post_init__(self) -> None:
        """Validate RGB fields."""
        for name in self._RGB_FIELDS:
            value = getattr(self, name)
            if value is not None:
                _validate_rgb(name, value)

    class Config(SendspinConfig):
        """Config for parsing json messages."""

        omit_none = True
