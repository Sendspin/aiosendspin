"""Color state for the Sendspin protocol."""

from __future__ import annotations

from dataclasses import dataclass, field

from aiosendspin.models.color import SessionUpdateColor, _validate_rgb

_RGB = tuple[int, int, int]
_WHITE: _RGB = (255, 255, 255)
_BLACK: _RGB = (0, 0, 0)
_MIN_CONTRAST = 4.5
_RGB_FIELDS: tuple[str, ...] = (
    "background_dark",
    "background_light",
    "primary",
    "accent",
    "on_dark",
    "on_light",
)


def _relative_luminance(rgb: _RGB) -> float:
    """Compute WCAG relative luminance for an sRGB color."""

    def channel(c: int) -> float:
        s = c / 255.0
        return s / 12.92 if s <= 0.03928 else ((s + 0.055) / 1.055) ** 2.4

    r, g, b = rgb
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def _contrast_ratio(a: _RGB, b: _RGB) -> float:
    """Compute WCAG contrast ratio between two RGB colors."""
    la = _relative_luminance(a)
    lb = _relative_luminance(b)
    lighter, darker = (la, lb) if la >= lb else (lb, la)
    return (lighter + 0.05) / (darker + 0.05)


def _assert_contrast(label: str, a: _RGB, b: _RGB) -> None:
    ratio = _contrast_ratio(a, b)
    if ratio < _MIN_CONTRAST:
        raise ValueError(f"{label}: contrast {ratio:.2f} < {_MIN_CONTRAST}:1")


@dataclass(frozen=True)
class Color:
    """Color palette for the current audio.

    Enforces the WCAG ≥4.5:1 contrast invariants from the Sendspin color@v1
    spec at construction time.
    """

    background_dark: _RGB | None = None
    """Background color for dark mode as (R, G, B)."""
    background_light: _RGB | None = None
    """Background color for light mode as (R, G, B)."""
    primary: _RGB | None = None
    """Dominant color as (R, G, B). Not adjusted for contrast."""
    accent: _RGB | None = None
    """Secondary or complementary color as (R, G, B). Not adjusted for contrast."""
    on_dark: _RGB | None = None
    """Light color for use on dark backgrounds as (R, G, B)."""
    on_light: _RGB | None = None
    """Dark color for use on light backgrounds as (R, G, B)."""
    timestamp_us: int | None = field(default=None, compare=False)
    """Server timestamp when this palette takes effect."""

    def __post_init__(self) -> None:
        """Validate RGB shape, range, and spec-mandated contrast pairs."""
        for name in _RGB_FIELDS:
            value = getattr(self, name)
            if value is not None:
                _validate_rgb(name, value)
        if self.background_dark is not None:
            _assert_contrast("background_dark vs white text", self.background_dark, _WHITE)
            if self.on_dark is not None:
                _assert_contrast("background_dark vs on_dark", self.background_dark, self.on_dark)
        if self.background_light is not None:
            _assert_contrast("background_light vs black text", self.background_light, _BLACK)
            if self.on_light is not None:
                _assert_contrast(
                    "background_light vs on_light", self.background_light, self.on_light
                )
        if self.on_dark is not None:
            _assert_contrast("on_dark vs black text", self.on_dark, _BLACK)
        if self.on_light is not None:
            _assert_contrast("on_light vs white text", self.on_light, _WHITE)

    def snapshot_update(self, timestamp: int) -> SessionUpdateColor:
        """Build a SessionUpdateColor carrying the full current state."""
        return SessionUpdateColor(
            timestamp=timestamp,
            background_dark=self.background_dark,
            background_light=self.background_light,
            primary=self.primary,
            accent=self.accent,
            on_dark=self.on_dark,
            on_light=self.on_light,
        )
