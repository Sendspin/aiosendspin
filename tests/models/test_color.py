"""Tests for SessionUpdateColor wire model."""

from __future__ import annotations

import pytest

from aiosendspin.models.color import SessionUpdateColor


def test_unset_fields_omitted_from_json() -> None:
    """Fields left unset are omitted from serialized JSON."""
    update = SessionUpdateColor(timestamp=42, primary=(10, 20, 30))
    payload = update.to_dict()
    assert payload == {"timestamp": 42, "primary": [10, 20, 30]}


def test_explicit_none_omitted_from_json() -> None:
    """Explicit None is omitted, so no leaf null reaches the wire."""
    update = SessionUpdateColor(timestamp=7, primary=None, on_light=(0, 0, 0))
    assert update.to_dict() == {"timestamp": 7, "on_light": [0, 0, 0]}


def test_roundtrip_decodes_to_tuples() -> None:
    """A wire dict with integer arrays decodes back to tuples."""
    raw = {
        "timestamp": 1,
        "background_dark": [1, 2, 3],
        "primary": [255, 0, 0],
    }
    update = SessionUpdateColor.from_dict(raw)
    assert update.background_dark == (1, 2, 3)
    assert update.primary == (255, 0, 0)


def test_rgb_component_out_of_range_raises() -> None:
    """Validation rejects component outside 0-255."""
    with pytest.raises(ValueError, match="primary values must be 0-255"):
        SessionUpdateColor(timestamp=0, primary=(300, 0, 0))  # type: ignore[arg-type]


def test_unset_field_passes_validation() -> None:
    """An unset field defaults to None and does not trigger validation."""
    update = SessionUpdateColor(timestamp=0)
    assert update.primary is None


def test_rgb_wrong_length_raises() -> None:
    """Validation rejects tuples that are not exactly 3 components."""
    with pytest.raises(ValueError, match=r"primary must be \(R, G, B\)"):
        SessionUpdateColor(timestamp=0, primary=(1, 2, 3, 4))  # type: ignore[arg-type]
