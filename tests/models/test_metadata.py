"""Tests for SessionUpdateMetadata wire model."""

from __future__ import annotations

import pytest

from aiosendspin.models.metadata import SessionUpdateMetadata


@pytest.mark.parametrize("year", [0, 999, 2050, 9999])
def test_year_accepts_any_non_negative_value(year: int) -> None:
    """Any non-negative year is accepted."""
    assert SessionUpdateMetadata(timestamp=0, year=year).year == year


def test_year_rejects_negative_value() -> None:
    """A negative year is rejected."""
    with pytest.raises(ValueError, match="non-negative"):
        SessionUpdateMetadata(timestamp=0, year=-1)
