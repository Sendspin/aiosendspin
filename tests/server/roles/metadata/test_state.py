"""Tests for Metadata.equals progress-drift handling."""

from __future__ import annotations

import json

import pytest

from aiosendspin.models.metadata import Progress
from aiosendspin.server.roles.metadata.state import Metadata


def test_equals_treats_paused_progress_as_frozen() -> None:
    """While paused (speed 0), unchanged progress across time stays equal (no phantom drift)."""
    first = Metadata(track_progress=42_000, playback_speed=0, timestamp_us=0)
    later = Metadata(track_progress=42_000, playback_speed=0, timestamp_us=2_000_000)

    assert first.equals(later)


def test_equals_detects_drift_at_normal_speed() -> None:
    """At 1x, progress that did not advance with elapsed time is not equal."""
    first = Metadata(track_progress=42_000, playback_speed=1000, timestamp_us=0)
    later = Metadata(track_progress=42_000, playback_speed=1000, timestamp_us=2_000_000)

    assert not first.equals(later)


def test_float_duration_from_caller_reaches_the_wire_as_an_integer() -> None:
    """A caller computing `duration_s * 1000` must not put a float on the wire.

    Music Assistant hands durations in as floats. The spec types track_duration as an
    integer and strict clients drop a mistyped field, so the coercion has to survive the
    whole Metadata -> SessionUpdateMetadata path, not just direct model construction.
    """
    metadata = Metadata(
        track_progress=1000.0,
        track_duration=217.0 * 1000,
        playback_speed=1000,
        year=2020.0,
        track=3.0,
    )

    payload = json.loads(metadata.snapshot_update(1).to_json())
    assert payload["progress"]["track_duration"] == 217_000
    assert type(payload["progress"]["track_duration"]) is int
    assert type(payload["year"]) is int
    assert type(payload["track"]) is int


def test_position_without_playback_speed_is_rejected() -> None:
    """A position cannot be sent without a speed, so it is rejected on construction."""
    with pytest.raises(ValueError, match="playback_speed is required"):
        Metadata(track_progress=5_000, track_duration=180_000)


def test_position_without_duration_is_sent_with_unknown_duration() -> None:
    """A position with no duration still reaches the wire, with duration 0 (unknown)."""
    update = Metadata(title="Live", track_progress=5_000, playback_speed=1000).snapshot_update(1)

    assert update.progress == Progress(track_progress=5_000, track_duration=0, playback_speed=1000)
