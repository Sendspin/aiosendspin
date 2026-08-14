"""Metadata handling for the Sendspin protocol."""

from __future__ import annotations

from collections.abc import Set as AbstractSet
from dataclasses import dataclass

from aiosendspin.models.metadata import Progress, SessionUpdateMetadata
from aiosendspin.models.types import RepeatMode


@dataclass
class Metadata:
    """Metadata for media playback."""

    title: str | None = None
    """Title of the current media."""
    artist: str | None = None
    """Artist of the current media."""
    album_artist: str | None = None
    """Album artist of the current media."""
    album: str | None = None
    """Album of the current media."""
    artwork_url: str | None = None
    """Artwork URL of the current media."""
    year: int | None = None
    """Release year of the current media."""
    track: int | None = None
    """Track number of the current media."""
    # DEPRECATED(spec-pr-175): remove in aiosendspin <version>
    repeat: RepeatMode | None = None
    """Ignored. Use `ControllerGroupRole.set_repeat` instead."""
    # DEPRECATED(spec-pr-175): remove in aiosendspin <version>
    shuffle: bool | None = None
    """Ignored. Use `ControllerGroupRole.set_shuffle` instead."""

    # Progress fields:
    # A track_progress requires a playback_speed; progress is sent whenever both are set
    track_progress: int | None = None
    """Track progress in milliseconds at the last update time. Requires `playback_speed`."""
    track_duration: int | None = None
    """
    Track duration in milliseconds.

    Use 0 or None for unlimited/unknown duration (e.g., live streams); None is sent as 0.
    """
    playback_speed: int | None = None
    """Playback speed multiplier * 1000 (e.g., 1000 = normal, 1500 = 1.5x, 0 = paused)."""

    timestamp_us: int | None = None
    """
    Timestamp in microseconds when this metadata was captured.

    You don't need to set this, since it will be set automatically by set_metadata() if not
    provided.
    """

    def __post_init__(self) -> None:
        """Reject a playback position that has no playback speed."""
        if self.track_progress is not None and self.playback_speed is None:
            raise ValueError("playback_speed is required when track_progress is set")

    def equals(self, other: Metadata | None, progress_tolerance_ms: int = 500) -> bool:
        """
        Check if metadata is meaningfully equal.

        Args:
            other: The other Metadata object to compare with.
            progress_tolerance_ms: Tolerance in milliseconds for track progress comparison.

        Returns:
            True if metadata is meaningfully equal, False otherwise.
        """
        if other is None:
            return False

        # Compare all non-progress fields
        if not (
            self.title == other.title
            and self.artist == other.artist
            and self.album_artist == other.album_artist
            and self.album == other.album
            and self.artwork_url == other.artwork_url
            and self.year == other.year
            and self.track == other.track
            and self.track_duration == other.track_duration
            and self.playback_speed == other.playback_speed
        ):
            return False

        # If both have no progress info, they're equal
        if self.track_progress is None and other.track_progress is None:
            return True

        # If only one has progress info, they're different
        if self.track_progress is None or other.track_progress is None:
            return False

        # If we don't have timestamps, fall back to simple tolerance check
        if self.timestamp_us is None or other.timestamp_us is None:
            return abs(self.track_progress - other.track_progress) <= progress_tolerance_ms

        # Calculate expected progress change based on elapsed time and playback speed
        time_diff_ms = (other.timestamp_us - self.timestamp_us) / 1000
        # Only None means unknown. A speed of 0 is a real paused rate.
        speed = 1000 if self.playback_speed is None else self.playback_speed
        playback_speed = speed / 1000
        expected_progress_change = time_diff_ms * playback_speed

        # Calculate actual progress change
        actual_progress_change = other.track_progress - self.track_progress

        # Check if the difference between expected and actual is within tolerance
        progress_drift = abs(actual_progress_change - expected_progress_change)
        return progress_drift <= progress_tolerance_ms

    def snapshot_update(self, timestamp: int) -> SessionUpdateMetadata:
        """Build a SessionUpdateMetadata carrying the full current state."""
        progress = None
        if self.track_progress is not None and self.playback_speed is not None:
            progress = Progress(
                track_progress=self.track_progress,
                track_duration=self.track_duration or 0,
                playback_speed=self.playback_speed,
            )
        return SessionUpdateMetadata(
            timestamp=timestamp,
            title=self.title,
            artist=self.artist,
            album_artist=self.album_artist,
            album=self.album,
            artwork_url=self.artwork_url,
            year=self.year,
            track=self.track,
            progress=progress,
        )
