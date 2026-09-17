"""
Metadata messages for the Sendspin protocol.

This module contains messages specific to clients with the metadata role, which
handle display of track information and playback progress. Metadata clients
receive state updates with track details.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import SendspinConfig, SendspinModel


@dataclass
class Progress(SendspinModel):
    """Playback progress information."""

    track_progress: int
    """Track progress in milliseconds, since start of track."""
    track_duration: int
    """Track duration in milliseconds. 0 for unlimited/unknown duration (e.g., live streams)."""
    playback_speed: int
    """Playback speed multiplier * 1000 (e.g., 1000 = normal, 1500 = 1.5x, 0 = paused)."""

    def __post_init__(self) -> None:
        """Validate field values."""
        # Validate track_progress is non-negative
        if self.track_progress < 0:
            raise ValueError(f"track_progress must be non-negative, got {self.track_progress}")

        # Validate track_duration is non-negative (0 allowed for live streams)
        if self.track_duration < 0:
            raise ValueError(f"track_duration must be non-negative, got {self.track_duration}")

        # Validate playback_speed is non-negative
        if self.playback_speed < 0:
            raise ValueError(f"playback_speed must be non-negative, got {self.playback_speed}")

    class Config(SendspinConfig):
        """Config for parsing json messages."""

        omit_default = True


# Server -> Client: server/state metadata object
@dataclass
class SessionUpdateMetadata(SendspinModel):
    """Metadata object in server/state message."""

    timestamp: int
    """Server clock time in microseconds for when this metadata is valid."""
    title: str | None = None
    artist: str | None = None
    album_artist: str | None = None
    album: str | None = None
    artwork_url: str | None = None
    year: int | None = None
    track: int | None = None
    progress: Progress | None = None
    """
    Playback progress information.

    Omitting it clears the client's playback position.
    """

    def __post_init__(self) -> None:
        """Validate field values."""
        if self.year is not None and self.year < 0:
            raise ValueError(f"year must be non-negative, got {self.year}")

        # Validate track number is positive
        if self.track is not None and self.track <= 0:
            raise ValueError(f"track must be positive, got {self.track}")

    class Config(SendspinConfig):
        """Config for parsing json messages."""

        omit_none = True
