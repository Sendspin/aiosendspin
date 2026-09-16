"""MetadataGroupRole - group-level metadata coordination."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from aiosendspin.models.core import ServerStateMessage, ServerStatePayload
from aiosendspin.server.roles.base import GroupRole, Role
from aiosendspin.server.roles.metadata.events import MetadataClearedEvent, MetadataUpdatedEvent
from aiosendspin.server.roles.metadata.state import Metadata

if TYPE_CHECKING:
    from aiosendspin.server.group import SendspinGroup

_UNSET = object()


class MetadataGroupRole(GroupRole):
    """Coordinate metadata across a group.

    Stores current metadata state and pushes updates to subscribed MetadataRoles.
    """

    role_family = "metadata"

    def __init__(self, group: SendspinGroup) -> None:
        """Initialize MetadataGroupRole."""
        super().__init__(group)
        self._current_metadata: Metadata | None = None
        self._track_progress_timestamp_us: int | None = None

    @property
    def metadata(self) -> Metadata | None:
        """Return current metadata."""
        return self._current_metadata

    def on_member_join(self, role: Role) -> None:
        """Send current metadata to newly joined member."""
        self._send_state_to_role(role)

    def _send_state_to_role(self, role: Role) -> None:
        """Send current metadata state to a single role."""
        if self._current_metadata is None:
            role.send_message(ServerStateMessage(ServerStatePayload(metadata=None)))
            return

        timestamp = self._group._server.clock.now_us()  # noqa: SLF001
        current = replace(self._current_metadata, track_progress=self._get_current_track_progress())
        metadata_update = current.snapshot_update(timestamp)
        state_message = ServerStateMessage(ServerStatePayload(metadata=metadata_update))
        role.send_message(state_message)

    def _get_current_track_progress(self) -> int | None:
        """Calculate current track progress in milliseconds."""
        if self._current_metadata is None or self._current_metadata.track_progress is None:
            return None

        if (
            self._track_progress_timestamp_us is not None
            and self._group.has_active_stream
            and self._current_metadata.playback_speed is not None
        ):
            current_time_us = self._group._server.clock.now_us()  # noqa: SLF001
            elapsed_us = current_time_us - self._track_progress_timestamp_us
            elapsed_ms = (elapsed_us * self._current_metadata.playback_speed) // 1_000_000
            calculated_progress = self._current_metadata.track_progress + elapsed_ms

            if (
                self._current_metadata.track_duration is not None
                and self._current_metadata.track_duration > 0
            ):
                calculated_progress = max(
                    0, min(calculated_progress, self._current_metadata.track_duration)
                )
            else:
                calculated_progress = max(0, calculated_progress)

            return calculated_progress

        return self._current_metadata.track_progress

    def freeze_progress(self) -> None:
        """Snapshot current progress and stop further client-side progress extrapolation."""
        metadata = self._current_metadata
        if metadata is None or (current_progress := self._get_current_track_progress()) is None:
            return

        self.set_metadata(
            replace(
                metadata,
                track_progress=current_progress,
                playback_speed=0,
            )
        )

    def set_metadata(self, metadata: Metadata | None) -> None:
        """Set metadata and push the full metadata state to all subscribed roles.

        Nothing is sent when the metadata is unchanged. `None` clears the metadata.
        """
        timestamp = self._group._server.clock.now_us()  # noqa: SLF001

        if metadata is not None:
            if metadata.timestamp_us is None:
                metadata = replace(metadata, timestamp_us=timestamp)
            else:
                timestamp = metadata.timestamp_us

        if metadata is None and self._current_metadata is None:
            return
        if metadata is not None and metadata.equals(self._current_metadata):
            return

        last_metadata = self._current_metadata
        metadata_update = None if metadata is None else metadata.snapshot_update(timestamp)

        self._current_metadata = metadata

        if metadata is not None and metadata.track_progress is not None:
            self._track_progress_timestamp_us = timestamp

        for role in self._members:
            state_message = ServerStateMessage(ServerStatePayload(metadata=metadata_update))
            role.send_message(state_message)

        if metadata is None:
            self.emit_group_event(
                MetadataClearedEvent(previous_metadata=last_metadata, timestamp_us=timestamp)
            )
            return
        self.emit_group_event(
            MetadataUpdatedEvent(
                metadata=metadata,
                previous_metadata=last_metadata,
                timestamp_us=timestamp,
            )
        )

    def update(
        self,
        *,
        title: str | None | object = _UNSET,
        artist: str | None | object = _UNSET,
        album_artist: str | None | object = _UNSET,
        album: str | None | object = _UNSET,
        artwork_url: str | None | object = _UNSET,
        year: int | None | object = _UNSET,
        track: int | None | object = _UNSET,
        track_progress: int | None | object = _UNSET,
        track_duration: int | None | object = _UNSET,
        playback_speed: int | None | object = _UNSET,
    ) -> None:
        """Batch update multiple metadata fields.

        Fields set to `_UNSET` are left unchanged. Passing `None` clears a field.
        A supplied `track_progress` is taken as the position now. Otherwise, during an
        active stream, the update carries the current extrapolated position.

        Raises ValueError if the result has a `track_progress` without a `playback_speed`.
        """
        current = self._current_metadata or Metadata()
        kwargs: dict[str, object] = {}
        if title is not _UNSET:
            kwargs["title"] = title
        if artist is not _UNSET:
            kwargs["artist"] = artist
        if album_artist is not _UNSET:
            kwargs["album_artist"] = album_artist
        if album is not _UNSET:
            kwargs["album"] = album
        if artwork_url is not _UNSET:
            kwargs["artwork_url"] = artwork_url
        if year is not _UNSET:
            kwargs["year"] = year
        if track is not _UNSET:
            kwargs["track"] = track
        if track_progress is not _UNSET:
            kwargs["track_progress"] = track_progress
        if track_duration is not _UNSET:
            kwargs["track_duration"] = track_duration
        if playback_speed is not _UNSET:
            kwargs["playback_speed"] = playback_speed

        if not kwargs:
            return

        if track_progress is not _UNSET or current.track_progress is None:
            kwargs["timestamp_us"] = None
        elif self._group.has_active_stream:
            # The stored position is only valid at its own timestamp, so move it to now.
            kwargs["track_progress"] = self._get_current_track_progress()
            kwargs["timestamp_us"] = None

        new_metadata = replace(current, **kwargs)  # type: ignore[arg-type]
        self.set_metadata(new_metadata)

    def clear(self) -> None:
        """Clear all metadata."""
        self.set_metadata(None)
