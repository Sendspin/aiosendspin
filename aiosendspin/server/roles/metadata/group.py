"""MetadataGroupRole - group-level metadata coordination."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from aiosendspin.models.core import (
    LegacyServerStateClearMessage,
    ServerStateMessage,
    ServerStatePayload,
)
from aiosendspin.models.metadata import SessionUpdateMetadata
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
        self._pending_metadata: Metadata | None = None
        self._pending_update: SessionUpdateMetadata | None = None
        self._track_progress_timestamp_us: int | None = None

    @property
    def metadata(self) -> Metadata | None:
        """Return current metadata."""
        return self._current_metadata

    def on_member_join(self, role: Role) -> None:
        """Send current metadata to newly joined member."""
        self._send_state_to_role(role)

    def _send_state_to_role(self, role: Role) -> None:
        """
        Send the complete current metadata state to a single role.

        Without metadata the state is a timestamp-only object.
        """
        timestamp = self._group._server.clock.now_us()  # noqa: SLF001
        if self._current_metadata is None:
            self._send_metadata(role, SessionUpdateMetadata(timestamp=timestamp), cleared=True)
            return

        current = replace(self._current_metadata, track_progress=self.track_progress)
        self._send_metadata(role, current.snapshot_update(timestamp), cleared=False)

    def _send_metadata(self, role: Role, update: SessionUpdateMetadata, *, cleared: bool) -> None:
        """Send a metadata object to a role; `cleared` marks the object for no metadata."""
        # DEPRECATED(spec-pr-275): remove in aiosendspin <version>
        if cleared and role.clears_state_with_null():
            role.send_message(LegacyServerStateClearMessage(self.role_family))
            return
        role.send_message(ServerStateMessage(ServerStatePayload(metadata=update)))

    @property
    def track_progress(self) -> int | None:
        """Return the playback position in milliseconds as of now, or None when unknown.

        During an active stream the stored position is extrapolated at the playback speed and
        clamped to the track duration.
        """
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
        if metadata is None or (current_progress := self.track_progress) is None:
            return

        self.set_metadata(
            replace(
                metadata,
                track_progress=current_progress,
                playback_speed=0,
                timestamp_us=None,
            )
        )

    def set_metadata(self, metadata: Metadata | None) -> None:
        """Set metadata and push the full metadata state to all subscribed roles.

        Nothing is sent when the metadata is unchanged. `None` clears the metadata.
        """
        self._apply_metadata(metadata, force=False)

    def seek(self, track_progress: int) -> None:
        """Set the playback position in milliseconds as of now and push it to all members.

        Unlike `update`, the new position is sent even when it is close to the current one.

        Raises ValueError if there is no metadata with a `playback_speed`, or if
        `track_progress` is negative.
        """
        metadata = self._current_metadata
        if metadata is None or metadata.playback_speed is None:
            raise ValueError("seek requires metadata with a playback_speed")
        self._apply_metadata(
            replace(metadata, track_progress=track_progress, timestamp_us=None), force=True
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
            kwargs["track_progress"] = self.track_progress
            kwargs["timestamp_us"] = None

        new_metadata = replace(current, **kwargs)  # type: ignore[arg-type]
        self.set_metadata(new_metadata)

    def clear(self, *, timestamp_us: int | None = None) -> None:
        """Clear all metadata."""
        self.set_metadata(None)

    def _apply_metadata(self, metadata: Metadata | None, *, force: bool) -> None:
        """Store metadata and push it, skipping unchanged metadata unless `force` is set."""
        timestamp = self._group._server.clock.now_us()  # noqa: SLF001

        if metadata is not None:
            if metadata.timestamp_us is None:
                metadata = replace(metadata, timestamp_us=timestamp)
            else:
                timestamp = metadata.timestamp_us

        if metadata is None and self._current_metadata is None:
            return
        if not force and metadata is not None and metadata.equals(self._current_metadata):
            return

        last_metadata = self._current_metadata
        metadata_update = (
            SessionUpdateMetadata(timestamp=timestamp)
            if metadata is None
            else metadata.snapshot_update(timestamp)
        )

        self._current_metadata = metadata

        if metadata is not None and metadata.track_progress is not None:
            self._track_progress_timestamp_us = timestamp

        for role in self._members:
            self._send_metadata(role, metadata_update, cleared=metadata is None)

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
