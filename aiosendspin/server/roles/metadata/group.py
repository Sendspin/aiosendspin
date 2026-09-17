"""MetadataGroupRole - group-level metadata coordination."""

from __future__ import annotations

from dataclasses import replace

from aiosendspin.models.core import ServerStateMessage, ServerStatePayload
from aiosendspin.models.metadata import SessionUpdateMetadata
from aiosendspin.server.roles.metadata.events import MetadataClearedEvent, MetadataUpdatedEvent
from aiosendspin.server.roles.metadata.state import Metadata
from aiosendspin.server.roles.scheduled_state import ScheduledStateGroupRole

_UNSET = object()


class MetadataGroupRole(ScheduledStateGroupRole[Metadata]):
    """Coordinate metadata across a group.

    Stores current metadata state and pushes updates to subscribed MetadataRoles.
    """

    role_family = "metadata"

    @property
    def metadata(self) -> Metadata | None:
        """Return current metadata."""
        return self._state.current(self._now_us())

    @property
    def track_progress(self) -> int | None:
        """Return the playback position in milliseconds as of now, or None when unknown.

        During an active stream the stored position is extrapolated at the playback speed and
        clamped to the track duration.
        """
        current_time_us = self._now_us()
        current = self._state.current(current_time_us)
        if current is None or current.track_progress is None:
            return None

        if (
            current.timestamp_us is not None
            and self._group.has_active_stream
            and current.playback_speed is not None
        ):
            elapsed_us = current_time_us - current.timestamp_us
            elapsed_ms = (elapsed_us * current.playback_speed) // 1_000_000
            calculated_progress = current.track_progress + elapsed_ms

            if current.track_duration is not None and current.track_duration > 0:
                calculated_progress = max(0, min(calculated_progress, current.track_duration))
            else:
                calculated_progress = max(0, calculated_progress)

            return calculated_progress

        return current.track_progress

    def freeze_progress(self) -> None:
        """Snapshot current progress and stop further client-side progress extrapolation."""
        metadata = self.metadata
        if metadata is None or (current_progress := self.track_progress) is None:
            return

        self.set_metadata(
            replace(
                metadata,
                track_progress=current_progress,
                playback_speed=0,
            )
        )

    def reset_progress(self) -> None:
        """Report the playback position as 0 from now and stop client-side extrapolation.

        Unlike `update`, the reset is sent even when the position is already close to 0.
        Nothing is sent when there is no position to report, or it already reads 0 while
        stopped; a scheduled metadata update is cancelled in every case.
        """
        metadata = self.metadata
        if metadata is None or metadata.track_progress is None:
            self.cancel_scheduled()
            return

        if metadata.track_progress == 0 and metadata.playback_speed == 0:
            # The position a client computes is already 0, so there is no change to convey.
            self.cancel_scheduled()
            return

        self._apply_metadata(
            replace(metadata, track_progress=0, playback_speed=0, timestamp_us=None), force=True
        )

    def set_metadata(self, metadata: Metadata | None) -> None:
        """Set metadata and push the full metadata state to all subscribed roles.

        Nothing is sent when the metadata is unchanged. `None` clears the metadata, and
        any scheduled metadata, at once.

        Metadata whose `timestamp_us` is in the future is scheduled to take effect then,
        replacing any metadata already scheduled. It is sent to clients at most 20
        seconds ahead, and `MetadataUpdatedEvent` fires now, carrying that timestamp.
        Metadata taking effect now cancels scheduled metadata; so do `update()`, `seek()`
        and `reset_progress()`. To show two tracks in sequence, schedule the second only after
        the first took effect.
        """
        self._apply_metadata(metadata, force=False)

    def seek(self, track_progress: int) -> None:
        """Set the playback position in milliseconds as of now and push it to all members.

        Unlike `update`, the new position is sent even when it is close to the current one.
        Like it, a seek cancels scheduled metadata.

        Raises ValueError if there is no metadata with a `playback_speed`, or if
        `track_progress` is negative.
        """
        metadata = self.metadata
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
        current = self.metadata or Metadata()
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

    def clear(self) -> None:
        """Clear all metadata, and any scheduled metadata, at once."""
        self.set_metadata(None)

    def _apply_metadata(self, metadata: Metadata | None, *, force: bool) -> None:
        """Apply or schedule metadata and push it, skipping unchanged metadata unless `force`."""
        now_us = self._now_us()
        last_metadata = self._state.current(now_us)
        timestamp = now_us
        if metadata is not None:
            if metadata.timestamp_us is None:
                metadata = replace(metadata, timestamp_us=timestamp)
            else:
                timestamp = metadata.timestamp_us

        if metadata is not None and timestamp > now_us:
            self._schedule(metadata, timestamp)
        else:
            if self._state.pending_timestamp_us is None and not force:
                if metadata is None and last_metadata is None:
                    return
                if metadata is not None and metadata.equals(last_metadata):
                    return
            self._apply(metadata, timestamp)

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

    def _state_message(self, state: Metadata | None, timestamp_us: int) -> ServerStateMessage:
        metadata_update = (
            SessionUpdateMetadata(timestamp=timestamp_us)
            if state is None
            else state.snapshot_update(timestamp_us)
        )
        return ServerStateMessage(ServerStatePayload(metadata=metadata_update))

    def _current_state(self, now_us: int) -> Metadata | None:
        current = self._state.current(now_us)
        if current is None:
            return None
        return replace(current, track_progress=self.track_progress, timestamp_us=now_us)
