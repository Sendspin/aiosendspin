"""VisualizerGroupRole - group-level visualizer coordination."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiosendspin.models.visualizer import BeatAvailability, BeatTiming
from aiosendspin.server.roles.base import GroupRole, Role
from aiosendspin.server.roles.visualizer.types import VisualizerRoleProtocol

if TYPE_CHECKING:
    from aiosendspin.server.group import SendspinGroup


class VisualizerGroupRole(GroupRole):
    """VisualizerGroupRole - group-level visualizer coordination.

    Reliable beat extraction is too computationally intensive to run on
    the fly, so the schedule is not calculated inside aiosendspin. The
    embedding server is responsible for computing beats offline and
    feeding them in via `append_beat_schedule()`.

    The group role is shared across all visualizer wire versions
    (`visualizer@v1`, `visualizer@_draft_r1`, …). Only members that
    satisfy `VisualizerRoleProtocol` are treated as beat-capable; older
    wire versions that lack the protocol attributes are silently
    skipped.
    """

    role_family = "visualizer"

    def __init__(self, group: SendspinGroup) -> None:
        """Initialize VisualizerGroupRole."""
        super().__init__(group)
        self._current_beats: list[BeatTiming] = []
        self._beat_availability: BeatAvailability = BeatAvailability.PENDING

    @property
    def current_beats(self) -> list[BeatTiming]:
        """Snapshot of beats in the current schedule."""
        return list(self._current_beats)

    @property
    def beat_availability(self) -> BeatAvailability:
        """Current beat availability for this source."""
        return self._beat_availability

    @property
    def beats_wanted(self) -> bool:
        """True if any subscribed visualizer client wants beats.

        The server should gate beat computation on this: skip the
        (expensive) beat analysis while it is False. It reflects
        membership live, so a client joining mid-stream flips it True
        on the next read. Re-check it on the client/group membership
        events the server already receives.
        """
        return any(
            member.wants_beats
            for member in self._members
            if isinstance(member, VisualizerRoleProtocol)
        )

    def set_beat_availability(self, availability: BeatAvailability) -> None:
        """Declare whether beats will arrive for the current source.

        Default is `PENDING`: beats may still arrive via
        `append_beat_schedule()`; roles defer `beat` in
        `stream/start.types` until the first schedule lands.

        `UNAVAILABLE` tells subscribers no beats will arrive for this
        source — `beat` stays out of the negotiated types and any
        currently stored schedule is dropped (a late joiner must not
        receive stale beats for a source that has been declared
        UNAVAILABLE).
        """
        self._beat_availability = availability
        if availability is BeatAvailability.UNAVAILABLE:
            self._current_beats = []
        # Snapshot members: callbacks may synchronously re-enter the
        # group (e.g. send_binary → backpressure → role disconnect →
        # unsubscribe → _members.remove).
        for role in list(self._members):
            if isinstance(role, VisualizerRoleProtocol):
                role.set_beat_availability(availability)

    def on_member_join(self, role: Role) -> None:
        """Replay availability + current schedule to a fresh beat-capable role.

        Roles that don't want beats are skipped (avoids replaying a
        whole-track schedule onto a wire that can't carry it). The
        schedule is filtered to future-only beats relative to the
        current playhead so the joining role's wire-ts guard doesn't
        burn cycles dropping every past beat one by one.
        """
        if not isinstance(role, VisualizerRoleProtocol):
            return
        role.set_beat_availability(self._beat_availability)
        if not role.wants_beats:
            return
        role.clear_beats()
        if not self._current_beats:
            return
        now_us = self._group._server.clock.now_us()  # noqa: SLF001
        future_beats = [b for b in self._current_beats if b.timestamp_us >= now_us]
        if future_beats:
            role.append_beats(future_beats)

    def on_member_leave(self, role: Role) -> None:  # noqa: ARG002
        """No-op override.

        Per-member state (timers, pending queues) is cleaned by the
        role's own `on_disconnect`. The base `GroupRole.unsubscribe`
        removes the member from `self._members` for us. Documented
        here so future maintainers don't add a cleanup that duplicates
        the role-side teardown.
        """
        return

    def append_beat_schedule(self, beats: list[BeatTiming]) -> None:
        """Append upcoming beats and broadcast to subscribers.

        Each BeatTiming carries its server-clock timestamp and a
        downbeat flag. Timestamps within a single call must be strictly
        increasing. Adjacent calls may share a boundary ts (server feeds
        `[..100]` then `[100..200]`); the duplicate head is dropped so
        producers don't have to shift timestamps by 1 µs.

        No-op when availability is `UNAVAILABLE`.
        """
        if self._beat_availability is BeatAvailability.UNAVAILABLE:
            return
        # Snapshot the caller's list so post-call mutations can't change
        # what subscribers see, and so validation runs against a stable
        # iteration order.
        snapshot = list(beats)
        if not snapshot:
            return
        last_ts = self._current_beats[-1].timestamp_us if self._current_beats else None
        # Allow a shared-boundary duplicate at the head (segment seam),
        # then require strictly-increasing inside the segment.
        if last_ts is not None and snapshot and snapshot[0].timestamp_us == last_ts:
            snapshot = snapshot[1:]
        for beat in snapshot:
            if last_ts is not None and beat.timestamp_us <= last_ts:
                raise ValueError(
                    f"beat timestamps must be strictly increasing; "
                    f"got {beat.timestamp_us} after {last_ts}"
                )
            last_ts = beat.timestamp_us
        if not snapshot:
            return
        self._current_beats.extend(snapshot)
        new_segment = list(snapshot)
        for role in list(self._members):
            if isinstance(role, VisualizerRoleProtocol):
                role.append_beats(new_segment)

    def clear_beat_schedule(self, *, from_us: int | None = None) -> None:
        """Drop the stored schedule and broadcast a clear marker.

        Also resets availability to `PENDING` — each track starts fresh,
        so a previous `UNAVAILABLE` declaration does not silently
        block the next track's schedule. Server can redeclare
        `UNAVAILABLE` after the next `append_beat_schedule` if needed.

        With `from_us`, only the stored beats at or after it are dropped and
        nothing is broadcast. Use it after `PushStream.clear_from()`, which
        already drops those beats from the roles.
        """
        if from_us is not None:
            self._current_beats = [b for b in self._current_beats if b.timestamp_us < from_us]
            return
        self._current_beats = []
        self._beat_availability = BeatAvailability.PENDING
        for role in list(self._members):
            if isinstance(role, VisualizerRoleProtocol):
                role.clear_beats()
                role.set_beat_availability(BeatAvailability.PENDING)
