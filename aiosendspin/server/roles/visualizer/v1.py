"""Visualizer role implementation for the `visualizer@v1` wire.

Each binary message carries exactly one frame of one type. The role emits:
- `loudness` (msg 16) — per audio chunk
- `beat` (msg 17) — fed in via `append_beats` from offline analysis
- `f_peak` (msg 18) — per audio chunk
- `spectrum` (msg 19) — per audio chunk
- `peak` (msg 20) — per audio chunk when the onset detector fires
- `pitch` (msg 21) — per audio chunk when a confident pitch is detected

`beat` is *deferred* from `stream/start.types` until the first non-empty
schedule actually lands. While beats are still being computed upstream
the role advertises only the FFT-driven types so clients can render a
`peak`-based fallback without flicker; once `append_beats` first
delivers, the role re-emits `stream/start` with `beat` added and beats
begin riding the wire interleaved with periodic frames.

All beats drain through `on_audio_chunk` (audio chunks are delivered to
this role's `on_audio_chunk` regardless of negotiated types), so no
separate clock-based scheduler is needed.
"""

from __future__ import annotations

import asyncio
import logging
import struct
from collections import deque
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from aiosendspin.models.core import (
    ClientStatePayload,
    StreamClearMessage,
    StreamClearPayload,
    StreamEndMessage,
    StreamEndPayload,
    StreamRequestFormatPayload,
    StreamStartMessage,
    StreamStartPayload,
)
from aiosendspin.models.types import BinaryMessageType
from aiosendspin.models.visualizer import (
    BeatAvailability,
    BeatTiming,
    ClientHelloVisualizerSupport,
    StreamStartVisualizer,
    SupportedVisualizerType,
    VisualizerStatePayload,
)
from aiosendspin.server.audio import BufferTracker
from aiosendspin.server.roles.base import (
    AudioChunk,
    AudioRequirements,
    BinaryHandling,
    Role,
    StreamRequirements,
)
from aiosendspin.server.roles.visualizer.features import (
    ExtractedFrame,
    VisualizerFeatureExtractor,
)
from aiosendspin.server.roles.visualizer.packing import (
    FLAG_DOWNBEAT,
    pack_visualizer_frame,
)

if TYPE_CHECKING:
    from aiosendspin.server.client import SendspinClient

_LOGGER = logging.getLogger(__name__)

# Types the reference implementation knows how to compute. Unsupported
# types requested by the client are silently omitted per spec.
_IMPLEMENTED_TYPES: frozenset[SupportedVisualizerType] = frozenset(
    {"loudness", "f_peak", "spectrum", "beat", "peak", "pitch"}
)
# Types whose computation requires the FFT extractor. `beat` is the only
# supplied-externally type and does not require the extractor.
_FFT_DRIVEN_TYPES: frozenset[SupportedVisualizerType] = frozenset(
    {"loudness", "f_peak", "spectrum", "peak", "pitch"}
)
# Periodic frames for a beat-wanting client are held to this lead ahead of the
# playhead. Keeping the wire-ts cursor near the playhead means a beat schedule
# landing mid-stream (a flow-mode track change re-pushes the whole schedule)
# sits at the cursor — only a small window trails it and is dropped — instead
# of the whole schedule landing behind a cursor already pushed seconds ahead by
# frames. Held the whole time beats are wanted; only `UNAVAILABLE` lifts it.
_WARMUP_LEAD_US = 3_000_000


class VisualizerV1Role(Role):
    """Role implementation for `visualizer@v1` streaming."""

    def __init__(self, client: SendspinClient | None = None) -> None:
        """Initialize VisualizerV1Role."""
        if client is None:
            raise ValueError("VisualizerV1Role requires a client")
        self._client = client
        self._stream_started = False
        self._buffer_tracker: BufferTracker | None = None
        self._buffer_capacity = 0
        # The client's requested stream configuration; None until one is known.
        self._request: VisualizerStatePayload | None = None
        self._stream_config: StreamStartVisualizer | None = None
        self._extractor: VisualizerFeatureExtractor | None = None
        # Beats queued for delivery on the next audio chunk's drain.
        self._pending_beats: deque[BeatTiming] = deque()
        # True once `append_beats` has delivered a non-empty schedule for
        # the current stream. Gates `beat` in the negotiated types.
        self._has_beats_landed: bool = False
        # Server-side capability metadata for downbeat tracking, set via
        # `set_tracks_downbeats()` before `stream/start` is sent.
        self._tracks_downbeats: bool = False
        # Last-emitted timestamp across all visualizer binaries. The spec
        # requires non-decreasing timestamp order within the role; beats
        # interleave with audio-chunk-driven periodic frames to satisfy
        # it. None means the cursor has not been primed yet.
        self._last_wire_emit_ts_us: int | None = None
        # Beat availability for the current source. UNAVAILABLE drops
        # `beat` from the negotiated set and discards pending beats.
        self._beat_availability: BeatAvailability = BeatAvailability.PENDING
        # Near-playhead cap: while the client wants beats, periodic frames
        # beyond `_WARMUP_LEAD_US` are parked here instead of sent, and released
        # by `_release_timer` as the playhead advances. This keeps the wire
        # cursor close to the playhead so a schedule pushed mid-stream (a
        # flow-mode track change) is not dropped behind it. `UNAVAILABLE` lifts
        # the cap (no beats coming, so full send-ahead is fine).
        self._holdback_active: bool = False
        self._pending_frames: deque[tuple[int, bytes, BinaryMessageType, int, int]] = deque()
        self._release_timer: asyncio.TimerHandle | None = None

    @property
    def role_id(self) -> str:
        """Versioned role identifier."""
        return "visualizer@v1"

    @property
    def role_family(self) -> str:
        """Role family name for protocol messages."""
        return "visualizer"

    @property
    def wants_beats(self) -> bool:
        """True if the client negotiated `beat` and beats are not unavailable.

        Reads the client's requested types (`_request`), not the exposed
        `stream/start` types, so it is True from the moment the client
        asks for beats — before the first schedule activates the type on
        the wire.
        """
        return (
            self._request is not None
            and "beat" in self._request.types
            and self._beat_availability is not BeatAvailability.UNAVAILABLE
        )

    def get_stream_requirements(self) -> StreamRequirements:
        """Visualizer role sends binary streams."""
        return StreamRequirements()

    def get_audio_requirements(self) -> AudioRequirements:
        """Return audio requirements for visualizer analysis."""
        return AudioRequirements(
            sample_rate=48_000,
            bit_depth=16,
            channels=2,
            frame_duration_us=25_000,
        )

    def replay_from_pcm_cache(self) -> bool:
        """Replay buffered PCM on late join (visualizer is analysis-only)."""
        return True

    def get_binary_handling(self, message_type: int) -> BinaryHandling | None:
        """Return handling policy for visualizer binary frames."""
        for member in (
            BinaryMessageType.VISUALIZATION_LOUDNESS,
            BinaryMessageType.VISUALIZATION_BEAT,
            BinaryMessageType.VISUALIZATION_F_PEAK,
            BinaryMessageType.VISUALIZATION_SPECTRUM,
            BinaryMessageType.VISUALIZATION_PEAK,
            BinaryMessageType.VISUALIZATION_PITCH,
        ):
            if message_type == member.value:
                return BinaryHandling(drop_late=True, grace_period_us=2_000_000, buffer_track=True)
        return None

    def get_buffer_tracker(self) -> BufferTracker | None:
        """Return the visualizer buffer tracker."""
        return self._buffer_tracker

    def set_tracks_downbeats(self, *, tracks: bool) -> None:
        """Mark whether the upstream beat detector identifies bar starts."""
        self._tracks_downbeats = bool(tracks)

    def set_beat_availability(self, availability: BeatAvailability) -> None:
        """Declare whether beats will arrive for the current source.

        `beat` rides in `stream/start.types` whenever the client wants
        beats AND a schedule has already landed. Switching to
        UNAVAILABLE drops any pending schedule and re-issues
        `stream/start` so the client falls back to the FFT-driven types.
        """
        if self._beat_availability is availability:
            return
        previous_beat_in_types = self._beat_in_negotiated_types()
        self._beat_availability = availability
        if availability is BeatAvailability.UNAVAILABLE:
            self._pending_beats.clear()
            self._has_beats_landed = False
            # No beats will arrive — lift the cap and release held frames.
            self._end_holdback()
        elif self._holdback_should_be_active() and not self._holdback_active:
            # Beats are wanted again (e.g. UNAVAILABLE → PENDING on a new track)
            # but the cap was lifted earlier. Re-arm it so a schedule landing
            # after a few seconds of audio is not dropped behind the cursor.
            self._rearm_warmup_holdback()
        if (
            self._request is None
            or "beat" not in self._request.types
            or self._stream_config is None
        ):
            return
        if previous_beat_in_types != self._beat_in_negotiated_types():
            self._reissue_stream_start()

    def _ensure_buffer_tracker(self) -> None:
        """Create or update the buffer tracker from negotiated config.

        Capacity changes update the limit but never reset the buffered
        byte count — the client still holds previously sent bytes and a
        reset would under-count, disabling backpressure until the real
        client buffer overflowed.
        """
        capacity = max(1, self._buffer_capacity)
        if self._buffer_tracker is None:
            self._buffer_tracker = BufferTracker(
                clock=self._client._server.clock,  # noqa: SLF001
                client_id=self._client.client_id,
                capacity_bytes=capacity,
            )
        else:
            self._buffer_tracker.capacity_bytes = capacity

    def requires_initial_state(self) -> bool:
        """Visualizer receives server binary, gated on the client's initial state."""
        return True

    def requires_activation_state(self) -> bool:
        """Visualizer waits for its client/state request unless the hello configured it."""
        support = self._client.info.visualizer_support
        # DEPRECATED(spec-pr-195): remove in aiosendspin <version>
        return support is None or not support.has_stream_config

    def on_connect(self) -> None:
        """Load the hello's visualizer support and subscribe to group role."""
        support = self._client.info.visualizer_support
        if support is None:
            raise ValueError("visualizer support object missing for visualizer@v1 role")
        self._buffer_capacity = support.buffer_capacity
        # A current client's request arrives in client/state; no stream starts before it.
        legacy_request = self._legacy_hello_request(support)
        self._request = None if legacy_request is None else self._filter_request(legacy_request)
        self._stream_config = None
        self._subscribe_to_group_role()

    def on_deactivate(self) -> None:
        """End the visualizer stream when the role is deactivated while still connected."""
        if self._stream_started:
            self.on_stream_end()
        super().on_deactivate()

    def on_disconnect(self) -> None:
        """Unsubscribe from VisualizerGroupRole and reset state."""
        self._unsubscribe_from_group_role()
        self._stream_started = False
        self._extractor = None
        self._pending_beats.clear()
        self._has_beats_landed = False
        self._last_wire_emit_ts_us = None
        self._beat_availability = BeatAvailability.PENDING
        self._cancel_release_timer()
        self._pending_frames.clear()
        self._holdback_active = False
        self.reset_binary_timing()

    def on_stream_start(self) -> None:
        """Start extractor state and emit `stream/start` on a fresh stream.

        No-op until the client's requested configuration is known.
        """
        if self._request is None:
            return
        # Rebuild the config so any beats that landed before this
        # `on_stream_start` (mid-stream join replay) are reflected.
        self._stream_config = self._build_stream_config()
        self.reset_binary_timing()
        # Advance the wire-ts cursor to the current playhead so a stale
        # replayed beat can't poison the cursor backward into the past. It only
        # moves forward: a restart within an announced stream must not let
        # timestamps decrease.
        self._reserve_wire_ts(self._client._server.clock.now_us())  # noqa: SLF001
        # Arm the warmup holdback if beats are wanted but none have landed yet.
        self._cancel_release_timer()
        self._pending_frames.clear()
        self._holdback_active = self._holdback_should_be_active()
        # `stream/start` is just a config update during an active stream
        # (`stream/clear` keeps the stream alive). Skip the resend when
        # we have already announced a config; mid-stream config changes
        # go through `_reissue_stream_start` which bypasses this guard.
        if not self._stream_started:
            self._send_stream_start()
        self._rebuild_extractor()
        self._stream_started = True
        self._ensure_buffer_tracker()

    def _rebuild_extractor(self) -> None:
        """Create the FFT extractor when at least one FFT-driven type is negotiated.

        Beat-only configurations have no use for the extractor; leaving
        it None lets `on_audio_chunk` early-return after the beat drain.
        """
        if self._stream_config is None:
            self._extractor = None
            return
        if not any(t in self._stream_config.types for t in _FFT_DRIVEN_TYPES):
            self._extractor = None
            return
        req = self.get_audio_requirements()
        self._extractor = VisualizerFeatureExtractor(
            sample_rate=req.sample_rate,
            channels=req.channels,
            config=self._stream_config,
        )

    def on_audio_chunk(self, chunk: AudioChunk) -> None:
        """Extract per-chunk features and emit one binary per enabled type.

        Beats are drained up to each emit ts so the visualizer wire stays
        in non-decreasing ts order. Audio chunks are delivered regardless
        of which types are negotiated, so this method is also the drain
        driver for beat-only configurations.
        """
        if not self.has_connection() or self._stream_config is None:
            return
        # Drain pending beats that fall at or before this chunk's start
        # so they precede the chunk's periodic frames. Frames inside the
        # chunk drain further as they emit.
        self._drain_beats_up_to(chunk.timestamp_us)
        if self._extractor is None:
            return
        end_time_us = chunk.timestamp_us + chunk.duration_us
        for frame in self._extractor.process_chunk(chunk.data, chunk.timestamp_us):
            self._drain_beats_up_to(frame.timestamp_us)
            self._emit_frame(frame, end_time_us=end_time_us, duration_us=chunk.duration_us)

    def _emit_frame(
        self,
        frame: ExtractedFrame,
        *,
        end_time_us: int,
        duration_us: int,
    ) -> None:
        """Pack and send all configured binaries for a single extractor frame."""
        if self._stream_config is None:
            return
        types = self._stream_config.types
        ts = frame.timestamp_us
        if "loudness" in types and frame.loudness is not None:
            payload = struct.pack(">H", int(np.clip(frame.loudness, 0, 65535)))
            self._dispatch_frame(
                pack_visualizer_frame(BinaryMessageType.VISUALIZATION_LOUDNESS, ts, payload),
                ts_us=ts,
                msg_type=BinaryMessageType.VISUALIZATION_LOUDNESS,
                end_time_us=end_time_us,
                duration_us=duration_us,
            )
        if "f_peak" in types and frame.f_peak_freq is not None and frame.f_peak_amp is not None:
            # Wire invariant: `freq == 0 implies amp == 0` so a misbehaving
            # extractor cannot emit "no peak with non-zero amp".
            freq = int(np.clip(frame.f_peak_freq, 0, 65535))
            amp = int(np.clip(frame.f_peak_amp, 0, 65535)) if freq != 0 else 0
            payload = struct.pack(">HH", freq, amp)
            self._dispatch_frame(
                pack_visualizer_frame(BinaryMessageType.VISUALIZATION_F_PEAK, ts, payload),
                ts_us=ts,
                msg_type=BinaryMessageType.VISUALIZATION_F_PEAK,
                end_time_us=end_time_us,
                duration_us=duration_us,
            )
        if "spectrum" in types and frame.spectrum is not None:
            payload = frame.spectrum.astype(">u2", copy=False).tobytes()
            self._dispatch_frame(
                pack_visualizer_frame(BinaryMessageType.VISUALIZATION_SPECTRUM, ts, payload),
                ts_us=ts,
                msg_type=BinaryMessageType.VISUALIZATION_SPECTRUM,
                end_time_us=end_time_us,
                duration_us=duration_us,
            )
        if "peak" in types and frame.peak is not None:
            payload = bytes((int(np.clip(frame.peak, 0, 255)),))
            self._dispatch_frame(
                pack_visualizer_frame(BinaryMessageType.VISUALIZATION_PEAK, ts, payload),
                ts_us=ts,
                msg_type=BinaryMessageType.VISUALIZATION_PEAK,
                end_time_us=end_time_us,
                duration_us=duration_us,
            )
        if (
            "pitch" in types
            and frame.pitch_midi_q88 is not None
            and frame.pitch_confidence is not None
        ):
            midi = int(np.clip(frame.pitch_midi_q88, 0, 65535))
            confidence = int(np.clip(frame.pitch_confidence, 0, 255))
            payload = struct.pack(">H", midi) + bytes((confidence,))
            self._dispatch_frame(
                pack_visualizer_frame(BinaryMessageType.VISUALIZATION_PITCH, ts, payload),
                ts_us=ts,
                msg_type=BinaryMessageType.VISUALIZATION_PITCH,
                end_time_us=end_time_us,
                duration_us=duration_us,
            )

    def _dispatch_frame(
        self,
        message: bytes,
        *,
        ts_us: int,
        msg_type: BinaryMessageType,
        end_time_us: int,
        duration_us: int,
    ) -> None:
        """Send a periodic frame now, or park it while the warmup cap is active."""
        if self._holdback_active and ts_us > self._warmup_cutoff_us():
            self._pending_frames.append((ts_us, message, msg_type, end_time_us, duration_us))
            self._arm_release_timer()
            return
        self._send_frame_now(ts_us, message, msg_type, end_time_us, duration_us)

    def _send_frame_now(
        self,
        ts_us: int,
        message: bytes,
        msg_type: BinaryMessageType,
        end_time_us: int,
        duration_us: int,
    ) -> None:
        """Reserve the wire ts and enqueue a periodic binary frame for sending."""
        last = self._last_wire_emit_ts_us
        # Strict `<`: the periodic types of one extractor frame share a timestamp.
        if last is not None and ts_us < last:
            return
        self._reserve_wire_ts(ts_us)
        self._client.send_binary(
            message,
            role_family=self.role_family,
            timestamp_us=ts_us,
            message_type=msg_type.value,
            buffer_end_time_us=end_time_us,
            buffer_byte_count=len(message),
            duration_us=duration_us,
        )

    def _holdback_should_be_active(self) -> bool:
        """Whether the near-playhead cap applies: the client wants beats.

        Active the whole time beats are wanted, not just before the first
        schedule. Keeping the wire cursor within `_WARMUP_LEAD_US` of the
        playhead lets a schedule pushed mid-stream — e.g. a flow-mode track
        change — land at the cursor instead of behind an already-advanced one.
        """
        return self.wants_beats

    def _rearm_warmup_holdback(self) -> None:
        """Re-arm the near-playhead cap (schedule cleared, or beats wanted again).

        Keeps already-parked periodic frames and the wire-ts cursor: a flow-mode
        track change keeps streaming the same continuous audio, so parked frames
        are still valid and must keep flowing, and a late beat landing below the
        cursor is dropped by the `<=` guard rather than emitted out of order.
        `on_stream_clear` (a seek) drops the parked frames and resets the cursor
        itself; a stream config change drops the parked frames and keeps the cursor.
        """
        self._holdback_active = self._holdback_should_be_active()
        self._arm_release_timer()

    def _warmup_cutoff_us(self) -> int:
        """Wire ts above which periodic frames are held during warmup."""
        return self._client._server.clock.now_us() + _WARMUP_LEAD_US  # noqa: SLF001

    def _cancel_release_timer(self) -> None:
        """Cancel the pending held-frame release timer, if any."""
        if self._release_timer is not None:
            self._release_timer.cancel()
            self._release_timer = None

    def _arm_release_timer(self) -> None:
        """Schedule the next held-frame release at the warmup lead."""
        if self._release_timer is not None or not self._pending_frames or not self.has_connection():
            return
        now_us = self._client._server.clock.now_us()  # noqa: SLF001
        head_ts = self._pending_frames[0][0]
        delay_s = max(0, head_ts - _WARMUP_LEAD_US - now_us) / 1_000_000
        loop = asyncio.get_running_loop()
        self._release_timer = loop.call_later(delay_s, self._run_release_scheduler)

    def _run_release_scheduler(self) -> None:
        """Release held periodic frames within the cap, interleaving due beats."""
        self._release_timer = None
        if not self.has_connection():
            return
        cutoff_us = self._warmup_cutoff_us()
        while self._pending_frames and self._pending_frames[0][0] <= cutoff_us:
            ts_us, message, msg_type, end_time_us, duration_us = self._pending_frames.popleft()
            self._drain_beats_up_to(ts_us)
            self._send_frame_now(ts_us, message, msg_type, end_time_us, duration_us)
        # Beats past the last released frame but still within the cap.
        self._drain_beats_up_to(cutoff_us)
        self._arm_release_timer()

    def _end_holdback(self) -> None:
        """Lift the warmup cap and flush held frames, interleaving due beats.

        Held periodic frames are released in ts order; pending beats at or
        below each frame's ts emit first so the wire stays non-decreasing.
        Beats beyond the held frontier stay queued for `on_audio_chunk` to
        drain. After this, periodic frames send immediately.
        """
        self._holdback_active = False
        self._cancel_release_timer()
        while self._pending_frames:
            frame_ts = self._pending_frames[0][0]
            self._drain_beats_up_to(frame_ts)
            ts_us, message, msg_type, end_time_us, duration_us = self._pending_frames.popleft()
            self._send_frame_now(ts_us, message, msg_type, end_time_us, duration_us)

    def _reserve_wire_ts(self, ts_us: int) -> None:
        """Advance the wire-ts cursor; callers must not regress below it."""
        last = self._last_wire_emit_ts_us
        self._last_wire_emit_ts_us = max(ts_us, last) if last is not None else ts_us

    def append_beats(self, beats: list[BeatTiming]) -> None:
        """Append beat timings for delivery interleaved with audio chunks.

        Beats land here from `VisualizerGroupRole.append_beat_schedule`
        (server-fed offline analysis). They drain on the next
        `on_audio_chunk` whose timestamp matches or exceeds each beat's
        ts.

        No-op while `BeatAvailability.UNAVAILABLE` or when the client
        did not request `beat`. The first non-empty delivery re-emits
        `stream/start` so the client sees `beat` added to the negotiated
        types.
        """
        if self._request is None or "beat" not in self._request.types:
            return
        if self._beat_availability is BeatAvailability.UNAVAILABLE:
            return
        if not beats:
            return
        first_landing = not self._has_beats_landed
        self._pending_beats.extend(beats)
        self._has_beats_landed = True
        if first_landing and self._stream_started and self._stream_config is not None:
            # Beat is now legitimately part of the negotiated types — tell
            # the client. Subsequent audio chunks (and the release timer) drain
            # the queue; the near-playhead cap stays active so beats are not
            # lost behind a cursor pushed ahead by frames.
            self._reissue_stream_start()

    def _reissue_stream_start(self) -> None:
        """Rebuild stream config from current state and re-send `stream/start`."""
        if self._request is None:
            return
        self._stream_config = self._build_stream_config()
        self._send_stream_start()

    def clear_beats(self) -> None:
        """Drop any pending beat schedule (`stream/clear` carries this on the wire)."""
        self._pending_beats.clear()
        if self._has_beats_landed:
            self._has_beats_landed = False
            # A landed schedule was dropped while the stream continues (track
            # change, analysis re-clear). Re-arm warmup so the next schedule is
            # not lost behind a wire cursor already pushed far ahead.
            self._rearm_warmup_holdback()
            if self._stream_started and self._stream_config is not None:
                self._reissue_stream_start()

    def _drain_beats_up_to(self, max_ts_us: int) -> None:
        """Emit any pending beats whose ts is <= `max_ts_us`.

        While the near-playhead cap is active, beats drain only up to the cap
        cutoff, so a far-ahead audio chunk cannot push the cursor past beats a
        later mid-stream schedule will need to sit at. Outside an announced
        stream, beats stay queued.
        """
        if (
            not self._stream_started
            or self._stream_config is None
            or "beat" not in self._stream_config.types
        ):
            return
        if self._holdback_active:
            max_ts_us = min(max_ts_us, self._warmup_cutoff_us())
        due: list[BeatTiming] = []
        while self._pending_beats and self._pending_beats[0].timestamp_us <= max_ts_us:
            due.append(self._pending_beats.popleft())
        if due:
            self._emit_beats(due)

    def _emit_beats(self, beats: list[BeatTiming]) -> None:
        """Emit each beat as its own msg 17 binary.

        Drops any beat whose ts would regress (or duplicate) the wire
        cursor — the wire must stay strictly non-decreasing.
        """
        if self._stream_config is None or not beats:
            return
        for beat in beats:
            last = self._last_wire_emit_ts_us
            # `<=` drops duplicates as well as strict regressions — the
            # group enforces strict monotonicity within a single schedule
            # but a resubscribe replay can deliver a beat whose ts equals
            # the most-recently-emitted one.
            if last is not None and beat.timestamp_us <= last:
                continue
            self._reserve_wire_ts(beat.timestamp_us)
            # The downbeat bit is only meaningful when the role tracks downbeats.
            flags = FLAG_DOWNBEAT if beat.is_downbeat and self._tracks_downbeats else 0
            message = pack_visualizer_frame(
                BinaryMessageType.VISUALIZATION_BEAT, beat.timestamp_us, bytes((flags,))
            )
            self._client.send_binary(
                message,
                role_family=self.role_family,
                timestamp_us=beat.timestamp_us,
                message_type=BinaryMessageType.VISUALIZATION_BEAT.value,
                buffer_end_time_us=beat.timestamp_us,
                buffer_byte_count=len(message),
                duration_us=0,
            )

    def on_stream_clear(self) -> None:
        """Reset extractor state, drop pending beats, notify client.

        `stream/clear` is sent only for a stream this role announced.
        """
        if self._extractor is not None:
            self._extractor.reset()
        self._pending_beats.clear()
        had_beats = self._has_beats_landed
        self._has_beats_landed = False
        # Seek re-pushes the schedule, so beats arrive again shortly: re-arm
        # warmup. The accompanying `stream/clear` makes the client discard
        # ahead binaries, so the wire-ts guard can be dropped too — post-seek
        # frames with earlier timestamps are then not silently blocked. Parked
        # frames are for the pre-seek position, so drop them here.
        self._cancel_release_timer()
        self._pending_frames.clear()
        self._rearm_warmup_holdback()
        self._last_wire_emit_ts_us = None
        if self._stream_started:
            self.send_message(StreamClearMessage(payload=StreamClearPayload(roles=["visualizer"])))
        self.reset_binary_timing()
        if self._buffer_tracker is not None:
            self._buffer_tracker.reset()
        if had_beats and self._stream_started and self._stream_config is not None:
            # `beat` is no longer in the negotiated types until a fresh
            # schedule arrives; tell the client.
            self._reissue_stream_start()

    def on_stream_end(self) -> None:
        """End the visualizer stream and reset state.

        `stream/end` is sent only for a stream this role announced.
        """
        announced = self._stream_started
        self._extractor = None
        self._stream_started = False
        self._pending_beats.clear()
        self._has_beats_landed = False
        self._last_wire_emit_ts_us = None
        self._cancel_release_timer()
        self._pending_frames.clear()
        self._holdback_active = False
        if announced:
            self.send_message(StreamEndMessage(payload=StreamEndPayload(roles=["visualizer"])))
        self.reset_binary_timing()
        if self._buffer_tracker is not None:
            self._buffer_tracker.reset()

    def initial_state_deviations(self, payload: ClientStatePayload) -> list[str]:
        """Report a missing visualizer object in the initial client/state."""
        support = self._client.info.visualizer_support
        # DEPRECATED(spec-pr-195): remove in aiosendspin <version>
        # A pre-#195 client configures its stream in the hello, which is flagged already.
        if support is not None and support.has_stream_config:
            return []
        if payload.visualizer is None:
            return ["has an active visualizer role but no visualizer state"]
        return []

    def client_state_deviations(self, payload: ClientStatePayload) -> list[str]:
        """Report visualizer fields in a client/state that violate the spec."""
        state = payload.visualizer
        if state is None:
            return []
        reasons: list[str] = []
        if "spectrum" in state.types and state.spectrum is None:
            reasons.append("requested visualizer 'spectrum' without a spectrum configuration")
        if state.rate_max <= 0:
            reasons.append(f"sent a non-positive visualizer rate_max: {state.rate_max}")
        return reasons

    def on_client_state(self, payload: ClientStatePayload) -> None:
        """Apply the visualizer stream configuration from client/state."""
        state = payload.visualizer
        # A non-positive rate_max was flagged by client_state_deviations; keep the prior request.
        if state is None or state.rate_max <= 0:
            return
        self._apply_request(self._filter_request(state))

    def on_initial_client_state(self, payload: ClientStatePayload) -> None:
        """Store the requested configuration so the stream join announces it from the start."""
        self.on_client_state(payload)

    # DEPRECATED(spec-pr-195): remove in aiosendspin <version>
    def on_stream_request_format(self, payload: StreamRequestFormatPayload) -> None:
        """Merge a pre-#195 visualizer request into the requested configuration."""
        request = payload.visualizer
        if request is None:
            return

        if request.buffer_capacity is not None:
            # buffer_capacity is not a visualizer stream/request-format field per spec.
            self._client.flag_noncompliance(
                "stream/request-format set buffer_capacity, not a visualizer field here"
            )

        invalid = [
            name
            for name, value in (
                ("rate_max", request.rate_max),
                ("buffer_capacity", request.buffer_capacity),
            )
            if value is not None and value <= 0
        ]
        if invalid:
            self._client.flag_noncompliance(
                "stream/request-format visualizer fields must be positive: " + ", ".join(invalid)
            )
            return

        if self._request is None:
            return

        if request.buffer_capacity is not None:
            self._buffer_capacity = request.buffer_capacity
            if self._stream_started:
                self._ensure_buffer_tracker()

        merged = VisualizerStatePayload(
            types=list(request.types) if request.types is not None else self._request.types,
            rate_max=request.rate_max or self._request.rate_max,
            spectrum=request.spectrum or self._request.spectrum,
        )
        self._apply_request(self._filter_request(merged))

    def _apply_request(self, request: VisualizerStatePayload) -> None:
        """Store the client's requested configuration and apply it.

        An active stream gets a new `stream/start` only when the derived config
        changed; with no active stream the request applies to the next one. The
        first request known on this connection joins a stream the group is
        already running.
        """
        if request == self._request:
            return
        first_request = self._request is None
        beats_were_wanted = self.wants_beats
        self._request = request

        if self._stream_started:
            self._apply_request_to_stream()
        if self.wants_beats and not beats_were_wanted:
            # The group role replays its beat schedule on join only to members
            # that want beats, so rejoin it now that this role does.
            self._unsubscribe_from_group_role()
            self._subscribe_to_group_role()
        if first_request and not self._stream_started:
            self._client.join_active_stream(self)

    def _apply_request_to_stream(self) -> None:
        """Re-derive the active stream's config; send `stream/start` only when it changed."""
        if self._build_stream_config() == self._stream_config:
            self._sync_holdback()
            return

        # Pending beats are pinned to the old config (e.g. stale rate);
        # drop them and re-arm `_has_beats_landed` so the new config
        # waits for fresh beats before re-advertising `beat`.
        self._pending_beats.clear()
        self._has_beats_landed = False

        # Rebuilt after the reset, which can drop `beat` from the config.
        self._stream_config = self._build_stream_config()
        # Held frames carry old-config payloads (e.g. stale spectrum bins);
        # drop them and re-evaluate the warmup cap against the new request.
        self._cancel_release_timer()
        self._pending_frames.clear()
        self._holdback_active = self._holdback_should_be_active()
        # rate_max / types change rebuilds the extractor (new hop). The wire-ts
        # cursor is kept: the stream continues, so timestamps must not decrease.
        # The new extractor's first frame lands at the next chunk's end.
        self._rebuild_extractor()
        self._ensure_buffer_tracker()
        self._send_stream_start()

    def _sync_holdback(self) -> None:
        """Arm or lift the near-playhead cap to match whether beats are wanted."""
        if self._holdback_should_be_active():
            if not self._holdback_active:
                self._rearm_warmup_holdback()
        elif self._holdback_active:
            self._end_holdback()

    def _filter_request(self, request: VisualizerStatePayload) -> VisualizerStatePayload:
        """Reduce a requested configuration to the types this role can stream."""
        types = [t for t in request.types if t in _IMPLEMENTED_TYPES]
        if "spectrum" in types and request.spectrum is None:
            # Flagged as a deviation; a lenient server streams the other types.
            types.remove("spectrum")
        # `pitch` rides spec-reserved binary type 21, so a compliance-strict server
        # never streams it.
        if not self._client._server.allow_noncompliant_clients:  # noqa: SLF001
            types = [t for t in types if t != "pitch"]
        return replace(request, types=types)

    # DEPRECATED(spec-pr-195): remove in aiosendspin <version>
    @staticmethod
    def _legacy_hello_request(
        support: ClientHelloVisualizerSupport,
    ) -> VisualizerStatePayload | None:
        """Return the stream configuration a pre-#195 hello carried, or None without one."""
        if not support.has_stream_config:
            return None
        return VisualizerStatePayload(
            types=["loudness", "f_peak"] if support.types is None else list(support.types),
            rate_max=30 if support.rate_max is None else support.rate_max,
            spectrum=support.spectrum,
        )

    def _beat_in_negotiated_types(self) -> bool:
        """Whether `beat` is currently exposed in `stream/start.types`."""
        return (
            self._request is not None
            and "beat" in self._request.types
            and self._has_beats_landed
            and self._beat_availability is not BeatAvailability.UNAVAILABLE
        )

    def _build_stream_config(self) -> StreamStartVisualizer:
        """Derive the current `stream/start` config from the request + beat state.

        `beat` is deferred: it is exposed only once a non-empty beat
        schedule has actually landed for the current stream (and the
        client requested it, and availability is not UNAVAILABLE).
        Until then the client sees only the FFT-driven types and can
        render a `peak`-based fallback without flicker.

        Exception: beat-only clients (`types == ["beat"]`) get `beat`
        from the start — there is no FFT-driven type to fall back to.
        """
        if self._request is None:
            raise ValueError("request must be known before building stream config")
        client_types = list(self._request.types)
        beat_only = client_types == ["beat"]
        if beat_only or self._beat_in_negotiated_types():
            exposed_types = client_types
        else:
            exposed_types = [t for t in client_types if t != "beat"]
        # Server-wide pitch shed: drop the (heavy) pitch feature unless it is
        # the only exposed type, so a pitch-only client still gets its data.
        # A compliance-strict server never gets here with pitch (`_filter_request`).
        if not self._client._server.visualizer_pitch_enabled:  # noqa: SLF001
            without_pitch: list[SupportedVisualizerType] = [
                t for t in exposed_types if t != "pitch"
            ]
            if without_pitch:
                exposed_types = without_pitch
        return StreamStartVisualizer.from_request(
            replace(self._request, types=exposed_types), tracks_downbeats=self._tracks_downbeats
        )

    def refresh_pitch_setting(self) -> None:
        """Re-apply the server-wide pitch toggle to the live stream config.

        Called by the server when `set_visualizer_pitch_enabled` flips. Rebuilds
        the config and, if the exposed types changed, rebuilds the extractor and
        re-emits `stream/start` so the client sees the new set.
        """
        if self._request is None or self._stream_config is None:
            return
        new_config = self._build_stream_config()
        if new_config.types == self._stream_config.types:
            return
        self._stream_config = new_config
        # Types changed (pitch added/removed). The wire-ts cursor is kept so an
        # in-stream `stream/start` never lets timestamps decrease.
        self._rebuild_extractor()
        if self._stream_started:
            self._send_stream_start()

    def _send_stream_start(self) -> None:
        """Send `stream/start` with the negotiated visualizer configuration."""
        if self._stream_config is None:
            return
        message = StreamStartMessage(payload=StreamStartPayload(visualizer=self._stream_config))
        self.send_message(message)
        self._stream_started = True
