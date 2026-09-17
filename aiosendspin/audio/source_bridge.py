"""Bridge captured audio into a steady pull-based stream."""

from __future__ import annotations

import importlib
import logging
import types
from typing import TYPE_CHECKING, cast

from aiosendspin.audio.format import (
    AudioFormat,
    _convert_s24_to_s32,
    _convert_s32_to_s24,
    _get_av,
    _get_numpy,
)

if TYPE_CHECKING:
    import av

logger = logging.getLogger(__name__)

# Timestamp lead over the sample-derived position beyond which a capture gap is filled.
_GAP_THRESHOLD_US = 15_000

# Persistent occupancy deviation from target before the simple tier drops or inserts.
_WATERMARK_US = 50_000

# Fed-audio duration over which occupancy extremes are observed before correcting.
_CORRECTION_WINDOW_US = 10_000_000

# Convergence threshold for persistent sub-watermark surplus.
_CONVERGENCE_EPSILON_US = 15_000
_CONVERGENCE_WINDOWS = 3

# Cap implausible clock-rate corrections.
_MAX_RATIO_ADJUST = 0.05

# Scales occupancy error to the ratio trim that regulates buffer level.
_RATIO_TIME_CONSTANT_US = 50_000_000

# Smoothing and per-window step bound for the measured supply/consumption rate.
_RATE_EST_ALPHA = 0.5
_RATE_EST_MAX_STEP = 0.01


def _require_av() -> types.ModuleType:
    """Return the av module or raise a friendly error if the extra is missing."""
    try:
        return _get_av()
    except ImportError as err:
        raise ImportError(
            "PyAV is required for SourceBridge. "
            "Install the 'source' extra: pip install aiosendspin[source]"
        ) from err


def _require_soxr() -> types.ModuleType:
    """Return the soxr module or raise a friendly error if the extra is missing."""
    try:
        return importlib.import_module("soxr")
    except ImportError as err:
        raise ImportError(
            "python-soxr is required for AsrcSourceBridge. "
            "Install the 'asrc' extra: pip install aiosendspin[asrc]"
        ) from err


class SourceBridge:
    """Buffer timestamped source PCM for a steady-rate consumer.

    Feed captured chunks with ``feed()``, pull exact frame counts with ``read()``,
    and call ``flush()`` once after input ends.
    """

    def __init__(
        self,
        *,
        input_format: AudioFormat,
        output_format: AudioFormat,
        target_latency_ms: int = 1000,
        max_latency_ms: int | None = None,
    ) -> None:
        """Create a bounded-latency source bridge.

        Args:
            input_format: PCM format passed to ``feed()``.
            output_format: PCM format returned by ``read()``.
            target_latency_ms: Buffered duration required before output begins.
            max_latency_ms: Maximum buffered duration before oldest audio is dropped.
        """
        if max_latency_ms is None:
            max_latency_ms = 2 * target_latency_ms
        if max_latency_ms <= target_latency_ms:
            raise ValueError("max_latency_ms must exceed target_latency_ms")
        self._in = input_format
        self._out = output_format
        in_wire, self._in_av_format, self._in_layout, in_av_bps = input_format.resolve_av_format()
        out_wire, _, _, out_av_bps = output_format.resolve_av_format()
        self._in_stride = in_wire * input_format.channels
        self._in_av_stride = in_av_bps * input_format.channels
        self._out_stride = out_wire * output_format.channels
        self._out_av_stride = out_av_bps * output_format.channels
        self._target_us = target_latency_ms * 1000
        self._max_us = max_latency_ms * 1000
        self._max_ratio_adjust = _MAX_RATIO_ADJUST
        self._buffer = bytearray()
        self._primed = False
        self._anchor_us: int | None = None
        self._frames_since_anchor = 0
        self._timestamp_skew_us = 0
        self._surplus_windows = 0
        self._window_fed_us = 0
        self._window_min_us: int | None = None
        self._window_max_us: int | None = None
        self._window_supplied_us = 0
        self._window_consumed_us = 0
        self._window_underruns = 0
        self._window_overflow_us = 0
        self._resampler_rate = output_format.sample_rate
        self._resampler = self._build_resampler(rate=self._resampler_rate)
        if input_format.channels != output_format.channels:
            # Remixing needs a channel layout, which PyAV has only for some counts.
            try:
                self._convert_via(
                    self._build_resampler(rate=self._resampler_rate), bytes(self._in_stride)
                )
            except ValueError as err:
                msg = (
                    f"Cannot remix {input_format.channels} channels "
                    f"to {output_format.channels} channels"
                )
                raise ValueError(msg) from err

    @property
    def occupancy_us(self) -> int:
        """Currently buffered audio in microseconds."""
        frames = len(self._buffer) // self._out_stride
        return frames * 1_000_000 // self._out.sample_rate

    def feed(self, pcm: bytes, capture_timestamp_us: int) -> None:
        """Add PCM whose first sample has the given capture timestamp."""
        if not pcm:
            return
        if len(pcm) % self._in_stride:
            raise ValueError("pcm length must be a whole number of frames")
        frames = len(pcm) // self._in_stride
        duration_us = frames * 1_000_000 // self._in.sample_rate
        if self._anchor_us is not None:
            self._track_timestamp(capture_timestamp_us, duration_us)
        # Also re-anchors on this chunk when tracking reset the bridge.
        if self._anchor_us is None:
            self._anchor_us = capture_timestamp_us
        self._frames_since_anchor += frames
        self._buffer += self._convert(pcm)
        overflow_us = self.occupancy_us - self._max_us
        if overflow_us > 0:
            if self._window_overflow_us == 0:
                logger.debug("Occupancy over max by %d us, dropping oldest audio", overflow_us)
            self._window_overflow_us += overflow_us
            del self._buffer[: self._us_to_bytes(overflow_us)]
        self._sample_occupancy()
        self._correct_drift()
        self._window_fed_us += duration_us
        self._window_supplied_us += duration_us
        if self._window_fed_us >= _CORRECTION_WINDOW_US:
            self._apply_window_correction()
            self._log_drift_heartbeat()
            self._window_fed_us = 0
            self._window_min_us = None
            self._window_max_us = None
            self._window_supplied_us = 0
            self._window_consumed_us = 0
            self._window_underruns = 0
            self._window_overflow_us = 0

    def read(self, frames: int) -> bytes:
        """Return one consumer pull of exactly ``frames``, padding with silence."""
        wanted = frames * self._out_stride
        if not self._primed:
            if self.occupancy_us < self._target_us:
                return bytes(wanted)
            self._primed = True
            # Rate measurement starts with playback, so the priming fill is not counted as surplus.
            self._window_supplied_us = 0
            self._window_consumed_us = 0
            # Trim inaudible startup surplus before playback begins.
            surplus_us = self.occupancy_us - self._target_us
            if surplus_us > 0:
                del self._buffer[: self._us_to_bytes(surplus_us)]
            logger.debug(
                "Bridge primed, trimmed %d us startup surplus to start at target", surplus_us
            )
        # Count every consumer pull, including silence padding.
        self._window_consumed_us += frames * 1_000_000 // self._out.sample_rate
        take = min(wanted, len(self._buffer))
        out = bytes(self._buffer[:take])
        del self._buffer[:take]
        if take < wanted:
            out += bytes(wanted - take)
            self._window_underruns += 1
            if self._window_underruns == 1:
                logger.debug("Underrun: buffer emptied, padding output with silence")
        self._sample_occupancy()
        return out

    def _track_timestamp(self, capture_timestamp_us: int, duration_us: int) -> None:
        """Compare a timestamp against the sample-derived position, filling gaps or resetting."""
        assert self._anchor_us is not None
        expected_us = (
            self._anchor_us + self._frames_since_anchor * 1_000_000 // self._in.sample_rate
        )
        # Offset from the position once the source clock's slow skew is taken out.
        offset_us = capture_timestamp_us - expected_us - self._timestamp_skew_us
        if abs(offset_us) > self._max_us:
            logger.warning("Source timestamp jumped %d us, resetting bridge", offset_us)
            self._reset()
        elif offset_us > _GAP_THRESHOLD_US:
            logger.debug("Capture gap of %d us, inserting silence to hold latency", offset_us)
            self._append_silence(offset_us)
            self._frames_since_anchor += offset_us * self._in.sample_rate // 1_000_000
            self._window_supplied_us += offset_us
        else:
            # Follow timestamp clock skew at up to the plausible rate; jitter stays within its band.
            step_us = int(duration_us * self._max_ratio_adjust)
            self._timestamp_skew_us += max(-step_us, min(step_us, offset_us))

    def _correct_drift(self) -> None:
        """SourceBridge corrects once per window, while AsrcSourceBridge corrects per feed."""

    def _apply_window_correction(self) -> None:
        """Correct persistent occupancy deviation over a window."""
        if self._window_min_us is None or self._window_max_us is None:
            return
        surplus_us = self._window_min_us - self._target_us
        deficit_us = self._target_us - self._window_max_us
        if surplus_us > _WATERMARK_US:
            logger.debug(
                "Drift: dropping %d us surplus (occupancy held %d-%d us vs %d us target)",
                surplus_us,
                self._window_min_us,
                self._window_max_us,
                self._target_us,
            )
            del self._buffer[: self._us_to_bytes(surplus_us)]
            self._surplus_windows = 0
            return
        if self._primed and deficit_us > _WATERMARK_US:
            logger.debug(
                "Drift: inserting %d us silence for deficit (occupancy %d-%d us vs %d us target)",
                deficit_us,
                self._window_min_us,
                self._window_max_us,
                self._target_us,
            )
            self._append_silence(deficit_us)
            self._surplus_windows = 0
            return
        if surplus_us > _CONVERGENCE_EPSILON_US:
            self._surplus_windows += 1
            if self._surplus_windows >= _CONVERGENCE_WINDOWS:
                logger.debug(
                    "Drift: trimming %d us surplus that persisted for %d windows",
                    surplus_us,
                    self._surplus_windows,
                )
                del self._buffer[: self._us_to_bytes(surplus_us)]
                self._surplus_windows = 0
        else:
            self._surplus_windows = 0

    def _build_resampler(self, *, rate: int) -> av.AudioResampler:
        av_mod = _require_av()
        _, av_format, layout, _ = self._out.resolve_av_format()
        return cast(
            "av.AudioResampler", av_mod.AudioResampler(format=av_format, layout=layout, rate=rate)
        )

    def _convert(self, pcm: bytes) -> bytes:
        # Normalize source PCM to the consumer format before it enters the output buffer.
        raw = self._convert_via(self._resampler, pcm)
        return _convert_s32_to_s24(raw) if self._out.bit_depth == 24 else raw

    def _convert_via(self, resampler: av.AudioResampler, pcm: bytes) -> bytes:
        """Run input PCM through an av resampler, returning av-stride output bytes."""
        av_mod = _require_av()
        if self._in.bit_depth == 24:
            pcm = _convert_s24_to_s32(pcm)
        samples = len(pcm) // self._in_av_stride
        frame = av_mod.AudioFrame(
            format=self._in_av_format, layout=self._in_layout, samples=samples
        )
        frame.sample_rate = self._in.sample_rate
        frame.planes[0].update(pcm)
        out = bytearray()
        for resampled in resampler.resample(frame):
            out += bytes(resampled.planes[0])[: resampled.samples * self._out_av_stride]
        return bytes(out)

    def _us_to_bytes(self, duration_us: int) -> int:
        frames = duration_us * self._out.sample_rate // 1_000_000
        return frames * self._out_stride

    def _append_silence(self, duration_us: int) -> None:
        self._buffer += bytes(self._us_to_bytes(duration_us))

    def _sample_occupancy(self) -> None:
        occ = self.occupancy_us
        self._window_min_us = occ if self._window_min_us is None else min(self._window_min_us, occ)
        self._window_max_us = occ if self._window_max_us is None else max(self._window_max_us, occ)

    def _log_drift_heartbeat(self) -> None:
        """Emit a periodic occupancy summary so drift behavior can be evaluated."""
        if not logger.isEnabledFor(logging.DEBUG):
            return
        if self._window_min_us is None or self._window_max_us is None:
            return
        events = ""
        if self._window_underruns:
            events += f", {self._window_underruns} underruns"
        if self._window_overflow_us:
            events += f", dropped {self._window_overflow_us} us overflow"
        logger.debug(
            "Bridge occupancy over last window: %d-%d us (target %d us)%s%s",
            self._window_min_us,
            self._window_max_us,
            self._target_us,
            self._drift_detail(),
            events,
        )

    def _drift_detail(self) -> str:
        """Return continuous-correction heartbeat detail."""
        return ""

    def _drain_av_tail(self) -> bytes:
        """Return raw AV-stride bytes flushed from the PyAV resampler."""
        out = bytearray()
        for resampled in self._resampler.resample(None):
            out += bytes(resampled.planes[0])[: resampled.samples * self._out_av_stride]
        return bytes(out)

    def flush(self) -> None:
        """Append the resampler tail after the final input chunk."""
        raw = self._drain_av_tail()
        if not raw:
            return
        data = _convert_s32_to_s24(raw) if self._out.bit_depth == 24 else raw
        self._buffer += data
        overflow_us = self.occupancy_us - self._max_us
        if overflow_us > 0:
            del self._buffer[: self._us_to_bytes(overflow_us)]

    def _reset(self) -> None:
        self._buffer.clear()
        self._primed = False
        self._anchor_us = None
        self._frames_since_anchor = 0
        self._timestamp_skew_us = 0
        self._surplus_windows = 0
        self._window_fed_us = 0
        self._window_min_us = None
        self._window_max_us = None
        self._window_supplied_us = 0
        self._window_consumed_us = 0
        self._window_underruns = 0
        self._window_overflow_us = 0
        self._resampler = self._build_resampler(rate=self._resampler_rate)


class AsrcSourceBridge(SourceBridge):
    """Continuously correct source drift while preserving target latency."""

    def __init__(
        self,
        *,
        input_format: AudioFormat,
        output_format: AudioFormat,
        target_latency_ms: int = 1000,
        max_latency_ms: int | None = None,
        quality: str = "HQ",
        max_ratio_adjust: float = _MAX_RATIO_ADJUST,
    ) -> None:
        """Create a continuously corrected source bridge.

        Args:
            input_format: PCM format passed to ``feed()``.
            output_format: PCM format returned by ``read()``.
            target_latency_ms: Buffered duration required before output begins.
            max_latency_ms: Maximum buffered duration before oldest audio is dropped.
            quality: Resampler quality setting.
            max_ratio_adjust: Maximum fractional source-rate correction.
        """
        super().__init__(
            input_format=input_format,
            output_format=output_format,
            target_latency_ms=target_latency_ms,
            max_latency_ms=max_latency_ms,
        )
        if max_ratio_adjust <= 0:
            raise ValueError("max_ratio_adjust must be positive")
        self._max_ratio_adjust = max_ratio_adjust
        soxr = _require_soxr()
        np = _get_numpy()
        if np is None:
            raise ImportError(
                "numpy is required for AsrcSourceBridge. "
                "Install the 'asrc' extra: pip install aiosendspin[asrc]"
            )
        self._np = np
        # PyAV converts format and layout while soxr owns the rate.
        self._resampler_rate = input_format.sample_rate
        self._resampler = self._build_resampler(rate=self._resampler_rate)
        if output_format.sample_type == "float":
            self._dtype = "float32"
        elif output_format.bit_depth == 16:
            self._dtype = "int16"
        else:
            self._dtype = "int32"
        # Leave headroom for variable ratios.
        self._soxr_stream = soxr.ResampleStream(
            input_format.sample_rate * (1.0 + max_ratio_adjust),
            output_format.sample_rate,
            output_format.channels,
            dtype=self._dtype,
            quality=quality,
            vr=True,
        )
        self._soxr_stream.set_io_ratio(input_format.sample_rate, output_format.sample_rate)
        self._rate_estimate = 0.0
        self._last_ratio_adjust = 0.0
        self._ratio_clamped = False

    def _correct_drift(self) -> None:
        """Apply the measured rate estimate plus a slow occupancy trim, clamped."""
        cap = self._max_ratio_adjust
        error_us = self.occupancy_us - self._target_us
        raw = self._rate_estimate + error_us / _RATIO_TIME_CONSTANT_US
        adjust = max(-cap, min(cap, raw))
        self._last_ratio_adjust = adjust
        clamped = abs(raw) >= cap
        if clamped and not self._ratio_clamped:
            logger.debug(
                "Resample ratio saturated at %+d ppm; the source's skew exceeds its range",
                round(adjust * 1_000_000),
            )
        self._ratio_clamped = clamped
        self._soxr_stream.set_io_ratio(self._in.sample_rate * (1.0 + adjust), self._out.sample_rate)

    def _drift_detail(self) -> str:
        """Report the applied resample ratio and the measured source rate skew."""
        return (
            f", resample ratio {round(self._last_ratio_adjust * 1_000_000):+d} ppm"
            f" (rate estimate {round(self._rate_estimate * 1_000_000):+d} ppm)"
        )

    def _apply_window_correction(self) -> None:
        """Update measured rate skew from the latest window."""
        if self._window_consumed_us <= 0:
            return
        skew = self._window_supplied_us / self._window_consumed_us - 1.0
        if abs(skew) > self._max_ratio_adjust:
            return
        corroborated = (self._window_underruns > 0 and skew < self._rate_estimate) or (
            self._window_overflow_us > 0 and skew > self._rate_estimate
        )
        if corroborated:
            self._rate_estimate = skew
            return
        step = _RATE_EST_ALPHA * (skew - self._rate_estimate)
        step = max(-_RATE_EST_MAX_STEP, min(_RATE_EST_MAX_STEP, step))
        self._rate_estimate += step

    def _convert(self, pcm: bytes) -> bytes:
        raw = self._convert_via(self._resampler, pcm)
        arr = self._np.frombuffer(raw, dtype=self._dtype).reshape(-1, self._out.channels)
        resampled = self._soxr_stream.resample_chunk(arr)
        data = self._np.ascontiguousarray(resampled).tobytes()
        return _convert_s32_to_s24(data) if self._out.bit_depth == 24 else data

    def flush(self) -> None:
        """Append the resampler tail after the final input chunk."""
        raw = self._drain_av_tail()
        arr = self._np.frombuffer(raw, dtype=self._dtype).reshape(-1, self._out.channels)
        resampled = self._soxr_stream.resample_chunk(arr, last=True)
        data = self._np.ascontiguousarray(resampled).tobytes()
        if not data:
            return
        if self._out.bit_depth == 24:
            data = _convert_s32_to_s24(data)
        self._buffer += data
        overflow_us = self.occupancy_us - self._max_us
        if overflow_us > 0:
            del self._buffer[: self._us_to_bytes(overflow_us)]

    def _reset(self) -> None:
        # The rate estimate survives resets: the hardware's clock skew does too.
        super()._reset()
        self._last_ratio_adjust = 0.0
        self._ratio_clamped = False
        self._soxr_stream.clear()
        self._soxr_stream.set_io_ratio(self._in.sample_rate, self._out.sample_rate)


__all__ = ["AsrcSourceBridge", "SourceBridge"]
