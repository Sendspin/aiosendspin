"""Feature extraction for draft visualizer role."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from aiosendspin.models.visualizer import StreamStartVisualizer

# Default FFT window at 48 kHz: 2048 samples ≈ 42 ms → ~23 Hz bin
# resolution, stable on constant tones. Decoupled from chunk cadence by
# the rolling buffer. At other rates the window scales so the time span
# (and therefore the achievable pitch F_MIN) stays roughly constant —
# see `_window_samples_for_rate`.
_DEFAULT_WINDOW_SAMPLES = 2048
_REFERENCE_SAMPLE_RATE = 48_000
# Below this many samples the FFT result is too coarse to be worth emitting.
_MIN_WINDOW_SAMPLES = 256
# Temporal smoothing factors (new sample weight). 0 = no smoothing, 1 = no memory.
_SPECTRUM_EMA_ALPHA = 0.4
_LOUDNESS_EMA_ALPHA = 0.5
# f_peak hysteresis: only switch reported peak bin when its magnitude exceeds
# the held peak's magnitude by this ratio. Suppresses bin-hop jitter between
# near-equal neighbours on steady tones.
_F_PEAK_SWITCH_RATIO = 1.10
# DEPRECATED(spec-pr-86): remove in aiosendspin <version>
# Pitch (YINFFT) search bounds and gating. The floor is set to the melody
# register (not the bass) so octave-down errors fall below it and are rejected.
_PITCH_F_MIN = 130.0
_PITCH_F_MAX = 1200.0
# Pitch ACF is computed via `rfft → |X|² → irfft`, which is the *circular*
# autocorrelation of the windowed signal — not the linear one YIN
# assumes. For `tau_max / n ≈ 0.36` (130 Hz at 48 kHz with a 2048-sample
# window) wrap-around contamination is bounded by the Hanning taper.
# Dropping F_MIN below ~100 Hz requires zero-padding the time-domain
# signal to `2n` before the rfft to recover a proper linear ACF.
# Below this windowed RMS (dBFS) the frame is treated as unvoiced.
_PITCH_RMS_FLOOR_DB = -45.0
# YIN absolute threshold: first CMNDF dip below this is taken as the period.
_PITCH_YIN_THRESHOLD = 0.12
# Voicing gate: above this normalized-difference at the chosen period the frame
# is aperiodic (percussion, noise, unpitched) and no pitch is emitted.
_PITCH_UNVOICED_DPRIME = 0.35
# Octave stabilization: EMA weight of the running pitch register, max semitone
# distance at which a raw note is snapped to the register's octave, and the gap
# after which the register is considered stale.
_PITCH_REGISTER_ALPHA = 0.25
_PITCH_SNAP_MAX_SEMITONES = 7.0
_PITCH_REGISTER_GAP_US = 400_000


@dataclass(frozen=True)
class ExtractedFrame:
    """Computed visualizer features for one emit timestamp."""

    timestamp_us: int
    loudness: int | None = None
    # f_peak fields are paired — both populated together.
    f_peak_freq: int | None = None
    f_peak_amp: int | None = None
    spectrum: np.ndarray | None = None
    # Onset detector output; None when no transient fired this frame.
    peak: int | None = None
    # DEPRECATED(spec-pr-86): remove in aiosendspin <version>
    # Pitch fields are paired — both None when no confident pitch was detected.
    pitch_midi_q88: int | None = None
    pitch_confidence: int | None = None


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _db_norm(value: float, ref: float) -> float:
    """Map a linear A-weighted magnitude to the spec's [0, 1] dB scale.

    -60 dB maps to 0, 0 dB to 1, linear in dB across that window. `ref` is the
    full-scale reference in the caller's domain. Shared by `loudness` and the
    `f_peak` amplitude so both use the single dB mapping the spec pins for
    visualizer amplitudes.
    """
    ratio = max(value / ref, 1e-10)
    db = 20.0 * np.log10(ratio)
    return float(np.clip((db + 60.0) / 60.0, 0.0, 1.0))


class VisualizerFeatureExtractor:
    """Compute visualizer features from PCM chunks at a configurable hop rate."""

    def __init__(
        self,
        *,
        sample_rate: int,
        channels: int,
        config: StreamStartVisualizer,
    ) -> None:
        """Create a feature extractor for negotiated stream config."""
        self._sample_rate = sample_rate
        self._channels = channels
        self._config = config

        # Hop is derived from rate_max. Zero/negative means "one frame per chunk":
        # cursor is reset to chunk_end after every chunk.
        self._hop_us: int = 1_000_000 // config.rate_max if config.rate_max > 0 else 0
        self._window_samples: int = min(self._window_samples_for_rate(sample_rate), sample_rate)

        # Rolling mono buffer + ts of its first sample.
        self._buffer: np.ndarray = np.zeros(0, dtype=np.float32)
        self._buffer_start_ts_us: int | None = None
        # Cursor: ts of the NEXT frame to emit. Set on first chunk.
        self._next_emit_ts_us: int | None = None

        # Per-FFT-size caches for values that are constant once the window
        # size settles (recomputing them every frame is pure overhead on
        # constrained hardware). Keyed by window length so the brief warmup
        # ramp, where the window grows, refreshes them and then they stick.
        self._hann_window: np.ndarray | None = None
        self._rfftfreq: np.ndarray | None = None
        # (freqs.size, valid mask, bin-index array) for the spectrum binning.
        self._spectrum_bin_cache: tuple[int, np.ndarray, np.ndarray] | None = None

        # Smoothing / hysteresis state.
        self._spectrum_ema: np.ndarray | None = None
        self._loudness_ema: float | None = None
        self._a_weight_cache: np.ndarray | None = None
        self._f_peak_last_idx: int | None = None
        # Onset detector state: EMA of A-weighted energy. A new frame fires a
        # `peak` when current energy exceeds the EMA by a threshold ratio.
        self._energy_ema: float | None = None
        self._last_peak_ts_us: int | None = None
        # DEPRECATED(spec-pr-86): remove in aiosendspin <version>
        # Pitch octave-stabilization state: EMA of recent raw MIDI and the ts of
        # the last voiced frame. Reset by any unvoiced frame.
        self._pitch_register: float | None = None
        self._pitch_last_ts_us: int | None = None

    @staticmethod
    def _window_samples_for_rate(sample_rate: int) -> int:
        """Scale the FFT window with sample rate, rounded to the next power of two.

        Keeps the time-domain span (and therefore the pitch F_MIN) roughly
        constant across input rates. At the reference rate the default
        2048 samples are used unchanged.
        """
        target = _DEFAULT_WINDOW_SAMPLES * sample_rate / _REFERENCE_SAMPLE_RATE
        size = 1
        while size < target:
            size <<= 1
        return size

    def reset(self) -> None:
        """Reset extractor state at stream boundaries."""
        self._buffer = np.zeros(0, dtype=np.float32)
        self._buffer_start_ts_us = None
        self._next_emit_ts_us = None
        self._spectrum_ema = None
        self._loudness_ema = None
        self._f_peak_last_idx = None
        self._energy_ema = None
        self._last_peak_ts_us = None
        self._pitch_register = None
        self._pitch_last_ts_us = None

    def process_chunk(self, pcm: bytes, timestamp_us: int) -> list[ExtractedFrame]:
        """Append PCM and emit one frame per hop boundary that fits.

        Returns frames in non-decreasing timestamp order. May be empty if the
        chunk is too short to advance past the next hop boundary.
        """
        mono = self._decode_pcm_to_mono_float32(pcm)
        if mono.size == 0:
            return []

        chunk_duration_us = round(mono.size * 1_000_000 / self._sample_rate)
        chunk_end_ts_us = timestamp_us + chunk_duration_us

        if self._buffer_start_ts_us is not None:
            expected_ts = self._buffer_start_ts_us + round(
                self._buffer.size * 1_000_000 / self._sample_rate
            )
            # Discontinuity in chunk timestamps (seek, codec restart, sparse
            # feeding) invalidates the rolling buffer's contiguous-sample
            # assumption. Reset and treat this chunk as a fresh start.
            if abs(timestamp_us - expected_ts) > 1_000:
                self._buffer = np.zeros(0, dtype=np.float32)
                self._buffer_start_ts_us = None
                self._next_emit_ts_us = None

        if self._buffer_start_ts_us is None:
            self._buffer = mono.copy()
            self._buffer_start_ts_us = timestamp_us
            # Anchor first emit at chunk end so the first frame's window
            # covers the whole chunk.
            self._next_emit_ts_us = chunk_end_ts_us
        else:
            self._buffer = np.concatenate([self._buffer, mono])

        # `rate_max <= 0` falls back to one frame per chunk at chunk end.
        if self._hop_us <= 0:
            self._next_emit_ts_us = chunk_end_ts_us

        frames: list[ExtractedFrame] = []
        assert self._next_emit_ts_us is not None
        while self._next_emit_ts_us <= chunk_end_ts_us:
            emit_ts = self._next_emit_ts_us
            window = self._extract_window(emit_ts)
            if window is None:
                if self._hop_us <= 0:
                    break
                self._next_emit_ts_us += self._hop_us
                continue
            frames.append(self._compute_frame(window, emit_ts))
            if self._hop_us <= 0:
                self._next_emit_ts_us = chunk_end_ts_us + 1
                break
            self._next_emit_ts_us += self._hop_us

        self._trim_buffer()
        return frames

    def _extract_window(self, emit_ts: int) -> np.ndarray | None:
        """Return up to `_window_samples` PCM samples ending at `emit_ts`.

        Returns None when fewer than `_MIN_WINDOW_SAMPLES` are available.
        """
        assert self._buffer_start_ts_us is not None
        emit_idx = round((emit_ts - self._buffer_start_ts_us) * self._sample_rate / 1_000_000)
        emit_idx = min(emit_idx, self._buffer.size)
        if emit_idx < _MIN_WINDOW_SAMPLES:
            return None
        n = min(self._window_samples, emit_idx)
        return self._buffer[emit_idx - n : emit_idx]

    def _trim_buffer(self) -> None:
        """Bound buffer growth — keep the last `_window_samples` only."""
        keep = self._window_samples
        if self._buffer.size > keep * 2:
            drop = self._buffer.size - keep
            self._buffer = self._buffer[-keep:]
            assert self._buffer_start_ts_us is not None
            self._buffer_start_ts_us += round(drop * 1_000_000 / self._sample_rate)

    def _compute_frame(self, mono: np.ndarray, emit_ts: int) -> ExtractedFrame:
        """Compute a single frame from a windowed mono PCM slice."""
        needs_fft = any(
            t in self._config.types for t in ("loudness", "f_peak", "spectrum", "peak", "pitch")
        )

        loudness: int | None = None
        f_peak_freq: int | None = None
        f_peak_amp: int | None = None
        spectrum: np.ndarray | None = None
        peak: int | None = None
        pitch_midi_q88: int | None = None
        pitch_confidence: int | None = None

        if needs_fft:
            freqs, magnitude = self._fft_magnitude(mono)
            compensated = self._apply_psychoacoustic_compensation(freqs, magnitude)

            if "loudness" in self._config.types:
                loudness = self._compute_loudness_db(compensated, mono.size)

            if "f_peak" in self._config.types:
                f_peak_freq, f_peak_amp = self._compute_f_peak(freqs, compensated, mono.size)

            if "spectrum" in self._config.types:
                spectrum = self._compute_spectrum(freqs, compensated)

            if "peak" in self._config.types:
                # TODO: onset detection runs only at the periodic hop, so `peak` is
                # throttled by rate_max. The spec says peak is event-driven and NOT
                # throttled. Decouple onset detection to run per audio chunk,
                # independent of the periodic frame hop.
                peak = self._detect_onset(compensated, emit_ts)

            # DEPRECATED(spec-pr-86): remove in aiosendspin <version>
            if "pitch" in self._config.types:
                pitch_midi_q88, pitch_confidence = self._compute_pitch_yinfft(mono, magnitude)
                if pitch_midi_q88 is None:
                    self._pitch_register = None
                    self._pitch_last_ts_us = None
                else:
                    pitch_midi_q88 = self._stabilize_pitch_octave(pitch_midi_q88, emit_ts)

        return ExtractedFrame(
            timestamp_us=emit_ts,
            loudness=loudness,
            f_peak_freq=f_peak_freq,
            f_peak_amp=f_peak_amp,
            spectrum=spectrum,
            peak=peak,
            pitch_midi_q88=pitch_midi_q88,
            pitch_confidence=pitch_confidence,
        )

    def _compute_loudness_db(self, compensated: np.ndarray, n_samples: int) -> int:
        """A-weighted, dB-scaled loudness in `[0, 65535]`, lightly smoothed.

        Mirrors the spectrum encoding: -60 dB → 0, 0 dB → 65535, linear in dB
        across that window.
        """
        if compensated.size == 0:
            return 0
        weighted_power = float(np.sum(np.square(compensated, dtype=np.float64)))
        rms = np.sqrt(2.0 * weighted_power) / max(n_samples, 1)
        # Full-scale sine through Hanning has RMS ≈ sqrt(3/16) ≈ 0.43.
        ref = float(np.sqrt(3.0 / 16.0))
        normalized = _db_norm(float(rms), ref)
        if self._loudness_ema is None:
            self._loudness_ema = normalized
        else:
            self._loudness_ema = (
                _LOUDNESS_EMA_ALPHA * normalized + (1.0 - _LOUDNESS_EMA_ALPHA) * self._loudness_ema
            )
        return int(self._loudness_ema * 65535.0)

    def _compute_f_peak(
        self, freqs: np.ndarray, compensated: np.ndarray, n_samples: int
    ) -> tuple[int, int]:
        """Return (freq_hz, amp_db_q16) for the dominant FFT bin with sub-bin interpolation."""
        if compensated.size == 0:
            return 0, 0
        candidate_idx = int(np.argmax(compensated))
        last_idx = self._f_peak_last_idx
        # Hysteresis: stay on the previously held bin unless the new candidate's
        # magnitude exceeds it by the switch ratio.
        if (
            last_idx is not None
            and 0 <= last_idx < compensated.size
            and candidate_idx != last_idx
            and compensated[candidate_idx] < _F_PEAK_SWITCH_RATIO * compensated[last_idx]
        ):
            idx = last_idx
        else:
            idx = candidate_idx
        self._f_peak_last_idx = idx
        if idx >= freqs.size or compensated[idx] <= 0.0:
            return 0, 0
        # Parabolic interpolation across the bin's neighbours for sub-bin freq.
        peak_mag = float(compensated[idx])
        peak_hz_float = float(freqs[idx])
        if 0 < idx < compensated.size - 1:
            a = float(compensated[idx - 1])
            b = float(compensated[idx])
            c = float(compensated[idx + 1])
            denom = a - 2.0 * b + c
            if abs(denom) > 1e-12:
                delta = 0.5 * (a - c) / denom
                # Clamp to ±1 bin so a bad neighbour can't fling the peak.
                delta = float(np.clip(delta, -1.0, 1.0))
                bin_hz = float(freqs[idx + 1] - freqs[idx]) if idx + 1 < freqs.size else 0.0
                peak_hz_float = float(freqs[idx]) + delta * bin_hz
                peak_mag = b - 0.25 * (a - c) * delta
        peak_hz = int(np.clip(round(peak_hz_float), 0, 65535))
        # Convert peak magnitude to the shared dB encoding so the amp value
        # matches `loudness`/`spectrum` scaling. Full-scale sine through Hanning
        # has rfft peak N/2, reduced to N/4 by the window.
        ref = max(float(n_samples) / 4.0, 1.0)
        amp = int(_db_norm(peak_mag, ref) * 65535.0)
        if peak_hz == 0:
            amp = 0
        return peak_hz, amp

    # DEPRECATED(spec-pr-86): remove in aiosendspin <version>
    def _compute_pitch_yinfft(
        self, mono: np.ndarray, magnitude: np.ndarray
    ) -> tuple[int | None, int | None]:
        """Return (midi_q88, confidence_uint8) via FFT-accelerated YIN.

        Derives the YIN difference function from the autocorrelation of
        the (Hanning-windowed, unweighted) magnitude spectrum via
        `irfft`. The first cumulative-mean-normalized-difference dip
        below `_PITCH_YIN_THRESHOLD` is the period, refined sub-sample
        by parabolic interpolation.

        Uses the unweighted magnitude (NOT A-weighted) on purpose: A-
        weighting attenuates the fundamental of any note below ~500 Hz
        by 3-10 dB while boosting the 1-4 kHz band, which biases the
        ACF toward harmonics and produces octave-up errors on
        mid-vocal-range pitches.

        Returns `(None, None)` below the voicing floor or when no
        confident period is found. `midi_q88` is a uint16 8.8 fixed-
        point MIDI note; `confidence_uint8` is 0-255.
        """
        n = mono.size
        # `irfft` requires `magnitude.size == n // 2 + 1` for a length-`n`
        # result; otherwise its zero-pad/truncate behaviour gives nonsense.
        if n < 64 or magnitude.size != n // 2 + 1:
            return None, None

        # DC-remove before the RMS voicing check: a small DC bias (e.g.
        # 0.01 amplitude → -40 dBFS) would otherwise pass the floor and,
        # combined with `_fft_magnitude` zeroing bin 0, produce a
        # confident ghost pitch from a constant signal.
        mono_dc_removed = mono - np.mean(mono, dtype=np.float64)
        rms = float(np.sqrt(np.mean(np.square(mono_dc_removed, dtype=np.float64))))
        if rms <= 0.0 or 20.0 * np.log10(rms) < _PITCH_RMS_FLOOR_DB:
            return None, None

        sr = float(self._sample_rate)
        tau_min = max(2, int(sr / _PITCH_F_MAX))
        tau_max = min(n // 2, int(sr / _PITCH_F_MIN))
        if tau_max <= tau_min + 1:
            return None, None

        # YIN difference via the windowed signal's autocorrelation
        # (Wiener-Khinchin on |X|²).
        power = np.square(magnitude.astype(np.float64))
        acf = np.fft.irfft(power, n=n)
        # Second silence gate: if the windowed signal has near-zero
        # energy (`acf[0]` ≈ 0), every `d_prime` collapses and the
        # search would otherwise pick `tau_min` with confidence 1.
        if acf[0] <= 1e-12:
            return None, None
        diff = 2.0 * (acf[0] - acf[: tau_max + 1])
        np.maximum(diff, 0.0, out=diff)

        d_prime = np.ones(tau_max + 1, dtype=np.float64)
        csum = np.cumsum(diff[1:])
        d_prime[1:] = diff[1:] * np.arange(1, tau_max + 1) / np.maximum(csum, 1e-12)

        selected = -1
        for tau in range(tau_min, tau_max):
            if d_prime[tau] < _PITCH_YIN_THRESHOLD and d_prime[tau] <= d_prime[tau + 1]:
                selected = tau
                break
        if selected == -1:
            selected = tau_min + int(np.argmin(d_prime[tau_min : tau_max + 1]))
        if d_prime[selected] > _PITCH_UNVOICED_DPRIME:
            return None, None

        # Parabolic interpolation around the chosen tau for sub-sample
        # precision. Clip the vertex offset to ±0.5 — a near-zero denom
        # can otherwise fling tau outside `[selected-1, selected+1]`.
        tau_refined = float(selected)
        if 0 < selected < tau_max:
            a = d_prime[selected - 1]
            b = d_prime[selected]
            c = d_prime[selected + 1]
            denom = 2.0 * (a - 2.0 * b + c)
            if abs(denom) > 1e-12:
                offset = float(np.clip((a - c) / denom, -0.5, 0.5))
                tau_refined = selected + offset
        if tau_refined <= 0.0:
            return None, None

        freq = sr / tau_refined
        midi = 69.0 + 12.0 * float(np.log2(freq / 440.0))
        if midi < 0.0 or midi >= 128.0:
            return None, None
        midi_q88 = int(np.clip(midi * 256.0, 0, 65535))
        # Spread `d_prime ∈ [0, _PITCH_UNVOICED_DPRIME]` across the full
        # 0-255 confidence range — the gate caps `d_prime[selected]`, so
        # `(1 - d_prime)` alone would crowd output into a narrow upper
        # band (~166-255 with the default 0.35 gate).
        confidence_value = 1.0 - d_prime[selected] / _PITCH_UNVOICED_DPRIME
        confidence_uint8 = int(np.clip(confidence_value * 255.0, 0, 255))
        return midi_q88, confidence_uint8

    # DEPRECATED(spec-pr-86): remove in aiosendspin <version>
    def _stabilize_pitch_octave(self, midi_q88: int, emit_ts: int) -> int:
        """Snap a raw pitch to the octave nearest the running register.

        Tracks an EMA of recent *raw* (pre-snap) notes. When a raw note sits
        closer to the register an octave (or two) away, it is shifted there —
        suppressing the octave flips that frame-independent estimation produces
        on steady notes. Following the raw note (not the snapped one) lets the
        register climb to a sustained octave leap within a few frames, so real
        melodic leaps survive while one-frame flips are absorbed. The register
        is reset by any unvoiced frame (handled by the caller) or a long gap.
        """
        midi = midi_q88 / 256.0
        register = self._pitch_register
        last_ts = self._pitch_last_ts_us
        stale = last_ts is not None and emit_ts - last_ts > _PITCH_REGISTER_GAP_US
        if register is None or stale:
            emit = midi
        else:
            emit = midi
            best_distance = abs(midi - register)
            for shift in (-24.0, -12.0, 12.0, 24.0):
                candidate = midi + shift
                if 0.0 <= candidate < 128.0 and abs(candidate - register) < best_distance:
                    emit, best_distance = candidate, abs(candidate - register)
            if best_distance > _PITCH_SNAP_MAX_SEMITONES:
                emit = midi

        if register is None or stale:
            self._pitch_register = midi
        else:
            a = _PITCH_REGISTER_ALPHA
            self._pitch_register = a * midi + (1.0 - a) * register
        self._pitch_last_ts_us = emit_ts
        return int(np.clip(emit * 256.0, 0, 65535))

    def _detect_onset(self, compensated: np.ndarray, timestamp_us: int) -> int | None:
        """Energy-based onset detector.

        Maintains an EMA of A-weighted broadband energy. When the current frame
        exceeds that average by a fixed multiplier, fires a peak whose strength
        (0-255) scales with the excess. Returns None when no onset is detected.
        """
        if compensated.size == 0:
            return None
        energy = float(np.sum(np.square(compensated.astype(np.float64))))
        # Bootstrap EMA on first frame.
        if self._energy_ema is None or self._energy_ema <= 0.0:
            self._energy_ema = max(energy, 1e-12)
            return None
        ema = self._energy_ema
        # Minimum 80 ms gap between successive onsets.
        if self._last_peak_ts_us is not None and timestamp_us - self._last_peak_ts_us < 80_000:
            # Still update EMA so the running average tracks loud passages.
            self._energy_ema = 0.9 * ema + 0.1 * energy
            return None
        threshold_multiplier = 1.6
        if energy > threshold_multiplier * ema:
            # Strength scales the excess into 0-255 over a ~6x range.
            excess = (energy / ema) - threshold_multiplier
            strength = int(np.clip(excess / 6.0, 0.0, 1.0) * 255.0)
            # Fast attack on EMA so we don't keep firing on a sustained rise.
            self._energy_ema = 0.6 * ema + 0.4 * energy
            self._last_peak_ts_us = timestamp_us
            return max(1, strength)
        # Standard EMA update outside onset events.
        self._energy_ema = 0.9 * ema + 0.1 * energy
        return None

    def _decode_pcm_to_mono_float32(self, pcm: bytes) -> np.ndarray:
        """Decode little-endian signed PCM16 to mono float32 in [-1, 1]."""
        raw = np.frombuffer(pcm, dtype="<i2")
        if raw.size == 0:
            return np.zeros(0, dtype=np.float32)

        if self._channels > 1:
            frame_count = raw.size // self._channels
            if frame_count <= 0:
                return np.zeros(0, dtype=np.float32)
            reshaped = raw[: frame_count * self._channels].reshape(frame_count, self._channels)
            mono = np.mean(reshaped.astype(np.float32), axis=1, dtype=np.float32)
        else:
            mono = raw.astype(np.float32)

        return np.asarray(np.clip(mono / 32768.0, -1.0, 1.0))

    def _fft_magnitude(self, mono: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return FFT frequency bins and normalized magnitudes."""
        if mono.size <= 1:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

        n = mono.size
        if self._hann_window is None or self._hann_window.size != n:
            self._hann_window = np.hanning(n).astype(np.float32)
            self._rfftfreq = np.fft.rfftfreq(n, d=1.0 / float(self._sample_rate)).astype(np.float32)
        assert self._rfftfreq is not None
        windowed = mono * self._hann_window
        spectrum = np.fft.rfft(windowed)
        magnitude = np.abs(spectrum).astype(np.float32)
        freqs = self._rfftfreq

        # Drop DC for peak/spectrum display.
        if magnitude.size > 0:
            magnitude[0] = 0.0
        return freqs, magnitude

    def _apply_psychoacoustic_compensation(
        self, freqs: np.ndarray, magnitude: np.ndarray
    ) -> np.ndarray:
        """Apply A-weighting gain so display tracks perceived loudness better."""
        if freqs.size == 0 or magnitude.size == 0:
            return magnitude

        # Cache A-weight gains for the current FFT size.
        if self._a_weight_cache is None or self._a_weight_cache.size != freqs.size:
            f = np.maximum(freqs.astype(np.float64), 1.0)
            f2 = f * f
            ra_num = (12194.0**2) * (f2**2)
            ra_den = (f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194.0**2)
            ra = np.clip(ra_num / np.maximum(ra_den, 1e-20), 1e-12, None)
            a_db = 2.0 + (20.0 * np.log10(ra))
            a_db = np.clip(a_db, -50.0, 6.0)
            self._a_weight_cache = np.power(10.0, a_db / 20.0).astype(np.float32)

        result: np.ndarray = magnitude * self._a_weight_cache
        return result

    def _compute_spectrum(self, freqs: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
        """Compute binned spectrum with per-bin EMA smoothing."""
        spectrum_cfg = self._config.spectrum
        if spectrum_cfg is None:
            raise ValueError("spectrum in config.types but config.spectrum is None")

        computed = self._compute_binned_spectrum(
            freqs=freqs,
            magnitude=magnitude,
            n_bins=spectrum_cfg.n_disp_bins,
            f_min=spectrum_cfg.f_min,
            f_max=spectrum_cfg.f_max,
            scale=spectrum_cfg.scale,
        )
        # Per-bin EMA in normalized space, then quantize to uint16.
        new_norm = computed.astype(np.float32) / 65535.0
        if self._spectrum_ema is None or self._spectrum_ema.size != new_norm.size:
            self._spectrum_ema = new_norm.copy()
        else:
            self._spectrum_ema = (
                _SPECTRUM_EMA_ALPHA * new_norm + (1.0 - _SPECTRUM_EMA_ALPHA) * self._spectrum_ema
            )
        return (np.clip(self._spectrum_ema, 0.0, 1.0) * 65535.0).astype(np.uint16)

    def _compute_binned_spectrum(
        self,
        *,
        freqs: np.ndarray,
        magnitude: np.ndarray,
        n_bins: int,
        f_min: int,
        f_max: int,
        scale: Literal["lin", "log", "mel"],
    ) -> np.ndarray:
        if freqs.size == 0 or magnitude.size == 0:
            return np.zeros(n_bins, dtype=np.uint16)

        lo = max(0, f_min)
        hi = min(int(self._sample_rate / 2), f_max)
        if hi <= lo:
            return np.zeros(n_bins, dtype=np.uint16)

        # Bin assignment is constant for a fixed freq grid + config; compute it
        # once per window size and reuse (digitize over ~1k bins per frame is
        # otherwise a steady cost). f_min/f_max/scale are fixed for the
        # extractor's lifetime, so freqs.size alone keys the cache.
        cache = self._spectrum_bin_cache
        if cache is not None and cache[0] == freqs.size:
            _, valid, idx = cache
        else:
            edges = self._frequency_bin_edges(n_bins=n_bins, f_min=lo, f_max=hi, scale=scale)
            bin_indices = np.digitize(freqs, edges) - 1
            valid = (bin_indices >= 0) & (bin_indices < n_bins)
            idx = bin_indices[valid]
            self._spectrum_bin_cache = (freqs.size, valid, idx)
        mag = magnitude[valid]

        sq = mag.astype(np.float64) ** 2
        sums = np.bincount(idx, weights=sq, minlength=n_bins).astype(np.float32)
        binned = np.sqrt(sums)

        # dB scale: map [-60 dB, 0 dB] relative to full-scale sine → [0, 65535].
        # Full-scale sine has rfft peak N/2, reduced to N/4 by Hanning window.
        n = freqs.size * 2 - 1
        ref = max(float(n) / 4.0, 1.0)
        ratio = np.maximum(binned / ref, 1e-10)
        db = 20.0 * np.log10(ratio)
        db_floor = -60.0
        t = (db - db_floor) / -db_floor
        t = np.maximum(t, 0.0)
        # Soft floor: bottom 10% (≈6 dB) folds into a quadratic fade so bins
        # crossing the floor don't snap between 0 and a positive value.
        soft = 10.0 * t * t
        t = np.where(t < 0.1, soft, t)
        t = np.minimum(t, 1.0)
        return (t * 65535.0).astype(np.uint16)

    def _frequency_bin_edges(
        self, *, n_bins: int, f_min: int, f_max: int, scale: Literal["lin", "log", "mel"]
    ) -> np.ndarray:
        if scale == "lin":
            return np.linspace(float(f_min), float(f_max), n_bins + 1, dtype=np.float32)
        if scale == "log":
            safe_min = max(f_min, 1)
            return np.logspace(
                np.log10(float(safe_min)),
                np.log10(float(f_max)),
                n_bins + 1,
                dtype=np.float32,
            )

        mel_edges = np.linspace(
            _hz_to_mel(np.array([float(f_min)], dtype=np.float32))[0],
            _hz_to_mel(np.array([float(f_max)], dtype=np.float32))[0],
            n_bins + 1,
            dtype=np.float32,
        )
        return _mel_to_hz(mel_edges).astype(np.float32)
