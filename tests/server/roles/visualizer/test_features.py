"""Tests for visualizer feature extraction and packing."""

from __future__ import annotations

import math
import struct

import numpy as np

from aiosendspin.models.types import BinaryMessageType
from aiosendspin.models.visualizer import (
    ClientHelloVisualizerSpectrum,
    StreamStartVisualizer,
)
from aiosendspin.server.roles.visualizer.features import VisualizerFeatureExtractor
from aiosendspin.server.roles.visualizer.packing import pack_visualizer_frame
from tests.server.roles.visualizer.conftest import sine_pcm_16bit


def _dual_sine_pcm_16bit(
    *,
    sample_rate: int,
    channels: int,
    hz_a: float,
    hz_b: float,
    duration_s: float,
) -> bytes:
    sample_count = int(sample_rate * duration_s)
    frame_bytes = bytearray()
    for i in range(sample_count):
        v_a = math.sin((2.0 * math.pi * hz_a * i) / sample_rate)
        v_b = math.sin((2.0 * math.pi * hz_b * i) / sample_rate)
        value = int(32767 * np.clip(0.5 * (v_a + v_b), -1.0, 1.0))
        packed = struct.pack("<h", value)
        for _ in range(channels):
            frame_bytes.extend(packed)
    return bytes(frame_bytes)


def _spectrum_config(rate_max: int = 60, n_bins: int = 12) -> StreamStartVisualizer:
    return StreamStartVisualizer(
        types=("loudness", "f_peak", "spectrum"),
        rate_max=rate_max,
        spectrum=ClientHelloVisualizerSpectrum(
            n_disp_bins=n_bins,
            scale="lin",
            f_min=20,
            f_max=16_000,
        ),
    )


def test_extractor_produces_loudness_peak_and_spectrum() -> None:
    """Extractor returns non-empty features for a sine wave."""
    config = _spectrum_config()
    extractor = VisualizerFeatureExtractor(sample_rate=48_000, channels=2, config=config)
    pcm = sine_pcm_16bit(sample_rate=48_000, channels=2, hz=1000.0, duration_s=0.025)

    frames = extractor.process_chunk(pcm, 1_000_000)

    assert frames, "first chunk should produce at least one frame"
    frame = frames[0]
    # Trailing window: first chunk's frame is anchored at chunk end.
    assert frame.timestamp_us == 1_025_000
    assert frame.loudness is not None
    assert frame.loudness > 0
    assert frame.f_peak_freq is not None
    assert abs(frame.f_peak_freq - 1000) < 300
    assert frame.f_peak_amp is not None
    assert frame.f_peak_amp > 0
    assert frame.spectrum is not None
    assert frame.spectrum.shape == (12,)
    assert frame.spectrum.dtype == np.uint16


def test_pack_visualizer_frame_layout_loudness() -> None:
    """[16][ts:8][uint16] = 11 bytes."""
    payload = struct.pack(">H", 0x7FFF)
    packed = pack_visualizer_frame(BinaryMessageType.VISUALIZATION_LOUDNESS, 1_234_000, payload)
    assert packed[0] == 16
    assert struct.unpack(">q", packed[1:9])[0] == 1_234_000
    assert struct.unpack(">H", packed[9:11])[0] == 0x7FFF
    assert len(packed) == 11


def test_pack_visualizer_frame_layout_spectrum() -> None:
    """[19][ts:8][uint16 * n_bins]."""
    bins = np.arange(8, dtype=np.uint16) * 100
    payload = bins.astype(">u2", copy=False).tobytes()
    packed = pack_visualizer_frame(BinaryMessageType.VISUALIZATION_SPECTRUM, 42, payload)
    assert packed[0] == 19
    assert struct.unpack(">q", packed[1:9])[0] == 42
    assert len(packed) == 1 + 8 + 16


def test_peak_frequency_uses_compensated_magnitude() -> None:
    """A-weighting should prefer perceptually louder mids over deep bass."""
    config = StreamStartVisualizer(
        types=("f_peak",),
        rate_max=30,
        spectrum=None,
    )
    extractor = VisualizerFeatureExtractor(sample_rate=48_000, channels=2, config=config)
    pcm = _dual_sine_pcm_16bit(
        sample_rate=48_000,
        channels=2,
        hz_a=80.0,
        hz_b=2_000.0,
        duration_s=0.05,
    )

    frames = extractor.process_chunk(pcm, 2_000_000)

    assert frames
    frame = frames[0]
    assert frame.f_peak_freq is not None
    assert abs(frame.f_peak_freq - 2_000) < 400


# ---------------------------------------------------------------------------
# Hop scheduling: rate_max drives multi-frame emission
# ---------------------------------------------------------------------------


def _feed_steady_chunks(
    extractor: VisualizerFeatureExtractor,
    *,
    sample_rate: int,
    channels: int,
    hz: float,
    chunks: int,
    chunk_duration_s: float = 0.025,
    start_ts_us: int = 0,
) -> list:
    """Feed `chunks` back-to-back 25 ms chunks; return concatenated frames."""
    frames: list = []
    chunk_duration_us = int(chunk_duration_s * 1_000_000)
    for i in range(chunks):
        pcm = sine_pcm_16bit(
            sample_rate=sample_rate, channels=channels, hz=hz, duration_s=chunk_duration_s
        )
        ts = start_ts_us + i * chunk_duration_us
        frames.extend(extractor.process_chunk(pcm, ts))
    return frames


def test_hop_matches_rate_max_60_over_one_second() -> None:
    """rate_max=60 should yield ~60 frames over 1 s of audio."""
    config = _spectrum_config(rate_max=60)
    extractor = VisualizerFeatureExtractor(sample_rate=48_000, channels=2, config=config)

    frames = _feed_steady_chunks(extractor, sample_rate=48_000, channels=2, hz=1000.0, chunks=40)
    # 40 chunks x 25 ms = 1 s. Expect 60 ± a few frames (quantization slack).
    assert 55 <= len(frames) <= 62
    # Steady-state timestamps spaced ~16_666 µs apart.
    diffs = [frames[i + 1].timestamp_us - frames[i].timestamp_us for i in range(1, len(frames) - 1)]
    assert all(15_000 <= d <= 18_000 for d in diffs), diffs


def test_hop_matches_rate_max_30_over_one_second() -> None:
    """rate_max=30 yields ~30 frames over 1 s."""
    config = _spectrum_config(rate_max=30)
    extractor = VisualizerFeatureExtractor(sample_rate=48_000, channels=2, config=config)

    frames = _feed_steady_chunks(extractor, sample_rate=48_000, channels=2, hz=1000.0, chunks=40)
    assert 28 <= len(frames) <= 32


def test_one_frame_per_chunk_with_rate_max_equal_chunk_rate() -> None:
    """At rate_max=40 (chunk cadence) every chunk yields exactly one frame."""
    config = _spectrum_config(rate_max=40)
    extractor = VisualizerFeatureExtractor(sample_rate=48_000, channels=2, config=config)

    frames = _feed_steady_chunks(extractor, sample_rate=48_000, channels=2, hz=1000.0, chunks=10)
    assert len(frames) == 10


def test_high_rate_max_caps_at_window_resolution() -> None:
    """rate_max much higher than chunk rate still emits each hop, bounded by buffer."""
    config = _spectrum_config(rate_max=120)
    extractor = VisualizerFeatureExtractor(sample_rate=48_000, channels=2, config=config)

    frames = _feed_steady_chunks(extractor, sample_rate=48_000, channels=2, hz=1000.0, chunks=40)
    # 120 fps x 1 s = 120 expected.
    assert 110 <= len(frames) <= 125


def test_reset_clears_emit_cursor_and_buffer() -> None:
    """reset() returns extractor to a pristine state."""
    config = _spectrum_config(rate_max=60)
    extractor = VisualizerFeatureExtractor(sample_rate=48_000, channels=2, config=config)

    _feed_steady_chunks(extractor, sample_rate=48_000, channels=2, hz=1000.0, chunks=5)
    extractor.reset()

    # After reset, first chunk anchors emit cursor to its chunk end again.
    pcm = sine_pcm_16bit(sample_rate=48_000, channels=2, hz=1000.0, duration_s=0.025)
    frames = extractor.process_chunk(pcm, 9_000_000)
    assert frames[0].timestamp_us == 9_025_000


# ---------------------------------------------------------------------------
# Steady-state stability (Problem 2: spectrum flicker)
# ---------------------------------------------------------------------------


def test_spectrum_stable_on_constant_tone() -> None:
    """A pure sine over 1 s should produce a low-variance spectrum bin."""
    config = _spectrum_config(rate_max=60)
    extractor = VisualizerFeatureExtractor(sample_rate=48_000, channels=2, config=config)

    frames = _feed_steady_chunks(extractor, sample_rate=48_000, channels=2, hz=1000.0, chunks=40)
    # Drop the warmup half: EMA + larger windows need time to stabilize.
    steady = frames[len(frames) // 2 :]
    assert steady
    spectra = np.stack([f.spectrum for f in steady if f.spectrum is not None])
    # Identify the loudest bin once stable.
    loudest_idx = int(np.argmax(spectra.mean(axis=0)))
    column = spectra[:, loudest_idx].astype(np.float64)
    mean = column.mean()
    std = column.std()
    assert mean > 1000, f"loudest bin should be well above floor (mean={mean})"
    # Coefficient of variation below 5% — far tighter than per-chunk FFT yields.
    assert std / mean < 0.05, f"loudest bin too jittery: mean={mean}, std={std}"


def test_f_peak_stable_on_constant_tone() -> None:
    """Reported f_peak frequency stays within ±5 Hz across steady-state frames."""
    config = StreamStartVisualizer(
        types=("f_peak",),
        rate_max=60,
        spectrum=None,
    )
    extractor = VisualizerFeatureExtractor(sample_rate=48_000, channels=2, config=config)

    frames = _feed_steady_chunks(extractor, sample_rate=48_000, channels=2, hz=1000.0, chunks=40)
    steady = frames[len(frames) // 2 :]
    freqs = np.array([f.f_peak_freq for f in steady if f.f_peak_freq is not None])
    assert freqs.size > 5
    # All frames should sit within a ±5 Hz envelope of the median peak.
    median = float(np.median(freqs))
    assert (np.abs(freqs - median) <= 5).all(), f"f_peak jitter: {freqs.tolist()}"
    # And the median itself should be near the true 1 kHz tone.
    assert abs(median - 1000.0) < 30.0


def test_spectrum_ema_converges_after_silence_to_tone_transition() -> None:
    """After step input, spectrum EMA reaches >=95% of steady value within a few hops."""
    config = _spectrum_config(rate_max=60)
    extractor = VisualizerFeatureExtractor(sample_rate=48_000, channels=2, config=config)

    # First half: silence. Second half: tone. Feed via independent calls so
    # the buffer transitions cleanly.
    silence = sine_pcm_16bit(sample_rate=48_000, channels=2, hz=1000.0, duration_s=0.025)
    silence = b"\x00" * len(silence)
    for i in range(10):
        extractor.process_chunk(silence, i * 25_000)
    # Now feed tone chunks and collect frames.
    tone_frames: list = []
    for i in range(40):
        pcm = sine_pcm_16bit(sample_rate=48_000, channels=2, hz=1000.0, duration_s=0.025)
        tone_frames.extend(extractor.process_chunk(pcm, (10 + i) * 25_000))

    assert tone_frames
    spectra = np.stack([f.spectrum for f in tone_frames if f.spectrum is not None])
    loudest_idx = int(np.argmax(spectra[-1]))
    steady_value = float(spectra[-5:, loudest_idx].mean())
    # Find when EMA crosses 95% of steady.
    column = spectra[:, loudest_idx].astype(np.float64)
    crossing = np.argmax(column >= 0.95 * steady_value)
    assert crossing > 0, "EMA never reached steady"
    # alpha=0.4 → 95% within ~6 frames; allow generous slack.
    assert crossing < 15, f"EMA convergence too slow: {crossing} frames"


def test_loudness_ema_smooths_step_input() -> None:
    """Loudness on step input rises monotonically over ~3-5 frames."""
    config = StreamStartVisualizer(
        types=("loudness",),
        rate_max=60,
        spectrum=None,
    )
    extractor = VisualizerFeatureExtractor(sample_rate=48_000, channels=2, config=config)

    silence = b"\x00" * (48_000 * 2 * 2 // 40)  # 25 ms stereo PCM16
    for i in range(5):
        extractor.process_chunk(silence, i * 25_000)
    loudness_trace: list = []
    for i in range(20):
        pcm = sine_pcm_16bit(sample_rate=48_000, channels=2, hz=1000.0, duration_s=0.025)
        loudness_trace.extend(
            frame.loudness
            for frame in extractor.process_chunk(pcm, (5 + i) * 25_000)
            if frame.loudness is not None
        )
    assert len(loudness_trace) >= 5
    # Loudness should rise (EMA-smoothed), not snap to peak instantly.
    first = loudness_trace[0]
    last = loudness_trace[-1]
    assert last > first
    # Smoothing visible: no single-frame jump from 0 to peak.
    assert first < 0.8 * last


# ---------------------------------------------------------------------------
# Pitch detection
# ---------------------------------------------------------------------------

# DEPRECATED(spec-pr-86): remove in aiosendspin <version>
# Every test in this section covers the deprecated `pitch` type.


def _pitch_only_config(rate_max: int = 30) -> StreamStartVisualizer:
    return StreamStartVisualizer(
        types=("pitch",),
        rate_max=rate_max,
        spectrum=None,
    )


def _dc_pcm_16bit(*, sample_rate: int, channels: int, level: int, duration_s: float) -> bytes:
    """Pure-DC PCM at a given level — used to verify the silence/DC gate."""
    sample_count = int(sample_rate * duration_s)
    packed = struct.pack("<h", level)
    return packed * (sample_count * channels)


def test_pitch_a4_detected_with_high_confidence() -> None:
    """440 Hz sine resolves to MIDI 69 with confidence > 128."""
    extractor = VisualizerFeatureExtractor(
        sample_rate=48_000, channels=2, config=_pitch_only_config()
    )
    pcm = sine_pcm_16bit(sample_rate=48_000, channels=2, hz=440.0, duration_s=0.05)
    frames = extractor.process_chunk(pcm, 1_000_000)
    assert frames
    frame = frames[0]
    assert frame.pitch_midi_q88 is not None
    assert frame.pitch_confidence is not None
    midi = frame.pitch_midi_q88 / 256.0
    assert abs(midi - 69.0) < 0.5
    assert frame.pitch_confidence > 128


def test_pitch_a3_vocal_range_not_octave_up() -> None:
    """220 Hz sine (A3) resolves correctly — no octave-up bias from A-weighting.

    Regression: A-weighting attenuates ~220 Hz by ~13 dB while boosting
    the 2nd-3rd harmonics by 0-1.2 dB. Feeding A-weighted magnitude to
    the ACF would shift the YIN dip to a harmonic period (MIDI ≈ 81
    instead of 69). The extractor now uses the unweighted magnitude for
    pitch.
    """
    extractor = VisualizerFeatureExtractor(
        sample_rate=48_000, channels=2, config=_pitch_only_config()
    )
    pcm = sine_pcm_16bit(sample_rate=48_000, channels=2, hz=220.0, duration_s=0.05)
    frames = extractor.process_chunk(pcm, 1_000_000)
    assert frames
    frame = frames[0]
    assert frame.pitch_midi_q88 is not None
    midi = frame.pitch_midi_q88 / 256.0
    # A3 is MIDI 57; an octave-up bias would land us near 69.
    assert abs(midi - 57.0) < 1.0, f"expected ~57, got {midi}"


def test_pitch_returns_none_for_dc_input() -> None:
    """Pure DC must not produce a confident ghost pitch."""
    extractor = VisualizerFeatureExtractor(
        sample_rate=48_000, channels=2, config=_pitch_only_config()
    )
    # 1% full-scale DC bias — well above the -45 dBFS RMS floor, so
    # the older code path emitted a spurious confident pitch.
    pcm = _dc_pcm_16bit(sample_rate=48_000, channels=2, level=327, duration_s=0.05)
    frames = extractor.process_chunk(pcm, 1_000_000)
    for frame in frames:
        assert frame.pitch_midi_q88 is None
        assert frame.pitch_confidence is None


def test_pitch_returns_none_for_silence() -> None:
    """Silence is unvoiced."""
    extractor = VisualizerFeatureExtractor(
        sample_rate=48_000, channels=2, config=_pitch_only_config()
    )
    pcm = b"\x00" * (48_000 * 2 * 2 // 20)  # 50 ms stereo silence
    frames = extractor.process_chunk(pcm, 1_000_000)
    for frame in frames:
        assert frame.pitch_midi_q88 is None


def test_window_samples_scales_with_sample_rate() -> None:
    """At 96 kHz the FFT window doubles to keep the time span constant."""
    extractor_48k = VisualizerFeatureExtractor(
        sample_rate=48_000, channels=2, config=_pitch_only_config()
    )
    extractor_96k = VisualizerFeatureExtractor(
        sample_rate=96_000, channels=2, config=_pitch_only_config()
    )
    assert extractor_48k._window_samples == 2048  # noqa: SLF001
    assert extractor_96k._window_samples == 4096  # noqa: SLF001


def test_fft_size_caches_reused_across_frames() -> None:
    """Per-window-size caches (hann, rfftfreq, bin assignment) are reused, not rebuilt."""
    config = _spectrum_config(rate_max=60)
    extractor = VisualizerFeatureExtractor(sample_rate=48_000, channels=2, config=config)
    # 100ms chunk fills the rolling buffer so the window settles to its full
    # size (2048) on the very first frame.
    pcm = sine_pcm_16bit(sample_rate=48_000, channels=2, hz=1000.0, duration_s=0.1)

    assert extractor.process_chunk(pcm, 1_000_000)
    hann = extractor._hann_window  # noqa: SLF001
    freqs = extractor._rfftfreq  # noqa: SLF001
    bin_cache = extractor._spectrum_bin_cache  # noqa: SLF001
    assert hann is not None
    assert freqs is not None
    assert bin_cache is not None

    # Contiguous second chunk → same window size → cached arrays reused verbatim.
    assert extractor.process_chunk(pcm, 1_100_000)
    assert extractor._hann_window is hann  # noqa: SLF001
    assert extractor._rfftfreq is freqs  # noqa: SLF001
    assert extractor._spectrum_bin_cache is bin_cache  # noqa: SLF001
