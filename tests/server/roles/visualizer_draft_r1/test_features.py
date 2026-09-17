"""Tests for visualizer feature extraction and packing."""

# DEPRECATED(spec-pr-86): remove in aiosendspin <version>

from __future__ import annotations

import math
import struct

import numpy as np

from aiosendspin.models.visualizer_draft_r1 import (
    ClientHelloVisualizerSpectrum,
    StreamStartVisualizer,
)
from aiosendspin.server.roles.visualizer_draft_r1.features import VisualizerFeatureExtractor
from aiosendspin.server.roles.visualizer_draft_r1.packing import pack_visualization_message
from tests.server.roles.visualizer_draft_r1.conftest import sine_pcm_16bit


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


def test_extractor_produces_loudness_peak_and_spectrum() -> None:
    """Extractor returns non-empty features for a sine wave."""
    config = StreamStartVisualizer(
        types=("loudness", "f_peak", "spectrum"),
        batch_max=4,
        spectrum=ClientHelloVisualizerSpectrum(
            n_disp_bins=12,
            scale="lin",
            f_min=20,
            f_max=16_000,
            rate_max=60,
        ),
    )
    extractor = VisualizerFeatureExtractor(sample_rate=48_000, channels=2, config=config)
    pcm = sine_pcm_16bit(sample_rate=48_000, channels=2, hz=1000.0, duration_s=0.025)

    frame = extractor.process_chunk(pcm, 1_000_000)

    assert frame.loudness is not None
    assert frame.loudness > 0
    assert frame.f_peak is not None
    assert abs(frame.f_peak - 1000) < 300
    assert frame.spectrum is not None
    assert isinstance(frame.spectrum, np.ndarray)
    assert frame.spectrum.shape == (12,)
    assert frame.spectrum.dtype == np.uint16


def test_pack_visualization_message_binary_layout() -> None:
    """Packed message starts with type byte, frame count, and then frame data."""
    config = StreamStartVisualizer(
        types=("loudness", "f_peak"),
        batch_max=4,
        spectrum=None,
    )
    extractor = VisualizerFeatureExtractor(sample_rate=48_000, channels=2, config=config)
    pcm = sine_pcm_16bit(sample_rate=48_000, channels=2, hz=500.0, duration_s=0.025)
    frame = extractor.process_chunk(pcm, 1_234_000)

    packed = pack_visualization_message(frames=[frame], config=config)

    assert packed[0] == 16  # message type
    assert packed[1] == 1  # frame count
    assert struct.unpack(">q", packed[2:10])[0] == 1_234_000
    # type(1) + count(1) + timestamp(8) + loudness(2) + f_peak(2) = 14
    assert len(packed) == 1 + 1 + 8 + 2 + 2


def test_peak_frequency_uses_compensated_magnitude() -> None:
    """A-weighting should prefer perceptually louder mids over deep bass."""
    config = StreamStartVisualizer(
        types=("f_peak",),
        batch_max=4,
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

    frame = extractor.process_chunk(pcm, 2_000_000)

    assert frame.f_peak is not None
    assert abs(frame.f_peak - 2_000) < 400
