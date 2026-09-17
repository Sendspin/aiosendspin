"""Round-trip tests for the StreamStartPayload.visualizer union.

`StreamStartPayload.visualizer` is `StreamStartVisualizer (v1) |
StreamStartVisualizerDraftR1 | None`. The two schemas overlap on `types` /
`spectrum` and differ only by the required `rate_max` (v1) vs `batch_max`
(draft). These guard that mashumaro resolves each payload to the correct
branch so the v1 client SDK's `isinstance` check does not silently treat a
message as "no visualizer config".
"""

# DEPRECATED(spec-pr-86): remove in aiosendspin <version>

from __future__ import annotations

from aiosendspin.models.core import StreamStartPayload
from aiosendspin.models.visualizer import (
    ClientHelloVisualizerSpectrum,
    StreamStartVisualizer,
)
from aiosendspin.models.visualizer_draft_r1 import (
    ClientHelloVisualizerSpectrum as ClientHelloVisualizerSpectrumDraftR1,
)
from aiosendspin.models.visualizer_draft_r1 import (
    StreamStartVisualizer as StreamStartVisualizerDraftR1,
)


def _spectrum() -> ClientHelloVisualizerSpectrum:
    return ClientHelloVisualizerSpectrum(n_disp_bins=8, scale="lin", f_min=20, f_max=16_000)


def _draft_spectrum() -> ClientHelloVisualizerSpectrumDraftR1:
    # The draft spectrum schema still carries `rate_max`, dropped in v1.
    return ClientHelloVisualizerSpectrumDraftR1(
        n_disp_bins=8, scale="lin", f_min=20, f_max=16_000, rate_max=30
    )


def test_v1_stream_start_round_trips_to_v1_branch() -> None:
    """A v1 visualizer payload deserializes back to the v1 schema."""
    payload = StreamStartPayload(
        visualizer=StreamStartVisualizer(
            types=("loudness", "spectrum"),
            rate_max=30,
            spectrum=_spectrum(),
        )
    )
    restored = StreamStartPayload.from_json(payload.to_json())
    assert isinstance(restored.visualizer, StreamStartVisualizer)
    assert restored.visualizer.rate_max == 30


def test_draft_stream_start_round_trips_to_draft_branch() -> None:
    """A draft visualizer payload deserializes back to the draft schema, not None."""
    payload = StreamStartPayload(
        visualizer=StreamStartVisualizerDraftR1(
            types=("loudness", "spectrum"),
            batch_max=8,
            spectrum=_draft_spectrum(),
        )
    )
    restored = StreamStartPayload.from_json(payload.to_json())
    assert isinstance(restored.visualizer, StreamStartVisualizerDraftR1)
    assert restored.visualizer.batch_max == 8


def test_v1_branch_preferred_when_rate_max_present() -> None:
    """A payload carrying rate_max resolves to v1 even with overlapping fields."""
    restored = StreamStartPayload.from_dict({"visualizer": {"types": ["loudness"], "rate_max": 45}})
    assert isinstance(restored.visualizer, StreamStartVisualizer)
    assert restored.visualizer.rate_max == 45
