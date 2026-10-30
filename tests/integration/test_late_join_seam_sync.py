"""Late joiner alignment across the catch-up to live hand-off, without rebases."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from aiosendspin.models.types import AudioCodec
from aiosendspin.server import push_stream
from aiosendspin.server.audio import AudioFormat
from aiosendspin.server.clock import ManualClock
from aiosendspin.server.roles import AudioRequirements
from tests.integration.sync_assertions import (
    band_limited_noise,
    content_offset_us,
    jumps_us,
    player_timeline,
    wire_spans,
)
from tests.integration.sync_harness import DummyServer, channel_resolver_for, make_pool_player

_SOURCE_RATE = 48_000
_SOURCE_FORMAT = AudioFormat(
    sample_rate=_SOURCE_RATE, bit_depth=32, channels=2, sample_type="float"
)
_SLICE_FRAMES = 4_800
_SLICE_US = 100_000
_TOTAL_SLICES = 200
# One 44.1 kHz sample is 22.7 us.
_TOLERANCE_US = 50
# Output block size, so a 100 ms slice releases two or three blocks.
_OUTPUT_BLOCK_FRAMES = 1_589
_CONTENT = band_limited_noise(_SLICE_FRAMES * _TOTAL_SLICES, _SOURCE_RATE).astype(np.float32)


class _BlockOutputGraph:
    """Resample graph that releases output in whole blocks, like a block-based resampler."""

    def __init__(self, graph: push_stream._AudioFilterGraph) -> None:
        self._graph = graph
        self._held: list[np.ndarray] = []
        self._held_frames = 0
        self._eof = False
        self._template: tuple[str, str, int] | None = None

    def push(self, frame: object) -> None:
        self._graph.push(frame)  # type: ignore[arg-type]
        self._eof = frame is None
        for out in push_stream._drain_audio_graph(self._graph):  # noqa: SLF001
            self._template = (out.format.name, out.layout.name, out.sample_rate)
            samples = out.to_ndarray()
            self._held.append(samples)
            self._held_frames += out.samples

    def pull(self) -> object:
        release = self._held_frames
        if not self._eof:
            release -= release % _OUTPUT_BLOCK_FRAMES
        if release == 0 or self._template is None:
            raise EOFError
        held = np.concatenate(self._held, axis=1)
        channels = held.shape[1] // self._held_frames
        cut = release * channels
        self._held = [held[:, cut:]] if cut < held.shape[1] else []
        self._held_frames -= release
        fmt, layout, rate = self._template
        av = push_stream._get_av()  # noqa: SLF001
        frame = av.AudioFrame.from_ndarray(held[:, :cut], format=fmt, layout=layout)
        frame.sample_rate = rate
        return frame


@pytest.fixture
def block_output_resampler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make rate-converting graphs emit block-sized output on any FFmpeg build."""
    build = push_stream._build_resample_graph  # noqa: SLF001

    def build_blocky(**kwargs: object) -> object:
        graph = build(**kwargs)  # type: ignore[arg-type]
        if kwargs["source_sample_rate"] == kwargs["target_sample_rate"]:
            return graph
        return _BlockOutputGraph(graph)

    monkeypatch.setattr(push_stream, "_build_resample_graph", build_blocky)


@pytest.mark.usefixtures("block_output_resampler")
# The join slice sets where the catch-up graph's output ends relative to the live graph's.
@pytest.mark.parametrize(
    ("live", "lead_slices", "join_after_slices"),
    [(False, 100, 52), (False, 100, 53), (True, 0, 51)],
    ids=["buffered-catchup-ahead", "buffered-catchup-behind", "live-short-replay"],
)
async def test_late_joiner_stays_aligned_after_catchup_handoff(
    live: bool,  # noqa: FBT001
    lead_slices: int,
    join_after_slices: int,
) -> None:
    """A replayed joiner must continue its content without a jump when it switches to live."""
    loop = asyncio.get_running_loop()
    clock = ManualClock()
    server = DummyServer(loop=loop, clock=clock)
    reference_player, reference_conn = make_pool_player(
        server, "reference-44k", codec=AudioCodec.PCM, sample_rate=44_100
    )
    joiner, joiner_conn = make_pool_player(
        server, "flac-44k", codec=AudioCodec.FLAC, sample_rate=44_100
    )
    group = reference_player.group
    stream = group.start_stream(channel_resolver=channel_resolver_for({}))
    stream.set_live_source(live)
    reference_role = reference_player.role("player@v1")
    joiner_role = joiner.role("player@v1")
    assert reference_role is not None
    assert joiner_role is not None
    reference_role._audio_requirements = AudioRequirements(  # noqa: SLF001
        sample_rate=44_100, bit_depth=16, channels=2, transformer=None
    )
    # A send-ahead below the 1 s floor puts the live tail past the joiner's target.
    joiner_role.min_buffer_ms = 150

    async def commit(index: int, *, advance: bool) -> None:
        pcm = _CONTENT[index * _SLICE_FRAMES : (index + 1) * _SLICE_FRAMES]
        stream.prepare_audio(pcm.tobytes(), _SOURCE_FORMAT)
        await stream.commit_audio()
        await asyncio.sleep(0)
        if advance:
            clock.advance_us(_SLICE_US)

    join_index = lead_slices + join_after_slices
    handoff_us = 0
    for index in range(_TOTAL_SLICES):
        if index == join_index:
            handoff_us = max(stream._channel_timing.values())  # noqa: SLF001
            await group.add_client(joiner)
            assert stream._catchup_state, "joiner did not replay from the PCM cache"  # noqa: SLF001
        await commit(index, advance=index >= lead_slices)

    reference = player_timeline(reference_conn.events)
    joined = player_timeline(joiner_conn.events)
    before = content_offset_us(
        reference, joined, window_start_us=handoff_us - 600_000, window_us=500_000
    )
    after = content_offset_us(reference, joined, window_start_us=handoff_us + 2_000_000)
    assert abs(after) <= _TOLERANCE_US, (
        f"joiner vs reference after the catch-up hand-off: {after:+.1f}us "
        f"(during replay: {before:+.1f}us, tolerance {_TOLERANCE_US}us)"
    )
    assert jumps_us(wire_spans(joiner_conn.events)) == []
