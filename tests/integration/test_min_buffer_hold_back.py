"""Players keep min_buffer queued although the resampler and encoder hold back output."""

from __future__ import annotations

import asyncio
import struct
from typing import Any

import numpy as np
import pytest

from aiosendspin.models.player import SupportedAudioFormat
from aiosendspin.models.types import AudioCodec
from aiosendspin.server import push_stream
from aiosendspin.server.audio import AudioFormat
from aiosendspin.server.clock import ManualClock
from tests.integration.sync_harness import CaptureConnection, DummyServer, make_player

_SOURCE_FORMAT = AudioFormat(sample_rate=48_000, bit_depth=16, channels=2)
_SLICE_US = 100_000
_SLICE_FRAMES = 4_800
_MIN_BUFFER_MS = 150
_HOLD_BACK_US = 40_000


class _HoldBackGraph:
    """Resample graph that keeps its newest output until more input arrives."""

    def __init__(self, graph: push_stream._AudioFilterGraph, sample_rate: int) -> None:
        self._graph = graph
        self._hold_frames = sample_rate * _HOLD_BACK_US // 1_000_000
        self._held: list[np.ndarray] = []
        self._held_frames = 0
        self._eof = False
        self._template: tuple[str, str, int] | None = None

    def push(self, frame: object) -> None:
        self._graph.push(frame)  # type: ignore[arg-type]
        self._eof = frame is None
        for out in push_stream._drain_audio_graph(self._graph):  # noqa: SLF001
            self._template = (out.format.name, out.layout.name, out.sample_rate)
            self._held.append(out.to_ndarray())
            self._held_frames += out.samples

    def pull(self) -> object:
        release = self._held_frames if self._eof else self._held_frames - self._hold_frames
        if release <= 0 or self._template is None:
            raise EOFError
        held = np.concatenate(self._held, axis=1)
        cut = release * (held.shape[1] // self._held_frames)
        self._held = [held[:, cut:]]
        self._held_frames -= release
        fmt, layout, rate = self._template
        frame = push_stream._get_av().AudioFrame.from_ndarray(  # noqa: SLF001
            held[:, :cut], format=fmt, layout=layout
        )
        frame.sample_rate = rate
        return frame


@pytest.fixture
def hold_back_resampler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make rate-converting graphs hold back a known amount of output on any FFmpeg build."""
    build = push_stream._build_resample_graph  # noqa: SLF001

    def build_held(**kwargs: Any) -> object:
        graph = build(**kwargs)
        if kwargs["source_sample_rate"] == kwargs["target_sample_rate"]:
            return graph
        return _HoldBackGraph(graph, kwargs["target_sample_rate"])

    monkeypatch.setattr(push_stream, "_build_resample_graph", build_held)


def _record_queued_before_send(conn: CaptureConnection, clock: ManualClock) -> list[int]:
    """Record how much audio the player has queued when each chunk is sent."""
    queued_us: list[int] = []
    send_binary = conn.send_binary

    def record(data: bytes, **kwargs: Any) -> bool:
        queued_us.append(kwargs["timestamp_us"] - clock.now_us())
        return send_binary(data, **kwargs)

    conn.send_binary = record  # type: ignore[method-assign]
    return queued_us


@pytest.mark.usefixtures("hold_back_resampler")
@pytest.mark.parametrize(
    ("codec", "rate"),
    [(AudioCodec.PCM, 44_100), (AudioCodec.FLAC, 48_000)],
    ids=["resampled-pcm", "flac"],
)
async def test_live_player_keeps_min_buffer(codec: AudioCodec, rate: int) -> None:
    """A player in a 48 kHz live stream never has less than min_buffer queued."""
    loop = asyncio.get_running_loop()
    clock = ManualClock()
    server = DummyServer(loop=loop, clock=clock)
    fmt = SupportedAudioFormat(codec=codec, channels=2, sample_rate=rate, bit_depth=16)
    client, group, conn = make_player(
        server, "player", supported_formats=[fmt], buffer_capacity=50_000_000
    )
    role = client.role("player@v1")
    assert role is not None
    role.min_buffer_ms = _MIN_BUFFER_MS
    queued_us = _record_queued_before_send(conn, clock)

    stream = group.start_stream()
    stream.set_live_source(True)
    pcm = struct.pack("<hh", 1000, -1000) * _SLICE_FRAMES
    for _ in range(30):
        stream.prepare_audio(pcm, _SOURCE_FORMAT)
        await stream.commit_audio()
        await asyncio.sleep(0)
        clock.advance_us(_SLICE_US)

    assert queued_us, "the player received no audio"
    assert min(queued_us) >= _MIN_BUFFER_MS * 1_000
