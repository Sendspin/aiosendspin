"""Tests for the client-side SourceCapture wire behaviour."""

from __future__ import annotations

from typing import Any

import pytest

from aiosendspin.client.connection import SendspinConnection
from aiosendspin.client.source import SourceCapture
from aiosendspin.models.player import SupportedAudioFormat
from aiosendspin.models.types import AudioCodec
from tests.conftest import sine_pcm_16bit


class _FakeConnection:
    def __init__(self, *, synchronized: bool = True) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.chunks: list[tuple[int, bytes]] = []
        self.synchronized = synchronized
        self.source_stream_active = False

    async def send_client_stream_start(self, **kwargs: Any) -> None:
        self.calls.append(("start", kwargs))
        self.source_stream_active = True

    async def send_source_chunk(self, frame: bytes, *, timestamp_us: int) -> None:
        self.chunks.append((timestamp_us, frame))

    async def send_client_stream_end(self) -> None:
        self.calls.append(("end", None))
        self.source_stream_active = False

    def compute_source_timestamp(self, capture_timestamp_us: int) -> int:
        # Identity mapping keeps the timestamp arithmetic easy to assert.
        return capture_timestamp_us

    def is_time_synchronized(self) -> bool:
        return self.synchronized

    def is_source_stream_active(self) -> bool:
        return self.source_stream_active


class _FakeClient:
    def now_us(self) -> int:
        return 5_000_000


def _pcm_format() -> SupportedAudioFormat:
    return SupportedAudioFormat(codec=AudioCodec.PCM, channels=2, sample_rate=48000, bit_depth=16)


def test_opus_capture_rejects_non_16_bit_pcm() -> None:
    """Opus encodes s16 only, so any other declared depth would stream garbage."""
    fmt = SupportedAudioFormat(codec=AudioCodec.OPUS, channels=2, sample_rate=48000, bit_depth=24)
    with pytest.raises(ValueError, match="16-bit"):
        SourceCapture(_FakeClient(), _FakeConnection(), fmt)  # type: ignore[arg-type]


async def test_opus_timestamps_lead_capture_by_the_encoder_pre_skip() -> None:
    """Opus chunks are stamped earlier by the pre-skip, since the decoder emits it first."""
    pytest.importorskip("av")
    fmt = SupportedAudioFormat(codec=AudioCodec.OPUS, channels=2, sample_rate=48000, bit_depth=16)
    conn = _FakeConnection()
    capture = SourceCapture(_FakeClient(), conn, fmt)  # type: ignore[arg-type]
    await capture.start()
    await capture.feed(sine_pcm_16bit(48000), capture_timestamp_us=1_000_000)
    assert conn.chunks
    # 6.5ms of libopus pre-skip precedes the first captured sample.
    assert conn.chunks[0][0] == 1_000_000 - 6_500


async def test_start_announces_client_stream_only() -> None:
    """start() sends client_stream/start and nothing else (framing is the lifecycle)."""
    conn = _FakeConnection()
    capture = SourceCapture(_FakeClient(), conn, _pcm_format())  # type: ignore[arg-type]
    await capture.start()
    kinds = [c[0] for c in conn.calls]
    assert kinds == ["start"]
    assert conn.calls[0][1]["codec"] is AudioCodec.PCM
    assert conn.calls[0][1]["codec_header"] is None


async def test_feed_streams_pcm_with_monotonic_server_timestamps() -> None:
    """Fed PCM is chunked into type-12 frames whose timestamps advance by audio duration."""
    conn = _FakeConnection()
    capture = SourceCapture(_FakeClient(), conn, _pcm_format())  # type: ignore[arg-type]
    await capture.start()
    pcm = sine_pcm_16bit(48000)
    await capture.feed(pcm, capture_timestamp_us=1_000_000)

    timestamps = [ts for ts, _ in conn.chunks]
    assert timestamps[0] == 1_000_000
    assert timestamps == sorted(timestamps)
    # PCM passthrough is lossless: the streamed frames reconstruct the input.
    assert b"".join(frame for _, frame in conn.chunks) == pcm


async def test_each_frame_uses_its_capture_timestamp() -> None:
    """Each output frame starts at its supplied capture timestamp."""
    conn = _FakeConnection()
    capture = SourceCapture(_FakeClient(), conn, _pcm_format())  # type: ignore[arg-type]
    await capture.start()
    frame = sine_pcm_16bit(1200)

    await capture.feed(frame, capture_timestamp_us=1_000_000)
    await capture.feed(frame, capture_timestamp_us=2_000_000)

    assert [ts for ts, _ in conn.chunks] == [1_000_000, 2_000_000]


async def test_stop_ends_client_stream() -> None:
    """stop() ends the input stream (framing is the lifecycle, no client/state)."""
    conn = _FakeConnection()
    capture = SourceCapture(_FakeClient(), conn, _pcm_format())  # type: ignore[arg-type]
    await capture.start()
    await capture.stop()
    assert conn.calls[-1] == ("end", None)
    assert not any(kind == "state" for kind, _ in conn.calls)


async def test_feed_before_start_raises() -> None:
    """feed() requires an active stream."""
    conn = _FakeConnection()
    capture = SourceCapture(_FakeClient(), conn, _pcm_format())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="start"):
        await capture.feed(b"\x00\x00\x00\x00")


async def test_start_before_time_sync_raises() -> None:
    """Capture cannot start until time synchronization converges."""
    conn = _FakeConnection(synchronized=False)
    capture = SourceCapture(_FakeClient(), conn, _pcm_format())  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="synchronized"):
        await capture.start()


async def test_start_recovers_after_connection_ends_stream() -> None:
    """A capture can restart after its connection closes the wire stream."""
    conn = _FakeConnection()
    capture = SourceCapture(_FakeClient(), conn, _pcm_format())  # type: ignore[arg-type]
    await capture.start()
    conn.source_stream_active = False

    await capture.start()

    assert [kind for kind, _ in conn.calls] == ["start", "start"]


async def test_stop_discards_buffer_after_connection_ends_stream() -> None:
    """Stopping after a wire end clears buffered capture state."""
    conn = _FakeConnection()
    capture = SourceCapture(_FakeClient(), conn, _pcm_format())  # type: ignore[arg-type]
    await capture.start()
    await capture.feed(sine_pcm_16bit(1), capture_timestamp_us=1_000_000)
    conn.source_stream_active = False

    await capture.stop()

    with pytest.raises(RuntimeError, match="start"):
        await capture.feed(sine_pcm_16bit(1))
    await capture.start()
    await capture.feed(sine_pcm_16bit(1200), capture_timestamp_us=2_000_000)
    assert [timestamp for timestamp, _ in conn.chunks] == [2_000_000]


def test_compute_source_timestamp_excludes_output_delay() -> None:
    """Capture timestamps skip the output delay that playback conversion applies."""
    conn = SendspinConnection.__new__(SendspinConnection)
    conn._output_delay_us = 250_000  # noqa: SLF001

    class _IdentityFilter:
        def compute_server_time(self, client_time: int) -> int:
            return client_time

    conn._time_filter = _IdentityFilter()  # type: ignore[assignment]  # noqa: SLF001
    assert conn.compute_source_timestamp(1_000_000) == 1_000_000
    assert conn.compute_server_time(1_000_000) == 1_250_000
