"""Integration tests for multi-player timing and group changes."""

from __future__ import annotations

import asyncio
import base64
import io
import math
from array import array
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import pytest

from aiosendspin.models.core import StreamClearMessage, StreamEndMessage, StreamStartMessage
from aiosendspin.models.player import (
    PLAYER_AUDIO_HEADER_SIZE,
    ClientHelloPlayerSupport,
    SupportedAudioFormat,
    pack_player_audio_header,
    unpack_player_audio_header,
)
from aiosendspin.models.types import AudioCodec, PlayerCommand, Roles
from aiosendspin.server.audio import AudioFormat
from aiosendspin.server.client import SendspinClient
from aiosendspin.server.clock import ManualClock
from aiosendspin.server.group import SendspinGroup
from aiosendspin.server.push_stream import PushStream
from aiosendspin.server.roles import AudioRequirements
from aiosendspin.server.roles.player.audio_transformers import FlacEncoder, PcmPassthrough
from tests.integration.sync_assertions import best_lag_samples


@dataclass(slots=True)
class _DummyServer:
    loop: asyncio.AbstractEventLoop
    clock: ManualClock
    id: str = "srv"
    name: str = "server"

    def _signal_client_connected(self, client_id: str) -> None:
        pass

    def _signal_client_disconnected(self, client_id: str, goodbye_reason: object = None) -> None:
        pass


EventKind = Literal["json", "bin"]


@dataclass(slots=True)
class _Event:
    kind: EventKind
    payload: object


class _CaptureConnection:
    """Capture connection that records JSON + binary messages in order."""

    def __init__(self) -> None:
        self.events: list[_Event] = []
        self.buffer_tracker = None

    async def disconnect(self, *, retry_connection: bool = True) -> None:  # noqa: ARG002
        return

    def send_message(self, message: object) -> None:
        self.events.append(_Event(kind="json", payload=message))

    def send_role_message(self, role: str, message: object) -> None:  # noqa: ARG002
        self.events.append(_Event(kind="json", payload=message))

    def send_binary(
        self,
        data: bytes,
        *,
        role: str,  # noqa: ARG002
        timestamp_us: int,
        message_type: int,  # noqa: ARG002
        buffer_end_time_us: int | None = None,
        buffer_byte_count: int | None = None,
        duration_us: int | None = None,
        player_audio_header: bool = False,
    ) -> bool:
        if player_audio_header:
            data = pack_player_audio_header(timestamp_us, 0) + data
        self.events.append(_Event(kind="bin", payload=data))
        if (
            self.buffer_tracker is not None
            and buffer_end_time_us is not None
            and buffer_byte_count is not None
        ):
            self.buffer_tracker.register(buffer_end_time_us, buffer_byte_count, duration_us or 0)
        return True


@dataclass(slots=True)
class _DecodedSegment:
    sample_rate: int
    channels: int
    start_timestamp_us: int
    pcm_s16le: bytes


@dataclass(slots=True)
class _EncodedSegment:
    codec: AudioCodec
    sample_rate: int
    channels: int
    start_timestamp_us: int
    codec_header_b64: str | None
    packets: list[bytes]


def _chirp(t: float, *, f0: float, k: float) -> float:
    """Frequency-swept sine with time-varying instantaneous frequency."""
    return math.sin(2.0 * math.pi * (f0 * t + 0.5 * k * t * t))


def _signal_left(t: float) -> float:
    """Deterministic continuous-time test signal (left channel)."""
    # Two chirps with different slopes; unique enough for correlation.
    return 0.55 * _chirp(t, f0=233.0, k=1137.0) + 0.35 * _chirp(t, f0=911.0, k=271.0)


def _pcm_s16le_stereo_for_range(
    start_timestamp_us: int,
    *,
    sample_rate: int,
    frame_count: int,
) -> bytes:
    """Generate deterministic stereo PCM for a given absolute time range."""
    out = array("h")
    out_extend = out.extend

    for i in range(frame_count):
        t = (start_timestamp_us + int(i * 1_000_000 / sample_rate)) / 1_000_000.0
        left = max(-1.0, min(1.0, _signal_left(t)))
        # Right channel uses a phase-shifted variant (still deterministic).
        right = max(-1.0, min(1.0, _signal_left(t + 0.0013)))
        out_extend((int(left * 32767.0), int(right * 32767.0)))

    return out.tobytes()


def _extract_left_channel_s16le(pcm_s16le: bytes, channels: int) -> list[int]:
    """Extract the left channel samples from packed s16le PCM bytes."""
    samples = array("h")
    samples.frombytes(pcm_s16le)
    return list(samples[0::channels])


def _make_player(
    server: _DummyServer,
    client_id: str,
    *,
    supported_formats: list[SupportedAudioFormat],
    buffer_capacity: int,
) -> tuple[SendspinClient, SendspinGroup, _CaptureConnection]:
    """Create a connected player with a preferred output format (first entry)."""
    client = SendspinClient(server, client_id=client_id)
    group = SendspinGroup(server, client)

    conn = _CaptureConnection()
    hello = type("Hello", (), {})()
    hello.client_id = client_id
    hello.name = client_id
    hello.player_support = ClientHelloPlayerSupport(
        supported_formats=supported_formats,
        buffer_capacity=buffer_capacity,
        supported_commands=[PlayerCommand.VOLUME, PlayerCommand.MUTE],
    )
    hello.artwork_support = None
    hello.visualizer_support = None

    client.attach_connection(
        conn,
        client_info=hello,
        negotiated_roles=[Roles.PLAYER.value],
        active_roles=[Roles.PLAYER.value],
    )
    client.mark_connected()
    role = client.role("player@v1")
    if role is not None:
        conn.buffer_tracker = role.get_buffer_tracker()

    # Set up audio requirements on the player role for hook-based streaming
    if role is not None and supported_formats:
        preferred_format = supported_formats[0]
        # Create transformer based on codec
        if preferred_format.codec == AudioCodec.FLAC:
            transformer = FlacEncoder(
                sample_rate=preferred_format.sample_rate,
                channels=preferred_format.channels,
                bit_depth=preferred_format.bit_depth,
            )
        else:
            transformer = PcmPassthrough(
                sample_rate=preferred_format.sample_rate,
                bit_depth=preferred_format.bit_depth,
                channels=preferred_format.channels,
            )
        role._audio_requirements = AudioRequirements(  # noqa: SLF001
            sample_rate=preferred_format.sample_rate,
            bit_depth=preferred_format.bit_depth,
            channels=preferred_format.channels,
            transformer=transformer,
        )

    return client, group, conn


def _decode_frames_to_pcm_s16le(
    frames: list[Any],
    *,
    sample_rate: int,
    channels: int,
) -> bytes:
    import av  # noqa: PLC0415

    layout = "stereo" if channels == 2 else "mono"
    resampler = av.AudioResampler(format="s16", layout=layout, rate=sample_rate)
    out = bytearray()
    for frame in frames:
        for out_frame in resampler.resample(frame):
            expected = out_frame.samples * channels * 2
            out.extend(bytes(out_frame.planes[0])[:expected])
    return bytes(out)


def _encoded_segments_from_events(events: list[_Event]) -> list[_EncodedSegment]:
    """Extract encoded segments (bounded by stream/clear and stream/end)."""
    segments: list[_EncodedSegment] = []
    current_start_msg: StreamStartMessage | None = None
    current_start_timestamp_us: int | None = None
    current_packets: list[bytes] = []

    def _flush() -> None:
        nonlocal current_start_msg, current_start_timestamp_us, current_packets
        if current_start_msg is None or current_start_timestamp_us is None or not current_packets:
            current_start_msg = None
            current_start_timestamp_us = None
            current_packets = []
            return

        player = current_start_msg.payload.player
        segments.append(
            _EncodedSegment(
                codec=player.codec,
                sample_rate=int(player.sample_rate),
                channels=int(player.channels),
                start_timestamp_us=current_start_timestamp_us,
                codec_header_b64=player.codec_header,
                packets=current_packets,
            )
        )
        current_start_msg = None
        current_start_timestamp_us = None
        current_packets = []

    for ev in events:
        if ev.kind == "json":
            msg = ev.payload
            if isinstance(msg, StreamStartMessage):
                _flush()
                current_start_msg = msg
                continue
            if isinstance(msg, StreamClearMessage | StreamEndMessage):
                _flush()
                continue
            continue

        data = ev.payload
        assert isinstance(data, (bytes, bytearray))
        header = unpack_player_audio_header(bytes(data))
        if current_start_msg is None:
            continue
        if current_start_timestamp_us is None:
            current_start_timestamp_us = header.timestamp_us
        current_packets.append(bytes(data)[PLAYER_AUDIO_HEADER_SIZE:])

    _flush()
    return segments


def _decode_flac_segment(seg: _EncodedSegment) -> bytes:
    header = base64.b64decode(seg.codec_header_b64) if seg.codec_header_b64 else b""
    bitstream = header + b"".join(seg.packets)
    import av  # noqa: PLC0415

    container = av.open(io.BytesIO(bitstream), format="flac")
    stream = container.streams.audio[0]
    decoded_frames: list[Any] = []
    for packet in container.demux(stream):
        decoded_frames.extend(packet.decode())
    return _decode_frames_to_pcm_s16le(
        decoded_frames,
        sample_rate=seg.sample_rate,
        channels=seg.channels,
    )


def _decode_segment(seg: _EncodedSegment) -> _DecodedSegment:
    if seg.codec == AudioCodec.PCM:
        pcm = b"".join(seg.packets)
    elif seg.codec == AudioCodec.FLAC:
        pcm = _decode_flac_segment(seg)
    else:
        pcm = b""
    return _DecodedSegment(
        sample_rate=seg.sample_rate,
        channels=seg.channels,
        start_timestamp_us=seg.start_timestamp_us,
        pcm_s16le=pcm,
    )


def _segments_from_events(events: list[_Event]) -> list[_DecodedSegment]:
    """Extract and decode segments (bounded by stream/clear and stream/end)."""
    return [_decode_segment(seg) for seg in _encoded_segments_from_events(events)]


def _choose_common_window(
    segments_by_player: list[list[_DecodedSegment]],
    *,
    window_duration_us: int,
    warmup_us: int,
) -> int:
    """Pick a start timestamp present in all players' segments."""
    starts: list[int] = []
    ends: list[int] = []

    for segments in segments_by_player:
        if not segments:
            raise AssertionError("expected at least one segment per player")
        seg = segments[-1]
        frame_count = len(seg.pcm_s16le) // (2 * seg.channels)
        dur_us = int(frame_count * 1_000_000 / seg.sample_rate)
        starts.append(seg.start_timestamp_us + warmup_us)
        ends.append(seg.start_timestamp_us + dur_us)

    start_us = max(starts)
    end_us = min(ends)
    if end_us - start_us < window_duration_us:
        raise AssertionError("not enough common audio coverage for window")
    return start_us


def _samples_for_window(
    seg: _DecodedSegment,
    window_start_us: int,
    window_duration_us: int,
) -> list[int]:
    """Extract left channel samples for a window based on timestamps."""
    frame_count_total = len(seg.pcm_s16le) // (2 * seg.channels)
    offset_frames = round((window_start_us - seg.start_timestamp_us) * seg.sample_rate / 1_000_000)
    window_frames = round(window_duration_us * seg.sample_rate / 1_000_000)
    offset_frames = max(0, min(offset_frames, frame_count_total))
    end_frames = max(0, min(offset_frames + window_frames, frame_count_total))

    # Extract raw frames from packed PCM.
    start_byte = offset_frames * seg.channels * 2
    end_byte = end_frames * seg.channels * 2
    return _extract_left_channel_s16le(seg.pcm_s16le[start_byte:end_byte], seg.channels)


def _expected_left_for_window(
    window_start_us: int,
    *,
    sample_rate: int,
    frame_count: int,
) -> list[int]:
    pcm = _pcm_s16le_stereo_for_range(
        window_start_us, sample_rate=sample_rate, frame_count=frame_count
    )
    return _extract_left_channel_s16le(pcm, 2)


def _first_audio_timestamp_after(
    events: list[_Event],
    *,
    start_index: int,
) -> int | None:
    for ev in events[start_index:]:
        if ev.kind != "bin":
            continue
        header = unpack_player_audio_header(ev.payload)  # type: ignore[arg-type]
        return header.timestamp_us
    return None


@dataclass(slots=True)
class _PendingJoin:
    player_id: str
    conn: _CaptureConnection
    start_index: int
    join_time_us: int


class _JoinTracker:
    def __init__(self, clock: ManualClock) -> None:
        self._clock = clock
        self._pending: list[_PendingJoin] = []
        self._join_delays_us: list[int] = []

    def track(self, player_id: str, conn: _CaptureConnection) -> None:
        self._pending.append(
            _PendingJoin(
                player_id=player_id,
                conn=conn,
                start_index=len(conn.events),
                join_time_us=self._clock.now_us(),
            )
        )

    def drain(self) -> None:
        still_pending: list[_PendingJoin] = []
        for join in self._pending:
            ts = _first_audio_timestamp_after(join.conn.events, start_index=join.start_index)
            if ts is None:
                still_pending.append(join)
                continue
            self._join_delays_us.append(ts - join.join_time_us)
        self._pending = still_pending

    def finalize(self) -> list[int]:
        self.drain()
        assert not self._pending
        return self._join_delays_us


async def _commit_pcm_block(
    stream: PushStream,
    *,
    play_start_us: int,
    source_format: AudioFormat,
    frame_count: int,
) -> int:
    pcm = _pcm_s16le_stereo_for_range(
        play_start_us, sample_rate=source_format.sample_rate, frame_count=frame_count
    )
    stream.prepare_audio(pcm, source_format)
    return await stream.commit_audio()


async def _add_client_and_track(
    group: SendspinGroup,
    *,
    player: SendspinClient,
    conn: _CaptureConnection,
    joins: _JoinTracker,
) -> None:
    if player in group.clients:
        return
    role = player.role("player@v1")
    if role is not None:
        role.get_join_delay_s = lambda: 0.0  # type: ignore[method-assign]
    joins.track(player.client_id, conn)
    await group.add_client(player)


def _assert_join_delays(
    join_delays_us: list[int],
    *,
    min_delay_us: int,
    max_delay_us: int,
) -> None:
    for delay_us in join_delays_us:
        assert min_delay_us <= delay_us <= max_delay_us


def _assert_three_player_sync_and_continuity(
    *,
    conn_a: _CaptureConnection,
    conn_b: _CaptureConnection,
    conn_c: _CaptureConnection,
    max_skew_us: int,
) -> None:
    seg_a = _segments_from_events(conn_a.events)
    seg_b = _segments_from_events(conn_b.events)
    seg_c = _segments_from_events(conn_c.events)
    assert seg_a
    assert seg_b
    assert seg_c

    window_duration_us = 500_000
    window_start_us = _choose_common_window(
        [seg_a, seg_b, seg_c], window_duration_us=window_duration_us, warmup_us=500_000
    )

    a_last = seg_a[-1]
    b_last = seg_b[-1]
    c_last = seg_c[-1]

    frames_a = round(window_duration_us * a_last.sample_rate / 1_000_000)
    frames_b = round(window_duration_us * b_last.sample_rate / 1_000_000)
    frames_c = round(window_duration_us * c_last.sample_rate / 1_000_000)

    rec_a = _samples_for_window(a_last, window_start_us, window_duration_us)
    rec_b = _samples_for_window(b_last, window_start_us, window_duration_us)
    rec_c = _samples_for_window(c_last, window_start_us, window_duration_us)

    exp_a = _expected_left_for_window(
        window_start_us, sample_rate=a_last.sample_rate, frame_count=frames_a
    )
    exp_b = _expected_left_for_window(
        window_start_us, sample_rate=b_last.sample_rate, frame_count=frames_b
    )
    exp_c = _expected_left_for_window(
        window_start_us, sample_rate=c_last.sample_rate, frame_count=frames_c
    )

    max_a = int(a_last.sample_rate * (max_skew_us / 1_000_000))
    max_b = int(b_last.sample_rate * (max_skew_us / 1_000_000))
    max_c = int(c_last.sample_rate * (max_skew_us / 1_000_000))

    lag_a, score_a = best_lag_samples(rec_a, exp_a, max_lag_samples=max_a)
    lag_b, score_b = best_lag_samples(rec_b, exp_b, max_lag_samples=max_b)
    lag_c, score_c = best_lag_samples(rec_c, exp_c, max_lag_samples=max_c)

    lag_a_us = abs(lag_a) * 1_000_000 / a_last.sample_rate
    lag_b_us = abs(lag_b) * 1_000_000 / b_last.sample_rate
    lag_c_us = abs(lag_c) * 1_000_000 / c_last.sample_rate
    assert lag_a_us <= max_skew_us
    assert lag_b_us <= max_skew_us
    assert lag_c_us <= max_skew_us
    assert max(lag_a_us, lag_b_us, lag_c_us) - min(lag_a_us, lag_b_us, lag_c_us) <= max_skew_us
    assert score_a >= 0.85
    assert score_b >= 0.85
    assert score_c >= 0.85


def _assert_pcm_chunks_continuous(events: list[_Event], *, max_gap_us: int) -> None:
    """Assert consecutive PCM audio chunks have continuous timestamps within a tolerance."""
    current_format: StreamStartMessage | None = None
    last_end_us: int | None = None
    for ev in events:
        if ev.kind == "json":
            msg = ev.payload
            if isinstance(msg, StreamStartMessage):
                current_format = msg
                last_end_us = None
            if isinstance(msg, StreamClearMessage | StreamEndMessage):
                current_format = None
                last_end_us = None
            continue

        if current_format is None or current_format.payload.player is None:
            continue
        fmt = current_format.payload.player
        if fmt.codec != AudioCodec.PCM:
            continue

        data = ev.payload
        assert isinstance(data, (bytes, bytearray))
        header = unpack_player_audio_header(bytes(data))
        payload = bytes(data)[PLAYER_AUDIO_HEADER_SIZE:]
        frame_count = len(payload) // (fmt.channels * 2)
        dur_us = int(frame_count * 1_000_000 / fmt.sample_rate)
        if last_end_us is not None:
            gap_us = header.timestamp_us - last_end_us
            assert 0 <= gap_us <= max_gap_us
        last_end_us = header.timestamp_us + dur_us


@pytest.mark.asyncio
async def test_multi_player_group_join_sync_stable_source() -> None:
    """Stable source: late joiner stays within +/- 5ms of the global clock."""
    loop = asyncio.get_running_loop()
    clock = ManualClock()
    server = _DummyServer(loop=loop, clock=clock)

    _player_a, group_a, conn_a = _make_player(
        server,
        "pA",
        supported_formats=[
            SupportedAudioFormat(codec=AudioCodec.PCM, channels=2, sample_rate=48_000, bit_depth=16)
        ],
        buffer_capacity=2_000_000,
    )
    player_b, _group_b, conn_b = _make_player(
        server,
        "pB",
        supported_formats=[
            SupportedAudioFormat(codec=AudioCodec.PCM, channels=2, sample_rate=32_000, bit_depth=16)
        ],
        buffer_capacity=60_000,
    )

    stream = group_a.start_stream()

    source_fmt = AudioFormat(sample_rate=48_000, bit_depth=16, channels=2)

    next_play_start_us = clock.now_us() + 1_000_000

    # Run 3 seconds virtual time; join B at t=1s.
    for i in range(120):  # 120 * 25ms = 3s
        if i == 40:
            await group_a.add_client(player_b)
        pcm = _pcm_s16le_stereo_for_range(
            next_play_start_us, sample_rate=source_fmt.sample_rate, frame_count=1200
        )
        stream.prepare_audio(pcm, source_fmt)
        play_start_us = await stream.commit_audio()
        assert abs(play_start_us - next_play_start_us) <= 1_000
        next_play_start_us = play_start_us + 25_000
        clock.advance_us(25_000)

    seg_a = _segments_from_events(conn_a.events)
    seg_b = _segments_from_events(conn_b.events)

    window_duration_us = 500_000  # 0.5s
    window_start_us = _choose_common_window(
        [seg_a, seg_b], window_duration_us=window_duration_us, warmup_us=250_000
    )

    a_last = seg_a[-1]
    b_last = seg_b[-1]
    frames_a = round(window_duration_us * a_last.sample_rate / 1_000_000)
    frames_b = round(window_duration_us * b_last.sample_rate / 1_000_000)

    received_a = _samples_for_window(a_last, window_start_us, window_duration_us)
    received_b = _samples_for_window(b_last, window_start_us, window_duration_us)

    expected_a = _expected_left_for_window(
        window_start_us, sample_rate=a_last.sample_rate, frame_count=frames_a
    )
    expected_b = _expected_left_for_window(
        window_start_us, sample_rate=b_last.sample_rate, frame_count=frames_b
    )

    lag_a, score_a = best_lag_samples(
        received_a, expected_a, max_lag_samples=int(a_last.sample_rate * 0.005)
    )
    lag_b, score_b = best_lag_samples(
        received_b, expected_b, max_lag_samples=int(b_last.sample_rate * 0.005)
    )

    lag_a_us = abs(lag_a) * 1_000_000 / a_last.sample_rate
    lag_b_us = abs(lag_b) * 1_000_000 / b_last.sample_rate
    assert lag_a_us <= 5_000
    assert lag_b_us <= 5_000
    assert abs(lag_a_us - lag_b_us) <= 5_000
    assert score_a >= 0.90
    assert score_b >= 0.90


@pytest.mark.asyncio
async def test_multi_player_sync_with_jittery_source_is_continuous() -> None:
    """Jittery chunk sizes: playback remains continuous and aligned."""
    loop = asyncio.get_running_loop()
    clock = ManualClock()
    server = _DummyServer(loop=loop, clock=clock)

    _player_a, group_a, conn_a = _make_player(
        server,
        "pA",
        supported_formats=[
            SupportedAudioFormat(codec=AudioCodec.PCM, channels=2, sample_rate=48_000, bit_depth=16)
        ],
        buffer_capacity=2_000_000,
    )
    player_b, _group_b, conn_b = _make_player(
        server,
        "pB",
        supported_formats=[
            SupportedAudioFormat(codec=AudioCodec.PCM, channels=2, sample_rate=32_000, bit_depth=16)
        ],
        buffer_capacity=60_000,
    )

    stream = group_a.start_stream()
    await group_a.add_client(player_b)

    source_fmt = AudioFormat(sample_rate=48_000, bit_depth=16, channels=2)

    next_play_start_us = clock.now_us() + 1_000_000

    # Alternate 20ms and 30ms blocks (still "continuous" overall).
    pattern_frames = [960, 1440]  # 20ms, 30ms at 48kHz
    for i in range(120):  # 120 commits, ~3 seconds
        frame_count = pattern_frames[i % 2]
        pcm = _pcm_s16le_stereo_for_range(
            next_play_start_us, sample_rate=source_fmt.sample_rate, frame_count=frame_count
        )
        stream.prepare_audio(pcm, source_fmt)
        play_start_us = await stream.commit_audio()
        assert abs(play_start_us - next_play_start_us) <= 1_000
        duration_us = int(frame_count * 1_000_000 / source_fmt.sample_rate)
        next_play_start_us = play_start_us + duration_us
        clock.advance_us(duration_us)

    seg_a = _segments_from_events(conn_a.events)
    seg_b = _segments_from_events(conn_b.events)

    window_duration_us = 500_000
    window_start_us = _choose_common_window(
        [seg_a, seg_b], window_duration_us=window_duration_us, warmup_us=500_000
    )

    a_last = seg_a[-1]
    b_last = seg_b[-1]
    frames_a = round(window_duration_us * a_last.sample_rate / 1_000_000)
    frames_b = round(window_duration_us * b_last.sample_rate / 1_000_000)

    received_a = _samples_for_window(a_last, window_start_us, window_duration_us)
    received_b = _samples_for_window(b_last, window_start_us, window_duration_us)
    expected_a = _expected_left_for_window(
        window_start_us, sample_rate=a_last.sample_rate, frame_count=frames_a
    )
    expected_b = _expected_left_for_window(
        window_start_us, sample_rate=b_last.sample_rate, frame_count=frames_b
    )

    lag_a, score_a = best_lag_samples(
        received_a, expected_a, max_lag_samples=int(a_last.sample_rate * 0.005)
    )
    lag_b, score_b = best_lag_samples(
        received_b, expected_b, max_lag_samples=int(b_last.sample_rate * 0.005)
    )

    lag_a_us = abs(lag_a) * 1_000_000 / a_last.sample_rate
    lag_b_us = abs(lag_b) * 1_000_000 / b_last.sample_rate
    assert lag_a_us <= 5_000
    assert lag_b_us <= 5_000
    assert abs(lag_a_us - lag_b_us) <= 5_000
    assert score_a >= 0.85
    assert score_b >= 0.85


@pytest.mark.asyncio
async def test_production_gap_rebases_timeline() -> None:
    """A long production gap should rebase timestamps so audio is not scheduled in the past."""
    loop = asyncio.get_running_loop()
    clock = ManualClock()
    server = _DummyServer(loop=loop, clock=clock)

    __player_a, group_a, conn_a = _make_player(
        server,
        "pA",
        supported_formats=[
            SupportedAudioFormat(codec=AudioCodec.PCM, channels=2, sample_rate=48_000, bit_depth=16)
        ],
        buffer_capacity=90_000,
    )

    stream = group_a.start_stream()
    source_fmt = AudioFormat(sample_rate=48_000, bit_depth=16, channels=2)

    next_play_start_us = clock.now_us() + 1_000_000

    # Produce 1s of audio.
    for _ in range(40):
        pcm = _pcm_s16le_stereo_for_range(
            next_play_start_us, sample_rate=source_fmt.sample_rate, frame_count=1200
        )
        stream.prepare_audio(pcm, source_fmt)
        play_start_us = await stream.commit_audio()
        assert abs(play_start_us - next_play_start_us) <= 1_000
        next_play_start_us = play_start_us + 25_000
        clock.advance_us(25_000)

    # Simulate a 2s gap with no audio production.
    events_before_gap = len(conn_a.events)
    clock.advance_us(2_000_000)
    resume_now_us = clock.now_us()

    pcm = _pcm_s16le_stereo_for_range(
        next_play_start_us, sample_rate=source_fmt.sample_rate, frame_count=1200
    )
    stream.prepare_audio(pcm, source_fmt)
    play_start_us = await stream.commit_audio()
    assert resume_now_us + 1_000_000 <= play_start_us <= resume_now_us + 1_100_000

    # Find the first audio chunk timestamp sent after the gap.
    first_after_gap_ts: int | None = None
    for ev in conn_a.events[events_before_gap:]:
        if ev.kind != "bin":
            continue
        header = unpack_player_audio_header(ev.payload)  # type: ignore[arg-type]
        first_after_gap_ts = header.timestamp_us
        break

    assert first_after_gap_ts is not None
    assert resume_now_us + 1_000_000 <= first_after_gap_ts <= resume_now_us + 1_100_000


@pytest.mark.asyncio
async def test_four_players_regroup_fast_start_and_sync() -> None:  # noqa: PLR0915
    """4 players (mixed formats): regroup/ungroup and stay synced within +/- 5ms."""
    loop = asyncio.get_running_loop()
    clock = ManualClock()
    server = _DummyServer(loop=loop, clock=clock)

    _player_a, group_a, conn_a = _make_player(
        server,
        "pA",
        supported_formats=[
            SupportedAudioFormat(codec=AudioCodec.PCM, channels=2, sample_rate=48_000, bit_depth=16)
        ],
        # Large buffer to simulate a stable/high-capacity device.
        buffer_capacity=2_000_000,
    )
    player_b, _group_b, conn_b = _make_player(
        server,
        "pB",
        supported_formats=[
            SupportedAudioFormat(
                codec=AudioCodec.FLAC, channels=2, sample_rate=44_100, bit_depth=16
            )
        ],
        # Compressed bytes; approximate ~500ms for typical FLAC frames in our test signal.
        buffer_capacity=60_000,
    )
    player_c, _group_c, conn_c = _make_player(
        server,
        "pC",
        supported_formats=[
            SupportedAudioFormat(codec=AudioCodec.PCM, channels=2, sample_rate=32_000, bit_depth=16)
        ],
        # ~500ms @ 32kHz stereo s16: 0.5 * 32_000 * 2ch * 2 bytes ≈ 64_000 bytes
        buffer_capacity=64_000,
    )
    player_d, _group_d, conn_d = _make_player(
        server,
        "pD",
        supported_formats=[
            SupportedAudioFormat(codec=AudioCodec.PCM, channels=2, sample_rate=32_000, bit_depth=16)
        ],
        buffer_capacity=64_000,
    )

    stream = group_a.start_stream()
    source_fmt = AudioFormat(sample_rate=48_000, bit_depth=16, channels=2)

    # 100ms per commit to ensure FLAC yields packets promptly.
    duration_us = 100_000
    frame_count = 4800  # 100ms @ 48kHz
    next_play_start_us = clock.now_us() + 1_000_000

    # Initial start latency (player A).
    first_idx_a = len(conn_a.events)
    play_start_us = await _commit_pcm_block(
        stream, play_start_us=next_play_start_us, source_format=source_fmt, frame_count=frame_count
    )
    assert abs(play_start_us - next_play_start_us) <= 1_000
    ts_a = _first_audio_timestamp_after(conn_a.events, start_index=first_idx_a)
    assert ts_a is not None
    assert 1_000_000 <= ts_a - clock.now_us() <= 1_100_000
    next_play_start_us = play_start_us + duration_us

    joins = _JoinTracker(clock)
    await _add_client_and_track(group_a, player=player_b, conn=conn_b, joins=joins)

    async def _safe_remove(client: SendspinClient) -> None:
        if len(group_a.clients) > 2:
            await group_a.remove_client(client)

    single_actions: dict[int, Callable[[], Awaitable[None]]] = {
        2: partial(_add_client_and_track, group_a, player=player_c, conn=conn_c, joins=joins),
        3: partial(_add_client_and_track, group_a, player=player_d, conn=conn_d, joins=joins),
        4: partial(_safe_remove, player_b),
        5: partial(_safe_remove, player_d),
        6: partial(_add_client_and_track, group_a, player=player_b, conn=conn_b, joins=joins),
        8: partial(_safe_remove, player_c),
        14: partial(_add_client_and_track, group_a, player=player_c, conn=conn_c, joins=joins),
        16: partial(_add_client_and_track, group_a, player=player_d, conn=conn_d, joins=joins),
    }
    burst_actions: dict[int, list[Callable[[], Awaitable[None]]]] = {
        10: [
            partial(_add_client_and_track, group_a, player=player_c, conn=conn_c, joins=joins),
            partial(_add_client_and_track, group_a, player=player_d, conn=conn_d, joins=joins),
            partial(_safe_remove, player_b),
        ],
        12: [
            partial(_safe_remove, player_c),
            partial(_add_client_and_track, group_a, player=player_b, conn=conn_b, joins=joins),
        ],
    }

    # Run long enough after the final regroup so all players have a shared window.
    for step in range(1, 40):
        clock.advance_us(duration_us)
        play_start_us = await _commit_pcm_block(
            stream,
            play_start_us=next_play_start_us,
            source_format=source_fmt,
            frame_count=frame_count,
        )
        assert abs(play_start_us - next_play_start_us) <= 1_000
        next_play_start_us = play_start_us + duration_us
        joins.drain()
        single_action = single_actions.get(step)
        if single_action is not None:
            await single_action()
        step_actions = burst_actions.get(step)
        if step_actions is not None:
            for action in step_actions:
                await action()

    join_delays_us = joins.finalize()
    _assert_join_delays(join_delays_us, min_delay_us=0, max_delay_us=1_000_000)
    _assert_three_player_sync_and_continuity(
        conn_a=conn_a,
        conn_b=conn_b,
        conn_c=conn_c,
        max_skew_us=5_000,
    )
    seg_a = _segments_from_events(conn_a.events)
    seg_b = _segments_from_events(conn_b.events)
    seg_c = _segments_from_events(conn_c.events)
    seg_d = _segments_from_events(conn_d.events)
    assert seg_d
    window_duration_us = 500_000
    window_start_us = _choose_common_window(
        [seg_a, seg_b, seg_c, seg_d],
        window_duration_us=window_duration_us,
        warmup_us=500_000,
    )
    d_last = seg_d[-1]
    frames_d = round(window_duration_us * d_last.sample_rate / 1_000_000)
    rec_d = _samples_for_window(d_last, window_start_us, window_duration_us)
    exp_d = _expected_left_for_window(
        window_start_us, sample_rate=d_last.sample_rate, frame_count=frames_d
    )
    max_d = int(d_last.sample_rate * (5_000 / 1_000_000))
    lag_d, score_d = best_lag_samples(rec_d, exp_d, max_lag_samples=max_d)
    lag_d_us = abs(lag_d) * 1_000_000 / d_last.sample_rate
    assert lag_d_us <= 5_000
    assert score_d >= 0.85
    _assert_pcm_chunks_continuous(conn_a.events, max_gap_us=6_000)
    _assert_pcm_chunks_continuous(conn_c.events, max_gap_us=6_000)


@pytest.mark.asyncio
async def test_first_time_join_unique_format_starts_under_1s_without_next_commit() -> None:
    """Late joiner with a unique format should start via PCM cache without waiting for commit."""
    loop = asyncio.get_running_loop()
    clock = ManualClock()
    server = _DummyServer(loop=loop, clock=clock)

    _player_a, group_a, _conn_a = _make_player(
        server,
        "pA",
        supported_formats=[
            SupportedAudioFormat(codec=AudioCodec.PCM, channels=2, sample_rate=48_000, bit_depth=16)
        ],
        # Large buffer to simulate a stable/high-capacity device.
        buffer_capacity=2_000_000,
    )
    player_b, _group_b, conn_b = _make_player(
        server,
        "pB",
        supported_formats=[
            SupportedAudioFormat(
                codec=AudioCodec.FLAC, channels=2, sample_rate=44_100, bit_depth=16
            )
        ],
        # Compressed bytes; approximate ~500ms for typical FLAC frames in our test signal.
        buffer_capacity=60_000,
    )

    stream = group_a.start_stream()
    source_fmt = AudioFormat(sample_rate=48_000, bit_depth=16, channels=2)

    # Build up future PCM cache for A only.
    duration_us = 100_000
    frame_count = 4800  # 100ms @ 48kHz
    next_play_start_us = clock.now_us() + 250_000
    lead_commits = 5  # keep +500ms of extra lead time before join
    for i in range(30):  # 3s of scheduled audio
        play_start_us = await _commit_pcm_block(
            stream,
            play_start_us=next_play_start_us,
            source_format=source_fmt,
            frame_count=frame_count,
        )
        next_play_start_us = play_start_us + duration_us
        # Join should not rely on a *future* commit, but it may rely on already-scheduled
        # buffered audio. Do not advance time after the last commit so the stream still
        # has ample future lead time relative to "now".
        if i < 29 - lead_commits:
            clock.advance_us(duration_us)

    # Join B, but do not commit any further audio afterwards.
    join_idx = len(conn_b.events)
    join_now_us = clock.now_us()
    role_b = player_b.role("player@v1")
    assert role_b is not None
    role_b.get_join_delay_s = lambda: 0.0  # type: ignore[method-assign]
    await group_a.add_client(player_b)

    first_ts = None
    for _ in range(100):
        first_ts = _first_audio_timestamp_after(conn_b.events, start_index=join_idx)
        if first_ts is not None:
            break
        await asyncio.sleep(0.01)
    assert first_ts is not None, "expected immediate catch-up audio from PCM cache"
    assert first_ts - join_now_us <= 1_000_000
