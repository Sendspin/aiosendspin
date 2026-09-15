"""Focused tests for PushStream behavior with persistent SendspinClient objects."""

from __future__ import annotations

import asyncio
import sys
from collections import deque
from dataclasses import dataclass
from itertools import pairwise
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from aiosendspin.models import unpack_binary_header
from aiosendspin.models.core import (
    StreamClearMessage,
    StreamEndMessage,
    StreamRequestFormatPayload,
    StreamStartMessage,
)
from aiosendspin.models.player import (
    ClientHelloPlayerSupport,
    StreamRequestFormatPlayer,
    SupportedAudioFormat,
)
from aiosendspin.models.types import AudioCodec, PlayerCommand, Roles
from aiosendspin.server import push_stream as push_stream_module
from aiosendspin.server.audio import AudioFormat
from aiosendspin.server.audio_transformers import TransformerPool
from aiosendspin.server.channels import MAIN_CHANNEL
from aiosendspin.server.client import SendspinClient
from aiosendspin.server.clock import LoopClock, ManualClock
from aiosendspin.server.push_stream import (
    DEFAULT_INITIAL_DELAY_US,
    CachedChunk,
    CachedPCMChunk,
    PushStream,
)
from aiosendspin.server.roles import AudioChunk, AudioRequirements
from aiosendspin.server.roles.player.audio_transformers import FlacEncoder, PcmPassthrough


@dataclass(slots=True)
class _DummyServer:
    loop: Any
    clock: Any
    id: str = "srv"
    name: str = "server"

    def is_external_player(self, client_id: str) -> bool:  # noqa: ARG002
        return False

    def _signal_client_connected(self, client_id: str) -> None:
        pass

    def _signal_client_disconnected(self, client_id: str, goodbye_reason: object = None) -> None:
        pass


class _DummyGroup:
    def __init__(self, clients: list[SendspinClient]) -> None:
        self.clients = clients
        self.transformer_pool = TransformerPool()
        self._push_stream: PushStream | None = None
        self.has_active_stream = False

    def on_client_connected(self, client: SendspinClient) -> None:  # noqa: ARG002
        return

    def _register_client_events(self, client: SendspinClient) -> None:  # noqa: ARG002
        return

    def group_role(self, family: str) -> None:  # noqa: ARG002
        return None

    def get_channel_for_player(self, player_id: str) -> UUID:  # noqa: ARG002
        return MAIN_CHANNEL

    def on_role_format_changed(self, role: Any) -> None:
        if self._push_stream is not None and not self._push_stream.is_stopped:
            self._push_stream.on_role_format_changed(role)


class _FakeConnection:
    def __init__(self) -> None:
        self.sent_json: list[object] = []
        self.sent_binary: list[bytes] = []
        self.buffer_tracker = None
        self.dropped_pending_binary: list[list[str] | None] = []

    async def disconnect(self, *, retry_connection: bool = True) -> None:  # noqa: ARG002
        return

    def drop_pending_binary(self, roles: list[str] | None) -> None:
        self.dropped_pending_binary.append(roles)

    def send_message(self, message: object) -> None:
        self.sent_json.append(message)

    def send_role_message(self, role: str, message: object) -> None:  # noqa: ARG002
        self.sent_json.append(message)

    def send_binary(
        self,
        data: bytes,
        *,
        role: str,  # noqa: ARG002
        timestamp_us: int,  # noqa: ARG002
        message_type: int,  # noqa: ARG002
        buffer_end_time_us: int | None = None,
        buffer_byte_count: int | None = None,
        duration_us: int | None = None,
    ) -> bool:
        self.sent_binary.append(data)
        if (
            self.buffer_tracker is not None
            and buffer_end_time_us is not None
            and buffer_byte_count is not None
        ):
            self.buffer_tracker.register(buffer_end_time_us, buffer_byte_count, duration_us or 0)
        return True


class _DummyRole:
    def __init__(
        self,
        requirements: AudioRequirements,
        *,
        output_delay_us: int = 0,
        replay_from_pcm_cache: bool = False,
        required_lead_time_us: int = DEFAULT_INITIAL_DELAY_US,
        min_buffer_us: int = 0,
    ) -> None:
        self._requirements = requirements
        self._output_delay_us = output_delay_us
        self._replay_from_pcm_cache = replay_from_pcm_cache
        self._required_lead_time_us = required_lead_time_us
        self._min_buffer_us = min_buffer_us
        self.received: list[AudioChunk] = []
        self.started = 0

    def get_audio_requirements(self) -> AudioRequirements | None:
        return self._requirements

    def replay_from_pcm_cache(self) -> bool:
        return self._replay_from_pcm_cache

    def get_output_delay_us(self) -> int:
        return self._output_delay_us

    def get_required_lead_time_us(self) -> int:
        return self._required_lead_time_us

    def get_min_buffer_us(self) -> int:
        return self._min_buffer_us

    def get_join_delay_s(self) -> float:
        return 0.0

    def on_stream_start(self) -> None:
        self.started += 1

    def on_audio_chunk(self, chunk: AudioChunk) -> None:
        self.received.append(chunk)

    def on_stream_end(self) -> None:
        return

    def on_stream_clear(self) -> None:
        return


class _DummyClient:
    def __init__(self, roles: list[_DummyRole]) -> None:
        self.is_connected = True
        self.active_roles = roles
        self.connection = _FakeConnection()


def _expand_packed_s24_to_s32(data: bytes) -> bytes:
    """Expand packed s24 PCM to PyAV's left-aligned s32 representation."""
    if sys.byteorder == "little":
        return b"".join(b"\x00" + data[i : i + 3] for i in range(0, len(data), 3))
    return b"".join(data[i : i + 3] + b"\x00" for i in range(0, len(data), 3))


def _packed_s24_pcm_25ms() -> bytes:
    """Build one 25ms stereo PCM chunk with a stable packed-s24 byte pattern."""
    return bytes([0x11, 0x21, 0x31, 0x12, 0x22, 0x32]) * 1200


def _make_connected_player(
    mock_loop: Any,
    group: _DummyGroup,
    client_id: str,
    *,
    clock: Any | None = None,
) -> tuple[SendspinClient, _FakeConnection]:
    """Create a connected player client with a fake connection."""
    server_clock = clock or LoopClock(mock_loop)
    server = _DummyServer(loop=mock_loop, clock=server_clock)
    client = SendspinClient(server, client_id=client_id)
    client._group = group  # noqa: SLF001
    group.clients.append(client)

    conn = _FakeConnection()
    hello = type("Hello", (), {})()
    hello.client_id = client_id
    hello.name = client_id
    hello.player_support = ClientHelloPlayerSupport(
        supported_formats=[
            SupportedAudioFormat(
                codec=AudioCodec.PCM,
                channels=2,
                sample_rate=48000,
                bit_depth=16,
            ),
            SupportedAudioFormat(
                codec=AudioCodec.FLAC,
                channels=2,
                sample_rate=48000,
                bit_depth=16,
            ),
        ],
        buffer_capacity=200_000,
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
    if role is not None:
        transformer = group.transformer_pool.get_or_create(
            PcmPassthrough,
            channel_id=MAIN_CHANNEL.int,
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            frame_duration_us=25_000,
        )
        role._audio_requirements = AudioRequirements(  # noqa: SLF001
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=transformer,
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )

    return client, conn


@pytest.mark.asyncio
async def test_late_join_target_includes_player_output_delay() -> None:
    """Late-join target should use raw timestamps far enough ahead for delayed players."""
    loop = asyncio.get_running_loop()
    clock = ManualClock(now_us_value=1_000_000)
    group = _DummyGroup(clients=[])
    client, _ = _make_connected_player(loop, group, "p1", clock=clock)
    role = client.role("player@v1")
    assert role is not None
    role.output_delay_ms = 5_000

    stream = PushStream(loop=loop, clock=clock, group=group)

    # now + max(min_buffer=1000ms, required_lead=250ms) + output_delay=5s
    assert stream.get_late_join_target_timestamp_us(role=role) == 7_000_000


@pytest.mark.asyncio
async def test_live_send_ahead_uses_min_buffer_not_required_lead() -> None:
    """Live streams floor at min_buffer + static and do not extend toward required_lead."""
    loop = asyncio.get_running_loop()
    clock = ManualClock(now_us_value=1_000_000)
    group = _DummyGroup(clients=[])
    client, _ = _make_connected_player(loop, group, "p1", clock=clock)
    role = client.role("player@v1")
    assert role is not None
    role.output_delay_ms = 0
    role.min_buffer_ms = 200
    role.required_lead_time_ms = 400  # must be ignored for live

    stream = PushStream(loop=loop, clock=clock, group=group)
    stream.set_live_source(is_live=True)

    assert stream._role_send_ahead_us(role) == 200_000  # noqa: SLF001


@pytest.mark.asyncio
async def test_late_join_target_uses_required_lead_time() -> None:
    """Required lead time bumps the floor when it exceeds the static minimum."""
    loop = asyncio.get_running_loop()
    clock = ManualClock(now_us_value=1_000_000)
    group = _DummyGroup(clients=[])
    client, _ = _make_connected_player(loop, group, "p1", clock=clock)
    role = client.role("player@v1")
    assert role is not None
    role.output_delay_ms = 5_000
    role.required_lead_time_ms = 400
    role.min_buffer_ms = 100  # below required_lead, so required_lead is the binding floor

    stream = PushStream(loop=loop, clock=clock, group=group)

    # now + max(100ms min_buffer, 400ms required_lead) + 5s output_delay
    assert stream.get_late_join_target_timestamp_us(role=role) == 6_400_000


@pytest.mark.asyncio
async def test_late_join_target_matches_fresh_start_floor() -> None:
    """A rejoin gets the buffer horizon the client advertised, same as a fresh start.

    Anchoring late joins at required_lead_time alone left a regrouped or reconnected
    player with a fraction of the jitter headroom the same client gets on a fresh
    start, so ordinary network jitter pushed chunks past their play deadline.
    """
    loop = asyncio.get_running_loop()
    clock = ManualClock(now_us_value=1_000_000)
    group = _DummyGroup(clients=[])
    client, _ = _make_connected_player(loop, group, "p1", clock=clock)
    role = client.role("player@v1")
    assert role is not None
    role.output_delay_ms = 0
    role.required_lead_time_ms = 250
    role.min_buffer_ms = 1_000

    stream = PushStream(loop=loop, clock=clock, group=group)

    # now + max(1s min_buffer, 250ms required_lead)
    assert stream.get_late_join_target_timestamp_us(role=role) == 2_000_000


@pytest.mark.asyncio
async def test_non_main_join_rebase_includes_player_output_delay() -> None:
    """Solo-channel rejoin should clamp raw timing high enough for output delay."""
    loop = asyncio.get_running_loop()
    clock = ManualClock(now_us_value=1_000_000)
    group = _DummyGroup(clients=[])
    client, _ = _make_connected_player(loop, group, "p1", clock=clock)
    role = client.role("player@v1")
    assert role is not None
    role.output_delay_ms = 5_000

    channel_id = UUID("77777777-7777-7777-7777-777777777777")
    stream = PushStream(loop=loop, clock=clock, group=group)
    stream._channel_timing[channel_id] = clock.now_us() + 30_000_000  # noqa: SLF001
    group.clients = [client]

    stream._rebase_far_ahead_join_tail(channel_id, role)  # noqa: SLF001

    assert stream._channel_timing[channel_id] == 7_000_000  # noqa: SLF001


@pytest.mark.asyncio
async def test_commit_audio_sends_stream_start_and_binary(mock_loop: Any) -> None:
    """commit_audio sends stream/start and at least one binary audio chunk."""
    group = _DummyGroup(clients=[])
    client, conn = _make_connected_player(mock_loop, group, "p1")

    stream = PushStream(loop=mock_loop, clock=LoopClock(mock_loop), group=group)
    stream.prepare_audio(
        bytes(4800),
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
    )
    await stream.commit_audio()

    assert any(isinstance(m, StreamStartMessage) for m in conn.sent_json)
    assert conn.sent_binary, "expected at least one binary chunk"
    header = unpack_binary_header(conn.sent_binary[0])
    assert header.message_type == 4  # BinaryMessageType.AUDIO_CHUNK
    role = client.role("player@v1")
    assert role is not None
    buffer_tracker = role.get_buffer_tracker()
    assert buffer_tracker is not None
    assert buffer_tracker.buffered_bytes > 0


@pytest.mark.asyncio
async def test_commit_audio_float_input_quantizes_at_output_edge(
    mock_loop: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Float input should be quantized at the output edge for integer player requirements."""
    group = _DummyGroup(clients=[])
    _client, conn = _make_connected_player(mock_loop, group, "p1")

    quantize_calls = 0
    original_quantizer = push_stream_module._quantize_float_pcm  # noqa: SLF001

    def _counted_quantizer(**kwargs: Any) -> object:
        nonlocal quantize_calls
        quantize_calls += 1
        return original_quantizer(**kwargs)

    monkeypatch.setattr(push_stream_module, "_quantize_float_pcm", _counted_quantizer)

    stream = PushStream(loop=mock_loop, clock=LoopClock(mock_loop), group=group)
    stream.prepare_audio(
        bytes(9600),  # 25ms @ 48kHz stereo f32
        AudioFormat(sample_rate=48_000, bit_depth=32, channels=2, sample_type="float"),
    )
    await stream.commit_audio()

    assert quantize_calls > 0
    assert conn.sent_binary


def test_quantize_float_to_s16_uses_triangular_hp_dither(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Float to s16 quantization should request triangular_hp dithering."""
    captured_dither_methods: list[str | None] = []
    original_build_resample_graph = push_stream_module._build_resample_graph  # noqa: SLF001

    def _record_build_resample_graph(
        *,
        source_av_format: str,
        source_layout: str,
        source_sample_rate: int,
        target_av_format: str,
        target_layout: str,
        target_sample_rate: int,
        dither_method: str | None = None,
    ) -> object:
        if source_av_format == "flt" and target_av_format == "s16":
            captured_dither_methods.append(dither_method)
        return original_build_resample_graph(
            source_av_format=source_av_format,
            source_layout=source_layout,
            source_sample_rate=source_sample_rate,
            target_av_format=target_av_format,
            target_layout=target_layout,
            target_sample_rate=target_sample_rate,
            dither_method=dither_method,
        )

    monkeypatch.setattr(push_stream_module, "_build_resample_graph", _record_build_resample_graph)

    group = _DummyGroup(clients=[])
    stream = PushStream(loop=MagicMock(), clock=ManualClock(), group=group)
    push_stream_module._quantize_float_pcm(  # noqa: SLF001
        channel_id=MAIN_CHANNEL,
        pcm_data=bytes(9600),  # 25ms @ 48kHz stereo f32
        output_ts=0,
        sample_rate=48_000,
        channels=2,
        target_bit_depth=16,
        resampler_cache=stream._resamplers,  # noqa: SLF001
    )

    assert captured_dither_methods
    assert captured_dither_methods[-1] == "triangular_hp"


def test_quantize_float_to_s16_preserves_dither_on_resampler_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drift-triggered graph rebuild must preserve triangular_hp dithering."""
    captured_dither_methods: list[str | None] = []
    original_build_resample_graph = push_stream_module._build_resample_graph  # noqa: SLF001

    def _record_build_resample_graph(
        *,
        source_av_format: str,
        source_layout: str,
        source_sample_rate: int,
        target_av_format: str,
        target_layout: str,
        target_sample_rate: int,
        dither_method: str | None = None,
    ) -> object:
        if source_av_format == "flt" and target_av_format == "s16":
            captured_dither_methods.append(dither_method)
        return original_build_resample_graph(
            source_av_format=source_av_format,
            source_layout=source_layout,
            source_sample_rate=source_sample_rate,
            target_av_format=target_av_format,
            target_layout=target_layout,
            target_sample_rate=target_sample_rate,
            dither_method=dither_method,
        )

    monkeypatch.setattr(push_stream_module, "_build_resample_graph", _record_build_resample_graph)

    group = _DummyGroup(clients=[])
    stream = PushStream(loop=MagicMock(), clock=ManualClock(), group=group)
    # First call creates the quantizer graph. Second call with stale timestamp creates >20ms
    # drift, forcing graph rebuild in _resample_pcm_standalone.
    push_stream_module._quantize_float_pcm(  # noqa: SLF001
        channel_id=MAIN_CHANNEL,
        pcm_data=bytes(9600),  # 25ms @ 48kHz stereo f32
        output_ts=0,
        sample_rate=48_000,
        channels=2,
        target_bit_depth=16,
        resampler_cache=stream._resamplers,  # noqa: SLF001
    )
    push_stream_module._quantize_float_pcm(  # noqa: SLF001
        channel_id=MAIN_CHANNEL,
        pcm_data=bytes(9600),  # 25ms @ 48kHz stereo f32
        output_ts=0,
        sample_rate=48_000,
        channels=2,
        target_bit_depth=16,
        resampler_cache=stream._resamplers,  # noqa: SLF001
    )

    assert len(captured_dither_methods) >= 2
    assert captured_dither_methods[0] == "triangular_hp"
    assert captured_dither_methods[-1] == "triangular_hp"


def test_resampler_drift_detection_uses_input_timeline() -> None:
    """Drift detection must compare against an input-timeline reference.

    Long-FIR resamplers (e.g. soxr precision=30) emit fewer samples than the
    rate-conversion ratio until warmed up, so the output-side pending_timestamp_us
    naturally lags input_timestamp_us by tens of ms even during steady-state
    contiguous streaming. If drift detection compares that output-side cursor to
    the input timestamp, every call exceeds the 20 ms threshold and rebuilds the
    graph cold — which was the 5.1.0 regression that discarded ~35% of audio for
    float@48k → flac@44.1k/16 (the MA Sendspin provider's default input path).
    """
    source = AudioFormat(sample_rate=48_000, bit_depth=32, channels=2, sample_type="float")
    target = AudioFormat(sample_rate=44_100, bit_depth=32, channels=2, sample_type="float")
    key = push_stream_module._ResamplerKey(  # noqa: SLF001
        channel_id=MAIN_CHANNEL,
        source_format=source,
        target_sample_rate=44_100,
        target_channels=2,
        target_bit_depth=32,
        target_sample_type="float",
    )
    state = push_stream_module._create_resampler_state(key, source, target)  # noqa: SLF001

    chunk_samples = 4_800
    chunk_bytes = bytes(chunk_samples * 2 * 4)  # stereo f32
    input_duration_us = 100_000
    first_input_ts = 250_000
    input_ts = first_input_ts
    num_chunks = 20

    for _ in range(num_chunks):
        push_stream_module._resample_pcm_standalone(  # noqa: SLF001
            state, chunk_bytes, source, input_ts
        )
        input_ts += input_duration_us

    # The input-side cursor must advance by exactly the cumulative input duration,
    # independent of how many output samples the resampler has emitted so far.
    expected_pending_input_us = first_input_ts + num_chunks * input_duration_us
    assert state.pending_input_timestamp_us == expected_pending_input_us, (
        "pending_input_timestamp_us must track the input timeline — drift detection "
        "that uses pending_timestamp_us (which tracks output duration) mis-fires "
        "on every call for long-FIR resamplers."
    )

    # The next contiguous call must not be flagged as drift.
    drift_us = abs(state.pending_input_timestamp_us - input_ts)
    assert drift_us == 0, (
        f"Contiguous input stream should produce zero drift; got {drift_us}us. "
        "This is the invariant that prevents the graph from being rebuilt on "
        "every commit."
    )


@pytest.mark.asyncio
async def test_stop_sends_stream_end_and_resets_buffer_tracker(mock_loop: Any) -> None:
    """Stop sends stream/end and resets BufferTracker state."""
    group = _DummyGroup(clients=[])
    client, conn = _make_connected_player(mock_loop, group, "p1")

    stream = PushStream(loop=mock_loop, clock=LoopClock(mock_loop), group=group)
    stream.prepare_audio(
        bytes(4800),
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
    )
    await stream.commit_audio()
    role = client.role("player@v1")
    assert role is not None
    buffer_tracker = role.get_buffer_tracker()
    assert buffer_tracker is not None
    assert buffer_tracker.buffered_bytes > 0

    stream.stop()
    assert any(isinstance(m, StreamEndMessage) for m in conn.sent_json)
    assert buffer_tracker.buffered_bytes == 0


@pytest.mark.asyncio
async def test_stop_during_inflight_commit_suppresses_audio_delivery(mock_loop: Any) -> None:
    """Stopping during an in-flight commit should not deliver post-stop audio chunks."""
    group = _DummyGroup(clients=[])
    _, conn = _make_connected_player(mock_loop, group, "p1")
    stream = PushStream(loop=mock_loop, clock=LoopClock(mock_loop), group=group)

    entered_delivery = asyncio.Event()
    release_delivery = asyncio.Event()
    original_deliver = stream._deliver_audio_to_roles  # noqa: SLF001

    async def _gated_deliver(
        prepared: dict[UUID, tuple[bytes, AudioFormat]],
        channel_play_start: dict[UUID, int],
        *,
        commit_generation: int | None = None,
    ) -> dict[object, list[CachedChunk]]:
        entered_delivery.set()
        await release_delivery.wait()
        return await original_deliver(
            prepared,
            channel_play_start,
            commit_generation=commit_generation,
        )

    stream._deliver_audio_to_roles = _gated_deliver  # type: ignore[method-assign]  # noqa: SLF001

    stream.prepare_audio(
        bytes(4800),
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
    )
    commit_task = asyncio.create_task(stream.commit_audio())
    await asyncio.wait_for(entered_delivery.wait(), timeout=1.0)

    stream.stop()
    release_delivery.set()
    await commit_task

    assert any(isinstance(m, StreamEndMessage) for m in conn.sent_json)
    assert not conn.sent_binary

    stream_end_index = next(
        i for i, message in enumerate(conn.sent_json) if isinstance(message, StreamEndMessage)
    )
    post_end_messages = conn.sent_json[stream_end_index + 1 :]
    assert not any(isinstance(message, StreamStartMessage) for message in post_end_messages)


@pytest.mark.asyncio
async def test_role_leave_during_inflight_commit_suppresses_stale_audio(mock_loop: Any) -> None:
    """A role ended/removed during commit must not receive stale audio."""
    group = _DummyGroup(clients=[])
    client, conn = _make_connected_player(mock_loop, group, "p1")
    stream = PushStream(loop=mock_loop, clock=LoopClock(mock_loop), group=group)

    entered_transform = asyncio.Event()
    release_transform = asyncio.Event()
    original_transform = stream._transform_and_deliver  # noqa: SLF001

    async def _gated_transform(
        roles_by_pcm: dict[tuple[UUID, int, int, int], list[tuple[Any, Any, Any]]],
        resampled_pcm: dict[tuple[UUID, int, int, int], Any],
        *,
        commit_generation: int | None = None,
    ) -> dict[object, list[CachedChunk]]:
        entered_transform.set()
        await release_transform.wait()
        return await original_transform(
            roles_by_pcm,
            resampled_pcm,
            commit_generation=commit_generation,
        )

    stream._transform_and_deliver = _gated_transform  # type: ignore[method-assign]  # noqa: SLF001

    stream.prepare_audio(
        bytes(4800),
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
    )
    commit_task = asyncio.create_task(stream.commit_audio())
    await asyncio.wait_for(entered_transform.wait(), timeout=1.0)

    role = client.role("player@v1")
    assert role is not None
    role.on_stream_end()
    stream.on_role_leave(role)
    group.clients.remove(client)

    release_transform.set()
    await commit_task

    assert any(isinstance(m, StreamEndMessage) for m in conn.sent_json)
    assert not conn.sent_binary


@pytest.mark.asyncio
async def test_clear_sends_stream_clear(mock_loop: Any) -> None:
    """Clear sends stream/clear to connected players."""
    group = _DummyGroup(clients=[])
    _, conn = _make_connected_player(mock_loop, group, "p1")

    stream = PushStream(loop=mock_loop, clock=LoopClock(mock_loop), group=group)
    stream.prepare_audio(
        bytes(4800),
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
    )
    await stream.commit_audio()

    stream.clear()
    assert any(isinstance(m, StreamClearMessage) for m in conn.sent_json)


@pytest.mark.asyncio
async def test_stop_resets_transformers_without_delivering_tail_frames(mock_loop: Any) -> None:
    """stop() resets transformers but never delivers flushed tail frames.

    Flushed frames carry timestamp_us=0, which has no scheduling
    semantics, so emitting them would only pollute the client's
    buffer tracker without serving any playback purpose.
    """
    reset_calls = 0

    class FlushingTransformer:
        pending_timestamp_us: int | None = None

        @property
        def frame_duration_us(self) -> int:
            return 25_000

        def process(self, pcm: bytes, _ts: int, _dur: int) -> list[tuple[bytes, int]]:
            return [(pcm, 25_000)]

        def flush(self) -> list[tuple[bytes, int]]:
            return [(b"tail", 25_000)]

        def get_header(self) -> bytes | None:
            return None

        def reset(self) -> None:
            nonlocal reset_calls
            reset_calls += 1

    group = _DummyGroup(clients=[])
    role = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=FlushingTransformer(),
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role]))

    stream = PushStream(loop=mock_loop, clock=LoopClock(mock_loop), group=group)
    stream.stop()

    assert role.received == []
    assert reset_calls == 1


@pytest.mark.asyncio
async def test_clear_does_not_send_stream_end(mock_loop: Any) -> None:
    """Clear must not emit stream/end (spec forbids it during transitions)."""
    group = _DummyGroup(clients=[])
    _, conn = _make_connected_player(mock_loop, group, "p1")

    stream = PushStream(loop=mock_loop, clock=LoopClock(mock_loop), group=group)
    stream.prepare_audio(
        bytes(4800),
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
    )
    await stream.commit_audio()

    stream.clear()
    assert not any(isinstance(m, StreamEndMessage) for m in conn.sent_json)


@pytest.mark.asyncio
async def test_clear_bumps_stream_generation(mock_loop: Any) -> None:
    """clear() bumps stream_generation so in-flight commits bail out.

    Without the bump, role state machines that preserve _stream_started
    across stream/clear would deliver stale chunks from concurrent
    commit_audio() coroutines after the clear takes effect.
    """
    group = _DummyGroup(clients=[])
    _make_connected_player(mock_loop, group, "p1")

    stream = PushStream(loop=mock_loop, clock=LoopClock(mock_loop), group=group)
    before = stream._stream_generation  # noqa: SLF001
    stream.clear()
    assert stream._stream_generation == before + 1  # noqa: SLF001


@pytest.mark.asyncio
async def test_clear_during_inflight_commit_suppresses_audio_delivery(mock_loop: Any) -> None:
    """A clear() running during an in-flight commit must abort the delivery."""
    group = _DummyGroup(clients=[])
    _, conn = _make_connected_player(mock_loop, group, "p1")
    stream = PushStream(loop=mock_loop, clock=LoopClock(mock_loop), group=group)

    entered_delivery = asyncio.Event()
    release_delivery = asyncio.Event()
    original_deliver = stream._deliver_audio_to_roles  # noqa: SLF001

    async def _gated_deliver(
        prepared: dict[UUID, tuple[bytes, AudioFormat]],
        channel_play_start: dict[UUID, int],
        *,
        commit_generation: int | None = None,
    ) -> dict[object, list[CachedChunk]]:
        entered_delivery.set()
        await release_delivery.wait()
        return await original_deliver(
            prepared,
            channel_play_start,
            commit_generation=commit_generation,
        )

    stream._deliver_audio_to_roles = _gated_deliver  # type: ignore[method-assign]  # noqa: SLF001

    stream.prepare_audio(
        bytes(4800),
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
    )
    commit_task = asyncio.create_task(stream.commit_audio())
    await asyncio.wait_for(entered_delivery.wait(), timeout=1.0)

    stream.clear()
    release_delivery.set()
    await commit_task

    assert any(isinstance(m, StreamClearMessage) for m in conn.sent_json)
    assert not conn.sent_binary


@pytest.mark.asyncio
async def test_stop_with_keep_stream_suppresses_stream_end(mock_loop: Any) -> None:
    """stop(keep_stream=True) tears down transport without emitting stream/end."""
    group = _DummyGroup(clients=[])
    _, conn = _make_connected_player(mock_loop, group, "p1")

    stream = PushStream(loop=mock_loop, clock=LoopClock(mock_loop), group=group)
    stream.prepare_audio(
        bytes(4800),
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
    )
    await stream.commit_audio()

    stream.stop(keep_stream=True)

    assert stream.is_stopped
    assert not any(isinstance(m, StreamEndMessage) for m in conn.sent_json)


@pytest.mark.asyncio
async def test_transient_disconnect_keeps_role_in_audio_pipeline(mock_loop: Any) -> None:
    """Transient disconnect keeps role processing active, but transport send remains no-op."""
    group = _DummyGroup(clients=[])
    client, conn = _make_connected_player(mock_loop, group, "p1")
    stream = PushStream(loop=mock_loop, clock=LoopClock(mock_loop), group=group)

    role = client.role("player@v1")
    assert role is not None
    original_on_audio_chunk = role.on_audio_chunk
    on_audio_chunk_spy = MagicMock(side_effect=original_on_audio_chunk)
    role.on_audio_chunk = on_audio_chunk_spy  # type: ignore[method-assign]

    client.detach_connection(None)
    assert client.has_warm_disconnected_roles

    conn.sent_binary.clear()
    stream.prepare_audio(
        bytes(4800),
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
    )
    await stream.commit_audio()

    assert on_audio_chunk_spy.call_count > 0
    assert not conn.sent_binary


@pytest.mark.asyncio
async def test_on_role_join_sends_catchup_chunks(mock_loop: Any) -> None:
    """Late join via on_role_join triggers stream/start and cached audio catch-up."""
    group = _DummyGroup(clients=[])
    _, conn1 = _make_connected_player(mock_loop, group, "p1")
    stream = PushStream(loop=mock_loop, clock=LoopClock(mock_loop), group=group)

    stream.prepare_audio(
        bytes(4800),
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
    )
    await stream.commit_audio()
    assert conn1.sent_binary

    client2, conn2 = _make_connected_player(mock_loop, group, "p2")
    role2 = client2.role("player@v1")
    assert role2 is not None
    role2.get_join_delay_s = MagicMock(return_value=0.0)
    stream.on_role_join(role2)

    assert any(isinstance(m, StreamStartMessage) for m in conn2.sent_json)
    assert conn2.sent_binary, "expected catch-up binary chunks"


@pytest.mark.asyncio
async def test_pcm_cache_catchup_for_uncached_codec() -> None:
    """PCM cache should enable catch-up when TransformKey cache is empty."""

    class TransformerA:
        pending_timestamp_us: int | None = None

        @property
        def frame_duration_us(self) -> int:
            return 25_000

        def process(self, pcm: bytes, _ts: int, _dur: int) -> list[tuple[bytes, int]]:
            return [(pcm, 25_000)]

        def flush(self) -> list[tuple[bytes, int]]:
            return []

        def get_header(self) -> bytes | None:
            return None

        def reset(self) -> None:
            return

    class TransformerB(TransformerA):
        pass

    group = _DummyGroup(clients=[])
    role1 = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=TransformerA(),
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role1]))

    loop = asyncio.get_running_loop()
    stream = PushStream(
        loop=loop,
        clock=LoopClock(loop),
        group=group,
    )
    stream.prepare_audio(
        bytes(4800),
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
    )
    await stream.commit_audio()

    role2 = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=TransformerB(),
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role2]))
    stream.on_role_join(role2)
    stream.prepare_audio(
        bytes(4800),
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
    )
    await stream.commit_audio()
    for _ in range(50):
        if role2.received:
            break
        await asyncio.sleep(0.01)

    assert role2.started == 1
    assert role2.received


async def _drain_catchup_tasks(stream: PushStream) -> None:
    """Run the loop until every catch-up task has finished, re-raising any failure."""
    while pending := [t for t in stream._catchup_tasks.values() if not t.done()]:  # noqa: SLF001
        done, _ = await asyncio.wait(pending)
        for task in done:
            if not task.cancelled():
                # surface a crashed catch-up here instead of leaving the caller to fail
                # on a missing-audio assertion that says nothing about the real cause
                task.result()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_non_main_pcm_catchup_does_not_anchor_to_far_channel_tail() -> None:
    """Non-main PCM catch-up should start near now, not at a far-ahead channel tail."""

    class TransformerA:
        pending_timestamp_us: int | None = None

        @property
        def frame_duration_us(self) -> int:
            return 25_000

        def process(self, pcm: bytes, _ts: int, _dur: int) -> list[tuple[bytes, int]]:
            return [(pcm, 25_000)]

        def flush(self) -> list[tuple[bytes, int]]:
            return []

        def get_header(self) -> bytes | None:
            return None

        def reset(self) -> None:
            return

    class TransformerB(TransformerA):
        pass

    channel_id = UUID("77777777-7777-7777-7777-777777777777")
    group = _DummyGroup(clients=[])
    role1 = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=TransformerA(),
            channel_id=channel_id,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role1]))

    loop = asyncio.get_running_loop()
    clock = ManualClock()
    stream = PushStream(loop=loop, clock=clock, group=group)
    stream.enable_pcm_cache_for_channel(channel_id)

    now_us = clock.now_us()
    frame_duration_us = 25_000

    # Simulate ~31s of cached PCM spanning near-now through far future.
    # Reconnect catch-up should start near now, not at the channel tail.
    pcm_chunks = deque[CachedPCMChunk]()
    for i in range(1240):
        ts = now_us - 100_000 + i * frame_duration_us
        pcm_chunks.append(
            CachedPCMChunk(
                timestamp_us=ts,
                duration_us=frame_duration_us,
                pcm_data=bytes(4800),
                sample_rate=48000,
                bit_depth=16,
                channels=2,
            )
        )
    stream._pcm_chunk_cache[channel_id.int] = pcm_chunks  # noqa: SLF001
    stream._channel_timing[channel_id] = now_us + 30_000_000  # noqa: SLF001

    role2 = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=TransformerB(),
            channel_id=channel_id,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role2]))

    stream.on_role_join(role2)
    await _drain_catchup_tasks(stream)

    assert role2.started == 1
    assert role2.received
    assert role2.received[0].timestamp_us - now_us < 500_000


async def _setup_deep_buffer_catchup_join() -> tuple[PushStream, _DummyRole, int]:
    """Build a deep-buffer main-channel catch-up join and return (stream, joiner, tail).

    A live role drives an established 24-to-16-bit resampler, the PCM cache
    spans near-now through a far-ahead tail, and the joiner has just joined
    with its own TransformKey so its catch-up task is created but not yet run.
    """

    class TransformerA:
        pending_timestamp_us: int | None = None

        @property
        def frame_duration_us(self) -> int:
            return 25_000

        def process(self, pcm: bytes, _ts: int, _dur: int) -> list[tuple[bytes, int]]:
            return [(pcm, 25_000)]

        def flush(self) -> list[tuple[bytes, int]]:
            return []

        def get_header(self) -> bytes | None:
            return None

        def reset(self) -> None:
            return

    class TransformerB(TransformerA):
        pass

    group = _DummyGroup(clients=[])
    role1 = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=TransformerA(),
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role1]))

    loop = asyncio.get_running_loop()
    clock = ManualClock()
    stream = PushStream(loop=loop, clock=clock, group=group)

    stream.prepare_audio(
        bytes(7200),  # 25ms @ 48kHz stereo 24-bit
        AudioFormat(sample_rate=48000, bit_depth=24, channels=2),
    )
    await stream.commit_audio()

    now_us = clock.now_us()
    frame_duration_us = 25_000
    chunk_count = 1240

    pcm_chunks = deque[CachedPCMChunk]()
    for i in range(chunk_count):
        ts = now_us - 100_000 + i * frame_duration_us
        pcm_chunks.append(
            CachedPCMChunk(
                timestamp_us=ts,
                duration_us=frame_duration_us,
                pcm_data=bytes(7200),
                sample_rate=48000,
                bit_depth=24,
                channels=2,
            )
        )
    tail_us = now_us - 100_000 + chunk_count * frame_duration_us
    stream._pcm_chunk_cache[MAIN_CHANNEL.int] = pcm_chunks  # noqa: SLF001
    stream._channel_timing[MAIN_CHANNEL] = tail_us  # noqa: SLF001
    stream._channel_timing_residue[MAIN_CHANNEL] = 0  # noqa: SLF001

    role2 = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=TransformerB(),
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role2]))

    stream.on_role_join(role2)
    return stream, role2, tail_us


@pytest.mark.asyncio
async def test_catchup_handoff_commit_race_does_not_overlap() -> None:
    """A commit racing the catch-up hand-off must not deliver duplicate audio.

    The commit caches its PCM, then yields so the catch-up task consumes it
    through the shared encoder. The resumed commit must not feed the same
    samples to that encoder again.
    """
    stream, role2, _tail_us = await _setup_deep_buffer_catchup_join()

    # Commit immediately so the hand-off happens mid-commit.
    stream.prepare_audio(
        bytes(7200),
        AudioFormat(sample_rate=48000, bit_depth=24, channels=2),
    )
    await stream.commit_audio()
    await _drain_catchup_tasks(stream)

    received = sorted(role2.received, key=lambda c: c.timestamp_us)
    assert received
    for prev, nxt in pairwise(received):
        assert nxt.timestamp_us >= prev.timestamp_us + prev.duration_us


@pytest.mark.asyncio
async def test_catchup_handoff_delivers_contiguous_audio() -> None:
    """Audio across the catch-up hand-off has no gaps through later commits."""
    stream, role2, tail_us = await _setup_deep_buffer_catchup_join()
    await _drain_catchup_tasks(stream)

    for _ in range(3):
        stream.prepare_audio(
            bytes(7200),
            AudioFormat(sample_rate=48000, bit_depth=24, channels=2),
        )
        await stream.commit_audio()

    received = sorted(role2.received, key=lambda c: c.timestamp_us)
    assert received
    for prev, nxt in pairwise(received):
        assert nxt.timestamp_us == prev.timestamp_us + prev.duration_us
    last = received[-1]
    assert last.timestamp_us + last.duration_us == tail_us + 3 * 25_000


@pytest.mark.asyncio
async def test_late_joiner_shares_group_timeline() -> None:
    """A late joiner must land on the same absolute timeline as the existing member.

    Every device schedules playback off the absolute timestamp in each chunk, so
    two players are in sync only if identical audio reaches them with identical
    timestamps. This drives a normal (near-now) group: role1 plays, role2 joins
    mid-stream through the PCM-cache catch-up path, then more live audio commits.

    Invariant: every timestamp role2 receives is also a timestamp role1 receives
    (a contiguous suffix of the shared timeline), with no anchor offset. A failure
    here would reproduce the reported "out of sync after grouping" symptom.
    """

    class TransformerA:
        pending_timestamp_us: int | None = None

        @property
        def frame_duration_us(self) -> int:
            return 25_000

        def process(self, pcm: bytes, _ts: int, _dur: int) -> list[tuple[bytes, int]]:
            return [(pcm, 25_000)]

        def flush(self) -> list[tuple[bytes, int]]:
            return []

        def get_header(self) -> bytes | None:
            return None

        def reset(self) -> None:
            return

    class TransformerB(TransformerA):
        pass

    group = _DummyGroup(clients=[])
    role1 = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=TransformerA(),
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role1]))

    loop = asyncio.get_running_loop()
    clock = ManualClock()
    stream = PushStream(loop=loop, clock=clock, group=group)

    def commit_one() -> None:
        stream.prepare_audio(
            bytes(7200),  # 25ms @ 48kHz stereo 24-bit
            AudioFormat(sample_rate=48000, bit_depth=24, channels=2),
        )

    # role1 plays two live chunks alone; both are cached as PCM for catch-up.
    commit_one()
    await stream.commit_audio()
    commit_one()
    await stream.commit_audio()

    # role2 joins mid-stream. replay_from_pcm_cache forces the catch-up path
    # (its own TransformKey, distinct transformer) rather than skipping replay.
    role2 = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=TransformerB(),
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        ),
        replay_from_pcm_cache=True,
    )
    group.clients.append(_DummyClient([role2]))
    stream.on_role_join(role2)

    # Two more live chunks land after the join.
    commit_one()
    await stream.commit_audio()
    commit_one()
    await stream.commit_audio()

    await _drain_catchup_tasks(stream)

    assert role2.started >= 1
    assert role2.received, "joiner was stranded with no audio"

    role2_ts = sorted(c.timestamp_us for c in role2.received)
    role1_ts = {c.timestamp_us for c in role1.received}

    # role2's timeline is contiguous (no gap that would glitch playback).
    for prev, nxt in pairwise(sorted(role2.received, key=lambda c: c.timestamp_us)):
        assert nxt.timestamp_us == prev.timestamp_us + prev.duration_us

    # Core sync invariant: role2 sits on role1's exact timeline, no offset.
    assert set(role2_ts) <= role1_ts, (
        f"joiner desynced from group: role2={role2_ts} not a subset of role1={sorted(role1_ts)}"
    )


@pytest.mark.asyncio
async def test_late_joiner_on_own_channel_shares_timeline_after_stall() -> None:
    """A joiner on its own channel must share the timeline even after a production stall.

    Members that need per-player audio processing get their own channel, so the
    join introduces a channel with no timing yet. If audio production has stalled
    meanwhile, the next commit rebases the timeline up to the send-ahead floor;
    the new channel must land on the existing member's timeline rather than one
    rebase beyond it.
    """
    joiner_channel = UUID("2a4f6e8c-0b1d-4f3a-9c5e-7d8f0a1b2c3d")
    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)

    group = _DummyGroup(clients=[])
    role1 = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=None,
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role1]))

    loop = asyncio.get_running_loop()
    clock = ManualClock()
    stream = PushStream(loop=loop, clock=clock, group=group)

    # The existing member plays alone, establishing the shared timeline.
    stream.prepare_audio(bytes(4800), fmt)  # 25ms @ 48kHz stereo 16-bit
    await stream.commit_audio()

    # Production stalls: the clock runs past the audio already scheduled, so the
    # timeline sits below now + send-ahead and the next commit has to rebase it.
    clock.advance_us(DEFAULT_INITIAL_DELAY_US + 200_000)

    role2 = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=None,
            channel_id=joiner_channel,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role2]))
    stream.on_role_join(role2)

    # First commit carrying the joiner's channel.
    role1_already_received = len(role1.received)
    stream.prepare_audio(bytes(4800), fmt)
    stream.prepare_audio(bytes(4800), fmt, channel_id=joiner_channel)
    await stream.commit_audio()

    role1_joint = role1.received[role1_already_received:]
    assert role1_joint, "existing member received nothing from the joint commit"
    assert role2.received, "joiner was stranded with no audio"

    role1_start_us = min(chunk.timestamp_us for chunk in role1_joint)
    role2_start_us = min(chunk.timestamp_us for chunk in role2.received)

    # Core sync invariant: the joiner's channel sits on the existing timeline.
    assert role2_start_us == role1_start_us, (
        f"joiner desynced by {role2_start_us - role1_start_us}us from the existing member"
    )
    # The rebase must still leave that shared timeline in the future.
    assert role1_start_us >= clock.now_us()


@pytest.mark.asyncio
async def test_main_join_with_established_resampler_backfills_near_now() -> None:
    """Deeply-buffered main-channel join must backfill near now, not inherit the tail.

    When a live role already drives a non-passthrough resampler at the joiner's
    target shape and the channel tail sits far ahead (deep producer buffer), the
    join must still replay buffered PCM from the playhead instead of skipping to
    live audio that does not arrive until the tail.
    """

    class TransformerA:
        pending_timestamp_us: int | None = None

        @property
        def frame_duration_us(self) -> int:
            return 25_000

        def process(self, pcm: bytes, _ts: int, _dur: int) -> list[tuple[bytes, int]]:
            return [(pcm, 25_000)]

        def flush(self) -> list[tuple[bytes, int]]:
            return []

        def get_header(self) -> bytes | None:
            return None

        def reset(self) -> None:
            return

    class TransformerB(TransformerA):
        pass

    group = _DummyGroup(clients=[])
    role1 = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=TransformerA(),
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role1]))

    loop = asyncio.get_running_loop()
    clock = ManualClock()
    stream = PushStream(loop=loop, clock=clock, group=group)

    # Commit one 24-bit chunk so role1 establishes a non-passthrough resampler
    # (source 24-bit to target 16-bit) at the joiner's target shape.
    stream.prepare_audio(
        bytes(7200),  # 25ms @ 48kHz stereo 24-bit
        AudioFormat(sample_rate=48000, bit_depth=24, channels=2),
    )
    await stream.commit_audio()
    role1_req = role1.get_audio_requirements()
    assert role1_req is not None
    assert stream._has_established_resampler_for(role1_req, MAIN_CHANNEL)  # noqa: SLF001

    now_us = clock.now_us()
    frame_duration_us = 25_000

    # ~31s of cached 24-bit PCM spanning near-now into the future, channel tail
    # parked 30s ahead to model a deep buffered source.
    pcm_chunks = deque[CachedPCMChunk]()
    for i in range(1240):
        ts = now_us - 100_000 + i * frame_duration_us
        pcm_chunks.append(
            CachedPCMChunk(
                timestamp_us=ts,
                duration_us=frame_duration_us,
                pcm_data=bytes(7200),
                sample_rate=48000,
                bit_depth=24,
                channels=2,
            )
        )
    stream._pcm_chunk_cache[MAIN_CHANNEL.int] = pcm_chunks  # noqa: SLF001
    stream._channel_timing[MAIN_CHANNEL] = now_us + 30_000_000  # noqa: SLF001

    role2 = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=TransformerB(),
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role2]))

    stream.on_role_join(role2)
    await _drain_catchup_tasks(stream)

    assert role2.started == 1
    assert role2.received
    assert role2.received[0].timestamp_us - now_us < 500_000


@pytest.mark.asyncio
async def test_non_main_join_without_cache_rebases_far_ahead_tail() -> None:
    """When no catch-up cache exists, non-main rejoin should not wait at a far channel tail."""

    class Transformer:
        pending_timestamp_us: int | None = None

        @property
        def frame_duration_us(self) -> int:
            return 25_000

        def process(self, pcm: bytes, _ts: int, _dur: int) -> list[tuple[bytes, int]]:
            return [(pcm, 25_000)]

        def flush(self) -> list[tuple[bytes, int]]:
            return []

        def get_header(self) -> bytes | None:
            return None

        def reset(self) -> None:
            return

    channel_id = UUID("66666666-6666-6666-6666-666666666666")
    role = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=Transformer(),
            channel_id=channel_id,
            frame_duration_us=25_000,
        )
    )
    group = _DummyGroup(clients=[_DummyClient([role])])

    loop = asyncio.get_running_loop()
    clock = ManualClock()
    stream = PushStream(loop=loop, clock=clock, group=group)

    now_us = clock.now_us()
    stream._channel_timing[channel_id] = now_us + 30_000_000  # noqa: SLF001

    stream.on_role_join(role)
    stream.prepare_audio(
        bytes(4800),
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
        channel_id=channel_id,
    )
    await stream.commit_audio()

    assert role.started == 1
    assert role.received
    assert role.received[0].timestamp_us - now_us < 500_000


@pytest.mark.asyncio
async def test_non_main_join_with_committed_timeline_does_not_rebase_backward() -> None:
    """Cache-miss rejoin must keep committed dedicated channels on shared timeline."""
    channel_id = UUID("55555555-5555-5555-5555-555555555555")
    role_main = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=None,
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    role_join = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=None,
            channel_id=channel_id,
            frame_duration_us=25_000,
        )
    )
    group = _DummyGroup(clients=[_DummyClient([role_main]), _DummyClient([role_join])])

    loop = asyncio.get_running_loop()
    clock = ManualClock(now_us_value=1_000_000)
    stream = PushStream(loop=loop, clock=clock, group=group)
    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)

    # Simulate an established shared timeline where both channels already committed audio.
    shared_start_us = clock.now_us() + 500_000
    stream._channel_timing[MAIN_CHANNEL] = shared_start_us  # noqa: SLF001
    stream._channel_timing[channel_id] = shared_start_us  # noqa: SLF001
    stream._channels_with_committed_audio.add(MAIN_CHANNEL)  # noqa: SLF001
    stream._channels_with_committed_audio.add(channel_id)  # noqa: SLF001

    stream.on_role_join(role_join)

    # Rejoin must not rewind dedicated-channel timing toward now.
    assert stream._channel_timing[channel_id] == shared_start_us  # noqa: SLF001

    stream.prepare_audio(bytes(4800), fmt, channel_id=MAIN_CHANNEL)
    stream.prepare_audio(bytes(4800), fmt, channel_id=channel_id)
    await stream.commit_audio()

    assert role_main.received
    assert role_join.received
    assert role_join.started == 1
    assert role_join.received[0].timestamp_us == role_main.received[0].timestamp_us


@pytest.mark.asyncio
async def test_transform_dedup_uses_transform_key_not_instance(mock_loop: Any) -> None:
    """Transformer dedupe should be based on TransformKey, not instance id."""

    class CountingTransformer:
        calls = 0
        pending_timestamp_us: int | None = None

        def __init__(self) -> None:
            self._frame_duration_us = 25_000

        @property
        def frame_duration_us(self) -> int:
            return self._frame_duration_us

        def process(self, pcm: bytes, _ts: int, _dur: int) -> list[tuple[bytes, int]]:
            CountingTransformer.calls += 1
            return [(pcm, self._frame_duration_us)]

        def flush(self) -> list[tuple[bytes, int]]:
            return []

        def get_header(self) -> bytes | None:
            return None

        def reset(self) -> None:
            return

    CountingTransformer.calls = 0
    group = _DummyGroup(clients=[])
    role1 = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=CountingTransformer(),
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    role2 = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=CountingTransformer(),
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    group.clients.extend([_DummyClient([role1]), _DummyClient([role2])])

    stream = PushStream(loop=mock_loop, clock=LoopClock(mock_loop), group=group)
    stream.prepare_audio(
        bytes(4800),
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
    )
    await stream.commit_audio()

    assert CountingTransformer.calls == 1


@pytest.mark.asyncio
async def test_transform_key_separates_frame_duration(mock_loop: Any) -> None:
    """Different frame_duration_us should not share transformer work."""

    class CountingTransformer:
        calls = 0
        pending_timestamp_us: int | None = None

        def __init__(self, frame_duration_us: int) -> None:
            self._frame_duration_us = frame_duration_us

        @property
        def frame_duration_us(self) -> int:
            return self._frame_duration_us

        def process(self, pcm: bytes, _ts: int, _dur: int) -> list[tuple[bytes, int]]:
            CountingTransformer.calls += 1
            return [(pcm, self._frame_duration_us)]

        def flush(self) -> list[tuple[bytes, int]]:
            return []

        def get_header(self) -> bytes | None:
            return None

        def reset(self) -> None:
            return

    CountingTransformer.calls = 0
    group = _DummyGroup(clients=[])
    role1 = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=CountingTransformer(25_000),
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    role2 = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=CountingTransformer(50_000),
            channel_id=MAIN_CHANNEL,
            frame_duration_us=50_000,
        )
    )
    group.clients.extend([_DummyClient([role1]), _DummyClient([role2])])

    stream = PushStream(loop=mock_loop, clock=LoopClock(mock_loop), group=group)
    stream.prepare_audio(
        bytes(4800),
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
    )
    await stream.commit_audio()

    assert CountingTransformer.calls == 2


@pytest.mark.asyncio
async def test_long_gap_reset_is_handled_in_push_stream() -> None:
    """PushStream resets transformer state after long production gaps."""

    class ResetTrackingTransformer:
        pending_timestamp_us: int | None = None

        def __init__(self) -> None:
            self.reset_calls = 0

        @property
        def frame_duration_us(self) -> int:
            return 25_000

        def process(self, pcm: bytes, _ts: int, _dur: int) -> list[tuple[bytes, int]]:
            return [(pcm, 25_000)]

        def flush(self) -> list[tuple[bytes, int]]:
            return []

        def get_header(self) -> bytes | None:
            return None

        def reset(self) -> None:
            self.reset_calls += 1

    transformer = ResetTrackingTransformer()
    group = _DummyGroup(clients=[])
    role = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=transformer,
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role]))

    loop = asyncio.get_running_loop()
    clock = ManualClock()
    stream = PushStream(loop=loop, clock=clock, group=group)
    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)

    stream.prepare_audio(bytes(4800), fmt)
    await stream.commit_audio()
    assert transformer.reset_calls == 0

    clock.advance_us(2_000_000)
    stream.prepare_audio(bytes(4800), fmt)
    await stream.commit_audio()
    assert transformer.reset_calls == 1


@pytest.mark.asyncio
async def test_medium_gap_does_not_reset_transformer() -> None:
    """PushStream does not reset transformer state for medium gaps."""

    class ResetTrackingTransformer:
        pending_timestamp_us: int | None = None

        def __init__(self) -> None:
            self.reset_calls = 0

        @property
        def frame_duration_us(self) -> int:
            return 25_000

        def process(self, pcm: bytes, _ts: int, _dur: int) -> list[tuple[bytes, int]]:
            return [(pcm, 25_000)]

        def flush(self) -> list[tuple[bytes, int]]:
            return []

        def get_header(self) -> bytes | None:
            return None

        def reset(self) -> None:
            self.reset_calls += 1

    transformer = ResetTrackingTransformer()
    group = _DummyGroup(clients=[])
    role = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=transformer,
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role]))

    loop = asyncio.get_running_loop()
    clock = ManualClock()
    stream = PushStream(loop=loop, clock=clock, group=group)
    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)

    stream.prepare_audio(bytes(4800), fmt)
    await stream.commit_audio()
    assert transformer.reset_calls == 0

    clock.advance_us(500_000)
    stream.prepare_audio(bytes(4800), fmt)
    await stream.commit_audio()
    assert transformer.reset_calls == 0


@pytest.mark.asyncio
async def test_late_join_uses_cached_chunks_across_role_recreation(mock_loop: Any) -> None:
    """Late join uses cache even if transformer instance changes."""

    class PassTransformer:
        pending_timestamp_us: int | None = None

        @property
        def frame_duration_us(self) -> int:
            return 25_000

        def process(self, pcm: bytes, _ts: int, _dur: int) -> list[tuple[bytes, int]]:
            return [(pcm, 25_000)]

        def flush(self) -> list[tuple[bytes, int]]:
            return []

        def get_header(self) -> bytes | None:
            return None

        def reset(self) -> None:
            return

    group = _DummyGroup(clients=[])
    role1 = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=PassTransformer(),
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    client1 = _DummyClient([role1])
    group.clients.append(client1)

    stream = PushStream(loop=mock_loop, clock=LoopClock(mock_loop), group=group)
    stream.prepare_audio(
        bytes(4800),
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
    )
    await stream.commit_audio()
    assert role1.received

    role2 = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=PassTransformer(),
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    stream.on_role_join(role2)

    assert role2.started == 1
    assert role2.received


@pytest.mark.asyncio
async def test_send_cached_chunks_keeps_chunk_overlapping_now(mock_loop: Any) -> None:
    """Cached replay should keep/send chunks that overlap now, not only future chunks."""
    group = _DummyGroup(clients=[])
    role = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2),
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role]))
    stream = PushStream(loop=mock_loop, clock=LoopClock(mock_loop), group=group)

    now_us = stream._clock.now_us()  # noqa: SLF001
    overlapping = CachedChunk(
        timestamp_us=now_us - 500_000,
        duration_us=1_000_000,
        payload=b"a",
        byte_count=1,
    )
    future = CachedChunk(
        timestamp_us=now_us + 100_000,
        duration_us=25_000,
        payload=b"b",
        byte_count=1,
    )

    stream._send_cached_chunks_to_role(  # noqa: SLF001
        role, [overlapping, future], now_us
    )
    assert len(role.received) == 2
    assert role.received[0].timestamp_us == overlapping.timestamp_us


@pytest.mark.asyncio
async def test_transform_key_separates_channels(mock_loop: Any) -> None:
    """TransformKey includes channel_id to avoid cross-channel sharing."""

    class CountingTransformer:
        calls = 0
        pending_timestamp_us: int | None = None

        def __init__(self) -> None:
            self._frame_duration_us = 25_000

        @property
        def frame_duration_us(self) -> int:
            return self._frame_duration_us

        def process(self, pcm: bytes, _ts: int, _dur: int) -> list[tuple[bytes, int]]:
            CountingTransformer.calls += 1
            return [(pcm, self._frame_duration_us)]

        def flush(self) -> list[tuple[bytes, int]]:
            return []

        def get_header(self) -> bytes | None:
            return None

        def reset(self) -> None:
            return

    CountingTransformer.calls = 0
    group = _DummyGroup(clients=[])
    other_channel = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    role1 = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=CountingTransformer(),
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    role2 = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=CountingTransformer(),
            channel_id=other_channel,
            frame_duration_us=25_000,
        )
    )
    group.clients.extend([_DummyClient([role1]), _DummyClient([role2])])

    stream = PushStream(loop=mock_loop, clock=LoopClock(mock_loop), group=group)
    stream.prepare_audio(
        bytes(4800),
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
        channel_id=MAIN_CHANNEL,
    )
    stream.prepare_audio(
        bytes(4800),
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
        channel_id=other_channel,
    )
    await stream.commit_audio()

    assert CountingTransformer.calls == 2


def _make_connected_player_multi_format(
    mock_loop: Any,
    group: _DummyGroup,
    client_id: str,
) -> tuple[SendspinClient, _FakeConnection]:
    """Create a connected player client that supports PCM 48kHz and PCM 44.1kHz."""
    server = _DummyServer(loop=mock_loop, clock=LoopClock(mock_loop))
    client = SendspinClient(server, client_id=client_id)
    client._group = group  # noqa: SLF001
    group.clients.append(client)

    conn = _FakeConnection()
    hello = type("Hello", (), {})()
    hello.client_id = client_id
    hello.name = client_id
    hello.player_support = ClientHelloPlayerSupport(
        supported_formats=[
            SupportedAudioFormat(
                codec=AudioCodec.PCM,
                channels=2,
                sample_rate=48000,
                bit_depth=16,
            ),
            SupportedAudioFormat(
                codec=AudioCodec.PCM,
                channels=2,
                sample_rate=44100,
                bit_depth=16,
            ),
        ],
        buffer_capacity=200_000,
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

    return client, conn


@pytest.mark.asyncio
async def test_format_change_during_active_stream(mock_loop: Any) -> None:
    """Mid-stream format change sends stream/start (deferred) with no stream/clear.

    Full PushStream flow:
    1. Create player with PCM 48kHz, start PushStream
    2. Commit audio N times
    3. Trigger format change via on_stream_request_format during active playback
    4. Commit more audio
    5. Assert: StreamStartMessage (with new format) in sent_json, NO StreamClearMessage
    6. Binary audio continues after format change
    7. Post-change audio resumes near now via new-format catch-up replay,
       not at the pre-change tail
    """
    group = _DummyGroup(clients=[])
    client, conn = _make_connected_player_multi_format(mock_loop, group, "p1")
    clock = LoopClock(mock_loop)

    stream = PushStream(loop=mock_loop, clock=clock, group=group)
    group._push_stream = stream  # noqa: SLF001
    group.has_active_stream = True

    # Commit enough 48kHz PCM that the channel tail runs well past the
    # resume floor (25ms per chunk).
    for _ in range(20):
        stream.prepare_audio(
            bytes(4800),
            AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
        )
        await stream.commit_audio()

    pre_change_binary_count = len(conn.sent_binary)
    assert pre_change_binary_count > 0

    # Record the last pre-change chunk's end timestamp
    last_pre_header = unpack_binary_header(conn.sent_binary[-1])
    # Duration of a 4800-byte PCM chunk at 48kHz stereo 16bit = 25ms = 25000us
    pre_change_end_us = last_pre_header.timestamp_us + 25_000

    # Clear sent_json to isolate format change messages
    conn.sent_json.clear()

    # Trigger mid-stream format change: PCM 48kHz -> PCM 44.1kHz
    request = StreamRequestFormatPayload(
        player=StreamRequestFormatPlayer(
            codec=AudioCodec.PCM,
            sample_rate=44100,
            channels=2,
            bit_depth=16,
        )
    )
    role = client.role("player@v1")
    assert role is not None
    role.on_stream_request_format(request)

    # No stream/clear. The stream/start may already have gone out with the
    # first new-format catch-up chunk re-encoded by the join path.
    assert not any(isinstance(msg, StreamClearMessage) for msg in conn.sent_json)

    # Let the boundary-scheduled catch-up task deliver its replay chunks.
    for _ in range(5):
        await asyncio.sleep(0)

    # Commit audio at the new format (44.1kHz)
    # 1102 samples * 2 bytes * 2 channels = 4408 bytes (~24.99ms)
    stream.prepare_audio(
        bytes(4408),
        AudioFormat(sample_rate=44100, bit_depth=16, channels=2),
    )
    await stream.commit_audio()

    # Stream/start should now be sent (deferred until first chunk)
    stream_starts = [msg for msg in conn.sent_json if isinstance(msg, StreamStartMessage)]
    assert len(stream_starts) == 1
    start_msg = stream_starts[0]
    assert start_msg.payload.player is not None
    assert start_msg.payload.player.sample_rate == 44100
    assert start_msg.payload.player.codec == AudioCodec.PCM

    # No stream/clear should have been sent
    assert not any(isinstance(msg, StreamClearMessage) for msg in conn.sent_json)

    # The change must evict queued old-format binary, as stream/clear does.
    assert conn.dropped_pending_binary == [["player"]]

    # Binary audio continued after the format change
    assert len(conn.sent_binary) > pre_change_binary_count

    # The client flushes its un-played buffer at the announcement, so the
    # replacement audio must resume near now, not continue at the pre-change tail.
    post_change_binary = conn.sent_binary[pre_change_binary_count:]
    first_post_header = unpack_binary_header(post_change_binary[0])
    now_us = clock.now_us()
    assert now_us <= first_post_header.timestamp_us < pre_change_end_us, (
        f"Post-change audio starts at {first_post_header.timestamp_us}us "
        f"(now={now_us}us, pre-change tail={pre_change_end_us}us)"
    )

    # No old-format frames (9-byte header + 4800-byte 48kHz payload) may
    # follow the boundary; replayed catch-up chunks are resampled to 44.1kHz
    # so their exact size can vary slightly.
    assert all(len(frame) != 9 + 4800 for frame in post_change_binary)


@pytest.mark.asyncio
async def test_format_flipflop_without_a_chunk_announces_the_return(mock_loop: Any) -> None:
    """Returning to the last announced format must still announce the boundary.

    Two requests with no chunk between them leave the client configured for
    the format it is already playing, so the sent-format guard suppresses the
    second stream/start. The boundary has meanwhile dropped the client's
    queued audio and re-anchored near the playhead, so without the
    announcement the client never flushes and plays the old buffer over the
    replacement audio.
    """
    group = _DummyGroup(clients=[])
    client, conn = _make_connected_player_multi_format(mock_loop, group, "p1")
    clock = LoopClock(mock_loop)

    stream = PushStream(loop=mock_loop, clock=clock, group=group)
    group._push_stream = stream  # noqa: SLF001
    group.has_active_stream = True

    role = client.role("player@v1")
    assert role is not None

    def request_format(sample_rate: int) -> None:
        role.on_stream_request_format(
            StreamRequestFormatPayload(
                player=StreamRequestFormatPlayer(
                    codec=AudioCodec.PCM,
                    sample_rate=sample_rate,
                    channels=2,
                    bit_depth=16,
                )
            )
        )

    for _ in range(20):
        stream.prepare_audio(
            bytes(4800),
            AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
        )
        await stream.commit_audio()

    pre_change_count = len(conn.sent_binary)
    pre_change_end_us = unpack_binary_header(conn.sent_binary[-1]).timestamp_us + 25_000
    conn.sent_json.clear()

    request_format(44100)
    request_format(48000)
    for _ in range(5):
        await asyncio.sleep(0)

    stream.prepare_audio(
        bytes(4800),
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
    )
    await stream.commit_audio()

    post_change_binary = conn.sent_binary[pre_change_count:]
    assert post_change_binary
    first_post_header = unpack_binary_header(post_change_binary[0])
    assert first_post_header.timestamp_us < pre_change_end_us

    stream_starts = [msg for msg in conn.sent_json if isinstance(msg, StreamStartMessage)]
    assert len(stream_starts) == 1, (
        f"Audio re-anchored to {first_post_header.timestamp_us}us behind the "
        f"{pre_change_end_us}us tail with {len(stream_starts)} stream/start(s)"
    )
    assert stream_starts[0].payload.player is not None
    assert stream_starts[0].payload.player.sample_rate == 48000


@pytest.mark.asyncio
async def test_format_change_during_inflight_commit_aborts_old_format_delivery(
    mock_loop: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A format request landing mid-commit must not deliver old-format audio.

    commit_audio() groups roles under the current AudioRequirements before
    its first await, so a request processed during resampling previously let
    the commit deliver old-format chunks behind a stream/start built from the
    NEW requirements. The boundary synchronously defers the role to the join
    path, which excludes it from that commit's live delivery.
    """
    group = _DummyGroup(clients=[])
    client, conn = _make_connected_player_multi_format(mock_loop, group, "p1")
    clock = LoopClock(mock_loop)

    stream = PushStream(loop=mock_loop, clock=clock, group=group)
    group._push_stream = stream  # noqa: SLF001
    group.has_active_stream = True

    # Establish the stream at 48kHz PCM
    for _ in range(2):
        stream.prepare_audio(
            bytes(4800),
            AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
        )
        await stream.commit_audio()
    assert conn.sent_binary

    role = client.role("player@v1")
    assert role is not None
    old_tkey = stream._build_transform_key(  # noqa: SLF001
        role.get_audio_requirements(), MAIN_CHANNEL, role
    )
    conn.sent_json.clear()
    conn.sent_binary.clear()

    # Land the format request after the commit has grouped roles under the
    # old requirements, but before encoding and delivery.
    stream.prepare_audio(
        bytes(4800),
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
    )
    original_resample = stream._resample_for_roles  # noqa: SLF001
    format_request_fired = False

    async def resample_with_midflight_format_change(*args: Any, **kwargs: Any) -> Any:
        nonlocal format_request_fired
        if not format_request_fired:
            format_request_fired = True
            role.on_stream_request_format(
                StreamRequestFormatPayload(
                    player=StreamRequestFormatPlayer(
                        codec=AudioCodec.PCM,
                        sample_rate=44100,
                        channels=2,
                        bit_depth=16,
                    )
                )
            )
        return await original_resample(*args, **kwargs)

    monkeypatch.setattr(stream, "_resample_for_roles", resample_with_midflight_format_change)
    await stream.commit_audio()

    # The boundary defers the rejoin until the in-flight commit's cleanup, so
    # let the catch-up task deliver its new-format replay chunks.
    for _ in range(5):
        await asyncio.sleep(0)

    # The changer gets only new-format replay from the boundary-scheduled
    # join; the excluded commit must not have delivered old-format frames.
    assert conn.sent_binary
    assert all(len(frame) != 9 + 4800 for frame in conn.sent_binary)

    # Nor may the excluded commit repopulate the evicted old-format cache.
    assert not stream._role_chunk_cache.get(old_tkey)  # noqa: SLF001

    # The next commit delivers the deferred stream/start, then only new-format audio.
    stream.prepare_audio(
        bytes(4408),
        AudioFormat(sample_rate=44100, bit_depth=16, channels=2),
    )
    await stream.commit_audio()

    stream_starts = [msg for msg in conn.sent_json if isinstance(msg, StreamStartMessage)]
    assert len(stream_starts) == 1
    assert stream_starts[0].payload.player is not None
    assert stream_starts[0].payload.player.sample_rate == 44100
    assert conn.sent_binary
    assert all(len(frame) != 9 + 4800 for frame in conn.sent_binary)


@pytest.mark.asyncio
async def test_format_change_preserves_peer_audio(
    mock_loop: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One player's mid-commit format change must not cost peers any audio.

    Regression for the stream-global generation bump: aborting the whole
    in-flight commit dropped its window for every player on the stream.
    The pending-join exclusion must contain the change to the requesting
    player.
    """
    group = _DummyGroup(clients=[])
    client_a, conn_a = _make_connected_player_multi_format(mock_loop, group, "p1")
    _client_b, conn_b = _make_connected_player_multi_format(mock_loop, group, "p2")
    clock = LoopClock(mock_loop)

    stream = PushStream(loop=mock_loop, clock=clock, group=group)
    group._push_stream = stream  # noqa: SLF001
    group.has_active_stream = True

    # Both players receive the same 48kHz PCM windows.
    for _ in range(3):
        stream.prepare_audio(
            bytes(4800),
            AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
        )
        await stream.commit_audio()
    peer_chunks_before = len(conn_b.sent_binary)
    assert peer_chunks_before > 0
    # B's own stream/start from the initial start is expected; ignore it here.
    conn_b.sent_json.clear()

    # Player A's format request lands after the commit grouped roles, but
    # before encoding and delivery.
    role_a = client_a.role("player@v1")
    assert role_a is not None
    stream.prepare_audio(
        bytes(4800),
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
    )
    original_resample = stream._resample_for_roles  # noqa: SLF001
    fired = False

    async def resample_with_midflight_format_change(*args: Any, **kwargs: Any) -> Any:
        nonlocal fired
        if not fired:
            fired = True
            role_a.on_stream_request_format(
                StreamRequestFormatPayload(
                    player=StreamRequestFormatPlayer(
                        codec=AudioCodec.PCM,
                        sample_rate=44100,
                        channels=2,
                        bit_depth=16,
                    )
                )
            )
        return await original_resample(*args, **kwargs)

    monkeypatch.setattr(stream, "_resample_for_roles", resample_with_midflight_format_change)
    await stream.commit_audio()
    for _ in range(5):
        await asyncio.sleep(0)

    # The unchanged peer received the in-flight commit's window: exactly one
    # more 48kHz chunk, contiguous with its previous timeline, and no
    # stream/start disruption of its own.
    peer_chunks = conn_b.sent_binary[peer_chunks_before:]
    assert len(peer_chunks) == 1
    assert len(peer_chunks[0]) == 9 + 4800
    prev_header = unpack_binary_header(conn_b.sent_binary[peer_chunks_before - 1])
    this_header = unpack_binary_header(peer_chunks[0])
    assert this_header.timestamp_us == prev_header.timestamp_us + 25_000
    assert not any(isinstance(msg, StreamStartMessage) for msg in conn_b.sent_json)

    # The changing player got no old-format frame behind its boundary.
    assert all(len(frame) != 9 + 4800 for frame in conn_a.sent_binary[peer_chunks_before:])


# --- Historical Audio Tests ---


@pytest.mark.asyncio
async def test_historical_audio_raises_on_active_channel(mock_loop: Any) -> None:
    """prepare_historical_audio() raises ValueError on channel with active timing."""
    group = _DummyGroup(clients=[])
    _make_connected_player(mock_loop, group, "p1")

    stream = PushStream(loop=mock_loop, clock=LoopClock(mock_loop), group=group)

    # Commit audio to establish timing on MAIN_CHANNEL
    stream.prepare_audio(
        bytes(4800),
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
    )
    await stream.commit_audio()

    # Now historical audio on MAIN_CHANNEL should raise
    with pytest.raises(ValueError, match="already has active timing"):
        stream.prepare_historical_audio(
            bytes(4800),
            AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
            channel_id=MAIN_CHANNEL,
        )


@pytest.mark.asyncio
async def test_historical_audio_allows_synthetic_timing_channel() -> None:
    """Historical audio is allowed when channel timing exists but no audio was committed on it."""
    group = _DummyGroup(clients=[])
    other_channel = UUID("99999999-9999-9999-9999-999999999999")
    role_main = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=None,
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    role_other = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=None,
            channel_id=other_channel,
            frame_duration_us=25_000,
        )
    )
    group.clients.extend([_DummyClient([role_main]), _DummyClient([role_other])])

    loop = asyncio.get_running_loop()
    clock = ManualClock()
    stream = PushStream(loop=loop, clock=clock, group=group)
    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)

    # Commit MAIN only: other_channel gets synthetic timing, but no committed audio.
    stream.prepare_audio(bytes(4800), fmt, channel_id=MAIN_CHANNEL)
    await stream.commit_audio()

    synthetic_tail_us = stream._channel_timing[other_channel]  # noqa: SLF001
    assert synthetic_tail_us > 0

    # Should not raise: channel has synthetic timing only.
    stream.prepare_historical_audio(bytes(4800), fmt, channel_id=other_channel)
    await stream.commit_audio()

    assert role_other.received
    first_hist = role_other.received[0]
    assert first_hist.timestamp_us + first_hist.duration_us == synthetic_tail_us
    assert stream._channel_timing[other_channel] == synthetic_tail_us  # noqa: SLF001


@pytest.mark.asyncio
async def test_historical_audio_raises_after_historical_commit_on_channel() -> None:
    """Reject historical injection after the channel has committed historical audio."""
    group = _DummyGroup(clients=[])
    channel = UUID("88888888-8888-8888-8888-888888888888")
    role = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=None,
            channel_id=channel,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role]))

    loop = asyncio.get_running_loop()
    clock = ManualClock()
    stream = PushStream(loop=loop, clock=clock, group=group)
    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)

    stream.prepare_historical_audio(bytes(4800), fmt, channel_id=channel)
    await stream.commit_audio()

    with pytest.raises(ValueError, match="already has active timing"):
        stream.prepare_historical_audio(bytes(4800), fmt, channel_id=channel)


@pytest.mark.asyncio
async def test_historical_audio_respects_explicit_start_time() -> None:
    """Historical audio can be anchored to an explicit start timestamp."""
    group = _DummyGroup(clients=[])
    channel = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    role = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=None,
            channel_id=channel,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role]))

    loop = asyncio.get_running_loop()
    clock = ManualClock()
    stream = PushStream(loop=loop, clock=clock, group=group)

    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)
    explicit_start_us = 5_000_000
    stream.prepare_historical_audio(
        bytes(4800),
        fmt,
        channel_id=channel,
        start_time_us=explicit_start_us,
    )
    stream.prepare_historical_audio(bytes(4800), fmt, channel_id=channel)

    await stream.commit_audio()

    assert role.received
    assert role.received[0].timestamp_us == explicit_start_us
    assert role.received[1].timestamp_us == explicit_start_us + role.received[0].duration_us


@pytest.mark.asyncio
async def test_historical_audio_skips_stale_delivery_but_advances_timing() -> None:
    """Historical chunks that are already stale are not delivered to active roles."""
    group = _DummyGroup(clients=[])
    channel = UUID("abababab-abab-abab-abab-aaaaaaaaaaaa")
    role = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=None,
            channel_id=channel,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role]))

    loop = asyncio.get_running_loop()
    clock = ManualClock(now_us_value=1_000_000)
    stream = PushStream(loop=loop, clock=clock, group=group)

    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)
    stream.prepare_historical_audio(bytes(4800), fmt, channel_id=channel, start_time_us=100_000)
    stream.prepare_historical_audio(bytes(4800), fmt, channel_id=channel)

    await stream.commit_audio()

    # Delivery is skipped because chunks are fully before now + DEFAULT_INITIAL_DELAY_US.
    assert not role.received
    assert role.started == 0

    # Timing/cache still advance so future live chunks stay aligned.
    assert stream._channel_timing[channel] == 150_000  # noqa: SLF001


@pytest.mark.asyncio
async def test_historical_audio_only_no_live() -> None:
    """Historical-only commit (no prepare_audio) bootstraps channel with correct timing."""
    group = _DummyGroup(clients=[])
    channel = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
    role = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=None,
            channel_id=channel,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role]))

    loop = asyncio.get_running_loop()
    stream = PushStream(loop=loop, clock=LoopClock(loop), group=group)

    # Queue two historical chunks
    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)
    stream.prepare_historical_audio(bytes(4800), fmt, channel_id=channel)
    stream.prepare_historical_audio(bytes(4800), fmt, channel_id=channel)

    await stream.commit_audio()

    # Channel should have received audio
    assert role.started == 1
    assert len(role.received) >= 2

    # Timestamps should be consecutive
    first_ts = role.received[0].timestamp_us
    second_ts = role.received[1].timestamp_us
    expected_duration = 25_000  # 4800 bytes at 48kHz/16bit/stereo = 25ms
    assert second_ts == first_ts + expected_duration


@pytest.mark.asyncio
async def test_historical_plus_live_seamless_transition(mock_loop: Any) -> None:
    """Historical audio followed by live audio has seamless timestamps."""
    group = _DummyGroup(clients=[])
    channel = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
    role = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=None,
            channel_id=channel,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role]))

    clock = LoopClock(mock_loop)
    stream = PushStream(loop=mock_loop, clock=clock, group=group)

    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)

    # Queue historical chunk
    stream.prepare_historical_audio(bytes(4800), fmt, channel_id=channel)
    # Queue live chunk
    stream.prepare_audio(bytes(4800), fmt, channel_id=channel)

    await stream.commit_audio()

    assert role.started == 1
    assert len(role.received) >= 2

    # With frozen clock, live chunk timestamp should exactly follow historical
    historical_end = role.received[0].timestamp_us + role.received[0].duration_us
    live_start = role.received[1].timestamp_us
    assert live_start == historical_end


@pytest.mark.asyncio
async def test_historical_on_one_channel_live_on_another() -> None:
    """Historical on one channel, live-only on another in same commit."""
    group = _DummyGroup(clients=[])
    hist_channel = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
    stale_channel = UUID("eeeeeeee-0000-0000-0000-eeeeeeeeeeee")

    role_hist = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=None,
            channel_id=hist_channel,
            frame_duration_us=25_000,
        )
    )
    role_live = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=None,
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    group.clients.extend([_DummyClient([role_hist]), _DummyClient([role_live])])

    loop = asyncio.get_running_loop()
    clock = ManualClock(now_us_value=1_000_000)
    stream = PushStream(loop=loop, clock=clock, group=group)

    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)

    # Seed main timeline, then add stale committed timing from an inactive channel.
    stream.prepare_audio(bytes(4800), fmt, channel_id=MAIN_CHANNEL)
    await stream.commit_audio()
    stream._channel_timing[stale_channel] = 900_000  # noqa: SLF001
    stream._channels_with_committed_audio.add(stale_channel)  # noqa: SLF001

    # Historical on hist_channel, live on MAIN_CHANNEL
    stream.prepare_historical_audio(bytes(4800), fmt, channel_id=hist_channel)
    stream.prepare_audio(bytes(4800), fmt, channel_id=MAIN_CHANNEL)

    await stream.commit_audio()

    assert role_hist.started == 1
    assert role_hist.received
    assert role_live.started == 1
    assert role_live.received
    assert (
        role_hist.received[-1].timestamp_us + role_hist.received[-1].duration_us
        == role_live.received[-1].timestamp_us
    )


@pytest.mark.asyncio
async def test_missing_channel_commits_keep_channel_timing_aligned() -> None:
    """Channels that miss commits should still advance on the shared timeline."""
    group = _DummyGroup(clients=[])
    other_channel = UUID("abababab-abab-abab-abab-abababababab")
    role_main = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=None,
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    role_other = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=None,
            channel_id=other_channel,
            frame_duration_us=25_000,
        )
    )
    group.clients.extend([_DummyClient([role_main]), _DummyClient([role_other])])

    loop = asyncio.get_running_loop()
    clock = ManualClock()
    stream = PushStream(loop=loop, clock=clock, group=group)
    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)

    # Two commits with only MAIN_CHANNEL prepared.
    stream.prepare_audio(bytes(4800), fmt, channel_id=MAIN_CHANNEL)
    play_start_1 = await stream.commit_audio()
    clock.advance_us(25_000)
    stream.prepare_audio(bytes(4800), fmt, channel_id=MAIN_CHANNEL)
    play_start_2 = await stream.commit_audio()
    clock.advance_us(25_000)

    # DSP channel resumes: first timestamp should be aligned to current shared timeline.
    stream.prepare_audio(bytes(4800), fmt, channel_id=other_channel)
    play_start_3 = await stream.commit_audio()

    assert play_start_2 == play_start_1 + 25_000
    assert play_start_3 == play_start_2 + 25_000
    assert role_other.received
    assert role_other.received[0].timestamp_us == play_start_3


@pytest.mark.asyncio
async def test_late_introduced_channel_aligns_with_existing_timeline() -> None:
    """A channel added late should start on the same timeline as existing channels."""
    group = _DummyGroup(clients=[])
    other_channel = UUID("cdcdcdcd-cdcd-cdcd-cdcd-cdcdcdcdcdcd")
    role_main = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=None,
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role_main]))

    loop = asyncio.get_running_loop()
    clock = ManualClock()
    stream = PushStream(loop=loop, clock=clock, group=group)
    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)

    # Let MAIN_CHANNEL get far ahead of wall clock (no clock advancement).
    for _ in range(40):
        stream.prepare_audio(bytes(4800), fmt, channel_id=MAIN_CHANNEL)
        await stream.commit_audio()

    role_other = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=None,
            channel_id=other_channel,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role_other]))
    stream.on_role_join(role_other)

    stream.prepare_audio(bytes(4800), fmt, channel_id=MAIN_CHANNEL)
    stream.prepare_audio(bytes(4800), fmt, channel_id=other_channel)
    await stream.commit_audio()

    assert role_main.received
    assert role_other.received
    assert role_main.received[-1].timestamp_us == role_other.received[-1].timestamp_us


@pytest.mark.asyncio
async def test_stale_committed_channel_does_not_inflate_active_timeline() -> None:
    """Stale committed timing must not force larger-than-audio timeline rebases."""
    group = _DummyGroup(clients=[])
    stale_channel = UUID("efeefeee-eeee-eeee-eeee-eeeeeeeeeeee")
    role_main = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=None,
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role_main]))

    loop = asyncio.get_running_loop()
    clock = ManualClock(now_us_value=1_000_000)
    stream = PushStream(loop=loop, clock=clock, group=group)
    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)

    # Seed a stale committed channel with no active subscribers.
    stream._channel_timing[stale_channel] = 1_250_000  # noqa: SLF001
    stream._channels_with_committed_audio.add(stale_channel)  # noqa: SLF001

    stream.prepare_audio(bytes(4800), fmt, channel_id=MAIN_CHANNEL)
    play_start_1 = await stream.commit_audio()
    clock.advance_us(25_000)
    stream.prepare_audio(bytes(4800), fmt, channel_id=MAIN_CHANNEL)
    play_start_2 = await stream.commit_audio()

    # Timeline should advance by chunk duration (25ms), not by stale-channel rebases.
    assert play_start_2 == play_start_1 + 25_000
    assert len(role_main.received) == 2
    assert role_main.received[1].timestamp_us == role_main.received[0].timestamp_us + 25_000


@pytest.mark.asyncio
async def test_historical_pcm_cache_populated() -> None:
    """Historical audio populates PCM cache when enabled for the channel."""
    group = _DummyGroup(clients=[])
    channel = UUID("eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee")
    role = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=None,
            channel_id=channel,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role]))

    loop = asyncio.get_running_loop()
    stream = PushStream(loop=loop, clock=LoopClock(loop), group=group)
    stream.enable_pcm_cache_for_channel(channel)

    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)
    stream.prepare_historical_audio(bytes(4800), fmt, channel_id=channel)
    stream.prepare_historical_audio(bytes(4800), fmt, channel_id=channel)

    await stream.commit_audio()

    cached = stream.get_cached_pcm_chunks(channel)
    assert len(cached) == 2
    assert cached[0].pcm_data == bytes(4800)
    assert cached[1].timestamp_us == cached[0].timestamp_us + cached[0].duration_us


@pytest.mark.asyncio
async def test_historical_no_pcm_cache_without_enable() -> None:
    """Historical audio does not populate PCM cache when not enabled."""
    group = _DummyGroup(clients=[])
    channel = UUID("ffffffff-ffff-ffff-ffff-ffffffffffff")
    role = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=None,
            channel_id=channel,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role]))

    loop = asyncio.get_running_loop()
    stream = PushStream(loop=loop, clock=LoopClock(loop), group=group)

    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)
    stream.prepare_historical_audio(bytes(4800), fmt, channel_id=channel)

    await stream.commit_audio()

    cached = stream.get_cached_pcm_chunks(channel)
    assert len(cached) == 0


@pytest.mark.asyncio
async def test_clear_clears_historical_buffers(mock_loop: Any) -> None:
    """clear() discards pending historical audio."""
    group = _DummyGroup(clients=[])
    stream = PushStream(loop=mock_loop, clock=LoopClock(mock_loop), group=group)

    channel = UUID("11111111-1111-1111-1111-111111111111")
    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)
    stream.prepare_historical_audio(bytes(4800), fmt, channel_id=channel)

    stream.clear()
    assert not stream._historical_buffers  # noqa: SLF001


@pytest.mark.asyncio
async def test_stop_clears_historical_buffers(mock_loop: Any) -> None:
    """stop() discards pending historical audio."""
    group = _DummyGroup(clients=[])
    stream = PushStream(loop=mock_loop, clock=LoopClock(mock_loop), group=group)

    channel = UUID("22222222-2222-2222-2222-222222222222")
    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)
    stream.prepare_historical_audio(bytes(4800), fmt, channel_id=channel)

    stream.stop()
    assert not stream._historical_buffers  # noqa: SLF001


@pytest.mark.asyncio
async def test_late_joiner_after_historical_injection() -> None:
    """Late joiner gets cached chunks after historical audio was injected."""
    group = _DummyGroup(clients=[])
    channel = UUID("33333333-3333-3333-3333-333333333333")
    role1 = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=None,
            channel_id=channel,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role1]))

    loop = asyncio.get_running_loop()
    stream = PushStream(loop=loop, clock=LoopClock(loop), group=group)

    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)
    stream.prepare_historical_audio(bytes(4800), fmt, channel_id=channel)
    stream.prepare_audio(bytes(4800), fmt, channel_id=channel)
    await stream.commit_audio()

    assert role1.received

    # Late joiner on the same channel
    role2 = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=None,
            channel_id=channel,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role2]))
    stream.on_role_join(role2)

    assert role2.started == 1
    assert role2.received


def test_drift_rebuild_flushes_old_graph_before_replacing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drift-triggered graph rebuild must flush the old graph before replacing it."""
    flush_calls: list[bool] = []
    original_build = push_stream_module._build_resample_graph  # noqa: SLF001

    class _SpyGraph:
        """Wraps a real graph to detect push(None) flush calls."""

        def __init__(self, real_graph: object) -> None:
            self._real = real_graph

        def push(self, frame: object) -> None:
            if frame is None:
                flush_calls.append(True)
            self._real.push(frame)  # type: ignore[union-attr]

        def pull(self) -> object:
            return self._real.pull()  # type: ignore[union-attr]

    def _spy_build(**kwargs: Any) -> object:
        graph = original_build(**kwargs)
        return _SpyGraph(graph)

    monkeypatch.setattr(push_stream_module, "_build_resample_graph", _spy_build)

    group = _DummyGroup(clients=[])
    stream = PushStream(loop=MagicMock(), clock=ManualClock(), group=group)
    pcm_25ms = bytes(9600)  # 25ms @ 48kHz stereo f32

    # First call: creates graph, advances pending_timestamp_us by ~25ms
    push_stream_module._quantize_float_pcm(  # noqa: SLF001
        channel_id=MAIN_CHANNEL,
        pcm_data=pcm_25ms,
        output_ts=0,
        sample_rate=48_000,
        channels=2,
        target_bit_depth=16,
        resampler_cache=stream._resamplers,  # noqa: SLF001
    )
    # Second call with same output_ts=0: drift > 20ms, triggers rebuild
    push_stream_module._quantize_float_pcm(  # noqa: SLF001
        channel_id=MAIN_CHANNEL,
        pcm_data=pcm_25ms,
        output_ts=0,
        sample_rate=48_000,
        channels=2,
        target_bit_depth=16,
        resampler_cache=stream._resamplers,  # noqa: SLF001
    )

    assert flush_calls, "Old graph should have been flushed with push(None) before rebuild"


@pytest.mark.asyncio
async def test_multi_role_fanout_quantizes_once_per_pcm_key(
    mock_loop: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multiple TransformKeys sharing a pcm_key must quantize float PCM only once."""
    group = _DummyGroup(clients=[])
    pool = group.transformer_pool

    # Two roles with different frame durations → different TransformKeys, same pcm_key
    role1 = _DummyRole(
        AudioRequirements(
            sample_rate=48_000,
            bit_depth=16,
            channels=2,
            transformer=pool.get_or_create(
                PcmPassthrough,
                channel_id=MAIN_CHANNEL.int,
                sample_rate=48_000,
                bit_depth=16,
                channels=2,
                frame_duration_us=25_000,
            ),
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    role2 = _DummyRole(
        AudioRequirements(
            sample_rate=48_000,
            bit_depth=16,
            channels=2,
            transformer=pool.get_or_create(
                PcmPassthrough,
                channel_id=MAIN_CHANNEL.int,
                sample_rate=48_000,
                bit_depth=16,
                channels=2,
                frame_duration_us=50_000,
            ),
            channel_id=MAIN_CHANNEL,
            frame_duration_us=50_000,
        )
    )
    group.clients.extend([_DummyClient([role1]), _DummyClient([role2])])

    quantize_calls = 0
    original_quantizer = push_stream_module._quantize_float_pcm  # noqa: SLF001

    def _counted_quantizer(**kwargs: Any) -> object:
        nonlocal quantize_calls
        quantize_calls += 1
        return original_quantizer(**kwargs)

    monkeypatch.setattr(push_stream_module, "_quantize_float_pcm", _counted_quantizer)

    stream = PushStream(loop=mock_loop, clock=LoopClock(mock_loop), group=group)
    stream.prepare_audio(
        bytes(9600),  # 25ms @ 48kHz stereo f32
        AudioFormat(sample_rate=48_000, bit_depth=32, channels=2, sample_type="float"),
    )
    await stream.commit_audio()

    assert quantize_calls == 1, f"Expected 1 quantize call per pcm_key, got {quantize_calls}"
    assert role1.received, "role1 should have received audio chunks"
    # role2 may not receive chunks yet because 25ms of data is below its
    # 50ms frame duration — the important assertion is quantize_calls == 1.


@pytest.mark.asyncio
async def test_catchup_quantizer_does_not_share_live_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch-up path must use its own quantizer state, not the shared live cache."""
    # Count real graph rebuilds by observing `state.graph` identity across the call,
    # not by replaying the drift-condition check. The old output-cursor drift proxy
    # would flag false positives under soxr (where FIR latency makes the output
    # cursor lag the input by tens of ms even in steady state).
    drift_rebuild_count = 0
    original_resample = push_stream_module._resample_pcm_standalone  # noqa: SLF001

    def _tracking_resample(state: Any, pcm: bytes, fmt: Any, ts: int) -> Any:
        nonlocal drift_rebuild_count
        graph_before = id(state.graph)
        result = original_resample(state, pcm, fmt, ts)
        if state.graph is not None and id(state.graph) != graph_before:
            drift_rebuild_count += 1
        return result

    monkeypatch.setattr(push_stream_module, "_resample_pcm_standalone", _tracking_resample)

    class TransformerA:
        pending_timestamp_us: int | None = None

        @property
        def frame_duration_us(self) -> int:
            return 25_000

        def process(self, pcm: bytes, _ts: int, _dur: int) -> list[tuple[bytes, int]]:
            return [(pcm, 25_000)]

        def flush(self) -> list[tuple[bytes, int]]:
            return []

        def get_header(self) -> bytes | None:
            return None

        def reset(self) -> None:
            return

    class TransformerB(TransformerA):
        pass

    group = _DummyGroup(clients=[])

    # role1 uses TransformerA — its live encoding builds quantizer state in self._resamplers
    role1 = _DummyRole(
        AudioRequirements(
            sample_rate=48_000,
            bit_depth=16,
            channels=2,
            transformer=TransformerA(),
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role1]))

    loop = asyncio.get_running_loop()
    clock = ManualClock(now_us_value=0)
    stream = PushStream(loop=loop, clock=clock, group=group)
    stream.enable_pcm_cache_for_channel(MAIN_CHANNEL)

    # Commit several float PCM chunks — builds live quantizer state.
    # Advance clock by 25ms (matching audio duration) between commits to
    # keep resampler timestamps monotonic without drift-triggered rebuilds.
    for _ in range(4):
        stream.prepare_audio(
            bytes(9600),  # 25ms @ 48kHz stereo f32
            AudioFormat(sample_rate=48_000, bit_depth=32, channels=2, sample_type="float"),
        )
        await stream.commit_audio()
        clock.advance_us(25_000)

    # Count drift rebuilds so far (live path — should be 0 with ManualClock)
    live_drifts = drift_rebuild_count

    # Add a late-joining role with TransformerB — different TransformKey, triggers PCM catch-up
    role2 = _DummyRole(
        AudioRequirements(
            sample_rate=48_000,
            bit_depth=16,
            channels=2,
            transformer=TransformerB(),
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role2]))
    stream.on_role_join(role2)

    # Wait for the async catch-up task to complete
    for _ in range(50):
        if role2.received:
            break
        await asyncio.sleep(0.01)

    # Catch-up should NOT trigger drift-rebuilds in the shared quantizer cache.
    # With separate cache: fresh quantizer state, no drift detection.
    # With shared cache: the jump from live timestamps (~350ms) to historical
    # timestamps (~250ms) causes drift > 20ms, triggering a graph rebuild.
    catchup_drifts = drift_rebuild_count - live_drifts
    assert catchup_drifts == 0, (
        f"Expected 0 drift-triggered rebuilds during catch-up, got {catchup_drifts} "
        f"(likely shared quantizer cache causing timestamp jump)"
    )


async def test_replay_from_pcm_cache_overrides_established_resampler_skip() -> None:
    """Late join with an established resampler at the role's shape.

    A live role builds a resampler at 48k/16/2. When a second role joins with
    that same target shape, the established-resampler guard normally skips PCM
    replay (re-running the shared resampler would shift the live role's audio).
    Analysis-only roles opt in via `replay_from_pcm_cache()` so they get the
    buffered PCM immediately instead of waiting for the next live commit, which
    may sit far ahead behind a producer buffer.

    Asserts the flag is the only thing that flips the decision: same scenario,
    replay=True is fed cached audio; replay=False is not.
    """

    class TransformerA:
        pending_timestamp_us: int | None = None

        @property
        def frame_duration_us(self) -> int:
            return 25_000

        def process(self, pcm: bytes, _ts: int, _dur: int) -> list[tuple[bytes, int]]:
            return [(pcm, 25_000)]

        def flush(self) -> list[tuple[bytes, int]]:
            return []

        def get_header(self) -> bytes | None:
            return None

        def reset(self) -> None:
            return

    group = _DummyGroup(clients=[])
    live_role = _DummyRole(
        AudioRequirements(
            sample_rate=48_000,
            bit_depth=16,
            channels=2,
            transformer=TransformerA(),
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([live_role]))

    loop = asyncio.get_running_loop()
    clock = ManualClock(now_us_value=0)
    stream = PushStream(loop=loop, clock=clock, group=group)
    stream.enable_pcm_cache_for_channel(MAIN_CHANNEL)

    for _ in range(4):
        stream.prepare_audio(
            bytes(9600),  # 25ms @ 48kHz stereo f32
            AudioFormat(sample_rate=48_000, bit_depth=32, channels=2, sample_type="float"),
        )
        await stream.commit_audio()
        clock.advance_us(25_000)

    # Force the guarded branch: pretend a non-passthrough resampler already
    # exists at the joining role's target shape. Without the opt-out this
    # short-circuits PCM replay.
    stream._has_established_resampler_for = lambda *_a, **_k: True  # type: ignore[method-assign]  # noqa: SLF001

    # Analysis-only joiner (no transformer => distinct TransformKey from the
    # live role) that opts into PCM-cache replay.
    viz_role = _DummyRole(
        AudioRequirements(
            sample_rate=48_000,
            bit_depth=16,
            channels=2,
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        ),
        replay_from_pcm_cache=True,
        required_lead_time_us=100_000,
    )
    group.clients.append(_DummyClient([viz_role]))
    stream.on_role_join(viz_role)

    for _ in range(50):
        if viz_role.received:
            break
        await asyncio.sleep(0.01)

    assert viz_role.received, "replay_from_pcm_cache role should receive buffered PCM on late join"

    # Control: same scenario, replay disabled => guard skips replay, no audio.
    class TransformerB(TransformerA):
        pass

    plain_role = _DummyRole(
        AudioRequirements(
            sample_rate=48_000,
            bit_depth=16,
            channels=2,
            transformer=TransformerB(),
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        ),
        replay_from_pcm_cache=False,
    )
    group.clients.append(_DummyClient([plain_role]))
    stream.on_role_join(plain_role)

    for _ in range(20):
        await asyncio.sleep(0.01)

    assert not plain_role.received, "replay-disabled role must not get historical replay"
    assert plain_role.started >= 1, "replay-disabled role should still be started for live audio"


def test_noop_resample_bypasses_graph_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resampling with identical source/target should skip graph construction."""
    build_calls = 0
    original_build = push_stream_module._build_resample_graph  # noqa: SLF001

    def _counting_build(**kwargs: Any) -> object:
        nonlocal build_calls
        build_calls += 1
        return original_build(**kwargs)

    monkeypatch.setattr(push_stream_module, "_build_resample_graph", _counting_build)

    source = AudioFormat(sample_rate=48_000, bit_depth=16, channels=2, sample_type="int")
    key = push_stream_module._ResamplerKey(  # noqa: SLF001
        channel_id=MAIN_CHANNEL,
        source_format=source,
        target_sample_rate=48_000,
        target_channels=2,
        target_bit_depth=16,
        target_sample_type="int",
    )
    state = push_stream_module._create_resampler_state(key, source, source)  # noqa: SLF001

    pcm_25ms = bytes(4800)  # 25ms @ 48kHz stereo s16
    result = push_stream_module._resample_pcm_standalone(  # noqa: SLF001
        state, pcm_25ms, source, 1_000_000
    )

    assert build_calls == 0, "No-op resample should not build a filter graph"
    assert result.pcm_data == pcm_25ms, "No-op resample should return input PCM unchanged"
    assert result.sample_count == 1200  # 48000 * 0.025
    assert result.output_start_ts == 1_000_000


def test_24bit_passthrough_expands_to_s32_and_marks_wire_conversion() -> None:
    """Packed s24 passthrough should still normalize to s32 for internal processing."""
    source = AudioFormat(sample_rate=48_000, bit_depth=24, channels=2, sample_type="int")
    key = push_stream_module._ResamplerKey(  # noqa: SLF001
        channel_id=MAIN_CHANNEL,
        source_format=source,
        target_sample_rate=48_000,
        target_channels=2,
        target_bit_depth=24,
        target_sample_type="int",
    )
    state = push_stream_module._create_resampler_state(key, source, source)  # noqa: SLF001

    packed_pcm = bytes([0x11, 0x21, 0x31, 0x12, 0x22, 0x32]) * 2
    result = push_stream_module._resample_pcm_standalone(  # noqa: SLF001
        state, packed_pcm, source, 1_000_000
    )

    assert state.is_passthrough
    assert result.pcm_data == _expand_packed_s24_to_s32(packed_pcm)
    assert result.sample_count == 2
    assert result.needs_s32_to_s24_conversion is True


def test_24bit_input_expands_to_s32_before_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    """Packed s24 input must be expanded before writing into an s32 PyAV frame."""
    captured_input: bytes | None = None

    class _CapturingGraph:
        def push(self, frame: Any | None) -> None:
            nonlocal captured_input
            assert frame is not None
            captured_input = bytes(frame.planes[0])

        def pull(self) -> Any:
            raise EOFError

    def _build_capturing_graph(**_kwargs: Any) -> _CapturingGraph:
        return _CapturingGraph()

    monkeypatch.setattr(push_stream_module, "_build_resample_graph", _build_capturing_graph)

    source = AudioFormat(sample_rate=48_000, bit_depth=24, channels=2, sample_type="int")
    target = AudioFormat(sample_rate=48_000, bit_depth=32, channels=2, sample_type="int")
    key = push_stream_module._ResamplerKey(  # noqa: SLF001
        channel_id=MAIN_CHANNEL,
        source_format=source,
        target_sample_rate=48_000,
        target_channels=2,
        target_bit_depth=32,
        target_sample_type="int",
    )
    state = push_stream_module._create_resampler_state(key, source, target)  # noqa: SLF001

    packed_pcm = bytes([0x11, 0x21, 0x31, 0x12, 0x22, 0x32]) * 2
    push_stream_module._resample_pcm_standalone(  # noqa: SLF001
        state, packed_pcm, source, 1_000_000
    )

    assert captured_input == _expand_packed_s24_to_s32(packed_pcm)


@pytest.mark.asyncio
async def test_encode_pcm_sequence_preserves_packed_s24_for_pcm_passthrough() -> None:
    """Raw PCM output should convert internal s32 back to packed s24 on the wire."""
    group = _DummyGroup(clients=[])
    stream = PushStream(loop=MagicMock(), clock=ManualClock(), group=group)
    encoder = PcmPassthrough(sample_rate=48_000, bit_depth=24, channels=2)
    req = AudioRequirements(
        sample_rate=48_000,
        bit_depth=24,
        channels=2,
        transformer=encoder,
        channel_id=MAIN_CHANNEL,
        frame_duration_us=25_000,
    )
    packed_pcm = _packed_s24_pcm_25ms()
    pcm_chunk = CachedPCMChunk(
        timestamp_us=1_000_000,
        duration_us=25_000,
        pcm_data=packed_pcm,
        sample_rate=48_000,
        bit_depth=24,
        channels=2,
    )

    encoded = await stream._encode_pcm_sequence(  # noqa: SLF001
        [pcm_chunk], encoder, req, MAIN_CHANNEL
    )

    assert len(encoded) == 1
    assert encoded[0].payload == packed_pcm


@pytest.mark.asyncio
async def test_encode_pcm_sequence_expands_s24_before_flac_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FLAC encoding should receive AV-format s32 bytes for 24-bit PCM."""
    group = _DummyGroup(clients=[])
    stream = PushStream(loop=MagicMock(), clock=ManualClock(), group=group)
    encoder = FlacEncoder(sample_rate=48_000, bit_depth=24, channels=2)
    captured_chunk: bytes | None = None

    def _fake_ensure_initialized() -> None:
        encoder._initialized = True  # noqa: SLF001
        encoder._chunk_samples = 1200  # noqa: SLF001
        encoder._chunk_duration_us = 25_000  # noqa: SLF001
        encoder._frame_stride = 8  # noqa: SLF001
        encoder._av_format = "s32"  # noqa: SLF001
        encoder._av_layout = "stereo"  # noqa: SLF001

    def _capture_chunk(chunk_pcm: bytes) -> bytes:
        nonlocal captured_chunk
        captured_chunk = chunk_pcm
        return b"flac"

    monkeypatch.setattr(encoder, "_ensure_initialized", _fake_ensure_initialized)
    monkeypatch.setattr(encoder, "_encode_chunk", _capture_chunk)

    req = AudioRequirements(
        sample_rate=48_000,
        bit_depth=24,
        channels=2,
        transformer=encoder,
        channel_id=MAIN_CHANNEL,
        frame_duration_us=25_000,
    )
    packed_pcm = _packed_s24_pcm_25ms()
    pcm_chunk = CachedPCMChunk(
        timestamp_us=1_000_000,
        duration_us=25_000,
        pcm_data=packed_pcm,
        sample_rate=48_000,
        bit_depth=24,
        channels=2,
    )

    encoded = await stream._encode_pcm_sequence(  # noqa: SLF001
        [pcm_chunk], encoder, req, MAIN_CHANNEL
    )

    assert len(encoded) == 1
    assert encoded[0].payload == b"flac"
    assert captured_chunk == _expand_packed_s24_to_s32(packed_pcm)


def test_encode_pcm_sequence_yields_while_replaying_deep_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deep PCM cache is encoded across several loop steps, without changing output."""

    def _encode() -> tuple[int, list[CachedChunk]]:
        stream = PushStream(loop=MagicMock(), clock=ManualClock(), group=_DummyGroup(clients=[]))
        encoder = PcmPassthrough(sample_rate=48_000, bit_depth=16, channels=2)
        req = AudioRequirements(
            sample_rate=48_000,
            bit_depth=16,
            channels=2,
            transformer=encoder,
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
        # 3 seconds of cached PCM in 100ms chunks, as the producer buffers it. The source
        # rate differs from the target so a resampler carries FIR state across the yields,
        # and the audio is a ramp rather than silence so any lost state changes the bytes.
        chunks = [
            CachedPCMChunk(
                timestamp_us=1_000_000 + index * 100_000,
                duration_us=100_000,
                pcm_data=bytes((index * 4410 + frame) % 251 for frame in range(4410 * 2 * 2)),
                sample_rate=44_100,
                bit_depth=16,
                channels=2,
            )
            for index in range(30)
        ]
        # Drive the coroutine by hand so every suspension point is counted.
        coro = stream._encode_pcm_sequence(chunks, encoder, req, MAIN_CHANNEL)  # noqa: SLF001
        yields = 0
        while True:
            try:
                coro.send(None)
            except StopIteration as stop:
                return yields, stop.value
            yields += 1

    yields, encoded = _encode()
    monkeypatch.setattr(push_stream_module, "_PCM_SEQUENCE_YIELD_INTERVAL_US", 10**12)
    single_step_yields, single_step_encoded = _encode()

    assert single_step_yields == 0
    assert yields >= 5
    assert encoded
    assert [(c.timestamp_us, c.duration_us, c.payload) for c in encoded] == [
        (c.timestamp_us, c.duration_us, c.payload) for c in single_step_encoded
    ]


def test_soxr_fallback_caches_failure_per_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After soxr fails for a format combo, subsequent calls skip straight to swr."""
    # Force _supports_soxr_resampler to return True
    monkeypatch.setattr(push_stream_module, "_supports_soxr_resampler", lambda: True)

    # Replace with a fresh set so test mutations don't leak; monkeypatch restores on teardown
    monkeypatch.setattr(push_stream_module, "_soxr_failed_configs", set())

    av_mod = push_stream_module._get_av()  # noqa: SLF001
    OriginalGraph = av_mod.filter.Graph  # noqa: N806
    soxr_attempt_count = 0

    class _TrackingSoxrGraph:
        """Graph wrapper that fails on soxr and tracks attempts."""

        def __init__(self) -> None:
            self._real = OriginalGraph()

        def add_abuffer(self, **kwargs: Any) -> Any:
            return self._real.add_abuffer(**kwargs)

        def add(self, name: str, args: str = "") -> Any:
            nonlocal soxr_attempt_count
            if "soxr" in args:
                soxr_attempt_count += 1
                raise OSError("simulated soxr failure")
            return self._real.add(name, args)

        def link_nodes(self, *nodes: Any) -> Any:
            return self._real.link_nodes(*nodes)

    monkeypatch.setattr(av_mod.filter, "Graph", _TrackingSoxrGraph)

    # First call: soxr fails, falls back to swr
    push_stream_module._build_resample_graph(  # noqa: SLF001
        source_av_format="s16",
        source_layout="stereo",
        source_sample_rate=44_100,
        target_av_format="s16",
        target_layout="stereo",
        target_sample_rate=48_000,
    )
    first_soxr_attempts = soxr_attempt_count

    # Second call with same format: should skip soxr entirely
    push_stream_module._build_resample_graph(  # noqa: SLF001
        source_av_format="s16",
        source_layout="stereo",
        source_sample_rate=44_100,
        target_av_format="s16",
        target_layout="stereo",
        target_sample_rate=48_000,
    )

    assert first_soxr_attempts == 1, "First call should attempt soxr once"
    assert soxr_attempt_count == 1, (
        f"Second call should skip soxr (cached failure), but got {soxr_attempt_count} attempts"
    )


# ---------------------------------------------------------------------------
# Output delay send-ahead budget tests
# ---------------------------------------------------------------------------

_OUTPUT_DELAY_US = 5_000_000  # 5 s


def _make_role(
    *,
    output_delay_us: int = 0,
    required_lead_time_us: int = DEFAULT_INITIAL_DELAY_US,
    min_buffer_us: int = 0,
    channel_id: UUID = MAIN_CHANNEL,
) -> _DummyRole:
    return _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=None,
            channel_id=channel_id,
            frame_duration_us=25_000,
        ),
        output_delay_us=output_delay_us,
        required_lead_time_us=required_lead_time_us,
        min_buffer_us=min_buffer_us,
    )


@pytest.mark.asyncio
async def test_commit_audio_bootstrap_includes_output_delay() -> None:
    """First commit with a large-delay player pushes timeline forward."""
    loop = asyncio.get_running_loop()
    clock = ManualClock(now_us_value=1_000_000)
    group = _DummyGroup(clients=[])
    role = _make_role(output_delay_us=_OUTPUT_DELAY_US)
    group.clients.append(_DummyClient([role]))

    stream = PushStream(loop=loop, clock=clock, group=group)
    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)
    stream.prepare_audio(bytes(4800), fmt)
    play_start = await stream.commit_audio()

    assert play_start >= clock.now_us() + DEFAULT_INITIAL_DELAY_US + _OUTPUT_DELAY_US


@pytest.mark.asyncio
async def test_mixed_delay_timeline_uses_largest_output_delay() -> None:
    """With 0 ms and 5 s delay players, timeline anchored for the largest."""
    loop = asyncio.get_running_loop()
    clock = ManualClock(now_us_value=1_000_000)
    group = _DummyGroup(clients=[])
    role_fast = _make_role(output_delay_us=0)
    role_slow = _make_role(output_delay_us=_OUTPUT_DELAY_US)
    group.clients.extend([_DummyClient([role_fast]), _DummyClient([role_slow])])

    stream = PushStream(loop=loop, clock=clock, group=group)
    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)
    stream.prepare_audio(bytes(4800), fmt)
    play_start = await stream.commit_audio()

    assert play_start >= clock.now_us() + DEFAULT_INITIAL_DELAY_US + _OUTPUT_DELAY_US


@pytest.mark.asyncio
async def test_commit_audio_uses_reported_lead_time() -> None:
    """A player reporting a low lead time starts ahead by exactly that lead."""
    loop = asyncio.get_running_loop()
    clock = ManualClock(now_us_value=1_000_000)
    group = _DummyGroup(clients=[])
    role = _make_role(required_lead_time_us=50_000, min_buffer_us=0, output_delay_us=0)
    group.clients.append(_DummyClient([role]))

    stream = PushStream(loop=loop, clock=clock, group=group)
    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)
    stream.prepare_audio(bytes(4800), fmt)
    play_start = await stream.commit_audio()

    assert play_start == clock.now_us() + 50_000


@pytest.mark.asyncio
async def test_commit_audio_min_buffer_floors_send_ahead() -> None:
    """Live streams: when min_buffer exceeds lead time, the floor follows min_buffer."""
    loop = asyncio.get_running_loop()
    clock = ManualClock(now_us_value=1_000_000)
    group = _DummyGroup(clients=[])
    role = _make_role(required_lead_time_us=50_000, min_buffer_us=2_000_000, output_delay_us=0)
    group.clients.append(_DummyClient([role]))

    stream = PushStream(loop=loop, clock=clock, group=group)
    stream.set_live_source(is_live=True)
    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)
    stream.prepare_audio(bytes(4800), fmt)
    play_start = await stream.commit_audio()

    assert play_start == clock.now_us() + 2_000_000


@pytest.mark.asyncio
async def test_buffered_source_floors_at_min_buffer_at_startup() -> None:
    """Buffered streams floor startup at min_buffer + static (spec 8181237)."""
    loop = asyncio.get_running_loop()
    clock = ManualClock(now_us_value=1_000_000)
    group = _DummyGroup(clients=[])
    role = _make_role(required_lead_time_us=0, min_buffer_us=15_000_000, output_delay_us=0)
    group.clients.append(_DummyClient([role]))

    stream = PushStream(loop=loop, clock=clock, group=group)
    stream.set_live_source(is_live=False)
    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)
    stream.prepare_audio(bytes(4800), fmt)
    play_start = await stream.commit_audio()

    # Buffered floor = max(min_buffer=15s, lead=0) + static(0) = 15s.
    assert play_start == clock.now_us() + 15_000_000


@pytest.mark.asyncio
async def test_explicit_play_start_us_overrides_stale_channel_timing() -> None:
    """Explicit play_start_us is authoritative across mode switches."""
    loop = asyncio.get_running_loop()
    clock = ManualClock(now_us_value=1_000_000)
    group = _DummyGroup(clients=[])
    role = _make_role(required_lead_time_us=0, min_buffer_us=0, output_delay_us=0)
    group.clients.append(_DummyClient([role]))

    stream = PushStream(loop=loop, clock=clock, group=group)
    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)

    # First commit anchors the channel at an explicit start far ahead of "now".
    first_play_start_us = 10_000_000
    stream.prepare_audio(bytes(4800), fmt)
    returned_first = await stream.commit_audio(play_start_us=first_play_start_us)
    assert returned_first == first_play_start_us

    # Subsequent commit with a different play_start_us must overwrite the
    # advanced channel timing, not be ignored because the channel already exists.
    second_play_start_us = 20_000_000
    stream.prepare_audio(bytes(4800), fmt)
    returned_second = await stream.commit_audio(play_start_us=second_play_start_us)
    assert returned_second == second_play_start_us

    # The channel timing now reflects the second play_start_us plus its chunk
    # duration (25ms @ 48kHz stereo s16 = 25_000us), not first + 2 * duration.
    expected_after_advance = second_play_start_us + 25_000
    assert stream._channel_timing[MAIN_CHANNEL] == expected_after_advance  # noqa: SLF001


@pytest.mark.asyncio
async def test_send_ahead_uses_largest_member_budget() -> None:
    """Live grouped players use a common floor: max(lead, min_buffer) + static across members."""
    loop = asyncio.get_running_loop()
    clock = ManualClock(now_us_value=1_000_000)
    group = _DummyGroup(clients=[])
    role_low = _make_role(required_lead_time_us=100_000, min_buffer_us=0, output_delay_us=0)
    role_high = _make_role(
        required_lead_time_us=0, min_buffer_us=1_500_000, output_delay_us=200_000
    )
    group.clients.extend([_DummyClient([role_low]), _DummyClient([role_high])])

    stream = PushStream(loop=loop, clock=clock, group=group)
    stream.set_live_source(is_live=True)
    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)
    stream.prepare_audio(bytes(4800), fmt)
    play_start = await stream.commit_audio()

    # role_high dominates: max(0, 1_500_000) + 200_000 = 1_700_000.
    assert play_start == clock.now_us() + 1_700_000


@pytest.mark.asyncio
async def test_sleep_to_limit_buffer_accounts_for_output_delay() -> None:
    """Buffer throttle allows extra lead equal to the largest output delay."""
    loop = asyncio.get_running_loop()
    clock = ManualClock(now_us_value=1_000_000)
    group = _DummyGroup(clients=[])
    role = _make_role(output_delay_us=_OUTPUT_DELAY_US)
    group.clients.append(_DummyClient([role]))

    stream = PushStream(loop=loop, clock=clock, group=group)
    # Timeline 5.25 s ahead — exactly DEFAULT + output_delay
    stream._channel_timing[MAIN_CHANNEL] = (  # noqa: SLF001
        clock.now_us() + DEFAULT_INITIAL_DELAY_US + _OUTPUT_DELAY_US
    )

    # With a 500 ms base buffer, effective limit = 500 ms + 5 s = 5.5 s.
    # ahead_us = 5.25 s < 5.5 s → no sleep.
    await stream.sleep_to_limit_buffer(max_buffer_us=500_000)
    # Test passes by returning immediately (no hang).


@pytest.mark.asyncio
async def test_historical_stale_filter_includes_output_delay() -> None:
    """Historical chunk between DEFAULT and DEFAULT+output_delay is filtered."""
    loop = asyncio.get_running_loop()
    clock = ManualClock(now_us_value=1_000_000)
    channel = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
    group = _DummyGroup(clients=[])
    role = _make_role(output_delay_us=_OUTPUT_DELAY_US, channel_id=channel)
    group.clients.append(_DummyClient([role]))

    stream = PushStream(loop=loop, clock=clock, group=group)
    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)
    # Place chunk just above the old threshold (now + DEFAULT = 1_250_000) but well
    # below the new threshold (now + DEFAULT + 5s = 6_250_000).  Without the fix
    # this chunk would have been delivered; with the fix it is correctly filtered.
    start_us = clock.now_us() + DEFAULT_INITIAL_DELAY_US + 100_000  # now + 350 ms
    stream.prepare_historical_audio(bytes(4800), fmt, channel_id=channel, start_time_us=start_us)
    await stream.commit_audio()

    assert not role.received
    # Channel timing still advanced (continuity preserved)
    assert channel in stream._channel_timing  # noqa: SLF001


@pytest.mark.asyncio
async def test_zero_delay_regression_commit_audio_unchanged() -> None:
    """With all delays at 0, commit_audio behaves identically to before."""
    loop = asyncio.get_running_loop()
    clock = ManualClock(now_us_value=1_000_000)
    group = _DummyGroup(clients=[])
    role = _make_role(output_delay_us=0)
    group.clients.append(_DummyClient([role]))

    stream = PushStream(loop=loop, clock=clock, group=group)
    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)
    stream.prepare_audio(bytes(4800), fmt)
    play_start = await stream.commit_audio()

    assert play_start == clock.now_us() + DEFAULT_INITIAL_DELAY_US


# ---------------------------------------------------------------------------
# Regression tests for the resampler timestamp-drift bug.
# ---------------------------------------------------------------------------
#
# `_resample_pcm_standalone` advances `pending_timestamp_us` per call by
# `int(output_samples * 1_000_000 / target_sample_rate)`. When the divisor
# does not divide cleanly (e.g. 44.1kHz with 1102 samples = 24988.66µs), the
# `int(...)` truncation accumulates per-call drift. Combined with the matching
# bug in `PcmPassthrough`, this produced the cliff at the 500ms transformer
# drift threshold in `_encode_for_transform_key`.
#
# The fix uses an integer residue accumulator on `_ResamplerState` so cumulative
# pending exactly matches `total_output_samples * 1_000_000 // target_rate`.


def _make_resampler_state_passthrough(
    *, sample_rate: int, bit_depth: int = 16, channels: int = 2
) -> push_stream_module._ResamplerState:  # type: ignore[name-defined]
    """Build a passthrough resampler state at the given target rate."""
    fmt = AudioFormat(
        sample_rate=sample_rate, bit_depth=bit_depth, channels=channels, sample_type="int"
    )
    key = push_stream_module._ResamplerKey(  # noqa: SLF001
        channel_id=MAIN_CHANNEL,
        source_format=fmt,
        target_sample_rate=sample_rate,
        target_channels=channels,
        target_bit_depth=bit_depth,
        target_sample_type="int",
    )
    return push_stream_module._create_resampler_state(key, fmt, fmt)  # noqa: SLF001


def test_resampler_passthrough_44100_no_drift_per_call() -> None:
    """Passthrough @ 44.1k with 1102 samples advances pending by 24988µs (not 24988.66)."""
    state = _make_resampler_state_passthrough(sample_rate=44100)
    fmt = AudioFormat(sample_rate=44100, bit_depth=16, channels=2, sample_type="int")
    pcm = bytes(1102 * 4)  # 1102 stereo s16 samples
    result = push_stream_module._resample_pcm_standalone(state, pcm, fmt, 0)  # noqa: SLF001
    assert result.output_start_ts == 0
    assert state.pending_timestamp_us == 24988  # int(1102 * 1e6 / 44100)


def test_resampler_passthrough_44100_no_cumulative_drift() -> None:
    """After 10k passthrough calls @ 44.1k, pending == sample-derived elapsed time exactly."""
    state = _make_resampler_state_passthrough(sample_rate=44100)
    fmt = AudioFormat(sample_rate=44100, bit_depth=16, channels=2, sample_type="int")
    pcm_one_chunk = bytes(1102 * 4)
    n_calls = 10_000
    # Each call advances input_ts by exactly 1102 samples worth of µs (using
    # rational arithmetic on the input side too).
    input_residue = 0
    input_ts = 0
    for _ in range(n_calls):
        push_stream_module._resample_pcm_standalone(  # noqa: SLF001
            state, pcm_one_chunk, fmt, input_ts
        )
        input_residue += 1102 * 1_000_000
        delta, input_residue = divmod(input_residue, 44100)
        input_ts += delta

    # Cumulative pending must equal n_calls * 1102 * 1_000_000 // 44100 exactly.
    expected = n_calls * 1102 * 1_000_000 // 44100
    actual = state.pending_timestamp_us
    assert actual is not None
    assert actual == expected, (
        f"resampler accumulated {actual - expected}µs of "
        f"drift over {n_calls} calls — regression of the per-call truncation bug"
    )


def test_resampler_passthrough_drift_reset_clears_residue() -> None:
    """Resampler drift-reset path (>20ms input/pending mismatch) must reset residue."""
    state = _make_resampler_state_passthrough(sample_rate=44100)
    fmt = AudioFormat(sample_rate=44100, bit_depth=16, channels=2, sample_type="int")
    pcm = bytes(1102 * 4)
    # Prime with a few calls so residue accumulates
    for i in range(5):
        push_stream_module._resample_pcm_standalone(state, pcm, fmt, i * 24988)  # noqa: SLF001
    # Now jump input forward by 5 seconds — way beyond the 20ms drift threshold
    new_input_ts = 10_000_000
    push_stream_module._resample_pcm_standalone(state, pcm, fmt, new_input_ts)  # noqa: SLF001
    # After the rebase + one fresh call: pending should be new_input_ts + 24988
    # (one chunk's worth from the fresh anchor, with residue having been reset to 0)
    assert state.pending_timestamp_us == new_input_ts + 24988
    # And the residue carried into the next call should reflect ONE call worth
    # (i.e. residue == 1102 * 1_000_000 % 44100 = 14800)
    assert state.pending_ts_residue == 1102 * 1_000_000 % 44100


def test_resampler_passthrough_48000_zero_drift() -> None:
    """At 48kHz/25ms (clean rate), drift is zero with old or new code — sanity check."""
    state = _make_resampler_state_passthrough(sample_rate=48000)
    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2, sample_type="int")
    pcm = bytes(1200 * 4)
    push_stream_module._resample_pcm_standalone(state, pcm, fmt, 0)  # noqa: SLF001
    assert state.pending_timestamp_us == 25_000


def test_resampler_passthrough_initial_state_has_zero_residue() -> None:
    """Freshly created _ResamplerState starts with residue=0."""
    state = _make_resampler_state_passthrough(sample_rate=44100)
    assert state.pending_ts_residue == 0
    assert state.pending_timestamp_us is None


def test_resampler_graph_path_44100_no_cumulative_drift() -> None:
    """48k→44.1k resample via the actual PyAV graph path — the path that triggered the bug."""
    source = AudioFormat(sample_rate=48_000, bit_depth=16, channels=2, sample_type="int")
    target = AudioFormat(sample_rate=44_100, bit_depth=16, channels=2, sample_type="int")
    key = push_stream_module._ResamplerKey(  # noqa: SLF001
        channel_id=MAIN_CHANNEL,
        source_format=source,
        target_sample_rate=44_100,
        target_channels=2,
        target_bit_depth=16,
        target_sample_type="int",
    )
    state = push_stream_module._create_resampler_state(key, source, target)  # noqa: SLF001
    assert not state.is_passthrough  # exercising the graph path

    # Push N calls of 25ms @ 48k = 1200 input samples each
    n_calls = 1000
    input_pcm = bytes(1200 * 4)
    input_residue = 0
    input_ts = 0
    total_output_samples = 0
    last_pending = 0
    for _ in range(n_calls):
        result = push_stream_module._resample_pcm_standalone(  # noqa: SLF001
            state, input_pcm, source, input_ts
        )
        total_output_samples += result.sample_count
        last_pending = state.pending_timestamp_us or 0
        input_residue += 1200 * 1_000_000
        delta, input_residue = divmod(input_residue, 48_000)
        input_ts += delta

    # Cumulative pending must exactly equal total_output_samples * 1e6 // 44100.
    # The graph may produce a slightly variable sample count per call due to
    # FFmpeg's filter buffering, but cumulative is what must be drift-free.
    expected = total_output_samples * 1_000_000 // 44_100
    drift = last_pending - expected
    assert drift == 0, (
        f"graph-path resampler drifted {drift}µs over {n_calls} calls "
        f"({total_output_samples} output samples) — regression of the per-call "
        f"truncation bug at push_stream.py:574"
    )


def test_resampler_pending_input_ts_has_zero_cumulative_drift_at_44100_source() -> None:
    """`pending_input_timestamp_us` must be drift-free for non-clean source rates.

    Plain `int(samples * 1e6 / source_rate)` truncation accumulates per-call error
    at rates that don't divide 1e6 evenly (e.g. 44.1k). Left unchecked, the
    input-side cursor would eventually lag the true input timeline past the
    20 ms drift threshold and cause a spurious graph rebuild. The divmod residue
    accumulator must match `total_samples * 1e6 // source_sample_rate` exactly.
    """
    source = AudioFormat(sample_rate=44_100, bit_depth=16, channels=2, sample_type="int")
    target = AudioFormat(sample_rate=44_100, bit_depth=16, channels=2, sample_type="int")
    key = push_stream_module._ResamplerKey(  # noqa: SLF001
        channel_id=MAIN_CHANNEL,
        source_format=source,
        target_sample_rate=44_100,
        target_channels=2,
        target_bit_depth=16,
        target_sample_type="int",
    )
    state = push_stream_module._create_resampler_state(key, source, target)  # noqa: SLF001
    assert state.pending_input_ts_residue == 0

    # 25 ms @ 44.1k = 1102 samples → 24988.66µs actual, rounds to 24988µs.
    # Plain `int(...)` would drop 0.66µs per call. Over 40_800 calls (17 min)
    # the old code would accumulate ~27 ms of input-side lag — enough to cross
    # the 20 ms rebuild threshold spuriously.
    samples_per_call = 1102
    input_pcm = bytes(samples_per_call * 4)
    n_calls = 40_800
    # The caller's `input_timestamp_us` advances along the true input timeline,
    # which is itself drift-free (we use the same divmod pattern here).
    external_residue = 0
    external_ts = 0
    for _ in range(n_calls):
        push_stream_module._resample_pcm_standalone(  # noqa: SLF001
            state, input_pcm, source, external_ts
        )
        external_residue += samples_per_call * 1_000_000
        delta, external_residue = divmod(external_residue, 44_100)
        external_ts += delta

    expected = n_calls * samples_per_call * 1_000_000 // 44_100
    actual = state.pending_input_timestamp_us
    drift = (actual or 0) - expected
    assert drift == 0, (
        f"pending_input_timestamp_us drifted {drift}µs over {n_calls} calls — "
        "regression of the input-side residue accumulator. This would eventually "
        "cross the 20ms drift threshold and trigger a spurious graph rebuild."
    )


def test_advance_channel_timing_is_drift_free() -> None:
    """Cumulative `_channel_timing` advance must equal `total_samples * 1e6 // rate`."""
    group = _DummyGroup(clients=[])
    stream = PushStream(loop=MagicMock(), clock=ManualClock(), group=group)
    channel_id = MAIN_CHANNEL
    stream._channel_timing[channel_id] = 0  # noqa: SLF001

    # 1024 samples @ 44.1k ≈ 23219.95µs; lossy int(...) drops ≈0.95µs/chunk.
    samples_per_chunk = 1024
    sample_rate = 44_100
    n_chunks = 1_000
    total_added = 0
    for _ in range(n_chunks):
        total_added += stream._advance_channel_timing(  # noqa: SLF001
            channel_id, samples_per_chunk, sample_rate
        )

    expected = n_chunks * samples_per_chunk * 1_000_000 // sample_rate
    assert stream._channel_timing[channel_id] == expected  # noqa: SLF001
    assert total_added == expected
    lossy = n_chunks * (samples_per_chunk * 1_000_000 // sample_rate)
    assert expected - lossy >= 1, "test rate must produce observable drift"


@pytest.mark.asyncio
async def test_commit_audio_advances_channel_timing_drift_free(mock_loop: Any) -> None:
    """End-to-end: residue accumulator must carry truncation across `commit_audio` calls."""
    group = _DummyGroup(clients=[])
    _client, _conn = _make_connected_player(mock_loop, group, "p1")

    clock = LoopClock(mock_loop)
    stream = PushStream(loop=mock_loop, clock=clock, group=group)

    samples_per_chunk = 1024
    sample_rate = 44_100
    pcm = bytes(samples_per_chunk * 4)
    fmt = AudioFormat(sample_rate=sample_rate, bit_depth=16, channels=2)

    n_commits = 100
    timings: list[int] = []
    for _ in range(n_commits):
        stream.prepare_audio(pcm, fmt)
        await stream.commit_audio()
        timings.append(stream._channel_timing[MAIN_CHANNEL])  # noqa: SLF001

    # Initial timing depends on wall clock (set by _resolve_channel_play_start);
    # subtract the post-first-commit baseline to isolate the residue invariant.
    elapsed = timings[-1] - timings[0]
    total_after_n = (n_commits * samples_per_chunk * 1_000_000) // sample_rate
    total_after_1 = (samples_per_chunk * 1_000_000) // sample_rate
    expected = total_after_n - total_after_1
    assert elapsed == expected, (
        f"channel timing drifted: got {elapsed}µs, expected {expected}µs over "
        f"{n_commits - 1} subsequent commits"
    )
    lossy = (n_commits - 1) * total_after_1
    assert expected - lossy >= 1


def test_advance_channel_timing_resets_residue_on_rate_change() -> None:
    """Residue is modulo the previous rate; switching rates must reset it."""
    group = _DummyGroup(clients=[])
    stream = PushStream(loop=MagicMock(), clock=ManualClock(), group=group)
    channel_id = MAIN_CHANNEL
    stream._channel_timing[channel_id] = 0  # noqa: SLF001

    # 1024 samples @ 44100 leaves a non-zero residue (42100, modulo 44100).
    delta1 = stream._advance_channel_timing(channel_id, 1024, 44_100)  # noqa: SLF001
    assert delta1 == 1024 * 1_000_000 // 44_100  # 23219

    # Switching to 48000 must not carry the 44100-modulus residue into the new
    # divmod, otherwise delta would be 21334 instead of 21333.
    delta2 = stream._advance_channel_timing(channel_id, 1024, 48_000)  # noqa: SLF001
    assert delta2 == 1024 * 1_000_000 // 48_000, (
        f"residue from prior rate bled into new-rate computation: "
        f"got {delta2}µs, expected {1024 * 1_000_000 // 48_000}µs"
    )


@pytest.mark.asyncio
async def test_catchup_drain_advances_encoder_pending_to_live_tip() -> None:
    """Drained FIR tail must advance encoder.pending_timestamp_us to the live tip.

    Without draining, the resampler's FIR holds samples past the catchup tail and
    `encoder.pending_timestamp_us` stays behind the live timeline. The next live
    commit then back-shifts its first chunk via `candidate_base` in
    `_encode_for_transform_key`, putting the joiner ahead of peers.
    """

    class Transformer:
        def __init__(self) -> None:
            self.pending_timestamp_us: int | None = None
            self._buffer = bytearray()
            self._frame_size = 25 * 44_100 * 2 * 2 // 1000  # 25ms @ 44.1kHz stereo s16

        @property
        def frame_duration_us(self) -> int:
            return 25_000

        def process(self, pcm: bytes, ts: int, _dur: int) -> list[tuple[bytes, int]]:
            if self.pending_timestamp_us is None:
                self.pending_timestamp_us = ts
            self._buffer.extend(pcm)
            frames: list[tuple[bytes, int]] = []
            while len(self._buffer) >= self._frame_size:
                frames.append((bytes(self._buffer[: self._frame_size]), 25_000))
                del self._buffer[: self._frame_size]
                if self.pending_timestamp_us is not None:
                    self.pending_timestamp_us += 25_000
            return frames

        def flush(self) -> list[tuple[bytes, int]]:
            return []

        def get_header(self) -> bytes | None:
            return None

        def reset(self) -> None:
            self._buffer.clear()
            self.pending_timestamp_us = None

    group = _DummyGroup(clients=[])
    role1 = _DummyRole(
        AudioRequirements(
            sample_rate=48_000,
            bit_depth=16,
            channels=2,
            transformer=Transformer(),
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role1]))

    loop = asyncio.get_running_loop()
    clock = ManualClock(now_us_value=0)
    stream = PushStream(loop=loop, clock=clock, group=group)
    stream.enable_pcm_cache_for_channel(MAIN_CHANNEL)

    for _ in range(8):
        stream.prepare_audio(
            bytes(19_200),  # 100ms @ 48kHz stereo f32
            AudioFormat(sample_rate=48_000, bit_depth=32, channels=2, sample_type="float"),
        )
        await stream.commit_audio()
        clock.advance_us(100_000)

    joining_transformer = Transformer()
    role2 = _DummyRole(
        AudioRequirements(
            sample_rate=44_100,  # rate change forces a real soxr/swr FIR
            bit_depth=16,
            channels=2,
            transformer=joining_transformer,
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        ),
        required_lead_time_us=100_000,
    )
    group.clients.append(_DummyClient([role2]))
    stream.on_role_join(role2)

    for _ in range(50):
        if role2.received:
            break
        await asyncio.sleep(0.01)

    assert role2.received, "catchup produced no chunks"
    catchup_last_end_us = role2.received[-1].timestamp_us + role2.received[-1].duration_us
    channel_tip_us = stream._channel_timing[MAIN_CHANNEL]  # noqa: SLF001
    # Without drain the gap is roughly one FIR group delay (a few ms with swr,
    # ~20 ms with libsoxr at precision=30). With drain it should land within one
    # encoder frame_dur of the live tip.
    gap_us = channel_tip_us - catchup_last_end_us
    assert 0 <= gap_us <= 25_000, (
        f"catchup tail too far behind live tip after drain: "
        f"channel_tip={channel_tip_us}, catchup_last_end={catchup_last_end_us}, "
        f"gap={gap_us}µs"
    )


@pytest.mark.asyncio
async def test_catchup_mid_pass_format_change_keeps_chunks_ordered() -> None:
    """Source-format change mid-catchup must flush the prior resampler in place.

    The PCM cache can hold chunks from multiple source formats (e.g. a sample-rate
    change earlier in the stream). With end-of-pass-only draining, the prior
    resampler's FIR tail would be appended after the new resampler's chunks,
    producing out-of-order timestamps. Flushing on key switch keeps output ordered.
    """

    class Transformer:
        pending_timestamp_us: int | None = None

        @property
        def frame_duration_us(self) -> int:
            return 25_000

        def process(self, pcm: bytes, _ts: int, _dur: int) -> list[tuple[bytes, int]]:
            return [(pcm, 25_000)]

        def flush(self) -> list[tuple[bytes, int]]:
            return []

        def get_header(self) -> bytes | None:
            return None

        def reset(self) -> None:
            return

    group = _DummyGroup(clients=[])
    role1 = _DummyRole(
        AudioRequirements(
            sample_rate=44_100,
            bit_depth=16,
            channels=2,
            transformer=Transformer(),
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role1]))

    loop = asyncio.get_running_loop()
    clock = ManualClock(now_us_value=0)
    stream = PushStream(loop=loop, clock=clock, group=group)
    stream.enable_pcm_cache_for_channel(MAIN_CHANNEL)

    # Two batches of 48k float, then 44.1k float, then back to 48k float — the
    # PCM cache will end up with mixed source formats requiring two distinct
    # resampler keys during catchup.
    formats = [
        (48_000, 4),
        (44_100, 4),
        (48_000, 4),
    ]
    for sample_rate, count in formats:
        # 100 ms of stereo float32 at the chosen rate.
        chunk_bytes = bytes(sample_rate * 2 * 4 // 10)
        for _ in range(count):
            stream.prepare_audio(
                chunk_bytes,
                AudioFormat(sample_rate=sample_rate, bit_depth=32, channels=2, sample_type="float"),
            )
            await stream.commit_audio()
            clock.advance_us(100_000)

    role2 = _DummyRole(
        AudioRequirements(
            sample_rate=44_100,
            bit_depth=16,
            channels=2,
            transformer=Transformer(),
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        ),
        required_lead_time_us=100_000,
    )
    group.clients.append(_DummyClient([role2]))
    stream.on_role_join(role2)

    for _ in range(50):
        if role2.received:
            break
        await asyncio.sleep(0.01)

    assert role2.received, "catchup produced no chunks"
    for prev, nxt in zip(role2.received, role2.received[1:], strict=False):
        assert nxt.timestamp_us >= prev.timestamp_us, (
            f"timestamps out of order across format change: "
            f"{prev.timestamp_us} -> {nxt.timestamp_us}"
        )
