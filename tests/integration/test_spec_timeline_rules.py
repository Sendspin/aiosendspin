"""Player timeline rules from the spec, checked on the wire across input jumps."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from aiosendspin.models.core import ClientStatePayload, StreamClearMessage, StreamEndMessage
from aiosendspin.models.player import PlayerStatePayload, SupportedAudioFormat
from aiosendspin.models.types import AudioCodec
from aiosendspin.server.audio import AudioFormat
from aiosendspin.server.channels import MAIN_CHANNEL
from aiosendspin.server.client import SendspinClient
from aiosendspin.server.clock import ManualClock
from aiosendspin.server.group import SendspinGroup
from aiosendspin.server.push_stream import PushStream
from tests.integration.sync_assertions import (
    WireSpan,
    band_limited_noise,
    content_offset_us,
    encoded_segments_from_events,
    jumps_us,
    player_timeline,
    wire_spans,
)
from tests.integration.sync_harness import (
    CaptureConnection,
    DummyServer,
    channel_resolver_for,
    make_pool_player,
)

_SOURCE_RATE = 48_000
_SOURCE_FORMAT = AudioFormat(sample_rate=_SOURCE_RATE, bit_depth=16, channels=2)
_SLICE_US = 100_000
_SLICE_FRAMES = _SOURCE_RATE * _SLICE_US // 1_000_000
_TOTAL_SLICES = 70
_JUMP_SLICE = 30
_JUMP_US = 30_000
_CONTENT = (band_limited_noise(_SLICE_FRAMES * _TOTAL_SLICES, _SOURCE_RATE) * 32767).astype(
    np.int16
)
# One 44.1 kHz sample is 22.7 us.
_TOLERANCE_US = 50
_MEMBERS = {
    "pcm-48k": (AudioCodec.PCM, 48_000),
    "pcm-44k": (AudioCodec.PCM, 44_100),
    "flac-44k": (AudioCodec.FLAC, 44_100),
    "opus-48k": (AudioCodec.OPUS, 48_000),
}


def _slice_pcm(index: int) -> bytes:
    return _CONTENT[index * _SLICE_FRAMES : (index + 1) * _SLICE_FRAMES].tobytes()


class _SendTimes:
    """Record the server clock at which each audio chunk is handed to a connection."""

    def __init__(self, conn: CaptureConnection, clock: ManualClock) -> None:
        self.times_us: list[int] = []
        send_binary = conn.send_binary

        def record(data: bytes, **kwargs: Any) -> bool:
            self.times_us.append(clock.now_us())
            return send_binary(data, **kwargs)

        conn.send_binary = record  # type: ignore[method-assign]

    def spans(self, events: Sequence[Any]) -> list[WireSpan]:
        spans = wire_spans(events)
        assert len(spans) == len(self.times_us)
        for span, sent_at_us in zip(spans, self.times_us, strict=True):
            span.sent_at_us = sent_at_us
        return spans


@dataclass(slots=True)
class _Group:
    clock: ManualClock
    server: DummyServer
    group: SendspinGroup
    stream: PushStream
    clients: dict[str, SendspinClient]
    connections: dict[str, CaptureConnection]


async def _group(
    members: dict[str, tuple[AudioCodec, int]], *, joiners: Sequence[str] = ()
) -> _Group:
    loop = asyncio.get_running_loop()
    clock = ManualClock()
    server = DummyServer(loop=loop, clock=clock)
    clients: dict[str, SendspinClient] = {}
    connections: dict[str, CaptureConnection] = {}
    for name, (codec, rate) in members.items():
        clients[name], connections[name] = make_pool_player(
            server, name, codec=codec, sample_rate=rate
        )
    names = [name for name in members if name not in joiners]
    group = clients[names[0]].group
    for name in names[1:]:
        await group.add_client(clients[name])
    stream = group.start_stream(channel_resolver=channel_resolver_for({}))
    return _Group(clock, server, group, stream, clients, connections)


async def _commit_explicit(sc: _Group, index: int, play_start_us: int) -> None:
    sc.stream.prepare_audio(_slice_pcm(index), _SOURCE_FORMAT)
    await sc.stream.commit_audio(play_start_us=play_start_us)
    await asyncio.sleep(0)
    sc.clock.advance_us(_SLICE_US)


async def _commit_auto(sc: _Group, index: int, *, stall_us: int = 0) -> None:
    sc.stream.prepare_audio(_slice_pcm(index), _SOURCE_FORMAT)
    await sc.stream.commit_audio()
    await asyncio.sleep(0)
    sc.clock.advance_us(_SLICE_US + stall_us)


def _explicit_start_us(index: int, *, jump_us: int, lead_us: int = 2_000_000) -> int:
    start_us = lead_us + index * _SLICE_US
    return start_us + jump_us if index >= _JUMP_SLICE else start_us


def _format_state(codec: AudioCodec, sample_rate: int) -> ClientStatePayload:
    return ClientStatePayload(
        player=PlayerStatePayload(
            format=SupportedAudioFormat(
                codec=codec, channels=2, sample_rate=sample_rate, bit_depth=16
            )
        )
    )


async def test_forward_explicit_jump_is_a_gap_in_one_stream() -> None:
    """A forward play_start_us jump reaches every member as a gap of its size, nothing else."""
    sc = await _group(_MEMBERS)
    for index in range(_TOTAL_SLICES):
        await _commit_explicit(sc, index, _explicit_start_us(index, jump_us=_JUMP_US))

    for name, conn in sc.connections.items():
        assert len(encoded_segments_from_events(conn.events)) == 1, f"{name} restarted its stream"
        assert not any(
            isinstance(ev.payload, (StreamClearMessage, StreamEndMessage)) for ev in conn.events
        ), f"{name} got a stream command for an input jump"
        jumps = jumps_us(wire_spans(conn.events))
        assert jumps, f"{name} played across the jump without a gap"
        assert abs(sum(jumps) - _JUMP_US) <= _TOLERANCE_US, f"{name} jumps {jumps}"
        assert min(jumps) > 0, f"{name} overlaps its own audio: {jumps}"


async def test_backward_explicit_jump_never_overlaps_on_the_wire() -> None:
    """A backward play_start_us jump must not give any member overlapping timestamps."""
    sc = await _group(_MEMBERS)
    for index in range(_TOTAL_SLICES):
        await _commit_explicit(sc, index, _explicit_start_us(index, jump_us=-_JUMP_US))

    overlaps = {
        name: [jump for jump in jumps_us(wire_spans(conn.events)) if jump < 0]
        for name, conn in sc.connections.items()
    }
    assert all(not found for found in overlaps.values()), f"overlaps per member: {overlaps}"


async def test_catchup_replay_of_a_backward_jump_never_overlaps_on_the_wire() -> None:
    """A joiner replayed from the PCM cache across a backward jump must not get overlap."""
    sc = await _group(
        {"pcm-48k": (AudioCodec.PCM, 48_000), "flac-48k": (AudioCodec.FLAC, 48_000)},
        joiners=["flac-48k"],
    )
    join_slice = _JUMP_SLICE + 5
    for index in range(_TOTAL_SLICES):
        if index == join_slice:
            await sc.group.add_client(sc.clients["flac-48k"])
            assert sc.stream._catchup_state, "joiner did not replay from the PCM cache"  # noqa: SLF001
        await _commit_explicit(sc, index, _explicit_start_us(index, jump_us=-_JUMP_US))

    joiner_overlaps = [j for j in jumps_us(wire_spans(sc.connections["flac-48k"].events)) if j < 0]
    assert joiner_overlaps == [], f"joiner replay overlaps by {joiner_overlaps} us"


@pytest.mark.parametrize("pcm_cache", [False, True], ids=["fresh-key-live", "fresh-key-catchup"])
async def test_format_change_after_rebases_resumes_without_going_back(
    pcm_cache: bool,  # noqa: FBT001
) -> None:
    """After rebases, a new format starts no earlier than the old one ended, on the same content."""
    sc = await _group({"pcm-44k": (AudioCodec.PCM, 44_100)})
    # Start on FLAC, which holds back audio.
    changer, changer_conn = make_pool_player(
        sc.server,
        "changer",
        codec=AudioCodec.FLAC,
        sample_rate=48_000,
        extra_formats=[
            SupportedAudioFormat(
                codec=AudioCodec.OPUS, channels=2, sample_rate=48_000, bit_depth=16
            )
        ],
    )
    await sc.group.add_client(changer)
    sc.connections["changer"] = changer_conn
    if not pcm_cache:
        sc.stream.disable_pcm_cache_for_channel(MAIN_CHANNEL)
    role = changer.role("player@v1")
    assert role is not None
    catchups: list[object] = []
    start_catchup = sc.stream._start_catchup_encoding  # noqa: SLF001

    async def record_catchup(*args: Any, **kwargs: Any) -> None:
        catchups.append(args[0])
        await start_catchup(*args, **kwargs)

    sc.stream._start_catchup_encoding = record_catchup  # type: ignore[method-assign]  # noqa: SLF001

    stalls = {10 + 3 * n: 20_000 for n in range(8)}
    change_slice = 40
    for index in range(_TOTAL_SLICES):
        if index == change_slice:
            role.on_client_state(_format_state(AudioCodec.OPUS, 48_000))
            for _ in range(5):
                await asyncio.sleep(0)
        await _commit_auto(sc, index, stall_us=stalls.get(index, 0))
    assert bool(catchups) is pcm_cache

    spans = wire_spans(changer_conn.events)
    old = [span for span in spans if span.segment == 0]
    new = [span for span in spans if span.segment == 1]
    assert old
    assert new, "no audio in the new format"
    assert new[0].start_us >= old[-1].end_us - 1, (
        f"new format starts {old[-1].end_us - new[0].start_us:.1f}us before the old one ended"
    )

    reference = player_timeline(sc.connections["pcm-44k"].events)
    changed = player_timeline(changer_conn.events)
    offset_us = content_offset_us(
        reference, changed, window_start_us=int(new[0].start_us) + 300_000, window_us=500_000
    )
    assert abs(offset_us) <= _TOLERANCE_US, f"new format plays other content: {offset_us:+.1f}us"


async def test_live_rebases_keep_every_member_at_min_buffer() -> None:
    """In live mode every chunk ends at least min_buffer past its send time across rebases."""
    sc = await _group(
        {
            "pcm-48k": (AudioCodec.PCM, 48_000),
            "flac-48k": (AudioCodec.FLAC, 48_000),
            "opus-48k": (AudioCodec.OPUS, 48_000),
        }
    )
    sc.stream.set_live_source(True)
    sends: dict[str, _SendTimes] = {}
    for name, client in sc.clients.items():
        role = client.role("player@v1")
        assert role is not None
        role.min_buffer_ms = 150
        sends[name] = _SendTimes(sc.connections[name], sc.clock)

    stalls = {10 + 2 * n: 20_000 for n in range(20)}
    for index in range(_TOTAL_SLICES):
        await _commit_auto(sc, index, stall_us=stalls.get(index, 0))

    short: dict[str, tuple[int, float]] = {}
    for name, recorder in sends.items():
        spans = recorder.spans(sc.connections[name].events)
        lacking = [
            span.end_us - span.sent_at_us
            for span in spans
            if span.sent_at_us is not None and span.end_us - span.sent_at_us < 150_000
        ]
        if lacking:
            short[name] = (len(lacking), min(lacking))
    assert short == {}, f"members queued less than min_buffer (count, shortest us): {short}"
