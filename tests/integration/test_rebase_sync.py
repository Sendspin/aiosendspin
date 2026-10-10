"""Group sync across forward timeline rebases in auto-timestamp mode."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

import numpy as np
import pytest

from aiosendspin.models.types import AudioCodec
from aiosendspin.server.audio import AudioFormat
from aiosendspin.server.channels import MAIN_CHANNEL
from aiosendspin.server.client import SendspinClient
from aiosendspin.server.clock import ManualClock
from aiosendspin.server.group import SendspinGroup
from aiosendspin.server.push_stream import PushStream
from aiosendspin.server.roles import AudioRequirements
from tests.integration.sync_assertions import (
    PlayerTimeline,
    band_limited_noise,
    content_offset_us,
    player_timeline,
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
_TOTAL_SLICES = 80
_CUSTOM_CHANNEL = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
_CONTENT = (band_limited_noise(_SLICE_FRAMES * _TOTAL_SLICES, _SOURCE_RATE) * 32767).astype(
    np.int16
)


def _slice_pcm(index: int) -> bytes:
    return _CONTENT[index * _SLICE_FRAMES : (index + 1) * _SLICE_FRAMES].tobytes()


def _tail_window_start(timelines: Sequence[PlayerTimeline], *, window_us: int = 1_000_000) -> int:
    """Start of a window that ends 500 ms before the shortest player's audio ends."""
    end = min(t.end_us for t in timelines) - 500_000 - window_us
    assert end > max(t.start_us for t in timelines), "not enough shared audio"
    return int(end)


@dataclass(slots=True)
class _Scenario:
    """Players on one group, fed 100 ms slices in auto-timestamp mode."""

    clock: ManualClock
    stream: PushStream
    group: SendspinGroup
    clients: dict[str, SendspinClient]
    connections: dict[str, CaptureConnection]

    async def commit(
        self,
        index: int,
        *,
        stall_us: int = 0,
        channels: Sequence[UUID] = (MAIN_CHANNEL,),
    ) -> None:
        """Commit one slice, then let the clock run that slice plus a producer stall."""
        for channel_id in channels:
            self.stream.prepare_audio(_slice_pcm(index), _SOURCE_FORMAT, channel_id=channel_id)
        await self.stream.commit_audio()
        await asyncio.sleep(0)
        self.clock.advance_us(_SLICE_US + stall_us)

    def offsets_us(self, reference: str, *, window: str) -> dict[str, float]:
        """Offset of every other player against `reference` in the head or tail window."""
        timelines = {name: player_timeline(conn.events) for name, conn in self.connections.items()}
        if window == "head":
            start_us = max(t.start_us for t in timelines.values()) + 200_000
        else:
            start_us = _tail_window_start(list(timelines.values()))
        return {
            name: content_offset_us(timelines[reference], timeline, window_start_us=start_us)
            for name, timeline in timelines.items()
            if name != reference
        }


async def _scenario(
    players: dict[str, tuple[AudioCodec, int]],
    *,
    grouped: Sequence[str],
    channels: dict[str, UUID] | None = None,
) -> _Scenario:
    loop = asyncio.get_running_loop()
    clock = ManualClock()
    server = DummyServer(loop=loop, clock=clock)
    clients: dict[str, SendspinClient] = {}
    connections: dict[str, CaptureConnection] = {}
    for name, (codec, rate) in players.items():
        clients[name], connections[name] = make_pool_player(
            server, name, codec=codec, sample_rate=rate
        )
    group = clients[grouped[0]].group
    for name in grouped[1:]:
        await group.add_client(clients[name])
    stream = group.start_stream(channel_resolver=channel_resolver_for(channels or {}))
    return _Scenario(clock, stream, group, clients, connections)


def _report(offsets_us: dict[str, float]) -> str:
    return ", ".join(f"{name} {value:+.1f}us" for name, value in offsets_us.items())


# One 44.1 kHz sample is 22.7 us.
_TOLERANCE_US = 50


async def test_resampled_member_stays_aligned_across_rebases() -> None:
    """A rebase that re-anchors the resampler must not move resampled content."""
    sc = await _scenario(
        {"pcm-48k": (AudioCodec.PCM, 48_000), "pcm-44k": (AudioCodec.PCM, 44_100)},
        grouped=["pcm-48k", "pcm-44k"],
    )
    stalls = {20: 30_000, 30: 30_000, 40: 30_000, 50: 30_000}
    for index in range(_TOTAL_SLICES):
        await sc.commit(index, stall_us=stalls.get(index, 0))

    head = sc.offsets_us("pcm-48k", window="head")
    tail = sc.offsets_us("pcm-48k", window="tail")
    assert abs(tail["pcm-44k"]) <= _TOLERANCE_US, (
        f"after 4 rebases of 30 ms the resampled player is off: {_report(tail)} "
        f"(before the rebases: {_report(head)}, tolerance {_TOLERANCE_US}us)"
    )


async def test_codecs_stay_aligned_across_accumulated_rebases() -> None:
    """Encoders with different frame sizes must follow accumulated rebases together."""
    sc = await _scenario(
        {
            "pcm-48k": (AudioCodec.PCM, 48_000),
            "flac-48k": (AudioCodec.FLAC, 48_000),
            "opus-48k": (AudioCodec.OPUS, 48_000),
        },
        grouped=["pcm-48k", "flac-48k", "opus-48k"],
    )
    stalls = {10 + 2 * n: 20_000 for n in range(30)}
    for index in range(_TOTAL_SLICES):
        await sc.commit(index, stall_us=stalls.get(index, 0))

    head = sc.offsets_us("pcm-48k", window="head")
    tail = sc.offsets_us("pcm-48k", window="tail")
    assert all(abs(v) <= _TOLERANCE_US for v in tail.values()), (
        f"after 30 rebases of 20 ms the codecs disagree: {_report(tail)} "
        f"(before the rebases: {_report(head)}, tolerance {_TOLERANCE_US}us)"
    )


async def test_flac_member_follows_rebase_before_its_first_packet() -> None:
    """A fresh FLAC encoder must follow a rebase that lands before it emits a packet."""
    sc = await _scenario(
        {"pcm-44k": (AudioCodec.PCM, 44_100), "flac-44k": (AudioCodec.FLAC, 44_100)},
        grouped=["pcm-44k", "flac-44k"],
    )
    # One 100 ms slice is 4410 frames, short of the 4608 FLAC needs for its first packet.
    for index in range(_TOTAL_SLICES):
        await sc.commit(index, stall_us=30_000 if index == 0 else 0)

    tail = sc.offsets_us("pcm-44k", window="tail")
    assert abs(tail["flac-44k"]) <= _TOLERANCE_US, (
        f"after a 30 ms rebase on the second commit: {_report(tail)} (tolerance {_TOLERANCE_US}us)"
    )


async def test_shared_encoder_follows_rebase_after_a_member_leaves() -> None:
    """A member leaving between a stall and the next commit must not hide the rebase."""
    sc = await _scenario(
        {
            "pcm-48k": (AudioCodec.PCM, 48_000),
            "flac-a": (AudioCodec.FLAC, 48_000),
            "flac-b": (AudioCodec.FLAC, 48_000),
        },
        grouped=["pcm-48k", "flac-a", "flac-b"],
    )
    for index in range(_TOTAL_SLICES):
        if index == 31:
            await sc.group.remove_client(sc.clients["flac-b"])
        await sc.commit(index, stall_us=30_000 if index == 30 else 0)
    del sc.connections["flac-b"]

    tail = sc.offsets_us("pcm-48k", window="tail")
    assert abs(tail["flac-a"]) <= _TOLERANCE_US, (
        f"after a 30 ms rebase and a member leaving: {_report(tail)} (tolerance {_TOLERANCE_US}us)"
    )


@pytest.mark.parametrize("replay", [False, True], ids=["live-tail-join", "pcm-cache-replay"])
async def test_late_joiner_encoder_matches_live_encoder_after_rebases(replay: bool) -> None:  # noqa: FBT001
    """A joiner on a new encoder must land on the timeline the live members play."""
    sc = await _scenario(
        {"pcm-48k": (AudioCodec.PCM, 48_000), "opus-joiner": (AudioCodec.OPUS, 48_000)},
        grouped=["pcm-48k"],
    )
    stalls = {20: 50_000, 25: 50_000, 30: 50_000, 35: 50_000}
    # Producing faster than realtime builds a lead past the send-ahead, so the join replays.
    bursts = {40, 41, 42, 43, 44} if replay else set()
    for index in range(_TOTAL_SLICES):
        if index == 50:
            await sc.group.add_client(sc.clients["opus-joiner"])
            assert bool(sc.stream._catchup_state) is replay  # noqa: SLF001
        stall_us = stalls.get(index, 0) - (_SLICE_US if index in bursts else 0)
        await sc.commit(index, stall_us=stall_us)

    tail = sc.offsets_us("pcm-48k", window="tail")
    assert abs(tail["opus-joiner"]) <= _TOLERANCE_US, (
        f"joiner after 4 rebases of 50 ms is off: {_report(tail)} (tolerance {_TOLERANCE_US}us)"
    )


async def test_encoderless_member_matches_encoder_members_after_rebase() -> None:
    """A role without a transformer must stay on the timeline the encoder roles play."""
    sc = await _scenario(
        {"no-encoder": (AudioCodec.PCM, 48_000), "pcm-48k": (AudioCodec.PCM, 48_000)},
        grouped=["no-encoder", "pcm-48k"],
    )
    role = sc.clients["no-encoder"].role("player@v1")
    assert role is not None
    role._audio_requirements = AudioRequirements(  # noqa: SLF001
        sample_rate=48_000, bit_depth=16, channels=2, transformer=None
    )
    for index in range(_TOTAL_SLICES):
        await sc.commit(index, stall_us=30_000 if index == 20 else 0)

    head = sc.offsets_us("pcm-48k", window="head")
    tail = sc.offsets_us("pcm-48k", window="tail")
    assert abs(tail["no-encoder"]) <= _TOLERANCE_US, (
        f"after one rebase of 30 ms: {_report(tail)} "
        f"(before the rebase: {_report(head)}, tolerance {_TOLERANCE_US}us)"
    )


async def test_channel_that_misses_a_commit_stays_aligned() -> None:
    """A channel skipped for one commit must resume on the shared timeline."""
    sc = await _scenario(
        {"main": (AudioCodec.PCM, 48_000), "custom": (AudioCodec.PCM, 48_000)},
        grouped=["main", "custom"],
        channels={"custom": _CUSTOM_CHANNEL},
    )
    for index in range(_TOTAL_SLICES):
        channels = (MAIN_CHANNEL,) if index == 40 else (MAIN_CHANNEL, _CUSTOM_CHANNEL)
        await sc.commit(index, channels=channels)

    head = sc.offsets_us("main", window="head")
    tail = sc.offsets_us("main", window="tail")
    assert abs(tail["custom"]) <= _TOLERANCE_US, (
        f"after the custom channel skipped one 100 ms commit: {_report(tail)} "
        f"(before the skip: {_report(head)}, tolerance {_TOLERANCE_US}us)"
    )
