"""Deterministic fuzzing of scheduled state between server group roles and a spec client."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from aiosendspin.clock import ManualClock
from aiosendspin.models.artwork import ArtworkChannel, ClientStateArtwork
from aiosendspin.models.color import SessionUpdateColor
from aiosendspin.models.core import ClientStatePayload, ServerStateMessage
from aiosendspin.models.metadata import SessionUpdateMetadata
from aiosendspin.models.types import ArtworkSource, PictureFormat, UndefinedField
from aiosendspin.server.connection import SendspinConnection
from aiosendspin.server.roles.artwork.v1 import MAX_ANNOUNCE_LEAD_US, ArtworkV1Role
from aiosendspin.server.roles.base import MAX_SCHEDULED_LEAD_US
from aiosendspin.server.roles.color.group import ColorGroupRole
from aiosendspin.server.roles.color.state import Color
from aiosendspin.server.roles.metadata.group import MetadataGroupRole
from aiosendspin.server.roles.metadata.state import Metadata
from aiosendspin.server.roles.scheduled_state import ScheduledRoleState

_FUZZ_SEED = 0x5EED5A7E
_SCENARIO_COUNT = 400
_STEP_COUNT = 30
# How far ahead a client's estimate of the server clock may be when a message arrives.
# Transit delay keeps a message from arriving before its send time on that estimate.
_CLOCK_ERROR_US = 2_000

_StateObject = SessionUpdateMetadata | SessionUpdateColor


@dataclass(order=True)
class _Timer:
    fire_at_us: int
    seq: int
    callback: Callable[[], None] = field(compare=False)
    cancelled: bool = field(default=False, compare=False)

    def cancel(self) -> None:
        self.cancelled = True


class _Loop:
    """Event loop stand-in whose timers fire on the manual server clock."""

    def __init__(self, clock: ManualClock) -> None:
        self._clock = clock
        self.timers: list[_Timer] = []

    def call_later(self, delay_s: float, callback: Callable[[], None]) -> _Timer:
        timer = _Timer(
            self._clock.now_us() + round(delay_s * 1_000_000), len(self.timers), callback
        )
        self.timers.append(timer)
        return timer

    def next_due(self, until_us: int) -> _Timer | None:
        due = [t for t in self.timers if not t.cancelled and t.fire_at_us <= until_us]
        if not due:
            return None
        timer = min(due)
        self.timers.remove(timer)
        return timer


class _SpecClient:
    """A client keeping current state plus one pending update per the spec."""

    def __init__(self, error_us: int) -> None:
        self.error_us = error_us
        self.current: _StateObject | None = None
        self.pending: _StateObject | None = None
        self.received = 0

    def receive(self, state: _StateObject | None, server_now_us: int) -> None:
        if self.received == 0:
            # messaging.md: the first server/state carries a past or present timestamp.
            assert state is None or state.timestamp <= server_now_us
        self.received += 1
        self.pending = None
        if state is not None and state.timestamp > server_now_us + self.error_us:
            self.pending = state
        else:
            self.current = state

    def promote(self, server_now_us: int) -> None:
        if self.pending is not None and self.pending.timestamp <= server_now_us + self.error_us:
            self.current = self.pending
            self.pending = None


class _Member:
    def __init__(self, client: _SpecClient) -> None:
        self.client = client
        self.outbox: list[ServerStateMessage] = []

    def send_message(self, message: ServerStateMessage) -> None:
        self.outbox.append(message)


class _Harness:
    """Drives one group role while its members' clients receive what it sends."""

    def __init__(self, rng: random.Random, name: str, role: Any, clock: ManualClock) -> None:
        self.rng = rng
        self.name = name
        self.role = role
        self.clock = clock
        self.loop: _Loop = role._group._server.loop  # noqa: SLF001
        self.members: list[_Member] = []
        self.connection = SendspinConnection.__new__(SendspinConnection)
        self.connection._server = MagicMock()  # noqa: SLF001
        self.connection._server.clock = clock  # noqa: SLF001

    def join(self) -> None:
        member = _Member(_SpecClient(self.rng.randint(0, _CLOCK_ERROR_US)))
        self.members.append(member)
        self.role.subscribe(member)

    def deliver(self) -> None:
        """Deliver every queued message, coalescing as the connection writer would."""
        now_us = self.clock.now_us()
        for member in self.members:
            member.client.promote(now_us)
            outbox, member.outbox = member.outbox, []
            while outbox:
                message = outbox.pop(0)
                while outbox and self.rng.random() < 0.5:
                    merged = self.connection._merge_state_messages(message, outbox[0])  # noqa: SLF001
                    if merged is None:
                        break
                    assert isinstance(merged, ServerStateMessage)
                    message = merged
                    outbox.pop(0)
                state = getattr(message.payload, self.name)
                assert not isinstance(state, UndefinedField)
                member.client.receive(state, now_us)

    def advance(self, delta_us: int) -> None:
        """Advance the clock, firing deferred sends and delivering at their times."""
        end_us = self.clock.now_us() + delta_us
        self.deliver()
        while (timer := self.loop.next_due(end_us)) is not None:
            self.clock.now_us_value = max(self.clock.now_us(), timer.fire_at_us)
            self.deliver()
            timer.callback()
            self.deliver()
        self.clock.now_us_value = end_us
        self.deliver()

    def check(self, snapshot: Callable[[Any], dict[str, Any] | None]) -> None:
        """Assert every client agrees with the server away from any scheduled boundary."""
        now_us = self.clock.now_us()
        state = self.role._state  # noqa: SLF001
        boundaries = [state.pending_timestamp_us] + [
            member.client.pending.timestamp
            for member in self.members
            if member.client.pending is not None
        ]
        if any(b is not None and abs(b - now_us) <= 2 * _CLOCK_ERROR_US for b in boundaries):
            return
        expected_current = snapshot(state.current(now_us))
        scheduled_us = state.pending_timestamp_us
        announced = scheduled_us is not None and scheduled_us - now_us <= MAX_SCHEDULED_LEAD_US
        expected_pending = snapshot(state.pending) if announced else None
        for member in self.members:
            member.client.promote(now_us)
            assert _state_fields(member.client.current, now_us) == expected_current, self.name
            if announced:
                assert member.client.pending is not None, self.name
                assert member.client.pending.timestamp == scheduled_us
            else:
                assert member.client.pending is None, self.name
            assert _state_fields(member.client.pending, None) == expected_pending, self.name


def _state_fields(state: _StateObject | None, now_us: int | None) -> dict[str, Any] | None:
    """Return a state object's fields, progress extrapolated to `now_us` when given."""
    if state is None:
        return None
    fields = state.to_dict()
    timestamp = fields.pop("timestamp")
    progress = fields.pop("progress", None)
    if progress is not None and now_us is not None:
        elapsed_ms = (now_us - timestamp) * progress["playback_speed"] // 1_000_000
        position = max(0, progress["track_progress"] + elapsed_ms)
        if progress["track_duration"]:
            position = min(position, progress["track_duration"])
        progress = {**progress, "track_progress": position}
    if progress is not None:
        fields["progress"] = progress
    return fields


def _make_group(clock: ManualClock) -> MagicMock:
    group = MagicMock()
    group._server.clock = clock  # noqa: SLF001
    group._server.loop = _Loop(clock)  # noqa: SLF001
    group.has_active_stream = True
    return group


def _random_metadata(rng: random.Random, timestamp_us: int | None) -> Metadata:
    return Metadata(
        title=rng.choice(["A", "B", "C"]),
        artist=rng.choice([None, "Artist"]),
        track_progress=rng.randint(0, 100_000),
        track_duration=rng.choice([0, 200_000]),
        playback_speed=rng.choice([0, 1000]),
        timestamp_us=timestamp_us,
    )


def _random_color(rng: random.Random) -> Color:
    return Color(primary=rng.choice([None, (1, 2, 3), (4, 5, 6)]), accent=(7, 8, 9))


def _run_state_scenario(rng: random.Random, *, metadata: bool) -> None:
    clock = ManualClock(now_us_value=1_000_000)
    group = _make_group(clock)
    if metadata:
        role: Any = MetadataGroupRole(group)
        harness = _Harness(rng, "metadata", role, clock)

        def snapshot(state: Metadata | None) -> dict[str, Any] | None:
            if state is None:
                return None
            assert state.timestamp_us is not None
            progress_at = clock.now_us() if state.timestamp_us <= clock.now_us() else None
            return _state_fields(state.snapshot_update(state.timestamp_us), progress_at)

        def set_now() -> None:
            role.set_metadata(_random_metadata(rng, None))

        def schedule(timestamp_us: int) -> None:
            role.set_metadata(_random_metadata(rng, timestamp_us))

    else:
        role = ColorGroupRole(group)
        harness = _Harness(rng, "color", role, clock)

        def snapshot(state: Color | None) -> dict[str, Any] | None:
            return None if state is None else _state_fields(state.snapshot_update(0), None)

        def set_now() -> None:
            role.set_color(_random_color(rng))

        def schedule(timestamp_us: int) -> None:
            role.set_color(_random_color(rng), timestamp_us=timestamp_us)

    harness.join()
    for _ in range(_STEP_COUNT):
        scheduled_us = role._state.pending_timestamp_us  # noqa: SLF001
        if scheduled_us is not None and abs(scheduled_us - clock.now_us()) <= 2 * _CLOCK_ERROR_US:
            # Clients may already have applied it; a change now races that by design.
            harness.advance(3 * _CLOCK_ERROR_US)
        action = rng.randrange(8)
        if action == 0:
            set_now()
        elif action in (1, 2):
            schedule(clock.now_us() + rng.choice([5_000, 1_000_000, 19_000_000, 25_000_000]))
        elif action == 3:
            role.cancel_scheduled()
        elif action == 4:
            role.clear()
        elif action == 5:
            harness.join()
        else:
            harness.advance(rng.choice([1_000, 500_000, 4_000_000, 10_000_000]))
            harness.check(snapshot)
            continue
        if rng.random() < 0.7:
            harness.deliver()
            harness.check(snapshot)
    harness.advance(60_000_000)
    harness.check(snapshot)


def test_scheduled_metadata_and_color_reach_clients_consistently() -> None:
    """Clients following the spec hold the server's current and announced scheduled state."""
    rng = random.Random(_FUZZ_SEED)  # noqa: S311
    for scenario in range(_SCENARIO_COUNT):
        _run_state_scenario(rng, metadata=scenario % 2 == 0)


@dataclass
class _ArtworkChannelModel:
    """A spec client's view of one artwork channel."""

    current: bytes | None = None
    pending: tuple[int, bytes] | None = None
    transfer: tuple[int, int, bytearray] | None = None


class _ArtworkClient:
    def __init__(self, clock: ManualClock) -> None:
        self.clock = clock
        self.channels = [_ArtworkChannelModel(), _ArtworkChannelModel()]
        self.in_flight: int | None = None

    def receive(self, data: bytes) -> None:
        channel = data[0] - 8
        model = self.channels[channel]
        self.promote()
        if data[1] == 0x01:
            model.pending = None
            model.transfer = None
            if self.in_flight == channel:
                self.in_flight = None
        elif data[1] == 0x02:
            assert self.in_flight is None, "announce while a transfer is in flight"
            timestamp_us = int.from_bytes(data[2:10], "big", signed=True)
            size = int.from_bytes(data[10:14], "big")
            assert timestamp_us - self.clock.now_us() <= MAX_ANNOUNCE_LEAD_US
            model.pending = None
            model.transfer = (timestamp_us, size, bytearray())
            self.in_flight = channel
        else:
            assert self.in_flight == channel, "part without a transfer in flight"
            assert model.transfer is not None
            model.transfer[2].extend(data[2:])
        if self.in_flight == channel and model.transfer is not None:
            timestamp_us, size, received = model.transfer
            assert len(received) <= size
            if len(received) == size:
                model.pending = (timestamp_us, bytes(received))
                model.transfer = None
                self.in_flight = None
        self.promote()

    def promote(self) -> None:
        for model in self.channels:
            if model.pending is not None and model.pending[0] <= self.clock.now_us():
                model.current = model.pending[1] or None
                model.pending = None


async def _run_artwork_scenario(rng: random.Random) -> None:
    clock = ManualClock(now_us_value=1_000_000)
    client_stub = MagicMock()
    client_stub.info.artwork_support = None
    client_stub.group.group_role.return_value = None
    client_stub._server.clock = clock  # noqa: SLF001
    written = asyncio.Semaphore(0)

    async def _wait_drained(_role_family: str) -> None:
        await written.acquire()

    client_stub.wait_role_drained = AsyncMock(side_effect=_wait_drained)
    spec_client = _ArtworkClient(clock)
    client_stub.send_binary.side_effect = lambda data, **_: spec_client.receive(data)
    role = ArtworkV1Role(client=client_stub)
    channel = ArtworkChannel(
        source=ArtworkSource.ALBUM, format=PictureFormat.PNG, width=1, height=1
    )
    role.on_client_state(
        ClientStatePayload(available=True, artwork=ClientStateArtwork(channels=[channel] * 2))
    )
    # What the group role asks for, per channel.
    intents: list[ScheduledRoleState[bytes]] = [ScheduledRoleState(), ScheduledRoleState()]

    async def settle() -> None:
        for _ in range(3):
            role._queue_changed.set()  # noqa: SLF001
            for _ in range(10):
                written.release()
                await asyncio.sleep(0)

    for step in range(_STEP_COUNT):
        target = rng.randrange(2)
        now_us = clock.now_us()
        intents[target].current(now_us)
        action = rng.randrange(6)
        image = rng.choice([b"", b"x" * rng.randint(1, 40_000)])
        if action == 0:
            intents[target].apply(image or None, now_us)
            role.send_artwork(target, image, now_us)
        elif action in (1, 2):
            timestamp_us = now_us + rng.choice([1_000, 1_000_000, MAX_ANNOUNCE_LEAD_US + 1])
            intents[target].schedule(image or None, timestamp_us)
            role.send_artwork(target, image, timestamp_us)
        elif action == 3:
            if intents[target].pending_timestamp_us is not None:
                intents[target].apply(intents[target].current(now_us), now_us)
                assert role.cancel_scheduled_artwork(target)
        else:
            clock.advance_us(rng.choice([1, 500_000, 2_000_000]))
        if rng.random() < 0.5 or step == _STEP_COUNT - 1:
            await settle()

    await settle()
    clock.advance_us(MAX_ANNOUNCE_LEAD_US)
    await settle()
    clock.advance_us(MAX_ANNOUNCE_LEAD_US + 1)
    await settle()
    spec_client.promote()
    role.on_disconnect()
    for intent, model in zip(intents, spec_client.channels, strict=True):
        assert model.current == intent.current(clock.now_us())
        assert model.pending is None
        assert model.transfer is None


async def test_scheduled_artwork_reaches_client_consistently() -> None:
    """A spec client ends up showing the last image the server made current per channel."""
    rng = random.Random(_FUZZ_SEED ^ 0xA47)  # noqa: S311
    for _ in range(_SCENARIO_COUNT // 4):
        await _run_artwork_scenario(rng)


def test_state_fields_extrapolate_progress() -> None:
    """The fuzz comparison extrapolates progress like a spec client."""
    update = SessionUpdateMetadata.from_dict(
        {
            "timestamp": 0,
            "title": "A",
            "progress": {"track_progress": 10, "track_duration": 15, "playback_speed": 1000},
        }
    )
    assert _state_fields(update, 1_000) == {
        "title": "A",
        "progress": {"track_progress": 11, "track_duration": 15, "playback_speed": 1000},
    }
    assert _state_fields(update, 60_000_000) == {
        "title": "A",
        "progress": {"track_progress": 15, "track_duration": 15, "playback_speed": 1000},
    }
