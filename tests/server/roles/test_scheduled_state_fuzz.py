"""Deterministic convergence fuzzing for scheduled state deltas."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, fields, replace

from PIL import Image

from aiosendspin.client.connection import SendspinConnection, _merge_session_update
from aiosendspin.client.scheduled_state import ScheduledStateUpdate
from aiosendspin.client.time_sync import SendspinTimeFilter
from aiosendspin.models.artwork import ArtworkChannel
from aiosendspin.models.color import SessionUpdateColor
from aiosendspin.models.core import ServerStateMessage
from aiosendspin.models.metadata import SessionUpdateMetadata
from aiosendspin.models.types import ArtworkSource, PictureFormat, RepeatMode, UndefinedField
from aiosendspin.server.roles.artwork.group import ArtworkGroupRole
from aiosendspin.server.roles.artwork.types import ArtworkRoleProtocol
from aiosendspin.server.roles.color.group import ColorGroupRole
from aiosendspin.server.roles.color.state import Color
from aiosendspin.server.roles.metadata.group import MetadataGroupRole
from aiosendspin.server.roles.metadata.state import Metadata

_FUZZ_SEED = 0x5EED5A7E
_SCENARIO_COUNT = 2_000
_ASYNC_SCENARIO_COUNT = 200


class _Clock:
    def __init__(self) -> None:
        self.value = 1_000_000

    def now_us(self) -> int:
        return self.value


class _Server:
    def __init__(self, clock: _Clock) -> None:
        self.clock = clock


class _Group:
    def __init__(self) -> None:
        self.clock = _Clock()
        self._server = _Server(self.clock)
        self.has_active_stream = False

    def _signal_event(self, _event: object) -> None:
        return


class _Member:
    def __init__(self) -> None:
        self.messages: list[ServerStateMessage] = []

    def send_message(self, message: ServerStateMessage) -> None:
        self.messages.append(message)


@dataclass(slots=True)
class _ArtworkFrame:
    timestamp: int
    payload: bytes | None


def _metadata(
    rng: random.Random,
    scenario: int,
    step: int,
    previous: Metadata | None = None,
) -> Metadata:
    if previous is not None and rng.randrange(5) == 0:
        return replace(previous, timestamp_us=None)
    progress = rng.choice([None, rng.randint(0, 300_000)])
    return Metadata(
        title=f"title-{scenario}-{step}",
        artist=rng.choice([None, "Artist A", "Artist B"]),
        album_artist=rng.choice([None, "Album Artist"]),
        album=rng.choice([None, "Album A", "Album B"]),
        artwork_url=rng.choice([None, "https://example.test/art.jpg"]),
        year=rng.choice([None, rng.randint(1980, 2030)]),
        track=rng.choice([None, rng.randint(1, 20)]),
        repeat=rng.choice([None, *RepeatMode]),
        shuffle=rng.choice([None, False, True]),
        track_progress=progress,
        track_duration=None if progress is None else rng.randint(progress, 400_000),
        playback_speed=None if progress is None else rng.choice([0, 500, 1_000, 1_500]),
    )


def _color(
    rng: random.Random,
    scenario: int,
    step: int,
    previous: Color | None = None,
) -> Color:
    if previous is not None and rng.randrange(5) == 0:
        return previous
    return Color(
        background_dark=rng.choice([None, (0, 0, 0)]),
        background_light=rng.choice([None, (255, 255, 255)]),
        primary=(scenario % 256, step % 256, (scenario + step) % 256),
        accent=rng.choice([None, (20, 40, 60), (200, 160, 120)]),
        on_dark=rng.choice([None, (255, 255, 255)]),
        on_light=rng.choice([None, (0, 0, 0)]),
    )


def _assert_converges[T: SessionUpdateMetadata | SessionUpdateColor](
    states: list[T], update: T, *, context: str
) -> T:
    applied = [_merge_session_update(state, update) for state in states]
    expected = applied[0].to_dict()
    assert all(state.to_dict() == expected for state in applied), context
    return applied[0]


def _semantic_state(update: SessionUpdateMetadata | SessionUpdateColor) -> dict[str, object]:
    return {
        field.name: None if isinstance(value, UndefinedField) else value
        for field in fields(update)
        if field.name != "timestamp"
        for value in [getattr(update, field.name)]
    }


def _scheduled_timestamp(rng: random.Random, now_us: int, scenario: int, step: int) -> int:
    if (scenario + step) % 7 == 0:
        return now_us
    return now_us + rng.randint(1, 1_000_000)


def _assert_metadata_replay(role: MetadataGroupRole, now_us: int) -> None:
    member = _Member()
    role.on_member_join(member)  # type: ignore[arg-type]
    assert member.messages
    first = member.messages[0].payload.metadata
    assert isinstance(first, SessionUpdateMetadata)
    assert first.timestamp == now_us
    assert len(member.messages) == 1 + int(role._state.has_pending)  # noqa: SLF001
    if role._state.has_pending:  # noqa: SLF001
        pending = member.messages[1].payload.metadata
        assert pending is role._state.pending_update  # noqa: SLF001


def _assert_color_replay(role: ColorGroupRole, now_us: int) -> None:
    member = _Member()
    role.on_member_join(member)  # type: ignore[arg-type]
    assert member.messages
    first = member.messages[0].payload.color
    assert isinstance(first, SessionUpdateColor)
    assert first.timestamp == now_us
    assert len(member.messages) == 1 + int(role._state.has_pending)  # noqa: SLF001
    if role._state.has_pending:  # noqa: SLF001
        pending = member.messages[1].payload.color
        assert pending is role._state.pending_update  # noqa: SLF001


def _run_metadata_scenario(rng: random.Random, scenario: int) -> None:
    group = _Group()
    group.has_active_stream = scenario % 2 == 0
    role = MetadataGroupRole(group)  # type: ignore[arg-type]
    member = _Member()
    initial_metadata = _metadata(rng, scenario, 0)
    if group.has_active_stream and initial_metadata.track_progress is None:
        initial_metadata = replace(
            initial_metadata,
            track_progress=rng.randint(0, 300_000),
            track_duration=400_000,
            playback_speed=rng.choice([500, 1_000, 1_500]),
        )
    role.set_metadata(initial_metadata)
    role.subscribe(member)  # type: ignore[arg-type]
    initial = member.messages.pop().payload.metadata
    assert isinstance(initial, SessionUpdateMetadata)
    reachable = [initial]
    pending_timestamp = group.clock.value

    for step in range(1, rng.randint(2, 7)):
        if rng.choice([False, True]):
            group.clock.value = pending_timestamp + rng.randint(0, 1_000)
        current = role.metadata
        pending_timestamp = _scheduled_timestamp(rng, group.clock.value, scenario, step)
        target = None if rng.randrange(8) == 0 else _metadata(rng, scenario, step, current)
        role.set_metadata(target, timestamp_us=pending_timestamp)
        if not member.messages:
            continue
        update = member.messages.pop().payload.metadata
        assert isinstance(update, SessionUpdateMetadata)
        applied = _assert_converges(
            reachable,
            update,
            context=f"metadata scenario={scenario} step={step} update={update.to_dict()}",
        )
        reachable = [*reachable, applied] if pending_timestamp > group.clock.value else [applied]
        if rng.randrange(4) == 0:
            _assert_metadata_replay(role, group.clock.value)

    if role._state.has_pending:  # noqa: SLF001
        current = role.metadata
        role.set_metadata(
            None if current is None else replace(current, timestamp_us=None),
            timestamp_us=group.clock.value,
        )
        assert not role._state.has_pending  # noqa: SLF001
        update = member.messages.pop().payload.metadata
        assert isinstance(update, SessionUpdateMetadata)
        reachable = [
            _assert_converges(
                reachable,
                update,
                context=f"metadata scenario={scenario} unchanged cancellation",
            )
        ]

    group.clock.value += 1
    final = _metadata(rng, scenario, 99, role.metadata)
    role.set_metadata(final)
    if not member.messages:
        assert role.metadata is not None
        assert _semantic_state(role.metadata.snapshot_update(group.clock.value)) == _semantic_state(
            final.snapshot_update(group.clock.value)
        )
        return
    update = member.messages.pop().payload.metadata
    assert isinstance(update, SessionUpdateMetadata)
    applied = _assert_converges(
        reachable,
        update,
        context=f"metadata scenario={scenario} final update={update.to_dict()}",
    )
    assert _semantic_state(applied) == _semantic_state(final.snapshot_update(group.clock.value))


def _run_color_scenario(rng: random.Random, scenario: int) -> None:
    group = _Group()
    role = ColorGroupRole(group)  # type: ignore[arg-type]
    member = _Member()
    role.set_color(_color(rng, scenario, 0))
    role.subscribe(member)  # type: ignore[arg-type]
    initial = member.messages.pop().payload.color
    assert isinstance(initial, SessionUpdateColor)
    reachable = [initial]
    pending_timestamp = group.clock.value

    for step in range(1, rng.randint(2, 7)):
        if rng.choice([False, True]):
            group.clock.value = pending_timestamp + rng.randint(0, 1_000)
        current = role.color
        pending_timestamp = _scheduled_timestamp(rng, group.clock.value, scenario, step)
        target = None if rng.randrange(8) == 0 else _color(rng, scenario, step, current)
        role.set_color(target, timestamp_us=pending_timestamp)
        if not member.messages:
            continue
        update = member.messages.pop().payload.color
        assert isinstance(update, SessionUpdateColor)
        applied = _assert_converges(
            reachable,
            update,
            context=f"color scenario={scenario} step={step} update={update.to_dict()}",
        )
        reachable = [*reachable, applied] if pending_timestamp > group.clock.value else [applied]
        if rng.randrange(4) == 0:
            _assert_color_replay(role, group.clock.value)

    if role._state.has_pending:  # noqa: SLF001
        current = role.color
        role.set_color(
            current,
            timestamp_us=group.clock.value,
        )
        assert not role._state.has_pending  # noqa: SLF001
        update = member.messages.pop().payload.color
        assert isinstance(update, SessionUpdateColor)
        reachable = [
            _assert_converges(
                reachable,
                update,
                context=f"color scenario={scenario} unchanged cancellation",
            )
        ]

    group.clock.value += 1
    final = _color(rng, scenario, 99, role.color)
    role.set_color(final)
    if not member.messages:
        assert role.color is not None
        assert _semantic_state(role.color.snapshot_update(group.clock.value)) == _semantic_state(
            final.snapshot_update(group.clock.value)
        )
        return
    update = member.messages.pop().payload.color
    assert isinstance(update, SessionUpdateColor)
    applied = _assert_converges(
        reachable,
        update,
        context=f"color scenario={scenario} final update={update.to_dict()}",
    )
    assert _semantic_state(applied) == _semantic_state(final.snapshot_update(group.clock.value))


async def _drain_scheduler() -> None:
    await asyncio.sleep(0)
    await asyncio.sleep(0)


async def _run_client_scheduler_scenario(rng: random.Random, scenario: int) -> None:
    clock = _Clock()
    offset = rng.randint(-100_000, 100_000)
    connection = SendspinConnection.__new__(SendspinConnection)
    connection._time_filter = SendspinTimeFilter()  # noqa: SLF001
    connection._time_filter.update(offset, 100, clock.value)  # noqa: SLF001
    commits: list[_ArtworkFrame | None] = []
    state = ScheduledStateUpdate[_ArtworkFrame](
        map_to_client_time=connection._map_to_client_time,  # noqa: SLF001
        now_us=clock.now_us,
        commit=commits.append,
    )
    current = _ArtworkFrame(timestamp=clock.value + offset, payload=b"current")
    state.handle_update(current)
    assert commits == [current]

    pending = _ArtworkFrame(
        timestamp=connection._time_filter.compute_server_time(  # noqa: SLF001
            clock.value + rng.randint(1, 1_000_000)
        ),
        payload=rng.choice([None, b"first"]),
    )
    state.handle_update(pending)
    await asyncio.sleep(0)
    action = scenario % 5
    if action == 0:
        clock.value = connection._map_to_client_time(pending.timestamp)  # noqa: SLF001
        state.reschedule_pending()
        await _drain_scheduler()
        assert commits[-1] is pending
    elif action == 1:
        connection._time_filter.update(  # noqa: SLF001
            pending.timestamp - clock.value,
            100,
            clock.value + 1,
        )
        state.reschedule_pending()
        await _drain_scheduler()
        assert commits[-1] is pending
    elif action == 2:
        replacement = _ArtworkFrame(
            timestamp=connection._time_filter.compute_server_time(  # noqa: SLF001
                clock.value + rng.randint(1, 1_000_000)
            ),
            payload=rng.choice([None, b"replacement"]),
        )
        state.handle_update(replacement)
        clock.value = connection._map_to_client_time(replacement.timestamp)  # noqa: SLF001
        state.reschedule_pending()
        await _drain_scheduler()
        assert commits[-1] is replacement
        assert pending not in commits
    elif action == 3:
        state.discard_pending()
        await _drain_scheduler()
        assert commits == [current]
    else:
        state.clear_immediately()
        await _drain_scheduler()
        assert commits == [current, None]


async def _run_artwork_replay_scenario(rng: random.Random, scenario: int) -> None:
    group = _Group()
    role = ArtworkGroupRole(group)  # type: ignore[arg-type]
    current = Image.new("RGB", (2, 2), (scenario % 256, 0, 0))
    await role.set_album_artwork(current)
    pending = None
    pending_timestamp = group.clock.value + rng.randint(1, 1_000_000)
    if scenario % 3:
        pending = Image.new("RGB", (2, 2), (0, scenario % 256, 0))
    await role.set_album_artwork(pending, timestamp_us=pending_timestamp)
    if scenario % 2:
        pending_timestamp = group.clock.value + rng.randint(1, 1_000_000)
        pending = None if scenario % 5 == 0 else Image.new("RGB", (2, 2), (0, 0, scenario % 256))
        await role.set_album_artwork(pending, timestamp_us=pending_timestamp)
    if scenario % 4 == 0:
        group.clock.value = pending_timestamp

    sent: list[tuple[Image.Image | None, int]] = []

    async def capture(
        _role: ArtworkRoleProtocol,
        image: Image.Image | None,
        _channel: int,
        _config: ArtworkChannel,
        timestamp_us: int,
    ) -> None:
        sent.append((image, timestamp_us))

    role._encode_and_send_artwork = capture  # type: ignore[method-assign]  # noqa: SLF001
    channel = ArtworkChannel(
        source=ArtworkSource.ALBUM,
        format=PictureFormat.PNG,
        media_width=2,
        media_height=2,
    )
    await role._send_artwork_replay(  # noqa: SLF001
        object(),  # type: ignore[arg-type]
        0,
        channel,
    )
    if group.clock.value < pending_timestamp:
        assert sent
        assert [timestamp for _, timestamp in sent] == [group.clock.value, pending_timestamp]
        assert sent[0][0] is not None
        assert (sent[1][0] is None) == (pending is None)
    elif pending is None:
        assert not sent
    else:
        assert [timestamp for _, timestamp in sent] == [group.clock.value]
        assert sent[0][0] is not None


def test_scheduled_state_deltas_converge_under_application_uncertainty() -> None:
    """Every scheduled delta converges from all reachable applied states."""
    rng = random.Random(_FUZZ_SEED)  # noqa: S311
    for scenario in range(_SCENARIO_COUNT):
        if scenario % 2:
            _run_metadata_scenario(rng, scenario)
        else:
            _run_color_scenario(rng, scenario)


async def test_client_timing_and_artwork_replay_state_machines() -> None:
    """Client timers and artwork replay preserve current and pending invariants."""
    rng = random.Random(_FUZZ_SEED ^ 0xA47)  # noqa: S311
    for scenario in range(_ASYNC_SCENARIO_COUNT):
        await _run_client_scheduler_scenario(rng, scenario)
        await _run_artwork_replay_scenario(rng, scenario)


def test_clock_mapping_uses_each_available_filter_estimate() -> None:
    """Timestamp mapping uses the current estimate without waiting for convergence."""
    rng = random.Random(_FUZZ_SEED ^ 0xC10C)  # noqa: S311
    for scenario in range(_SCENARIO_COUNT):
        connection = SendspinConnection.__new__(SendspinConnection)
        connection._time_filter = SendspinTimeFilter()  # noqa: SLF001
        client_time = 1_000_000 + scenario * 10_000
        server_timestamp = rng.randint(1, 10_000_000_000)
        assert connection._map_to_client_time(server_timestamp) == server_timestamp  # noqa: SLF001

        offset = rng.randint(-500_000, 500_000)
        connection._time_filter.update(offset, 100, client_time)  # noqa: SLF001
        assert connection._map_to_client_time(server_timestamp) == server_timestamp - offset  # noqa: SLF001

        connection._time_filter.update(offset, 100, client_time + 1_000)  # noqa: SLF001
        assert connection._map_to_client_time(server_timestamp) == server_timestamp - offset  # noqa: SLF001
