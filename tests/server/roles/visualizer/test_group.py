"""Tests for VisualizerGroupRole."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aiosendspin.models.visualizer import BeatAvailability, BeatTiming
from aiosendspin.server.roles.visualizer.group import VisualizerGroupRole
from aiosendspin.server.roles.visualizer.types import VisualizerRoleProtocol


def _make_group_stub(*, now_us: int = 0) -> MagicMock:
    """Create a mock group for testing."""
    group = MagicMock()
    group._server.clock.now_us.return_value = now_us  # noqa: SLF001
    return group


def _make_member_stub(*, wants_beats: bool = True) -> MagicMock:
    """Create a member mock that satisfies VisualizerRoleProtocol."""
    member = MagicMock(spec=VisualizerRoleProtocol)
    member.wants_beats = wants_beats
    return member


def test_visualizer_group_role_family() -> None:
    """VisualizerGroupRole has role_family of 'visualizer'."""
    group = _make_group_stub()
    vgr = VisualizerGroupRole(group)
    assert vgr.role_family == "visualizer"


def test_visualizer_group_role_subscribe() -> None:
    """VisualizerGroupRole accepts subscriptions."""
    group = _make_group_stub()
    vgr = VisualizerGroupRole(group)

    member = _make_member_stub()
    vgr.subscribe(member)

    assert member in vgr._members  # noqa: SLF001


def test_visualizer_group_role_unsubscribe() -> None:
    """VisualizerGroupRole handles unsubscriptions."""
    group = _make_group_stub()
    vgr = VisualizerGroupRole(group)

    member = _make_member_stub()
    vgr.subscribe(member)
    vgr.unsubscribe(member)

    assert member not in vgr._members  # noqa: SLF001


def test_append_beat_schedule_broadcasts() -> None:
    """append_beat_schedule pushes beats to every subscribed member."""
    group = _make_group_stub()
    vgr = VisualizerGroupRole(group)

    member_a = _make_member_stub()
    member_b = _make_member_stub()
    vgr.subscribe(member_a)
    vgr.subscribe(member_b)

    beats = [BeatTiming(100), BeatTiming(200, is_downbeat=True), BeatTiming(300)]
    vgr.append_beat_schedule(beats)

    member_a.append_beats.assert_called_once_with(beats)
    member_b.append_beats.assert_called_once_with(beats)


def test_beats_wanted_false_when_no_members() -> None:
    """beats_wanted is False with no subscribers."""
    vgr = VisualizerGroupRole(_make_group_stub())
    assert vgr.beats_wanted is False


def test_beats_wanted_true_when_member_wants_beats() -> None:
    """beats_wanted is True when a subscribed member wants beats."""
    vgr = VisualizerGroupRole(_make_group_stub())
    member = _make_member_stub()
    member.wants_beats = True
    vgr.subscribe(member)
    assert vgr.beats_wanted is True


def test_beats_wanted_false_after_last_wanter_leaves() -> None:
    """beats_wanted drops to False once the only beat-wanting member unsubscribes."""
    vgr = VisualizerGroupRole(_make_group_stub())
    member = _make_member_stub()
    member.wants_beats = True
    vgr.subscribe(member)
    vgr.unsubscribe(member)
    assert vgr.beats_wanted is False


def test_beats_wanted_false_when_no_member_wants() -> None:
    """beats_wanted is False when members exist but none want beats."""
    vgr = VisualizerGroupRole(_make_group_stub())
    member = _make_member_stub()
    member.wants_beats = False
    vgr.subscribe(member)
    assert vgr.beats_wanted is False


def test_append_beat_schedule_extends_existing() -> None:
    """A second append extends the stored schedule and pushes only the new tail."""
    group = _make_group_stub()
    vgr = VisualizerGroupRole(group)

    member = _make_member_stub()
    vgr.subscribe(member)

    vgr.append_beat_schedule([BeatTiming(100), BeatTiming(200)])
    vgr.append_beat_schedule([BeatTiming(300), BeatTiming(400)])

    assert [b.timestamp_us for b in vgr.current_beats] == [100, 200, 300, 400]
    assert member.append_beats.call_count == 2
    second_call = member.append_beats.call_args_list[1]
    assert [b.timestamp_us for b in second_call.args[0]] == [300, 400]


def test_append_beat_schedule_rejects_non_monotonic() -> None:
    """Out-of-order timestamps raise ValueError."""
    group = _make_group_stub()
    vgr = VisualizerGroupRole(group)
    vgr.append_beat_schedule([BeatTiming(100), BeatTiming(200)])
    with pytest.raises(ValueError, match="strictly increasing"):
        vgr.append_beat_schedule([BeatTiming(150)])


def test_append_beat_schedule_empty_is_noop() -> None:
    """Empty append does not broadcast."""
    group = _make_group_stub()
    vgr = VisualizerGroupRole(group)

    member = _make_member_stub()
    vgr.subscribe(member)
    vgr.append_beat_schedule([])

    member.append_beats.assert_not_called()


def test_clear_beat_schedule_broadcasts_clear() -> None:
    """clear_beat_schedule wipes state and calls clear_beats on every member."""
    group = _make_group_stub()
    vgr = VisualizerGroupRole(group)

    member = _make_member_stub()
    vgr.subscribe(member)

    vgr.append_beat_schedule([BeatTiming(42)])
    member.reset_mock()
    vgr.clear_beat_schedule()

    assert vgr.current_beats == []
    member.clear_beats.assert_called_once_with()
    member.append_beats.assert_not_called()


def test_on_member_join_replays_clear_then_append() -> None:
    """A late joiner gets a clear followed by the current schedule."""
    group = _make_group_stub()
    vgr = VisualizerGroupRole(group)
    vgr.append_beat_schedule([BeatTiming(10), BeatTiming(20)])

    member = _make_member_stub()
    vgr.subscribe(member)

    member.clear_beats.assert_called_once_with()
    member.append_beats.assert_called_once()
    assert [b.timestamp_us for b in member.append_beats.call_args.args[0]] == [10, 20]


def test_on_member_join_empty_schedule_only_clears() -> None:
    """A late joiner with no schedule sees only the clear."""
    group = _make_group_stub()
    vgr = VisualizerGroupRole(group)

    member = _make_member_stub()
    vgr.subscribe(member)

    member.clear_beats.assert_called_once_with()
    member.append_beats.assert_not_called()


def test_non_protocol_members_skipped() -> None:
    """Members lacking the VisualizerRoleProtocol surface are silently skipped."""
    group = _make_group_stub()
    vgr = VisualizerGroupRole(group)

    plain_member = MagicMock(spec=[])
    vgr.subscribe(plain_member)

    vgr.append_beat_schedule([BeatTiming(1), BeatTiming(2)])
    # No exception, no spurious method calls fabricated.
    assert not hasattr(plain_member, "append_beats") or not plain_member.append_beats.called


def test_set_beat_availability_propagates_to_members() -> None:
    """Group propagates beat availability to subscribed roles."""
    group = _make_group_stub()
    vgr = VisualizerGroupRole(group)
    member = _make_member_stub()
    vgr.subscribe(member)
    member.set_beat_availability.reset_mock()

    vgr.set_beat_availability(BeatAvailability.UNAVAILABLE)

    member.set_beat_availability.assert_called_once_with(BeatAvailability.UNAVAILABLE)
    assert vgr.beat_availability is BeatAvailability.UNAVAILABLE


def test_append_beat_schedule_noop_when_unavailable() -> None:
    """UNAVAILABLE blocks beat schedule from being broadcast."""
    group = _make_group_stub()
    vgr = VisualizerGroupRole(group)
    member = _make_member_stub()
    vgr.subscribe(member)
    vgr.set_beat_availability(BeatAvailability.UNAVAILABLE)
    member.append_beats.reset_mock()

    vgr.append_beat_schedule([BeatTiming(1)])

    member.append_beats.assert_not_called()
    assert vgr.current_beats == []


def test_on_member_join_replays_availability() -> None:
    """A late-joining role gets the current availability."""
    group = _make_group_stub()
    vgr = VisualizerGroupRole(group)
    vgr.set_beat_availability(BeatAvailability.UNAVAILABLE)

    member = _make_member_stub()
    vgr.subscribe(member)

    member.set_beat_availability.assert_called_once_with(BeatAvailability.UNAVAILABLE)


def test_unavailable_clears_current_beats() -> None:
    """set_beat_availability(UNAVAILABLE) drops any stored schedule."""
    vgr = VisualizerGroupRole(_make_group_stub())
    vgr.append_beat_schedule([BeatTiming(1), BeatTiming(2)])
    assert vgr.current_beats != []

    vgr.set_beat_availability(BeatAvailability.UNAVAILABLE)

    assert vgr.current_beats == []


def test_clear_beat_schedule_resets_availability_to_pending() -> None:
    """clear_beat_schedule resets availability for the next track."""
    vgr = VisualizerGroupRole(_make_group_stub())
    vgr.set_beat_availability(BeatAvailability.UNAVAILABLE)
    assert vgr.beat_availability is BeatAvailability.UNAVAILABLE

    vgr.clear_beat_schedule()

    assert vgr.beat_availability is BeatAvailability.PENDING


def test_clear_beat_schedule_re_declares_pending_to_members() -> None:
    """Members hear the PENDING reset alongside the clear."""
    vgr = VisualizerGroupRole(_make_group_stub())
    member = _make_member_stub()
    vgr.subscribe(member)
    vgr.set_beat_availability(BeatAvailability.UNAVAILABLE)
    member.reset_mock()

    vgr.clear_beat_schedule()

    member.clear_beats.assert_called_once_with()
    member.set_beat_availability.assert_called_once_with(BeatAvailability.PENDING)


def test_append_beat_schedule_allows_shared_segment_boundary() -> None:
    """Adjacent calls may share a boundary ts; the duplicate head is dropped."""
    vgr = VisualizerGroupRole(_make_group_stub())
    member = _make_member_stub()
    vgr.subscribe(member)
    vgr.append_beat_schedule([BeatTiming(100), BeatTiming(200)])
    member.reset_mock()

    vgr.append_beat_schedule([BeatTiming(200), BeatTiming(300)])  # shared 200

    assert [b.timestamp_us for b in vgr.current_beats] == [100, 200, 300]
    second_call_args = member.append_beats.call_args.args[0]
    # Duplicate boundary dropped; only the new tail is broadcast.
    assert [b.timestamp_us for b in second_call_args] == [300]


def test_append_beat_schedule_strict_inside_segment() -> None:
    """Strict-`>` still enforced within a single call (only the head dedupe is allowed)."""
    vgr = VisualizerGroupRole(_make_group_stub())
    with pytest.raises(ValueError, match="strictly increasing"):
        vgr.append_beat_schedule([BeatTiming(100), BeatTiming(100)])


def test_on_member_join_skips_role_that_does_not_want_beats() -> None:
    """A subscribing role whose wants_beats is False gets no schedule replay."""
    vgr = VisualizerGroupRole(_make_group_stub())
    vgr.append_beat_schedule([BeatTiming(10), BeatTiming(20)])
    member = _make_member_stub(wants_beats=False)

    vgr.subscribe(member)

    member.set_beat_availability.assert_called_once_with(BeatAvailability.PENDING)
    member.append_beats.assert_not_called()
    member.clear_beats.assert_not_called()


def test_on_member_join_filters_past_beats() -> None:
    """Beats whose ts is < current playhead are not replayed (joiner-side burden)."""
    vgr = VisualizerGroupRole(_make_group_stub(now_us=1_000))
    vgr.append_beat_schedule([BeatTiming(500), BeatTiming(1_500), BeatTiming(2_500)])
    member = _make_member_stub()

    vgr.subscribe(member)

    replayed = [b.timestamp_us for b in member.append_beats.call_args.args[0]]
    assert replayed == [1_500, 2_500]


def test_append_beat_schedule_snapshots_caller_list() -> None:
    """Post-call mutation of the caller's list must not change broadcast contents."""
    vgr = VisualizerGroupRole(_make_group_stub())
    member = _make_member_stub()
    vgr.subscribe(member)

    mutable: list[BeatTiming] = [BeatTiming(1), BeatTiming(2)]
    vgr.append_beat_schedule(mutable)
    mutable.append(BeatTiming(3))

    broadcast = member.append_beats.call_args.args[0]
    assert [b.timestamp_us for b in broadcast] == [1, 2]


def test_clear_beat_schedule_from_keeps_earlier_beats() -> None:
    """clear_beat_schedule(from_us=...) drops later beats and leaves the members alone."""
    vgr = VisualizerGroupRole(_make_group_stub())
    member = _make_member_stub()
    vgr.subscribe(member)
    vgr.append_beat_schedule([BeatTiming(1_000_000), BeatTiming(2_000_000), BeatTiming(3_000_000)])
    member.clear_beats.reset_mock()

    vgr.clear_beat_schedule(from_us=2_000_000)
    vgr.append_beat_schedule([BeatTiming(2_200_000)])

    assert [beat.timestamp_us for beat in vgr.current_beats] == [1_000_000, 2_200_000]
    member.clear_beats.assert_not_called()
