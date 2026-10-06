"""Tests for ColorGroupRole."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aiosendspin.clock import ManualClock
from aiosendspin.models.core import ServerStateMessage
from aiosendspin.server.roles.color import ColorClearedEvent, ColorUpdatedEvent
from aiosendspin.server.roles.color.group import ColorGroupRole
from aiosendspin.server.roles.color.state import Color


def _make_group_stub() -> MagicMock:
    group = MagicMock()
    group._server = MagicMock()  # noqa: SLF001
    group._server.clock.now_us.return_value = 1_000_000  # noqa: SLF001
    return group


def test_color_group_role_family() -> None:
    """ColorGroupRole has role_family of 'color'."""
    group = _make_group_stub()
    cgr = ColorGroupRole(group)
    assert cgr.role_family == "color"


def test_color_group_role_initial_color_is_none() -> None:
    """Initial color is None."""
    group = _make_group_stub()
    cgr = ColorGroupRole(group)
    assert cgr.color is None


def test_set_color_stores_and_broadcasts() -> None:
    """set_color() stores the color and sends update to members."""
    group = _make_group_stub()
    cgr = ColorGroupRole(group)

    member = MagicMock()
    cgr._members = [member]  # noqa: SLF001

    color = Color(primary=(255, 0, 0), accent=(0, 255, 0))
    cgr.set_color(color)

    assert cgr.color is not None
    assert cgr.color.primary == (255, 0, 0)

    member.send_message.assert_called_once()
    msg = member.send_message.call_args.args[0]
    assert isinstance(msg, ServerStateMessage)
    assert msg.payload.color is not None
    assert msg.payload.color.primary == (255, 0, 0)

    group._signal_event.assert_called_once()  # noqa: SLF001
    event = group._signal_event.call_args.args[0]  # noqa: SLF001
    assert isinstance(event, ColorUpdatedEvent)
    assert event.color.primary == (255, 0, 0)
    assert event.previous_color is None


def test_set_color_no_op_when_equal() -> None:
    """set_color() with the same color does nothing."""
    group = _make_group_stub()
    cgr = ColorGroupRole(group)

    color = Color(primary=(255, 0, 0))
    cgr.set_color(color)
    group._signal_event.reset_mock()  # noqa: SLF001

    cgr.set_color(Color(primary=(255, 0, 0)))

    group._signal_event.assert_not_called()  # noqa: SLF001


def test_clear_color() -> None:
    """clear() sets color to None and sends a timestamp-only color object."""
    group = _make_group_stub()
    cgr = ColorGroupRole(group)

    member = MagicMock()
    cgr._members = [member]  # noqa: SLF001

    cgr.set_color(Color(primary=(255, 0, 0)))
    member.send_message.reset_mock()
    group._signal_event.reset_mock()  # noqa: SLF001

    cgr.clear()

    assert cgr.color is None
    member.send_message.assert_called_once()
    msg = member.send_message.call_args.args[0]
    assert isinstance(msg, ServerStateMessage)
    assert msg.payload.to_dict() == {"color": {"timestamp": 1_000_000}}

    event = group._signal_event.call_args.args[0]  # noqa: SLF001
    assert isinstance(event, ColorClearedEvent)


def test_on_member_join_sends_current_color() -> None:
    """on_member_join sends a snapshot to the new member."""
    group = _make_group_stub()
    cgr = ColorGroupRole(group)
    cgr.set_color(Color(primary=(100, 150, 200)))

    member = MagicMock()
    cgr.on_member_join(member)

    member.send_message.assert_called_once()
    msg = member.send_message.call_args.args[0]
    assert isinstance(msg, ServerStateMessage)
    assert msg.payload.color is not None
    assert msg.payload.color.primary == (100, 150, 200)


def test_on_member_join_sends_timestamp_only_when_no_color() -> None:
    """on_member_join sends a timestamp-only color object when no color is set."""
    group = _make_group_stub()
    cgr = ColorGroupRole(group)

    member = MagicMock()
    cgr.on_member_join(member)

    member.send_message.assert_called_once()
    msg = member.send_message.call_args.args[0]
    assert isinstance(msg, ServerStateMessage)
    assert msg.payload.to_dict() == {"color": {"timestamp": 1_000_000}}


def test_set_color_sends_full_state_on_partial_change() -> None:
    """A change to one field still sends every set field, with no leaf nulls."""
    group = _make_group_stub()
    cgr = ColorGroupRole(group)

    member = MagicMock()
    cgr._members = [member]  # noqa: SLF001

    cgr.set_color(Color(primary=(255, 0, 0), accent=(0, 255, 0)))
    member.send_message.reset_mock()

    cgr.set_color(Color(primary=(255, 0, 0), accent=(0, 0, 255)))

    msg = member.send_message.call_args.args[0]
    assert isinstance(msg, ServerStateMessage)
    assert msg.payload.to_dict() == {
        "color": {"timestamp": 1_000_000, "primary": [255, 0, 0], "accent": [0, 0, 255]}
    }


def _make_scheduling_group() -> tuple[MagicMock, ManualClock]:
    clock = ManualClock(now_us_value=1_000_000)
    group = MagicMock()
    group._server.clock = clock  # noqa: SLF001
    return group, clock


def _sent_colors(member: MagicMock) -> list[dict[str, object] | None]:
    return [call.args[0].payload.to_dict()["color"] for call in member.send_message.call_args_list]


def test_scheduled_color_is_sent_and_takes_effect_later() -> None:
    """A future palette is sent with its timestamp and becomes current once due."""
    group, clock = _make_scheduling_group()
    cgr = ColorGroupRole(group)
    member = MagicMock()
    cgr._members = [member]  # noqa: SLF001
    cgr.set_color(Color(primary=(1, 2, 3)))

    cgr.set_color(Color(primary=(4, 5, 6)), timestamp_us=1_500_000)

    assert cgr.color == Color(primary=(1, 2, 3))
    assert _sent_colors(member)[-1] == {"timestamp": 1_500_000, "primary": [4, 5, 6]}
    event = group._signal_event.call_args.args[0]  # noqa: SLF001
    assert isinstance(event, ColorUpdatedEvent)
    assert event.timestamp_us == 1_500_000
    clock.advance_us(500_000)
    assert cgr.color == Color(primary=(4, 5, 6))


def test_late_join_gets_current_then_scheduled_color() -> None:
    """A joining member gets the current palette as of now, then the scheduled one."""
    group, clock = _make_scheduling_group()
    cgr = ColorGroupRole(group)
    cgr.set_color(Color(primary=(1, 2, 3)))
    cgr.set_color(Color(primary=(4, 5, 6)), timestamp_us=1_500_000)
    clock.advance_us(100_000)

    member = MagicMock()
    cgr.on_member_join(member)

    assert _sent_colors(member) == [
        {"timestamp": 1_100_000, "primary": [1, 2, 3]},
        {"timestamp": 1_500_000, "primary": [4, 5, 6]},
    ]


def test_late_join_after_scheduled_color_took_effect() -> None:
    """Once due, the scheduled palette is the joining member's current state."""
    group, clock = _make_scheduling_group()
    cgr = ColorGroupRole(group)
    cgr.set_color(Color(primary=(4, 5, 6)), timestamp_us=1_500_000)
    clock.advance_us(600_000)

    member = MagicMock()
    cgr.on_member_join(member)

    assert _sent_colors(member) == [{"timestamp": 1_600_000, "primary": [4, 5, 6]}]


def test_unchanged_present_color_cancels_scheduled_one() -> None:
    """Re-setting the current palette now is sent, since it cancels the scheduled one."""
    group, _clock = _make_scheduling_group()
    cgr = ColorGroupRole(group)
    member = MagicMock()
    cgr._members = [member]  # noqa: SLF001
    cgr.set_color(Color(primary=(1, 2, 3)))
    cgr.set_color(Color(primary=(4, 5, 6)), timestamp_us=1_500_000)

    cgr.set_color(Color(primary=(1, 2, 3)))

    assert _sent_colors(member)[-1] == {"timestamp": 1_000_000, "primary": [1, 2, 3]}
    assert cgr._state.pending_timestamp_us is None  # noqa: SLF001


def test_cancel_scheduled_color_resends_current() -> None:
    """cancel_scheduled() re-sends the current palette now; without a schedule it does nothing."""
    group, _clock = _make_scheduling_group()
    cgr = ColorGroupRole(group)
    member = MagicMock()
    cgr._members = [member]  # noqa: SLF001
    cgr.set_color(Color(primary=(1, 2, 3)))
    cgr.set_color(Color(primary=(4, 5, 6)), timestamp_us=1_500_000)

    cgr.cancel_scheduled()
    cgr.cancel_scheduled()

    assert _sent_colors(member)[2:] == [{"timestamp": 1_000_000, "primary": [1, 2, 3]}]
    assert cgr.color == Color(primary=(1, 2, 3))


def test_clear_discards_scheduled_color() -> None:
    """clear() sends null at once and the scheduled palette never takes effect."""
    group, clock = _make_scheduling_group()
    cgr = ColorGroupRole(group)
    cgr.set_color(Color(primary=(4, 5, 6)), timestamp_us=1_500_000)

    cgr.clear()
    clock.advance_us(600_000)

    assert cgr.color is None


def test_scheduling_a_color_clear_is_rejected() -> None:
    """A clear cannot be scheduled; an empty palette can."""
    group, _clock = _make_scheduling_group()
    cgr = ColorGroupRole(group)

    with pytest.raises(ValueError, match="Color"):
        cgr.set_color(None, timestamp_us=1_500_000)
    cgr.set_color(Color(), timestamp_us=1_500_000)


def test_color_beyond_lead_limit_is_sent_20s_ahead() -> None:
    """A palette more than 20 s ahead is sent, also to joining members, only 20 s ahead."""
    group, clock = _make_scheduling_group()
    cgr = ColorGroupRole(group)
    member = MagicMock()
    cgr._members = [member]  # noqa: SLF001

    cgr.set_color(Color(primary=(4, 5, 6)), timestamp_us=26_000_000)

    member.send_message.assert_not_called()
    (delay_s, send), _kwargs = group._server.loop.call_later.call_args  # noqa: SLF001
    assert delay_s == 5.0
    joiner = MagicMock()
    cgr.on_member_join(joiner)
    assert _sent_colors(joiner) == [{"timestamp": 1_000_000}]

    clock.advance_us(5_000_000)
    send()
    assert _sent_colors(member) == [{"timestamp": 26_000_000, "primary": [4, 5, 6]}]


# DEPRECATED(spec-pr-135): remove in aiosendspin <version>
def test_color_reaches_member_applying_on_receipt_once_due() -> None:
    """A member that applies a palette on receipt gets a scheduled one only once it is due."""
    group, clock = _make_scheduling_group()
    cgr = ColorGroupRole(group)
    member = MagicMock()
    member.supports_scheduled_updates.return_value = False
    cgr._members = [member]  # noqa: SLF001

    cgr.set_color(Color(primary=(4, 5, 6)), timestamp_us=1_500_000)
    joiner = MagicMock()
    joiner.supports_scheduled_updates.return_value = False
    cgr.subscribe(joiner)

    member.send_message.assert_not_called()
    assert _sent_colors(joiner) == [{"timestamp": 1_000_000}]
    (delay_s, send_due, *args), _kwargs = group._server.loop.call_later.call_args  # noqa: SLF001
    assert delay_s == 0.5
    clock.advance_us(500_000)
    send_due(*args)
    assert _sent_colors(member) == [{"timestamp": 1_500_000, "primary": [4, 5, 6]}]
    assert _sent_colors(joiner)[1:] == [{"timestamp": 1_500_000, "primary": [4, 5, 6]}]


def test_cancel_before_deferred_send_sends_nothing() -> None:
    """Cancelling a palette not yet sent stops its send and sends nothing else."""
    group, _clock = _make_scheduling_group()
    cgr = ColorGroupRole(group)
    member = MagicMock()
    cgr._members = [member]  # noqa: SLF001
    cgr.set_color(Color(primary=(4, 5, 6)), timestamp_us=26_000_000)

    cgr.cancel_scheduled()

    group._server.loop.call_later.return_value.cancel.assert_called_once()  # noqa: SLF001
    member.send_message.assert_not_called()
    assert cgr._state.pending_timestamp_us is None  # noqa: SLF001


def test_sent_color_replaced_by_deferred_one_is_cancelled() -> None:
    """Replacing a sent palette with one sent only later restates the current palette now."""
    group, _clock = _make_scheduling_group()
    cgr = ColorGroupRole(group)
    member = MagicMock()
    cgr._members = [member]  # noqa: SLF001
    cgr.set_color(Color(primary=(4, 5, 6)), timestamp_us=1_500_000)

    cgr.set_color(Color(primary=(7, 8, 9)), timestamp_us=26_000_000)

    assert _sent_colors(member) == [
        {"timestamp": 1_500_000, "primary": [4, 5, 6]},
        {"timestamp": 1_000_000},
    ]
