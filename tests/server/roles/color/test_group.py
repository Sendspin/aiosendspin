"""Tests for ColorGroupRole."""

from __future__ import annotations

from unittest.mock import MagicMock

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
    """clear() sets color to None and sends a color null."""
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
    assert msg.payload.to_dict() == {"color": None}

    event = group._signal_event.call_args.args[0]  # noqa: SLF001
    assert isinstance(event, ColorClearedEvent)


def test_on_member_join_sends_current_color() -> None:
    """on_member_join sends a snapshot to the new member."""
    group = _make_group_stub()
    cgr = ColorGroupRole(group)
    cgr._current_color = Color(primary=(100, 150, 200))  # noqa: SLF001

    member = MagicMock()
    cgr.on_member_join(member)

    member.send_message.assert_called_once()
    msg = member.send_message.call_args.args[0]
    assert isinstance(msg, ServerStateMessage)
    assert msg.payload.color is not None
    assert msg.payload.color.primary == (100, 150, 200)


def test_on_member_join_sends_null_when_no_color() -> None:
    """on_member_join sends a color null when no color is set."""
    group = _make_group_stub()
    cgr = ColorGroupRole(group)

    member = MagicMock()
    cgr.on_member_join(member)

    member.send_message.assert_called_once()
    msg = member.send_message.call_args.args[0]
    assert isinstance(msg, ServerStateMessage)
    assert msg.payload.to_dict() == {"color": None}


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
