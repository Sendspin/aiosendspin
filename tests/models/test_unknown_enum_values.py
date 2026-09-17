"""Tests for unrecognized values of known enums in inbound messages."""

from __future__ import annotations

import pytest

from aiosendspin.models.controller import ControllerStatePayload
from aiosendspin.models.core import (
    ClientGoodbyePayload,
    GroupUpdateServerPayload,
    ServerActivatePayload,
)
from aiosendspin.models.player import PlayerStatePayload
from aiosendspin.models.types import Activity, GoodbyeReason, MediaCommand, PlaybackStateType


def test_goodbye_unrecognized_reason_is_recorded() -> None:
    """An unrecognized goodbye reason parses as no reason and is recorded as sent."""
    payload = ClientGoodbyePayload.from_dict({"reason": "moving_house"})
    assert payload.reason is None
    assert payload.unrecognized_reason == "moving_house"


def test_goodbye_record_cannot_be_spoofed() -> None:
    """A recognized reason clears any unrecognized_reason sent on the wire."""
    payload = ClientGoodbyePayload.from_dict(
        {"reason": "restart", "unrecognized_reason": "spoofed"}
    )
    assert payload.reason is GoodbyeReason.RESTART
    assert payload.unrecognized_reason is None
    assert payload.to_dict() == {"reason": "restart"}


def test_goodbye_without_reason_is_still_rejected() -> None:
    """Tolerance covers unknown identifiers, not a missing required reason."""
    with pytest.raises(LookupError, match="reason"):
        ClientGoodbyePayload.from_dict({})


def test_activate_drops_unrecognized_activities() -> None:
    """Unknown activities are dropped and recorded; known ones remain."""
    payload = ServerActivatePayload.from_dict(
        {"activities": ["teleport", "pairing"], "ignored_activities": ["spoofed"]}
    )
    assert payload.activities == [Activity.PAIRING]
    assert payload.ignored_activities == ["teleport"]
    assert "ignored_activities" not in ServerActivatePayload.from_dict({"activities": []}).to_dict()


def test_activate_non_string_activity_is_still_rejected() -> None:
    """A non-string activity entry is a shape error, not an unknown identifier."""
    with pytest.raises(ValueError, match="activities"):
        ServerActivatePayload.from_dict({"activities": [1]})


def test_group_update_drops_unrecognized_playback_state() -> None:
    """An unknown playback state is ignored and the other group fields apply."""
    payload = GroupUpdateServerPayload.from_dict(
        {"playback_state": "rewinding", "group_id": "g1", "group_name": "Kitchen"}
    )
    assert payload.playback_state is None
    assert payload.group_id == "g1"
    assert payload.group_name == "Kitchen"
    assert (
        GroupUpdateServerPayload.from_dict({"playback_state": "paused"}).playback_state
        is PlaybackStateType.PAUSED
    )


def test_controller_state_drops_unrecognized_commands() -> None:
    """Unknown controller commands are dropped from supported_commands."""
    payload = ControllerStatePayload.from_dict(
        {
            "supported_commands": ["play", "teleport"],
            "volume": 50,
            "muted": False,
            "repeat": "off",
            "shuffle": False,
        }
    )
    assert payload.supported_commands == [MediaCommand.PLAY]


def test_player_state_record_cannot_be_spoofed() -> None:
    """ignored_commands is taken from the parse, never from the wire."""
    payload = PlayerStatePayload.from_dict(
        {"supported_commands": ["volume"], "ignored_commands": ["spoofed"]}
    )
    assert payload.ignored_commands is None
    assert "supported_commands" not in PlayerStatePayload.from_dict({}).to_dict()
