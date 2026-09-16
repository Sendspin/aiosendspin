"""Tests for controller protocol models."""

from __future__ import annotations

import pytest
from mashumaro.exceptions import MissingField

from aiosendspin.models.controller import ControllerCommandPayload, ControllerStatePayload
from aiosendspin.models.types import MediaCommand, RepeatMode


def test_seek_command_requires_position_ms() -> None:
    """A 'seek' command without position_ms is rejected."""
    with pytest.raises(ValueError, match="position_ms"):
        ControllerCommandPayload(command=MediaCommand.SEEK)


def test_seek_command_rejects_offset_ms() -> None:
    """A 'seek' command must not carry offset_ms."""
    with pytest.raises(ValueError, match="offset_ms"):
        ControllerCommandPayload(command=MediaCommand.SEEK, position_ms=1000, offset_ms=500)


def test_seek_relative_command_requires_offset_ms() -> None:
    """A 'seek_relative' command without offset_ms is rejected."""
    with pytest.raises(ValueError, match="offset_ms"):
        ControllerCommandPayload(command=MediaCommand.SEEK_RELATIVE)


def test_seek_relative_command_rejects_position_ms() -> None:
    """A 'seek_relative' command must not carry position_ms."""
    with pytest.raises(ValueError, match="position_ms"):
        ControllerCommandPayload(
            command=MediaCommand.SEEK_RELATIVE, offset_ms=500, position_ms=1000
        )


def test_non_seek_command_rejects_position_ms() -> None:
    """Commands other than seek must not carry position_ms."""
    with pytest.raises(ValueError, match="position_ms"):
        ControllerCommandPayload(command=MediaCommand.PLAY, position_ms=1000)


def test_non_seek_command_rejects_offset_ms() -> None:
    """Commands other than seek must not carry offset_ms."""
    with pytest.raises(ValueError, match="offset_ms"):
        ControllerCommandPayload(command=MediaCommand.PLAY, offset_ms=500)


def test_state_requires_repeat_and_shuffle() -> None:
    """Controller state without repeat/shuffle is rejected rather than backfilled."""
    with pytest.raises(MissingField):
        ControllerStatePayload.from_dict({"supported_commands": [], "volume": 0, "muted": False})


def test_state_omits_seek_max_ms_when_absent() -> None:
    """seek_max_ms is omitted from the wire when not provided."""
    payload = ControllerStatePayload(
        supported_commands=[MediaCommand.PLAY],
        volume=50,
        muted=False,
        repeat=RepeatMode.OFF,
        shuffle=False,
    )
    assert "seek_max_ms" not in payload.to_dict()


def test_seek_command_rejects_negative_position_ms() -> None:
    """A 'seek' with a negative position_ms is rejected."""
    with pytest.raises(ValueError, match="non-negative"):
        ControllerCommandPayload(command=MediaCommand.SEEK, position_ms=-1)
