"""Regression tests for server/state message merging."""

from __future__ import annotations

import pytest

from aiosendspin.models.color import SessionUpdateColor
from aiosendspin.models.controller import ControllerStatePayload
from aiosendspin.models.core import (
    LegacyServerStateClearMessage,
    ServerStateMessage,
    ServerStatePayload,
)
from aiosendspin.models.metadata import Progress, SessionUpdateMetadata
from aiosendspin.models.types import MediaCommand, RepeatMode, ServerMessage, UndefinedField


def test_server_state_absent_role_omitted_from_wire() -> None:
    """A role left unset is UndefinedField and omitted from serialization."""
    payload = ServerStatePayload(metadata=SessionUpdateMetadata(timestamp=100, title="X"))
    encoded = payload.to_dict()
    assert "color" not in encoded
    assert "controller" not in encoded
    assert isinstance(ServerStatePayload.from_dict(encoded).color, UndefinedField)


@pytest.mark.parametrize("role", ["metadata", "controller", "color", "_acme"])
def test_server_state_null_role_object_is_rejected(role: str) -> None:
    """A null role object fails to parse rather than reading as an absent role."""
    with pytest.raises(ValueError, match=f"must not be null, got \\['{role}'\\]"):
        ServerStatePayload.from_dict({role: None})
    with pytest.raises(ValueError, match="ServerStatePayload"):
        ServerMessage.from_json(f'{{"type":"server/state","payload":{{"{role}":null}}}}')


def test_server_state_merge_timestamp_only_object_clears_role() -> None:
    """A timestamp-only role object replaces every field of the queued one."""
    existing = ServerStateMessage(
        payload=ServerStatePayload(
            metadata=SessionUpdateMetadata(timestamp=100, title="Song Title"),
        )
    )
    incoming = ServerStateMessage(
        payload=ServerStatePayload(metadata=SessionUpdateMetadata(timestamp=200))
    )

    merged = existing.merge(incoming)

    assert isinstance(merged, ServerStateMessage)
    assert merged.payload.to_dict() == {"metadata": {"timestamp": 200}}


# DEPRECATED(spec-pr-275): remove in aiosendspin <version>
def test_legacy_clear_message_serializes_null_role_object() -> None:
    """The legacy clear message sends a server/state with a null role object."""
    message = LegacyServerStateClearMessage("metadata")

    assert message.to_json() == '{"type":"server/state","payload":{"metadata":null}}'
    assert message.merge(ServerStateMessage(ServerStatePayload())) is None


# DEPRECATED(spec-pr-275): remove in aiosendspin <version>
def test_legacy_clear_message_is_never_parsed() -> None:
    """A parsed server/state is always a ServerStateMessage, never the legacy clear message."""
    message = ServerMessage.from_json(
        '{"type":"server/state","payload":{"metadata":{"timestamp":1}}}'
    )

    assert type(message) is ServerStateMessage


def test_server_state_merge_absent_role_preserved() -> None:
    """An incoming message that omits a role leaves the existing role state intact."""
    existing = ServerStateMessage(
        payload=ServerStatePayload(
            metadata=SessionUpdateMetadata(timestamp=100, title="Song Title"),
        )
    )
    color = SessionUpdateColor(timestamp=200, primary=(1, 2, 3))
    incoming = ServerStateMessage(payload=ServerStatePayload(color=color))

    merged = existing.merge(incoming)

    assert isinstance(merged, ServerStateMessage)
    assert merged.payload.metadata == SessionUpdateMetadata(timestamp=100, title="Song Title")
    assert merged.payload.color == color


def test_server_state_merge_replaces_metadata_object_wholesale() -> None:
    """A later metadata object replaces the earlier one; no stale field survives."""
    existing = ServerStateMessage(
        payload=ServerStatePayload(
            metadata=SessionUpdateMetadata(
                timestamp=100,
                title="Song Title",
                album="Some Album",
                progress=Progress(
                    track_progress=30_000,
                    track_duration=213_000,
                    playback_speed=1_000,
                ),
            )
        )
    )
    incoming = ServerStateMessage(
        payload=ServerStatePayload(
            metadata=SessionUpdateMetadata(timestamp=200, title="Other Title"),
        )
    )

    merged = existing.merge(incoming)

    assert isinstance(merged, ServerStateMessage)
    assert merged.payload.metadata == SessionUpdateMetadata(timestamp=200, title="Other Title")
    assert merged.payload.to_dict() == {"metadata": {"timestamp": 200, "title": "Other Title"}}


def test_server_state_merge_replaces_each_role_object_independently() -> None:
    """Each role object is replaced or kept on its own; omitted roles keep queued state."""
    controller = ControllerStatePayload(
        supported_commands=[MediaCommand.PLAY],
        volume=50,
        muted=False,
        repeat=RepeatMode.OFF,
        shuffle=False,
    )
    color = SessionUpdateColor(timestamp=100, primary=(1, 2, 3))
    existing = ServerStateMessage(
        payload=ServerStatePayload(
            metadata=SessionUpdateMetadata(timestamp=100, title="Song Title"),
            color=color,
        )
    )
    incoming = ServerStateMessage(
        payload=ServerStatePayload(
            metadata=SessionUpdateMetadata(timestamp=200), controller=controller
        )
    )

    merged = existing.merge(incoming)

    assert isinstance(merged, ServerStateMessage)
    assert merged.payload.metadata == SessionUpdateMetadata(timestamp=200)
    assert merged.payload.controller == controller
    assert merged.payload.color == color


def test_server_state_merge_controller_overwrites_repeat_and_shuffle() -> None:
    """Incoming controller state overwrites existing repeat/shuffle (required fields)."""
    existing = ServerStateMessage(
        payload=ServerStatePayload(
            controller=ControllerStatePayload(
                supported_commands=[MediaCommand.PLAY],
                volume=50,
                muted=False,
                repeat=RepeatMode.OFF,
                shuffle=False,
            )
        )
    )
    incoming = ServerStateMessage(
        payload=ServerStatePayload(
            controller=ControllerStatePayload(
                supported_commands=[MediaCommand.PLAY],
                volume=50,
                muted=False,
                repeat=RepeatMode.ALL,
                shuffle=True,
            )
        )
    )

    merged = existing.merge(incoming)

    assert isinstance(merged, ServerStateMessage)
    assert isinstance(merged.payload.controller, ControllerStatePayload)
    assert merged.payload.controller.repeat == RepeatMode.ALL
    assert merged.payload.controller.shuffle is True


def test_legacy_metadata_repeat_and_shuffle_are_ignored_on_parse() -> None:
    """A metadata object that still carries repeat/shuffle parses without them."""
    decoded = ServerStatePayload.from_dict(
        {"metadata": {"timestamp": 1, "title": "X", "repeat": "all", "shuffle": True}}
    )
    assert decoded.metadata == SessionUpdateMetadata(timestamp=1, title="X")


def test_metadata_object_never_carries_leaf_nulls() -> None:
    """Unset metadata fields are omitted rather than sent as null."""
    payload = ServerStatePayload(metadata=SessionUpdateMetadata(timestamp=1, title=None))
    assert payload.to_dict() == {"metadata": {"timestamp": 1}}
