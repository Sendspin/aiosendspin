"""Regression tests for server/state message merging."""

from __future__ import annotations

from aiosendspin.models.color import SessionUpdateColor
from aiosendspin.models.controller import ControllerStatePayload
from aiosendspin.models.core import ServerStateMessage, ServerStatePayload
from aiosendspin.models.metadata import Progress, SessionUpdateMetadata
from aiosendspin.models.types import MediaCommand, RepeatMode, UndefinedField


def test_server_state_absent_role_omitted_from_wire() -> None:
    """A role left unset is UndefinedField and omitted from serialization."""
    payload = ServerStatePayload(metadata=SessionUpdateMetadata(timestamp=100, title="X"))
    encoded = payload.to_dict()
    assert "color" not in encoded
    assert "controller" not in encoded
    assert isinstance(ServerStatePayload.from_dict(encoded).color, UndefinedField)


def test_server_state_whole_role_null_round_trips() -> None:
    """A whole-role object set to null serializes as null and decodes to None."""
    payload = ServerStatePayload(metadata=None)
    assert payload.to_dict() == {"metadata": None}
    decoded = ServerStatePayload.from_dict({"metadata": None})
    assert decoded.metadata is None
    assert isinstance(decoded.color, UndefinedField)
    assert isinstance(decoded.controller, UndefinedField)


def test_server_state_merge_whole_role_null_clears_role() -> None:
    """A whole-role null clears that role; an absent role keeps its existing state."""
    existing = ServerStateMessage(
        payload=ServerStatePayload(
            metadata=SessionUpdateMetadata(timestamp=100, title="Song Title"),
        )
    )
    incoming = ServerStateMessage(payload=ServerStatePayload(metadata=None))

    merged = existing.merge(incoming)

    assert isinstance(merged, ServerStateMessage)
    assert merged.payload.metadata is None


def test_server_state_merge_absent_role_preserved() -> None:
    """An incoming message that omits a role leaves the existing role state intact."""
    existing = ServerStateMessage(
        payload=ServerStatePayload(
            metadata=SessionUpdateMetadata(timestamp=100, title="Song Title"),
        )
    )
    incoming = ServerStateMessage(
        payload=ServerStatePayload(
            color=None,
        )
    )

    merged = existing.merge(incoming)

    assert isinstance(merged, ServerStateMessage)
    assert merged.payload.metadata is not None
    assert merged.payload.metadata.title == "Song Title"
    assert merged.payload.color is None


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
    incoming = ServerStateMessage(payload=ServerStatePayload(metadata=None, controller=controller))

    merged = existing.merge(incoming)

    assert isinstance(merged, ServerStateMessage)
    assert merged.payload.metadata is None
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
    assert merged.payload.controller is not None
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
