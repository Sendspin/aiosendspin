"""Tests for application-specific role objects in role-object messages."""

from __future__ import annotations

from typing import Any

import orjson
import pytest

from aiosendspin.models.base import SendspinModel
from aiosendspin.models.core import (
    ClientCommandMessage,
    ClientCommandPayload,
    ClientStateMessage,
    ClientStatePayload,
    ServerCommandMessage,
    ServerCommandPayload,
    ServerStateMessage,
    ServerStatePayload,
    StreamStartMessage,
    StreamStartPayload,
)
from aiosendspin.models.metadata import SessionUpdateMetadata
from aiosendspin.models.types import ClientMessage, ServerMessage

PAYLOADS = pytest.mark.parametrize(
    ("payload_cls", "known"),
    [
        (ClientStatePayload, {"available": True}),
        (ClientCommandPayload, {}),
        (ServerStatePayload, {"metadata": {"timestamp": 1}}),
        (ServerCommandPayload, {}),
        (StreamStartPayload, {"server_transmitted": 7}),
    ],
)


@pytest.mark.parametrize(
    ("message_type", "message_base"),
    [
        (ClientStateMessage, ClientMessage),
        (ClientCommandMessage, ClientMessage),
        (ServerStateMessage, ServerMessage),
        (ServerCommandMessage, ServerMessage),
        (StreamStartMessage, ServerMessage),
    ],
)
def test_application_objects_on_the_wire(
    message_type: type[ClientMessage | ServerMessage],
    message_base: type[ClientMessage | ServerMessage],
) -> None:
    """A full message puts application objects at the top level of its payload."""
    wire_type = message_type.__dataclass_fields__["type"].default
    text = orjson.dumps({"type": wire_type, "payload": {"_acme": {"on": True}}}).decode()

    message = message_base.from_json(text)

    assert isinstance(message, message_type)
    assert message.payload.application_objects == {"_acme": {"on": True}}  # type: ignore[attr-defined]
    sent = orjson.loads(message.to_json())["payload"]
    sent.pop("server_transmitted", None)
    assert sent == {"_acme": {"on": True}}


@PAYLOADS
def test_application_objects_round_trip(
    payload_cls: type[SendspinModel], known: dict[str, Any]
) -> None:
    """`_`-prefixed keys survive a parse and serialize at the top level again."""
    wire = known | {"_acme_lights": {"scene": "dim"}, "_acme_hint": [1, 2]}

    payload = payload_cls.from_dict(wire)

    assert payload.application_objects == {  # type: ignore[attr-defined]
        "_acme_lights": {"scene": "dim"},
        "_acme_hint": [1, 2],
    }
    assert payload.to_dict() == wire


@PAYLOADS
def test_application_objects_omitted_when_empty(
    payload_cls: type[SendspinModel], known: dict[str, Any]
) -> None:
    """Without application objects nothing extra is serialized."""
    payload = payload_cls.from_dict(known)

    assert payload.application_objects == {}  # type: ignore[attr-defined]
    assert payload.to_dict() == known


@PAYLOADS
def test_unknown_non_prefixed_keys_still_ignored(
    payload_cls: type[SendspinModel], known: dict[str, Any]
) -> None:
    """Unknown keys without `_` are dropped, and the field name cannot be set from the wire."""
    wire = known | {"future_role": {"x": 1}, "application_objects": {"_spoofed": 1}}

    payload = payload_cls.from_dict(wire)

    assert payload.application_objects == {}  # type: ignore[attr-defined]
    assert payload.to_dict() == known


def test_application_object_keys_must_be_prefixed() -> None:
    """Serializing an application object key without `_` is rejected."""
    payload = ServerCommandPayload(application_objects={"acme": {}})

    with pytest.raises(ValueError, match="must start with '_'"):
        payload.to_dict()


def test_server_state_merge_keeps_application_objects_per_key() -> None:
    """A merged server/state replaces present application objects and keeps the rest."""
    existing = ServerStateMessage(
        payload=ServerStatePayload(
            metadata=SessionUpdateMetadata(timestamp=100, title="Song"),
            application_objects={"_a": {"v": 1}, "_b": {"v": 1}},
        )
    )
    incoming = ServerStateMessage(
        payload=ServerStatePayload(application_objects={"_b": {"v": 2}, "_c": {"v": 3}})
    )

    merged = existing.merge(incoming)

    assert isinstance(merged, ServerStateMessage)
    assert merged.payload.metadata == SessionUpdateMetadata(timestamp=100, title="Song")
    assert merged.payload.application_objects == {"_a": {"v": 1}, "_b": {"v": 2}, "_c": {"v": 3}}
    assert existing.payload.application_objects == {"_a": {"v": 1}, "_b": {"v": 1}}
