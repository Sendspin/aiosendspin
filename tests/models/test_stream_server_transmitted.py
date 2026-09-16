"""Stream start/clear messages carry a server_transmitted timestamp; stream/end does not."""

from __future__ import annotations

import pytest

from aiosendspin.models.core import (
    StreamClearPayload,
    StreamEndMessage,
    StreamEndPayload,
    StreamStartPayload,
)
from aiosendspin.models.types import ServerMessage


@pytest.mark.parametrize(
    "payload",
    [StreamStartPayload(), StreamClearPayload()],
)
def test_stream_payload_always_serializes_server_transmitted(
    payload: StreamStartPayload | StreamClearPayload,
) -> None:
    """server_transmitted is a required field, present on the wire even when unset."""
    assert "server_transmitted" in payload.to_dict()


def test_stream_end_omits_server_transmitted() -> None:
    """stream/end carries only roles."""
    assert "server_transmitted" not in StreamEndPayload().to_dict()
    message = StreamEndMessage(payload=StreamEndPayload(roles=["player"]))
    assert "server_transmitted" not in message.to_json()


def test_stream_end_with_legacy_server_transmitted_still_parses() -> None:
    """A stream/end from an older server that still sends server_transmitted is understood."""
    legacy = '{"type":"stream/end","payload":{"server_transmitted":5000000,"roles":["player"]}}'
    message = ServerMessage.from_json(legacy)
    assert isinstance(message, StreamEndMessage)
    assert message.payload.roles == ["player"]
