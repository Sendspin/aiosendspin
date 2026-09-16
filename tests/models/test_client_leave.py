"""client/leave carries no payload fields."""

from __future__ import annotations

import orjson
import pytest

from aiosendspin.models.core import ClientLeaveMessage
from aiosendspin.models.types import ClientMessage


def test_client_leave_serializes_without_payload() -> None:
    """The wire form is the message type alone."""
    assert orjson.loads(ClientLeaveMessage().to_json()) == {"type": "client/leave"}


@pytest.mark.parametrize(
    "raw",
    [
        '{"type":"client/leave"}',
        '{"type":"client/leave","payload":{}}',
        '{"type":"client/leave","payload":{"future_field":1}}',
    ],
)
def test_client_leave_parses_with_or_without_payload(raw: str) -> None:
    """An absent, empty, or unrecognized payload still dispatches to client/leave."""
    assert isinstance(ClientMessage.from_json(raw), ClientLeaveMessage)
