"""server/unpair carries no payload fields."""

from __future__ import annotations

import orjson

from aiosendspin.models.core import ServerUnpairMessage


def test_server_unpair_sends_empty_payload() -> None:
    """The wire form keeps an empty payload object."""
    assert orjson.loads(ServerUnpairMessage().to_json()) == {
        "type": "server/unpair",
        "payload": {},
    }
