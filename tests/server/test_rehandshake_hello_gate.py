"""Which client/hello marks a connection for a hello pair after every re-handshake."""

# DEPRECATED(spec-pr-287): remove in aiosendspin <version>

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import orjson
import pytest

from aiosendspin.models.core import ClientHelloMessage
from aiosendspin.server.clock import LoopClock
from aiosendspin.server.connection import SendspinConnection

_PLAYER_SUPPORT: dict[str, Any] = {
    "supported_formats": [{"codec": "pcm", "sample_rate": 48000, "bit_depth": 16, "channels": 2}],
    "buffer_capacity": 100_000,
}
_ARTWORK_SUPPORT: dict[str, Any] = {
    "channels": [{"source": "album", "format": "jpeg", "media_width": 300, "media_height": 200}]
}


@dataclass(slots=True)
class _DummyServer:
    loop: asyncio.AbstractEventLoop
    clock: Any
    id: str = "srv"
    name: str = "server"


def _hello(**payload: Any) -> ClientHelloMessage:
    raw = orjson.dumps(
        {
            "type": "client/hello",
            "payload": {
                "name": "Client",
                "supported_roles": ["player@v1"],
                "player@v1_support": _PLAYER_SUPPORT,
                **payload,
            },
        }
    ).decode()
    message = SendspinConnection._deserialize_client_message(raw)  # noqa: SLF001
    assert isinstance(message, ClientHelloMessage)
    return message


@pytest.mark.parametrize(
    ("hello", "expected"),
    [
        (_hello(), False),
        (
            _hello(**{"player@v1_support": None, "player_support": _PLAYER_SUPPORT}),
            True,
        ),
        (
            _hello(**{"player@v1_support": {**_PLAYER_SUPPORT, "supported_commands": []}}),
            True,
        ),
        (
            _hello(
                supported_roles=["player@v1", "artwork@v1"],
                **{"artwork@v1_support": _ARTWORK_SUPPORT},
            ),
            True,
        ),
        (
            _hello(
                supported_roles=["player@v1", "visualizer@v1"],
                **{"visualizer@v1_support": {"buffer_capacity": 1000, "rate_max": 30}},
            ),
            True,
        ),
        (_hello(supported_pair_methods=[{"method": "pairing_psk"}]), True),
        (_hello(**{"visualizer@v1_support": {"buffer_capacity": 1000}}), False),
        (_hello(supported_roles=["player@v1", "visualizer@v1"]), False),
    ],
    ids=[
        "current",
        "unversioned-support-key",
        "player-supported-commands",
        "artwork-support",
        "visualizer-stream-config",
        "list-form-pair-methods",
        "unlisted-support-role",
        "missing-support-object",
    ],
)
@pytest.mark.asyncio
async def test_gate_follows_pre_spec_287_wire(
    hello: ClientHelloMessage,
    expected: bool,  # noqa: FBT001
) -> None:
    """Only a tolerance for a wire predating spec #287 marks the connection."""
    loop = asyncio.get_running_loop()
    conn = SendspinConnection(
        _DummyServer(loop=loop, clock=LoopClock(loop)), wsock_client=MagicMock()
    )
    conn._client = MagicMock()  # noqa: SLF001
    conn._note_client_hello_wire(hello.payload)  # noqa: SLF001
    assert conn._expects_rehandshake_hellos is expected  # noqa: SLF001
