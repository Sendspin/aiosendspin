"""A malformed client/hello is rejected instead of raising out of the connection."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiosendspin.models.core import ClientGoodbyeMessage, ClientGoodbyePayload
from aiosendspin.models.types import GoodbyeReason
from aiosendspin.server.clock import LoopClock
from aiosendspin.server.connection import SendspinConnection


@dataclass(slots=True)
class _DummyServer:
    loop: asyncio.AbstractEventLoop
    clock: Any
    id: str = "srv"
    name: str = "server"


@pytest.mark.asyncio
async def test_malformed_client_hello_rejects_without_raising() -> None:
    """Undeserializable hello text disconnects and returns False, it does not raise."""
    loop = asyncio.get_running_loop()
    conn = SendspinConnection(
        _DummyServer(loop=loop, clock=LoopClock(loop)), wsock_client=MagicMock()
    )
    conn.disconnect = AsyncMock()  # type: ignore[method-assign]

    assert await conn._ingest_client_hello("{not json") is False  # noqa: SLF001

    conn.disconnect.assert_awaited_once_with(retry_connection=False)


@pytest.mark.asyncio
async def test_restart_goodbye_in_place_of_hello_keeps_reconnecting() -> None:
    """A client/goodbye sent instead of the hello is honored, so a restart is redialed."""
    loop = asyncio.get_running_loop()
    conn = SendspinConnection(
        _DummyServer(loop=loop, clock=LoopClock(loop)), wsock_client=MagicMock()
    )
    goodbye = ClientGoodbyeMessage(payload=ClientGoodbyePayload(reason=GoodbyeReason.RESTART))

    assert await conn._ingest_client_hello(goodbye.to_json()) is False  # noqa: SLF001

    assert conn.goodbye_reason is GoodbyeReason.RESTART
    assert conn.should_retry_server_initiated_connection


@pytest.mark.asyncio
async def test_null_reason_goodbye_in_place_of_hello_is_flagged() -> None:
    """A client/goodbye sent instead of the hello has its deviations flagged."""
    loop = asyncio.get_running_loop()
    conn = SendspinConnection(
        _DummyServer(loop=loop, clock=LoopClock(loop)), wsock_client=MagicMock()
    )
    conn._flag_noncompliance = MagicMock()  # type: ignore[method-assign]  # noqa: SLF001

    goodbye = '{"type": "client/goodbye", "payload": {"reason": null}}'
    assert await conn._ingest_client_hello(goodbye) is False  # noqa: SLF001

    conn._flag_noncompliance.assert_called_once_with(  # noqa: SLF001
        "client/goodbye sent null for 'reason' instead of a value"
    )
