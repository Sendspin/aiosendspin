"""The client SDK sends client/leave without touching reported availability."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest

from aiosendspin.client.client import SendspinClient
from aiosendspin.client.connection import SendspinConnection
from aiosendspin.models.types import Roles
from tests.conftest import make_sdk_client


def _connection(sent: list[str], *, connected: bool = True) -> SendspinConnection:
    conn = SendspinConnection(make_sdk_client(client_name="c", roles=[Roles.CONTROLLER]))
    conn._ws = MagicMock(closed=False)  # noqa: SLF001
    conn._connected = connected  # noqa: SLF001

    async def _capture(payload: str) -> None:
        sent.append(payload)

    conn._send_message = _capture  # type: ignore[method-assign]  # noqa: SLF001
    return conn


async def test_send_leave_sends_payloadless_client_leave() -> None:
    """send_leave emits only the message type and keeps the reported availability."""
    sent: list[str] = []
    conn = _connection(sent)
    available_before = conn._reported_available  # noqa: SLF001

    await conn.send_leave()

    assert [orjson.loads(message) for message in sent] == [{"type": "client/leave"}]
    assert conn._reported_available is available_before  # noqa: SLF001


async def test_connection_send_leave_requires_connection() -> None:
    """A disconnected connection refuses to send client/leave."""
    sent: list[str] = []
    conn = _connection(sent, connected=False)

    with pytest.raises(RuntimeError, match="not connected"):
        await conn.send_leave()
    assert sent == []


async def test_client_send_leave_uses_admitted_connection() -> None:
    """The public leave API delegates to the admitted connection."""
    client = SendspinClient.__new__(SendspinClient)
    connection = AsyncMock()
    client._admitted_connection = connection  # type: ignore[assignment]  # noqa: SLF001

    await client.send_leave()

    connection.send_leave.assert_awaited_once_with()


async def test_client_send_leave_requires_connection() -> None:
    """The public leave API raises when no connection is admitted."""
    client = SendspinClient.__new__(SendspinClient)
    client._admitted_connection = None  # noqa: SLF001

    with pytest.raises(RuntimeError, match="not connected"):
        await client.send_leave()
