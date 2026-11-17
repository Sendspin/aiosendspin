"""The client checks the `group_id` of each group/update."""

from __future__ import annotations

import json
import secrets
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import WSMessage, WSMsgType

from aiosendspin.client.connection import SendspinConnection
from aiosendspin.models.core import GroupUpdateServerPayload
from aiosendspin.models.types import Roles
from tests.conftest import make_sdk_client


def _connection() -> tuple[SendspinConnection, list[GroupUpdateServerPayload]]:
    client = make_sdk_client(client_name="Test Client", roles=[Roles.CONTROLLER])
    updates: list[GroupUpdateServerPayload] = []
    client.add_group_update_listener(updates.append)
    conn = SendspinConnection(client)
    conn._ws = MagicMock(closed=False, send_str=AsyncMock(), close=AsyncMock())  # noqa: SLF001
    conn._connected = True  # noqa: SLF001
    conn.disconnect = AsyncMock()  # type: ignore[method-assign]
    return conn, updates


async def _receive_group_update(conn: SendspinConnection, group_id: str) -> None:
    payload = {"playback_state": "stopped", "group_id": group_id, "group_name": "Kitchen"}
    await conn._handle_ws_message(  # noqa: SLF001
        WSMessage(WSMsgType.TEXT, json.dumps({"type": "group/update", "payload": payload}), ""),
        conn.now_us(),
    )


@pytest.mark.parametrize(
    "group_id",
    [
        "0b7d5f0e-3c1a-4e4b-9a52-6d1c1f7a2b90",
        "A" * 42 + "+",
        "A" * 44,
    ],
)
async def test_malformed_group_id_closes_connection(group_id: str) -> None:
    """A malformed group_id closes the connection before any listener sees it."""
    conn, updates = _connection()

    await _receive_group_update(conn, group_id)
    assert conn._protocol_error_task is not None  # noqa: SLF001
    await conn._protocol_error_task  # noqa: SLF001

    conn.disconnect.assert_awaited_once()  # type: ignore[attr-defined]
    assert updates == []


async def test_base64url_group_id_is_delivered() -> None:
    """A group_id of 32 random bytes in base64url reaches the listeners."""
    conn, updates = _connection()
    group_id = secrets.token_urlsafe(32)

    await _receive_group_update(conn, group_id)

    conn.disconnect.assert_not_awaited()  # type: ignore[attr-defined]
    assert [update.group_id for update in updates] == [group_id]
