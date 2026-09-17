"""The message loop dispatches client/leave and skips unknown client message types."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
from aiohttp import WSMessage, WSMsgType

from aiosendspin.models.core import ClientLeaveMessage
from aiosendspin.server.compliance import ClientComplianceError
from aiosendspin.server.connection import _MAX_WARNED_UNKNOWN_TYPES, SendspinConnection


class _AsyncIterTransport:
    close_code = 1000

    def __init__(self, texts: list[str]) -> None:
        self._msgs = [WSMessage(WSMsgType.TEXT, text, "") for text in texts]

    def __aiter__(self) -> _AsyncIterTransport:
        return self

    async def __anext__(self) -> WSMessage:
        if not self._msgs:
            raise StopAsyncIteration
        return self._msgs.pop(0)


class _FakeClient:
    def __init__(self, *, strict: bool) -> None:
        self._strict = strict
        self.handle_leave = AsyncMock()
        self.noncompliance: list[str] = []

    def flag_noncompliance(self, reason: str) -> None:
        self.noncompliance.append(reason)
        if self._strict:
            raise ClientComplianceError(reason)


def _connection(
    texts: list[str], *, strict: bool = False
) -> tuple[SendspinConnection, _FakeClient]:
    server = MagicMock(allow_noncompliant_clients=not strict)
    conn = SendspinConnection(server, wsock_client=AsyncMock())
    client = _FakeClient(strict=strict)
    conn._client = client  # type: ignore[assignment]  # noqa: SLF001
    conn._transport = _AsyncIterTransport(texts)  # type: ignore[assignment]  # noqa: SLF001
    return conn, client


_LEAVE = orjson.dumps({"type": "client/leave"}).decode()


def _unknown(message_type: str) -> str:
    return orjson.dumps({"type": message_type, "payload": {"field": 1}}).decode()


async def test_client_leave_is_handed_to_the_client() -> None:
    """client/leave reaches the persistent client's leave handler."""
    conn, client = _connection([])

    await conn._handle_message(ClientLeaveMessage(), timestamp_us=0)  # noqa: SLF001

    client.handle_leave.assert_awaited_once_with()


async def test_client_leave_without_client_is_ignored() -> None:
    """client/leave before a client is attached is dropped without raising."""
    conn, _ = _connection([])
    conn._client = None  # noqa: SLF001

    await conn._handle_message(ClientLeaveMessage(), timestamp_us=0)  # noqa: SLF001


@pytest.mark.parametrize("strict", [False, True])
async def test_unknown_message_type_keeps_connection_open(
    strict: bool,  # noqa: FBT001
) -> None:
    """An unknown type is skipped without a compliance flag, in default and strict mode."""
    conn, client = _connection([_unknown("client/from-the-future"), _LEAVE], strict=strict)

    await conn._run_message_loop()  # noqa: SLF001

    client.handle_leave.assert_awaited_once_with()
    assert client.noncompliance == []
    assert conn._closing is False  # noqa: SLF001


async def test_unknown_message_type_is_warned_once_per_type(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Each distinct unknown type is warned about once per connection."""
    conn, _ = _connection([_unknown("client/a"), _unknown("client/a"), _unknown("client/b")])

    with caplog.at_level(logging.WARNING):
        await conn._run_message_loop()  # noqa: SLF001

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert "client/a" in warnings[0]
    assert "client/b" in warnings[1]


async def test_unknown_message_type_warnings_are_capped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Distinct unknown types beyond the cap are ignored without a warning."""
    types = [f"client/unknown-{i}" for i in range(_MAX_WARNED_UNKNOWN_TYPES + 1)]
    conn, client = _connection([*(_unknown(t) for t in types), _LEAVE])

    with caplog.at_level(logging.WARNING):
        await conn._run_message_loop()  # noqa: SLF001

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == _MAX_WARNED_UNKNOWN_TYPES
    client.handle_leave.assert_awaited_once_with()


@pytest.mark.parametrize(
    "text",
    [
        orjson.dumps({"type": "client/goodbye", "payload": {}}).decode(),
        orjson.dumps({"payload": {}}).decode(),
        "not json",
    ],
)
async def test_malformed_message_ends_the_loop(text: str) -> None:
    """A malformed known type, a missing type, or non-JSON still ends the message loop."""
    conn, client = _connection([text, _LEAVE])

    await conn._run_message_loop()  # noqa: SLF001

    client.handle_leave.assert_not_awaited()


async def test_unrecognized_goodbye_reason_disconnects_without_retry() -> None:
    """An unrecognized goodbye reason parses, is recorded, and ends without a reconnect."""
    goodbye = orjson.dumps(
        {"type": "client/goodbye", "payload": {"reason": "moving_house"}}
    ).decode()
    conn, client = _connection([goodbye, _LEAVE])
    conn.disconnect = AsyncMock()  # type: ignore[method-assign]

    await conn._run_message_loop()  # noqa: SLF001

    conn.disconnect.assert_awaited_once_with(retry_connection=False)
    client.handle_leave.assert_awaited_once_with()
    assert conn.goodbye_reason is None
    assert client.noncompliance == []
