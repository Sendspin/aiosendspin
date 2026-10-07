"""The message loop dispatches client/leave and skips unknown client message types."""

from __future__ import annotations

import logging
from types import SimpleNamespace
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
    def __init__(self, *, strict: bool, controller: bool) -> None:
        self._strict = strict
        self._controller = controller
        self.handle_leave = AsyncMock()
        self.noncompliance: list[str] = []
        self.active_roles: tuple[SimpleNamespace, ...] = ()

    def roles_by_family(self, family: str) -> list[object]:
        return [object()] if family == "controller" and self._controller else []

    def flag_noncompliance(self, reason: str) -> None:
        self.noncompliance.append(reason)
        if self._strict:
            raise ClientComplianceError(reason)


def _connection(
    texts: list[str], *, strict: bool = False, controller: bool = False
) -> tuple[SendspinConnection, _FakeClient]:
    server = MagicMock(allow_noncompliant_clients=not strict)
    conn = SendspinConnection(server, wsock_client=AsyncMock())
    client = _FakeClient(strict=strict, controller=controller)
    conn._client = client  # type: ignore[assignment]  # noqa: SLF001
    conn._transport = _AsyncIterTransport(texts)  # type: ignore[assignment]  # noqa: SLF001
    return conn, client


_LEAVE = orjson.dumps({"type": "client/leave", "payload": {}}).decode()
_NO_PAYLOAD_OBJECT = "sent a message without a payload object"


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
async def test_client_leave_without_payload_is_flagged_before_leaving(
    strict: bool,  # noqa: FBT001
) -> None:
    """A client/leave without a payload object is flagged, then handled unless strict."""
    conn, client = _connection([orjson.dumps({"type": "client/leave"}).decode()], strict=strict)

    await conn._run_message_loop()  # noqa: SLF001

    assert client.noncompliance == [_NO_PAYLOAD_OBJECT]
    assert client.handle_leave.await_count == (0 if strict else 1)


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("message_type", ["client/from-the-future", "server/from-the-future"])
async def test_unknown_message_type_keeps_connection_open(
    message_type: str,
    strict: bool,  # noqa: FBT001
) -> None:
    """An unknown type is skipped without a compliance flag, in default and strict mode."""
    conn, client = _connection([_unknown(message_type), _LEAVE], strict=strict)

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
    ("message_type", "reason"),
    [
        ("server/state", "sent a server-to-client message"),
        ("server/init", "sent a server-to-client message"),
        ("client/init", "sent client/init after the connection was established"),
        ("noise/handshake", "sent noise/handshake outside a re-handshake"),
    ],
)
async def test_message_type_the_client_must_not_send_is_flagged(
    message_type: str, reason: str
) -> None:
    """A type only the server sends, or one only valid during a handshake, is flagged."""
    conn, client = _connection([_unknown(message_type), _LEAVE])

    await conn._run_message_loop()  # noqa: SLF001

    assert client.noncompliance == [reason]
    client.handle_leave.assert_awaited_once_with()


@pytest.mark.parametrize(
    "message",
    [{"type": "client/from-the-future"}, {"type": "client/command", "payload": 5}],
)
async def test_message_without_payload_object_is_flagged(message: dict[str, object]) -> None:
    """A message without a payload object is flagged and skipped."""
    conn, client = _connection([orjson.dumps(message).decode(), _LEAVE])

    await conn._run_message_loop()  # noqa: SLF001

    assert client.noncompliance == [_NO_PAYLOAD_OBJECT]
    client.handle_leave.assert_awaited_once_with()


_MALFORMED = "sent a malformed client/command controller object"
_NOT_OFFERED = "sent a controller command the server did not offer"


@pytest.mark.parametrize(
    ("controller", "controller_active", "noncompliance"),
    [
        ({"command": "volume", "volume": 101}, True, [_MALFORMED]),
        ({"command": "volume", "volume": 101}, False, []),
        ({"command": "from-the-future"}, True, [_NOT_OFFERED]),
        ({"command": "from-the-future"}, False, []),
        ({"command": "seek", "position_ms": -1}, True, []),
    ],
)
async def test_controller_command_is_skipped_or_flagged(
    controller: dict[str, object],
    controller_active: bool,  # noqa: FBT001
    noncompliance: list[str],
) -> None:
    """A bad controller object is flagged only from an active controller, never a negative seek."""
    command = orjson.dumps({"type": "client/command", "payload": {"controller": controller}})
    conn, client = _connection([command.decode(), _LEAVE], controller=controller_active)

    await conn._run_message_loop()  # noqa: SLF001

    assert client.noncompliance == noncompliance
    client.handle_leave.assert_awaited_once_with()


async def test_request_format_without_role_object_is_flagged() -> None:
    """A stream/request-format carrying no role object is flagged."""
    request = orjson.dumps({"type": "stream/request-format", "payload": {}}).decode()
    conn, client = _connection([request, _LEAVE])

    await conn._run_message_loop()  # noqa: SLF001

    assert client.noncompliance == ["sent a stream/request-format without a role object"]


_NOT_AN_ENVELOPE = "sent a message that is not a valid message envelope"


@pytest.mark.parametrize(
    ("text", "reason", "strict"),
    [
        (
            orjson.dumps({"type": "client/goodbye", "payload": {}}).decode(),
            "sent a malformed client/goodbye",
            False,
        ),
        (orjson.dumps({"type": "client/time"}).decode(), _NO_PAYLOAD_OBJECT, False),
        (
            orjson.dumps({"type": "client/hello", "payload": {}}).decode(),
            "sent a second client/hello after the hello exchange",
            False,
        ),
        (orjson.dumps({"payload": {}}).decode(), _NOT_AN_ENVELOPE, False),
        (orjson.dumps({"type": [], "payload": {}}).decode(), _NOT_AN_ENVELOPE, False),
        ("[]", _NOT_AN_ENVELOPE, False),
        ("not json", _NOT_AN_ENVELOPE, False),
        ("not json", _NOT_AN_ENVELOPE, True),
    ],
)
async def test_undecodable_message_is_flagged_and_ends_the_loop(
    text: str,
    reason: str,
    strict: bool,  # noqa: FBT001
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A text message that fails to parse is flagged and ends the loop, rejecting when strict."""
    conn, client = _connection([text, _LEAVE], strict=strict)

    with caplog.at_level(logging.ERROR):
        await conn._run_message_loop()  # noqa: SLF001

    assert client.noncompliance == [reason]
    assert conn._closing is strict  # noqa: SLF001
    client.handle_leave.assert_not_awaited()
    assert not caplog.records


@pytest.mark.parametrize(
    ("reason", "noncompliance"),
    [("moving_house", []), (None, ["sent client/goodbye with a null reason"])],
)
async def test_unrecognized_or_null_goodbye_reason_disconnects_without_retry(
    reason: str | None, noncompliance: list[str]
) -> None:
    """An unrecognized or null goodbye reason ends without a reconnect, flagging only null."""
    goodbye = orjson.dumps({"type": "client/goodbye", "payload": {"reason": reason}}).decode()
    conn, client = _connection([goodbye, _LEAVE])
    conn.disconnect = AsyncMock()  # type: ignore[method-assign]

    await conn._run_message_loop()  # noqa: SLF001

    conn.disconnect.assert_awaited_once_with(retry_connection=False)
    client.handle_leave.assert_awaited_once_with()
    assert conn.goodbye_reason is None
    assert client.noncompliance == noncompliance


_BAD_PLAYER_STATE = orjson.dumps(
    {"type": "client/state", "payload": {"available": True, "player": {"volume": 150}}}
).decode()


async def test_malformed_inactive_role_object_is_ignored() -> None:
    """A client/state object for an inactive role is dropped unparsed and the rest applied."""
    conn, client = _connection([_BAD_PLAYER_STATE, _LEAVE], strict=True)
    conn._handle_client_state = AsyncMock()  # type: ignore[method-assign]  # noqa: SLF001

    await conn._run_message_loop()  # noqa: SLF001

    payload = conn._handle_client_state.await_args.args[0]  # noqa: SLF001
    assert payload.available is True
    assert payload.player is None
    client.handle_leave.assert_awaited_once_with()
    assert client.noncompliance == []


@pytest.mark.parametrize("strict", [False, True])
async def test_wire_type_deviation_is_flagged_before_dispatch(
    strict: bool,  # noqa: FBT001
) -> None:
    """A tolerated wire type is flagged first, and in strict mode the message never dispatches."""
    time = orjson.dumps({"type": "client/time", "payload": {"client_transmitted": 5.7}})
    conn, client = _connection([time.decode(), _LEAVE], strict=strict)
    conn.send_priority_message = MagicMock()  # type: ignore[method-assign]

    await conn._run_message_loop()  # noqa: SLF001

    assert client.noncompliance == [
        "client/time sent a number for 'client_transmitted' instead of an integer"
    ]
    if strict:
        conn.send_priority_message.assert_not_called()
        client.handle_leave.assert_not_awaited()
        assert conn._closing is True  # noqa: SLF001
    else:
        reply = conn.send_priority_message.call_args.args[0]
        assert reply.payload.client_transmitted == 5
        client.handle_leave.assert_awaited_once_with()


@pytest.mark.parametrize("strict", [False, True])
async def test_malformed_active_role_object_is_flagged(
    strict: bool,  # noqa: FBT001
) -> None:
    """A malformed active-role object is flagged and skipped, or rejected when strict."""
    conn, client = _connection([_BAD_PLAYER_STATE, _LEAVE], strict=strict)
    client.active_roles = (SimpleNamespace(role_family="player"),)
    conn._handle_client_state = AsyncMock()  # type: ignore[method-assign]  # noqa: SLF001

    await conn._run_message_loop()  # noqa: SLF001

    conn._handle_client_state.assert_not_awaited()  # noqa: SLF001
    assert client.noncompliance == ["sent a malformed client/state"]
    assert conn._closing is strict  # noqa: SLF001
    assert client.handle_leave.await_count == (0 if strict else 1)
