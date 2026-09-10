"""How the server records what a client/hello reveals about its pair-method wire."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiosendspin.models.core import ClientHelloPayload
from aiosendspin.noise.keys import Identity
from aiosendspin.noise.trust_store import InMemoryServerPairingStore
from aiosendspin.server import SendspinServer
from aiosendspin.server.clock import LoopClock
from aiosendspin.server.compliance import ClientComplianceError
from aiosendspin.server.connection import SendspinConnection


@dataclass(slots=True)
class _DummyServer:
    loop: asyncio.AbstractEventLoop
    clock: Any
    id: str = "srv"
    name: str = "server"


def _conn_with_client() -> tuple[SendspinConnection, MagicMock]:
    loop = asyncio.get_running_loop()
    conn = SendspinConnection(
        _DummyServer(loop=loop, clock=LoopClock(loop)), wsock_client=MagicMock()
    )
    client = MagicMock()
    conn._client = client  # noqa: SLF001
    return conn, client


def _hello(pair_methods: Any) -> ClientHelloPayload:
    return ClientHelloPayload.from_dict(
        {
            "client_id": "c1",
            "name": "Client",
            "version": 1,
            "supported_roles": ["controller@v1"],
            "supported_pair_methods": pair_methods,
        }
    )


@pytest.mark.asyncio
async def test_current_wire_is_not_flagged() -> None:
    """A conformant advertisement raises nothing with the server."""
    conn, client = _conn_with_client()
    conn._note_client_hello_wire(_hello({"pairing_psk": {}}))  # noqa: SLF001
    client.flag_noncompliance.assert_not_called()


@pytest.mark.asyncio
async def test_superseded_list_shape_is_flagged() -> None:
    """The superseded list shape is a tolerated deviation, so the server records it."""
    conn, client = _conn_with_client()
    conn._note_client_hello_wire(_hello([{"method": "pairing_psk"}]))  # noqa: SLF001
    client.flag_noncompliance.assert_called_once()
    assert "supported_pair_methods as a list" in client.flag_noncompliance.call_args.args[0]


@pytest.mark.asyncio
async def test_offering_both_pairing_code_methods_is_flagged() -> None:
    """Offering both code methods breaks a MUST NOT, so it is a compliance failure."""
    conn, client = _conn_with_client()
    hello = _hello(
        {
            "static_pairing_code": {},
            "dynamic_pairing_code": {"formats": ["digits"], "out_channels": ["display"]},
        }
    )
    conn._note_client_hello_wire(hello)  # noqa: SLF001
    client.flag_noncompliance.assert_called_once()
    assert "both pairing-code methods" in client.flag_noncompliance.call_args.args[0]


@pytest.mark.asyncio
async def test_unrecognized_method_is_logged_but_not_flagged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A method from a newer revision is conformant: a strict server must still admit it."""
    conn, client = _conn_with_client()
    hello = _hello({"pairing_psk": {}, "telepathy": {"anything": [1, 2]}})
    with caplog.at_level(logging.INFO):
        conn._note_client_hello_wire(hello)  # noqa: SLF001
    client.flag_noncompliance.assert_not_called()
    assert any("telepathy" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_unusable_method_is_reported_apart_from_an_unknown_one(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A method dropped for unusable values must not read as a newer-revision identifier."""
    conn, client = _conn_with_client()
    hello = _hello(
        {
            "pairing_psk": {},
            "dynamic_pairing_code": {"formats": ["holographic"], "out_channels": ["display"]},
        }
    )
    with caplog.at_level(logging.INFO):
        conn._note_client_hello_wire(hello)  # noqa: SLF001
    client.flag_noncompliance.assert_not_called()
    assert any("no usable values" in record.message for record in caplog.records)
    assert not any("unrecognized pairing methods" in record.message for record in caplog.records)


def _strict_server() -> SendspinServer:
    loop = asyncio.get_running_loop()
    client_session = MagicMock()
    client_session.closed = True
    client_session.close = AsyncMock()
    return SendspinServer(
        loop=loop,
        identity=Identity.generate(),
        server_name="server",
        client_session=client_session,
        pairing_store=InMemoryServerPairingStore(),
        allow_noncompliant_clients=False,
    )


@pytest.mark.asyncio
async def test_strict_server_rejects_a_client_offering_both_code_methods() -> None:
    """The flag is not cosmetic: a strict server turns the MUST NOT breach into a rejection."""
    server = _strict_server()
    conn = SendspinConnection(server, wsock_client=MagicMock())
    conn._client = server.get_or_create_client("dev")  # noqa: SLF001
    hello = _hello(
        {
            "static_pairing_code": {},
            "dynamic_pairing_code": {"formats": ["digits"], "out_channels": ["display"]},
        }
    )

    with pytest.raises(ClientComplianceError, match="both pairing-code methods"):
        conn._note_client_hello_wire(hello)  # noqa: SLF001


@pytest.mark.asyncio
async def test_strict_server_admits_a_client_offering_an_unknown_method() -> None:
    """A client speaking a newer revision stays admissible even under strict compliance."""
    server = _strict_server()
    conn = SendspinConnection(server, wsock_client=MagicMock())
    conn._client = server.get_or_create_client("dev")  # noqa: SLF001

    conn._note_client_hello_wire(_hello({"pairing_psk": {}, "telepathy": {}}))  # noqa: SLF001
