"""Tests for ClientListener."""

import pytest
from aiohttp import WSServerHandshakeError, web
from pytest_aiohttp import AiohttpClient

from aiosendspin.client.listener import DEFAULT_PORT, ClientListener


async def _noop_handler(_ws: web.WebSocketResponse) -> None:
    """No-op websocket handler for listener construction tests."""


def test_client_listener_default_port_matches_spec() -> None:
    """ClientListener should default to the spec-recommended listener port."""
    listener = ClientListener(client_id="client-1", on_connection=_noop_handler)
    assert DEFAULT_PORT == 8928
    assert listener.port == 8928


async def test_client_listener_rejects_upgrade_with_origin(aiohttp_client: AiohttpClient) -> None:
    """An upgrade carrying an Origin header gets 403."""
    listener = ClientListener(client_id="client-1", on_connection=_noop_handler)
    client = await aiohttp_client(listener._create_web_application())  # noqa: SLF001

    with pytest.raises(WSServerHandshakeError) as exc_info:
        await client.ws_connect("/sendspin", origin="http://example.com")

    assert exc_info.value.status == 403


async def test_client_listener_accepts_upgrade_without_origin(
    aiohttp_client: AiohttpClient,
) -> None:
    """An upgrade without an Origin header connects and reaches the handler."""
    connections: list[web.WebSocketResponse] = []

    async def handler(ws: web.WebSocketResponse) -> None:
        connections.append(ws)
        await ws.close()

    listener = ClientListener(client_id="client-1", on_connection=handler)
    client = await aiohttp_client(listener._create_web_application())  # noqa: SLF001

    ws = await client.ws_connect("/sendspin")
    await ws.receive()

    assert len(connections) == 1
