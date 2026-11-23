"""Tests for ClientListener."""

from unittest.mock import AsyncMock

import pytest
from aiohttp import WSServerHandshakeError, web
from pytest_aiohttp import AiohttpClient

from aiosendspin.client.listener import DEFAULT_PORT, ClientListener


class _FakeAsyncZeroconf:
    def __init__(self) -> None:
        self.async_register_service = AsyncMock()
        self.goodbyes = AsyncMock()
        self.async_unregister_service = AsyncMock(side_effect=lambda _info: self.goodbyes())
        self.goodbyes_sent_before_close: bool | None = None
        self.async_close = AsyncMock(side_effect=self._record_close)

    def _record_close(self) -> None:
        self.goodbyes_sent_before_close = self.goodbyes.await_count == 1


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


async def test_client_listener_sends_goodbyes_before_zeroconf_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stopping the listener withdraws its service before Zeroconf shuts down."""
    zeroconf = _FakeAsyncZeroconf()
    monkeypatch.setattr("aiosendspin.client.listener.AsyncZeroconf", lambda: zeroconf)
    listener = ClientListener(
        client_id="client-1", on_connection=_noop_handler, port=0, host="127.0.0.1"
    )
    await listener.start()
    await listener.stop()
    assert zeroconf.goodbyes_sent_before_close is True
