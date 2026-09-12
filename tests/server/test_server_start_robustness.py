"""Tests for SendspinServer.start_server failure-path robustness."""

from __future__ import annotations

import asyncio
import socket
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiosendspin.noise.keys import Identity
from aiosendspin.noise.trust_store import InMemoryServerPairingStore
from aiosendspin.server.server import SendspinServer


def _make_server() -> SendspinServer:
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
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _FakeAsyncZeroconf:
    """Zeroconf test double avoiding real multicast sockets in tests."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.zeroconf = MagicMock()
        self.async_register_service = AsyncMock()
        self.async_unregister_service = AsyncMock()
        self.async_close = AsyncMock()


class _FakeAsyncServiceBrowser:
    """Service browser test double that skips real mDNS discovery."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.async_cancel = AsyncMock()


@pytest.mark.asyncio
async def test_start_server_survives_invalid_advertise_address(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unparsable advertise address must not tear down the HTTP listener or discovery."""
    monkeypatch.setattr("aiosendspin.server.server.AsyncZeroconf", _FakeAsyncZeroconf)
    monkeypatch.setattr("aiosendspin.server.server.AsyncServiceBrowser", _FakeAsyncServiceBrowser)

    server = _make_server()
    port = _free_port()

    with caplog.at_level("ERROR"):
        await server.start_server(
            port=port,
            host="127.0.0.1",
            advertise_addresses=["homeassistant.local"],
        )

    try:
        assert server._tcp_site is not None  # noqa: SLF001
        assert server._mdns_service is None  # noqa: SLF001
        assert server._mdns_browser is not None  # noqa: SLF001

        assert any("not advertised over mDNS" in record.getMessage() for record in caplog.records)

        # The port must still accept connections; the HTTP listener is unaffected.
        _reader, writer = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", port), timeout=2.0
        )
        writer.close()
        await writer.wait_closed()
    finally:
        await server.close()  # must not raise


@pytest.mark.asyncio
async def test_start_server_port_in_use_raises_and_close_is_safe() -> None:
    """A failed start_server (port already bound) must not corrupt teardown state."""
    server = _make_server()
    port = _free_port()

    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind(("127.0.0.1", port))
    blocker.listen(1)
    try:
        with pytest.raises(OSError, match="address already in use"):
            await server.start_server(port=port, host="127.0.0.1")
    finally:
        blocker.close()

    assert server._tcp_site is None  # noqa: SLF001
    assert server._app_runner is None  # noqa: SLF001
    assert server._app is None  # noqa: SLF001

    await server.close()  # must not raise
