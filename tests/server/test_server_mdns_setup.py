"""Tests for the interfaces and addresses in the server's mDNS setup."""

from __future__ import annotations

import asyncio
from typing import ClassVar
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiosendspin.noise.keys import Identity
from aiosendspin.noise.trust_store import InMemoryServerPairingStore
from aiosendspin.server.server import SendspinServer


class _FakeAsyncZeroconf:
    instances: ClassVar[list[_FakeAsyncZeroconf]] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.zeroconf = MagicMock()
        self.async_register_service = AsyncMock()
        self.async_unregister_service = AsyncMock()
        self.async_close = AsyncMock()
        _FakeAsyncZeroconf.instances.append(self)


@pytest.fixture
def fake_zeroconf(monkeypatch: pytest.MonkeyPatch) -> type[_FakeAsyncZeroconf]:
    """Replace Zeroconf with a double that records how the server set it up."""
    _FakeAsyncZeroconf.instances = []
    monkeypatch.setattr("aiosendspin.server.server.AsyncZeroconf", _FakeAsyncZeroconf)
    return _FakeAsyncZeroconf


def _make_server() -> SendspinServer:
    return SendspinServer(
        loop=asyncio.get_running_loop(),
        identity=Identity.generate(),
        server_name="server",
        pairing_store=InMemoryServerPairingStore(),
    )


async def test_chosen_interfaces_reach_zeroconf(fake_zeroconf: type[_FakeAsyncZeroconf]) -> None:
    """Zeroconf runs on the interfaces the host app chose."""
    server = _make_server()
    try:
        await server.start_server(
            port=0,
            zeroconf_interfaces=["192.168.1.10", "10.20.0.10"],
            discover_clients=False,
        )
        assert fake_zeroconf.instances[0].kwargs["interfaces"] == ["192.168.1.10", "10.20.0.10"]
    finally:
        await server.close()


async def test_non_ipv4_interface_is_rejected_before_listening(
    fake_zeroconf: type[_FakeAsyncZeroconf],
) -> None:
    """An IPv6 interface fails start_server without leaving the HTTP listener running."""
    server = _make_server()
    try:
        with pytest.raises(ValueError, match="IPv4"):
            await server.start_server(port=0, zeroconf_interfaces=["192.168.1.10", "fe80::1%2"])
        assert server._tcp_site is None  # noqa: SLF001
        assert not fake_zeroconf.instances
    finally:
        await server.close()
