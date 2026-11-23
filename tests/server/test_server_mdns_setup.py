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
        self.goodbyes = AsyncMock()
        self.async_unregister_service = AsyncMock(side_effect=lambda _info: self.goodbyes())
        self.goodbyes_sent_before_close: bool | None = None
        self.async_close = AsyncMock(side_effect=self._record_close)
        _FakeAsyncZeroconf.instances.append(self)

    def _record_close(self) -> None:
        self.goodbyes_sent_before_close = self.goodbyes.await_count == 1


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


def _advertised_addresses(zeroconf: _FakeAsyncZeroconf) -> list[str]:
    info = zeroconf.async_register_service.await_args.args[0]
    return info.parsed_addresses()


async def test_specific_host_is_advertised(
    fake_zeroconf: type[_FakeAsyncZeroconf], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A server bound to one address advertises it, not the default route's address."""
    monkeypatch.setattr("aiosendspin.server.server.get_local_ip", lambda: "10.0.0.5")
    server = _make_server()
    try:
        await server.start_server(port=0, host="127.0.0.1", discover_clients=False)
        assert _advertised_addresses(fake_zeroconf.instances[0]) == ["127.0.0.1"]
    finally:
        await server.close()


async def test_chosen_interfaces_are_advertised_without_default_route(
    fake_zeroconf: type[_FakeAsyncZeroconf], monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no default route, the record carries the addresses mDNS runs on."""
    monkeypatch.setattr("aiosendspin.server.server.get_local_ip", lambda: None)
    server = _make_server()
    try:
        await server.start_server(
            port=0, zeroconf_interfaces=["192.168.50.2"], discover_clients=False
        )
        assert _advertised_addresses(fake_zeroconf.instances[0]) == ["192.168.50.2"]
    finally:
        await server.close()


async def test_goodbyes_are_sent_before_zeroconf_closes(
    fake_zeroconf: type[_FakeAsyncZeroconf],
) -> None:
    """Stopping the server withdraws its service before Zeroconf shuts down."""
    server = _make_server()
    await server.start_server(port=0, host="127.0.0.1", discover_clients=False)
    await server.close()
    assert fake_zeroconf.instances[0].goodbyes_sent_before_close is True
