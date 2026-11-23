"""Tests for renaming a running SendspinServer."""

from __future__ import annotations

import asyncio
from typing import ClassVar
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiosendspin.noise.keys import Identity
from aiosendspin.noise.trust_store import InMemoryServerPairingStore
from aiosendspin.server.connection import SendspinConnection
from aiosendspin.server.server import SendspinServer


class _FakeAsyncZeroconf:
    instances: ClassVar[list[_FakeAsyncZeroconf]] = []

    def __init__(self, **_kwargs: object) -> None:
        self.zeroconf = MagicMock()
        self.async_register_service = AsyncMock()
        self.async_unregister_service = AsyncMock(side_effect=lambda _info: AsyncMock()())
        self.broadcast = AsyncMock()
        self.async_update_service = AsyncMock(side_effect=lambda _info: self.broadcast())
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


async def test_rename_updates_the_record_and_later_hellos(
    fake_zeroconf: type[_FakeAsyncZeroconf],
) -> None:
    """A rename republishes the same record under the new name and reaches later hellos."""
    server = _make_server()
    try:
        await server.start_server(port=0, host="127.0.0.1", discover_clients=False)
        zeroconf = fake_zeroconf.instances[0]
        registered = zeroconf.async_register_service.await_args.args[0]

        await server.set_name("Living room")

        updated = zeroconf.async_update_service.await_args.args[0]
        assert updated.name == registered.name
        assert updated.parsed_addresses() == registered.parsed_addresses()
        assert updated.port == registered.port
        assert updated.properties[b"name"] == b"Living room"
        zeroconf.broadcast.assert_awaited_once()
        connection = MagicMock(_server=server)
        assert SendspinConnection._server_hello(connection).name == "Living room"  # noqa: SLF001
    finally:
        await server.close()
