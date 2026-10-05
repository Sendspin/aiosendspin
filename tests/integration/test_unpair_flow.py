"""End-to-end unpair tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from aiohttp import web
from aiohttp.test_utils import TestServer

from aiosendspin.client.client import SendspinClient as SdkClient
from aiosendspin.models.types import (
    GoodbyeReason,
    Roles,
)
from aiosendspin.noise.keys import Identity, generate_psk, psk_id_for
from aiosendspin.noise.trust_store import (
    ClientPairingRecord,
    InMemoryClientPairingStore,
    InMemoryServerPairingStore,
    ServerPairingRecord,
)
from aiosendspin.server.client import SendspinClient
from aiosendspin.server.server import SendspinServer
from tests.conftest import make_sdk_client


def _make_server(store: InMemoryServerPairingStore) -> SendspinServer:
    return SendspinServer(
        loop=asyncio.get_running_loop(),
        identity=Identity.generate(),
        server_name="test-server",
        pairing_store=store,
    )


@asynccontextmanager
async def _serve(server: SendspinServer) -> AsyncIterator[str]:
    app = web.Application()
    app.router.add_get(SendspinServer.API_PATH, server.on_client_connect)
    test_server = TestServer(app)
    await test_server.start_server()
    try:
        yield f"ws://127.0.0.1:{test_server.port}{SendspinServer.API_PATH}"
    finally:
        await test_server.close()
        await server.close()


async def _seed_pairing(
    server: SendspinServer,
    server_store: InMemoryServerPairingStore,
    client_store: InMemoryClientPairingStore,
    client_id: str,
) -> str:
    """Pre-establish a long-term record on both sides; return its psk_id."""
    psk = generate_psk()
    psk_id = psk_id_for(psk)
    await server_store.store_record(
        ServerPairingRecord(psk_id=psk_id, psk=psk, client_id=client_id, pair_methods=[])
    )
    await client_store.store_record(
        ClientPairingRecord(psk_id=psk_id, psk=psk, server_id=server.id)
    )
    return psk_id


async def _await_connected_client(server: SendspinServer, client_id: str) -> SendspinClient:
    async with asyncio.timeout(5):
        while True:
            client = server.get_client(client_id)
            if client is not None and client.is_connected and client.connection is not None:
                return client
            await asyncio.sleep(0.01)


async def _await_disconnected(client: SdkClient) -> None:
    async with asyncio.timeout(5):
        while client.connected:  # noqa: ASYNC110
            await asyncio.sleep(0.01)


async def test_unpair_drops_record_and_closes() -> None:
    """server.unpair removes both sides' records and closes the client with goodbye 'unpaired'."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    identity = Identity.generate()
    client_store = InMemoryClientPairingStore()
    psk_id = await _seed_pairing(server, server_store, client_store, identity.peer_id)

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
        )
        try:
            await client.connect(url)
            server_client = await _await_connected_client(server, identity.peer_id)
            conn = server_client.connection
            assert conn is not None

            await server.unpair(identity.peer_id)
            await _await_disconnected(client)

            assert conn.goodbye_reason is GoodbyeReason.UNPAIRED
            assert await client_store.record_by_psk_id(psk_id) is None
            assert await server_store.record_by_client_id(identity.peer_id) is None
        finally:
            await client.disconnect()
