"""End-to-end pairing-record capacity: eviction on pairing and the open-connection limit."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

import pytest
from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer

from aiosendspin.client.client import SendspinClient as SdkClient
from aiosendspin.client.connection import SendspinConnection as SdkConnection
from aiosendspin.models.types import Activity, GoodbyeReason, Roles
from aiosendspin.noise.keys import Identity, generate_psk, psk_id_for
from aiosendspin.noise.trust_store import (
    ClientPairingRecord,
    InMemoryClientPairingStore,
    InMemoryServerPairingStore,
    PairingPsk,
    PskCategory,
    ServerPairingRecord,
    StagedPairingPsk,
)
from aiosendspin.server.connection import SendspinConnection
from aiosendspin.server.server import SendspinServer
from tests.conftest import make_sdk_client
from tests.pairing_stores import seed_used_client_records


class _OpenConnection:
    """Stand-in for another open connection, backed by the given pairing records."""

    def __init__(self, *record_psk_ids: str) -> None:
        self.record_psk_ids = set(record_psk_ids)


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


@asynccontextmanager
async def _host_incoming_client(client: SdkClient) -> AsyncIterator[str]:
    """Host an SDK client's server-initiated (incoming) endpoint; yield its URL."""

    async def handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await client.attach_websocket(ws)
        return ws

    app = web.Application()
    app.router.add_get("/sendspin", handler)
    test_server = TestServer(app)
    await test_server.start_server()
    try:
        yield f"ws://127.0.0.1:{test_server.port}/sendspin"
    finally:
        await test_server.close()


async def _stage_pairing_psk(
    client_store: InMemoryClientPairingStore,
    server_store: InMemoryServerPairingStore,
    client_id: str,
) -> None:
    pairing = generate_psk()
    psk_id = psk_id_for(pairing)
    await client_store.set_pairing_psk(PairingPsk(psk_id=psk_id, psk=pairing))
    await server_store.stage_pairing_psk(client_id, StagedPairingPsk(psk_id=psk_id, psk=pairing))


async def _await_paired_session(client: SdkClient) -> None:
    async with asyncio.timeout(5):
        while (  # noqa: ASYNC110
            client.noise_psk is None
            or client.noise_psk.category is not PskCategory.LONG_TERM
            or Activity.PAIRING in client.activities
        ):
            await asyncio.sleep(0.01)


async def _per_server_psk_ids(store: InMemoryClientPairingStore) -> set[str]:
    return {r.psk_id for r in await store.list_records() if r.server_id is not None}


@pytest.mark.parametrize("protect_oldest", [False, True])
async def test_pairing_at_capacity_persists_by_evicting(*, protect_oldest: bool) -> None:
    """A pairing at capacity persists, evicting the oldest record no open connection uses."""
    identity = Identity.generate()
    client_store = InMemoryClientPairingStore(record_capacity=5)
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    seeded = await seed_used_client_records(client_store, 5)
    await _stage_pairing_psk(client_store, server_store, identity.peer_id)
    client = make_sdk_client(
        identity=identity, pairing_store=client_store, client_name="c", roles=[Roles.CONTROLLER]
    )
    if protect_oldest:
        other = _OpenConnection(seeded[0].psk_id)
        assert client._claim_connection_slot(other)  # type: ignore[arg-type]  # noqa: SLF001
    evicted = seeded[1] if protect_oldest else seeded[0]

    async with _serve(server) as url:
        try:
            await client.connect(url)
            await _await_paired_session(client)
            new = await client_store.record_by_server_id(server.id)
            assert new is not None
            assert client.noise_psk is not None
            assert client.noise_psk.psk_id == new.psk_id
            kept = {r.psk_id for r in seeded if r is not evicted}
            assert await _per_server_psk_ids(client_store) == kept | {new.psk_id}
            assert await server_store.record_by_client_id(identity.peer_id) is not None
        finally:
            await client.disconnect()


async def test_incoming_connection_over_the_limit_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With every connection slot taken, a server dial gets client/goodbye concurrent_attempt."""
    identity = Identity.generate()
    client_store = InMemoryClientPairingStore(record_capacity=5)
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    psk = generate_psk()
    await server_store.store_record(
        ServerPairingRecord(
            psk_id=psk_id_for(psk), psk=psk, client_id=identity.peer_id, pair_methods=[]
        )
    )
    await client_store.store_record(
        ClientPairingRecord(psk_id=psk_id_for(psk), psk=psk, server_id=server.id)
    )
    client = make_sdk_client(
        identity=identity, pairing_store=client_store, client_name="c", roles=[Roles.CONTROLLER]
    )
    for _ in range(4):
        assert client._claim_connection_slot(_OpenConnection())  # type: ignore[arg-type]  # noqa: SLF001
    goodbyes: list[GoodbyeReason] = []
    send_goodbye = SdkConnection.send_goodbye

    async def record_goodbye(self: SdkConnection, reason: GoodbyeReason) -> None:
        goodbyes.append(reason)
        await send_goodbye(self, reason)

    monkeypatch.setattr(SdkConnection, "send_goodbye", record_goodbye)

    try:
        async with (
            _host_incoming_client(client) as url,
            ClientSession() as session,
            session.ws_connect(url) as wsock,
        ):
            conn = SendspinConnection(server, wsock_client=wsock, url=url)
            task = asyncio.create_task(conn.handle_client())
            try:
                async with asyncio.timeout(5):
                    while not goodbyes:  # noqa: ASYNC110
                        await asyncio.sleep(0.01)
            finally:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
        assert goodbyes == [GoodbyeReason.CONCURRENT_ATTEMPT]
        assert not client.connected
        assert len(client._open_connections) == 4  # noqa: SLF001
    finally:
        await client.disconnect()
        await server.close()
