"""End-to-end: a server whose dial the client rejects keeps the client for later playback."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from aiohttp import web
from aiohttp.test_utils import TestServer

from aiosendspin.client.client import SendspinClient as SdkClient
from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.models.types import AudioCodec, GoodbyeReason, Roles
from aiosendspin.noise.keys import Identity, generate_psk, psk_id_for
from aiosendspin.noise.trust_store import (
    ClientPairingRecord,
    InMemoryClientPairingStore,
    InMemoryServerPairingStore,
    ServerPairingRecord,
)
from aiosendspin.server.server import (
    ClientAddedEvent,
    ClientConnectedEvent,
    ClientDisconnectedEvent,
    SendspinEvent,
    SendspinServer,
)
from tests.conftest import make_sdk_client


async def _paired_server(
    identity: Identity, client_store: InMemoryClientPairingStore
) -> SendspinServer:
    server = SendspinServer(
        loop=asyncio.get_running_loop(),
        identity=Identity.generate(),
        server_name="test-server",
        pairing_store=InMemoryServerPairingStore(),
    )
    psk = generate_psk()
    await server.pairing_store.store_record(
        ServerPairingRecord(
            psk_id=psk_id_for(psk), psk=psk, client_id=identity.peer_id, pair_methods=[]
        )
    )
    await client_store.store_record(
        ClientPairingRecord(psk_id=psk_id_for(psk), psk=psk, server_id=server.id)
    )
    return server


@asynccontextmanager
async def _host_incoming_client(client: SdkClient) -> AsyncIterator[str]:
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


async def _wait_connected(server: SendspinServer, client_id: str) -> None:
    async with asyncio.timeout(5):
        while (c := server.get_client(client_id)) is None or not c.is_connected:  # noqa: ASYNC110
            await asyncio.sleep(0.01)


async def test_rejected_dial_keeps_client_until_reclaimed() -> None:
    """A rejected dial announces and keeps the client, and playback on this server reclaims it."""
    identity = Identity.generate()
    client_id = identity.peer_id
    client_store = InMemoryClientPairingStore()
    holder = await _paired_server(identity, client_store)
    rejected = await _paired_server(identity, client_store)
    events: list[SendspinEvent] = []
    rejected.add_event_listener(lambda _server, event: events.append(event))
    # A player client holds back client/state until admitted, so a rejection precedes it.
    client = make_sdk_client(
        identity=identity,
        pairing_store=client_store,
        client_name="c",
        roles=[Roles.PLAYER],
        player_support=ClientHelloPlayerSupport(
            supported_formats=[
                SupportedAudioFormat(
                    codec=AudioCodec.PCM, sample_rate=48_000, bit_depth=16, channels=2
                )
            ],
            buffer_capacity=100_000,
        ),
    )

    try:
        async with _host_incoming_client(client) as url:
            try:
                holder.connect_to_client(url)
                await _wait_connected(holder, client_id)

                rejected.connect_to_client(url)
                async with asyncio.timeout(5):
                    while not any(isinstance(e, ClientDisconnectedEvent) for e in events):  # noqa: ASYNC110
                        await asyncio.sleep(0.01)

                assert events == [
                    ClientConnectedEvent(client_id),
                    ClientAddedEvent(client_id),
                    ClientDisconnectedEvent(client_id, GoodbyeReason.CONCURRENT_ATTEMPT),
                ]
                kept = rejected.get_client(client_id)
                assert kept is not None
                assert kept.info_or_none is not None
                assert kept.cleanup_on_mdns_removal

                assert rejected.request_client_playback_connection(client_id)
                await _wait_connected(rejected, client_id)
                assert rejected.get_client(client_id) is kept
            finally:
                # The incoming handler serves the client until it disconnects.
                await client.disconnect()
    finally:
        await holder.close()
        await rejected.close()
