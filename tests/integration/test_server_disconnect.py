"""Server-side disconnect of a client that dialed in."""

from __future__ import annotations

import asyncio
import socket

from aiohttp import web

from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.models.types import AudioCodec, Roles
from aiosendspin.noise.keys import Identity, generate_psk, psk_id_for
from aiosendspin.noise.trust_store import (
    ClientPairingRecord,
    InMemoryClientPairingStore,
    InMemoryServerPairingStore,
    ServerPairingRecord,
)
from aiosendspin.server.server import SendspinServer
from tests.conftest import make_sdk_client


async def test_server_disconnect_ends_incoming_handler() -> None:
    """Disconnecting a client from the server side returns its incoming request handler."""
    server_store = InMemoryServerPairingStore()
    server = SendspinServer(
        loop=asyncio.get_running_loop(),
        identity=Identity.generate(),
        server_name="test-server",
        pairing_store=server_store,
    )
    identity = Identity.generate()
    client_store = InMemoryClientPairingStore()
    psk = generate_psk()
    await server_store.store_record(
        ServerPairingRecord(
            psk_id=psk_id_for(psk), psk=psk, client_id=identity.peer_id, pair_methods=[]
        )
    )
    await client_store.store_record(
        ClientPairingRecord(psk_id=psk_id_for(psk), psk=psk, server_id=server.id)
    )

    handler_done = asyncio.Event()

    async def handler(request: web.Request) -> web.StreamResponse:
        try:
            return await server.on_client_connect(request)
        finally:
            handler_done.set()

    app = web.Application()
    app.router.add_get(SendspinServer.API_PATH, handler)
    # Unlike aiohttp's TestServer, a plain AppRunner keeps handlers running after the peer leaves.
    runner = web.AppRunner(app)
    await runner.setup()
    sock = socket.create_server(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    await web.SockSite(runner, sock).start()
    client = make_sdk_client(
        identity=identity,
        pairing_store=client_store,
        client_name="c",
        roles=[Roles.PLAYER],
        player_support=ClientHelloPlayerSupport(
            supported_formats=[
                SupportedAudioFormat(
                    codec=AudioCodec.PCM, channels=2, sample_rate=48000, bit_depth=16
                )
            ],
            buffer_capacity=1_000_000,
        ),
    )
    try:
        await client.connect(f"ws://127.0.0.1:{port}{SendspinServer.API_PATH}")
        async with asyncio.timeout(5):
            while (server_client := server.get_client(identity.peer_id)) is None or (  # noqa: ASYNC110
                not server_client.is_connected
            ):
                await asyncio.sleep(0.01)
        connection = server_client.connection
        assert connection is not None

        await connection.disconnect(retry_connection=False)

        async with asyncio.timeout(5):
            await handler_done.wait()
    finally:
        await client.disconnect()
        await server.close()
        await runner.cleanup()
