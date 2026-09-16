"""End-to-end visualizer stream configuration: requested in client/state, echoed in stream/start."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from aiohttp import web
from aiohttp.test_utils import TestServer

from aiosendspin.models.core import StreamStartMessage
from aiosendspin.models.types import Roles
from aiosendspin.models.visualizer import (
    ClientHelloVisualizerSupport,
    StreamStartVisualizer,
    VisualizerStatePayload,
)
from aiosendspin.noise.keys import Identity, generate_psk, psk_id_for
from aiosendspin.noise.trust_store import (
    ClientPairingRecord,
    InMemoryClientPairingStore,
    InMemoryServerPairingStore,
    ServerPairingRecord,
)
from aiosendspin.server.audio import AudioFormat
from aiosendspin.server.client import SendspinClient as ServerClient
from aiosendspin.server.server import SendspinServer
from tests.conftest import make_sdk_client


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


async def _await_connected(server: SendspinServer, client_id: str) -> ServerClient:
    async with asyncio.timeout(5):
        while True:
            client = server.get_client(client_id)
            if client is not None and client.is_connected:
                return client
            await asyncio.sleep(0.01)


async def _await_visualizer_start(
    starts: list[StreamStartMessage], count: int
) -> StreamStartVisualizer:
    async with asyncio.timeout(5):
        while True:
            configs = [m.payload.visualizer for m in starts if m.payload.visualizer is not None]
            if len(configs) >= count:
                config = configs[count - 1]
                assert isinstance(config, StreamStartVisualizer)
                return config
            await asyncio.sleep(0.01)


async def test_strict_server_streams_sdk_visualizer_state() -> None:
    """A strict server admits the SDK visualizer and follows its client/state requests."""
    server_store = InMemoryServerPairingStore()
    server = SendspinServer(
        loop=asyncio.get_running_loop(),
        identity=Identity.generate(),
        server_name="test-server",
        pairing_store=server_store,
        allow_noncompliant_clients=False,
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

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.VISUALIZER],
            visualizer_support=ClientHelloVisualizerSupport(buffer_capacity=1_000_000),
            visualizer_state=VisualizerStatePayload(types=["loudness"], rate_max=30),
        )
        starts: list[StreamStartMessage] = []
        client.add_stream_start_listener(starts.append)
        try:
            await client.connect(url)
            server_client = await _await_connected(server, identity.peer_id)

            stream = server_client.group.start_stream()
            stream.prepare_audio(
                bytes(19_200), AudioFormat(sample_rate=48_000, bit_depth=16, channels=2)
            )
            await stream.commit_audio()

            first = await _await_visualizer_start(starts, 1)
            assert first.types == ("loudness",)
            assert first.rate_max == 30

            await client.set_visualizer_state(
                VisualizerStatePayload(types=["loudness", "peak"], rate_max=15)
            )
            second = await _await_visualizer_start(starts, 2)
            assert second.types == ("loudness", "peak")
            assert second.rate_max == 15
            assert client.connected
        finally:
            await client.disconnect()
