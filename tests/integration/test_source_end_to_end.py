"""End-to-end source role: a paired client streams capture through a real server.

Drives a full SendspinServer + client SDK over a real WebSocket: pair, reconnect
with the long-term PSK, activate source@v1, then SourceCapture audio and assert the
server's SourceV1Role hands it back out of the decoded stream handle bit-exact.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any

from aiohttp import web
from aiohttp.test_utils import TestServer

from aiosendspin.models.player import SupportedAudioFormat
from aiosendspin.models.source import ClientHelloSourceFeatures, ClientHelloSourceSupport
from aiosendspin.models.types import AudioCodec, Roles
from aiosendspin.noise.keys import Identity, generate_psk, psk_id_for
from aiosendspin.noise.trust_store import (
    InMemoryClientPairingStore,
    InMemoryServerPairingStore,
    PairingPsk,
    PskCategory,
)
from aiosendspin.server.roles.source import SourceStreamStartedEvent
from aiosendspin.server.roles.source.v1 import SourceV1Role
from aiosendspin.server.server import SendspinServer
from tests.conftest import make_sdk_client, sine_pcm_16bit


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


def _source_support() -> ClientHelloSourceSupport:
    return ClientHelloSourceSupport(features=ClientHelloSourceFeatures(line_sense=True))


def _pcm_format() -> SupportedAudioFormat:
    return SupportedAudioFormat(codec=AudioCodec.PCM, channels=2, sample_rate=48000, bit_depth=16)


async def _request_start(client: Any, role: SourceV1Role) -> None:
    """Request a start and wait until the client receives it."""
    commanded: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    def _on_command(payload: Any) -> None:
        if payload.source is not None and not commanded.done():
            commanded.set_result(None)

    client.add_server_command_listener(_on_command)
    # Queued until the source state that follows the client's clock sync arrives.
    role.request_start()
    await asyncio.wait_for(commanded, timeout=5)


async def test_paired_source_client_streams_pcm_end_to_end() -> None:
    """A paired source client's captured PCM reaches the server handle bit-exact."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()

    pairing = generate_psk()
    pp = PairingPsk(psk_id=psk_id_for(pairing), psk=pairing)
    await client_store.set_pairing_psk(pp)
    await server_store.stage_pairing_psk(client_identity.peer_id, pp)

    async with _serve(server) as url:
        # First connect performs pairing onto a long-term PSK.
        pair_client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="src",
            roles=[Roles.SOURCE],
            source_support=_source_support(),
        )
        await pair_client.connect(url)
        async with asyncio.timeout(5):
            while (  # noqa: ASYNC110
                pair_client.noise_psk is None
                or pair_client.noise_psk.category is not PskCategory.LONG_TERM
            ):
                await asyncio.sleep(0.01)
        await pair_client.disconnect()

        # Reconnect on the long-term PSK; source@v1 is now activatable.
        play_client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="src",
            roles=[Roles.SOURCE],
            source_support=_source_support(),
        )
        try:
            await play_client.connect(url)
            assert play_client.connected

            server_client = server.get_client(client_identity.peer_id)
            assert server_client is not None
            assert "source@v1" in server_client.active_role_ids
            started: asyncio.Future[Any] = asyncio.get_running_loop().create_future()

            def _on_event(_client: Any, event: Any) -> None:
                if isinstance(event, SourceStreamStartedEvent) and not started.done():
                    started.set_result(event)

            server_client.add_event_listener(_on_event)

            source_role = server_client.role("source@v1")
            assert isinstance(source_role, SourceV1Role)
            await _request_start(play_client, source_role)

            pcm = sine_pcm_16bit(48000)
            capture = play_client.create_source_capture(_pcm_format())
            await capture.start()

            event = await asyncio.wait_for(started, timeout=5)
            handle = event.handle

            async def _drain() -> bytes:
                buf = bytearray()
                async for chunk, _ts in handle:
                    buf += chunk
                return bytes(buf)

            drain_task = asyncio.create_task(_drain())
            await capture.feed(pcm, capture_timestamp_us=play_client.now_us())
            await capture.stop()

            received = await asyncio.wait_for(drain_task, timeout=5)
            assert received == pcm
        finally:
            await play_client.disconnect()


async def test_unpaired_source_client_cannot_activate_source() -> None:
    """On an unpaired (sentinel) connection the server never activates source@v1.

    Source captures local audio (potentially a microphone), so the spec requires a
    paired connection; the server must withhold the role from unpaired clients.
    """
    server = _make_server(InMemoryServerPairingStore())
    client_store = InMemoryClientPairingStore()
    await client_store.store_pairing_config(
        replace(await client_store.get_pairing_config(), unpaired_access_enabled=True)
    )

    identity = Identity.generate()
    await server.trust_unpaired(identity.peer_id)

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=identity,
            pairing_store=client_store,
            client_name="unpaired-src",
            roles=[Roles.SOURCE],
            source_support=_source_support(),
        )
        try:
            await client.connect(url)
            assert client.noise_psk is not None
            assert client.noise_psk.category is PskCategory.SENTINEL
            server_client = server.get_client(client.identity.peer_id)
            assert server_client is not None
            assert "source@v1" not in server_client.active_role_ids
        finally:
            await client.disconnect()
