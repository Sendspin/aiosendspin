"""End-to-end player supported_commands: declared in client/state, gating server/command."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from aiohttp import web
from aiohttp.test_utils import TestServer

from aiosendspin.models.core import ServerCommandPayload
from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.models.types import AudioCodec, PlayerCommand, Roles
from aiosendspin.noise.keys import Identity, generate_psk, psk_id_for
from aiosendspin.noise.trust_store import (
    ClientPairingRecord,
    InMemoryClientPairingStore,
    InMemoryServerPairingStore,
    ServerPairingRecord,
)
from aiosendspin.server.roles.player.v1 import PlayerV1Role
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


async def _await_player_role(server: SendspinServer, client_id: str) -> PlayerV1Role:
    async with asyncio.timeout(5):
        while True:
            client = server.get_client(client_id)
            if client is not None and client.is_connected:
                for role in client.active_roles:
                    if isinstance(role, PlayerV1Role):
                        return role
            await asyncio.sleep(0.01)


async def test_strict_server_gates_commands_on_sdk_state_list() -> None:
    """A strict server admits the SDK player and sends only the commands its state declared."""
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
            roles=[Roles.PLAYER],
            # Declared on player_support, so the SDK moves it into client/state.
            player_support=ClientHelloPlayerSupport(
                supported_formats=[
                    SupportedAudioFormat(
                        codec=AudioCodec.PCM, channels=2, sample_rate=48000, bit_depth=16
                    )
                ],
                buffer_capacity=1_000_000,
                supported_commands=[PlayerCommand.VOLUME],
            ),
        )
        received: list[ServerCommandPayload] = []
        client.add_server_command_listener(received.append)
        try:
            await client.connect(url)
            role = await _await_player_role(server, identity.peer_id)
            assert role.state_supported_commands == [PlayerCommand.VOLUME]

            role.set_player_mute(True)
            role.set_player_volume(30)
            async with asyncio.timeout(5):
                while not received:  # noqa: ASYNC110
                    await asyncio.sleep(0.01)

            assert [p.player.command for p in received if p.player] == [PlayerCommand.VOLUME]
        finally:
            await client.disconnect()
