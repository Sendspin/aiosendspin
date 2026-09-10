"""End-to-end management command tests: records, gating, interleaving."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from aiosendspin.client.client import SendspinClient as SdkClient
from aiosendspin.client.models import PairingSupport
from aiosendspin.models.management import ManagementSetPairingConfigPayload, SetPairingPskConfig
from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.models.types import (
    Activity,
    AudioCodec,
    GoodbyeReason,
    ManagementResult,
    PlayerCommand,
    Roles,
)
from aiosendspin.noise.keys import Identity, b64url_encode, generate_psk, psk_id_for
from aiosendspin.noise.trust_store import (
    ClientPairingRecord,
    InMemoryClientPairingStore,
    InMemoryServerPairingStore,
    PairingPsk,
    ServerPairingRecord,
)
from aiosendspin.server.client import SendspinClient
from aiosendspin.server.connection import SendspinConnection
from aiosendspin.server.server import SendspinServer
from tests.conftest import make_sdk_client
from tests.pairing_stores import BoundedClientStore


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


async def _await_activity(client: SdkClient, activity: Activity) -> None:
    async with asyncio.timeout(5):
        while activity not in client.activities:  # noqa: ASYNC110
            await asyncio.sleep(0.01)


async def _await_without_activity(client: SdkClient, activity: Activity) -> None:
    async with asyncio.timeout(5):
        while activity in client.activities:  # noqa: ASYNC110
            await asyncio.sleep(0.01)


async def _await_disconnected(client: SdkClient) -> None:
    async with asyncio.timeout(5):
        while client.connected:  # noqa: ASYNC110
            await asyncio.sleep(0.01)


async def _await_player_state(conn: SendspinConnection, *, volume: int, muted: bool) -> None:
    async with asyncio.timeout(5):
        while True:
            server_client = conn._client  # noqa: SLF001
            if server_client is not None:
                for role in server_client.active_roles:
                    if (
                        role.role_family == "player"
                        and role.get_player_volume() == volume
                        and role.get_player_muted() == muted
                    ):
                        return
            await asyncio.sleep(0.01)


async def test_enable_management_adds_activity() -> None:
    """Enabling management adds the management activity without dropping the connection."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    identity = Identity.generate()
    client_store = InMemoryClientPairingStore()
    await _seed_pairing(server, server_store, client_store, identity.peer_id)

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
        )
        try:
            await client.connect(url)
            await _await_connected_client(server, identity.peer_id)
            server.enable_management(identity.peer_id)
            await _await_activity(client, Activity.MANAGEMENT)
            assert client.connected
        finally:
            await client.disconnect()


async def test_disable_management_keeps_connection() -> None:
    """Disabling management drops the activity and re-engages the gate, keeping the connection."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    identity = Identity.generate()
    client_store = InMemoryClientPairingStore()
    await _seed_pairing(server, server_store, client_store, identity.peer_id)

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
        )
        try:
            await client.connect(url)
            await _await_connected_client(server, identity.peer_id)
            conn = server.enable_management(identity.peer_id)
            await _await_activity(client, Activity.MANAGEMENT)

            server.disable_management(identity.peer_id)
            await _await_without_activity(client, Activity.MANAGEMENT)
            assert client.connected

            # The gate is re-engaged: a management/* request is now denied.
            result = await conn.remove_record(psk_id=psk_id_for(generate_psk()))
            assert result is ManagementResult.PERMISSION_DENIED
        finally:
            await client.disconnect()


async def test_records_round_trip() -> None:
    """list/add/remove-record over a management session mutate the client's store."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    identity = Identity.generate()
    client_store = InMemoryClientPairingStore()
    seeded_id = await _seed_pairing(server, server_store, client_store, identity.peer_id)

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
        )
        try:
            await client.connect(url)
            await _await_connected_client(server, identity.peer_id)
            conn = server.enable_management(identity.peer_id)
            await _await_activity(client, Activity.MANAGEMENT)

            result, records, _ = await conn.list_records()
            assert result is ManagementResult.OK
            # The seeded long-term record plus the pre-provisioned shared fallback.
            assert len(records) == 2
            by_id = {r.psk_id: r for r in records}
            # Connecting authenticated with the seeded record; the unused shared
            # fallback stays false.
            assert by_id[seeded_id].used is True
            assert all(not r.used for r in records if r.psk_id != seeded_id)

            added = generate_psk()
            added_id = psk_id_for(added)
            assert (await conn.add_record(psk=added, server_id="srv2")) is ManagementResult.OK
            _, records, _ = await conn.list_records()
            assert {r.psk_id for r in records} == {
                psk_id_for(s.psk) for s in await client_store.list_records()
            }
            assert await client_store.record_by_psk_id(added_id) is not None

            assert (await conn.remove_record(psk_id=added_id)) is ManagementResult.OK
            assert await client_store.record_by_psk_id(added_id) is None
            _, records, _ = await conn.list_records()
            assert len(records) == 2
        finally:
            await client.disconnect()


async def test_list_records_reports_storage_and_tracks_usage() -> None:
    """A bounded client reports capacity/free/costs on list-records, and free tracks usage."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    identity = Identity.generate()
    client_store = BoundedClientStore()
    await _seed_pairing(server, server_store, client_store, identity.peer_id)

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
        )
        try:
            await client.connect(url)
            await _await_connected_client(server, identity.peer_id)
            conn = server.enable_management(identity.peer_id)
            await _await_activity(client, Activity.MANAGEMENT)

            # Pre-provisioned shared record plus one seeded record: free = capacity - 2.
            _, _, storage = await conn.list_records()
            assert storage is not None
            assert (storage.capacity, storage.cost_individual, storage.cost_shared) == (4, 1, 1)
            assert storage.free == 2

            added = generate_psk()
            assert (await conn.add_record(psk=added, server_id="srv2")) is ManagementResult.OK
            _, _, after_add = await conn.list_records()
            assert after_add is not None
            assert after_add.free == 1

            assert (await conn.remove_record(psk_id=psk_id_for(added))) is ManagementResult.OK
            _, _, after_remove = await conn.list_records()
            assert after_remove is not None
            assert after_remove.free == 2
        finally:
            await client.disconnect()


async def test_management_request_without_session_is_denied() -> None:
    """A management/* request on a connection without the management activity is denied."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    identity = Identity.generate()
    client_store = InMemoryClientPairingStore()
    await _seed_pairing(server, server_store, client_store, identity.peer_id)

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
            result = await conn.remove_record(psk_id=psk_id_for(generate_psk()))
            assert result is ManagementResult.PERMISSION_DENIED
        finally:
            await client.disconnect()


async def test_paired_connection_allows_management() -> None:
    """Any paired (long-term) connection may run management; it is not closed."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    identity = Identity.generate()
    client_store = InMemoryClientPairingStore()
    await _seed_pairing(server, server_store, client_store, identity.peer_id)

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
        )
        try:
            await client.connect(url)
            await _await_connected_client(server, identity.peer_id)
            conn = server.enable_management(identity.peer_id)
            await _await_activity(client, Activity.MANAGEMENT)
            result, records, _ = await conn.list_records()
            assert result is ManagementResult.OK
            # The seeded long-term record plus the pre-provisioned shared fallback.
            assert len(records) == 2
            assert client.connected
        finally:
            await client.disconnect()


async def test_second_request_in_flight_raises() -> None:
    """At most one management request may be in flight on a connection."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    identity = Identity.generate()
    client_store = InMemoryClientPairingStore()
    await _seed_pairing(server, server_store, client_store, identity.peer_id)

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
        )
        try:
            await client.connect(url)
            await _await_connected_client(server, identity.peer_id)
            conn = server.enable_management(identity.peer_id)
            await _await_activity(client, Activity.MANAGEMENT)
            first = asyncio.ensure_future(conn.list_records())
            await asyncio.sleep(0)  # let the first request register as in flight
            with pytest.raises(RuntimeError, match="in flight"):
                await conn.list_records()
            await first
        finally:
            await client.disconnect()


async def test_get_pairing_config_returns_view_without_secrets() -> None:
    """get-pairing-config returns the config and never leaks the configured Pairing PSK."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    identity = Identity.generate()
    client_store = InMemoryClientPairingStore()
    await _seed_pairing(server, server_store, client_store, identity.peer_id)
    secret = generate_psk()
    await client_store.set_pairing_psk(PairingPsk(psk_id=psk_id_for(secret), psk=secret))

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
        )
        try:
            await client.connect(url)
            await _await_connected_client(server, identity.peer_id)
            conn = server.enable_management(identity.peer_id)
            await _await_activity(client, Activity.MANAGEMENT)

            result, data, _ = await conn.get_pairing_config()
            assert result is ManagementResult.OK
            assert data.pairing_psk is not None
            assert data.pairing_psk.enabled is True
            assert data.unpaired_access is not None
            # The configured PSK secret never appears in the response.
            assert b64url_encode(secret) not in data.to_json()
        finally:
            await client.disconnect()


async def test_set_pairing_config_disables_offered_method() -> None:
    """Disabling the Pairing PSK method stops the client offering it on the next connection."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    identity = Identity.generate()
    client_store = InMemoryClientPairingStore()
    await _seed_pairing(server, server_store, client_store, identity.peer_id)

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
        )
        try:
            await client.connect(url)
            first = await _await_connected_client(server, identity.peer_id)
            assert first.connection is not None
            info = first.connection._client_info  # noqa: SLF001
            assert info.supported_pair_methods is not None
            assert info.supported_pair_methods.pairing_psk is not None

            conn = server.enable_management(identity.peer_id)
            await _await_activity(client, Activity.MANAGEMENT)
            result = await conn.set_pairing_config(
                ManagementSetPairingConfigPayload(pairing_psk=SetPairingPskConfig(enabled=False))
            )
            assert result is ManagementResult.OK
        finally:
            await client.disconnect()

        # Reconnect (same store): the client no longer offers the disabled method.
        reconnect = make_sdk_client(
            identity=identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
        )
        try:
            await reconnect.connect(url)
            again = await _await_connected_client(server, identity.peer_id)
            assert again.connection is not None
            info = again.connection._client_info  # noqa: SLF001
            assert info.supported_pair_methods is not None
            assert info.supported_pair_methods.pairing_psk is None
        finally:
            await reconnect.disconnect()


async def test_open_pairing_window_round_trip() -> None:
    """open-pairing-window opens the window; with no pairing-code method it is invalid."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    identity = Identity.generate()
    client_store = InMemoryClientPairingStore()
    await _seed_pairing(server, server_store, client_store, identity.peer_id)

    async def display(pairing_code: str | None) -> None:
        pass

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
            pairing_support=PairingSupport(pairing_code_display=display),
        )
        try:
            await client.connect(url)
            await _await_connected_client(server, identity.peer_id)
            conn = server.enable_management(identity.peer_id)
            await _await_activity(client, Activity.MANAGEMENT)

            assert not client.pairing_window_open
            assert await conn.open_pairing_window() is ManagementResult.OK
            assert client.pairing_window_open
            # A no-op ok while the window is already open.
            assert await conn.open_pairing_window() is ManagementResult.OK
        finally:
            await client.disconnect()

        # A client with no pairing-code method enabled rejects the request as invalid.
        no_code = make_sdk_client(
            identity=identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
        )
        try:
            await no_code.connect(url)
            await _await_connected_client(server, identity.peer_id)
            conn = server.enable_management(identity.peer_id)
            await _await_activity(no_code, Activity.MANAGEMENT)
            assert await conn.open_pairing_window() is ManagementResult.INVALID
            assert not no_code.pairing_window_open
        finally:
            await no_code.disconnect()


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


async def test_unpair_keeps_shared_psk_record() -> None:
    """server/unpair closes the client but never removes a shared-PSK record."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    identity = Identity.generate()
    client_store = InMemoryClientPairingStore()
    psk = generate_psk()
    psk_id = psk_id_for(psk)
    await server_store.store_record(
        ServerPairingRecord(psk_id=psk_id, psk=psk, client_id=identity.peer_id, pair_methods=[])
    )
    # Shared-PSK record: no bound server_id.
    await client_store.store_record(ClientPairingRecord(psk_id=psk_id, psk=psk, server_id=None))

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
            # The shared-PSK record may back other servers: it is preserved on the client.
            assert await client_store.record_by_psk_id(psk_id) is not None
        finally:
            await client.disconnect()


async def test_management_interleaves_with_role_traffic() -> None:
    """A management session neither pauses the writer nor blocks active-role traffic."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    identity = Identity.generate()
    client_store = InMemoryClientPairingStore()
    await _seed_pairing(server, server_store, client_store, identity.peer_id)

    player_support = ClientHelloPlayerSupport(
        supported_formats=[
            SupportedAudioFormat(codec=AudioCodec.PCM, channels=2, sample_rate=44100, bit_depth=16)
        ],
        buffer_capacity=1_000_000,
        supported_commands=[PlayerCommand.VOLUME, PlayerCommand.MUTE],
    )

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.PLAYER],
            player_support=player_support,
        )
        try:
            await client.connect(url)
            await _await_connected_client(server, identity.peer_id)
            conn = server.enable_management(identity.peer_id)
            await _await_activity(client, Activity.MANAGEMENT)

            # The writer is never paused (unlike the pairing exchange).
            assert conn._writer_task is not None  # noqa: SLF001

            # Active-role traffic still flows while the management session is open.
            await client.send_player_state(available=True, volume=42, muted=True)
            await _await_player_state(conn, volume=42, muted=True)

            result, records, _ = await conn.list_records()
            assert result is ManagementResult.OK
            # The seeded long-term record plus the pre-provisioned shared fallback.
            assert len(records) == 2
            assert client.connected
            assert conn._writer_task is not None  # noqa: SLF001
        finally:
            await client.disconnect()
