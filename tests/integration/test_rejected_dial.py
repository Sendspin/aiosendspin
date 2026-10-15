"""End-to-end: a server whose dial the client rejects keeps the client for later playback."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace

import pytest
from aiohttp import ClientConnectionError, web
from aiohttp.test_utils import TestServer, unused_port

from aiosendspin.client.client import SendspinClient as SdkClient
from aiosendspin.client.models import PairingSupport
from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.models.types import (
    Activity,
    AudioCodec,
    ConnectionReason,
    GoodbyeReason,
    PairAbortReason,
    PairMethod,
    Roles,
)
from aiosendspin.noise.keys import Identity, generate_psk, psk_id_for
from aiosendspin.noise.pairing import (
    PairingAttempt,
    PairingError,
    RemotePairingAbortError,
)
from aiosendspin.noise.trust_store import (
    ClientPairingRecord,
    InMemoryClientPairingStore,
    InMemoryServerPairingStore,
    ServerPairingRecord,
)
from aiosendspin.server import client as client_module
from aiosendspin.server.server import (
    ClientAddedEvent,
    ClientConnectedEvent,
    ClientDisconnectedEvent,
    SendspinEvent,
    SendspinServer,
)
from tests.conftest import make_sdk_client

_STATIC_PAIRING_CODE = "12345678"


def _player_client(
    identity: Identity,
    store: InMemoryClientPairingStore,
    pairing_support: PairingSupport | None = None,
) -> SdkClient:
    return make_sdk_client(
        identity=identity,
        pairing_store=store,
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
        pairing_support=pairing_support,
    )


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
    client = _player_client(identity, client_store)

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


def _static_code_attempt(code: str) -> PairingAttempt:
    async def provide() -> str:
        return code

    return PairingAttempt(method=PairMethod.STATIC_PAIRING_CODE, pairing_code_provider=provide)


@asynccontextmanager
async def _held_elsewhere(
    holder_reason: ConnectionReason, *, opens_window: bool = True
) -> AsyncIterator[tuple[SdkClient, str, SendspinServer, str]]:
    """Yield a client another server holds, its id, the unpaired server it rejected and its URL."""
    identity = Identity.generate()
    client_id = identity.peer_id
    client_store = InMemoryClientPairingStore()
    await client_store.store_pairing_config(
        replace(
            await client_store.get_pairing_config(),
            static_pairing_code_enabled=True,
            unpaired_access_enabled=True,
        )
    )
    await client_store.set_static_pairing_code(_STATIC_PAIRING_CODE)
    holder = await _paired_server(identity, client_store)
    pairing = SendspinServer(
        loop=asyncio.get_running_loop(),
        identity=Identity.generate(),
        server_name="test-server",
        pairing_store=InMemoryServerPairingStore(),
    )
    rejected = asyncio.Event()

    def on_event(_server: SendspinServer, event: SendspinEvent) -> None:
        if (
            isinstance(event, ClientDisconnectedEvent)
            and event.goodbye_reason is GoodbyeReason.CONCURRENT_ATTEMPT
        ):
            rejected.set()

    pairing.add_event_listener(on_event)

    async def gesture_prompt(active: bool) -> None:  # noqa: FBT001
        if active and opens_window:
            client.open_pairing_window()

    client = _player_client(identity, client_store, PairingSupport(gesture_prompt=gesture_prompt))
    try:
        async with _host_incoming_client(client) as url:
            try:
                holder.connect_to_client(url, connection_reason=holder_reason)
                await _wait_connected(holder, client_id)
                pairing.connect_to_client(url)
                async with asyncio.timeout(5):
                    await rejected.wait()
                yield client, client_id, pairing, url
            finally:
                await client.disconnect()
    finally:
        await holder.close()
        await pairing.close()


async def test_initiate_pairing_pairs_a_client_held_elsewhere() -> None:
    """Pairing a client another server holds dials it, pairs, and declares no playback."""
    async with _held_elsewhere(ConnectionReason.DISCOVERY) as (client, client_id, pairing, _url):
        async with asyncio.timeout(5):
            await pairing.initiate_pairing(client_id, _static_code_attempt(_STATIC_PAIRING_CODE))
            assert await pairing.pairing_store.record_by_client_id(client_id) is not None
            while Activity.PAIRING in client.activities:  # noqa: ASYNC110
                await asyncio.sleep(0.01)
        assert Activity.PLAYBACK not in client.activities


async def test_initiate_pairing_raises_the_client_abort() -> None:
    """A pairing dial the client aborts raises its abort reason and stores no record."""
    async with _held_elsewhere(ConnectionReason.DISCOVERY) as (_client, client_id, pairing, _url):
        async with asyncio.timeout(5):
            with pytest.raises(RemotePairingAbortError) as excinfo:
                await pairing.initiate_pairing(client_id, _static_code_attempt("87654321"))
        assert excinfo.value.reason is PairAbortReason.PAIRING_CODE_MISMATCH
        assert await pairing.pairing_store.record_by_client_id(client_id) is None


async def test_pairing_refused_by_a_held_client_keeps_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pairing dial refused with concurrent_attempt keeps the client past the cleanup delay."""
    monkeypatch.setattr(client_module, "CLIENT_CLEANUP_DELAY", 0.1)
    async with _held_elsewhere(ConnectionReason.PLAYBACK) as (_client, client_id, pairing, _url):
        async with asyncio.timeout(5):
            with pytest.raises(RemotePairingAbortError) as excinfo:
                await pairing.initiate_pairing(
                    client_id, _static_code_attempt(_STATIC_PAIRING_CODE)
                )
            assert excinfo.value.reason is PairAbortReason.CONCURRENT_ATTEMPT
            while (c := pairing.get_client(client_id)) is not None and c.connection is not None:  # noqa: ASYNC110
                await asyncio.sleep(0.01)
        await asyncio.sleep(0.3)
        kept = pairing.get_client(client_id)
        assert kept is not None
        assert kept.cleanup_on_mdns_removal


async def test_cancelled_initiate_pairing_closes_its_dial() -> None:
    """Cancelling a pairing that dialed the client closes that dial."""
    async with _held_elsewhere(ConnectionReason.DISCOVERY, opens_window=False) as (
        _client,
        client_id,
        pairing,
        url,
    ):
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.3):
                await pairing.initiate_pairing(
                    client_id, _static_code_attempt(_STATIC_PAIRING_CODE)
                )
        async with asyncio.timeout(5):
            while url in pairing._connection_tasks:  # noqa: ASYNC110, SLF001
                await asyncio.sleep(0.01)


async def test_initiate_pairing_refuses_a_different_client_at_the_address() -> None:
    """A pairing dial answered by a different client pairs nothing and raises."""
    server = SendspinServer(
        loop=asyncio.get_running_loop(),
        identity=Identity.generate(),
        server_name="test-server",
        pairing_store=InMemoryServerPairingStore(),
    )
    identity = Identity.generate()
    client_store = InMemoryClientPairingStore()
    await client_store.store_pairing_config(
        replace(
            await client_store.get_pairing_config(),
            static_pairing_code_enabled=True,
            unpaired_access_enabled=True,
        )
    )
    await client_store.set_static_pairing_code(_STATIC_PAIRING_CODE)

    async def gesture_prompt(active: bool) -> None:  # noqa: FBT001
        if active:
            client.open_pairing_window()

    client = _player_client(identity, client_store, PairingSupport(gesture_prompt=gesture_prompt))
    try:
        async with _host_incoming_client(client) as url:
            try:
                server.get_or_create_client("client-1")
                server.register_client_url("client-1", url)
                async with asyncio.timeout(5):
                    with pytest.raises(PairingError, match="different client"):
                        await server.initiate_pairing(
                            "client-1", _static_code_attempt(_STATIC_PAIRING_CODE)
                        )
                assert await server.pairing_store.record_by_client_id(identity.peer_id) is None
            finally:
                await client.disconnect()
    finally:
        await server.close()


async def test_initiate_pairing_pairs_its_client_after_the_address_moved() -> None:
    """A pairing dial pairs its client even after another client took over the address."""
    server = SendspinServer(
        loop=asyncio.get_running_loop(),
        identity=Identity.generate(),
        server_name="test-server",
        pairing_store=InMemoryServerPairingStore(),
    )
    identity = Identity.generate()
    client_store = InMemoryClientPairingStore()
    await client_store.store_pairing_config(
        replace(
            await client_store.get_pairing_config(),
            static_pairing_code_enabled=True,
            unpaired_access_enabled=True,
        )
    )
    await client_store.set_static_pairing_code(_STATIC_PAIRING_CODE)

    async def gesture_prompt(active: bool) -> None:  # noqa: FBT001
        if active:
            client.open_pairing_window()

    client = _player_client(identity, client_store, PairingSupport(gesture_prompt=gesture_prompt))
    try:
        async with _host_incoming_client(client) as url:
            try:
                server.get_or_create_client(identity.peer_id)
                server.register_client_url(identity.peer_id, url)
                pairing = asyncio.create_task(
                    server.initiate_pairing(
                        identity.peer_id, _static_code_attempt(_STATIC_PAIRING_CODE)
                    )
                )
                await asyncio.sleep(0)
                server.register_client_url("other-client", url)
                async with asyncio.timeout(5):
                    await pairing
                assert await server.pairing_store.record_by_client_id(identity.peer_id) is not None
            finally:
                await client.disconnect()
    finally:
        await server.close()


async def test_initiate_pairing_raises_when_the_client_is_unreachable() -> None:
    """A dial for pairing that cannot connect raises the connection error."""
    server = SendspinServer(
        loop=asyncio.get_running_loop(),
        identity=Identity.generate(),
        server_name="test-server",
        pairing_store=InMemoryServerPairingStore(),
    )
    server.get_or_create_client("client-1")
    server.register_client_url("client-1", f"ws://127.0.0.1:{unused_port()}/sendspin")

    try:
        async with asyncio.timeout(5):
            with pytest.raises(ClientConnectionError):
                await server.initiate_pairing(
                    "client-1", _static_code_attempt(_STATIC_PAIRING_CODE)
                )
    finally:
        await server.close()
