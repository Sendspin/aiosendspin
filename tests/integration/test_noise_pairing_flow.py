"""End-to-end Noise tests: pairing, paired playback, bad PSK, and transition mode."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import replace

import pytest
from aiohttp import ClientSession, WSMsgType, web
from aiohttp.test_utils import TestServer

from aiosendspin.client.client import SendspinClient as SdkClient
from aiosendspin.client.models import PairingSupport
from aiosendspin.models.core import (
    ActivatePairing,
    ClientHelloMessage,
    ClientHelloPayload,
    ClientStateMessage,
    ServerActivateMessage,
    ServerActivatePayload,
    ServerHelloMessage,
)
from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.models.types import (
    Activity,
    AudioCodec,
    ClientMessage,
    PairAbortReason,
    PairingCodeFormat,
    PairMethod,
    PlayerCommand,
    Roles,
    ServerMessage,
    TrustLevel,
)
from aiosendspin.noise.keys import Identity, generate_psk, psk_id_for
from aiosendspin.noise.pairing import (
    PairingAbortError,
    PairingAttempt,
    PairingError,
    PairingTimeoutError,
)
from aiosendspin.noise.trust_store import (
    ClientPairingRecord,
    InMemoryClientPairingStore,
    InMemoryServerPairingStore,
    PairingPsk,
    PskCategory,
    ServerPairingRecord,
    StagedPairingPsk,
    TrustedUnpairedClient,
)
from aiosendspin.server.client import SendspinClient
from aiosendspin.server.connection import SendspinConnection
from aiosendspin.server.server import (
    ClientCredentialMismatchEvent,
    SendspinEvent,
    SendspinServer,
)
from tests.conftest import make_sdk_client


def _make_server(
    store: InMemoryServerPairingStore,
    *,
    allow_unencrypted: bool = False,
) -> SendspinServer:
    return SendspinServer(
        loop=asyncio.get_running_loop(),
        identity=Identity.generate(),
        server_name="test-server",
        pairing_store=store,
        allow_unencrypted=allow_unencrypted,
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


def _legacy_hello() -> str:
    return ClientHelloMessage(
        payload=ClientHelloPayload(
            client_id="legacy-client",
            name="legacy",
            version=1,
            supported_roles=[Roles.CONTROLLER.value],
        )
    ).to_json()


async def test_pairing_psk_flow_then_paired_playback() -> None:
    """Pair via a Pairing PSK, then reconnect with the established long-term PSK."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()

    # Operator-style setup: a Pairing PSK the client accepts and the server stages.
    pairing = generate_psk()
    psk_id = psk_id_for(pairing)
    await client_store.set_pairing_psk(PairingPsk(psk_id=psk_id, psk=pairing))
    await server_store.stage_pairing_psk(
        client_identity.peer_id, StagedPairingPsk(psk_id=psk_id, psk=pairing)
    )

    async with _serve(server) as url:
        pair_client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
        )
        # Pairing finalizes, the server re-handshakes onto the long-term PSK, and the
        # connection continues as a normal session (no disconnect).
        await pair_client.connect(url)
        assert pair_client.connected
        assert pair_client.noise_psk is not None
        assert pair_client.noise_psk.category is PskCategory.LONG_TERM

        client_record = await client_store.record_by_server_id(server.id)
        server_record = await server_store.record_by_client_id(client_identity.peer_id)
        assert client_record is not None
        assert server_record is not None
        assert client_record.psk == server_record.psk
        assert client_record.psk_id == server_record.psk_id
        await pair_client.disconnect()

        # Reconnect with the long-term PSK for a playback connection.
        play_client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
        )
        try:
            await play_client.connect(url)
            assert play_client.connected
            assert play_client.server_info is not None
            assert play_client.server_info.server_id == server.id
            assert play_client.noise_psk is not None
            assert play_client.noise_psk.category is PskCategory.LONG_TERM
        finally:
            await play_client.disconnect()


async def test_transition_mode_accepts_legacy_client() -> None:
    """With allow_unencrypted, a legacy client opening with client/hello gets server/hello."""
    server = _make_server(InMemoryServerPairingStore(), allow_unencrypted=True)
    async with (
        _serve(server) as url,
        ClientSession() as session,
        session.ws_connect(url) as ws,
    ):
        await ws.send_str(_legacy_hello())
        msg = await asyncio.wait_for(ws.receive(), timeout=5)
        assert msg.type is WSMsgType.TEXT
        assert isinstance(ServerMessage.from_json(msg.data), ServerHelloMessage)


async def test_default_server_rejects_legacy_client() -> None:
    """Without transition mode, a legacy client/hello is closed without a server/hello."""
    server = _make_server(InMemoryServerPairingStore())
    async with (
        _serve(server) as url,
        ClientSession() as session,
        session.ws_connect(url) as ws,
    ):
        await ws.send_str(_legacy_hello())
        msg = await asyncio.wait_for(ws.receive(), timeout=5)
        assert msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED)


async def test_transition_mode_rejects_paired_client_downgrade() -> None:
    """A legacy hello claiming a client_id with a pairing record is refused."""
    store = InMemoryServerPairingStore()
    psk = generate_psk()
    await store.store_record(
        ServerPairingRecord(
            psk_id=psk_id_for(psk), psk=psk, client_id="legacy-client", pair_methods=[]
        )
    )
    server = _make_server(store, allow_unencrypted=True)
    async with (
        _serve(server) as url,
        ClientSession() as session,
        session.ws_connect(url) as ws,
    ):
        await ws.send_str(_legacy_hello())
        msg = await asyncio.wait_for(ws.receive(), timeout=5)
        assert msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED)
    assert server.get_client("legacy-client") is None


async def test_transition_mode_rejects_pairing_staged_client() -> None:
    """A legacy hello claiming a client_id with a staged Pairing PSK is refused."""
    store = InMemoryServerPairingStore()
    pairing = generate_psk()
    await store.stage_pairing_psk(
        "legacy-client", StagedPairingPsk(psk_id=psk_id_for(pairing), psk=pairing)
    )
    server = _make_server(store, allow_unencrypted=True)
    async with (
        _serve(server) as url,
        ClientSession() as session,
        session.ws_connect(url) as ws,
    ):
        await ws.send_str(_legacy_hello())
        msg = await asyncio.wait_for(ws.receive(), timeout=5)
        assert msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED)
    assert server.get_client("legacy-client") is None


async def test_transition_mode_rejects_trusted_unpaired_client() -> None:
    """A legacy hello claiming a trusted-unpaired client_id is refused."""
    store = InMemoryServerPairingStore()
    await store.add_trusted_unpaired(TrustedUnpairedClient(client_id="legacy-client"))
    server = _make_server(store, allow_unencrypted=True)
    async with (
        _serve(server) as url,
        ClientSession() as session,
        session.ws_connect(url) as ws,
    ):
        await ws.send_str(_legacy_hello())
        msg = await asyncio.wait_for(ws.receive(), timeout=5)
        assert msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED)
    assert server.get_client("legacy-client") is None


@asynccontextmanager
async def _serve_legacy_peer() -> AsyncIterator[tuple[str, asyncio.Event, list[str]]]:
    """Serve a fake legacy client: sends client/hello on connect, records TEXT frames."""
    closed = asyncio.Event()
    frames: list[str] = []

    async def handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_str(_legacy_hello())
        frames.extend([msg.data async for msg in ws if msg.type is WSMsgType.TEXT])
        closed.set()
        return ws

    app = web.Application()
    app.router.add_get("/sendspin", handler)
    test_server = TestServer(app)
    await test_server.start_server()
    try:
        yield f"ws://127.0.0.1:{test_server.port}/sendspin", closed, frames
    finally:
        await test_server.close()


async def test_pairing_dial_refuses_legacy_client() -> None:
    """A dial carrying a pairing intent aborts when the peer answers with a legacy hello."""
    server = _make_server(InMemoryServerPairingStore(), allow_unencrypted=True)
    try:
        async with _serve_legacy_peer() as (url, closed, frames):
            async with ClientSession() as session, session.ws_connect(url) as wsock:
                conn = SendspinConnection(
                    server,
                    wsock_client=wsock,
                    url=url,
                    pairing_attempt=PairingAttempt(
                        method=PairMethod.PAIRING_PSK, pairing_psk=generate_psk()
                    ),
                )
                await asyncio.wait_for(conn.handle_client(), timeout=5)
            await asyncio.wait_for(closed.wait(), timeout=5)
            assert frames == []
            assert server.get_client("legacy-client") is None
    finally:
        await server.close()


async def test_dial_enforces_expected_client_id_for_legacy_hello() -> None:
    """A legacy hello on a dial pinned to another client_id is refused."""
    server = _make_server(InMemoryServerPairingStore(), allow_unencrypted=True)
    try:
        async with _serve_legacy_peer() as (url, closed, frames):
            async with ClientSession() as session, session.ws_connect(url) as wsock:
                conn = SendspinConnection(
                    server,
                    wsock_client=wsock,
                    url=url,
                    expected_client_id="some-other-client",
                )
                await asyncio.wait_for(conn.handle_client(), timeout=5)
            await asyncio.wait_for(closed.wait(), timeout=5)
            assert frames == []
            assert server.get_client("legacy-client") is None
    finally:
        await server.close()


async def test_initiate_pairing_refuses_legacy_connection() -> None:
    """Operator-initiated pairing on an unencrypted connection raises PairingError."""
    server = _make_server(InMemoryServerPairingStore(), allow_unencrypted=True)
    async with (
        _serve(server) as url,
        ClientSession() as session,
        session.ws_connect(url) as ws,
    ):
        await ws.send_str(_legacy_hello())
        msg = await asyncio.wait_for(ws.receive(), timeout=5)
        assert msg.type is WSMsgType.TEXT  # admitted: legacy server/hello
        conn = await _find_connection_by_client_id(server, "legacy-client")
        with pytest.raises(PairingError, match="unencrypted"):
            await conn.initiate_pairing(
                PairingAttempt(method=PairMethod.PAIRING_PSK, pairing_psk=generate_psk())
            )


async def _find_connection_by_client_id(
    server: SendspinServer, client_id: str
) -> SendspinConnection:
    async with asyncio.timeout(5):
        while True:
            for conn in server._pending_connections:  # noqa: SLF001
                if conn._client_id == client_id:  # noqa: SLF001
                    return conn
            await asyncio.sleep(0.01)


async def _await_long_term_record(store: InMemoryClientPairingStore, server_id: str) -> None:
    async with asyncio.timeout(5):
        while await store.record_by_server_id(server_id) is None:  # noqa: ASYNC110
            await asyncio.sleep(0.01)


async def test_unknown_client_admitted_idle_on_sentinel() -> None:
    """An unknown client lands on Sentinel and receives server/activate(activities=[])."""
    server = _make_server(InMemoryServerPairingStore())
    async with _serve(server) as url:
        client = make_sdk_client(
            identity=Identity.generate(),
            pairing_store=InMemoryClientPairingStore(),
            client_name="c",
            roles=[Roles.CONTROLLER],
        )
        try:
            await client.connect(url)
            assert client.connected
            assert client.noise_psk is not None
            assert client.noise_psk.category is PskCategory.SENTINEL
            assert client.activities == []
        finally:
            await client.disconnect()


async def _unpaired_enabled_store() -> InMemoryClientPairingStore:
    """Return a client store that advertises and admits unpaired access."""
    store = InMemoryClientPairingStore()
    config = await store.get_pairing_config()
    await store.store_pairing_config(replace(config, unpaired_access_enabled=True))
    return store


def _server_active_role_count(server: SendspinServer, client_id: str) -> int:
    """Return the count of roles the server has activated for ``client_id`` (0 if unknown)."""
    client = server.get_client(client_id)
    return len(client.active_roles) if client is not None else 0


async def test_unpaired_sentinel_untrusted_activates_no_roles() -> None:
    """Sentinel client, client-side unpaired access on, server offers neither → no roles."""
    server = _make_server(InMemoryServerPairingStore())
    identity = Identity.generate()
    async with _serve(server) as url:
        client = make_sdk_client(
            identity=identity,
            pairing_store=await _unpaired_enabled_store(),
            client_name="c",
            roles=[Roles.CONTROLLER],
        )
        try:
            await client.connect(url)
            assert client.noise_psk is not None
            assert client.noise_psk.category is PskCategory.SENTINEL
            assert _server_active_role_count(server, identity.peer_id) == 0
        finally:
            await client.disconnect()


async def test_trust_unpaired_before_connect_activates_roles() -> None:
    """A client pinned as trusted-unpaired while offline is admitted on connect."""
    server = _make_server(InMemoryServerPairingStore())
    identity = Identity.generate()
    await server.trust_unpaired(identity.peer_id)
    async with _serve(server) as url:
        client = make_sdk_client(
            identity=identity,
            pairing_store=await _unpaired_enabled_store(),
            client_name="c",
            roles=[Roles.CONTROLLER],
        )
        try:
            await client.connect(url)
            assert _server_active_role_count(server, identity.peer_id) == 1
        finally:
            await client.disconnect()
    trusted = await server.pairing_store.list_trusted_unpaired()
    assert [c.client_id for c in trusted] == [identity.peer_id]


async def test_live_trust_then_untrust_toggles_roles() -> None:
    """trust_unpaired/untrust_unpaired re-activate a live Sentinel session without reconnect."""
    server = _make_server(InMemoryServerPairingStore())
    identity = Identity.generate()
    async with _serve(server) as url:
        client = make_sdk_client(
            identity=identity,
            pairing_store=await _unpaired_enabled_store(),
            client_name="c",
            roles=[Roles.CONTROLLER],
        )
        try:
            await client.connect(url)
            assert _server_active_role_count(server, identity.peer_id) == 0
            await server.trust_unpaired(identity.peer_id)
            assert _server_active_role_count(server, identity.peer_id) == 1
            await server.untrust_unpaired(identity.peer_id)
            assert _server_active_role_count(server, identity.peer_id) == 0
        finally:
            await client.disconnect()


async def test_trusted_client_still_blocked_when_client_disables_unpaired() -> None:
    """A trusted client that itself refuses unpaired access gets no roles (client guard wins)."""
    server = _make_server(InMemoryServerPairingStore())
    identity = Identity.generate()
    await server.trust_unpaired(identity.peer_id)
    async with _serve(server) as url:
        client = make_sdk_client(
            identity=identity,
            pairing_store=InMemoryClientPairingStore(),  # unpaired access off (default)
            client_name="c",
            roles=[Roles.CONTROLLER],
        )
        try:
            await client.connect(url)
            assert _server_active_role_count(server, identity.peer_id) == 0
        finally:
            await client.disconnect()


async def test_live_pairing_dynamic_pairing_code() -> None:
    """Operator pairs a Sentinel-idle connection via a dynamic pairing code."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()

    shown: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def display(pairing_code: str | None) -> None:
        if pairing_code is not None and not shown.done():
            shown.set_result(pairing_code)

    async def provide() -> str:
        return await shown

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
            pairing_support=PairingSupport(pairing_code_display=display),
        )
        try:
            await client.connect(url)
            conn = await _find_connection_by_client_id(server, client_identity.peer_id)
            await conn.initiate_pairing(
                PairingAttempt(
                    method=PairMethod.DYNAMIC_PAIRING_CODE,
                    pairing_code_provider=provide,
                    pairing_format=PairingCodeFormat.DIGITS,
                )
            )
            await _await_long_term_record(client_store, server.id)
            assert client.noise_psk is not None
            assert client.noise_psk.category is PskCategory.LONG_TERM

            client_record = await client_store.record_by_server_id(server.id)
            server_record = await server_store.record_by_client_id(client_identity.peer_id)
            assert client_record is not None
            assert server_record is not None
            assert client_record.psk == server_record.psk
            assert client_record.psk_id == server_record.psk_id
            # The dynamic pairing code is always 6 digits.
            assert len(shown.result()) == 6
        finally:
            await client.disconnect()


@pytest.mark.parametrize(
    ("languages", "expected"),
    [(("ca", "es", "en"), ["ca", "es", "en"]), ((), None)],
)
async def test_live_pairing_dynamic_pairing_code_language_hint(
    languages: tuple[str, ...], expected: list[str] | None
) -> None:
    """The attempt's languages reach the client on the pairing server/activate, or are omitted."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()

    loop = asyncio.get_running_loop()
    shown: asyncio.Future[str] = loop.create_future()
    activation: asyncio.Future[ActivatePairing] = loop.create_future()

    async def display(pairing_code: str | None) -> None:
        if pairing_code is None or shown.done():
            return
        shown.set_result(pairing_code)
        conn = client._admitted_connection  # noqa: SLF001 - assert on the received activation
        assert conn is not None
        assert conn._selected_pairing is not None  # noqa: SLF001
        activation.set_result(conn._selected_pairing)  # noqa: SLF001

    async def provide() -> str:
        return await shown

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
            pairing_support=PairingSupport(pairing_code_display=display),
        )
        try:
            await client.connect(url)
            conn = await _find_connection_by_client_id(server, client_identity.peer_id)
            await conn.initiate_pairing(
                PairingAttempt(
                    method=PairMethod.DYNAMIC_PAIRING_CODE,
                    pairing_code_provider=provide,
                    languages=languages,
                    pairing_format=PairingCodeFormat.DIGITS,
                )
            )
            await _await_long_term_record(client_store, server.id)
            assert activation.result().languages == expected
        finally:
            await client.disconnect()


async def test_live_pairing_updates_connection_security_trust() -> None:
    """Pairing promotes the connection to the long-term PSK, and trust follows the category."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()

    shown: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def display(pairing_code: str | None) -> None:
        if pairing_code is not None and not shown.done():
            shown.set_result(pairing_code)

    async def provide() -> str:
        return await shown

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
            pairing_support=PairingSupport(pairing_code_display=display),
        )
        try:
            await client.connect(url)
            conn = await _find_connection_by_client_id(server, client_identity.peer_id)
            await conn.initiate_pairing(
                PairingAttempt(
                    method=PairMethod.DYNAMIC_PAIRING_CODE,
                    pairing_code_provider=provide,
                    pairing_format=PairingCodeFormat.DIGITS,
                )
            )
            server_client = conn._client  # noqa: SLF001
            assert server_client is not None
            security = server_client.connection_security
            assert security is not None
            assert security.psk_category is PskCategory.LONG_TERM
            assert security.trust_level is TrustLevel.USER
        finally:
            await client.disconnect()


async def test_live_pairing_method_enabled_after_hello_still_pairs() -> None:
    """A method enabled after client/hello can still pair: the client arbitrates, not the hello."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()
    config = await client_store.get_pairing_config()
    await client_store.store_pairing_config(replace(config, dynamic_pairing_code_enabled=False))

    shown: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def display(pairing_code: str | None) -> None:
        if pairing_code is not None and not shown.done():
            shown.set_result(pairing_code)

    async def provide() -> str:
        return await shown

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
            pairing_support=PairingSupport(pairing_code_display=display),
        )
        try:
            await client.connect(url)
            conn = await _find_connection_by_client_id(server, client_identity.peer_id)
            assert conn._client_info is not None  # noqa: SLF001
            info = conn._client_info  # noqa: SLF001
            offered = {d.method for d in (info.supported_pair_methods or [])}
            assert PairMethod.DYNAMIC_PAIRING_CODE not in offered

            config = await client_store.get_pairing_config()
            await client_store.store_pairing_config(
                replace(config, dynamic_pairing_code_enabled=True)
            )
            await conn.initiate_pairing(
                PairingAttempt(
                    method=PairMethod.DYNAMIC_PAIRING_CODE,
                    pairing_code_provider=provide,
                    pairing_format=PairingCodeFormat.DIGITS,
                )
            )
            await _await_long_term_record(client_store, server.id)
        finally:
            await client.disconnect()


async def test_live_pairing_qr_code() -> None:
    """Operator pairs by scanning the client-rendered token; digits channels stay silent."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()

    shown: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    digits_calls: list[str | None] = []
    spoken_calls: list[tuple[str | None, tuple[str, ...]]] = []

    async def qr_display(token: str | None) -> None:
        if token is not None and not shown.done():
            shown.set_result(token)

    async def digits_display(pairing_code: str | None) -> None:
        digits_calls.append(pairing_code)

    async def speak(pairing_code: str | None, *, languages: tuple[str, ...]) -> None:
        spoken_calls.append((pairing_code, languages))

    async def provide() -> str:
        return await shown

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
            pairing_support=PairingSupport(
                pairing_code_display=digits_display,
                pairing_code_speaker=speak,
                qr_code_display=qr_display,
            ),
        )
        try:
            await client.connect(url)
            conn = await _find_connection_by_client_id(server, client_identity.peer_id)
            assert conn._client_info is not None  # noqa: SLF001
            descriptor = next(
                d
                for d in (conn._client_info.supported_pair_methods or [])  # noqa: SLF001
                if d.method is PairMethod.DYNAMIC_PAIRING_CODE
            )
            assert descriptor.formats == ["digits", "qr_code"]
            await conn.initiate_pairing(
                PairingAttempt(
                    method=PairMethod.DYNAMIC_PAIRING_CODE,
                    pairing_code_provider=provide,
                    pairing_format=PairingCodeFormat.QR_CODE,
                )
            )
            await _await_long_term_record(client_store, server.id)
            assert shown.result().startswith("SP:1")
            assert digits_calls == []
            assert spoken_calls == []
        finally:
            await client.disconnect()


async def test_live_pairing_ignores_unrecognized_advertised_formats() -> None:
    """A descriptor format from a newer spec revision is ignored; the known ones still pair."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()

    shown: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def display(pairing_code: str | None) -> None:
        if pairing_code is not None and not shown.done():
            shown.set_result(pairing_code)

    async def provide() -> str:
        return await shown

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
            pairing_support=PairingSupport(pairing_code_display=display),
        )
        try:
            await client.connect(url)
            conn = await _find_connection_by_client_id(server, client_identity.peer_id)
            assert conn._client_info is not None  # noqa: SLF001
            descriptor = next(
                d
                for d in (conn._client_info.supported_pair_methods or [])  # noqa: SLF001
                if d.method is PairMethod.DYNAMIC_PAIRING_CODE
            )
            descriptor.formats = ["holographic", "digits"]

            await conn.initiate_pairing(
                PairingAttempt(
                    method=PairMethod.DYNAMIC_PAIRING_CODE,
                    pairing_code_provider=provide,
                    pairing_format=PairingCodeFormat.DIGITS,
                )
            )
            await _await_long_term_record(client_store, server.id)
        finally:
            await client.disconnect()


@pytest.mark.parametrize(
    ("formats", "match"),
    [
        (["holographic"], "does not offer the digits"),
        (None, "missing formats"),
    ],
)
async def test_live_pairing_unusable_advertised_formats(
    formats: list[str] | None, match: str
) -> None:
    """Only unknown formats offer nothing to select; a missing field is nonconformant."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
            pairing_support=PairingSupport(pairing_code_display=lambda _code: asyncio.sleep(0)),
        )
        try:
            await client.connect(url)
            conn = await _find_connection_by_client_id(server, client_identity.peer_id)
            assert conn._client_info is not None  # noqa: SLF001
            descriptor = next(
                d
                for d in (conn._client_info.supported_pair_methods or [])  # noqa: SLF001
                if d.method is PairMethod.DYNAMIC_PAIRING_CODE
            )
            descriptor.formats = formats

            with pytest.raises(PairingError, match=match):
                await conn.initiate_pairing(
                    PairingAttempt(
                        method=PairMethod.DYNAMIC_PAIRING_CODE,
                        pairing_code_provider=lambda: asyncio.sleep(0),
                        pairing_format=PairingCodeFormat.DIGITS,
                    )
                )
        finally:
            await client.disconnect()


async def test_live_pairing_unoffered_format_fails_before_activation() -> None:
    """Requesting qr_code from a digits-only client fails server-side, before any attempt."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
            pairing_support=PairingSupport(pairing_code_display=lambda _code: asyncio.sleep(0)),
        )
        try:
            await client.connect(url)
            conn = await _find_connection_by_client_id(server, client_identity.peer_id)
            with pytest.raises(PairingError, match="does not offer the qr_code"):
                await conn.initiate_pairing(
                    PairingAttempt(
                        method=PairMethod.DYNAMIC_PAIRING_CODE,
                        pairing_code_provider=lambda: asyncio.sleep(0),
                        pairing_format=PairingCodeFormat.QR_CODE,
                    )
                )
        finally:
            await client.disconnect()


async def test_live_pairing_unadvertised_format_client_aborts() -> None:
    """With no descriptor to gate on, an unoffered format reaches the client, which aborts."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()
    config = await client_store.get_pairing_config()
    await client_store.store_pairing_config(replace(config, dynamic_pairing_code_enabled=False))

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
            pairing_support=PairingSupport(pairing_code_display=lambda _code: asyncio.sleep(0)),
        )
        try:
            await client.connect(url)
            conn = await _find_connection_by_client_id(server, client_identity.peer_id)
            await client_store.store_pairing_config(
                replace(config, dynamic_pairing_code_enabled=True)
            )
            with pytest.raises(PairingAbortError) as exc_info:
                await conn.initiate_pairing(
                    PairingAttempt(
                        method=PairMethod.DYNAMIC_PAIRING_CODE,
                        pairing_code_provider=lambda: asyncio.sleep(0),
                        pairing_format=PairingCodeFormat.QR_CODE,
                    )
                )
            assert exc_info.value.reason is PairAbortReason.METHOD_NOT_SUPPORTED
        finally:
            await client.disconnect()


async def test_live_pairing_method_disabled_after_hello_aborts() -> None:
    """A method disabled after the hello is refused without closing; a retry can succeed."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()

    shown: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def display(pairing_code: str | None) -> None:
        if pairing_code is not None and not shown.done():
            shown.set_result(pairing_code)

    async def provide() -> str:
        return await shown

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
            pairing_support=PairingSupport(pairing_code_display=display),
        )
        try:
            await client.connect(url)
            conn = await _find_connection_by_client_id(server, client_identity.peer_id)
            config = await client_store.get_pairing_config()
            await client_store.store_pairing_config(
                replace(config, dynamic_pairing_code_enabled=False)
            )
            with pytest.raises(PairingAbortError) as exc_info:
                await conn.initiate_pairing(
                    PairingAttempt(
                        method=PairMethod.DYNAMIC_PAIRING_CODE,
                        pairing_code_provider=provide,
                        pairing_format=PairingCodeFormat.DIGITS,
                    )
                )
            assert exc_info.value.reason is PairAbortReason.METHOD_NOT_SUPPORTED
            await client_store.store_pairing_config(
                replace(config, dynamic_pairing_code_enabled=True)
            )
            await conn.initiate_pairing(
                PairingAttempt(
                    method=PairMethod.DYNAMIC_PAIRING_CODE,
                    pairing_code_provider=provide,
                    pairing_format=PairingCodeFormat.DIGITS,
                )
            )
            await _await_long_term_record(client_store, server.id)
        finally:
            await client.disconnect()


async def _await_left_pairing(client: SdkClient) -> None:
    async with asyncio.timeout(2):
        while Activity.PAIRING in client.activities:  # noqa: ASYNC110
            await asyncio.sleep(0.01)


async def test_live_pairing_dynamic_pairing_code_wrong_then_retry() -> None:
    """A wrong pairing code aborts the attempt but stays in pairing; a retry succeeds."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()

    shown_pins: list[str] = []
    pairing_code_ready = asyncio.Event()

    async def display(pairing_code: str | None) -> None:
        if pairing_code is not None:
            shown_pins.append(pairing_code)
            pairing_code_ready.set()

    attempts = 0

    async def provide() -> str:
        nonlocal attempts
        await pairing_code_ready.wait()
        pairing_code_ready.clear()
        attempts += 1
        correct = shown_pins[-1]
        if attempts == 1:
            return "000000" if correct != "000000" else "111111"  # wrong on the first try
        return correct

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
            pairing_support=PairingSupport(pairing_code_display=display),
        )
        try:
            await client.connect(url)
            conn = await _find_connection_by_client_id(server, client_identity.peer_id)

            with pytest.raises(PairingAbortError) as excinfo:
                await conn.initiate_pairing(
                    PairingAttempt(
                        method=PairMethod.DYNAMIC_PAIRING_CODE,
                        pairing_code_provider=provide,
                        pairing_format=PairingCodeFormat.DIGITS,
                    )
                )
            assert excinfo.value.reason is PairAbortReason.PAIRING_CODE_MISMATCH
            assert client.connected
            assert Activity.PAIRING in client.activities
            assert await client_store.pairing_code_failure_count() == 1
            # One attempt consumed, no re-handshake between attempts, so the index advanced.
            assert conn._pairing_index == 1  # noqa: SLF001

            # Retry on the same connection: fresh activate, fresh attempt index and code.
            await conn.initiate_pairing(
                PairingAttempt(
                    method=PairMethod.DYNAMIC_PAIRING_CODE,
                    pairing_code_provider=provide,
                    pairing_format=PairingCodeFormat.DIGITS,
                )
            )
            await _await_long_term_record(client_store, server.id)
            assert client.noise_psk is not None
            assert client.noise_psk.category is PskCategory.LONG_TERM
            assert await client_store.pairing_code_failure_count() == 0
            # The success re-handshake to the long-term PSK reset the per-handshake index.
            assert conn._pairing_index == 0  # noqa: SLF001
        finally:
            await client.disconnect()


async def test_stray_pairing_frame_outside_pairing_is_discarded() -> None:
    """A pairing frame reaching the server outside pairing is discarded, not fatal."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
        )
        try:
            await client.connect(url)
            await _find_connection_by_client_id(server, client_identity.peer_id)
            assert client._admitted_connection is not None  # noqa: SLF001
            await client._admitted_connection.send_pair_abort(  # noqa: SLF001
                PairAbortReason.USER_CANCELLED
            )
            await asyncio.sleep(0.1)  # a fatal frame would have torn the connection down
            assert client.connected
        finally:
            await client.disconnect()


async def test_end_pairing_after_failed_attempt_leaves_pairing() -> None:
    """After a failed attempt, end_pairing leaves pairing without dropping the connection."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()

    shown: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def display(pairing_code: str | None) -> None:
        if pairing_code is not None and not shown.done():
            shown.set_result(pairing_code)

    async def wrong_code() -> str:
        pairing_code = await shown
        return "000000" if pairing_code != "000000" else "111111"

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
            pairing_support=PairingSupport(pairing_code_display=display),
        )
        try:
            await client.connect(url)
            conn = await _find_connection_by_client_id(server, client_identity.peer_id)
            with pytest.raises(PairingAbortError):
                await conn.initiate_pairing(
                    PairingAttempt(
                        method=PairMethod.DYNAMIC_PAIRING_CODE,
                        pairing_code_provider=wrong_code,
                        pairing_format=PairingCodeFormat.DIGITS,
                    )
                )
            assert Activity.PAIRING in client.activities

            await server.end_pairing(client_identity.peer_id)
            await _await_left_pairing(client)
            assert client.connected
            assert await client_store.record_by_server_id(server.id) is None
        finally:
            await client.disconnect()


async def test_end_pairing_during_attempt_leaves_pairing() -> None:
    """end_pairing aborts a stalled attempt with user_cancelled, stays connected, re-pairs."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()

    shown_pins: list[str] = []
    displayed = asyncio.Event()
    never: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def display(pairing_code: str | None) -> None:
        if pairing_code is not None:
            shown_pins.append(pairing_code)
            displayed.set()

    async def stalling_provide() -> str:
        return await never  # the first attempt stalls in the pairing code provider until cancelled

    async def correct_provide() -> str:
        await displayed.wait()
        return shown_pins[-1]

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
            pairing_support=PairingSupport(pairing_code_display=display),
        )
        attempt: asyncio.Future[None] | None = None
        try:
            await client.connect(url)
            conn = await _find_connection_by_client_id(server, client_identity.peer_id)
            attempt = asyncio.ensure_future(
                conn.initiate_pairing(
                    PairingAttempt(
                        method=PairMethod.DYNAMIC_PAIRING_CODE,
                        pairing_code_provider=stalling_provide,
                        pairing_format=PairingCodeFormat.DIGITS,
                    )
                )
            )
            await displayed.wait()  # the client showed the pairing code: the attempt is in progress
            displayed.clear()

            await server.end_pairing(client_identity.peer_id)
            with pytest.raises(PairingAbortError) as excinfo:
                await attempt
            attempt = None
            assert excinfo.value.reason is PairAbortReason.USER_CANCELLED
            assert client.connected
            await _await_left_pairing(client)
            assert await client_store.record_by_server_id(server.id) is None
            assert conn._pairing_index == 1  # noqa: SLF001

            # The connection is reusable: a fresh attempt on it pairs successfully.
            await conn.initiate_pairing(
                PairingAttempt(
                    method=PairMethod.DYNAMIC_PAIRING_CODE,
                    pairing_code_provider=correct_provide,
                    pairing_format=PairingCodeFormat.DIGITS,
                )
            )
            await _await_long_term_record(client_store, server.id)
            assert client.noise_psk is not None
            assert client.noise_psk.category is PskCategory.LONG_TERM
        finally:
            if not never.done():
                never.cancel()
            if attempt is not None:
                attempt.cancel()
                with suppress(asyncio.CancelledError, PairingAbortError):
                    await attempt
            await client.disconnect()


async def test_gesture_timeout_leaves_pairing_without_dropping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The server's gesture bound cancels the attempt in band: no abort frame, connection alive."""
    monkeypatch.setattr("aiosendspin.noise.pairing.SERVER_GESTURE_TIMEOUT_S", 0.1)
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()
    await client_store.store_pairing_config(
        replace(await client_store.get_pairing_config(), static_pairing_code_enabled=True)
    )
    await client_store.set_static_pairing_code("12345678")

    aborts: list[PairAbortReason] = []

    async def gesture_prompt(active: bool) -> None:  # noqa: FBT001
        pass  # never opens a window: the server's gesture bound expires

    async def provide() -> str:
        return "12345678"

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
            pairing_support=PairingSupport(gesture_prompt=gesture_prompt),
        )
        client.add_pairing_abort_listener(aborts.append)
        try:
            await client.connect(url)
            conn = await _find_connection_by_client_id(server, client_identity.peer_id)
            with pytest.raises(PairingTimeoutError):
                await conn.initiate_pairing(
                    PairingAttempt(
                        method=PairMethod.STATIC_PAIRING_CODE, pairing_code_provider=provide
                    )
                )
            # The leave activate unparks the client; no pair/abort reason exists for this.
            await _await_left_pairing(client)
            assert client.connected
            assert aborts == []
            assert await client_store.record_by_server_id(server.id) is None

            # The connection is reusable: an opened window admits a fresh attempt.
            client.open_pairing_window()
            await conn.initiate_pairing(
                PairingAttempt(method=PairMethod.STATIC_PAIRING_CODE, pairing_code_provider=provide)
            )
            await _await_long_term_record(client_store, server.id)
        finally:
            await client.disconnect()


async def test_end_pairing_during_gesture_wait_unparks_client() -> None:
    """end_pairing reaches a client parked in the static gesture wait; it re-pairs after."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()
    await client_store.store_pairing_config(
        replace(await client_store.get_pairing_config(), static_pairing_code_enabled=True)
    )
    await client_store.set_static_pairing_code("12345678")

    prompts: list[bool] = []
    prompted = asyncio.Event()

    async def gesture_prompt(active: bool) -> None:  # noqa: FBT001
        prompts.append(active)
        if active:
            prompted.set()

    async def provide() -> str:
        return "12345678"

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
            pairing_support=PairingSupport(gesture_prompt=gesture_prompt),
        )
        attempt: asyncio.Future[None] | None = None
        try:
            await client.connect(url)
            conn = await _find_connection_by_client_id(server, client_identity.peer_id)
            attempt = asyncio.ensure_future(
                conn.initiate_pairing(
                    PairingAttempt(
                        method=PairMethod.STATIC_PAIRING_CODE, pairing_code_provider=provide
                    )
                )
            )
            await prompted.wait()  # the client is parked awaiting the operator gesture

            await server.end_pairing(client_identity.peer_id)
            with pytest.raises(PairingAbortError) as excinfo:
                await attempt
            attempt = None
            assert excinfo.value.reason is PairAbortReason.USER_CANCELLED
            assert client.connected
            await _await_left_pairing(client)
            assert prompts == [True, False]  # the SDK cleared the gesture prompt
            assert await client_store.record_by_server_id(server.id) is None

            # The connection is reusable: a proactively opened window admits a fresh attempt.
            client.open_pairing_window()
            await conn.initiate_pairing(
                PairingAttempt(method=PairMethod.STATIC_PAIRING_CODE, pairing_code_provider=provide)
            )
            await _await_long_term_record(client_store, server.id)
            assert client.noise_psk is not None
            assert client.noise_psk.category is PskCategory.LONG_TERM
        finally:
            if attempt is not None:
                attempt.cancel()
                with suppress(asyncio.CancelledError, PairingAbortError):
                    await attempt
            await client.disconnect()


async def test_external_cancel_of_initiate_pairing_stays_cancelled() -> None:
    """Cancelling the task running initiate_pairing ends it cancelled, not with the abort."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()

    displayed = asyncio.Event()
    never: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def display(pairing_code: str | None) -> None:
        if pairing_code is not None:
            displayed.set()

    async def stalling_provide() -> str:
        return await never

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
            pairing_support=PairingSupport(pairing_code_display=display),
        )
        try:
            await client.connect(url)
            conn = await _find_connection_by_client_id(server, client_identity.peer_id)
            attempt = asyncio.ensure_future(
                conn.initiate_pairing(
                    PairingAttempt(
                        method=PairMethod.DYNAMIC_PAIRING_CODE,
                        pairing_code_provider=stalling_provide,
                        pairing_format=PairingCodeFormat.DIGITS,
                    )
                )
            )
            await displayed.wait()  # the attempt is in progress, stalled on the pairing code

            attempt.cancel()
            with pytest.raises(asyncio.CancelledError):
                await attempt
            assert attempt.cancelled()
            # The forwarded cancel still aborted the attempt in-band: connection survives.
            assert client.connected
            assert Activity.PAIRING in client.activities
        finally:
            if not never.done():
                never.cancel()
            await client.disconnect()


async def _paired_client_with_stalled_success_tail(
    server: SendspinServer,
    url: str,
    client_identity: Identity,
    client_store: InMemoryClientPairingStore,
) -> tuple[SdkClient, asyncio.Future[None], asyncio.Event]:
    """Run a dynamic attempt up to the success re-handshake, which stalls until released.

    Returns (client, attempt future, release event); the attempt has finalized on return.
    """
    shown_pins: list[str] = []
    displayed = asyncio.Event()

    async def display(pairing_code: str | None) -> None:
        if pairing_code is not None:
            shown_pins.append(pairing_code)
            displayed.set()

    async def provide() -> str:
        await displayed.wait()
        return shown_pins[-1]

    client = make_sdk_client(
        identity=client_identity,
        pairing_store=client_store,
        client_name="c",
        roles=[Roles.CONTROLLER],
        pairing_support=PairingSupport(pairing_code_display=display),
    )
    await client.connect(url)
    conn = await _find_connection_by_client_id(server, client_identity.peer_id)

    original_rehandshake = conn._rehandshake_to  # noqa: SLF001
    entered = asyncio.Event()
    release = asyncio.Event()

    async def stalled_rehandshake(*args: object) -> bool:
        entered.set()
        await release.wait()
        return await original_rehandshake(*args)

    conn._rehandshake_to = stalled_rehandshake  # type: ignore[method-assign]  # noqa: SLF001
    attempt: asyncio.Future[None] = asyncio.ensure_future(
        conn.initiate_pairing(
            PairingAttempt(
                method=PairMethod.DYNAMIC_PAIRING_CODE,
                pairing_code_provider=provide,
                pairing_format=PairingCodeFormat.DIGITS,
            )
        )
    )
    await entered.wait()
    return client, attempt, release


async def test_cancel_racing_success_completes_pairing() -> None:
    """A cancel landing after finalize is absorbed: the attempt completes as a success."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()

    async with _serve(server) as url:
        client, attempt, release = await _paired_client_with_stalled_success_tail(
            server, url, client_identity, client_store
        )
        try:
            attempt.cancel()
            await asyncio.sleep(0)  # let the cancel forward into the attempt task
            release.set()

            await attempt  # completes: the cancel came too late to abort the pairing
            assert not attempt.cancelled()
            await _await_long_term_record(client_store, server.id)
            assert await server_store.record_by_client_id(client_identity.peer_id) is not None
            await _await_left_pairing(client)
            assert client.connected
            assert client.noise_psk is not None
            assert client.noise_psk.category is PskCategory.LONG_TERM
        finally:
            release.set()
            await client.disconnect()


async def test_end_pairing_racing_success_completes_pairing() -> None:
    """end_pairing after finalize completes the pairing instead of aborting it."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()

    async with _serve(server) as url:
        client, attempt, release = await _paired_client_with_stalled_success_tail(
            server, url, client_identity, client_store
        )
        try:
            end_task: asyncio.Future[None] = asyncio.ensure_future(
                server.end_pairing(client_identity.peer_id)
            )
            await asyncio.sleep(0)  # let end_pairing cancel the attempt task
            release.set()

            await end_task
            await attempt  # completes: end_pairing came too late to abort the pairing
            assert not attempt.cancelled()
            await _await_long_term_record(client_store, server.id)
            assert await server_store.record_by_client_id(client_identity.peer_id) is not None
            await _await_left_pairing(client)
            assert client.connected
            assert client.noise_psk is not None
            assert client.noise_psk.category is PskCategory.LONG_TERM
        finally:
            release.set()
            await client.disconnect()


async def test_live_pairing_pairing_psk() -> None:
    """Operator pairs a Sentinel-idle connection via Pairing PSK."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()

    pairing = generate_psk()
    await client_store.set_pairing_psk(PairingPsk(psk_id=psk_id_for(pairing), psk=pairing))

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
        )
        try:
            await client.connect(url)
            conn = await _find_connection_by_client_id(server, client_identity.peer_id)
            await conn.initiate_pairing(
                PairingAttempt(method=PairMethod.PAIRING_PSK, pairing_psk=pairing)
            )
            await _await_long_term_record(client_store, server.id)
            assert client.noise_psk is not None
            assert client.noise_psk.category is PskCategory.LONG_TERM

            client_record = await client_store.record_by_server_id(server.id)
            server_record = await server_store.record_by_client_id(client_identity.peer_id)
            assert client_record is not None
            assert server_record is not None
            assert client_record.psk == server_record.psk
        finally:
            await client.disconnect()


async def test_pairing_finalize_clears_staged_and_trusted_unpaired() -> None:
    """A finalized pairing removes the client's staged Pairing PSK and unpaired-trust grant."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()

    pairing = generate_psk()
    psk_id = psk_id_for(pairing)
    await client_store.set_pairing_psk(PairingPsk(psk_id=psk_id, psk=pairing))
    await server_store.stage_pairing_psk(
        client_identity.peer_id, StagedPairingPsk(psk_id=psk_id, psk=pairing)
    )
    await server_store.add_trusted_unpaired(
        TrustedUnpairedClient(client_id=client_identity.peer_id)
    )

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
        )
        try:
            await client.connect(url)
            assert client.noise_psk is not None
            assert client.noise_psk.category is PskCategory.LONG_TERM
            assert await server_store.record_by_client_id(client_identity.peer_id) is not None
            assert await server_store.staged_pairing_psk(client_identity.peer_id) is None
            assert await server_store.trusted_unpaired(client_identity.peer_id) is None
        finally:
            await client.disconnect()


async def test_reverification_leaves_staged_and_trusted_unpaired() -> None:
    """A verify attempt finalizes no record and leaves staged/trusted-unpaired entries alone."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()

    long_term = generate_psk()
    long_term_id = psk_id_for(long_term)
    await server_store.store_record(
        ServerPairingRecord(
            psk_id=long_term_id, psk=long_term, client_id=client_identity.peer_id, pair_methods=[]
        )
    )
    await client_store.store_record(
        ClientPairingRecord(psk_id=long_term_id, psk=long_term, server_id=server.id)
    )
    staged = generate_psk()
    await server_store.stage_pairing_psk(
        client_identity.peer_id, StagedPairingPsk(psk_id=psk_id_for(staged), psk=staged)
    )
    await server_store.add_trusted_unpaired(
        TrustedUnpairedClient(client_id=client_identity.peer_id)
    )

    shown: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def display(pairing_code: str | None) -> None:
        if pairing_code is not None and not shown.done():
            shown.set_result(pairing_code)

    async def provide() -> str:
        return await shown

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
            pairing_support=PairingSupport(pairing_code_display=display),
        )
        try:
            await client.connect(url)
            await server.initiate_pairing(
                client_identity.peer_id,
                PairingAttempt(
                    method=PairMethod.DYNAMIC_PAIRING_CODE,
                    pairing_code_provider=provide,
                    verify=True,
                    pairing_format=PairingCodeFormat.DIGITS,
                ),
            )
            assert await server_store.staged_pairing_psk(client_identity.peer_id) is not None
            assert await server_store.trusted_unpaired(client_identity.peer_id) is not None
        finally:
            await client.disconnect()


async def test_live_pairing_static_pairing_code() -> None:
    """Operator pairs a Sentinel-idle connection via a static pairing code once the window opens."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()
    await client_store.store_pairing_config(
        replace(await client_store.get_pairing_config(), static_pairing_code_enabled=True)
    )
    await client_store.set_static_pairing_code("12345678")
    await (
        client_store.record_pairing_code_failure()
    )  # dynamic pairing-code counter; static pairing ignores it

    window_opened = asyncio.get_running_loop().create_future()

    async def gesture_prompt(active: bool) -> None:  # noqa: FBT001
        if active and not window_opened.done():
            window_opened.set_result(None)
            client.open_pairing_window()

    async def provide() -> str:
        return "12345678"

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
            pairing_support=PairingSupport(gesture_prompt=gesture_prompt),
        )
        try:
            await client.connect(url)
            conn = await _find_connection_by_client_id(server, client_identity.peer_id)
            await conn.initiate_pairing(
                PairingAttempt(method=PairMethod.STATIC_PAIRING_CODE, pairing_code_provider=provide)
            )
            await _await_long_term_record(client_store, server.id)
            assert window_opened.done()
            assert client.noise_psk is not None
            assert client.noise_psk.category is PskCategory.LONG_TERM

            client_record = await client_store.record_by_server_id(server.id)
            server_record = await server_store.record_by_client_id(client_identity.peer_id)
            assert client_record is not None
            assert server_record is not None
            assert client_record.psk == server_record.psk
            assert client_record.psk_id == server_record.psk_id
            # The static flow leaves the dynamic-pairing-code failure counter alone.
            assert await client_store.pairing_code_failure_count() == 1
        finally:
            await client.disconnect()


async def test_live_pairing_escalated_dynamic_pairing_code_waits_for_window() -> None:
    """An escalated dynamic attempt is gesture-gated; the gesture unparks it and it pairs."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()
    for _ in range(10):
        await client_store.record_pairing_code_failure()
    assert await client_store.is_pairing_code_escalated()

    window_opened = asyncio.get_running_loop().create_future()
    pending_signals = 0

    def on_pending() -> None:
        nonlocal pending_signals
        pending_signals += 1

    async def gesture_prompt(active: bool) -> None:  # noqa: FBT001
        if active and not window_opened.done():
            window_opened.set_result(None)
            client.open_pairing_window()

    shown: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def display(pairing_code: str | None) -> None:
        if pairing_code is not None and not shown.done():
            shown.set_result(pairing_code)

    async def provide() -> str:
        return await shown

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
            pairing_support=PairingSupport(
                gesture_prompt=gesture_prompt, pairing_code_display=display
            ),
        )
        try:
            await client.connect(url)
            conn = await _find_connection_by_client_id(server, client_identity.peer_id)
            await conn.initiate_pairing(
                PairingAttempt(
                    method=PairMethod.DYNAMIC_PAIRING_CODE,
                    pairing_code_provider=provide,
                    on_pair_pending=on_pending,
                    pairing_format=PairingCodeFormat.DIGITS,
                )
            )
            await _await_long_term_record(client_store, server.id)
            assert window_opened.done()  # the attempt waited for the gesture
            assert pending_signals == 1  # the server surfaced the pending gesture
            assert client.noise_psk is not None
            assert client.noise_psk.category is PskCategory.LONG_TERM
            # Successful inner authentication de-escalates the method.
            assert not await client_store.is_pairing_code_escalated()
        finally:
            await client.disconnect()


async def test_live_pairing_pauses_writer_during_exchange() -> None:
    """initiate_pairing pauses the writer for the duration of the pairing exchange.

    The pairing/re-handshake sends and the writer share one Noise send-cipher, so a
    writer frame interleaved with them would advance the cipher nonce out of order and
    break the session. The pause is set in initiate_pairing for every method, so the
    dynamic-pairing-code flow here exercises it for all of them.
    """
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()

    loop = asyncio.get_running_loop()
    shown: asyncio.Future[str] = loop.create_future()
    writer_paused_mid_exchange: asyncio.Future[bool] = loop.create_future()

    async def display(pairing_code: str | None) -> None:
        if pairing_code is not None and not shown.done():
            shown.set_result(pairing_code)

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
            pairing_support=PairingSupport(pairing_code_display=display),
        )
        try:
            await client.connect(url)
            conn = await _find_connection_by_client_id(server, client_identity.peer_id)

            async def provide() -> str:
                # Mid-exchange: server/pair-init is out and the server awaits the pairing code.
                if not writer_paused_mid_exchange.done():
                    writer_paused_mid_exchange.set_result(conn._writer_task is None)  # noqa: SLF001
                # Queue writer work; with the writer paused it must wait for resume
                # rather than interleave with the rest of the exchange.
                for _ in range(64):
                    conn.send_priority_message(
                        ServerActivateMessage(payload=ServerActivatePayload(activities=[]))
                    )
                return await shown

            await conn.initiate_pairing(
                PairingAttempt(
                    method=PairMethod.DYNAMIC_PAIRING_CODE,
                    pairing_code_provider=provide,
                    pairing_format=PairingCodeFormat.DIGITS,
                )
            )

            assert await writer_paused_mid_exchange, "writer ran during the pairing exchange"
            assert conn._writer_task is not None  # noqa: SLF001  # resumed after the exchange
            await _await_long_term_record(client_store, server.id)
            assert client.connected
            assert client.noise_psk is not None
            assert client.noise_psk.category is PskCategory.LONG_TERM
        finally:
            await client.disconnect()


async def test_live_pairing_psk_pauses_writer_across_rehandshakes() -> None:
    """Pairing-PSK live pairing re-handshakes twice (Sentinel→Pairing→long-term).

    The writer stays paused across both re-handshakes, so it cannot interleave with
    either one. The store_record hook observes the writer state mid-exchange (after the
    first re-handshake, before the second).
    """
    loop = asyncio.get_running_loop()
    writer_paused_mid_exchange: asyncio.Future[bool] = loop.create_future()
    conn_holder: list[SendspinConnection] = []

    class _ObservingStore(InMemoryServerPairingStore):
        async def store_record(self, record: ServerPairingRecord) -> None:
            if conn_holder and not writer_paused_mid_exchange.done():
                paused = conn_holder[0]._writer_task is None  # noqa: SLF001
                writer_paused_mid_exchange.set_result(paused)
            await super().store_record(record)

    server_store = _ObservingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()

    pairing = generate_psk()
    await client_store.set_pairing_psk(PairingPsk(psk_id=psk_id_for(pairing), psk=pairing))

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
        )
        try:
            await client.connect(url)
            conn = await _find_connection_by_client_id(server, client_identity.peer_id)
            conn_holder.append(conn)
            await conn.initiate_pairing(
                PairingAttempt(method=PairMethod.PAIRING_PSK, pairing_psk=pairing)
            )
            assert await writer_paused_mid_exchange, "writer ran during the pairing exchange"
            assert conn._writer_task is not None  # noqa: SLF001  # resumed after the exchange
            await _await_long_term_record(client_store, server.id)
            assert client.connected
            assert client.noise_psk is not None
            assert client.noise_psk.category is PskCategory.LONG_TERM
        finally:
            await client.disconnect()


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


async def test_resync_resends_current_player_state() -> None:
    """After a re-verification, the client re-pushes its *current* player state.

    Dynamic pairing code over the long-term PSK re-verifies the pairing: the channel stays on the
    long-term PSK (no re-handshake), and the leave-pairing server/activate reactivates the
    player role. The client follows it with a fresh client/state carrying the volume/mute it
    last reported, not the construction-time initial values.
    """
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()

    # Pre-stage a shared long-term PSK so the client connects directly as paired playback.
    long_term = generate_psk()
    long_term_id = psk_id_for(long_term)
    await server_store.store_record(
        ServerPairingRecord(
            psk_id=long_term_id, psk=long_term, client_id=client_identity.peer_id, pair_methods=[]
        )
    )
    await client_store.store_record(
        ClientPairingRecord(psk_id=long_term_id, psk=long_term, server_id=server.id)
    )

    loop = asyncio.get_running_loop()
    shown: asyncio.Future[str] = loop.create_future()

    async def display(pairing_code: str | None) -> None:
        if pairing_code is not None and not shown.done():
            shown.set_result(pairing_code)

    async def provide() -> str:
        return await shown

    player_support = ClientHelloPlayerSupport(
        supported_formats=[
            SupportedAudioFormat(codec=AudioCodec.PCM, channels=2, sample_rate=44100, bit_depth=16)
        ],
        buffer_capacity=1_000_000,
        supported_commands=[PlayerCommand.VOLUME, PlayerCommand.MUTE],
    )

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.PLAYER],
            player_support=player_support,
            pairing_support=PairingSupport(pairing_code_display=display),
        )
        try:
            await client.connect(url)
            assert client.connected
            conn = await _find_connection_by_client_id(server, client_identity.peer_id)

            # The app moves volume/mute off the initial defaults (100/False).
            await client.send_player_state(available=True, volume=42, muted=True)
            await _await_player_state(conn, volume=42, muted=True)

            resync_state: asyncio.Future[tuple[int | None, bool | None]] = loop.create_future()
            pairing_started = False
            original_handle = conn._handle_message  # noqa: SLF001

            async def spy(message: ClientMessage, timestamp_us: int) -> None:
                if (
                    pairing_started
                    and isinstance(message, ClientStateMessage)
                    and message.payload.player is not None
                    and not resync_state.done()
                ):
                    resync_state.set_result(
                        (message.payload.player.volume, message.payload.player.muted)
                    )
                await original_handle(message, timestamp_us)

            conn._handle_message = spy  # type: ignore[method-assign]  # noqa: SLF001

            pairing_started = True
            await conn.initiate_pairing(
                PairingAttempt(
                    method=PairMethod.DYNAMIC_PAIRING_CODE,
                    pairing_code_provider=provide,
                    verify=True,
                    pairing_format=PairingCodeFormat.DIGITS,
                )
            )

            async with asyncio.timeout(5):
                resent = await resync_state
            assert resent == (42, True)
            assert client.noise_psk is not None
            assert client.noise_psk.category is PskCategory.LONG_TERM
        finally:
            await client.disconnect()


async def _await_connected_client(server: SendspinServer, client_id: str) -> SendspinClient:
    async with asyncio.timeout(5):
        while True:
            client = server.get_client(client_id)
            if client is not None and client.is_connected:
                return client
            await asyncio.sleep(0.01)


async def test_reverification_over_long_term_keeps_pairing() -> None:
    """Dynamic pairing code over a long-term PSK re-verifies without disturbing the pairing.

    The server runs the dynamic PAKE round but leaves pairing instead of finalizing: the
    connection stays on the *same* long-term PSK, no new record is stored on either side,
    and roles are reactivated. A successful round resets the failure counter like any other.
    """
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()

    long_term = generate_psk()
    long_term_id = psk_id_for(long_term)
    await server_store.store_record(
        ServerPairingRecord(
            psk_id=long_term_id, psk=long_term, client_id=client_identity.peer_id, pair_methods=[]
        )
    )
    await client_store.store_record(
        ClientPairingRecord(psk_id=long_term_id, psk=long_term, server_id=server.id)
    )
    # A pre-existing dynamic-pairing-code failure count is reset by a successful re-verification.
    await client_store.record_pairing_code_failure()

    shown: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def display(pairing_code: str | None) -> None:
        if pairing_code is not None and not shown.done():
            shown.set_result(pairing_code)

    async def provide() -> str:
        return await shown

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
            pairing_support=PairingSupport(pairing_code_display=display),
        )
        try:
            await client.connect(url)
            assert client.noise_psk is not None
            assert client.noise_psk.category is PskCategory.LONG_TERM

            server_client = await _await_connected_client(server, client_identity.peer_id)
            # The accessor reports the pre-check security state.
            assert server_client.is_paired
            security = server_client.connection_security
            assert security is not None
            assert security.psk_category is PskCategory.LONG_TERM
            assert security.trust_level is TrustLevel.USER

            await server.initiate_pairing(
                client_identity.peer_id,
                PairingAttempt(
                    method=PairMethod.DYNAMIC_PAIRING_CODE,
                    pairing_code_provider=provide,
                    verify=True,
                    pairing_format=PairingCodeFormat.DIGITS,
                ),
            )

            # The connection survives and stays on the *same* long-term PSK.
            assert client.connected
            assert client.noise_psk is not None
            assert client.noise_psk.category is PskCategory.LONG_TERM
            assert client.noise_psk.psk == long_term
            assert server_client.is_paired

            # The pairing PSK is unchanged on both sides; the verification is recorded server-side.
            client_record = await client_store.record_by_server_id(server.id)
            server_record = await server_store.record_by_client_id(client_identity.peer_id)
            assert client_record is not None
            assert server_record is not None
            assert client_record.psk == long_term
            assert server_record.psk == long_term
            assert server_record.pair_methods == [PairMethod.DYNAMIC_PAIRING_CODE]
            # The seeded long-term record plus the pre-provisioned shared fallback; nothing new.
            stored_pubkey = [
                r for r in await client_store.list_records() if r.server_id is not None
            ]
            assert len(stored_pubkey) == 1
            # Inner authentication succeeded, so the failure counter resets to zero.
            assert await client_store.pairing_code_failure_count() == 0
        finally:
            await client.disconnect()


async def test_reverification_under_escalation_is_gesture_gated() -> None:
    """Re-verification follows the escalation rules: gated on a window, de-escalated on success."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()

    long_term = generate_psk()
    long_term_id = psk_id_for(long_term)
    await server_store.store_record(
        ServerPairingRecord(
            psk_id=long_term_id, psk=long_term, client_id=client_identity.peer_id, pair_methods=[]
        )
    )
    await client_store.store_record(
        ClientPairingRecord(psk_id=long_term_id, psk=long_term, server_id=server.id)
    )
    # Drive dynamic pairing code into escalation (counter reaches 10).
    for _ in range(10):
        await client_store.record_pairing_code_failure()
    assert await client_store.is_pairing_code_escalated()

    window_opened = asyncio.get_running_loop().create_future()

    async def gesture_prompt(active: bool) -> None:  # noqa: FBT001
        if active and not window_opened.done():
            window_opened.set_result(None)
            client.open_pairing_window()

    shown: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def display(pairing_code: str | None) -> None:
        if pairing_code is not None and not shown.done():
            shown.set_result(pairing_code)

    async def provide() -> str:
        return await shown

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
            pairing_support=PairingSupport(
                gesture_prompt=gesture_prompt, pairing_code_display=display
            ),
        )
        try:
            await client.connect(url)
            conn = await _find_connection_by_client_id(server, client_identity.peer_id)
            await conn.initiate_pairing(
                PairingAttempt(
                    method=PairMethod.DYNAMIC_PAIRING_CODE,
                    pairing_code_provider=provide,
                    verify=True,
                    pairing_format=PairingCodeFormat.DIGITS,
                )
            )
            assert window_opened.done()  # the attempt waited for the gesture
            assert client.connected
            assert client.noise_psk is not None
            assert client.noise_psk.psk == long_term  # same long-term PSK, no re-pair
            assert not await client_store.is_pairing_code_escalated()
        finally:
            await client.disconnect()


async def test_initiate_pairing_raises_when_client_not_connected() -> None:
    """The server-level wrapper rejects a presence/pairing request for an absent client."""
    server = _make_server(InMemoryServerPairingStore())
    async with _serve(server):
        with pytest.raises(ValueError, match="not connected"):
            await server.initiate_pairing(
                "unknown-client",
                PairingAttempt(method=PairMethod.PAIRING_PSK, pairing_psk=generate_psk()),
            )


async def test_connection_security_reports_sentinel_for_unpaired() -> None:
    """An unpaired (Sentinel) connection reports is_paired=False and trust none."""
    server = _make_server(InMemoryServerPairingStore())
    identity = Identity.generate()
    async with _serve(server) as url:
        client = make_sdk_client(
            identity=identity,
            pairing_store=InMemoryClientPairingStore(),
            client_name="c",
            roles=[Roles.CONTROLLER],
        )
        try:
            await client.connect(url)
            server_client = await _await_connected_client(server, identity.peer_id)
            assert not server_client.is_paired
            security = server_client.connection_security
            assert security is not None
            assert security.psk_category is PskCategory.SENTINEL
            assert security.trust_level is TrustLevel.NONE
        finally:
            await client.disconnect()


async def _seed_long_term(
    server: SendspinServer,
    server_store: InMemoryServerPairingStore,
    client_store: InMemoryClientPairingStore,
    client_id: str,
) -> None:
    psk = generate_psk()
    psk_id = psk_id_for(psk)
    await server_store.store_record(
        ServerPairingRecord(psk_id=psk_id, psk=psk, client_id=client_id, pair_methods=[])
    )
    await client_store.store_record(
        ClientPairingRecord(psk_id=psk_id, psk=psk, server_id=server.id)
    )


@asynccontextmanager
async def _host_incoming_client(
    client: SdkClient, *, expected_server_id: str | None = None
) -> AsyncIterator[str]:
    """Host an SDK client's server-initiated (incoming) endpoint; yield its URL."""

    async def handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await client.attach_websocket(ws, expected_server_id=expected_server_id)
        return ws

    app = web.Application()
    app.router.add_get("/sendspin", handler)
    test_server = TestServer(app)
    await test_server.start_server()
    try:
        yield f"ws://127.0.0.1:{test_server.port}/sendspin"
    finally:
        await test_server.close()


@asynccontextmanager
async def _dial(server: SendspinServer, url: str) -> AsyncIterator[None]:
    """Dial ``url`` from ``server`` (server-initiated), running the connection in the background."""
    async with ClientSession() as session, session.ws_connect(url) as wsock:
        conn = SendspinConnection(server, wsock_client=wsock, url=url)
        task = asyncio.create_task(conn.handle_client())
        try:
            yield
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


async def _await_sdk_connected(client: SdkClient) -> None:
    async with asyncio.timeout(5):
        while not client.connected:  # noqa: ASYNC110
            await asyncio.sleep(0.01)


async def test_attach_websocket_admits_server_initiated_dial() -> None:
    """A server dial into the SDK client's incoming endpoint is admitted end-to-end.

    Drives the public ``SdkClient.attach_websocket`` orchestration — provisional
    tracking, the admission lock, admit, and steady-state — over a real socket.
    """
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    identity = Identity.generate()
    client_store = InMemoryClientPairingStore()
    await _seed_long_term(server, server_store, client_store, identity.peer_id)

    sdk = make_sdk_client(
        identity=identity,
        pairing_store=client_store,
        client_name="c",
        roles=[Roles.CONTROLLER],
    )
    try:
        async with _host_incoming_client(sdk) as url, _dial(server, url):
            await _await_sdk_connected(sdk)
            assert sdk.connected
            assert sdk.noise_psk is not None
            assert sdk.noise_psk.category is PskCategory.LONG_TERM
            server_client = await _await_connected_client(server, identity.peer_id)
            assert server_client.is_paired
    finally:
        await sdk.disconnect()
        await server.close()


async def test_attach_websocket_bringup_failure_is_swallowed() -> None:
    """A dial whose server_id fails the client's expectation aborts bring-up without admitting.

    Hosting the incoming endpoint with a mismatched ``expected_server_id`` makes
    the handshake abort; the public entry point must discard the provisional
    connection and leave the client unconnected rather than propagating the error.
    """
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    identity = Identity.generate()
    client_store = InMemoryClientPairingStore()
    await _seed_long_term(server, server_store, client_store, identity.peer_id)

    sdk = make_sdk_client(
        identity=identity,
        pairing_store=client_store,
        client_name="c",
        roles=[Roles.CONTROLLER],
    )
    try:
        async with (
            _host_incoming_client(sdk, expected_server_id="not-the-real-server") as url,
            _dial(server, url),
        ):
            # Bring-up aborts on the server_id mismatch; the client never connects.
            with pytest.raises(TimeoutError):
                await _await_sdk_connected(sdk)
            assert not sdk.connected
    finally:
        await sdk.disconnect()
        await server.close()


async def test_concurrent_server_dials_arbitrate_to_single_connection() -> None:
    """Two servers dialing one client concurrently converge on a single admitted connection.

    The admission lock must serialize the two incoming ``server/activate`` decisions
    so the client ends attached to exactly one server, not wedged or double-attached.
    """
    identity = Identity.generate()
    client_store = InMemoryClientPairingStore()

    store_a = InMemoryServerPairingStore()
    store_b = InMemoryServerPairingStore()
    server_a = _make_server(store_a)
    server_b = _make_server(store_b)
    # The same client identity is paired with both servers.
    await _seed_long_term(server_a, store_a, client_store, identity.peer_id)
    await _seed_long_term(server_b, store_b, client_store, identity.peer_id)

    sdk = make_sdk_client(
        identity=identity,
        pairing_store=client_store,
        client_name="c",
        roles=[Roles.CONTROLLER],
    )
    try:
        async with (
            _host_incoming_client(sdk) as url,
            _dial(server_a, url),
            _dial(server_b, url),
        ):
            await _await_sdk_connected(sdk)
            # The admission lock serialized the two incoming server/activate decisions,
            # so the client converged on a single admitted connection to one server.
            assert sdk.connected
            assert sdk._admitted_connection is not None  # noqa: SLF001
            assert sdk.server_info is not None
            assert sdk.server_info.server_id in {server_a.id, server_b.id}
    finally:
        await sdk.disconnect()
        await server_a.close()
        await server_b.close()


async def _await_server_disconnect(server: SendspinServer, client_id: str) -> None:
    async with asyncio.timeout(5):
        while True:
            client = server.get_client(client_id)
            if client is None or not client.is_connected:
                return
            await asyncio.sleep(0.01)


async def test_poisoned_transport_frame_drops_connection_cleanly() -> None:
    """A frame that fails Noise auth on a live session tears that connection down.

    Injecting undecryptable bytes at the raw transport of an established encrypted
    connection must drop that connection without wedging the server: a second,
    independent client still connects afterwards.
    """
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    victim = Identity.generate()
    victim_store = InMemoryClientPairingStore()
    await _seed_long_term(server, server_store, victim_store, victim.peer_id)

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=victim,
            pairing_store=victim_store,
            client_name="victim",
            roles=[Roles.CONTROLLER],
        )
        await client.connect(url)
        await _await_connected_client(server, victim.peer_id)

        # Inject garbage beneath the client's encryption layer: the server will
        # try to Noise-decrypt it, fail authentication, and drop the connection.
        raw_transport = client._admitted_connection._ws._ws  # noqa: SLF001
        await raw_transport.send_bytes(b"\x00" * 64)

        await _await_server_disconnect(server, victim.peer_id)
        await client.disconnect()

        # The server survived: a fresh, independent client still pairs and connects.
        survivor = Identity.generate()
        survivor_store = InMemoryClientPairingStore()
        await _seed_long_term(server, server_store, survivor_store, survivor.peer_id)
        other = make_sdk_client(
            identity=survivor,
            pairing_store=survivor_store,
            client_name="survivor",
            roles=[Roles.CONTROLLER],
        )
        try:
            await other.connect(url)
            survivor_client = await _await_connected_client(server, survivor.peer_id)
            assert survivor_client.is_connected
        finally:
            await other.disconnect()


async def test_lost_client_record_connects_on_the_sentinel_and_is_surfaced() -> None:
    """A client whose record is gone still connects, is reported, and gets no roles.

    The server keeps the record it holds — the mismatch says the client cannot use the
    credential, not that the record is wrong — but withholds playback until re-pairing.
    """
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    identity = Identity.generate()

    # The server holds a record the client no longer has: an eviction, a factory reset,
    # or a pairing finalize the client never persisted.
    psk = generate_psk()
    psk_id = psk_id_for(psk)
    await server_store.store_record(
        ServerPairingRecord(psk_id=psk_id, psk=psk, client_id=identity.peer_id, pair_methods=[])
    )

    seen: list[SendspinEvent] = []
    server.add_event_listener(lambda _server, event: seen.append(event))

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=identity,
            pairing_store=InMemoryClientPairingStore(),  # empty: the record is gone
            client_name="c",
            roles=[Roles.CONTROLLER],
        )
        try:
            await client.connect(url)
            conn = await _find_connection_by_client_id(server, identity.peer_id)

            assert client.connected
            assert conn._credential_mismatch is True  # noqa: SLF001
            assert conn._roles_to_activate == []  # noqa: SLF001
            assert [e.client_id for e in seen if isinstance(e, ClientCredentialMismatchEvent)] == [
                identity.peer_id
            ]
            # The record the server holds is untouched by the signal.
            assert await server_store.record_by_client_id(identity.peer_id) is not None
        finally:
            await client.disconnect()


async def test_re_pairing_restores_service_after_a_credential_mismatch() -> None:
    """The remedy the spec offers must actually work on the same connection.

    Pairing replaces the record, so the mismatch no longer stands and the session
    regains its roles without the client having to reconnect.
    """
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    identity = Identity.generate()

    psk = generate_psk()
    await server_store.store_record(
        ServerPairingRecord(
            psk_id=psk_id_for(psk), psk=psk, client_id=identity.peer_id, pair_methods=[]
        )
    )

    shown: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def display(pairing_code: str | None) -> None:
        if pairing_code is not None and not shown.done():
            shown.set_result(pairing_code)

    async def provide() -> str:
        return await shown

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=identity,
            pairing_store=InMemoryClientPairingStore(),  # the record is gone
            client_name="c",
            roles=[Roles.CONTROLLER],
            pairing_support=PairingSupport(pairing_code_display=display),
        )
        try:
            await client.connect(url)
            conn = await _find_connection_by_client_id(server, identity.peer_id)
            assert conn._credential_mismatch is True  # noqa: SLF001
            assert conn._roles_to_activate == []  # noqa: SLF001

            await conn.initiate_pairing(
                PairingAttempt(
                    method=PairMethod.DYNAMIC_PAIRING_CODE,
                    pairing_code_provider=provide,
                    pairing_format=PairingCodeFormat.DIGITS,
                )
            )

            assert conn._credential_mismatch is False  # noqa: SLF001
            assert conn._noise_psk is not None  # noqa: SLF001
            assert conn._noise_psk.category is PskCategory.LONG_TERM  # noqa: SLF001
            assert conn._roles_to_activate == ["controller@v1"]  # noqa: SLF001
        finally:
            await client.disconnect()


async def test_forgetting_a_mismatched_client_reactivates_it_in_place() -> None:
    """Forgetting the client is the other remedy, and it must not leave the session idle.

    A Sentinel session ignores ``server/unpair`` and stays connected, so the roles it may
    now carry have to be announced to it rather than waiting for a reconnect.
    """
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    identity = Identity.generate()

    psk = generate_psk()
    await server_store.store_record(
        ServerPairingRecord(
            psk_id=psk_id_for(psk), psk=psk, client_id=identity.peer_id, pair_methods=[]
        )
    )

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=identity,
            pairing_store=await _unpaired_enabled_store(),  # the record is gone
            client_name="c",
            roles=[Roles.CONTROLLER],
        )
        try:
            await client.connect(url)
            conn = await _find_connection_by_client_id(server, identity.peer_id)
            await server.trust_unpaired(identity.peer_id)

            # Trusted-unpaired alone cannot lift the hold while the record stands.
            assert conn._credential_mismatch is True  # noqa: SLF001
            assert conn._roles_to_activate == []  # noqa: SLF001

            await server.unpair(identity.peer_id)

            assert conn._credential_mismatch is False  # noqa: SLF001
            assert conn._roles_to_activate == ["controller@v1"]  # noqa: SLF001
            # Announced, not merely permitted: unpair awaits the re-activation.
            assert _server_active_role_count(server, identity.peer_id) == 1
        finally:
            await client.disconnect()
