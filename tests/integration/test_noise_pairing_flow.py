"""End-to-end Noise tests: pairing, paired playback, bad PSK, and transition mode."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import AsyncIterator, Callable, Iterator
from collections.abc import Set as AbstractSet
from contextlib import asynccontextmanager, contextmanager, suppress
from dataclasses import replace
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
from aiohttp import ClientSession, WSMsgType, web
from aiohttp.test_utils import TestServer

from aiosendspin.client import connection as client_connection_module
from aiosendspin.client.client import SendspinClient as SdkClient
from aiosendspin.client.connection import SendspinConnection as SdkConnection
from aiosendspin.client.models import PairingSupport
from aiosendspin.models.core import (
    ActivatePairing,
    ClientHelloMessage,
    ClientHelloPayload,
    ClientStateMessage,
    ClientStatePayload,
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
    ServerErrorReason,
    ServerMessage,
    TrustLevel,
)
from aiosendspin.noise import pairing as pairing_module
from aiosendspin.noise.driver import InitRejectedError
from aiosendspin.noise.keys import Identity, b64url_encode, generate_psk, psk_id_for
from aiosendspin.noise.models import (
    ClientPairFinalizeMessage,
    ClientPairFinalizePayload,
    ClientPairInitMessage,
    ClientPairInitPayload,
    ClientPairRetryMessage,
    ServerErrorMessage,
    ServerErrorPayload,
)
from aiosendspin.noise.pairing import (
    InvalidPairingCodeError,
    PairingAbortError,
    PairingAttempt,
    PairingError,
    PairingTimeoutError,
)
from aiosendspin.noise.trust_store import (
    PAIRING_ROUND_LIMIT,
    ClientPairingRecord,
    InMemoryClientPairingStore,
    InMemoryServerPairingStore,
    PairingPsk,
    PskCategory,
    ServerPairingRecord,
    StagedPairingPsk,
    TrustedUnpairedClient,
)
from aiosendspin.server import connection as connection_module
from aiosendspin.server.client import SendspinClient
from aiosendspin.server.compliance import ClientComplianceError
from aiosendspin.server.connection import SendspinConnection
from aiosendspin.server.server import (
    ClientCredentialMismatchEvent,
    SendspinEvent,
    SendspinServer,
)
from tests.conftest import make_sdk_client

if TYPE_CHECKING:
    from aiosendspin.noise.trust_store import ClientPairingStore
    from aiosendspin.noise.wire import EncryptedWebSocket


def _make_server(
    store: InMemoryServerPairingStore,
    *,
    allow_unencrypted: bool = False,
    languages: tuple[str, ...] | None = None,
    allow_noncompliant_clients: bool = True,
) -> SendspinServer:
    return SendspinServer(
        loop=asyncio.get_running_loop(),
        identity=Identity.generate(),
        server_name="test-server",
        pairing_store=store,
        allow_unencrypted=allow_unencrypted,
        languages=languages,
        allow_noncompliant_clients=allow_noncompliant_clients,
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


# DEPRECATED(spec-pr-287): remove in aiosendspin <version>
async def _rehandshake_with_hellos(self: SdkConnection, data: str) -> None:
    """Run a server-initiated re-handshake that re-exchanges hellos, as pre-#287 clients do."""
    await self._cancel_pairing_attempt()
    async with (
        self._exchange(),
        asyncio.timeout(client_connection_module.REHANDSHAKE_TIMEOUT_S),
    ):
        await self._rehandshake(data)
        activate = await self._exchange_hellos()
    await self._handle_server_activate(activate, resync=True)


# DEPRECATED(spec-pr-287): remove in aiosendspin <version>
def _pre_spec_287_rehandshake() -> Any:
    """Patch the SDK client to re-exchange hellos after every re-handshake."""
    return patch.object(SdkConnection, "_handle_handshake", _rehandshake_with_hellos)


@contextmanager
def _count_hellos() -> Iterator[Counter[str]]:
    """Count every server/hello the server sends and every client/hello it receives."""
    counts: Counter[str] = Counter()
    server_hello = SendspinConnection._server_hello  # noqa: SLF001
    deserialize = SendspinConnection._deserialize_client_message  # noqa: SLF001

    def counting_server_hello(self: SendspinConnection) -> Any:
        counts["server/hello"] += 1
        return server_hello(self)

    def counting_deserialize(_cls: type[SendspinConnection], raw: str) -> ClientMessage:
        message = deserialize(raw)
        if isinstance(message, ClientHelloMessage):
            counts["client/hello"] += 1
        return message

    with (
        patch.object(SendspinConnection, "_server_hello", counting_server_hello),
        patch.object(
            SendspinConnection, "_deserialize_client_message", classmethod(counting_deserialize)
        ),
    ):
        yield counts


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
        await _await_paired_session(pair_client)
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


@pytest.mark.parametrize(
    "first_text",
    [
        pytest.param(_legacy_hello(), id="legacy-hello"),
        pytest.param("this is not json", id="not-json"),
        pytest.param("[]", id="not-object"),
        pytest.param('{"type":"client/goodbye","payload":{}}', id="unknown-type"),
    ],
)
async def test_default_server_answers_non_init_first_frame_with_server_error(
    first_text: str,
) -> None:
    """Without transition mode, a TEXT first frame other than client/init gets malformed."""
    server = _make_server(InMemoryServerPairingStore())
    async with (
        _serve(server) as url,
        ClientSession() as session,
        session.ws_connect(url) as ws,
    ):
        await ws.send_str(first_text)
        msg = await asyncio.wait_for(ws.receive(), timeout=5)
        assert msg.type is WSMsgType.TEXT
        assert ServerErrorMessage.from_json(msg.data).payload.reason is ServerErrorReason.MALFORMED
        msg = await asyncio.wait_for(ws.receive(), timeout=5)
        assert msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED)


async def test_server_closes_silently_on_binary_first_frame() -> None:
    """A non-TEXT first frame closes the connection without a server/error."""
    server = _make_server(InMemoryServerPairingStore(), allow_unencrypted=True)
    async with (
        _serve(server) as url,
        ClientSession() as session,
        session.ws_connect(url) as ws,
    ):
        await ws.send_bytes(b"\x00")
        msg = await asyncio.wait_for(ws.receive(), timeout=5)
        assert msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED)


async def test_client_surfaces_server_error_as_init_rejected(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The SDK raises InitRejectedError with the reason, logs it, and closes the socket."""
    closed = asyncio.Event()

    async def handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.receive()  # client/init
        error = ServerErrorMessage(
            payload=ServerErrorPayload(reason=ServerErrorReason.UNSUPPORTED_VERSION)
        )
        await ws.send_str(error.to_json())
        msg = await ws.receive()
        assert msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED)
        closed.set()
        return ws

    app = web.Application()
    app.router.add_get("/sendspin", handler)
    test_server = TestServer(app)
    await test_server.start_server()
    client = make_sdk_client(client_name="c", roles=[Roles.CONTROLLER])
    try:
        with pytest.raises(InitRejectedError) as exc_info:
            await client.connect(f"ws://127.0.0.1:{test_server.port}/sendspin")
        assert exc_info.value.reason is ServerErrorReason.UNSUPPORTED_VERSION
        await asyncio.wait_for(closed.wait(), timeout=5)
        assert "unsupported_version" in caplog.text
        assert not client.connected
    finally:
        await client.disconnect()
        await test_server.close()


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


@pytest.mark.parametrize("allow_unencrypted", [True, False])
async def test_pairing_dial_refuses_legacy_client(allow_unencrypted: bool) -> None:  # noqa: FBT001
    """A pairing dial answered with a legacy hello closes without a reply."""
    server = _make_server(InMemoryServerPairingStore(), allow_unencrypted=allow_unencrypted)
    try:
        async with _serve_legacy_peer() as (url, closed, frames):
            async with ClientSession() as session, session.ws_connect(url) as wsock:
                conn = SendspinConnection(
                    server,
                    wsock_client=wsock,
                    url=url,
                    pairing_attempt=PairingAttempt(
                        method=PairMethod.PAIRING_PSK,
                        pairing_psk=generate_psk(),
                        client_id="legacy-client",
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
                PairingAttempt(
                    method=PairMethod.PAIRING_PSK,
                    pairing_psk=generate_psk(),
                    client_id="legacy-client",
                )
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

    async def display(pairing_code: str | None, **_kwargs: object) -> None:
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


async def _pair_via_spoken_dynamic_code(
    server: SendspinServer, *, player_support: ClientHelloPlayerSupport | None = None
) -> tuple[ActivatePairing, tuple[str, ...]]:
    """Pair a speaker-only client; return its pairing activation and the languages it spoke."""
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()

    loop = asyncio.get_running_loop()
    spoken: asyncio.Future[tuple[str, ...]] = loop.create_future()
    code: asyncio.Future[str] = loop.create_future()
    activation: asyncio.Future[ActivatePairing] = loop.create_future()

    async def speak(pairing_code: str | None, *, languages: tuple[str, ...]) -> None:
        if pairing_code is None or code.done():
            return
        spoken.set_result(languages)
        code.set_result(pairing_code)
        conn = client._admitted_connection  # noqa: SLF001 - assert on the received activation
        assert conn is not None
        assert conn._selected_pairing is not None  # noqa: SLF001
        activation.set_result(conn._selected_pairing)  # noqa: SLF001

    async def provide() -> str:
        return await code

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.PLAYER] if player_support is not None else [Roles.CONTROLLER],
            player_support=player_support,
            pairing_support=PairingSupport(pairing_code_speaker=speak),
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
        finally:
            await client.disconnect()
    return activation.result(), spoken.result()


@pytest.mark.parametrize("languages", [("ca", "es", "en"), None])
async def test_live_pairing_language_hint_rides_server_hello(
    languages: tuple[str, ...] | None,
) -> None:
    """The server's languages reach the speaker from server/hello, never the activation."""
    server = _make_server(InMemoryServerPairingStore(), languages=languages)
    activation, spoken = await _pair_via_spoken_dynamic_code(server)
    assert activation.languages is None
    assert spoken == (languages or ())


# DEPRECATED(spec-pr-241): remove in aiosendspin <version>
async def test_live_pairing_language_hint_on_activation_for_pre_spec_177_hello() -> None:
    """A client whose hello predates spec PR 177 also gets the languages on the activation."""
    server = _make_server(InMemoryServerPairingStore(), languages=("ca", "en"))
    build_client_hello = SdkConnection._build_client_hello  # noqa: SLF001

    async def pre_spec_177_hello(self: SdkConnection) -> ClientHelloMessage:
        hello = await build_client_hello(self)
        assert hello.payload.player_support is not None
        hello.payload.player_support.supported_commands = [PlayerCommand.VOLUME]
        return hello

    player_support = ClientHelloPlayerSupport(
        supported_formats=[
            SupportedAudioFormat(codec=AudioCodec.PCM, channels=2, sample_rate=44100, bit_depth=16)
        ],
        buffer_capacity=1_000_000,
    )
    with (
        patch.object(SdkConnection, "_build_client_hello", pre_spec_177_hello),
        _pre_spec_287_rehandshake(),
    ):
        activation, spoken = await _pair_via_spoken_dynamic_code(
            server, player_support=player_support
        )
    assert activation.languages == ["ca", "en"]
    assert spoken == ("ca", "en")


async def test_live_pairing_updates_connection_security_trust() -> None:
    """Pairing promotes the connection to the long-term PSK, and trust follows the category."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()

    shown: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def display(pairing_code: str | None, **_kwargs: object) -> None:
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

    async def display(pairing_code: str | None, **_kwargs: object) -> None:
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
            assert info.supported_pair_methods is not None
            assert info.supported_pair_methods.dynamic_pairing_code is None

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

    async def digits_display(pairing_code: str | None, **_kwargs: object) -> None:
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
            methods = conn._client_info.supported_pair_methods  # noqa: SLF001
            assert methods is not None
            descriptor = methods.dynamic_pairing_code
            assert descriptor is not None
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
    """A descriptor format from a newer spec revision is ignored; the known ones still pair.

    The parse filters unknown formats out, so the descriptor is mutated afterwards to reach
    the server's own selection check directly.
    """
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()

    shown: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def display(pairing_code: str | None, **_kwargs: object) -> None:
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
            methods = conn._client_info.supported_pair_methods  # noqa: SLF001
            assert methods is not None
            descriptor = methods.dynamic_pairing_code
            assert descriptor is not None
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


async def test_live_pairing_unusable_advertised_formats() -> None:
    """A format the client does not offer is refused before the attempt starts.

    As above, the descriptor is mutated past the parse to exercise the selection check itself.
    """
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
            pairing_support=PairingSupport(
                pairing_code_display=lambda _code, **_kwargs: asyncio.sleep(0)
            ),
        )
        try:
            await client.connect(url)
            conn = await _find_connection_by_client_id(server, client_identity.peer_id)
            assert conn._client_info is not None  # noqa: SLF001
            methods = conn._client_info.supported_pair_methods  # noqa: SLF001
            assert methods is not None
            descriptor = methods.dynamic_pairing_code
            assert descriptor is not None
            descriptor.formats = ["holographic"]

            with pytest.raises(PairingError, match="does not offer the digits"):
                await conn.initiate_pairing(
                    PairingAttempt(
                        method=PairMethod.DYNAMIC_PAIRING_CODE,
                        pairing_code_provider=lambda: asyncio.sleep(0),
                        pairing_format=PairingCodeFormat.DIGITS,
                    )
                )
        finally:
            await client.disconnect()


async def test_live_pairing_dropped_unusable_descriptor_is_refused() -> None:
    """A dynamic descriptor the parse dropped as unusable is not selected.

    The parse result is set directly, as the SDK client only advertises usable values.
    """
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
            pairing_support=PairingSupport(
                pairing_code_display=lambda _code, **_kwargs: asyncio.sleep(0)
            ),
        )
        try:
            await client.connect(url)
            conn = await _find_connection_by_client_id(server, client_identity.peer_id)
            assert conn._client_info is not None  # noqa: SLF001
            methods = conn._client_info.supported_pair_methods  # noqa: SLF001
            assert methods is not None
            methods.dynamic_pairing_code = None
            methods.unusable_methods = [PairMethod.DYNAMIC_PAIRING_CODE.value]

            with pytest.raises(PairingError, match="no usable dynamic_pairing_code"):
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
            pairing_support=PairingSupport(
                pairing_code_display=lambda _code, **_kwargs: asyncio.sleep(0)
            ),
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
            pairing_support=PairingSupport(
                pairing_code_display=lambda _code, **_kwargs: asyncio.sleep(0)
            ),
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

    async def display(pairing_code: str | None, **_kwargs: object) -> None:
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


async def _await_paired_session(client: SdkClient) -> None:
    """Wait for a pairing started on connect to land the session on its long-term PSK."""
    async with asyncio.timeout(5):
        while (  # noqa: ASYNC110
            client.noise_psk is None
            or client.noise_psk.category is not PskCategory.LONG_TERM
            or Activity.PAIRING in client.activities
        ):
            await asyncio.sleep(0.01)


async def test_live_pairing_dynamic_pairing_code_wrong_then_retry() -> None:
    """A wrong code fails a round; the next round of the same attempt pairs on the same code."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()
    shown: asyncio.Queue[str] = asyncio.Queue()
    shown_pins: list[str] = []

    async def display(pairing_code: str | None, **_kwargs: object) -> None:
        if pairing_code is not None:
            shown_pins.append(pairing_code)
            shown.put_nowait(pairing_code)

    entered: list[str] = []

    async def provide() -> str:
        correct = await shown.get()
        entered.append(correct if entered else ("000000" if correct != "000000" else "111111"))
        return entered[-1]

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
            assert len(entered) == 2
            assert len(shown_pins) == 2
            assert shown_pins[0] == shown_pins[1]  # the code is stable across rounds
            assert await client_store.pairing_round_count() == 0
        finally:
            await client.disconnect()


async def test_live_pairing_round_limit_holds_back_until_pairing_window() -> None:
    """Exhausting the rounds aborts the attempt; the next one waits for the operator action."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()
    shown: asyncio.Queue[str] = asyncio.Queue()

    async def display(pairing_code: str | None, **_kwargs: object) -> None:
        if pairing_code is not None:
            shown.put_nowait(pairing_code)

    async def wrong_code() -> str:
        correct = await shown.get()
        return "000000" if correct != "000000" else "111111"

    async def right_code() -> str:
        return await shown.get()

    window_opened = asyncio.get_running_loop().create_future()
    pending_signals = 0

    def on_pending(_message: str | None) -> None:
        nonlocal pending_signals
        pending_signals += 1

    async def gesture_prompt(active: bool) -> None:  # noqa: FBT001
        if active and not window_opened.done():
            window_opened.set_result(None)
            client.open_pairing_window()

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

            with pytest.raises(PairingAbortError) as excinfo:
                await conn.initiate_pairing(
                    PairingAttempt(
                        method=PairMethod.DYNAMIC_PAIRING_CODE,
                        pairing_code_provider=wrong_code,
                        pairing_format=PairingCodeFormat.DIGITS,
                    )
                )
            assert excinfo.value.reason is PairAbortReason.PAIRING_CODE_MISMATCH
            assert client.connected
            assert Activity.PAIRING in client.activities
            assert await client_store.pairing_round_count() == PAIRING_ROUND_LIMIT
            assert await client_store.is_pairing_round_limit_reached()
            assert not window_opened.done()

            await conn.initiate_pairing(
                PairingAttempt(
                    method=PairMethod.DYNAMIC_PAIRING_CODE,
                    pairing_code_provider=right_code,
                    on_pair_pending=on_pending,
                    pairing_format=PairingCodeFormat.DIGITS,
                )
            )
            await _await_long_term_record(client_store, server.id)
            assert window_opened.done()  # the attempt waited for the operator action
            assert pending_signals == 1  # the server surfaced the held-back attempt
            assert client.noise_psk is not None
            assert client.noise_psk.category is PskCategory.LONG_TERM
            assert await client_store.pairing_round_count() == 0
        finally:
            await client.disconnect()


async def test_live_pairing_invalid_operator_input_leaves_pairing() -> None:
    """Malformed operator input ends the attempt and leaves pairing, keeping the connection."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()

    async def display(pairing_code: str | None, **_kwargs: object) -> None:
        pass

    async def typo() -> str:
        return "12x456"

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
            await _find_connection_by_client_id(server, client_identity.peer_id)

            with pytest.raises(InvalidPairingCodeError):
                await server.initiate_pairing(
                    client_identity.peer_id,
                    PairingAttempt(
                        method=PairMethod.DYNAMIC_PAIRING_CODE,
                        pairing_code_provider=typo,
                        pairing_format=PairingCodeFormat.DIGITS,
                    ),
                )
            await _await_left_pairing(client)
            assert client.connected
            assert await _find_connection_by_client_id(server, client_identity.peer_id)
            assert await client_store.record_by_server_id(server.id) is None
        finally:
            await client.disconnect()


async def _code_pairing_client(
    identity: Identity,
    store: InMemoryClientPairingStore,
    method: PairMethod,
    shown: asyncio.Queue[str],
) -> SdkClient:
    """Build a client offering ``method``; shown codes land on ``shown``, windows open on ask."""
    if method is PairMethod.STATIC_PAIRING_CODE:
        await store.store_pairing_config(
            replace(await store.get_pairing_config(), static_pairing_code_enabled=True)
        )
        await store.set_static_pairing_code(_STATIC_PAIRING_CODE)
        shown.put_nowait(_STATIC_PAIRING_CODE)

    async def display(pairing_code: str | None, **_kwargs: object) -> None:
        if pairing_code is not None:
            shown.put_nowait(pairing_code)

    async def gesture_prompt(active: bool) -> None:  # noqa: FBT001
        if active:
            client.open_pairing_window()

    client = make_sdk_client(
        identity=identity,
        pairing_store=store,
        client_name="c",
        roles=[Roles.CONTROLLER],
        pairing_support=PairingSupport(
            gesture_prompt=gesture_prompt,
            pairing_code_display=display if method is PairMethod.DYNAMIC_PAIRING_CODE else None,
        ),
    )
    return client


def _code_attempt(
    method: PairMethod, provide: pairing_module.PairingCodeProvider
) -> PairingAttempt:
    return PairingAttempt(
        method=method,
        pairing_code_provider=provide,
        pairing_format=(
            PairingCodeFormat.DIGITS if method is PairMethod.DYNAMIC_PAIRING_CODE else None
        ),
    )


def _grouped(code: str, separator: str) -> str:
    half = len(code) // 2
    return f"{code[:half]}{separator}{code[half:]}"


_STATIC_PAIRING_CODE = "12345678"
_CODE_METHODS = (PairMethod.DYNAMIC_PAIRING_CODE, PairMethod.STATIC_PAIRING_CODE)


@pytest.mark.parametrize("separator", ["-", " "])
@pytest.mark.parametrize("method", _CODE_METHODS)
async def test_live_pairing_strips_separators_from_the_entered_code(
    method: PairMethod, separator: str
) -> None:
    """A grouped entry (``123-456``, ``1234 5678``) pairs like the contiguous code."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()
    shown: asyncio.Queue[str] = asyncio.Queue()

    async def provide() -> str:
        return _grouped(await shown.get(), separator)

    async with _serve(server) as url:
        client = await _code_pairing_client(client_identity, client_store, method, shown)
        try:
            await client.connect(url)
            await _find_connection_by_client_id(server, client_identity.peer_id)
            await server.initiate_pairing(client_identity.peer_id, _code_attempt(method, provide))
            await _await_long_term_record(client_store, server.id)
            assert await server_store.record_by_client_id(client_identity.peer_id) is not None
        finally:
            await client.disconnect()


async def test_live_static_pairing_wrong_code_keeps_the_connection() -> None:
    """A wrong static code surfaces as a ``pairing_code_mismatch`` abort, not a disconnect."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()
    method = PairMethod.STATIC_PAIRING_CODE

    async def wrong_code() -> str:
        return "8765-4321"

    async with _serve(server) as url:
        client = await _code_pairing_client(client_identity, client_store, method, asyncio.Queue())
        try:
            await client.connect(url)
            await _find_connection_by_client_id(server, client_identity.peer_id)
            with pytest.raises(PairingAbortError) as excinfo:
                await server.initiate_pairing(
                    client_identity.peer_id, _code_attempt(method, wrong_code)
                )
            assert excinfo.value.reason is PairAbortReason.PAIRING_CODE_MISMATCH
            assert client.connected
            assert await _find_connection_by_client_id(server, client_identity.peer_id)
            assert await client_store.record_by_server_id(server.id) is None
        finally:
            await client.disconnect()


@pytest.mark.parametrize("method", _CODE_METHODS)
async def test_dial_pairing_accepts_a_separated_code(method: PairMethod) -> None:
    """A pairing dial strips separators from the entered code."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()
    shown: asyncio.Queue[str] = asyncio.Queue()

    async def provide() -> str:
        return _grouped(await shown.get(), "-")

    sdk = await _code_pairing_client(client_identity, client_store, method, shown)
    try:
        async with (
            _host_incoming_client(sdk) as url,
            _dial(server, url, pairing_attempt=_code_attempt(method, provide)),
        ):
            await _await_paired_session(sdk)
            assert await server_store.record_by_client_id(client_identity.peer_id) is not None
    finally:
        await sdk.disconnect()
        await server.close()


@pytest.mark.parametrize(
    ("method", "entered"),
    [
        pytest.param(PairMethod.DYNAMIC_PAIRING_CODE, "12x456", id="dynamic_malformed"),
        pytest.param(PairMethod.STATIC_PAIRING_CODE, "1234567", id="static_malformed"),
        pytest.param(PairMethod.STATIC_PAIRING_CODE, "8765-4321", id="static_mismatch"),
    ],
)
async def test_dial_pairing_failed_entry_keeps_the_connection(
    method: PairMethod, entered: str
) -> None:
    """Malformed or wrong operator input on a pairing dial leaves pairing, still connected."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()
    shown: asyncio.Queue[str] = asyncio.Queue()
    asked = asyncio.Event()

    async def provide() -> str:
        await shown.get()
        asked.set()
        return entered

    sdk = await _code_pairing_client(client_identity, client_store, method, shown)
    try:
        async with (
            _host_incoming_client(sdk) as url,
            _dial(server, url, pairing_attempt=_code_attempt(method, provide)),
        ):
            async with asyncio.timeout(5):
                await asked.wait()
            await _await_left_pairing(sdk)
            assert sdk.connected
            server_client = await _await_connected_client(server, client_identity.peer_id)
            assert not server_client.is_paired
            assert await client_store.record_by_server_id(server.id) is None
    finally:
        await sdk.disconnect()
        await server.close()


async def test_pair_retry_in_flight_does_not_fail_the_next_attempt() -> None:
    """A retry sent before the client saw a leave does not fail an attempt started right after."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()
    shown: asyncio.Queue[str] = asyncio.Queue()

    async def display(pairing_code: str | None, **_kwargs: object) -> None:
        if pairing_code is not None:
            shown.put_nowait(pairing_code)

    async def provide() -> str:
        return await shown.get()

    run_client = client_connection_module.run_dynamic_pairing_code_client

    async def client_with_retry_in_flight(ws: EncryptedWebSocket, **kwargs: Any) -> str | None:
        # The previous attempt's retry reaches the server after its next pairing activate.
        await ws.send_str(ClientPairRetryMessage().to_json())
        return await run_client(ws, **kwargs)

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
            with patch.object(
                client_connection_module,
                "run_dynamic_pairing_code_client",
                client_with_retry_in_flight,
            ):
                await conn.initiate_pairing(
                    PairingAttempt(
                        method=PairMethod.DYNAMIC_PAIRING_CODE,
                        pairing_code_provider=provide,
                        pairing_format=PairingCodeFormat.DIGITS,
                    )
                )
            await _await_long_term_record(client_store, server.id)
            assert client.connected
            assert client.noise_psk is not None
            assert client.noise_psk.category is PskCategory.LONG_TERM
        finally:
            await client.disconnect()


# DEPRECATED(spec-pr-237): remove in aiosendspin <version>
async def _send_list_form_hello(self: SdkConnection) -> None:
    """Send client/hello with supported_pair_methods in the superseded list form."""
    assert self._ws is not None
    hello = (await self._build_client_hello()).to_dict()
    methods = hello["payload"]["supported_pair_methods"]
    hello["payload"]["supported_pair_methods"] = [
        {"method": method, **descriptor} for method, descriptor in methods.items()
    ]
    await self._ws.send_str(json.dumps(hello))


# DEPRECATED(spec-pr-237): remove in aiosendspin <version>
async def test_list_form_client_pairs_under_the_pre_round_sid(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A client predating rounds pairs with the pre-round sid and is flagged as non-compliant."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()
    shown: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def display(pairing_code: str | None, **_kwargs: object) -> None:
        if pairing_code is not None and not shown.done():
            shown.set_result(pairing_code)

    async def provide() -> str:
        return await shown

    legacy_pake_sid = pairing_module._legacy_pake_sid  # noqa: SLF001
    server_legacy_sids = 0

    def client_sid(handshake_hash: bytes, pairing_index: int, _round_number: int) -> bytes:
        return legacy_pake_sid(handshake_hash, pairing_index)

    def server_sid(handshake_hash: bytes, pairing_index: int) -> bytes:
        nonlocal server_legacy_sids
        server_legacy_sids += 1
        return legacy_pake_sid(handshake_hash, pairing_index)

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
            pairing_support=PairingSupport(pairing_code_display=display),
        )
        try:
            with (
                patch.object(SdkConnection, "_send_client_hello", _send_list_form_hello),
                _pre_spec_287_rehandshake(),
                patch.object(pairing_module, "_pake_sid", client_sid),
                patch.object(pairing_module, "_legacy_pake_sid", server_sid),
            ):
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
            assert server_legacy_sids == 1
            assert client.noise_psk is not None
            assert client.noise_psk.category is PskCategory.LONG_TERM
        finally:
            await client.disconnect()
    assert (
        "non-compliant client c: client/hello sent supported_pair_methods as a list"
        in caplog.messages
    )


# DEPRECATED(spec-pr-237): remove in aiosendspin <version>
async def test_strict_server_rejects_list_form_client(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A strict server rejects a client predating rounds before any pairing."""
    server = _make_server(InMemoryServerPairingStore(), allow_noncompliant_clients=False)
    client_identity = Identity.generate()

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=client_identity,
            pairing_store=InMemoryClientPairingStore(),
            client_name="c",
            roles=[Roles.CONTROLLER],
        )
        try:
            with (
                patch.object(SdkConnection, "_send_client_hello", _send_list_form_hello),
                suppress(Exception),
            ):
                await client.connect(url)
            async with asyncio.timeout(5):
                while server._pending_connections:  # noqa: SLF001, ASYNC110
                    await asyncio.sleep(0.01)
            server_client = server.get_client(client_identity.peer_id)
            assert server_client is None or not server_client.is_connected
        finally:
            await client.disconnect()
    assert (
        "rejecting non-compliant client c: client/hello sent supported_pair_methods as a list"
        in caplog.messages
    )


async def test_pair_retry_after_leaving_pairing_is_discarded() -> None:
    """A client/pair-retry still in flight when pairing ends is discarded, not fatal."""
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
            connection = client._admitted_connection  # noqa: SLF001
            assert connection is not None
            await connection._send_message(  # noqa: SLF001
                ClientPairRetryMessage().to_json(), force=True
            )
            await asyncio.sleep(0.1)  # a fatal frame would have torn the connection down
            assert client.connected
        finally:
            await client.disconnect()


async def test_pairing_frames_between_attempts_are_discarded() -> None:
    """After a non-closing abort nothing queues pairing frames until the next attempt."""
    server = _make_server(InMemoryServerPairingStore())
    identity = Identity.generate()
    client_store = InMemoryClientPairingStore()
    config = await client_store.get_pairing_config()
    # Offers no dynamic pairing code, so the attempt aborts with method_not_supported.
    await client_store.store_pairing_config(replace(config, dynamic_pairing_code_enabled=False))

    async def provide() -> str:
        return "000000"

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
        )
        try:
            await client.connect(url)
            conn = await _find_connection_by_client_id(server, identity.peer_id)
            with pytest.raises(PairingAbortError):
                await conn.initiate_pairing(
                    PairingAttempt(
                        method=PairMethod.DYNAMIC_PAIRING_CODE,
                        pairing_code_provider=provide,
                        pairing_format=PairingCodeFormat.DIGITS,
                    )
                )
            assert conn._in_pairing  # noqa: SLF001
            assert conn._pairing_message_queue is None  # noqa: SLF001
            routed = _track_routed_types(conn)

            sdk_conn = client._admitted_connection  # noqa: SLF001
            assert sdk_conn is not None
            await sdk_conn._send_message(ClientPairRetryMessage().to_json())  # noqa: SLF001
            replies = sdk_conn._time_filter.count  # noqa: SLF001
            await sdk_conn._send_time_message()  # noqa: SLF001
            await _wait_until(lambda: sdk_conn._time_filter.count > replies)  # noqa: SLF001

            assert routed == []
            assert conn._pairing_message_queue is None  # noqa: SLF001
            assert client.connected
            await conn.end_pairing()
            await _await_left_pairing(client)
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

    async def display(pairing_code: str | None, **_kwargs: object) -> None:
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

    async def display(pairing_code: str | None, **_kwargs: object) -> None:
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

    async def display(pairing_code: str | None, **_kwargs: object) -> None:
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

    async def display(pairing_code: str | None, **_kwargs: object) -> None:
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


async def test_success_rehandshake_handles_client_messages_sent_before_message_1() -> None:
    """Client messages in flight when the success re-handshake starts do not fail the pairing."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()

    async with _serve(server) as url:
        client, attempt, release = await _paired_client_with_stalled_success_tail(
            server, url, client_identity, client_store
        )
        try:
            conn = await _find_connection_by_client_id(server, client_identity.peer_id)
            queue = conn._pairing_message_queue  # noqa: SLF001
            assert queue is not None
            client_ws = client._admitted_connection._ws  # noqa: SLF001
            assert client_ws is not None
            queued = queue.qsize()
            handled: list[bytes] = []
            route_binary = conn._route_inbound_binary  # noqa: SLF001

            def tracking_route_binary(data: bytes) -> None:
                handled.append(data)
                route_binary(data)

            conn._route_inbound_binary = tracking_route_binary  # type: ignore[method-assign]  # noqa: SLF001
            await client_ws.send_str(
                ClientStateMessage(payload=ClientStatePayload(available=True)).to_json()
            )
            await client_ws.send_bytes(b"\x04audio")
            # The reader handles them in place; they never reach the stalled pairing task.
            await _wait_until(lambda: bool(handled))
            assert queue.qsize() == queued
            release.set()

            await attempt
            await _await_long_term_record(client_store, server.id)
            await _await_left_pairing(client)
            assert client.connected
            assert client.noise_psk is not None
            assert client.noise_psk.category is PskCategory.LONG_TERM
            assert conn.psk_category is PskCategory.LONG_TERM
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
                PairingAttempt(
                    method=PairMethod.PAIRING_PSK,
                    pairing_psk=pairing,
                    client_id=client_identity.peer_id,
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
        finally:
            await client.disconnect()


# DEPRECATED(spec-pr-247): remove in aiosendspin <version>
async def _legacy_pairing_psk_client(
    ws: EncryptedWebSocket,
    *,
    pairing_index: int,  # noqa: ARG001
    server_id: str,
    store: ClientPairingStore,
    on_finalize: Callable[[], None] | None = None,
    protected_psk_ids: Callable[[], AbstractSet[str]] = frozenset,
) -> None:
    """Pairing PSK client that goes straight to client/pair-finalize."""
    await pairing_module._finalize_client(  # noqa: SLF001
        ws,
        server_id=server_id,
        store=store,
        on_finalize=on_finalize,
        protected_psk_ids=protected_psk_ids,
    )


async def _staged_pairing_psk_stores(
    client_identity: Identity,
) -> tuple[InMemoryServerPairingStore, InMemoryClientPairingStore]:
    server_store = InMemoryServerPairingStore()
    client_store = InMemoryClientPairingStore()
    pairing = generate_psk()
    psk_id = psk_id_for(pairing)
    await client_store.set_pairing_psk(PairingPsk(psk_id=psk_id, psk=pairing))
    await server_store.stage_pairing_psk(
        client_identity.peer_id, StagedPairingPsk(psk_id=psk_id, psk=pairing)
    )
    return server_store, client_store


# DEPRECATED(spec-pr-247): remove in aiosendspin <version>
async def test_legacy_pairing_psk_client_pairs_and_is_flagged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A finalize-first Pairing PSK client still pairs, flagged as non-compliant."""
    client_identity = Identity.generate()
    server_store, client_store = await _staged_pairing_psk_stores(client_identity)
    server = _make_server(server_store)

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
        )
        try:
            with patch.object(
                client_connection_module, "run_pairing_psk_client", _legacy_pairing_psk_client
            ):
                await client.connect(url)
                await _await_paired_session(client)
            assert client.noise_psk is not None
            assert client.noise_psk.category is PskCategory.LONG_TERM
            client_record = await client_store.record_by_server_id(server.id)
            server_record = await server_store.record_by_client_id(client_identity.peer_id)
            assert client_record is not None
            assert server_record is not None
            assert client_record.psk == server_record.psk
        finally:
            await client.disconnect()
    assert (
        "non-compliant client c: Pairing PSK client/pair-finalize sent without client/pair-init"
        in caplog.messages
    )


# DEPRECATED(spec-pr-247): remove in aiosendspin <version>
async def test_strict_server_rejects_legacy_pairing_psk_client_on_connect(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A strict server drops a finalize-first client for good, persisting nothing."""
    client_identity = Identity.generate()
    server_store, client_store = await _staged_pairing_psk_stores(client_identity)
    server = _make_server(server_store, allow_noncompliant_clients=False)

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=client_identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
        )
        try:
            with (
                patch.object(
                    client_connection_module,
                    "run_pairing_psk_client",
                    _legacy_pairing_psk_client,
                ),
                patch.object(
                    SendspinConnection,
                    "disconnect",
                    autospec=True,
                    side_effect=SendspinConnection.disconnect,
                ) as disconnect,
            ):
                await client.connect(url)
                async with asyncio.timeout(5):
                    while server._pending_connections or client.connected:  # noqa: SLF001, ASYNC110
                        await asyncio.sleep(0.01)
            assert disconnect.await_args_list[0].kwargs == {"retry_connection": False}
            assert await server_store.record_by_client_id(client_identity.peer_id) is None
            assert (
                "rejecting non-compliant client c: "
                "Pairing PSK client/pair-finalize sent without client/pair-init"
            ) in caplog.messages
        finally:
            await client.disconnect()


# DEPRECATED(spec-pr-247): remove in aiosendspin <version>
async def test_strict_server_rejects_legacy_pairing_psk_client_live() -> None:
    """A strict server's live Pairing PSK attempt rejects a finalize-first client."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store, allow_noncompliant_clients=False)
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
            await _find_connection_by_client_id(server, client_identity.peer_id)
            with (
                patch.object(
                    client_connection_module,
                    "run_pairing_psk_client",
                    _legacy_pairing_psk_client,
                ),
                pytest.raises(ClientComplianceError),
            ):
                await server.initiate_pairing(
                    client_identity.peer_id,
                    PairingAttempt(
                        method=PairMethod.PAIRING_PSK,
                        pairing_psk=pairing,
                        client_id=client_identity.peer_id,
                    ),
                )
            assert await server_store.record_by_client_id(client_identity.peer_id) is None
            async with asyncio.timeout(5):
                while server._pending_connections:  # noqa: SLF001, ASYNC110
                    await asyncio.sleep(0.01)
        finally:
            await client.disconnect()


# DEPRECATED(spec-pr-247): remove in aiosendspin <version>
async def test_finalize_first_is_discarded_after_a_pairing_psk_pair_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once a connection has sent a Pairing PSK pair-init, a leading finalize is a leftover."""
    monkeypatch.setattr(pairing_module, "SERVER_FIRST_MESSAGE_TIMEOUT_S", 0.2)
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()
    pairing = generate_psk()
    await client_store.set_pairing_psk(PairingPsk(psk_id=psk_id_for(pairing), psk=pairing))
    attempt = PairingAttempt(
        method=PairMethod.PAIRING_PSK, pairing_psk=pairing, client_id=client_identity.peer_id
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
            conn = await _find_connection_by_client_id(server, client_identity.peer_id)
            await conn.initiate_pairing(attempt)
            first = await server_store.record_by_client_id(client_identity.peer_id)
            assert first is not None

            with (
                patch.object(
                    client_connection_module,
                    "run_pairing_psk_client",
                    _legacy_pairing_psk_client,
                ),
                pytest.raises(PairingTimeoutError, match="client/pair-init"),
            ):
                await conn.initiate_pairing(attempt)
            assert await server_store.record_by_client_id(client_identity.peer_id) == first
        finally:
            await client.disconnect()


def _track_routed_types(conn: SendspinConnection) -> list[str]:
    """Record the type of each message ``conn`` routes to its pairing attempt from now on."""
    routed: list[str] = []
    route = conn._try_route_to_pairing_queue  # noqa: SLF001

    def tracking_route(msg: Any) -> bool:
        if routed_now := route(msg):
            routed.append(json.loads(msg.data)["type"])
        return routed_now

    conn._try_route_to_pairing_queue = tracking_route  # type: ignore[method-assign]  # noqa: SLF001
    return routed


async def _wait_until(predicate: Callable[[], bool]) -> None:
    async with asyncio.timeout(5):
        while not predicate():  # noqa: ASYNC110
            await asyncio.sleep(0.01)


# DEPRECATED(spec-pr-247): remove in aiosendspin <version>
class _AbandonedPairingPskClient:
    """Pairing PSK client exchange whose first attempt's messages arrive after it is abandoned.

    The first attempt sends ``client/pair-init`` once ``release_init`` is set, and a finalize-only
    ``client/pair-finalize`` on ``send_finalize()``. Every attempt waits to be cancelled.
    """

    def __init__(self) -> None:
        self.release_init = asyncio.Event()
        self._release_finalize = asyncio.Event()
        self._stale: asyncio.Task[None] | None = None

    async def run(
        self,
        ws: EncryptedWebSocket,
        *,
        pairing_index: int,
        server_id: str,  # noqa: ARG002
        store: ClientPairingStore,  # noqa: ARG002
        on_finalize: Callable[[], None] | None = None,  # noqa: ARG002
        protected_psk_ids: Callable[[], AbstractSet[str]] = frozenset,  # noqa: ARG002
    ) -> None:
        if self._stale is None:
            self._stale = asyncio.create_task(self._send_stale(ws, pairing_index))
        await asyncio.Event().wait()

    async def send_finalize(self) -> None:
        assert self._stale is not None
        self._release_finalize.set()
        await self._stale

    async def _send_stale(self, ws: EncryptedWebSocket, pairing_index: int) -> None:
        await self.release_init.wait()
        await ws.send_str(
            ClientPairInitMessage(
                payload=ClientPairInitPayload(pairing_index=pairing_index)
            ).to_json()
        )
        await self._release_finalize.wait()
        await ws.send_str(
            ClientPairFinalizeMessage(
                payload=ClientPairFinalizePayload(long_term_psk=b64url_encode(generate_psk()))
            ).to_json()
        )


# DEPRECATED(spec-pr-247): remove in aiosendspin <version>
@pytest.mark.parametrize("init_after_cancel", [False, True])
async def test_unconsumed_pair_init_of_a_cancelled_attempt_blocks_the_legacy_fallback(
    init_after_cancel: bool,  # noqa: FBT001
) -> None:
    """A cancelled attempt's unconsumed pair-init, queued or late, marks the client as new-flow.

    Its late finalize, reaching the next attempt, is discarded instead of persisting a stale PSK.
    """
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()
    pairing = generate_psk()
    await client_store.set_pairing_psk(PairingPsk(psk_id=psk_id_for(pairing), psk=pairing))
    attempt = PairingAttempt(
        method=PairMethod.PAIRING_PSK, pairing_psk=pairing, client_id=client_identity.peer_id
    )
    client_exchange = _AbandonedPairingPskClient()
    release_init = client_exchange.release_init
    second_attempt_started = asyncio.Event()
    real_server_exchange = connection_module.run_pairing_psk_server
    server_calls = 0

    async def server_exchange(ws: EncryptedWebSocket, **kwargs: Any) -> ServerPairingRecord:
        nonlocal server_calls
        server_calls += 1
        if server_calls == 1:
            await asyncio.Event().wait()  # never consumes the first attempt's messages
        second_attempt_started.set()
        return await real_server_exchange(ws, **kwargs)

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
            with (
                patch.object(
                    client_connection_module, "run_pairing_psk_client", client_exchange.run
                ),
                patch.object(connection_module, "run_pairing_psk_server", server_exchange),
            ):
                first = asyncio.create_task(conn.initiate_pairing(attempt))
                await _wait_until(lambda: server_calls > 0)
                if not init_after_cancel:
                    release_init.set()
                    await _wait_until(
                        lambda: not conn._pairing_message_queue.empty()  # noqa: SLF001
                    )
                await conn.end_pairing()
                with pytest.raises(PairingAbortError):
                    await first
                if init_after_cancel:
                    release_init.set()
                    await _wait_until(lambda: conn._sent_psk_pair_init)  # noqa: SLF001

                second = asyncio.create_task(conn.initiate_pairing(attempt))
                await asyncio.wait_for(second_attempt_started.wait(), timeout=5)
                queue = conn._pairing_message_queue  # noqa: SLF001
                assert queue is not None
                routed = _track_routed_types(conn)
                await client_exchange.send_finalize()
                # The next attempt consumes the late finalize and discards it.
                await _wait_until(lambda: "client/pair-finalize" in routed and queue.empty())
                await conn.end_pairing()
                with pytest.raises(PairingAbortError):
                    await second
            assert await server_store.record_by_client_id(client_identity.peer_id) is None
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
            await _await_paired_session(client)
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

    async def display(pairing_code: str | None, **_kwargs: object) -> None:
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
    await client_store.record_pairing_round()  # a dynamic-pairing-code round; static ignores it

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
            # The static flow leaves the dynamic-pairing-code round count alone.
            assert await client_store.pairing_round_count() == 1
        finally:
            await client.disconnect()


async def test_live_pairing_keeps_the_writer_running_during_exchange() -> None:
    """Without a re-handshake the writer keeps running, so client/time is answered mid-attempt."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()

    loop = asyncio.get_running_loop()
    shown: asyncio.Future[str] = loop.create_future()
    during: dict[str, object] = {}

    async def display(pairing_code: str | None, **_kwargs: object) -> None:
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
            sdk_conn = client._admitted_connection  # noqa: SLF001
            assert sdk_conn is not None
            time_replies = 0
            handle_server_time = sdk_conn._handle_server_time  # noqa: SLF001

            async def counting_handle_server_time(payload: Any) -> None:
                nonlocal time_replies
                time_replies += 1
                await handle_server_time(payload)

            sdk_conn._handle_server_time = counting_handle_server_time  # type: ignore[method-assign]  # noqa: SLF001

            async def provide() -> str:
                # Mid-exchange: server/pair-init is out and the server awaits the pairing code.
                during["writer_running"] = conn._writer_task is not None  # noqa: SLF001
                # A refresh that changes nothing leaves the attempt alone, and an approval
                # granted now waits for the end of pairing instead of cancelling it.
                await conn.refresh_trusted_unpaired()
                await server.trust_unpaired(client_identity.peer_id)
                replies = time_replies
                await sdk_conn._send_time_message()  # noqa: SLF001
                await _wait_until(lambda: time_replies > replies)
                return await shown

            await conn.initiate_pairing(
                PairingAttempt(
                    method=PairMethod.DYNAMIC_PAIRING_CODE,
                    pairing_code_provider=provide,
                    pairing_format=PairingCodeFormat.DIGITS,
                )
            )

            assert during["writer_running"] is True
            await _await_paired_session(client)
            assert conn._writer_task is not None  # noqa: SLF001
        finally:
            await client.disconnect()


async def test_live_pairing_psk_pauses_writer_across_rehandshakes() -> None:
    """Pairing-PSK live pairing re-handshakes twice (Sentinel→Pairing→long-term).

    The writer is paused for each re-handshake, so it cannot interleave with either one,
    and runs again for the exchange between them.
    """
    writer_running: list[bool] = []
    conn_holder: list[SendspinConnection] = []
    real_rehandshake = connection_module.run_rehandshake_server

    async def observing_rehandshake(*args: Any, **kwargs: Any) -> Any:
        writer_running.append(conn_holder[0]._writer_task is not None)  # noqa: SLF001
        return await real_rehandshake(*args, **kwargs)

    class _ObservingStore(InMemoryServerPairingStore):
        async def store_record(self, record: ServerPairingRecord) -> None:
            if conn_holder:
                writer_running.append(conn_holder[0]._writer_task is not None)  # noqa: SLF001
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
            with patch.object(connection_module, "run_rehandshake_server", observing_rehandshake):
                await conn.initiate_pairing(
                    PairingAttempt(
                        method=PairMethod.PAIRING_PSK,
                        pairing_psk=pairing,
                        client_id=client_identity.peer_id,
                    )
                )
            # Paused for the first re-handshake, running for the exchange, paused for the last.
            assert writer_running == [False, True, False]
            assert conn._writer_task is not None  # noqa: SLF001  # resumed after the exchange
            await _await_paired_session(client)
        finally:
            await client.disconnect()


def _player_support() -> ClientHelloPlayerSupport:
    return ClientHelloPlayerSupport(
        supported_formats=[
            SupportedAudioFormat(codec=AudioCodec.PCM, channels=2, sample_rate=48000, bit_depth=16)
        ],
        buffer_capacity=1_000_000,
    )


async def _pair_while_playing(
    server: SendspinServer,
    client: SdkClient,
    attempt: Callable[[Callable[[], Any]], PairingAttempt],
    code: asyncio.Future[str],
) -> dict[str, object]:
    """Start a stream on the client's group, run a pairing attempt, and snapshot mid-attempt.

    ``attempt`` builds the attempt from a pairing code provider that takes the snapshot.
    """
    sdk_conn = client._admitted_connection  # noqa: SLF001
    assert sdk_conn is not None
    assert sdk_conn.server_id is not None
    activations: list[ServerActivatePayload] = []
    apply_activation = sdk_conn._apply_activation  # noqa: SLF001

    async def recording_apply_activation(payload: ServerActivatePayload) -> Any:
        activations.append(payload)
        return await apply_activation(payload)

    sdk_conn._apply_activation = recording_apply_activation  # type: ignore[method-assign]  # noqa: SLF001
    conn = await _find_connection_by_client_id(server, client.identity.peer_id)
    server_client = conn._client  # noqa: SLF001
    assert server_client is not None
    group = server_client.group
    # An unsynchronized player reports itself unavailable, which would stop its group.
    await _wait_until(client.is_time_synchronized)
    await _wait_until(lambda: server_client.available)
    group.start_stream()
    await _wait_until(lambda: client.activities == [Activity.PLAYBACK])
    during: dict[str, object] = {}

    async def snapshot() -> None:
        await _wait_until(lambda: Activity.PAIRING in client.activities)
        during["activities"] = client.activities
        during["client_roles"] = list(sdk_conn._active_roles)  # noqa: SLF001
        during["server_roles"] = list(server_client.active_role_ids)
        during["same_group"] = server_client.group is group
        during["stream"] = group.has_active_stream

    async def provide() -> str:
        await snapshot()
        return await code

    await conn.initiate_pairing(attempt(provide))
    await _await_paired_session(client)
    during["pairing_active_roles"] = [
        payload.active_roles for payload in activations if payload.pairing is not None
    ]
    during["after_roles"] = list(server_client.active_role_ids)
    during["after_stream"] = group.has_active_stream and server_client.group is group
    return during


async def test_live_pairing_runs_alongside_playback() -> None:
    """A playing unpaired client pairs without leaving its group, stream or roles."""
    server = _make_server(InMemoryServerPairingStore())
    identity = Identity.generate()
    await server.trust_unpaired(identity.peer_id)
    code: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    suspended: list[bool] = []

    async def display(pairing_code: str | None, **_kwargs: object) -> None:
        if pairing_code is not None and not code.done():
            code.set_result(pairing_code)

    async def suspend(active: bool) -> None:  # noqa: FBT001
        suspended.append(active)

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=identity,
            pairing_store=await _unpaired_enabled_store(),
            client_name="c",
            roles=[Roles.PLAYER],
            player_support=_player_support(),
            pairing_support=PairingSupport(
                pairing_code_display=display, out_channel_suspend=suspend
            ),
        )
        try:
            await client.connect(url)
            during = await _pair_while_playing(
                server,
                client,
                lambda provide: PairingAttempt(
                    method=PairMethod.DYNAMIC_PAIRING_CODE,
                    pairing_code_provider=provide,
                    pairing_format=PairingCodeFormat.DIGITS,
                ),
                code,
            )
            assert during["activities"] == [Activity.PLAYBACK, Activity.PAIRING]
            assert during["pairing_active_roles"] == [None]
            assert during["client_roles"] == ["player@v1"]
            assert during["server_roles"] == ["player@v1"]
            assert during["same_group"] is True
            assert during["stream"] is True
            assert during["after_roles"] == ["player@v1"]
            assert during["after_stream"] is True
            assert client.activities == [Activity.PLAYBACK]
            assert suspended == [True, False]
        finally:
            await client.disconnect()


async def test_revoking_approval_mid_attempt_ends_pairing() -> None:
    """A revoked unpaired approval ends the attempt and withdraws playback and roles."""
    server = _make_server(InMemoryServerPairingStore())
    identity = Identity.generate()
    await server.trust_unpaired(identity.peer_id)
    waiting = asyncio.Event()

    async def display(_pairing_code: str | None, **_kwargs: object) -> None:
        return

    async def provide() -> str:
        waiting.set()
        await asyncio.Event().wait()
        return ""

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=identity,
            pairing_store=await _unpaired_enabled_store(),
            client_name="c",
            roles=[Roles.CONTROLLER],
            pairing_support=PairingSupport(pairing_code_display=display),
        )
        try:
            await client.connect(url)
            conn = await _find_connection_by_client_id(server, identity.peer_id)
            assert _server_active_role_count(server, identity.peer_id) == 1
            attempt = asyncio.create_task(
                conn.initiate_pairing(
                    PairingAttempt(
                        method=PairMethod.DYNAMIC_PAIRING_CODE,
                        pairing_code_provider=provide,
                        pairing_format=PairingCodeFormat.DIGITS,
                    )
                )
            )
            await asyncio.wait_for(waiting.wait(), timeout=5)

            await server.untrust_unpaired(identity.peer_id)

            with pytest.raises(PairingAbortError):
                await attempt
            assert _server_active_role_count(server, identity.peer_id) == 0
            await _await_left_pairing(client)
            assert client.connected
            assert client._admitted_connection._active_roles == []  # noqa: SLF001
        finally:
            await client.disconnect()


# DEPRECATED(spec-pr-272): remove in aiosendspin <version>
async def test_live_pairing_quiesces_a_legacy_generation_client() -> None:
    """A client on the previous wire generation leaves playback and its roles for pairing."""
    server = _make_server(InMemoryServerPairingStore())
    identity = Identity.generate()
    await server.trust_unpaired(identity.peer_id)
    code: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    build_client_hello = SdkConnection._build_client_hello  # noqa: SLF001

    async def pre_spec_177_hello(self: SdkConnection) -> ClientHelloMessage:
        hello = await build_client_hello(self)
        assert hello.payload.player_support is not None
        hello.payload.player_support.supported_commands = [PlayerCommand.VOLUME]
        return hello

    async def display(pairing_code: str | None, **_kwargs: object) -> None:
        if pairing_code is not None and not code.done():
            code.set_result(pairing_code)

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=identity,
            pairing_store=await _unpaired_enabled_store(),
            client_name="c",
            roles=[Roles.PLAYER],
            player_support=_player_support(),
            pairing_support=PairingSupport(pairing_code_display=display),
        )
        try:
            with (
                patch.object(SdkConnection, "_build_client_hello", pre_spec_177_hello),
                _pre_spec_287_rehandshake(),
            ):
                await client.connect(url)
                during = await _pair_while_playing(
                    server,
                    client,
                    lambda provide: PairingAttempt(
                        method=PairMethod.DYNAMIC_PAIRING_CODE,
                        pairing_code_provider=provide,
                        pairing_format=PairingCodeFormat.DIGITS,
                    ),
                    code,
                )
            assert during["activities"] == [Activity.PAIRING]
            assert during["pairing_active_roles"] == [[]]
            assert during["client_roles"] == []
            assert during["server_roles"] == []
            assert during["stream"] is False
            assert during["after_roles"] == ["player@v1"]
        finally:
            await client.disconnect()


async def test_pairing_on_a_long_term_session_quiesces_first() -> None:
    """A long-term session re-keyed onto the pairing PSK leaves playback before pairing.

    The client admits unpaired access, so only an explicit empty role set clears its roles.
    """
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    identity = Identity.generate()
    client_store = await _unpaired_enabled_store()
    long_term = generate_psk()
    long_term_id = psk_id_for(long_term)
    await server_store.store_record(
        ServerPairingRecord(
            psk_id=long_term_id, psk=long_term, client_id=identity.peer_id, pair_methods=[]
        )
    )
    await client_store.store_record(
        ClientPairingRecord(psk_id=long_term_id, psk=long_term, server_id=server.id)
    )
    pairing = generate_psk()
    await client_store.set_pairing_psk(PairingPsk(psk_id=psk_id_for(pairing), psk=pairing))
    code: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    code.set_result("")
    real_exchange = connection_module.run_pairing_psk_server

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.PLAYER],
            player_support=_player_support(),
        )
        try:
            await client.connect(url)
            snapshots: list[Callable[[], Any]] = []

            async def observe_then_run(*args: Any, **kwargs: Any) -> ServerPairingRecord:
                await snapshots[0]()
                return await real_exchange(*args, **kwargs)  # type: ignore[no-any-return]

            def attempt(provide: Callable[[], Any]) -> PairingAttempt:
                snapshots.append(provide)
                return PairingAttempt(
                    method=PairMethod.PAIRING_PSK, pairing_psk=pairing, client_id=identity.peer_id
                )

            with patch.object(connection_module, "run_pairing_psk_server", observe_then_run):
                during = await _pair_while_playing(server, client, attempt, code)
            assert during["activities"] == [Activity.PAIRING]
            assert during["pairing_active_roles"] == [[]]
            assert during["client_roles"] == []
            assert during["server_roles"] == []
            assert during["stream"] is False
            assert during["after_roles"] == ["player@v1"]
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

    async def display(pairing_code: str | None, **_kwargs: object) -> None:
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
    # A pre-existing dynamic-pairing-code round count is reset by a successful re-verification.
    await client_store.record_pairing_round()

    shown: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def display(pairing_code: str | None, **_kwargs: object) -> None:
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
            # server_kc verified, so the round count resets to zero.
            assert await client_store.pairing_round_count() == 0
        finally:
            await client.disconnect()


async def test_reverification_at_round_limit_is_held_back() -> None:
    """Re-verification at the round limit waits for the pairing window, then resets the count."""
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
    for _ in range(PAIRING_ROUND_LIMIT):
        await client_store.record_pairing_round()
    assert await client_store.is_pairing_round_limit_reached()

    window_opened = asyncio.get_running_loop().create_future()

    async def gesture_prompt(active: bool) -> None:  # noqa: FBT001
        if active and not window_opened.done():
            window_opened.set_result(None)
            client.open_pairing_window()

    shown: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def display(pairing_code: str | None, **_kwargs: object) -> None:
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
            assert await client_store.pairing_round_count() == 0
        finally:
            await client.disconnect()


async def test_initiate_pairing_refuses_a_pairing_psk_token_for_another_client() -> None:
    """A Pairing PSK token naming another client raises before pairing, keeping the connection."""
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
            with pytest.raises(InvalidPairingCodeError, match="another client"):
                await server.initiate_pairing(
                    client_identity.peer_id,
                    PairingAttempt(
                        method=PairMethod.PAIRING_PSK,
                        pairing_psk=pairing,
                        client_id=Identity.generate().peer_id,
                    ),
                )
            assert not conn._in_pairing  # noqa: SLF001
            assert client.connected
            assert Activity.PAIRING not in client.activities
            assert client.noise_psk is not None
            assert client.noise_psk.category is PskCategory.SENTINEL
            assert await server_store.record_by_client_id(client_identity.peer_id) is None
        finally:
            await client.disconnect()


@pytest.mark.parametrize("already_paired", [False, True])
async def test_pairing_psk_dial_refuses_a_token_for_another_client(
    already_paired: bool,  # noqa: FBT001
) -> None:
    """A Pairing PSK dial reaching a client other than the token's aborts the handshake."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    identity = Identity.generate()
    client_store = InMemoryClientPairingStore()
    pairing = generate_psk()
    await client_store.set_pairing_psk(PairingPsk(psk_id=psk_id_for(pairing), psk=pairing))
    if already_paired:
        await _seed_long_term(server, server_store, client_store, identity.peer_id)
    records_before = await server_store.list_records()
    attempt = PairingAttempt(
        method=PairMethod.PAIRING_PSK, pairing_psk=pairing, client_id=Identity.generate().peer_id
    )

    sdk = make_sdk_client(
        identity=identity, pairing_store=client_store, client_name="c", roles=[Roles.CONTROLLER]
    )
    try:
        async with (
            _host_incoming_client(sdk) as url,
            ClientSession() as session,
            session.ws_connect(url) as wsock,
        ):
            conn = SendspinConnection(server, wsock_client=wsock, url=url, pairing_attempt=attempt)
            await asyncio.wait_for(conn.handle_client(), timeout=5)
        assert not sdk.connected
        assert server.get_client(identity.peer_id) is None
        assert await server_store.list_records() == records_before
        assert (await client_store.record_by_server_id(server.id) is not None) is already_paired
    finally:
        await sdk.disconnect()
        await server.close()


async def test_pairing_psk_dial_pairs_the_token_client() -> None:
    """A Pairing PSK dial reaching the token's client pairs it."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    identity = Identity.generate()
    client_store = InMemoryClientPairingStore()
    pairing = generate_psk()
    await client_store.set_pairing_psk(PairingPsk(psk_id=psk_id_for(pairing), psk=pairing))
    attempt = PairingAttempt(
        method=PairMethod.PAIRING_PSK, pairing_psk=pairing, client_id=identity.peer_id
    )

    sdk = make_sdk_client(
        identity=identity, pairing_store=client_store, client_name="c", roles=[Roles.CONTROLLER]
    )
    try:
        async with _host_incoming_client(sdk) as url, _dial(server, url, pairing_attempt=attempt):
            await _await_paired_session(sdk)
            assert await server_store.record_by_client_id(identity.peer_id) is not None
    finally:
        await sdk.disconnect()
        await server.close()


async def test_initiate_pairing_raises_when_client_not_connected() -> None:
    """The server-level wrapper rejects a presence/pairing request for an absent client."""
    server = _make_server(InMemoryServerPairingStore())
    async with _serve(server):
        with pytest.raises(ValueError, match="not connected"):
            await server.initiate_pairing(
                "unknown-client",
                PairingAttempt(
                    method=PairMethod.PAIRING_PSK,
                    pairing_psk=generate_psk(),
                    client_id="unknown-client",
                ),
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
async def _dial(
    server: SendspinServer, url: str, *, pairing_attempt: PairingAttempt | None = None
) -> AsyncIterator[None]:
    """Dial ``url`` from ``server`` (server-initiated), running the connection in the background."""
    async with ClientSession() as session, session.ws_connect(url) as wsock:
        conn = SendspinConnection(
            server, wsock_client=wsock, url=url, pairing_attempt=pairing_attempt
        )
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

    async def display(pairing_code: str | None, **_kwargs: object) -> None:
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


async def test_moving_onto_a_pairing_psk_keeps_the_playback_hold() -> None:
    """The hold must outlive the re-handshake an attempt makes to reach its own PSK.

    A pairing-PSK attempt moves the session off the Sentinel before its exchange runs.
    Nothing has been agreed at that point and the record the client could not use is
    still there, so the constraint has to stand until the pairing actually replaces it.

    ``test_pairing_attempts_that_abort_never_admit_playback`` covers what a lifted hold
    would let through once such an attempt lands the session back on the Sentinel.
    """
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    identity = Identity.generate()
    client_store = await _unpaired_enabled_store()

    psk = generate_psk()
    await server_store.store_record(
        ServerPairingRecord(
            psk_id=psk_id_for(psk), psk=psk, client_id=identity.peer_id, pair_methods=[]
        )
    )
    # The client kept a Pairing PSK but not the record, so it can answer a pairing-PSK
    # attempt while still being unable to use the credential the server references.
    pairing = generate_psk()
    await client_store.set_pairing_psk(PairingPsk(psk_id=psk_id_for(pairing), psk=pairing))

    real_exchange = connection_module.run_pairing_psk_server
    during: dict[str, object] = {}

    async def _observe_then_run(*args: object, **kwargs: object) -> ServerPairingRecord | None:
        """Capture the hold as it stands once the re-handshake is done, then pair for real."""
        during["mismatch"] = conn._credential_mismatch  # noqa: SLF001
        during["category"] = conn._noise_psk.category  # noqa: SLF001
        during["record"] = await server_store.record_by_client_id(identity.peer_id)
        return await real_exchange(*args, **kwargs)  # type: ignore[operator, no-any-return]

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
        )
        try:
            await client.connect(url)
            conn = await _find_connection_by_client_id(server, identity.peer_id)
            # Granted for an ordinary unpaired client and never revoked; it must not
            # become the thing that admits playback once the hold is lost.
            await server.trust_unpaired(identity.peer_id)
            assert conn._credential_mismatch is True  # noqa: SLF001

            with patch.object(connection_module, "run_pairing_psk_server", _observe_then_run):
                await conn.initiate_pairing(
                    PairingAttempt(
                        method=PairMethod.PAIRING_PSK,
                        pairing_psk=pairing,
                        client_id=identity.peer_id,
                    )
                )

            # Mid-attempt: off the Sentinel, but nothing agreed and the old record intact.
            assert during["category"] is PskCategory.PAIRING
            assert during["record"] is not None
            assert during["mismatch"] is True

            # Only the finalized pairing releases it, by replacing what the client lost.
            assert conn._noise_psk is not None  # noqa: SLF001
            assert conn._noise_psk.category is PskCategory.LONG_TERM  # noqa: SLF001
            assert conn._credential_mismatch is False  # noqa: SLF001
            assert conn._roles_to_activate == ["controller@v1"]  # noqa: SLF001
        finally:
            await client.disconnect()


async def test_an_aborted_attempt_off_a_long_term_session_admits_no_playback() -> None:
    """A session re-keyed away from its record carries no playback until the pairing lands."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    identity = Identity.generate()
    client_store = await _unpaired_enabled_store()
    config = await client_store.get_pairing_config()
    # Holds the Pairing PSK but does not offer the method, so the attempt aborts.
    await client_store.store_pairing_config(replace(config, pairing_psk_enabled=False))
    pairing = generate_psk()
    await client_store.set_pairing_psk(PairingPsk(psk_id=psk_id_for(pairing), psk=pairing))
    long_term = generate_psk()
    long_term_id = psk_id_for(long_term)
    await server_store.store_record(
        ServerPairingRecord(
            psk_id=long_term_id, psk=long_term, client_id=identity.peer_id, pair_methods=[]
        )
    )
    await client_store.store_record(
        ClientPairingRecord(psk_id=long_term_id, psk=long_term, server_id=server.id)
    )

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
        )
        try:
            await client.connect(url)
            conn = await _find_connection_by_client_id(server, identity.peer_id)
            await server.trust_unpaired(identity.peer_id)
            assert _server_active_role_count(server, identity.peer_id) == 1

            with pytest.raises(PairingAbortError):
                await conn.initiate_pairing(
                    PairingAttempt(
                        method=PairMethod.PAIRING_PSK,
                        pairing_psk=pairing,
                        client_id=identity.peer_id,
                    )
                )
            await conn.end_pairing()

            assert conn._noise_psk is not None  # noqa: SLF001
            assert conn._noise_psk.category is PskCategory.PAIRING  # noqa: SLF001
            assert conn._playback_capable is False  # noqa: SLF001
            assert _server_active_role_count(server, identity.peer_id) == 0
            await _wait_until(lambda: client.activities == [])
            assert client.connected
        finally:
            await client.disconnect()


async def test_pairing_attempts_that_abort_never_admit_playback() -> None:
    """The user-visible half: attempts that agree nothing must not unblock the session.

    Two attempts of different methods walk the session onto a Pairing PSK and back to the
    Sentinel without replacing the record. Landing back on the Sentinel with a standing
    trusted-unpaired grant is where a lost hold would show up as playback.
    """
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    identity = Identity.generate()

    # Holds the Pairing PSK so the re-handshake onto it lands, but offers neither method,
    # so each attempt is aborted by the client once it sees the activation.
    client_store = InMemoryClientPairingStore()
    config = await client_store.get_pairing_config()
    await client_store.store_pairing_config(
        replace(
            config,
            unpaired_access_enabled=True,
            pairing_psk_enabled=False,
            dynamic_pairing_code_enabled=False,
        )
    )
    pairing = generate_psk()
    await client_store.set_pairing_psk(PairingPsk(psk_id=psk_id_for(pairing), psk=pairing))

    psk = generate_psk()
    await server_store.store_record(
        ServerPairingRecord(
            psk_id=psk_id_for(psk), psk=psk, client_id=identity.peer_id, pair_methods=[]
        )
    )

    async def display(_pairing_code: str | None, **_kwargs: object) -> None:
        """Offer a dynamic out-channel; the attempt aborts before a code is emitted."""
        return

    async def provide() -> str:
        return "000000"

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
            conn = await _find_connection_by_client_id(server, identity.peer_id)
            await server.trust_unpaired(identity.peer_id)
            assert conn._credential_mismatch is True  # noqa: SLF001

            with pytest.raises(PairingAbortError):
                await conn.initiate_pairing(
                    PairingAttempt(
                        method=PairMethod.PAIRING_PSK,
                        pairing_psk=pairing,
                        client_id=identity.peer_id,
                    )
                )
            assert conn._noise_psk is not None  # noqa: SLF001
            assert conn._noise_psk.category is PskCategory.PAIRING  # noqa: SLF001

            with pytest.raises(PairingAbortError):
                await conn.initiate_pairing(
                    PairingAttempt(
                        method=PairMethod.DYNAMIC_PAIRING_CODE,
                        pairing_code_provider=provide,
                        pairing_format=PairingCodeFormat.DIGITS,
                    )
                )
            await conn.end_pairing()

            # Back where trusted-unpaired admits playback, with the record still unusable.
            assert conn._noise_psk.category is PskCategory.SENTINEL  # noqa: SLF001
            assert conn._trusted_unpaired is True  # noqa: SLF001
            assert await server_store.record_by_client_id(identity.peer_id) is not None
            assert conn._playback_capable is False  # noqa: SLF001
            assert conn._roles_to_activate == []  # noqa: SLF001
            assert _server_active_role_count(server, identity.peer_id) == 0
            assert conn._credential_mismatch is True  # noqa: SLF001
        finally:
            await client.disconnect()


async def test_pairing_psk_switch_exchanges_hellos_once() -> None:
    """Re-keying onto the Pairing PSK and then the long-term PSK carries one hello pair."""
    server = _make_server(InMemoryServerPairingStore(), allow_noncompliant_clients=False)
    identity = Identity.generate()
    client_store = InMemoryClientPairingStore()
    pairing = generate_psk()
    await client_store.set_pairing_psk(PairingPsk(psk_id=psk_id_for(pairing), psk=pairing))

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
        )
        try:
            with _count_hellos() as hellos:
                await client.connect(url)
                conn = await _find_connection_by_client_id(server, identity.peer_id)
                await conn.initiate_pairing(
                    PairingAttempt(
                        method=PairMethod.PAIRING_PSK,
                        pairing_psk=pairing,
                        client_id=identity.peer_id,
                    )
                )
                await _await_paired_session(client)
            assert hellos == {"server/hello": 1, "client/hello": 1}
            assert conn.psk_category is PskCategory.LONG_TERM
            assert client.connected
        finally:
            await client.disconnect()


async def test_sentinel_switch_exchanges_hellos_once() -> None:
    """Moving a Pairing PSK session to the Sentinel, then long-term, keeps one hello pair."""
    server = _make_server(InMemoryServerPairingStore(), allow_noncompliant_clients=False)
    identity = Identity.generate()
    # Holds the Pairing PSK, so the first attempt lands on it, but refuses that method.
    client_store = InMemoryClientPairingStore()
    config = await client_store.get_pairing_config()
    await client_store.store_pairing_config(replace(config, pairing_psk_enabled=False))
    pairing = generate_psk()
    await client_store.set_pairing_psk(PairingPsk(psk_id=psk_id_for(pairing), psk=pairing))
    shown: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def display(pairing_code: str | None, **_kwargs: object) -> None:
        if pairing_code is not None and not shown.done():
            shown.set_result(pairing_code)

    async def provide() -> str:
        return await shown

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.CONTROLLER],
            pairing_support=PairingSupport(pairing_code_display=display),
        )
        try:
            with _count_hellos() as hellos:
                await client.connect(url)
                conn = await _find_connection_by_client_id(server, identity.peer_id)
                with pytest.raises(PairingAbortError):
                    await conn.initiate_pairing(
                        PairingAttempt(
                            method=PairMethod.PAIRING_PSK,
                            pairing_psk=pairing,
                            client_id=identity.peer_id,
                        )
                    )
                assert conn.psk_category is PskCategory.PAIRING

                sentinel_rehandshakes: list[PskCategory] = []
                rehandshake = conn._rehandshake_to  # noqa: SLF001

                async def tracking_rehandshake(transport: Any, psk: Any) -> bool:
                    sentinel_rehandshakes.append(psk.category)
                    return await rehandshake(transport, psk)

                conn._rehandshake_to = tracking_rehandshake  # type: ignore[method-assign]  # noqa: SLF001
                await conn.initiate_pairing(
                    PairingAttempt(
                        method=PairMethod.DYNAMIC_PAIRING_CODE,
                        pairing_code_provider=provide,
                        pairing_format=PairingCodeFormat.DIGITS,
                    )
                )
                await _await_paired_session(client)
            assert sentinel_rehandshakes == [PskCategory.SENTINEL, PskCategory.LONG_TERM]
            assert hellos == {"server/hello": 1, "client/hello": 1}
            assert conn.psk_category is PskCategory.LONG_TERM
            assert client.connected
        finally:
            await client.disconnect()


async def test_long_term_adoption_exchanges_hellos_once_and_activates_roles() -> None:
    """Adopting the long-term PSK after pairing activates the roles with no repeated hellos."""
    server = _make_server(InMemoryServerPairingStore(), allow_noncompliant_clients=False)
    identity = Identity.generate()
    client_store = InMemoryClientPairingStore()
    shown: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def display(pairing_code: str | None, **_kwargs: object) -> None:
        if pairing_code is not None and not shown.done():
            shown.set_result(pairing_code)

    async def provide() -> str:
        return await shown

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.PLAYER],
            player_support=_player_support(),
            pairing_support=PairingSupport(pairing_code_display=display),
        )
        try:
            with _count_hellos() as hellos:
                await client.connect(url)
                conn = await _find_connection_by_client_id(server, identity.peer_id)
                assert _server_active_role_count(server, identity.peer_id) == 0
                await conn.initiate_pairing(
                    PairingAttempt(
                        method=PairMethod.DYNAMIC_PAIRING_CODE,
                        pairing_code_provider=provide,
                        pairing_format=PairingCodeFormat.DIGITS,
                    )
                )
                await _await_paired_session(client)
            assert hellos == {"server/hello": 1, "client/hello": 1}
            assert conn.psk_category is PskCategory.LONG_TERM
            assert client.activities == []
            server_client = server.get_client(identity.peer_id)
            assert server_client is not None
            assert [role.role_id for role in server_client.active_roles] == ["player@v1"]
            assert client._admitted_connection is not None  # noqa: SLF001
            assert client._admitted_connection._active_roles == ["player@v1"]  # noqa: SLF001
        finally:
            await client.disconnect()


# DEPRECATED(spec-pr-287): remove in aiosendspin <version>
# The list-form wire is used because a pre-#177 hello on this path receives a queued
# server/time between the repeated client/hello and server/activate.
async def test_pre_spec_287_client_gets_hellos_after_each_rehandshake() -> None:
    """A client whose hello uses a pre-#287 wire gets a hello pair after every re-handshake."""
    server = _make_server(InMemoryServerPairingStore())
    identity = Identity.generate()
    client_store = InMemoryClientPairingStore()
    pairing = generate_psk()
    await client_store.set_pairing_psk(PairingPsk(psk_id=psk_id_for(pairing), psk=pairing))

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.PLAYER],
            player_support=_player_support(),
        )
        try:
            with (
                patch.object(SdkConnection, "_send_client_hello", _send_list_form_hello),
                _pre_spec_287_rehandshake(),
                _count_hellos() as hellos,
            ):
                await client.connect(url)
                conn = await _find_connection_by_client_id(server, identity.peer_id)
                assert conn._expects_rehandshake_hellos is True  # noqa: SLF001
                await conn.initiate_pairing(
                    PairingAttempt(
                        method=PairMethod.PAIRING_PSK,
                        pairing_psk=pairing,
                        client_id=identity.peer_id,
                    )
                )
                await _await_paired_session(client)
            # The initial pair, then one for each of the Pairing PSK and long-term re-handshakes.
            assert hellos == {"server/hello": 3, "client/hello": 3}
            assert conn.psk_category is PskCategory.LONG_TERM
            assert conn._writer_paused is False  # noqa: SLF001
            assert conn._writer_task is not None  # noqa: SLF001
            assert not conn._writer_task.done()  # noqa: SLF001
            assert client.connected
        finally:
            await client.disconnect()


@pytest.mark.parametrize("strict", [False, True])
async def test_repeated_client_hello_is_flagged(
    caplog: pytest.LogCaptureFixture,
    strict: bool,  # noqa: FBT001
) -> None:
    """A current client's second client/hello is a compliance violation, not a new exchange."""
    server = _make_server(InMemoryServerPairingStore(), allow_noncompliant_clients=not strict)
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
            sdk_conn = client._admitted_connection  # noqa: SLF001
            assert sdk_conn is not None
            assert sdk_conn._ws is not None  # noqa: SLF001
            hello = await sdk_conn._build_client_hello()  # noqa: SLF001
            await sdk_conn._ws.send_str(hello.to_json())  # noqa: SLF001
            reason = "sent a second client/hello after the hello exchange"
            if strict:
                await _wait_until(lambda: not client.connected)
                assert f"rejecting non-compliant client c: {reason}" in caplog.messages
            else:
                await _wait_until(lambda: f"non-compliant client c: {reason}" in caplog.messages)
                assert client.connected
        finally:
            await client.disconnect()


async def test_repeated_client_hello_during_pairing_is_flagged_not_routed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A current client's second client/hello mid-attempt is flagged and never reaches pairing."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    client_identity = Identity.generate()
    client_store = InMemoryClientPairingStore()

    async with _serve(server) as url:
        client, attempt, release = await _paired_client_with_stalled_success_tail(
            server, url, client_identity, client_store
        )
        try:
            conn = await _find_connection_by_client_id(server, client_identity.peer_id)
            routed = _track_routed_types(conn)
            sdk_conn = client._admitted_connection  # noqa: SLF001
            assert sdk_conn is not None
            assert sdk_conn._ws is not None  # noqa: SLF001
            hello = await sdk_conn._build_client_hello()  # noqa: SLF001
            await sdk_conn._ws.send_str(hello.to_json())  # noqa: SLF001
            await _wait_until(
                lambda: (
                    "non-compliant client c: sent a second client/hello after the hello exchange"
                    in caplog.messages
                )
            )
            assert "client/hello" not in routed
            release.set()

            await attempt
            await _await_paired_session(client)
            assert conn.psk_category is PskCategory.LONG_TERM
        finally:
            release.set()
            await client.disconnect()


async def test_rehandshake_reloads_trusted_unpaired_for_the_new_psk() -> None:
    """Leaving the long-term PSK re-reads the unpaired-access grant the session now runs under."""
    server_store = InMemoryServerPairingStore()
    server = _make_server(server_store)
    identity = Identity.generate()
    await server_store.add_trusted_unpaired(TrustedUnpairedClient(client_id=identity.peer_id))
    # Admits unpaired access and holds the Pairing PSK, but refuses to pair with it.
    client_store = await _unpaired_enabled_store()
    config = await client_store.get_pairing_config()
    await client_store.store_pairing_config(replace(config, pairing_psk_enabled=False))
    long_term = generate_psk()
    long_term_id = psk_id_for(long_term)
    await server_store.store_record(
        ServerPairingRecord(
            psk_id=long_term_id, psk=long_term, client_id=identity.peer_id, pair_methods=[]
        )
    )
    await client_store.store_record(
        ClientPairingRecord(psk_id=long_term_id, psk=long_term, server_id=server.id)
    )
    pairing = generate_psk()
    await client_store.set_pairing_psk(PairingPsk(psk_id=psk_id_for(pairing), psk=pairing))

    async with _serve(server) as url:
        client = make_sdk_client(
            identity=identity,
            pairing_store=client_store,
            client_name="c",
            roles=[Roles.PLAYER],
            player_support=_player_support(),
        )
        try:
            await client.connect(url)
            conn = await _find_connection_by_client_id(server, identity.peer_id)
            # A long-term session never reads the grant.
            assert conn._trusted_unpaired is False  # noqa: SLF001

            with pytest.raises(PairingAbortError):
                await conn.initiate_pairing(
                    PairingAttempt(
                        method=PairMethod.PAIRING_PSK,
                        pairing_psk=pairing,
                        client_id=identity.peer_id,
                    )
                )
            assert conn.psk_category is PskCategory.PAIRING
            assert conn._trusted_unpaired is True  # noqa: SLF001

            # Once the record is gone, the grant alone admits playback on the Pairing PSK.
            await server_store.remove_record(identity.peer_id)
            conn.forget_credential_mismatch()
            await conn.end_pairing()
            assert conn._playback_capable is True  # noqa: SLF001
            assert _server_active_role_count(server, identity.peer_id) == 1
        finally:
            await client.disconnect()
