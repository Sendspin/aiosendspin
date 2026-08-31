"""Tests for multi-server support (connection reasons and client reclaim)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
from aiohttp import WSMessage, WSMsgType, web
from mashumaro.exceptions import MissingField

from aiosendspin.models.core import (
    ClientHelloMessage,
    ClientHelloPayload,
    ServerActivateMessage,
    ServerStateMessage,
    ServerStatePayload,
    UnpairedAccess,
)
from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.models.types import (
    Activity,
    AudioCodec,
    ConnectionReason,
    PairAbortReason,
    PlaybackStateType,
    PlayerCommand,
    Roles,
)
from aiosendspin.noise.keys import generate_psk, psk_id_for
from aiosendspin.noise.pairing import PairingAbortError, RemotePairingAbortError
from aiosendspin.noise.trust_store import (
    InMemoryServerPairingStore,
    PskCategory,
    ResolvedPsk,
    ServerPairingStore,
    TrustedUnpairedClient,
)
from aiosendspin.noise.wire import EncryptedWebSocket
from aiosendspin.server.client import SendspinClient
from aiosendspin.server.clock import LoopClock
from aiosendspin.server.compliance import ClientComplianceError
from aiosendspin.server.connection import SendspinConnection
from aiosendspin.server.group import SendspinGroup
from aiosendspin.server.roles.negotiation import negotiate_roles
from aiosendspin.server.roles.registry import ROLE_FACTORIES

if TYPE_CHECKING:
    from aiosendspin.models.types import ServerMessage


@dataclass
class _MockServer:
    """Mock server for testing connection reason lookup."""

    loop: asyncio.AbstractEventLoop
    clock: LoopClock
    id: str = "srv"
    name: str = "server"
    allow_noncompliant_clients: bool = True
    pairing_store: ServerPairingStore = field(default_factory=InMemoryServerPairingStore)
    remove_client: AsyncMock = field(default_factory=AsyncMock)

    _connection_reasons: dict[str, ConnectionReason] = field(default_factory=dict)
    _client_urls: dict[str, str] = field(default_factory=dict)
    _clients: dict[str, SendspinClient] = field(default_factory=dict)
    _connection_tasks: dict[str, asyncio.Task[None]] = field(default_factory=dict)

    def is_external_player(self, client_id: str) -> bool:  # noqa: ARG002
        return False

    def on_client_first_connect(self, client_id: str) -> None:  # noqa: ARG002
        return

    def get_connection_reason(self, url: str) -> ConnectionReason:
        return self._connection_reasons.get(url, ConnectionReason.DISCOVERY)

    def register_client_url(self, client_id: str, url: str) -> None:
        self._client_urls[client_id] = url

    def get_client_url(self, client_id: str) -> str | None:
        return self._client_urls.get(client_id)

    def get_or_create_client(self, client_id: str) -> SendspinClient:
        client = self._clients.get(client_id)
        if client is None:
            client = SendspinClient(self, client_id=client_id)
            self._clients[client_id] = client
            SendspinGroup(self, client)
        return client

    def _signal_client_connected(self, client_id: str) -> None:
        pass

    def _signal_client_disconnected(self, client_id: str, goodbye_reason: object = None) -> None:
        pass


class _DummyConnection:
    def __init__(self) -> None:
        self.sent_messages: list[ServerMessage] = []

    async def disconnect(self, *, retry_connection: bool = True) -> None:  # noqa: ARG002
        return

    def send_message(self, message: ServerMessage) -> None:
        self.sent_messages.append(message)

    def send_role_message(self, role: str, message: ServerMessage) -> None:  # noqa: ARG002
        self.sent_messages.append(message)

    def send_binary(
        self,
        data: bytes,  # noqa: ARG002
        *,
        role: str,  # noqa: ARG002
        timestamp_us: int,  # noqa: ARG002
        message_type: int,  # noqa: ARG002
        buffer_end_time_us: int | None = None,  # noqa: ARG002
        buffer_byte_count: int | None = None,  # noqa: ARG002
        duration_us: int | None = None,  # noqa: ARG002
    ) -> bool:
        return True


def _player_hello(client_id: str, *, unpaired_access: bool = False) -> ClientHelloPayload:
    return ClientHelloPayload(
        client_id=client_id,
        name=client_id,
        version=1,
        supported_roles=[Roles.PLAYER.value],
        unpaired_access=UnpairedAccess(enabled=unpaired_access),
        player_support=ClientHelloPlayerSupport(
            supported_formats=[
                SupportedAudioFormat(
                    codec=AudioCodec.PCM,
                    channels=2,
                    sample_rate=48000,
                    bit_depth=16,
                )
            ],
            buffer_capacity=100_000,
            supported_commands=[PlayerCommand.VOLUME, PlayerCommand.MUTE],
        ),
    )


def _client_hello_frame(client_id: str, *, unpaired_access: bool = False) -> WSMessage:
    return WSMessage(
        WSMsgType.TEXT,
        ClientHelloMessage(
            payload=_player_hello(client_id, unpaired_access=unpaired_access)
        ).to_json(),
        "",
    )


class _FakeTransport:
    """Transport double: records sent frames and yields queued inbound frames."""

    def __init__(self, incoming: list[WSMessage] | None = None) -> None:
        self.sent: list[str] = []
        self._incoming = iter(incoming or [])
        self.closed = False
        self.close_code: int | None = None

    async def send_str(self, data: str) -> None:
        self.sent.append(data)

    async def receive(self) -> WSMessage | None:
        return next(self._incoming, None)

    def sent_payloads(self) -> list[dict]:
        return [orjson.loads(s) for s in self.sent]


async def _exchange_hellos_encrypted(
    conn: SendspinConnection,
    *,
    client_id: str = "client-1",
    category: PskCategory = PskCategory.LONG_TERM,
    unpaired_access: bool = False,
) -> _FakeTransport:
    """Drive a non-legacy (encrypted) hello exchange and return the recording transport."""
    psk = generate_psk()
    conn._client_id = client_id  # noqa: SLF001
    conn._noise_psk = ResolvedPsk(  # noqa: SLF001
        psk_id=psk_id_for(psk), psk=psk, category=category, counterparty_id=client_id
    )
    fake = _FakeTransport([_client_hello_frame(client_id, unpaired_access=unpaired_access)])
    conn._transport = fake  # type: ignore[assignment]  # noqa: SLF001
    await conn._exchange_hellos()  # noqa: SLF001
    return fake


@pytest.fixture
async def mock_server() -> _MockServer:
    """Create a mock server with connection reason tracking."""
    loop = asyncio.get_running_loop()
    return _MockServer(loop=loop, clock=LoopClock(loop))


class TestConnectionReasonLookup:
    """Tests for connection reason lookup from server."""

    def test_get_connection_reason_defaults_to_discovery(self, mock_server: _MockServer) -> None:
        """Connection reason defaults to DISCOVERY for unknown URLs."""
        assert mock_server.get_connection_reason("ws://unknown:1234") == ConnectionReason.DISCOVERY

    def test_get_connection_reason_returns_stored_reason(self, mock_server: _MockServer) -> None:
        """Connection reason returns the stored value for known URLs."""
        url = "ws://192.168.1.100:8927/sendspin"
        mock_server._connection_reasons[url] = ConnectionReason.PLAYBACK  # noqa: SLF001

        assert mock_server.get_connection_reason(url) == ConnectionReason.PLAYBACK


class TestClientUrlTracking:
    """Tests for client URL registration and lookup."""

    def test_register_and_get_client_url(self, mock_server: _MockServer) -> None:
        """Client URL can be registered and retrieved."""
        mock_server.register_client_url("client-1", "ws://192.168.1.50:8927/sendspin")

        assert mock_server.get_client_url("client-1") == "ws://192.168.1.50:8927/sendspin"

    def test_get_client_url_returns_none_for_unknown(self, mock_server: _MockServer) -> None:
        """get_client_url returns None for unknown client IDs."""
        assert mock_server.get_client_url("unknown-client") is None


class TestEncryptedActivities:
    """Tests that the encrypted hello exchange declares activities from PSK + capabilities."""

    @pytest.mark.asyncio
    async def test_long_term_idle_declares_no_activities(self, mock_server: _MockServer) -> None:
        """An idle long-term connection declares no activities but advertises active roles."""
        conn = SendspinConnection(mock_server, wsock_client=AsyncMock())
        fake = await _exchange_hellos_encrypted(conn, category=PskCategory.LONG_TERM)

        payloads = fake.sent_payloads()
        assert [p["type"] for p in payloads] == ["server/hello", "server/activate"]
        activate = payloads[1]["payload"]
        assert activate["activities"] == []
        # active_roles present because the connection is playback-capable.
        assert Roles.PLAYER.value in activate["active_roles"]

    @pytest.mark.asyncio
    async def test_legacy_support_keys_logged_on_ingest(
        self, mock_server: _MockServer, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A hello carrying unversioned support keys is admitted and logged."""
        raw = orjson.dumps(
            {
                "type": "client/hello",
                "payload": {
                    "client_id": "client-1",
                    "name": "client-1",
                    "version": 1,
                    "supported_roles": ["player@v1"],
                    "player_support": {
                        "supported_formats": [
                            {"codec": "pcm", "channels": 2, "sample_rate": 48000, "bit_depth": 16}
                        ],
                        "buffer_capacity": 100_000,
                        "supported_commands": [],
                    },
                },
            }
        ).decode()
        conn = SendspinConnection(mock_server, wsock_client=AsyncMock())
        psk = generate_psk()
        conn._client_id = "client-1"  # noqa: SLF001
        conn._noise_psk = ResolvedPsk(  # noqa: SLF001
            psk_id=psk_id_for(psk),
            psk=psk,
            category=PskCategory.LONG_TERM,
            counterparty_id="client-1",
        )
        conn._transport = _FakeTransport([WSMessage(WSMsgType.TEXT, raw, "")])  # type: ignore[assignment]  # noqa: SLF001

        with caplog.at_level("INFO"):
            assert await conn._exchange_hellos() is True  # noqa: SLF001
        assert "unversioned support keys" in caplog.text

    @pytest.mark.asyncio
    async def test_strict_server_excludes_draft_visualizer_without_rejecting(self) -> None:
        """Strict mode does not activate the legacy draft wire, but admits the client."""
        loop = asyncio.get_running_loop()
        strict_server = _MockServer(
            loop=loop, clock=LoopClock(loop), allow_noncompliant_clients=False
        )
        raw = orjson.dumps(
            {
                "type": "client/hello",
                "payload": {
                    "client_id": "client-1",
                    "name": "client-1",
                    "version": 1,
                    "supported_roles": ["visualizer@_draft_r1"],
                    "visualizer@_draft_r1_support": {
                        "buffer_capacity": 1024,
                        "types": ["loudness"],
                        "batch_max": 8,
                    },
                },
            }
        ).decode()
        conn = SendspinConnection(strict_server, wsock_client=AsyncMock())
        psk = generate_psk()
        conn._client_id = "client-1"  # noqa: SLF001
        conn._noise_psk = ResolvedPsk(  # noqa: SLF001
            psk_id=psk_id_for(psk),
            psk=psk,
            category=PskCategory.LONG_TERM,
            counterparty_id="client-1",
        )
        conn._transport = _FakeTransport([WSMessage(WSMsgType.TEXT, raw, "")])  # type: ignore[assignment]  # noqa: SLF001

        assert await conn._exchange_hellos() is True  # noqa: SLF001
        assert "visualizer@_draft_r1" not in conn._negotiated_roles  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_strict_server_rejects_noncompliant_hello(self) -> None:
        """When noncompliance is disallowed, an unversioned-support-key hello is rejected."""
        loop = asyncio.get_running_loop()
        strict_server = _MockServer(
            loop=loop, clock=LoopClock(loop), allow_noncompliant_clients=False
        )
        raw = orjson.dumps(
            {
                "type": "client/hello",
                "payload": {
                    "client_id": "client-1",
                    "name": "client-1",
                    "version": 1,
                    "supported_roles": ["player@v1"],
                    "player_support": {
                        "supported_formats": [
                            {"codec": "pcm", "channels": 2, "sample_rate": 48000, "bit_depth": 16}
                        ],
                        "buffer_capacity": 100_000,
                        "supported_commands": [],
                    },
                },
            }
        ).decode()
        conn = SendspinConnection(strict_server, wsock_client=AsyncMock())
        psk = generate_psk()
        conn._client_id = "client-1"  # noqa: SLF001
        conn._noise_psk = ResolvedPsk(  # noqa: SLF001
            psk_id=psk_id_for(psk),
            psk=psk,
            category=PskCategory.LONG_TERM,
            counterparty_id="client-1",
        )
        conn._transport = _FakeTransport([WSMessage(WSMsgType.TEXT, raw, "")])  # type: ignore[assignment]  # noqa: SLF001

        assert await conn._exchange_hellos() is False  # noqa: SLF001
        assert "client-1" not in strict_server._clients  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_message_loop_hard_rejects_on_compliance_error(
        self, mock_server: _MockServer
    ) -> None:
        """A ClientComplianceError during dispatch ends the loop with no warm reconnect."""

        class _AsyncIterTransport:
            close_code = 1000

            def __init__(self, msgs: list[WSMessage]) -> None:
                self._msgs = msgs

            def __aiter__(self) -> _AsyncIterTransport:
                return self

            async def __anext__(self) -> WSMessage:
                if not self._msgs:
                    raise StopAsyncIteration
                return self._msgs.pop(0)

        conn = SendspinConnection(mock_server, wsock_client=AsyncMock())
        text = orjson.dumps({"type": "client/time", "payload": {"client_transmitted": 1}}).decode()
        conn._transport = _AsyncIterTransport([WSMessage(WSMsgType.TEXT, text, "")])  # type: ignore[assignment]  # noqa: SLF001
        conn._handle_message = AsyncMock(side_effect=ClientComplianceError("bad"))  # type: ignore[method-assign]  # noqa: SLF001
        conn.disconnect = AsyncMock()  # type: ignore[method-assign]

        await conn._run_message_loop()  # noqa: SLF001
        # Cleanup must tear the connection down without a warm reconnect.
        await conn._cleanup_connection()  # noqa: SLF001
        conn.disconnect.assert_awaited_once_with(retry_connection=False)

    @pytest.mark.asyncio
    async def test_client_binary_frame_is_ignored_not_rejected(self) -> None:
        """A client binary frame is logged and skipped, never rejecting the connection."""

        class _AsyncIterTransport:
            close_code = 1000

            def __init__(self, msgs: list[WSMessage]) -> None:
                self._msgs = msgs

            def __aiter__(self) -> _AsyncIterTransport:
                return self

            async def __anext__(self) -> WSMessage:
                if not self._msgs:
                    raise StopAsyncIteration
                return self._msgs.pop(0)

        loop = asyncio.get_running_loop()
        strict_server = _MockServer(
            loop=loop, clock=LoopClock(loop), allow_noncompliant_clients=False
        )
        conn = SendspinConnection(strict_server, wsock_client=AsyncMock())
        conn._transport = _AsyncIterTransport([WSMessage(WSMsgType.BINARY, b"\x00", "")])  # type: ignore[assignment]  # noqa: SLF001

        await conn._run_message_loop()  # noqa: SLF001
        assert conn._closing is False  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_unencrypted_hello_without_version_is_flagged(
        self, mock_server: _MockServer, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An unencrypted hello that omits the required version is admitted and flagged."""
        conn = SendspinConnection(mock_server, wsock_client=AsyncMock())  # no PSK => unencrypted
        raw = orjson.dumps(
            {
                "type": "client/hello",
                "payload": {
                    "client_id": "legacy-1",
                    "name": "legacy-1",
                    "supported_roles": [],
                },
            }
        ).decode()
        with caplog.at_level("INFO"):
            assert await conn._ingest_client_hello(raw) is True  # noqa: SLF001
        assert "omitted required version" in caplog.text

    @pytest.mark.asyncio
    async def test_dialed_for_playback_declares_playback(self, mock_server: _MockServer) -> None:
        """A connection dialed for playback declares the playback activity up front."""
        url = "ws://192.168.1.100:8927/sendspin"
        mock_server._connection_reasons[url] = ConnectionReason.PLAYBACK  # noqa: SLF001
        conn = SendspinConnection(mock_server, wsock_client=AsyncMock(), url=url)
        fake = await _exchange_hellos_encrypted(conn, category=PskCategory.LONG_TERM)

        activate = fake.sent_payloads()[1]["payload"]
        assert activate["activities"] == [Activity.PLAYBACK.value]
        assert Roles.PLAYER.value in activate["active_roles"]

    @pytest.mark.asyncio
    async def test_playback_state_change_resends_activate(self, mock_server: _MockServer) -> None:
        """A group playback-state change re-sends server/activate (active_roles omitted)."""
        conn = SendspinConnection(mock_server, wsock_client=AsyncMock())
        await _exchange_hellos_encrypted(conn, category=PskCategory.LONG_TERM)
        conn._subscribe_activity_events()  # noqa: SLF001
        assert conn._client is not None  # noqa: SLF001

        # Idle → playing: a fresh server/activate with [playback] is enqueued.
        conn._client.group._set_playback_state(PlaybackStateType.PLAYING)  # noqa: SLF001
        activates = [m for m in conn._priority_messages if isinstance(m, ServerActivateMessage)]  # noqa: SLF001
        assert activates, "expected a re-sent server/activate"
        assert activates[-1].payload.activities == [Activity.PLAYBACK]
        assert activates[-1].payload.active_roles is None  # sticky: omitted on re-send

        # Playing → stopped: re-sent with an empty activity set.
        conn._client.group._set_playback_state(PlaybackStateType.STOPPED)  # noqa: SLF001
        activates = [m for m in conn._priority_messages if isinstance(m, ServerActivateMessage)]  # noqa: SLF001
        assert activates[-1].payload.activities == []

    @pytest.mark.asyncio
    async def test_sentinel_untrusted_declares_no_roles(self, mock_server: _MockServer) -> None:
        """A Sentinel client the server hasn't trusted activates no roles."""
        conn = SendspinConnection(mock_server, wsock_client=AsyncMock())
        fake = await _exchange_hellos_encrypted(
            conn, category=PskCategory.SENTINEL, unpaired_access=True
        )

        activate = fake.sent_payloads()[1]["payload"]
        assert activate["activities"] == []
        # active_roles is empty (no active roles) when not playback-capable; absent would
        # mean "unchanged" (sticky).
        assert activate["active_roles"] == []

    @pytest.mark.asyncio
    async def test_sentinel_trusted_activates_roles(self, mock_server: _MockServer) -> None:
        """A trusted Sentinel client that admits unpaired access gets its roles provisioned."""
        await mock_server.pairing_store.add_trusted_unpaired(
            TrustedUnpairedClient(client_id="client-1")
        )
        conn = SendspinConnection(mock_server, wsock_client=AsyncMock())
        fake = await _exchange_hellos_encrypted(
            conn, category=PskCategory.SENTINEL, unpaired_access=True
        )

        activate = fake.sent_payloads()[1]["payload"]
        # Idle (no playback yet) so no activity declared, but roles are provisioned.
        assert activate["activities"] == []
        assert activate["active_roles"] == [Roles.PLAYER.value]

    @pytest.mark.asyncio
    async def test_sentinel_trusted_but_client_disables_unpaired(
        self, mock_server: _MockServer
    ) -> None:
        """A trusted client that itself refuses unpaired access gets no roles."""
        await mock_server.pairing_store.add_trusted_unpaired(
            TrustedUnpairedClient(client_id="client-1")
        )
        conn = SendspinConnection(mock_server, wsock_client=AsyncMock())
        fake = await _exchange_hellos_encrypted(
            conn, category=PskCategory.SENTINEL, unpaired_access=False
        )

        activate = fake.sent_payloads()[1]["payload"]
        assert activate["active_roles"] == []


class TestLegacyServerHello:
    """Tests for the legacy (transition-mode) server/hello path."""

    @pytest.mark.asyncio
    async def test_legacy_hello_carries_connection_reason(self, mock_server: _MockServer) -> None:
        """A legacy (unencrypted) hello exchange uses connection_reason, no server/activate."""
        url = "ws://192.168.1.100:8927/sendspin"
        mock_server._connection_reasons[url] = ConnectionReason.PLAYBACK  # noqa: SLF001
        conn = SendspinConnection(mock_server, wsock_client=AsyncMock(), url=url)

        fake = _FakeTransport()
        conn._transport = fake  # type: ignore[assignment]  # noqa: SLF001
        # Legacy connections peek client/hello before the transport is encrypted.
        conn._pending_first_text = ClientHelloMessage(  # noqa: SLF001
            payload=_player_hello("client-1")
        ).to_json()
        await conn._exchange_hellos()  # noqa: SLF001

        payloads = fake.sent_payloads()
        assert [p["type"] for p in payloads] == ["server/hello"]
        assert payloads[0]["payload"]["connection_reason"] == ConnectionReason.PLAYBACK.value

    @pytest.mark.asyncio
    async def test_legacy_hello_clamps_post_legacy_reasons(self, mock_server: _MockServer) -> None:
        """Reasons legacy clients cannot parse are sent as discovery."""
        url = "ws://192.168.1.100:8927/sendspin"
        mock_server._connection_reasons[url] = ConnectionReason.MANAGEMENT  # noqa: SLF001
        conn = SendspinConnection(mock_server, wsock_client=AsyncMock(), url=url)

        fake = _FakeTransport()
        conn._transport = fake  # type: ignore[assignment]  # noqa: SLF001
        conn._pending_first_text = ClientHelloMessage(  # noqa: SLF001
            payload=_player_hello("client-1")
        ).to_json()
        await conn._exchange_hellos()  # noqa: SLF001

        reason = fake.sent_payloads()[0]["payload"]["connection_reason"]
        assert reason == ConnectionReason.DISCOVERY.value


class _FakePairingTransport(_FakeTransport, EncryptedWebSocket):
    """A recording transport that satisfies the encrypted-transport check in pairing."""

    # Shadow the EncryptedWebSocket properties so _FakeTransport.__init__ can assign them.
    closed = False
    close_code: int | None = None

    def __init__(self, incoming: list[WSMessage] | None = None) -> None:
        _FakeTransport.__init__(self, incoming)


class TestInitialConnectPairingAbort:
    """Aborted initial-connect pairing follows the spec's closing/non-closing split."""

    @staticmethod
    def _pairing_connection(
        mock_server: _MockServer, reason: PairAbortReason
    ) -> tuple[SendspinConnection, _FakePairingTransport]:
        conn = SendspinConnection(mock_server, wsock_client=AsyncMock())
        psk = generate_psk()
        conn._client_id = "client-1"  # noqa: SLF001
        conn._noise_psk = ResolvedPsk(  # noqa: SLF001
            psk_id=psk_id_for(psk), psk=psk, category=PskCategory.PAIRING
        )
        fake = _FakePairingTransport([_client_hello_frame("client-1")])
        conn._transport = fake  # noqa: SLF001
        conn._pair = AsyncMock(  # type: ignore[method-assign]  # noqa: SLF001
            side_effect=RemotePairingAbortError(reason)
        )
        return conn, fake

    @pytest.mark.asyncio
    async def test_nonclosing_abort_keeps_connection(self, mock_server: _MockServer) -> None:
        """A non-closing abort reason ends pairing with a leave activate, not a teardown."""
        conn, fake = self._pairing_connection(mock_server, PairAbortReason.ATTEMPT_TIMEOUT)

        assert await conn._exchange_hellos() is True  # noqa: SLF001

        payloads = fake.sent_payloads()
        assert payloads[-1]["type"] == "server/activate"
        assert Activity.PAIRING.value not in payloads[-1]["payload"]["activities"]

    @pytest.mark.asyncio
    async def test_closing_abort_propagates(self, mock_server: _MockServer) -> None:
        """A closing abort reason still tears the connection down."""
        conn, _ = self._pairing_connection(mock_server, PairAbortReason.CONCURRENT_ATTEMPT)

        with pytest.raises(PairingAbortError):
            await conn._exchange_hellos()  # noqa: SLF001


class TestHandshakeOrdering:
    """Tests for server/hello ordering relative to role messages."""

    @pytest.mark.asyncio
    async def test_no_role_messages_sent_before_server_activate(
        self, mock_server: _MockServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Role messages enqueued during attach must not be sent until after the hello exchange.

        The writer only starts after server/activate, so a role message enqueued
        while attaching the client cannot land on the wire before the hello-exchange frames.
        """
        conn = SendspinConnection(mock_server, wsock_client=AsyncMock())

        original_attach_connection = SendspinClient.attach_connection

        def attach_with_early_role_message(
            self: SendspinClient,
            connection: SendspinConnection,
            *,
            client_info: ClientHelloPayload,
            active_roles: list[str],
            negotiated_roles: list[str] | None = None,
        ) -> None:
            connection.send_role_message(
                "metadata",
                ServerStateMessage(payload=ServerStatePayload()),
            )
            original_attach_connection(
                self,
                connection,
                client_info=client_info,
                active_roles=active_roles,
                negotiated_roles=negotiated_roles,
            )

        monkeypatch.setattr(SendspinClient, "attach_connection", attach_with_early_role_message)

        fake = await _exchange_hellos_encrypted(conn)

        # Only the hello-exchange frames reach the wire; the role message stays queued.
        assert [p["type"] for p in fake.sent_payloads()] == ["server/hello", "server/activate"]


class TestCustomRoleSupportParsing:
    """Tests for custom role support-key handling in inbound client/hello messages."""

    @pytest.mark.parametrize(
        ("role_id", "support_key", "expected_attr", "support_payload"),
        [
            (
                "player@_custom_version",
                "player@_custom_version_support",
                "player_support",
                {
                    "supported_formats": [
                        {"codec": "pcm", "sample_rate": 48000, "bit_depth": 16, "channels": 2}
                    ],
                    "buffer_capacity": 100_000,
                    "supported_commands": [],
                },
            ),
            (
                "artwork@_custom_version",
                "artwork@_custom_version_support",
                "artwork_support",
                {
                    "channels": [
                        {
                            "source": "album",
                            "format": "jpeg",
                            "width": 300,
                            "height": 300,
                        }
                    ]
                },
            ),
            (
                "visualizer@_custom_version",
                "visualizer@_custom_version_support",
                "visualizer_support",
                {"buffer_capacity": 100_000, "rate_max": 30, "types": ["loudness"]},
            ),
        ],
    )
    def test_deserialize_client_hello_maps_custom_support(
        self,
        role_id: str,
        support_key: str,
        expected_attr: str,
        support_payload: dict[str, object],
    ) -> None:
        """Custom role support keys are mapped into existing support fields."""
        raw = orjson.dumps(
            {
                "type": "client/hello",
                "payload": {
                    "client_id": "c1",
                    "name": "Client",
                    "version": 1,
                    "supported_roles": [role_id],
                    support_key: support_payload,
                },
            }
        ).decode()

        msg = SendspinConnection._deserialize_client_message(raw)  # noqa: SLF001
        assert isinstance(msg, ClientHelloMessage)
        assert getattr(msg.payload, expected_attr) is not None

    @pytest.mark.parametrize(
        ("role_id", "missing_support_key"),
        [
            ("player@_custom_version", "player@_custom_version_support"),
            ("artwork@_custom_version", "artwork@_custom_version_support"),
            ("visualizer@_custom_version", "visualizer@_custom_version_support"),
        ],
    )
    def test_deserialize_client_hello_requires_custom_support_key(
        self, role_id: str, missing_support_key: str
    ) -> None:
        """Custom role IDs require their matching custom support keys."""
        raw = orjson.dumps(
            {
                "type": "client/hello",
                "payload": {
                    "client_id": "c1",
                    "name": "Client",
                    "version": 1,
                    "supported_roles": [role_id],
                },
            }
        ).decode()

        with pytest.raises(ValueError, match=missing_support_key):
            SendspinConnection._deserialize_client_message(raw)  # noqa: SLF001

    def test_deserialize_client_hello_records_legacy_support_key_use(self) -> None:
        """Unversioned support keys are rewritten to v1 aliases and recorded, not logged."""
        raw = orjson.dumps(
            {
                "type": "client/hello",
                "payload": {
                    "client_id": "c1",
                    "name": "Client",
                    "version": 1,
                    "supported_roles": ["player@v1"],
                    "player_support": {
                        "supported_formats": [
                            {
                                "codec": "pcm",
                                "sample_rate": 48000,
                                "bit_depth": 16,
                                "channels": 2,
                            }
                        ],
                        "buffer_capacity": 100_000,
                        "supported_commands": [],
                    },
                },
            }
        ).decode()

        msg = SendspinConnection._deserialize_client_message(raw)  # noqa: SLF001
        assert isinstance(msg, ClientHelloMessage)
        assert msg.payload.player_support is not None
        assert msg.payload.legacy_support_keys_used == ["player_support"]

    def test_deserialize_client_hello_records_legacy_key_alongside_canonical(self) -> None:
        """A hello sending both the legacy and versioned support keys still records the legacy."""
        support = {
            "supported_formats": [
                {"codec": "pcm", "sample_rate": 48000, "bit_depth": 16, "channels": 2}
            ],
            "buffer_capacity": 100_000,
            "supported_commands": [],
        }
        raw = orjson.dumps(
            {
                "type": "client/hello",
                "payload": {
                    "client_id": "c1",
                    "name": "Client",
                    "version": 1,
                    "supported_roles": ["player@v1"],
                    "player@v1_support": support,
                    "player_support": support,
                },
            }
        ).decode()

        msg = SendspinConnection._deserialize_client_message(raw)  # noqa: SLF001
        assert isinstance(msg, ClientHelloMessage)
        assert msg.payload.player_support is not None
        assert msg.payload.legacy_support_keys_used == ["player_support"]

    def test_deserialize_client_hello_records_unlisted_support(self) -> None:
        """A support object for a role not in supported_roles is dropped and recorded."""
        raw = orjson.dumps(
            {
                "type": "client/hello",
                "payload": {
                    "client_id": "c1",
                    "name": "Client",
                    "version": 1,
                    "supported_roles": [],
                    "player@v1_support": {
                        "supported_formats": [
                            {
                                "codec": "pcm",
                                "sample_rate": 48000,
                                "bit_depth": 16,
                                "channels": 2,
                            }
                        ],
                        "buffer_capacity": 100_000,
                        "supported_commands": [],
                    },
                },
            }
        ).decode()

        msg = SendspinConnection._deserialize_client_message(raw)  # noqa: SLF001
        assert isinstance(msg, ClientHelloMessage)
        assert msg.payload.player_support is None
        assert msg.payload.unlisted_support_roles == ["player@v1"]

    def test_deserialize_client_hello_ignores_spoofed_legacy_record(self) -> None:
        """A versioned hello records nothing, even if it spoofs legacy_support_keys_used."""
        raw = orjson.dumps(
            {
                "type": "client/hello",
                "payload": {
                    "client_id": "c1",
                    "name": "Client",
                    "version": 1,
                    "supported_roles": ["player@v1"],
                    "legacy_support_keys_used": ["player_support"],
                    "player@v1_support": {
                        "supported_formats": [
                            {
                                "codec": "pcm",
                                "sample_rate": 48000,
                                "bit_depth": 16,
                                "channels": 2,
                            }
                        ],
                        "buffer_capacity": 100_000,
                        "supported_commands": [],
                    },
                },
            }
        ).decode()

        msg = SendspinConnection._deserialize_client_message(raw)  # noqa: SLF001
        assert isinstance(msg, ClientHelloMessage)
        assert msg.payload.legacy_support_keys_used is None

    def test_deserialize_client_hello_ignores_spoofed_unlisted_record(self) -> None:
        """A hello with no unlisted support records nothing, even if it spoofs the field."""
        raw = orjson.dumps(
            {
                "type": "client/hello",
                "payload": {
                    "client_id": "c1",
                    "name": "Client",
                    "version": 1,
                    "supported_roles": ["player@v1"],
                    "unlisted_support_roles": ["artwork@v1"],
                    "player@v1_support": {
                        "supported_formats": [
                            {
                                "codec": "pcm",
                                "sample_rate": 48000,
                                "bit_depth": 16,
                                "channels": 2,
                            }
                        ],
                        "buffer_capacity": 100_000,
                        "supported_commands": [],
                    },
                },
            }
        ).decode()

        msg = SendspinConnection._deserialize_client_message(raw)  # noqa: SLF001
        assert isinstance(msg, ClientHelloMessage)
        assert msg.payload.unlisted_support_roles is None

    def test_legacy_visualizer_support_key_is_not_normalized_to_v1(self) -> None:
        """Legacy visualizer_support must not be rewritten to visualizer@v1_support."""
        raw = orjson.dumps(
            {
                "type": "client/hello",
                "payload": {
                    "client_id": "c1",
                    "name": "Client",
                    "version": 1,
                    "supported_roles": ["visualizer@_custom_r1"],
                    "visualizer_support": {
                        "types": ["loudness", "f_peak"],
                        "buffer_capacity": 65536,
                        "rate_max": 30,
                    },
                },
            }
        ).decode()

        msg = SendspinConnection._deserialize_client_message(raw)  # noqa: SLF001
        assert isinstance(msg, ClientHelloMessage)
        assert msg.payload.visualizer_support is not None

    def test_family_order_prefers_first_role_and_does_not_require_second_version_support(
        self,
    ) -> None:
        """When v1 is listed before v2, parser must not require v2 support key."""
        raw = orjson.dumps(
            {
                "type": "client/hello",
                "payload": {
                    "client_id": "c1",
                    "name": "Client",
                    "version": 1,
                    "supported_roles": ["player@v1", "player@v2"],
                    "player@v1_support": {
                        "supported_formats": [
                            {
                                "codec": "pcm",
                                "sample_rate": 48000,
                                "bit_depth": 16,
                                "channels": 2,
                            }
                        ],
                        "buffer_capacity": 100_000,
                        "supported_commands": [],
                    },
                },
            }
        ).decode()

        msg = SendspinConnection._deserialize_client_message(raw)  # noqa: SLF001
        assert isinstance(msg, ClientHelloMessage)
        assert msg.payload.player_support is not None

    def test_family_order_prefers_first_custom_role_and_requires_matching_support_key(self) -> None:
        """When v2 is unregistered, parser falls back to first registered role in family."""
        raw = orjson.dumps(
            {
                "type": "client/hello",
                "payload": {
                    "client_id": "c1",
                    "name": "Client",
                    "version": 1,
                    "supported_roles": ["player@v2", "player@v1"],
                    "player@v1_support": {
                        "supported_formats": [
                            {
                                "codec": "pcm",
                                "sample_rate": 48000,
                                "bit_depth": 16,
                                "channels": 2,
                            }
                        ],
                        "buffer_capacity": 100_000,
                        "supported_commands": [],
                    },
                },
            }
        ).decode()

        msg = SendspinConnection._deserialize_client_message(raw)  # noqa: SLF001
        assert isinstance(msg, ClientHelloMessage)
        assert msg.payload.player_support is not None

    def test_family_order_prefers_registered_v2_and_requires_matching_support_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When v2 is registered and listed first, parser requires v2 support key."""
        monkeypatch.setitem(ROLE_FACTORIES, "player@v2", lambda _client: None)  # type: ignore[arg-type]

        raw = orjson.dumps(
            {
                "type": "client/hello",
                "payload": {
                    "client_id": "c1",
                    "name": "Client",
                    "version": 1,
                    "supported_roles": ["player@v2", "player@v1"],
                    "player@v1_support": {
                        "supported_formats": [
                            {
                                "codec": "pcm",
                                "sample_rate": 48000,
                                "bit_depth": 16,
                                "channels": 2,
                            }
                        ],
                        "buffer_capacity": 100_000,
                        "supported_commands": [],
                    },
                },
            }
        ).decode()

        with pytest.raises(ValueError, match="player@v2_support"):
            SendspinConnection._deserialize_client_message(raw)  # noqa: SLF001

    def test_legacy_family_support_key_used_for_custom_role(self) -> None:
        """A client that sends the legacy <family>_support key still binds for custom roles."""
        raw = orjson.dumps(
            {
                "type": "client/hello",
                "payload": {
                    "client_id": "c1",
                    "name": "Client",
                    "version": 1,
                    "supported_roles": ["visualizer@v1", "visualizer@_custom_legacy"],
                    "visualizer_support": {
                        "types": ["loudness"],
                        "buffer_capacity": 65_536,
                        "rate_max": 30,
                    },
                },
            }
        ).decode()

        msg = SendspinConnection._deserialize_client_message(raw)  # noqa: SLF001
        assert isinstance(msg, ClientHelloMessage)
        assert msg.payload.visualizer_support is not None

    def test_unknown_future_visualizer_version_ignored_with_schema_drift(self) -> None:
        """Unregistered spec-versioned roles must not be parsed against the family schema.

        Reproduces the forward-compat trap: a client running a future protocol
        version sends `visualizer@v99_support` whose schema diverges from the
        registered v1 spec. The server has no role registered for `v99` so per
        spec it must ignore the role and proceed. The current implementation
        greedy-parses against the v1 schema and raises MissingField.
        """
        raw = orjson.dumps(
            {
                "type": "client/hello",
                "payload": {
                    "client_id": "c1",
                    "name": "Client",
                    "version": 1,
                    "supported_roles": ["visualizer@v99"],
                    "visualizer@v99_support": {
                        "buffer_capacity": 65536,
                        "spectrum": {
                            "n_disp_bins": 48,
                            "scale": "mel",
                            "f_min": 20,
                            "f_max": 20000,
                            "rate_max": 30,
                        },
                    },
                },
            }
        ).decode()

        msg = SendspinConnection._deserialize_client_message(raw)  # noqa: SLF001
        assert isinstance(msg, ClientHelloMessage)
        assert msg.payload.visualizer_support is None

    def test_unknown_future_player_version_ignored_with_schema_drift(self) -> None:
        """A future player@vN support payload missing v1's required fields must not crash."""
        raw = orjson.dumps(
            {
                "type": "client/hello",
                "payload": {
                    "client_id": "c1",
                    "name": "Client",
                    "version": 1,
                    "supported_roles": ["player@v99"],
                    "player@v99_support": {"buffer_capacity": 100_000},
                },
            }
        ).decode()

        msg = SendspinConnection._deserialize_client_message(raw)  # noqa: SLF001
        assert isinstance(msg, ClientHelloMessage)
        assert msg.payload.player_support is None

    def test_unknown_future_artwork_version_ignored_with_schema_drift(self) -> None:
        """A future artwork@vN support payload missing v1's required fields must not crash."""
        raw = orjson.dumps(
            {
                "type": "client/hello",
                "payload": {
                    "client_id": "c1",
                    "name": "Client",
                    "version": 1,
                    "supported_roles": ["artwork@v99"],
                    "artwork@v99_support": {"some_new_field": "value"},
                },
            }
        ).decode()

        msg = SendspinConnection._deserialize_client_message(raw)  # noqa: SLF001
        assert isinstance(msg, ClientHelloMessage)
        assert msg.payload.artwork_support is None

    def test_hello_with_draft_r1_only_parses_and_negotiates(self) -> None:
        """Legacy `visualizer@_draft_r1` clients are still fully supported.

        Wire-format round-trip: deserialize the hello, then run negotiation and
        confirm the legacy role is the one that gets activated.
        """
        raw = orjson.dumps(
            {
                "type": "client/hello",
                "payload": {
                    "client_id": "c1",
                    "name": "Client",
                    "version": 1,
                    "supported_roles": ["visualizer@_draft_r1"],
                    "visualizer@_draft_r1_support": {
                        "types": ["loudness"],
                        "buffer_capacity": 65_536,
                        "spectrum": {
                            "n_disp_bins": 48,
                            "scale": "mel",
                            "f_min": 20,
                            "f_max": 20000,
                            "rate_max": 30,
                        },
                    },
                },
            }
        ).decode()

        msg = SendspinConnection._deserialize_client_message(raw)  # noqa: SLF001
        assert isinstance(msg, ClientHelloMessage)
        assert msg.payload.visualizer_draft_r1_support is not None
        assert msg.payload.visualizer_support is None
        assert "visualizer@_draft_r1" in negotiate_roles(msg.payload.supported_roles)

    def test_hello_with_v2_and_v1_mixed_falls_back_to_v1(self) -> None:
        """When client offers `[v2, v1]` and server knows only v1, v1 is activated.

        Reaches the architecture limit gracefully: client lists a newer protocol
        version first, but the server picks the highest version it actually
        registers. The unknown v2 support payload must be ignored, not parsed.
        """
        raw = orjson.dumps(
            {
                "type": "client/hello",
                "payload": {
                    "client_id": "c1",
                    "name": "Client",
                    "version": 1,
                    "supported_roles": ["visualizer@v2", "visualizer@v1"],
                    "visualizer@v2_support": {"completely": "different schema"},
                    "visualizer@v1_support": {
                        "buffer_capacity": 65_536,
                        "rate_max": 30,
                        "types": ["loudness"],
                    },
                },
            }
        ).decode()

        msg = SendspinConnection._deserialize_client_message(raw)  # noqa: SLF001
        assert isinstance(msg, ClientHelloMessage)
        assert msg.payload.visualizer_support is not None
        assert negotiate_roles(msg.payload.supported_roles) == ["visualizer@v1"]

    def test_hello_with_only_v2_drops_visualizer_role_entirely(self) -> None:
        """Lone unsupported version → deserialize succeeds and family is inactive.

        Spec: server should ignore unknown roles. Active_roles must reflect this
        so the client can detect outdated servers and degrade gracefully.
        """
        raw = orjson.dumps(
            {
                "type": "client/hello",
                "payload": {
                    "client_id": "c1",
                    "name": "Client",
                    "version": 1,
                    "supported_roles": ["visualizer@v2"],
                    "visualizer@v2_support": {"unknown_field": 1},
                },
            }
        ).decode()

        msg = SendspinConnection._deserialize_client_message(raw)  # noqa: SLF001
        assert isinstance(msg, ClientHelloMessage)
        assert msg.payload.visualizer_support is None
        assert msg.payload.visualizer_draft_r1_support is None
        assert negotiate_roles(msg.payload.supported_roles) == []

    @pytest.mark.parametrize(
        ("supported_roles", "expected"),
        [
            (["controller@v2"], ["controller@v2"]),  # unknown version of a known family
            (["foobar@v1"], ["foobar@v1"]),  # unknown family
            (["controller@v1", "metadata@v1"], []),  # implemented
            (["_custom@v1", "player@_draft"], []),  # custom family / version excluded
            (["player@v1", "controller@v2", "foobar@v3"], ["controller@v2", "foobar@v3"]),
        ],
    )
    def test_unimplemented_roles_flags_unknown_non_custom(
        self, supported_roles: list[str], expected: list[str]
    ) -> None:
        """Roles/versions the server does not implement are tracked, excluding `_` custom ones."""
        assert SendspinConnection._unimplemented_roles(supported_roles) == expected  # noqa: SLF001

    def test_hello_with_brand_new_family_does_not_crash(self) -> None:
        """A family the server has never heard of is silently ignored end-to-end.

        Guards against future role additions on the client side that the server
        doesn't know about yet.
        """
        raw = orjson.dumps(
            {
                "type": "client/hello",
                "payload": {
                    "client_id": "c1",
                    "name": "Client",
                    "version": 1,
                    "supported_roles": ["crystalball@v1"],
                    "crystalball@v1_support": {"forecast": "cloudy"},
                },
            }
        ).decode()

        msg = SendspinConnection._deserialize_client_message(raw)  # noqa: SLF001
        assert isinstance(msg, ClientHelloMessage)
        assert negotiate_roles(msg.payload.supported_roles) == []

    def test_custom_underscore_player_version_with_v1_compatible_support_parses(self) -> None:
        """`player@_experimental` (no registered factory) parses against v1 schema.

        PairingCodes the implicit contract: an underscore-prefixed custom version of a
        spec family keeps its support payload v1-compatible since the family's
        registered schema is what the deserializer uses. The custom role
        implementer owns both ends, so this trade-off is intentional.
        """
        raw = orjson.dumps(
            {
                "type": "client/hello",
                "payload": {
                    "client_id": "c1",
                    "name": "Client",
                    "version": 1,
                    "supported_roles": ["player@_experimental"],
                    "player@_experimental_support": {
                        "supported_formats": [
                            {
                                "codec": "pcm",
                                "channels": 2,
                                "sample_rate": 48000,
                                "bit_depth": 16,
                            },
                        ],
                        "buffer_capacity": 100_000,
                        "supported_commands": [],
                    },
                },
            }
        ).decode()

        msg = SendspinConnection._deserialize_client_message(raw)  # noqa: SLF001
        assert isinstance(msg, ClientHelloMessage)
        assert msg.payload.player_support is not None
        assert msg.payload.player_support.buffer_capacity == 100_000

    def test_custom_underscore_player_version_with_incompatible_support_raises(self) -> None:
        """`player@_experimental` with schema-incompatible support raises.

        Documents the implicit contract from the previous test: when an
        underscore-prefixed custom version is the only role in its family, the
        family's registered schema gets applied. Diverging from that schema is
        the implementer's responsibility — the server cannot guess otherwise.
        """
        raw = orjson.dumps(
            {
                "type": "client/hello",
                "payload": {
                    "client_id": "c1",
                    "name": "Client",
                    "version": 1,
                    "supported_roles": ["player@_experimental"],
                    "player@_experimental_support": {"only": "garbage"},
                },
            }
        ).decode()

        with pytest.raises(MissingField):
            SendspinConnection._deserialize_client_message(raw)  # noqa: SLF001


class TestClientUrlRegistration:
    """Tests that client URLs are registered after successful handshake."""

    @pytest.mark.asyncio
    async def test_url_registered_after_handshake(self, mock_server: _MockServer) -> None:
        """Client URL is registered after the hello exchange for a server-initiated connection."""
        url = "ws://192.168.1.100:8927/sendspin"
        conn = SendspinConnection(mock_server, wsock_client=AsyncMock(), url=url)

        await _exchange_hellos_encrypted(conn, client_id="my-speaker")

        assert mock_server.get_client_url("my-speaker") == url

    @pytest.mark.asyncio
    async def test_url_not_registered_for_client_initiated(self, mock_server: _MockServer) -> None:
        """Client URL is NOT registered for client-initiated connections (no URL to store)."""
        request = MagicMock(spec=web.Request)
        request.remote = "192.168.1.50"

        conn = SendspinConnection(mock_server, request=request)
        conn._wsock_server = AsyncMock()  # noqa: SLF001
        conn._wsock_server.closed = False  # noqa: SLF001

        await _exchange_hellos_encrypted(conn, client_id="my-speaker")

        # No URL should be registered since we don't know the client's WebSocket URL
        assert mock_server.get_client_url("my-speaker") is None


@dataclass
class _MockServerWithReclaim:
    """Mock server that tracks reclaim calls."""

    loop: asyncio.AbstractEventLoop
    clock: LoopClock
    id: str = "srv"
    name: str = "server"

    _reclaim_calls: list[str] = field(default_factory=list)

    def request_client_playback_connection(self, client_id: str) -> bool:
        self._reclaim_calls.append(client_id)
        return True

    def is_external_player(self, client_id: str) -> bool:  # noqa: ARG002
        return False

    def _signal_client_connected(self, client_id: str) -> None:
        pass

    def _signal_client_disconnected(self, client_id: str, goodbye_reason: object) -> None:
        pass


class TestAutomaticReclaim:
    """Tests for automatic client reclaim on playback start and group join."""

    @pytest.mark.asyncio
    async def test_start_stream_reclaims_disconnected_clients(self) -> None:
        """start_stream() reclaims disconnected clients in the group."""
        loop = asyncio.get_running_loop()
        server = _MockServerWithReclaim(loop=loop, clock=LoopClock(loop))

        # Create a client and group
        client = SendspinClient(server, client_id="speaker-1")
        group = SendspinGroup(server, client)

        # Connect and then disconnect the client (simulating another_server goodbye)
        conn = _DummyConnection()
        client.attach_connection(
            conn,
            client_info=_player_hello("speaker-1"),
            negotiated_roles=[Roles.PLAYER.value],
            active_roles=[Roles.PLAYER.value],
        )
        client.mark_connected()

        # Store URL for reclaim
        server._client_urls = {"speaker-1": "ws://192.168.1.50:8927/sendspin"}  # type: ignore[attr-defined]  # noqa: SLF001

        # Disconnect the client
        client.detach_connection(None)
        assert not client.is_connected

        # Start a stream - should trigger reclaim
        group.start_stream()

        assert "speaker-1" in server._reclaim_calls  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_add_client_reclaims_if_group_has_active_playback(self) -> None:
        """add_client() reclaims disconnected client when group has active playback."""
        loop = asyncio.get_running_loop()
        server = _MockServerWithReclaim(loop=loop, clock=LoopClock(loop))

        # Create two clients - one connected, one disconnected
        client1 = SendspinClient(server, client_id="speaker-1")
        client2 = SendspinClient(server, client_id="speaker-2")
        group1 = SendspinGroup(server, client1)
        SendspinGroup(server, client2)

        # Connect client1 and start playback
        conn1 = _DummyConnection()
        client1.attach_connection(
            conn1,
            client_info=_player_hello("speaker-1"),
            negotiated_roles=[Roles.PLAYER.value],
            active_roles=[Roles.PLAYER.value],
        )
        client1.mark_connected()
        group1.start_stream()

        # Clear reclaim calls from start_stream (client1 was connected)
        server._reclaim_calls.clear()  # noqa: SLF001

        # Connect and disconnect client2
        conn2 = _DummyConnection()
        client2.attach_connection(
            conn2,
            client_info=_player_hello("speaker-2"),
            negotiated_roles=[Roles.PLAYER.value],
            active_roles=[Roles.PLAYER.value],
        )
        client2.mark_connected()
        client2.detach_connection(None)
        assert not client2.is_connected

        # Store URL for reclaim
        server._client_urls = {"speaker-2": "ws://192.168.1.51:8927/sendspin"}  # type: ignore[attr-defined]  # noqa: SLF001

        # Add disconnected client2 to group1 (which has active playback)
        await group1.add_client(client2)

        assert "speaker-2" in server._reclaim_calls  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_add_client_does_not_reclaim_if_connected(self) -> None:
        """add_client() does not reclaim if client is already connected."""
        loop = asyncio.get_running_loop()
        server = _MockServerWithReclaim(loop=loop, clock=LoopClock(loop))

        # Create two clients
        client1 = SendspinClient(server, client_id="speaker-1")
        client2 = SendspinClient(server, client_id="speaker-2")
        group1 = SendspinGroup(server, client1)
        SendspinGroup(server, client2)

        # Connect both clients
        conn1 = _DummyConnection()
        client1.attach_connection(
            conn1,
            client_info=_player_hello("speaker-1"),
            negotiated_roles=[Roles.PLAYER.value],
            active_roles=[Roles.PLAYER.value],
        )
        client1.mark_connected()

        conn2 = _DummyConnection()
        client2.attach_connection(
            conn2,
            client_info=_player_hello("speaker-2"),
            negotiated_roles=[Roles.PLAYER.value],
            active_roles=[Roles.PLAYER.value],
        )
        client2.mark_connected()

        # Start playback on group1
        group1.start_stream()
        server._reclaim_calls.clear()  # noqa: SLF001

        # Add connected client2 to group1
        await group1.add_client(client2)

        # Should not try to reclaim since client2 is connected
        assert "speaker-2" not in server._reclaim_calls  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_add_client_does_not_reclaim_if_no_active_playback(self) -> None:
        """add_client() does not reclaim if group has no active playback."""
        loop = asyncio.get_running_loop()
        server = _MockServerWithReclaim(loop=loop, clock=LoopClock(loop))

        # Create two clients
        client1 = SendspinClient(server, client_id="speaker-1")
        client2 = SendspinClient(server, client_id="speaker-2")
        group1 = SendspinGroup(server, client1)
        SendspinGroup(server, client2)

        # Connect client1 but don't start playback
        conn1 = _DummyConnection()
        client1.attach_connection(
            conn1,
            client_info=_player_hello("speaker-1"),
            negotiated_roles=[Roles.PLAYER.value],
            active_roles=[Roles.PLAYER.value],
        )
        client1.mark_connected()

        # Connect and disconnect client2
        conn2 = _DummyConnection()
        client2.attach_connection(
            conn2,
            client_info=_player_hello("speaker-2"),
            negotiated_roles=[Roles.PLAYER.value],
            active_roles=[Roles.PLAYER.value],
        )
        client2.mark_connected()
        client2.detach_connection(None)

        # Add disconnected client2 to group1 (no active playback)
        await group1.add_client(client2)

        # Should not try to reclaim since no active playback
        assert "speaker-2" not in server._reclaim_calls  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_add_client_replaces_stale_client_with_same_client_id(self) -> None:
        """add_client() replaces stale client objects that share the same client_id."""
        loop = asyncio.get_running_loop()
        server = _MockServerWithReclaim(loop=loop, clock=LoopClock(loop))

        owner = SendspinClient(server, client_id="speaker-1")
        stale = SendspinClient(server, client_id="speaker-2")
        replacement = SendspinClient(server, client_id="speaker-2")

        group1 = SendspinGroup(server, owner, stale)
        SendspinGroup(server, replacement)

        await group1.add_client(replacement)

        # Group membership should contain only the replacement object for speaker-2.
        speaker2_members = [c for c in group1.clients if c.client_id == "speaker-2"]
        assert speaker2_members == [replacement]
        assert stale not in group1.clients
