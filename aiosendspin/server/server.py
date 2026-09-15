"""Sendspin Server implementation to connect to and manage many Sendspin clients."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from ipaddress import ip_address
from urllib.parse import urlsplit

from aiohttp import (
    ClientConnectionError,
    ClientResponseError,
    ClientTimeout,
    ClientWSTimeout,
    web,
)
from aiohttp.client import ClientSession
from zeroconf import (
    InterfaceChoice,
    IPVersion,
    NonUniqueNameException,
    ServiceStateChange,
    Zeroconf,
)
from zeroconf.asyncio import AsyncServiceBrowser, AsyncServiceInfo, AsyncZeroconf

from aiosendspin.clock import Clock, RawMonotonicClock
from aiosendspin.models.core import ClientHelloPayload
from aiosendspin.models.types import ConnectionReason, GoodbyeReason
from aiosendspin.noise.keys import Identity
from aiosendspin.noise.pairing import PairingAbortError, PairingAttempt, PairingTimeoutError
from aiosendspin.noise.trust_store import ServerPairingStore, TrustedUnpairedClient
from aiosendspin.util import create_task, get_local_ip

from .client import SendspinClient
from .connection import SendspinConnection
from .group import SendspinGroup

logger = logging.getLogger(__name__)


# Abort reconnection attempts after exponential backoff reaches this ceiling
MAX_RECONNECT_BACKOFF_S = 300.0
# Only consider a connection stable if it lasts at least this long, otherwise
# a successful but broken session may cause a reconnection every second.
STABLE_SERVER_INITIATED_SESSION_S = 10.0


@dataclass(frozen=True, slots=True)
class _ServerInitiatedConnectionOptions:
    """Retry policy for a server-initiated client URL."""

    retry_initial_connection: bool = False
    retry_indefinitely: bool = False


class SendspinEvent:
    """Base event type used by SendspinServer.add_event_listener()."""


@dataclass
class ClientAddedEvent(SendspinEvent):
    """A new persistent client/device was added."""

    client_id: str


@dataclass
class ClientUpdatedEvent(SendspinEvent):
    """A client's hello payload changed on reconnect."""

    client_id: str


@dataclass
class ClientRemovedEvent(SendspinEvent):
    """A persistent client/device was removed from the server."""

    client_id: str


@dataclass
class ClientConnectedEvent(SendspinEvent):
    """A client established a transport connection (first or reconnect).

    Fired on every ``attach_connection()`` — the first connection of a
    brand-new client and subsequent reconnects alike. Complementary to
    ``ClientDisconnectedEvent`` which fires when the transport goes down.

    .. note::
       This event may fire **before** ``ClientAddedEvent`` for new clients
       (the client is added to the registry later in the attach flow).
       Consumers should not assume the client is already in the registry
       when handling this event.
    """

    client_id: str


@dataclass
class ClientDisconnectedEvent(SendspinEvent):
    """A client's transport connection was lost.

    Fired on every ``detach_connection()`` regardless of goodbye reason.
    The ``goodbye_reason`` is ``None`` for an unexpected disconnect
    (e.g. WebSocket close without protocol goodbye).
    """

    client_id: str
    goodbye_reason: GoodbyeReason | None


@dataclass(frozen=True, slots=True)
class ExternalStreamStartRequest:
    """Request payload for externally managed player connection on stream start."""

    client_id: str
    server: SendspinServer
    connection_reason: ConnectionReason = ConnectionReason.PLAYBACK


ExternalStreamStartCallback = Callable[[ExternalStreamStartRequest], None]


def _get_first_valid_ip(addresses: list[str]) -> str | None:
    """Get the first valid IP address, filtering out link-local and unspecified addresses."""
    for addr_str in addresses:
        try:
            addr = ip_address(addr_str)
        except ValueError:
            continue
        if not addr.is_link_local and not addr.is_unspecified:
            return addr_str
    return None


class SendspinServer:
    """Sendspin Server implementation to connect to and manage many Sendspin clients."""

    API_PATH = "/sendspin"  # Fixed by protocol
    _pending_connections: set[SendspinConnection]
    """Incoming connections that have not finished their handshake/message loop yet."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        identity: Identity,
        server_name: str,
        client_session: ClientSession | None = None,
        *,
        pairing_store: ServerPairingStore,
        allow_unencrypted: bool = False,
        allow_noncompliant_clients: bool = True,
        clock: Clock | None = None,
    ) -> None:
        """Initialize a Sendspin server instance.

        Args:
            loop: Event loop the server runs on.
            identity: The server's long-term Noise identity.
            server_name: Human-readable name advertised to clients.
            client_session: Shared aiohttp session, or None to create and own one.
            pairing_store: Persistent store for pairing records and config.
            allow_unencrypted: Accept legacy unencrypted clients over the non-spec
                transition-mode hello, off by default. Enable it only to bridge
                pre-encryption clients during migration.
            allow_noncompliant_clients: Tolerate and log deviations from clients
                built against pre-1.0 spec drafts when True, reject the client when
                False. Tolerance is transitional and will be removed in a future
                version.
            clock: Clock source, or None for the default monotonic clock.
        """
        self._loop = loop
        self._identity = identity
        self._id = identity.peer_id
        self._name = server_name
        self._pairing_store = pairing_store
        self._allow_unencrypted = allow_unencrypted
        self._allow_noncompliant_clients = allow_noncompliant_clients
        self._clock: Clock = clock or RawMonotonicClock()

        self._clients: dict[str, SendspinClient] = {}
        self._event_cbs: list[Callable[[SendspinServer, SendspinEvent], None]] = []
        # Server-wide toggle for the visualizer `pitch` feature. Off by default:
        # `pitch` rides reserved binary type 21, so enabling it puts a
        # spec-reserved type on the wire and is technically non-compliant. It is
        # kept as an opt-in extension for constrained/experimental setups. Read by
        # VisualizerV1Role when building its stream config.
        self._visualizer_pitch_enabled: bool = False

        if client_session is None:
            self._client_session = ClientSession(loop=self._loop, timeout=ClientTimeout(total=30))
            self._owns_session = True
        else:
            self._client_session = client_session
            self._owns_session = False

        self._connection_tasks: dict[str, asyncio.Task[None]] = {}
        self._retry_events: dict[str, asyncio.Event] = {}
        self._initial_connect_waiters: dict[str, list[asyncio.Future[None]]] = {}
        self._initial_connect_succeeded: set[str] = set()
        self._connection_options: dict[str, _ServerInitiatedConnectionOptions] = {}
        self._connection_reasons: dict[str, ConnectionReason] = {}  # url → reason
        # Pairing intents handed to an already-running dial task, consumed on its next dial.
        self._pending_pairing_attempts: dict[str, PairingAttempt] = {}
        self._client_urls: dict[str, str] = {}  # client_id → url
        self._external_stream_start_cbs: dict[str, ExternalStreamStartCallback] = {}
        self._external_registration_timeouts: dict[str, asyncio.Handle] = {}
        self._reclaim_timeouts: dict[str, asyncio.Handle] = {}
        # Clients whose unregister a one-shot timer deferred to an in-flight reconnect.
        self._deferred_unregister: set[str] = set()
        self._pending_connections = set()

        self._mdns_client_urls: dict[str, str] = {}
        self._app: web.Application | None = None
        self._app_runner: web.AppRunner | None = None
        self._tcp_site: web.TCPSite | None = None
        self._zc: AsyncZeroconf | None = None
        self._mdns_service: AsyncServiceInfo | None = None
        self._mdns_browser: AsyncServiceBrowser | None = None

        logger.debug("SendspinServer initialized: id=%s, name=%s", self._id, server_name)

    def _create_web_application(self) -> web.Application:
        app = web.Application()
        app.router.add_get(self.API_PATH, self.on_client_connect)
        return app

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """Return the asyncio loop used by the server."""
        return self._loop

    @property
    def clock(self) -> Clock:
        """Time source used for timestamping."""
        return self._clock

    @property
    def id(self) -> str:
        """Return the server identifier advertised to clients."""
        return self._id

    @property
    def identity(self) -> Identity:
        """Return the server's static X25519 identity (its public key is the server_id)."""
        return self._identity

    @property
    def pairing_store(self) -> ServerPairingStore:
        """Return the trust store of long-term records the server holds for clients."""
        return self._pairing_store

    @property
    def allow_unencrypted(self) -> bool:
        """Whether transition mode is enabled (accepts legacy unencrypted clients)."""
        return self._allow_unencrypted

    @property
    def allow_noncompliant_clients(self) -> bool:
        """Whether non-spec-compliant clients are tolerated instead of rejected."""
        return self._allow_noncompliant_clients

    @property
    def name(self) -> str:
        """Return the human-readable server name."""
        return self._name

    @property
    def clients(self) -> list[SendspinClient]:
        """All known clients (including disconnected)."""
        return list(self._clients.values())

    @property
    def connected_clients(self) -> list[SendspinClient]:
        """All currently connected clients."""
        return [c for c in self._clients.values() if c.is_connected]

    def get_client(self, client_id: str) -> SendspinClient | None:
        """Get a persistent client device by id, if known."""
        return self._clients.get(client_id)

    @property
    def visualizer_pitch_enabled(self) -> bool:
        """Whether visualizer roles compute the `pitch` feature (default False)."""
        return self._visualizer_pitch_enabled

    def set_visualizer_pitch_enabled(self, *, enabled: bool) -> None:
        """Enable or disable the visualizer `pitch` feature server-wide.

        This option is ignored while `allow_noncompliant_clients` is False: `pitch`
        uses reserved binary type 21, which the spec forbids, so a compliance-strict
        server never emits it regardless of this toggle. It otherwise stays available
        as an opt-in extension. Pitch (YINFFT) is also the heaviest per-frame
        visualizer computation, so leaving it off sheds that cost on constrained
        hardware.
        Toggling drops/adds `pitch` on live roles' negotiated types and re-emits
        `stream/start`; new roles pick the setting up when they connect.
        """
        if enabled == self._visualizer_pitch_enabled:
            return
        self._visualizer_pitch_enabled = enabled
        for client in self._clients.values():
            for role in client.active_roles:
                refresh = getattr(role, "refresh_pitch_setting", None)
                if callable(refresh):
                    refresh()

    def get_or_create_client(self, client_id: str) -> SendspinClient:
        """Get or create a persistent client device by id."""
        client = self._clients.get(client_id)
        if client is None:
            client = SendspinClient(self, client_id=client_id)
            self._clients[client_id] = client
            SendspinGroup(self, client)
        return client

    def on_client_first_connect(self, client_id: str) -> None:
        """Fire ClientAddedEvent when a client completes its first handshake."""
        client = self._clients.get(client_id)
        if client is not None:
            self._fire_client_added_event_once(client)

    async def remove_client(self, client_id: str) -> None:
        """Remove a client from the persistent registry."""
        self._cancel_external_registration_timeout(client_id)
        self._cancel_reclaim_timeout(client_id)
        self._external_stream_start_cbs.pop(client_id, None)
        self._client_urls.pop(client_id, None)

        client = self._clients.pop(client_id, None)
        if client is None:
            return
        await client.group.remove_client(client)
        self._signal_event(ClientRemovedEvent(client_id))

    def register_external_player(
        self,
        hello: ClientHelloPayload,
        *,
        on_stream_start: ExternalStreamStartCallback,
        timeout_s: float = 0.0,
    ) -> SendspinClient:
        """Register an externally orchestrated client identity.

        Use this for clients that do not connect through Sendspin's standard
        discovery/reconnect mechanisms, or that require an external call to
        start their Sendspin client connection.

        Args:
            hello: Client identity/capability payload to preload while disconnected.
                This must match the hello payload sent when the client later
                connects to this server.
            on_stream_start: Callback invoked when playback needs this client to
                connect to the server through its usual external mechanism.
            timeout_s: Optional registration deadline in seconds. If > 0 and the
                client transport does not attach before the deadline, the client
                is fully unregistered from this server.
        """
        if timeout_s < 0:
            raise ValueError("timeout_s must be >= 0")
        if hello.client_id is None:
            raise ValueError("external player hello must include client_id")
        client_id = hello.client_id

        client = self.get_or_create_client(client_id)
        if client.is_connected:
            raise RuntimeError(
                f"Cannot register external player {client_id!r} while client is connected"
            )
        client.preinitialize_client_from_hello(hello)
        self._fire_client_added_event_once(client)
        self._external_stream_start_cbs[client_id] = on_stream_start
        self._cancel_reclaim_timeout(client_id)
        if timeout_s > 0:
            self._schedule_external_registration_timeout(client_id, timeout_s)
        else:
            self._cancel_external_registration_timeout(client_id)
        return client

    def _fire_client_added_event_once(self, client: SendspinClient) -> None:
        """Fire ClientAddedEvent once per persistent client lifetime."""
        if client._added_event_fired:  # noqa: SLF001
            return
        client._added_event_fired = True  # noqa: SLF001
        self._signal_event(ClientAddedEvent(client.client_id))

    def unregister_external_player(self, client_id: str) -> None:
        """Remove external stream-start callback for a registered client."""
        self._cancel_external_registration_timeout(client_id)
        self._external_stream_start_cbs.pop(client_id, None)

    def is_external_player(self, client_id: str) -> bool:
        """Whether this client is externally managed for playback connects."""
        return client_id in self._external_stream_start_cbs

    def on_client_transport_attached(self, client_id: str) -> None:
        """Cancel pending connection-deadline timeouts once transport is attached."""
        self._cancel_external_registration_timeout(client_id)
        self._cancel_reclaim_timeout(client_id)

    def add_event_listener(
        self, callback: Callable[[SendspinServer, SendspinEvent], None]
    ) -> Callable[[], None]:
        """Register a callback for server events and return an unsubscribe callable."""
        self._event_cbs.append(callback)

        def _remove() -> None:
            with suppress(ValueError):
                self._event_cbs.remove(callback)

        return _remove

    def _signal_event(self, event: SendspinEvent) -> None:
        for cb in self._event_cbs:
            try:
                cb(self, event)
            except Exception:
                logger.exception("Error in event listener")

    def _signal_client_updated(self, client_id: str) -> None:
        """Emit a ClientUpdatedEvent (called from SendspinClient)."""
        self._signal_event(ClientUpdatedEvent(client_id))

    def _signal_client_connected(self, client_id: str) -> None:
        """Emit a ClientConnectedEvent (called from SendspinClient)."""
        self._signal_event(ClientConnectedEvent(client_id))

    def _signal_client_disconnected(
        self, client_id: str, goodbye_reason: GoodbyeReason | None
    ) -> None:
        """Emit a ClientDisconnectedEvent (called from SendspinClient)."""
        self._signal_event(ClientDisconnectedEvent(client_id, goodbye_reason))

    async def on_client_connect(self, request: web.Request) -> web.StreamResponse:
        """Handle an incoming WebSocket connection from a Sendspin client."""
        logger.debug("Incoming client connection from %s", request.remote)

        conn = SendspinConnection(self, request=request)
        self._pending_connections.add(conn)
        try:
            await conn.handle_client()
        finally:
            self._pending_connections.discard(conn)

        websocket = conn.websocket_connection
        assert isinstance(websocket, web.WebSocketResponse)
        return websocket

    def connect_to_client(
        self,
        url: str,
        *,
        connection_reason: ConnectionReason = ConnectionReason.DISCOVERY,
        retry_initial_connection: bool = False,
        retry_indefinitely: bool = False,
        pairing_attempt: PairingAttempt | None = None,
    ) -> None:
        """Start a background connection attempt to a client URL.

        By default, initial connection failures are logged and stop the background task,
        and automatic retries only happen after at least one successful connection.
        If mDNS discovery is unavailable, callers can build a full client WebSocket URL
        from a configured hostname/IP, port, and path, then pass
        retry_initial_connection=True and retry_indefinitely=True.

        ``pairing_attempt`` carries an operator-initiated pairing intent for this dial;
        when a dial task already exists it is queued for that task's next dial.
        """
        self._set_connection_options(
            url,
            retry_initial_connection=retry_initial_connection,
            retry_indefinitely=retry_indefinitely,
        )
        self._connection_reasons[url] = connection_reason
        prev_task = self._connection_tasks.get(url)
        if prev_task is not None:
            if pairing_attempt is not None:
                self._pending_pairing_attempts[url] = pairing_attempt
            if retry_event := self._retry_events.get(url):
                retry_event.set()
            return

        self._initial_connect_succeeded.discard(url)
        self._retry_events[url] = asyncio.Event()
        self._connection_tasks[url] = create_task(
            self._handle_client_connection(url, pairing_attempt=pairing_attempt),
            eager_start=False,
        )

    async def connect_to_client_and_wait(
        self,
        url: str,
        *,
        connection_reason: ConnectionReason = ConnectionReason.DISCOVERY,
        retry_initial_connection: bool = False,
        retry_indefinitely: bool = False,
        pairing_attempt: PairingAttempt | None = None,
    ) -> None:
        """Connect to a client and wait for the initial connection attempt.

        Raises:
            ClientConnectionError: If the initial connection to the client fails.
            ClientResponseError: If the client responds with an error HTTP status.
            TimeoutError: If the initial connection attempt times out, or the backoff
                ceiling is reached with retry_initial_connection=True.
            Exception: Other unexpected errors during the initial connection attempt.
        """
        self._set_connection_options(
            url,
            retry_initial_connection=retry_initial_connection,
            retry_indefinitely=retry_indefinitely,
        )
        self._connection_reasons[url] = connection_reason
        if url in self._initial_connect_succeeded:
            return

        waiter: asyncio.Future[None] = self._loop.create_future()
        self._initial_connect_waiters.setdefault(url, []).append(waiter)

        prev_task = self._connection_tasks.get(url)
        if prev_task is not None:
            if pairing_attempt is not None:
                self._pending_pairing_attempts[url] = pairing_attempt
            if retry_event := self._retry_events.get(url):
                retry_event.set()
        else:
            self._initial_connect_succeeded.discard(url)
            self._retry_events[url] = asyncio.Event()
            self._connection_tasks[url] = create_task(
                self._handle_client_connection(url, pairing_attempt=pairing_attempt),
                eager_start=False,
            )

        await waiter

    async def initiate_pairing(self, client_id: str, attempt: PairingAttempt) -> None:
        """Run a pairing attempt on a connected client.

        A pair abort raises and leaves the connection open (retry with another
        ``initiate_pairing`` or drop out with ``end_pairing``); a server-side timeout
        raises with pairing already left; other failures disconnect.
        """
        connection = self._connection_for(client_id)
        try:
            await connection.initiate_pairing(attempt)
        except (PairingAbortError, PairingTimeoutError):
            raise
        except BaseException:
            await connection.disconnect(retry_connection=False)
            raise

    async def end_pairing(self, client_id: str) -> None:
        """End pairing on a connected client without finalizing.

        No-op if not in pairing. Aborts any in-progress attempt with ``user_cancelled``, keeping
        the connection alive.
        If an attempt has already been finalized by the client, it completes as a success instead.
        """
        await self._connection_for(client_id).end_pairing()

    def enable_management(self, client_id: str) -> SendspinConnection:
        """Enable a management session on a connected client and return its connection."""
        connection = self._connection_for(client_id)
        connection.enable_management()
        return connection

    def disable_management(self, client_id: str) -> None:
        """End a client's management session, leaving any playback on the connection intact."""
        self._connection_for(client_id).disable_management()

    async def unpair(self, client_id: str) -> None:
        """Drop the pairing with a connected client: remove our record and tell it to drop its own.

        Raises ``ValueError`` if the client is not currently connected.
        """
        connection = self._connection_for(client_id)
        await self.pairing_store.remove_record(client_id)
        connection.unpair()

    async def trust_unpaired(self, client_id: str) -> None:
        """Approve ``client_id`` for unpaired playback, re-activating it if connected."""
        await self.pairing_store.add_trusted_unpaired(TrustedUnpairedClient(client_id=client_id))
        await self._refresh_trusted_unpaired(client_id)

    async def untrust_unpaired(self, client_id: str) -> None:
        """Revoke ``client_id``'s unpaired-playback approval, re-activating it if connected."""
        await self.pairing_store.remove_trusted_unpaired(client_id)
        await self._refresh_trusted_unpaired(client_id)

    async def _refresh_trusted_unpaired(self, client_id: str) -> None:
        """Re-activate a connected client's roles after a trust change (no-op if offline)."""
        client = self.get_client(client_id)
        connection = client.connection if client is not None else None
        if connection is not None:
            await connection.refresh_trusted_unpaired()

    def _connection_for(self, client_id: str) -> SendspinConnection:
        """Return the connected client's connection, or raise if it is not connected."""
        client = self.get_client(client_id)
        connection = client.connection if client is not None else None
        if connection is None:
            raise ValueError(f"client {client_id} is not connected")
        return connection

    def _set_connection_options(
        self,
        url: str,
        *,
        retry_initial_connection: bool,
        retry_indefinitely: bool,
    ) -> None:
        """Store retry options without downgrading an existing background task."""
        previous = self._connection_options.get(url)
        if previous is None:
            self._connection_options[url] = _ServerInitiatedConnectionOptions(
                retry_initial_connection=retry_initial_connection,
                retry_indefinitely=retry_indefinitely,
            )
            return

        self._connection_options[url] = _ServerInitiatedConnectionOptions(
            retry_initial_connection=(
                previous.retry_initial_connection or retry_initial_connection
            ),
            retry_indefinitely=previous.retry_indefinitely or retry_indefinitely,
        )

    def _get_connection_options(self, url: str) -> _ServerInitiatedConnectionOptions:
        """Return retry options for a server-initiated client URL."""
        return self._connection_options.get(url, _ServerInitiatedConnectionOptions())

    def get_connection_reason(self, url: str) -> ConnectionReason:
        """Get the connection reason for a URL (for use by SendspinConnection)."""
        return self._connection_reasons.get(url, ConnectionReason.DISCOVERY)

    def register_client_url(self, client_id: str, url: str) -> None:
        """Record the URL used to connect to a client.

        If the client was previously registered with a different URL, the
        connection to that stale URL is cancelled.
        """
        previous_url = self._client_urls.get(client_id)
        self._client_urls[client_id] = url
        if previous_url is not None and previous_url != url:
            self.disconnect_from_client(previous_url)

    def get_client_url(self, client_id: str) -> str | None:
        """Get the URL for a client (for reconnection)."""
        return self._client_urls.get(client_id)

    def get_client_id_for_url(self, url: str) -> str | None:
        """Return the unique ``client_id`` known at ``url``, or ``None`` if unknown/ambiguous."""
        matches = [cid for cid, known_url in self._client_urls.items() if known_url == url]
        return matches[0] if len(matches) == 1 else None

    def reclaim_client_for_playback(self, client_id: str, timeout_s: float = 30.0) -> bool:
        """Attempt to reconnect to a client for playback.

        Returns True if reconnection was initiated, False if no URL available.
        Used when starting playback to reclaim clients that disconnected with 'another_server'.

        The client must reconnect within timeout_s seconds; otherwise it is
        fully unregistered from this server.
        """
        if timeout_s < 0:
            raise ValueError("timeout_s must be >= 0")
        url = self._client_urls.get(client_id)
        if url is None:
            return False

        self.connect_to_client(url, connection_reason=ConnectionReason.PLAYBACK)
        if timeout_s > 0:
            self._schedule_reclaim_timeout(client_id, timeout_s)
        else:
            self._cancel_reclaim_timeout(client_id)
        return True

    def request_client_playback_connection(self, client_id: str) -> bool:
        """Request that a disconnected client connect for playback."""
        external_cb = self._external_stream_start_cbs.get(client_id)
        if external_cb is None:
            return self.reclaim_client_for_playback(client_id)

        request = ExternalStreamStartRequest(
            client_id=client_id,
            server=self,
        )
        try:
            external_cb(request)
        except Exception:
            logger.exception(
                "External stream-start callback failed for client_id=%s",
                client_id,
            )
            return False
        return True

    def disconnect_from_client(self, url: str) -> None:
        """Disconnect a server-initiated connection previously established via connect_to_client."""
        self._connection_options.pop(url, None)
        self._connection_reasons.pop(url, None)
        self._initial_connect_succeeded.discard(url)
        connection_task = self._connection_tasks.pop(url, None)
        if connection_task is not None:
            connection_task.cancel()

    def _cancel_external_registration_timeout(self, client_id: str) -> None:
        """Cancel a pending external-registration timeout for a client."""
        handle = self._external_registration_timeouts.pop(client_id, None)
        if handle is not None:
            handle.cancel()

    def _cancel_reclaim_timeout(self, client_id: str) -> None:
        """Cancel a pending reclaim timeout for a client."""
        handle = self._reclaim_timeouts.pop(client_id, None)
        if handle is not None:
            handle.cancel()

    def _cancel_all_timers(self) -> None:
        """Disarm all pending registry timers."""
        for handle in (
            *self._external_registration_timeouts.values(),
            *self._reclaim_timeouts.values(),
        ):
            handle.cancel()
        self._external_registration_timeouts.clear()
        self._reclaim_timeouts.clear()
        for client in self._clients.values():
            client._cancel_cleanup()  # noqa: SLF001

    def _schedule_external_registration_timeout(self, client_id: str, timeout_s: float) -> None:
        """Schedule full unregister if an externally registered client never connects."""
        self._cancel_external_registration_timeout(client_id)
        self._external_registration_timeouts[client_id] = self._loop.call_later(
            timeout_s,
            lambda: create_task(self._expire_external_registration(client_id)),
        )

    def _schedule_reclaim_timeout(self, client_id: str, timeout_s: float) -> None:
        """Schedule full unregister if reclaim never reconnects the client."""
        self._cancel_reclaim_timeout(client_id)
        self._reclaim_timeouts[client_id] = self._loop.call_later(
            timeout_s,
            lambda: create_task(self._expire_reclaim(client_id)),
        )

    async def _expire_external_registration(self, client_id: str) -> None:
        """Handle external-registration timeout."""
        self._external_registration_timeouts.pop(client_id, None)
        await self._full_unregister_disconnected_client(client_id)

    async def _expire_reclaim(self, client_id: str) -> None:
        """Handle reclaim timeout."""
        self._reclaim_timeouts.pop(client_id, None)
        await self._full_unregister_disconnected_client(client_id)

    async def _full_unregister_disconnected_client(self, client_id: str) -> None:
        """Fully unregister a client if it still has no active connection."""
        client = self._clients.get(client_id)
        if client is not None and client.connection is not None:
            return
        # A reconnect mid-handshake: defer removal until the task settles so a
        # failed reconnect is still unregistered instead of lingering forever.
        url = self._client_urls.get(client_id)
        if url is not None and url in self._connection_tasks:
            self._deferred_unregister.add(client_id)
            return
        self._deferred_unregister.discard(client_id)
        self.unregister_external_player(client_id)
        await self.remove_client(client_id)

    async def _complete_deferred_unregister(
        self, url: str, *, connection_succeeded: bool, cancelled: bool
    ) -> None:
        """Finish an unregister deferred while this URL's reconnect task was in flight."""
        client_id = self.get_client_id_for_url(url)
        if client_id is None or client_id not in self._deferred_unregister:
            return
        self._deferred_unregister.discard(client_id)
        # A reconnect that attached (or a shutdown cancellation) is handled elsewhere.
        if connection_succeeded or cancelled:
            return
        await self._full_unregister_disconnected_client(client_id)

    def _resolve_initial_connect_waiters(self, url: str, err: BaseException | None = None) -> None:
        """Resolve or fail waiters for an initial connection attempt."""
        waiters = self._initial_connect_waiters.pop(url, [])
        for waiter in waiters:
            if waiter.done():
                continue
            if err is None:
                waiter.set_result(None)
            else:
                waiter.set_exception(err)

    async def _handle_client_connection(  # noqa: PLR0912, PLR0915
        self, url: str, *, pairing_attempt: PairingAttempt | None = None
    ) -> None:
        """Handle a server-initiated WebSocket connection task."""
        backoff = 1.0
        first_connection_succeeded = False

        try:
            while True:
                retry_event = self._retry_events.get(url)
                try:
                    async with self._client_session.ws_connect(
                        url,
                        heartbeat=30,
                        timeout=ClientWSTimeout(ws_close=10, ws_receive=60),  # pyright: ignore[reportCallIssue]
                    ) as wsock:
                        if not first_connection_succeeded:
                            first_connection_succeeded = True
                            self._initial_connect_succeeded.add(url)
                            self._resolve_initial_connect_waiters(url)
                        connection_started_s = time.monotonic()
                        if pairing_attempt is None:
                            pairing_attempt = self._pending_pairing_attempts.pop(url, None)
                        conn = SendspinConnection(
                            self,
                            wsock_client=wsock,
                            url=url,
                            expected_client_id=self.get_client_id_for_url(url),
                            pairing_attempt=pairing_attempt,
                        )
                        pairing_attempt = None
                        await conn.handle_client()
                        session_duration_s = time.monotonic() - connection_started_s

                    if session_duration_s >= STABLE_SERVER_INITIATED_SESSION_S:
                        backoff = 1.0

                    if not conn.should_retry_server_initiated_connection:
                        reason = conn.goodbye_reason
                        logger.debug(
                            "Not reconnecting to %s (goodbye reason: %s)",
                            url,
                            reason.value if reason is not None else "none",
                        )
                        break

                    if self._client_session.closed:
                        break

                except asyncio.CancelledError:
                    if not first_connection_succeeded:
                        self._resolve_initial_connect_waiters(url, asyncio.CancelledError())
                    break
                except TimeoutError:
                    if not first_connection_succeeded:
                        if self._get_connection_options(url).retry_initial_connection:
                            logger.debug("Initial connection to %s timed out, retrying", url)
                        else:
                            logger.debug("Initial connection to %s timed out", url)
                            self._resolve_initial_connect_waiters(url, TimeoutError())
                            return
                    else:
                        logger.debug("Connection task for %s timed out", url)
                except (ClientConnectionError, ClientResponseError) as err:
                    if not first_connection_succeeded:
                        if self._get_connection_options(url).retry_initial_connection:
                            logger.debug("Initial connection to %s failed, retrying: %s", url, err)
                        else:
                            logger.debug("Initial connection to %s failed: %s", url, err)
                            self._resolve_initial_connect_waiters(url, err)
                            return
                    else:
                        logger.debug("Connection task for %s failed: %s", url, err)

                options = self._get_connection_options(url)
                if not options.retry_indefinitely and backoff >= MAX_RECONNECT_BACKOFF_S:
                    if not first_connection_succeeded:
                        self._resolve_initial_connect_waiters(
                            url,
                            TimeoutError(
                                "Initial connection did not succeed before "
                                "the reconnect backoff ceiling was reached"
                            ),
                        )
                    break

                sleep_s = min(backoff, MAX_RECONNECT_BACKOFF_S)
                logger.debug("Trying to reconnect to client at %s in %.1fs", url, sleep_s)
                if retry_event is not None:
                    try:
                        await asyncio.wait_for(retry_event.wait(), timeout=sleep_s)
                        retry_event.clear()
                    except TimeoutError:
                        pass
                else:
                    await asyncio.sleep(sleep_s)
                backoff = min(backoff * 2, MAX_RECONNECT_BACKOFF_S)
        except asyncio.CancelledError:
            if not first_connection_succeeded:
                self._resolve_initial_connect_waiters(url, asyncio.CancelledError())
        except Exception as err:
            if not first_connection_succeeded:
                self._resolve_initial_connect_waiters(url, err)
            logger.exception("Unexpected error occurred")
        finally:
            self._connection_tasks.pop(url, None)
            self._retry_events.pop(url, None)
            self._pending_pairing_attempts.pop(url, None)
            self._initial_connect_succeeded.discard(url)
            self._connection_options.pop(url, None)
            self._connection_reasons.pop(url, None)
            current = asyncio.current_task()
            await self._complete_deferred_unregister(
                url,
                connection_succeeded=first_connection_succeeded,
                cancelled=current is not None and current.cancelling() > 0,
            )

    async def start_server(
        self,
        port: int = 8927,
        host: str = "0.0.0.0",
        advertise_addresses: list[str] | None = None,
        *,
        discover_clients: bool = True,
    ) -> None:
        """Start the server HTTP listener and (optionally) mDNS discovery/advertising."""
        if self._app is not None:
            logger.warning("Server is already running")
            return

        logger.info("Starting Sendspin server on port %d", port)
        self._app = self._create_web_application()
        self._app_runner = web.AppRunner(self._app)
        await self._app_runner.setup()

        try:
            self._tcp_site = web.TCPSite(
                self._app_runner,
                host=host if host != "0.0.0.0" else None,
                port=port,
            )
            await self._tcp_site.start()
            logger.info("Sendspin server started successfully on %s:%d", host, port)

            self._zc = AsyncZeroconf(
                ip_version=IPVersion.V4Only,
                interfaces=[host] if host != "0.0.0.0" else InterfaceChoice.Default,
            )

            if advertise_addresses is not None:
                addresses = advertise_addresses
            elif local_ip := get_local_ip():
                addresses = [local_ip]
            else:
                addresses = []

            if addresses:
                await self._start_mdns_advertising(
                    addresses=addresses, port=port, path=self.API_PATH
                )
            else:
                logger.warning(
                    "No IP addresses available for mDNS advertising. "
                    "Clients may not be able to discover this server. "
                    "Consider specifying addresses manually via advertise_addresses."
                )

            if discover_clients:
                await self._start_mdns_discovery()
        except OSError as e:
            logger.error("Failed to start server on %s:%d: %s", host, port, e)
            await self._stop_mdns()
            if self._app_runner:
                # cleanup() already stops/unregisters any site added to this runner.
                await self._app_runner.cleanup()
                self._app_runner = None
            self._tcp_site = None
            if self._app:
                await self._app.shutdown()
                self._app = None
            raise

    async def stop_server(self) -> None:
        """Stop the server HTTP listener and mDNS services."""
        await self._stop_mdns()

        if self._tcp_site:
            await self._tcp_site.stop()
            self._tcp_site = None

        if self._app_runner:
            await self._app_runner.cleanup()
            self._app_runner = None

        if self._app:
            await self._app.shutdown()
            self._app = None

    async def close(self) -> None:
        """Close the server and disconnect all active connections."""
        for task in self._connection_tasks.values():
            task.cancel()

        # Close pending incoming connections so their handlers can exit promptly.
        for conn in list(self._pending_connections):
            wsock = conn.websocket_connection
            # An incoming socket not past wsock.prepare() has nothing to close.
            if wsock.closed or (isinstance(wsock, web.WebSocketResponse) and not wsock.prepared):
                continue
            logger.debug("Closing pending client connection")
            try:
                async with asyncio.timeout(1.0):
                    await wsock.close()
            except TimeoutError:
                logger.debug("Timeout while closing pending client websocket")

        disconnect_tasks: list[asyncio.Task[None]] = []
        for client in self._clients.values():
            if client.connection is None:
                continue
            disconnect_tasks.append(
                create_task(client.connection.disconnect(retry_connection=False))
            )
        if disconnect_tasks:
            await asyncio.gather(*disconnect_tasks, return_exceptions=True)

        # Disarm timers after disconnect, which re-arms per-client cleanup.
        self._cancel_all_timers()

        await self.stop_server()
        if self._owns_session and not self._client_session.closed:
            await self._client_session.close()

    async def _start_mdns_advertising(self, addresses: list[str], port: int, path: str) -> None:
        assert self._zc is not None
        if self._mdns_service is not None:
            await self._zc.async_unregister_service(self._mdns_service)

        service_type = "_sendspin-server._tcp.local."
        properties: dict[str, str] = {}
        if self._name:
            properties["name"] = self._name
        properties["path"] = path

        try:
            info = AsyncServiceInfo(
                type_=service_type,
                name=f"{self._id}.{service_type}",
                server=f"{self._id}.local.",
                parsed_addresses=addresses,
                port=port,
                properties=properties,
            )
            await self._zc.async_register_service(info)
            self._mdns_service = info
            logger.debug("mDNS advertising server on port %d with path %s", port, path)
        except NonUniqueNameException:
            logger.error("Sendspin server with identical name present in the local network!")
        except OSError as e:
            # e.g. an advertise address that is not an IP literal; the HTTP listener
            # and client discovery are unaffected, the server just isn't announced
            logger.error(
                "Server is running but not advertised over mDNS (addresses %s): %s",
                addresses,
                e,
            )

    async def _start_mdns_discovery(self) -> None:
        assert self._zc is not None

        service_type = "_sendspin._tcp.local."
        self._mdns_browser = AsyncServiceBrowser(
            self._zc.zeroconf,
            service_type,
            handlers=[self._on_mdns_service_state_change],
        )

    def _on_mdns_service_state_change(
        self,
        zeroconf: Zeroconf,
        service_type: str,
        name: str,
        state_change: ServiceStateChange,
    ) -> None:
        if state_change in (ServiceStateChange.Added, ServiceStateChange.Updated):

            def _schedule_add() -> None:
                create_task(self._handle_service_added(zeroconf, service_type, name))

            self._loop.call_soon_threadsafe(_schedule_add)
        elif state_change is ServiceStateChange.Removed:
            self._loop.call_soon_threadsafe(lambda: self._handle_service_removed(name))

    async def _handle_service_added(self, zeroconf: Zeroconf, service_type: str, name: str) -> None:
        info = AsyncServiceInfo(service_type, name)
        if not info.load_from_cache(zeroconf):
            await info.async_request(zeroconf, 3000)

        addresses = info.parsed_addresses()
        if not addresses:
            return

        address = _get_first_valid_ip(addresses)
        if address is None:
            return

        port = info.port
        path = None
        if info.properties:
            for k, v in info.properties.items():
                key = k.decode() if isinstance(k, bytes) else k
                if key == "path" and v is not None:
                    path = v.decode() if isinstance(v, bytes) else v
                    break

        if port is None:
            return
        if path is None or not str(path).startswith("/"):
            return

        url = f"ws://{address}:{port}{path}"
        old_url = self._mdns_client_urls.get(name)
        if old_url is not None and old_url != url and old_url in self._connection_tasks:
            old_parts = urlsplit(old_url)
            if (
                old_parts.hostname in addresses
                and old_parts.port == port
                and old_parts.path == path
            ):
                logger.debug(
                    "mDNS preferred address changed for %s (%s -> %s) "
                    "but current address still advertised, keeping existing connection",
                    name,
                    old_url,
                    url,
                )
                return
            logger.debug(
                "mDNS address changed for %s (%s -> %s), reconnecting",
                name,
                old_url,
                url,
            )
        self._mdns_client_urls[name] = url
        self.connect_to_client(url)

    def _handle_service_removed(self, name: str) -> None:
        url = self._mdns_client_urls.pop(name, None)
        if url is None:
            return
        # Keep the retry loop alive if a normal client still wants to be reached.
        # mDNS can drop ahead of the device coming back, and the device may not
        # re-announce until something else nudges it (router-side delay, ESP
        # firmware that announces only on cold boot).
        if not self._has_persistent_client_for_url(url):
            self.disconnect_from_client(url)
        create_task(self._cleanup_retained_clients_for_removed_mdns_url(url))

    def _has_persistent_client_for_url(self, url: str) -> bool:
        for client_id, known_url in self._client_urls.items():
            if known_url != url:
                continue
            client = self._clients.get(client_id)
            if client is not None and not client.cleanup_on_mdns_removal:
                return True
        return False

    async def _cleanup_retained_clients_for_removed_mdns_url(self, url: str) -> None:
        """Remove retained ANOTHER_SERVER clients when their mDNS URL disappears."""
        client_ids = [
            client_id for client_id, known_url in self._client_urls.items() if known_url == url
        ]
        for client_id in client_ids:
            client = self._clients.get(client_id)
            # Only forget the URL for clients we're about to remove. Persistent
            # clients keep the URL so the still-running retry task can reconnect.
            if client is None or client.cleanup_on_mdns_removal:
                self._client_urls.pop(client_id, None)

        for client_id in client_ids:
            client = self._clients.get(client_id)
            if client is None:
                continue

            if client.cleanup_on_mdns_removal and not client.is_connected:
                await self.remove_client(client_id)

    async def _stop_mdns(self) -> None:
        if self._zc is None:
            return
        try:
            if self._mdns_browser is not None:
                await self._mdns_browser.async_cancel()
            if self._mdns_service is not None:
                await self._zc.async_unregister_service(self._mdns_service)
        finally:
            await self._zc.async_close()
            self._zc = None
            self._mdns_service = None
            self._mdns_browser = None
