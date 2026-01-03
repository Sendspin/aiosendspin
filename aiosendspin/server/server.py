"""Sendspin Server implementation to connect to and manage many Sendspin Clients."""

import asyncio
import logging
import socket
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass

from aiohttp import ClientConnectionError, ClientResponseError, ClientTimeout, ClientWSTimeout, web
from aiohttp.client import ClientSession
from zeroconf import (
    InterfaceChoice,
    IPVersion,
    NonUniqueNameException,
    ServiceStateChange,
    Zeroconf,
)
from zeroconf.asyncio import AsyncServiceBrowser, AsyncServiceInfo, AsyncZeroconf

from .client import SendspinClient

logger = logging.getLogger(__name__)


class SendspinEvent:
    """Base event type used by SendspinServer.add_event_listener()."""


@dataclass
class ClientAddedEvent(SendspinEvent):
    """A new client was added."""

    client_id: str


@dataclass
class ClientRemovedEvent(SendspinEvent):
    """A client disconnected from the server."""

    client_id: str


async def _get_ip_pton(ip_string: str) -> bytes:
    """Return socket pton for a local ip."""
    try:
        return await asyncio.to_thread(socket.inet_pton, socket.AF_INET, ip_string)
    except OSError:
        return await asyncio.to_thread(socket.inet_pton, socket.AF_INET6, ip_string)


class SendspinServer:
    """Sendspin Server implementation to connect to and manage many Sendspin Clients."""

    API_PATH = "/sendspin"  # Fixed by protocol

    _clients: set[SendspinClient]
    """All groups managed by this server."""
    _loop: asyncio.AbstractEventLoop
    _event_cbs: list[Callable[["SendspinServer", SendspinEvent], None]]
    _connection_tasks: dict[str, asyncio.Task[None]]
    """
    All tasks managing client connections.

    This only includes connections initiated via connect_to_client (Server -> Client).
    """
    _retry_events: dict[str, asyncio.Event]
    """
    For each connection task in _connection_tasks, this holds an asyncio.Event.

    This event is used to signal an immediate retry of the connection, in case the connection is
    sleeping during a backoff period.
    """
    _id: str
    _name: str
    _client_session: ClientSession
    """The client session used to connect to clients."""
    _owns_session: bool
    """Whether this server instance owns the client session."""
    _app: web.Application | None
    """
    Web application instance for the server.

    This is used to handle incoming WebSocket connections from clients.
    """
    _app_runner: web.AppRunner | None
    """App runner for the web application."""
    _tcp_site: web.TCPSite | None
    """TCP site for the web application."""
    _zc: AsyncZeroconf | None
    """AsyncZeroconf instance."""
    _mdns_service: AsyncServiceInfo | None
    """Registered mDNS service."""
    _mdns_browser: AsyncServiceBrowser | None
    """mDNS service browser for client discovery."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        server_id: str,
        server_name: str,
        client_session: ClientSession | None = None,
    ) -> None:
        """
        Initialize a new Sendspin Server.

        Args:
            loop: The asyncio event loop to use for asynchronous operations.
            server_id: Unique identifier for this server instance.
            server_name: Human-readable name for this server.
            client_session: Optional ClientSession for outgoing connections.
                If None, a new session will be created.
        """
        self._clients = set()
        self._loop = loop
        self._event_cbs = []
        self._id = server_id
        self._name = server_name
        if client_session is None:
            self._client_session = ClientSession(loop=self._loop, timeout=ClientTimeout(total=30))
            self._owns_session = True
        else:
            self._client_session = client_session
            self._owns_session = False
        self._connection_tasks = {}
        self._retry_events = {}
        self._app = None
        self._app_runner = None
        self._tcp_site = None
        self._zc = None
        self._mdns_service = None
        self._mdns_browser = None
        logger.debug("SendspinServer initialized: id=%s, name=%s", server_id, server_name)

    def _create_web_application(self) -> web.Application:
        """
        Create and configure the aiohttp web application.

        Returns:
            Configured aiohttp web.Application instance.
        """
        app = web.Application()
        app.router.add_get(self.API_PATH, self.on_client_connect)
        return app

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """Read-only access to the event loop used by this server."""
        return self._loop

    async def on_client_connect(self, request: web.Request) -> web.StreamResponse:
        """Handle an incoming WebSocket connection from a Sendspin client."""
        logger.debug("Incoming client connection from %s", request.remote)

        client = SendspinClient(
            self,
            handle_client_connect=self._handle_client_connect,
            handle_client_disconnect=self._handle_client_disconnect,
            request=request,
        )
        await client._handle_client()  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

        websocket = client.websocket_connection
        # This is a WebSocketResponse since we just created client
        # as client-initiated.
        assert isinstance(websocket, web.WebSocketResponse)
        return websocket

    def connect_to_client(self, url: str) -> None:
        """
        Connect to the Sendspin client at the given URL.

        If an active connection already exists for this URL, nothing will happen.
        In case a connection attempt fails, a new connection will be attempted automatically.
        """
        logger.debug("Connecting to client at URL: %s", url)
        prev_task = self._connection_tasks.get(url)
        if prev_task is not None:
            logger.debug("Connection is already active for URL: %s", url)
            # Signal immediate retry if we have a retry event (connection is in backoff)
            if retry_event := self._retry_events.get(url):
                logger.debug("Signaling immediate retry for URL: %s", url)
                retry_event.set()
        else:
            # Create retry event for this connection
            self._retry_events[url] = asyncio.Event()
            self._connection_tasks[url] = self._loop.create_task(
                self._handle_client_connection(url)
            )

    def disconnect_from_client(self, url: str) -> None:
        """
        Disconnect from the Sendspin client that was previously connected at the given URL.

        If no connection was established at this URL, or the connection is already closed,
        this will do nothing.

        NOTE: this will only disconnect connections that were established via connect_to_client.
        """
        connection_task = self._connection_tasks.pop(url, None)
        if connection_task is not None:
            logger.debug("Disconnecting from client at URL: %s", url)
            connection_task.cancel()

    async def _handle_client_connection(self, url: str) -> None:
        """Handle the actual connection to a client."""
        # Exponential backoff settings
        backoff = 1.0
        max_backoff = 300.0  # 5 minutes

        try:
            while True:
                client: SendspinClient | None = None
                retry_event = self._retry_events.get(url)

                try:
                    async with self._client_session.ws_connect(
                        url,
                        heartbeat=30,
                        # Pyright doesn't recognise the signature
                        timeout=ClientWSTimeout(ws_close=10, ws_receive=60),  # pyright: ignore[reportCallIssue]
                    ) as wsock:
                        # Reset backoff on successful connect
                        backoff = 1.0
                        client = SendspinClient(
                            self,
                            handle_client_connect=self._handle_client_connect,
                            handle_client_disconnect=self._handle_client_disconnect,
                            wsock_client=wsock,
                        )
                        await client._handle_client()  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                    if self._client_session.closed:
                        logger.debug("Client session closed, stopping connection task for %s", url)
                        break
                    if client.closing:
                        break
                except asyncio.CancelledError:
                    break
                except TimeoutError:
                    logger.debug("Connection task for %s timed out", url)
                except (ClientConnectionError, ClientResponseError) as err:
                    logger.debug("Connection task for %s failed: %s", url, err)

                if backoff >= max_backoff:
                    break

                logger.debug("Trying to reconnect to client at %s in %.1fs", url, backoff)

                # Use asyncio.wait_for with the retry event to allow immediate retry
                if retry_event is not None:
                    try:
                        # Always returns True when event is set
                        await asyncio.wait_for(retry_event.wait(), timeout=backoff)
                        logger.debug("Immediate retry requested for %s", url)
                        # Clear the event for next time
                        retry_event.clear()
                    except TimeoutError:
                        # Normal timeout, continue with exponential backoff
                        pass
                else:
                    await asyncio.sleep(backoff)

                # Increase backoff for next retry (exponential)
                backoff *= 2
        except asyncio.CancelledError:
            pass
        except Exception:
            # NOTE: Intentional catch-all to log unexpected exceptions so they are visible.
            logger.exception("Unexpected error occurred")
        finally:
            self._connection_tasks.pop(url, None)  # Cleanup connection tasks dict
            self._retry_events.pop(url, None)  # Cleanup retry events dict

    def add_event_listener(
        self, callback: Callable[["SendspinServer", SendspinEvent], None]
    ) -> Callable[[], None]:
        """
        Register a callback to listen for state changes of the server.

        State changes include:
        - A new client was connected
        - A client disconnected

        Returns a function to remove the listener.
        """
        self._event_cbs.append(callback)

        def _remove() -> None:
            with suppress(ValueError):
                self._event_cbs.remove(callback)

        return _remove

    def _signal_event(self, event: SendspinEvent) -> None:
        """Signal an event to all registered listeners."""
        for cb in self._event_cbs:
            try:
                cb(self, event)
            except Exception:
                logger.exception("Error in event listener")

    async def _handle_client_connect(self, client: SendspinClient) -> None:
        """
        Register the client to the server.

        Should only be called once all data like the client id was received.
        If a client with the same ID already exists, the old connection is closed.
        """
        if client in self._clients:
            return

        # Check for existing client with same ID and disconnect it
        if (existing := self.get_client(client.client_id)) is not None:
            logger.info("Client %s reconnected, closing previous connection", client.client_id)
            # Fire removal event before awaiting disconnect to ensure correct event order
            self._handle_client_disconnect(existing)
            try:
                await existing.disconnect(retry_connection=False)
            except Exception:
                logger.exception("Error disconnecting replaced client %s", client.client_id)

        logger.debug("Adding client %s (%s) to server", client.client_id, client.name)
        self._clients.add(client)
        self._signal_event(ClientAddedEvent(client.client_id))

    def _handle_client_disconnect(self, client: SendspinClient) -> None:
        """Unregister the client from the server."""
        if client not in self._clients:
            return

        logger.debug("Removing client %s from server", client.client_id)
        self._clients.remove(client)
        self._signal_event(ClientRemovedEvent(client.client_id))

    @property
    def clients(self) -> set[SendspinClient]:
        """Get the set of all clients connected to this server."""
        return self._clients

    def get_client(self, client_id: str) -> SendspinClient | None:
        """Get the client with the given id."""
        logger.debug("Looking for client with id: %s", client_id)
        for client in self.clients:
            if client.client_id == client_id:
                logger.debug("Found client %s", client_id)
                return client
        logger.debug("Client %s not found", client_id)
        return None

    @property
    def id(self) -> str:
        """Get the unique identifier of this server."""
        return self._id

    @property
    def name(self) -> str:
        """Get the name of this server."""
        return self._name

    async def start_server(
        self,
        port: int = 8927,
        host: str = "0.0.0.0",
        advertise_host: str = "0.0.0.0",
        *,
        discover_clients: bool = True,
    ) -> None:
        """
        Start the Sendspin Server.

        This will start the Sendspin server to connect to clients for both:
        - Client initiated connections: This will advertise this server via mDNS as _sendspin-server
        - Server initiated connections: This will listen for all _sendspin._tcp mDNS services and
          automatically connect to them.

        :param port: The TCP port to bind the server to.
        :param host: The IP address for the server to listen on
            (e.g., "0.0.0.0" for all interfaces).
        :param advertise_host: The IP address to advertise via mDNS.
        :param discover_clients: If True, enable automatic mDNS discovery of clients.
            If False, the server will still advertise itself and accept incoming connections,
            but will not actively connect to discovered clients.
        """
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
            # Start mDNS advertise and discovery
            self._zc = AsyncZeroconf(
                ip_version=IPVersion.V4Only, interfaces=InterfaceChoice.Default
            )
            await self._start_mdns_advertising(host=advertise_host, port=port, path=self.API_PATH)
            if discover_clients:
                await self._start_mdns_discovery()
        except OSError as e:
            logger.error("Failed to start server on %s:%d: %s", host, port, e)
            await self._stop_mdns()
            if self._app_runner:
                await self._app_runner.cleanup()
                self._app_runner = None
            if self._app:
                await self._app.shutdown()
                self._app = None
            raise

    async def stop_server(self) -> None:
        """Stop the HTTP server."""
        await self._stop_mdns()

        if self._tcp_site:
            await self._tcp_site.stop()
            self._tcp_site = None
            logger.debug("TCP site stopped")

        if self._app_runner:
            await self._app_runner.cleanup()
            self._app_runner = None
            logger.debug("App runner cleaned up")

        if self._app:
            await self._app.shutdown()
            self._app = None

    async def close(self) -> None:
        """Close the server and cleanup resources."""
        await self.stop_server()
        # Stop mDNS if active
        await self._stop_mdns()
        if self._owns_session and not self._client_session.closed:
            await self._client_session.close()
            logger.debug("Closed internal client session for server %s", self._name)

    async def _start_mdns_advertising(self, host: str, port: int, path: str) -> None:
        """Start advertising this server via mDNS."""
        assert self._zc is not None
        if self._mdns_service is not None:
            await self._zc.async_unregister_service(self._mdns_service)

        service_type = "_sendspin-server._tcp.local."
        properties = {"path": path}
        info = AsyncServiceInfo(
            type_=service_type,
            name=f"{self._id}.{service_type}",
            addresses=[await _get_ip_pton(host)] if host != "0.0.0.0" else None,
            port=port,
            properties=properties,
        )
        try:
            await self._zc.async_register_service(info)
            self._mdns_service = info
            logger.debug("mDNS advertising server on port %d with path %s", port, path)
        except NonUniqueNameException:
            logger.error("Sendspin server with identical name present in the local network!")

    async def _start_mdns_discovery(self) -> None:
        """Automatically connect to Sendspin clients when discovered via mDNS."""
        assert self._zc is not None

        service_type = "_sendspin._tcp.local."
        self._mdns_browser = AsyncServiceBrowser(
            self._zc.zeroconf,
            service_type,
            handlers=[self._on_mdns_service_state_change],
        )
        logger.debug("mDNS discovery started for clients")

    def _on_mdns_service_state_change(
        self,
        zeroconf: Zeroconf,
        service_type: str,
        name: str,
        state_change: ServiceStateChange,
    ) -> None:
        """Handle mDNS service state callback (called from zeroconf thread)."""
        if state_change in (ServiceStateChange.Added, ServiceStateChange.Updated):

            def _schedule() -> None:
                task = self._loop.create_task(
                    self._handle_service_added(zeroconf, service_type, name)
                )
                task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

            self._loop.call_soon_threadsafe(_schedule)
        # We don't listen on removals since connect_to_client has its own disconnect/retry logic

    async def _handle_service_added(self, zeroconf: Zeroconf, service_type: str, name: str) -> None:
        """Handle a new mDNS service being added."""
        # Get service info asynchronously
        info = AsyncServiceInfo(service_type, name)
        await info.async_request(zeroconf, 3000)

        if not info or not info.parsed_addresses():
            return

        address = info.parsed_addresses()[0]
        port = info.port
        path = None
        if info.properties:
            for k, v in info.properties.items():
                key = k.decode() if isinstance(k, bytes) else k
                if key == "path" and v is not None:
                    path = v.decode() if isinstance(v, bytes) else v
                    break

        if port is None:
            logger.warning("Sendspin client discovered at %s has no port, ignoring", address)
            return
        if path is None or not str(path).startswith("/"):
            logger.warning(
                "Sendspin client discovered at %s:%i has no or invalid path property, ignoring",
                address,
                port,
            )
            return

        url = f"ws://{address}:{port}{path}"
        logger.debug("mDNS discovered client at %s", url)
        self.connect_to_client(url)

    async def _stop_mdns(self) -> None:
        """Stop mDNS advertise and discovery if active."""
        if self._zc is None:
            return
        try:
            if self._mdns_browser is not None:
                # AsyncServiceBrowser cleanup
                await self._mdns_browser.async_cancel()
            if self._mdns_service is not None:
                await self._zc.async_unregister_service(self._mdns_service)
        finally:
            await self._zc.async_close()
            self._zc = None
            self._mdns_service = None
            self._mdns_browser = None
