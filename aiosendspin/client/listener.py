"""ClientListener - accepts incoming WebSocket connections from Sendspin servers."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from aiohttp import hdrs, web
from zeroconf import NonUniqueNameException
from zeroconf.asyncio import AsyncServiceInfo, AsyncZeroconf

from aiosendspin.util import get_local_ip

logger = logging.getLogger(__name__)

# Default recommended port per spec
DEFAULT_PORT = 8928
DEFAULT_PATH = "/sendspin"
DEFAULT_HOST = "0.0.0.0"

# Callback type for new server connections.
# Receives the WebSocketResponse and should handle the connection lifecycle.
# The callback should return when the connection is closed.
ConnectionHandler = Callable[[web.WebSocketResponse], Awaitable[None]]


class ClientListener:
    """
    WebSocket server that accepts incoming connections from Sendspin servers.

    This implements the "Server Initiated Connections" pattern from the spec,
    where the client advertises via mDNS and servers connect to it.

    Usage:
        async def handle_connection(ws: web.WebSocketResponse) -> None:
            client = SendspinClient(...)
            await client.attach_websocket(ws)
            # Wait for disconnect
            disconnect_event = asyncio.Event()
            client.add_disconnect_listener(disconnect_event.set)
            await disconnect_event.wait()

        listener = ClientListener(
            client_id="my-client",
            on_connection=handle_connection,
        )
        await listener.start()
    """

    def __init__(
        self,
        client_id: str,
        on_connection: ConnectionHandler,
        *,
        port: int = DEFAULT_PORT,
        path: str = DEFAULT_PATH,
        host: str = DEFAULT_HOST,
        advertise_mdns: bool = True,
        client_name: str | None = None,
    ) -> None:
        """
        Initialize the ClientListener.

        Args:
            client_id: Unique identifier for this client (used for mDNS).
            on_connection: Async callback invoked for each incoming connection.
                The callback receives the WebSocketResponse and should handle
                the full connection lifecycle (attach to SendspinClient, wait
                for disconnect, etc.).
            port: Port to listen on (default: 8928).
            path: WebSocket endpoint path (default: /sendspin).
            host: Host/IP address to bind to (default: 0.0.0.0). Use "127.0.0.1"
                for local-only access.
            advertise_mdns: Whether to advertise via mDNS (default: True).
            client_name: Optional friendly name for this client (advertised via mDNS).
        """
        self._client_id = client_id
        self._on_connection = on_connection
        self._port = port
        self._path = path
        self._host = host
        self._advertise_mdns = advertise_mdns
        self._client_name = client_name

        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None

        self._zc: AsyncZeroconf | None = None
        self._mdns_service: AsyncServiceInfo | None = None

    @property
    def port(self) -> int:
        """Return the port being listened on."""
        return self._port

    def _create_web_application(self) -> web.Application:
        """Create and configure the aiohttp web application."""
        app = web.Application()
        app.router.add_get(self._path, self._handle_websocket)
        return app

    async def start(self) -> None:
        """Start listening for incoming server connections."""
        self._app = self._create_web_application()

        # Start the server
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()

        logger.info("ClientListener started on port %d, path %s", self._port, self._path)

        # Start mDNS advertising
        if self._advertise_mdns:
            await self._start_mdns()

    async def stop(self) -> None:
        """Stop listening and clean up resources."""
        # Stop mDNS
        await self._stop_mdns()

        # Stop web server
        if self._site is not None:
            await self._site.stop()
            self._site = None
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        if self._app is not None:
            await self._app.shutdown()
            self._app = None

        logger.info("ClientListener stopped")

    async def _handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """Handle an incoming WebSocket connection from a server."""
        if hdrs.ORIGIN in request.headers:
            raise web.HTTPForbidden
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        logger.debug("Incoming server connection from %s", request.remote)

        # Delegate to the connection handler
        try:
            await self._on_connection(ws)
        except Exception:
            logger.exception(
                "Unhandled exception in on_connection callback for %s",
                request.remote,
            )
            if not ws.closed:
                await ws.close(code=1011, message=b"Internal error")

        return ws

    async def _start_mdns(self) -> None:
        """Start mDNS advertising for this client."""
        self._zc = AsyncZeroconf()

        # Get local IP address
        local_ip = get_local_ip()
        if local_ip is None:
            logger.warning("Could not determine local IP address, mDNS may not work properly")
            addresses = ["127.0.0.1"]
        else:
            addresses = [local_ip]

        service_type = "_sendspin._tcp.local."
        properties: dict[str, str] = {}
        if self._client_name is not None:
            properties["name"] = self._client_name
        properties["path"] = self._path

        self._mdns_service = AsyncServiceInfo(
            type_=service_type,
            name=f"{self._client_id}.{service_type}",
            server=f"{self._client_id}.local.",
            parsed_addresses=addresses,
            port=self._port,
            properties=properties,
        )

        try:
            await self._zc.async_register_service(self._mdns_service)
            logger.info(
                "mDNS advertising client '%s' on port %d with path %s",
                self._client_id,
                self._port,
                self._path,
            )
        except NonUniqueNameException:
            logger.error("Sendspin client with identical name present in the local network!")

    async def _stop_mdns(self) -> None:
        """Stop mDNS advertising."""
        if self._zc is None:
            return
        try:
            if self._mdns_service is not None:
                await self._zc.async_unregister_service(self._mdns_service)
        finally:
            await self._zc.async_close()
            self._zc = None
            self._mdns_service = None
