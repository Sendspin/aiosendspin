"""Represents a single client device connected to the server."""

import asyncio
import logging
from collections.abc import Callable, Coroutine
from contextlib import suppress
from enum import Enum
from typing import TYPE_CHECKING, Any, cast

from aiohttp import ClientWebSocketResponse, WSMessage, WSMsgType, web

from aiosendspin.models import unpack_binary_header
from aiosendspin.models.core import (
    ClientCommandMessage,
    ClientGoodbyeMessage,
    ClientHelloMessage,
    ClientHelloPayload,
    ClientStateMessage,
    ClientTimeMessage,
    GroupUpdateServerMessage,
    GroupUpdateServerPayload,
    ServerHelloMessage,
    ServerHelloPayload,
    ServerTimeMessage,
    ServerTimePayload,
    StreamEndMessage,
    StreamRequestFormatMessage,
)
from aiosendspin.models.types import (
    BinaryMessageType,
    ClientMessage,
    ClientStateType,
    ConnectionReason,
    GoodbyeReason,
    PlaybackStateType,
    Roles,
    ServerMessage,
)

from .controller import ControllerClient
from .events import ClientEvent, ClientGroupChangedEvent
from .group import SendspinGroup
from .metadata import MetadataClient
from .player import PlayerClient
from .visualizer import VisualizerClient

MAX_PENDING_MSG = 4096


logger = logging.getLogger(__name__)

# The cyclic import is not an issue during runtime, so hide it
# pyright: reportImportCycles=none
if TYPE_CHECKING:
    from .server import SendspinServer


class DisconnectBehaviour(Enum):
    """Enum for disconnect behaviour options."""

    UNGROUP = "ungroup"
    """
    The client will ungroup itself from its current group when it gets disconnected.

    Playback will continue on the remaining group members.
    """
    STOP = "stop"
    """
    The client will stop playback of the whole group when it gets disconnected.
    """


class SendspinClient:
    """
    A Client that is connected to a SendspinServer.

    Playback is handled through groups, use Client.group to get the
    assigned group.
    """

    _server: "SendspinServer"
    """Reference to the SendspinServer instance this client belongs to."""
    _wsock_client: ClientWebSocketResponse | None = None
    """
    WebSocket connection from the server to the client.

    This is only set for server-initiated connections.
    """
    _wsock_server: web.WebSocketResponse | None = None
    """
    WebSocket connection from the client to the server.

    This is only set for client-initiated connections.
    """
    _request: web.Request | None = None
    """
    Web Request used for client-initiated connections.

    This is only set for client-initiated connections.
    """
    _client_id: str | None = None
    _client_info: ClientHelloPayload | None = None
    _writer_task: asyncio.Task[None] | None = None
    """Task responsible for sending JSON and binary data."""
    _to_write: asyncio.Queue[ServerMessage | bytes]
    """Queue for messages to be sent to the client through the WebSocket."""
    _group: SendspinGroup
    _event_cbs: list[Callable[["SendspinClient", ClientEvent], Coroutine[None, None, None]]]
    _closing: bool = False
    _disconnecting: bool = False
    """Flag to prevent multiple concurrent disconnect tasks."""
    _server_hello_sent: bool = False
    """Flag to track if server/hello has been sent to complete handshake."""
    _initial_state_received: bool = False
    """Flag to track if initial client/state has been received (for roles that require it)."""
    _initial_state_timeout_handle: asyncio.TimerHandle | None = None
    """Timeout handle for initial state reception (for roles that require it)."""
    disconnect_behaviour: DisconnectBehaviour
    """
    Controls the disconnect behavior for this client.

    UNGROUP (default): Client leaves its current group but playback continues
        on remaining group members.
    STOP: Client stops playback for the entire group when disconnecting.
    """
    _handle_client_connect: Callable[["SendspinClient"], Coroutine[Any, Any, None]]
    _handle_client_disconnect: Callable[["SendspinClient"], None]
    _logger: logging.Logger
    _roles: list[Roles]
    _player: PlayerClient | None = None
    _controller: ControllerClient | None = None
    _metadata_client: MetadataClient | None = None
    _visualizer: VisualizerClient | None = None
    _client_state: ClientStateType
    """Current operational state of the client."""
    _previous_group_id: str | None = None
    """Group ID to rejoin after external_source ends (for switch command priority)."""
    _external_source_solo_group_id: str | None = None
    """Solo group ID created by an external_source transition."""
    _stream_start_time_us: int | None = None
    """Timestamp when first audio chunk was sent, for grace period on timing warnings."""

    def __init__(
        self,
        server: "SendspinServer",
        handle_client_connect: Callable[["SendspinClient"], Coroutine[Any, Any, None]],
        handle_client_disconnect: Callable[["SendspinClient"], None],
        request: web.Request | None = None,
        wsock_client: ClientWebSocketResponse | None = None,
    ) -> None:
        """
        DO NOT CALL THIS CONSTRUCTOR. INTERNAL USE ONLY.

        Use SendspinServer.on_client_connect or SendspinServer.connect_to_client instead.

        Args:
            server: The SendspinServer instance this client belongs to.
            handle_client_connect: Callback function called when the client's handshake is complete.
            handle_client_disconnect: Callback function called when the client disconnects.
            request: Optional web request object for client-initiated connections.
                Only one of request or wsock_client must be provided.
            wsock_client: Optional client WebSocket response for server-initiated connections.
                Only one of request or wsock_client must be provided.
        """
        self._server = server
        self._handle_client_connect = handle_client_connect
        self._handle_client_disconnect = handle_client_disconnect
        if request is not None:
            assert wsock_client is None
            self._request = request
            self._wsock_server = web.WebSocketResponse(heartbeat=55)
            self._logger = logger.getChild(f"unknown-{self._request.remote}")
            self._logger.debug("Client initialized")
        elif wsock_client is not None:
            assert request is None
            self._logger = logger.getChild("unknown-client")
            self._wsock_client = wsock_client
        else:
            raise ValueError("Either request or wsock_client must be provided")
        self._to_write = asyncio.Queue(maxsize=MAX_PENDING_MSG)
        self._event_cbs = []
        self._closing = False
        self._disconnecting = False
        self._server_hello_sent = False
        self._initial_state_received = False
        self._initial_state_timeout_handle = None
        self._roles = []
        self._client_state = ClientStateType.SYNCHRONIZED
        self._previous_group_id = None
        self._external_source_solo_group_id = None
        self.disconnect_behaviour = DisconnectBehaviour.UNGROUP
        self._set_group(SendspinGroup(server, self))

    async def disconnect(self, *, retry_connection: bool = True) -> None:
        """Disconnect this client from the server."""
        if not retry_connection:
            self._closing = True
        self._disconnecting = True
        self._logger.debug("Disconnecting client")

        if self.disconnect_behaviour == DisconnectBehaviour.UNGROUP:
            await self.ungroup()
            # Try to stop playback if we were playing alone before disconnecting
            await self.group.stop()
        elif self.disconnect_behaviour == DisconnectBehaviour.STOP:
            await self.group.stop()
            await self.ungroup()

        # Cancel running tasks
        if self._writer_task and not self._writer_task.done():
            self._logger.debug("Cancelling writer task")
            self._writer_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._writer_task
        # Handle task is cancelled implicitly when wsock closes or externally

        # Close WebSocket
        if self._wsock_client is not None and not self._wsock_client.closed:
            await self._wsock_client.close()
        elif self._wsock_server is not None and not self._wsock_server.closed:
            await self._wsock_server.close()

        if self._client_id is not None:
            self._handle_client_disconnect(self)

        self._logger.info("Client disconnected")

    @property
    def group(self) -> SendspinGroup:
        """Get the group assigned to this client."""
        return self._group

    @property
    def client_id(self) -> str:
        """The unique identifier of this Client."""
        # This should only be called once the client was correctly initialized
        assert self._client_id
        return self._client_id

    @property
    def name(self) -> str:
        """The human-readable name of this Client."""
        assert self._client_info  # Client should be fully initialized by now
        return self._client_info.name

    @property
    def info(self) -> ClientHelloPayload:
        """List of information and capabilities reported by this client."""
        assert self._client_info  # Client should be fully initialized by now
        return self._client_info

    @property
    def websocket_connection(self) -> web.WebSocketResponse | ClientWebSocketResponse:
        """
        Returns the active WebSocket connection for this client.

        This provides access to the underlying WebSocket connection, which can be
        either a server-side WebSocketResponse (for client-initiated connections)
        or a ClientWebSocketResponse (for server-initiated connections).
        """
        wsock = self._wsock_server or self._wsock_client
        assert wsock is not None
        return wsock

    @property
    def closing(self) -> bool:
        """Whether this player is in the process of closing/disconnecting."""
        return self._closing

    @property
    def is_server_initiated(self) -> bool:
        """True if the server initiated this connection (connected to the client)."""
        return self._wsock_client is not None

    @property
    def roles(self) -> list[Roles]:
        """List of roles this client supports."""
        return self._roles

    def check_role(self, role: Roles) -> bool:
        """Check if the client supports a specific role."""
        return role in self._roles

    def _ensure_role(self, role: Roles) -> None:
        """Raise a ValueError if the client does not support a specific role."""
        if role not in self._roles:
            raise ValueError(f"Client does not support role: {role}")

    @property
    def player(self) -> PlayerClient | None:
        """Return the attached player instance, if available."""
        if self._player is None and Roles.PLAYER in self._roles:
            self._player = PlayerClient(self)
        return self._player

    @property
    def require_player(self) -> PlayerClient:
        """Return the player or raise if the role is unsupported."""
        if self._player is None:
            raise ValueError(f"Client does not support role: {Roles.PLAYER}")
        return self._player

    @property
    def controller(self) -> ControllerClient | None:
        """Return the controller role helper, if initialized."""
        return self._controller

    @property
    def require_controller(self) -> ControllerClient:
        """Return controller helper or raise if role unsupported."""
        if self._controller is None:
            raise ValueError(f"Client does not support role: {Roles.CONTROLLER}")
        return self._controller

    @property
    def metadata(self) -> MetadataClient | None:
        """Return the metadata role helper, if initialized."""
        return self._metadata_client

    @property
    def require_metadata(self) -> MetadataClient:
        """Return metadata helper or raise if role unsupported."""
        if self._metadata_client is None:
            raise ValueError(f"Client does not support role: {Roles.METADATA}")
        return self._metadata_client

    @property
    def visualizer(self) -> VisualizerClient | None:
        """Return the visualizer role helper, if initialized."""
        return self._visualizer

    @property
    def require_visualizer(self) -> VisualizerClient:
        """Return visualizer helper or raise if role unsupported."""
        if self._visualizer is None:
            raise ValueError(f"Client does not support role: {Roles.VISUALIZER}")
        return self._visualizer

    def requires_initial_state(self) -> bool:
        """Check if this client's roles require sending initial state."""
        return Roles.PLAYER in self._roles

    def _initial_state_timeout_callback(self) -> None:
        """
        Handle initial state timeout.

        Logs a warning and disconnects the client for spec violation.
        """
        if self._initial_state_received:
            # State was received just as timeout fired
            return

        self._logger.warning(
            "Client %s failed to send required initial state within timeout (spec violation)",
            self._client_id or "unknown",
        )
        # Disconnect and allow retry in case of transient network issues
        task = self._server.loop.create_task(self.disconnect(retry_connection=True))
        task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

    def _set_group(self, group: "SendspinGroup") -> None:
        """
        Set the group for this client. For internal use by SendspinGroup only.

        NOTE: this does not update the group's client list

        Args:
            group: The SendspinGroup to assign this client to.
        """
        if hasattr(self, "_group"):
            # If we are leaving the solo group created by an external_source transition via any
            # means other than the recovery switch logic, clear the stored previous group.
            if (
                self._external_source_solo_group_id is not None
                and self._group.group_id == self._external_source_solo_group_id
                and group.group_id != self._external_source_solo_group_id
            ):
                self._previous_group_id = None
                self._external_source_solo_group_id = None
            # Don't unregister if this is the initial setup in __init__
            self._group._unregister_client_events(self)  # noqa: SLF001

        self._group = group

        # Register event listeners with new group
        self._group._register_client_events(self)  # noqa: SLF001

        # Emit event for group change
        self._signal_event(ClientGroupChangedEvent(group))

    async def ungroup(self) -> None:
        """
        Remove the client from the group.

        If the client is already alone, this function does nothing.
        """
        if len(self._group.clients) > 1:
            self._logger.debug("Ungrouping client from group")
            await self._group.remove_client(self)
        else:
            self._logger.debug("Client already alone in group, no ungrouping needed")

    async def _setup_connection(self) -> None:
        """Establish WebSocket connection."""
        if self._wsock_server is not None:
            assert self._request is not None
            try:
                async with asyncio.timeout(10):
                    # Prepare response, writer not needed
                    await self._wsock_server.prepare(self._request)
            except TimeoutError:
                self._logger.warning("Timeout preparing request")
                raise

        self._logger.info("Connection established")

        self._logger.debug("Creating writer task")
        self._writer_task = self._server.loop.create_task(self._writer())
        # server/hello will be sent after receiving client/hello

    async def _run_message_loop(self) -> None:
        """Run the main message processing loop."""
        wsock = self._wsock_server or self._wsock_client
        assert wsock is not None
        receive_task: asyncio.Task[WSMessage] | None = None
        # Listen for all incoming messages
        try:
            while not wsock.closed:
                # Wait for either a message or the writer task to complete (meaning the client
                # disconnected or errored)
                receive_task = self._server.loop.create_task(wsock.receive())
                assert self._writer_task is not None  # for type checking
                done, pending = await asyncio.wait(
                    [receive_task, self._writer_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if self._writer_task in done:
                    self._logger.debug("Writer task ended, closing connection")
                    # Cancel the receive task if it's still pending
                    if receive_task in pending:
                        receive_task.cancel()
                    break

                # Get the message from the completed receive task
                try:
                    msg = await receive_task
                except (ConnectionError, asyncio.CancelledError, TimeoutError) as e:
                    self._logger.error("Error receiving message: %s", e)
                    break

                timestamp = int(self._server.loop.time() * 1_000_000)

                if msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED):
                    break

                if msg.type == WSMsgType.BINARY:
                    # Per spec, clients should not send binary messages
                    # Binary messages should be rejected if there is no active stream
                    if not self._group.has_active_stream:
                        self._logger.warning(
                            "Received binary message from client with no active stream, rejecting"
                        )
                    else:
                        self._logger.warning(
                            "Received binary message from client "
                            "(clients should not send binary data)"
                        )
                    continue

                if msg.type != WSMsgType.TEXT:
                    continue

                try:
                    await self._handle_message(
                        ClientMessage.from_json(cast("str", msg.data)), timestamp
                    )
                except Exception:
                    self._logger.exception("error parsing message")
            self._logger.debug("wsock was closed")

        except asyncio.CancelledError:
            self._logger.debug("Connection closed by client")
        except Exception:
            self._logger.exception("Unexpected error inside websocket API")
        finally:
            if receive_task and not receive_task.done():
                receive_task.cancel()

    async def _cleanup_connection(self) -> None:
        """Clean up WebSocket connection and tasks."""
        wsock = self._wsock_client or self._wsock_server
        try:
            if wsock and not wsock.closed:
                await wsock.close()
        except Exception:
            self._logger.exception("Failed to close websocket")
        await self.disconnect()

    async def _handle_client(self) -> None:
        """
        Handle the complete websocket connection lifecycle.

        This method is private and should only be called by SendspinServer
        during client connection handling.
        """
        try:
            # Establish connection and setup
            await self._setup_connection()

            # Run the main message loop
            await self._run_message_loop()
        finally:
            # Clean up connection and tasks
            await self._cleanup_connection()

    async def _handle_message(  # noqa: PLR0915
        self, message: ClientMessage, timestamp: int
    ) -> None:
        """Handle incoming commands from the client."""
        if self._client_info is None and not isinstance(message, ClientHelloMessage):
            raise ValueError("First message must be client/hello")

        # Check that other messages are not sent before server/hello
        if (
            self._client_info is not None
            and not self._server_hello_sent
            and not isinstance(message, ClientHelloMessage)
        ):
            raise ValueError("Client must wait for server/hello before sending other messages")

        match message:
            # Core messages
            case ClientHelloMessage(client_info):
                self._logger.info("Received client/hello")
                if client_info.version != 1:
                    self._logger.error(
                        "Incompatible protocol version %s (only '1' is supported)",
                        client_info.version,
                    )
                    await self.disconnect(retry_connection=False)
                    return

                self._client_info = client_info
                self._roles = client_info.supported_roles
                self._client_id = client_info.client_id
                self._logger.info("Client ID set to %s", self._client_id)
                self._logger = logger.getChild(self._client_id)

                # Initialize role helpers based on supported roles
                if Roles.PLAYER in self._roles:
                    self._player = PlayerClient(self)
                if Roles.CONTROLLER in self._roles:
                    self._controller = ControllerClient(self)
                if Roles.METADATA in self._roles:
                    self._metadata_client = MetadataClient(self)
                if Roles.VISUALIZER in self._roles:
                    self._visualizer = VisualizerClient(self)

                self._logger.debug("Sending server/hello in response to client/hello")
                self.send_message(
                    ServerHelloMessage(
                        payload=ServerHelloPayload(
                            server_id=self._server.id,
                            name=self._server.name,
                            version=1,
                            active_roles=self._roles,
                            connection_reason=ConnectionReason.DISCOVERY,
                        )
                    )
                )
                self._server_hello_sent = True
                # Send initial group/update after handshake
                self._logger.debug("Sending initial group/update after handshake")
                self.send_message(
                    GroupUpdateServerMessage(
                        payload=GroupUpdateServerPayload(
                            playback_state=PlaybackStateType.STOPPED,
                            group_id=self._group.group_id,
                            group_name=self._group.group_name,
                        )
                    )
                )

                # For roles requiring a initial state update per spect,
                # only register the client once we have received it
                if self.requires_initial_state():
                    # Start timeout (5 seconds) for initial state reception
                    self._initial_state_timeout_handle = self._server.loop.call_later(
                        5.0, self._initial_state_timeout_callback
                    )
                else:
                    await self._handle_client_connect(self)
            case ClientTimeMessage(client_time):
                self.send_message(
                    ServerTimeMessage(
                        ServerTimePayload(
                            client_transmitted=client_time.client_transmitted,
                            server_received=timestamp,
                            server_transmitted=int(self._server.loop.time() * 1_000_000),
                        )
                    )
                )
            # Player messages
            case ClientStateMessage(payload):
                # Track initial state reception for roles that require it
                if self.requires_initial_state() and not self._initial_state_received:
                    self._initial_state_received = True
                    self._logger.debug("Received initial client state")

                    # Cancel timeout if set
                    if self._initial_state_timeout_handle:
                        self._initial_state_timeout_handle.cancel()
                        self._initial_state_timeout_handle = None

                    # Now that we have initial state, register client
                    await self._handle_client_connect(self)

                # Handle client-level state (new spec location)
                new_state = payload.state

                # DEPRECATED(before-spec-pr-50): Remove once all clients use client-level state
                # Fall back to player.state for backward compatibility with older clients
                if new_state is None and payload.player is not None:
                    new_state = payload.player.state

                if new_state is not None and new_state != self._client_state:
                    await self._handle_state_transition(new_state)

                if payload.player:
                    self.require_player.handle_player_update(payload.player)
            case StreamRequestFormatMessage(payload):
                await self.group.handle_stream_format_request(self, payload)
            # Controller messages
            case ClientCommandMessage(payload):
                if payload.controller:
                    await self.require_controller.handle_command(payload.controller)
            # Goodbye message (multi-server support)
            case ClientGoodbyeMessage(payload):
                self._logger.info("Received client/goodbye with reason: %s", payload.reason)
                # Per spec: auto-reconnect only for 'restart' reason
                retry = payload.reason == GoodbyeReason.RESTART
                await self.disconnect(retry_connection=retry)

    async def _writer(self) -> None:
        """Write outgoing messages from the queue."""
        # Exceptions if socket disconnected or cancelled by connection handler
        wsock = self._wsock_server or self._wsock_client
        assert wsock is not None
        try:
            while not wsock.closed and not self._closing:
                item = await self._to_write.get()

                if isinstance(item, bytes):
                    # Unpack binary header using helper function
                    header = unpack_binary_header(item)

                    # Only validate timestamps for audio chunks, since they are time-sensitive
                    if header.message_type == BinaryMessageType.AUDIO_CHUNK.value:
                        now = int(self._server.loop.time() * 1_000_000)
                        # Track stream start time for grace period on timing warnings
                        if self._stream_start_time_us is None:
                            self._stream_start_time_us = now
                        # Grace period: skip timing warnings for first 2 seconds after stream start
                        in_grace_period = (now - self._stream_start_time_us) < 2_000_000
                        if header.timestamp_us - now < 0:
                            if not in_grace_period:
                                self._logger.warning(
                                    "Audio chunk should have played already, skipping it"
                                )
                            continue
                        if header.timestamp_us - now < 500_000 and not in_grace_period:
                            self._logger.warning(
                                "sending audio chunk that needs to be played very soon (in %d us)",
                                (header.timestamp_us - now),
                            )
                    try:
                        await wsock.send_bytes(item)
                    except ConnectionError:
                        self._logger.warning(
                            "Connection error sending binary data, ending writer task"
                        )
                        break
                else:
                    assert isinstance(item, ServerMessage)  # for type checking
                    if isinstance(item, ServerTimeMessage):
                        item.payload.server_transmitted = int(self._server.loop.time() * 1_000_000)
                    try:
                        await wsock.send_str(item.to_json())
                    except ConnectionError:
                        self._logger.warning(
                            "Connection error sending JSON data, ending writer task"
                        )
                        break
            self._logger.debug("WebSocket Connection was closed for the client, ending writer task")
        except Exception:
            self._logger.exception("Error in writer task for client")

    def send_message(self, message: ServerMessage | bytes) -> None:
        """
        Enqueue a JSON or binary message to be sent directly to the client.

        It is recommended to not use this method, but to use the higher-level
        API of this library instead.

        NOTE: Binary messages are directly sent to the client, you need to add the
        header yourself using pack_binary_header().
        """
        try:
            self._to_write.put_nowait(message)
        except asyncio.QueueFull:
            # Only trigger disconnect once, even if queue fills repeatedly
            if not self._disconnecting:
                self._logger.error("Message queue full, client too slow - disconnecting")
                task = self._server.loop.create_task(self.disconnect(retry_connection=True))
                task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
            return

        if isinstance(message, bytes):
            pass
        elif isinstance(message, StreamEndMessage):
            # Reset stream start time so next stream gets a fresh grace period
            self._stream_start_time_us = None
        elif not isinstance(message, ServerTimeMessage):
            self._logger.debug("Enqueueing message: %s", type(message).__name__)

    def add_event_listener(
        self, callback: Callable[["SendspinClient", ClientEvent], Coroutine[None, None, None]]
    ) -> Callable[[], None]:
        """
        Register a callback to listen for state changes of this client.

        State changes include:
        - The volume was changed
        - The client joined a group

        Returns a function to remove the listener.
        """
        self._event_cbs.append(callback)

        def _remove() -> None:
            with suppress(ValueError):
                self._event_cbs.remove(callback)

        return _remove

    def _signal_event(self, event: ClientEvent) -> None:
        for cb in self._event_cbs:
            task = self._server.loop.create_task(cb(self, event))
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

    async def _handle_state_transition(self, new_state: ClientStateType) -> None:
        """
        Handle client state transitions.

        When transitioning to external_source:
        - If in multi-client group: remember previous group, move to solo group
        - If already in solo group: stop playback
        """
        old_state = self._client_state
        self._client_state = new_state

        self._logger.info(
            "Client state transition: %s -> %s",
            old_state.value,
            new_state.value,
        )

        if new_state == ClientStateType.EXTERNAL_SOURCE:
            is_multi_client_group = len(self._group.clients) > 1

            if is_multi_client_group:
                # Remember current group for later rejoin via switch command
                self._previous_group_id = self._group.group_id
                self._logger.debug(
                    "Storing previous group %s for external_source client",
                    self._previous_group_id,
                )
                # Move to solo group - remove_client sends stream/end automatically
                await self._group.remove_client(self)
                self._external_source_solo_group_id = self._group.group_id
            else:
                # Already in solo group - just stop playback
                self._logger.debug(
                    "Client already in solo group, stopping playback for external_source"
                )
                await self._group.stop()
