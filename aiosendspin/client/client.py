"""Sendspin Client implementation to connect to a Sendspin Server."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass

from aiohttp import ClientSession, ClientWebSocketResponse, WSMessage, WSMsgType

from aiosendspin.models import BINARY_HEADER_SIZE, BinaryMessageType, unpack_binary_header
from aiosendspin.models.artwork import ClientHelloArtworkSupport
from aiosendspin.models.controller import ControllerCommandPayload
from aiosendspin.models.core import (
    ClientCommandMessage,
    ClientCommandPayload,
    ClientHelloMessage,
    ClientHelloPayload,
    ClientStateMessage,
    ClientStatePayload,
    ClientTimeMessage,
    ClientTimePayload,
    DeviceInfo,
    GroupUpdateServerMessage,
    GroupUpdateServerPayload,
    ServerCommandMessage,
    ServerCommandPayload,
    ServerHelloMessage,
    ServerHelloPayload,
    ServerStateMessage,
    ServerStatePayload,
    ServerTimeMessage,
    ServerTimePayload,
    StreamClearMessage,
    StreamEndMessage,
    StreamStartMessage,
)
from aiosendspin.models.player import (
    ClientHelloPlayerSupport,
    PlayerStatePayload,
    StreamStartPlayer,
)
from aiosendspin.models.types import AudioCodec, MediaCommand, PlayerStateType, Roles, ServerMessage

from .time_sync import SendspinTimeFilter

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PCMFormat:
    """PCM audio format description."""

    sample_rate: int
    """Sample rate in Hz (e.g., 48000, 44100)."""
    channels: int
    """Number of audio channels (1=mono, 2=stereo)."""
    bit_depth: int
    """Bits per sample (e.g., 16, 24, 32)."""

    def __post_init__(self) -> None:
        """Validate the provided PCM audio format."""
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.channels not in (1, 2):
            raise ValueError("channels must be 1 or 2")
        if self.bit_depth not in (16, 24, 32):
            raise ValueError("bit_depth must be 16, 24, or 32")

    @property
    def frame_size(self) -> int:
        """Return bytes per PCM frame."""
        return self.channels * (self.bit_depth // 8)


# Callback invoked when server state metadata updates are received.
MetadataCallback = Callable[[ServerStatePayload], None]

# Callback invoked when group state updates are received.
GroupUpdateCallback = Callable[[GroupUpdateServerPayload], None]

# Callback invoked when controller state updates are received.
ControllerStateCallback = Callable[[ServerStatePayload], None]

# Callback invoked when audio streaming begins.
StreamStartCallback = Callable[[StreamStartMessage], None]

# Callback invoked when audio streaming ends.
# Receives list of roles to end, or None if all roles should be ended.
StreamEndCallback = Callable[[list[Roles] | None], None]

# Callback invoked when stream buffers should be cleared (e.g., seek operation).
# Receives list of roles to clear, or None if all roles should be cleared.
StreamClearCallback = Callable[[list[Roles] | None], None]

# Callback invoked with (server_timestamp_us, audio_data, format) when audio chunks arrive.
AudioChunkCallback = Callable[[int, bytes, PCMFormat], None]

# Callback invoked when the client disconnects from the server.
DisconnectCallback = Callable[[], None]

# Callback invoked when server sends player commands (volume, mute).
ServerCommandCallback = Callable[[ServerCommandPayload], None]


@dataclass(slots=True)
class ServerInfo:
    """Information about the connected server."""

    server_id: str
    name: str
    version: int


class SendspinClient:
    """
    Async Sendspin client for handling playback and metadata.

    The client must be created within an async context and requires explicit
    role specification. Player and metadata support configs are required if
    their respective roles are enabled.
    """

    _client_id: str
    """Unique identifier for this client."""
    _client_name: str
    """Human-readable name for this client."""
    _device_info: DeviceInfo | None
    """Optional device information."""
    _roles: list[Roles]
    """List of roles this client supports."""
    _player_support: ClientHelloPlayerSupport | None
    """Player capabilities (only set if PLAYER role is supported)."""
    _artwork_support: ClientHelloArtworkSupport | None
    """Artwork capabilities (only set if ARTWORK role is supported)."""
    _session: ClientSession | None
    """Optional aiohttp ClientSession for WebSocket connection."""

    _loop: asyncio.AbstractEventLoop
    """Event loop for this client."""
    _ws: ClientWebSocketResponse | None = None
    """WebSocket connection to the server."""
    _owns_session: bool
    """Whether this client owns and should close the session."""
    _connected: bool = False
    """Whether the client is currently connected."""
    _server_info: ServerInfo | None = None
    """Information about the connected server."""
    _server_hello_event: asyncio.Event | None = None
    """Event signaled when server hello is received."""

    _reader_task: asyncio.Task[None] | None = None
    """Background task reading messages from server."""
    _time_task: asyncio.Task[None] | None = None
    """Background task for time synchronization."""

    _static_delay_us: int = 0
    """Static playback delay in microseconds."""
    _send_lock: asyncio.Lock
    """Lock for serializing WebSocket message sends."""
    _time_filter: SendspinTimeFilter
    """Kalman filter for time synchronization."""

    _current_player: StreamStartPlayer | None = None
    """Current active player configuration."""
    _current_pcm_format: PCMFormat | None = None
    """Current PCM audio format for active stream."""
    _stream_active: bool = False
    """True if stream is active (stream/start received, stream/end not yet received)."""

    _group_state: GroupUpdateServerPayload | None = None
    """Latest group state received from server."""
    _server_state: ServerStatePayload | None = None
    """Latest server state received from server."""

    _metadata_callbacks: list[MetadataCallback]
    """Callbacks invoked on server/state messages with metadata."""
    _group_callbacks: list[GroupUpdateCallback]
    """Callbacks invoked on group/update messages."""
    _controller_callbacks: list[ControllerStateCallback]
    """Callbacks invoked on server/state messages."""
    _stream_start_callbacks: list[StreamStartCallback]
    """Callbacks invoked when a stream starts."""
    _stream_end_callbacks: list[StreamEndCallback]
    """Callbacks invoked when a stream ends."""
    _stream_clear_callbacks: list[StreamClearCallback]
    """Callbacks invoked when stream buffers should be cleared."""
    _audio_chunk_callbacks: list[AudioChunkCallback]
    """Callbacks invoked when audio chunks are received."""
    _disconnect_callbacks: list[DisconnectCallback]
    """Callbacks invoked when the client disconnects."""
    _server_command_callbacks: list[ServerCommandCallback]
    """Callbacks invoked when server sends player commands."""

    _initial_volume: int
    """Initial volume level for player role (0-100)."""
    _initial_muted: bool
    """Initial mute state for player role."""

    def __init__(
        self,
        client_id: str,
        client_name: str,
        roles: Sequence[Roles],
        *,
        device_info: DeviceInfo | None = None,
        player_support: ClientHelloPlayerSupport | None = None,
        artwork_support: ClientHelloArtworkSupport | None = None,
        session: ClientSession | None = None,
        static_delay_ms: float = 0.0,
        initial_volume: int = 100,
        initial_muted: bool = False,
    ) -> None:
        """
        Create a new Sendspin client instance.

        Args:
            client_id: Unique identifier for this client.
            client_name: Human-readable name for this client.
            roles: Sequence of roles this client supports. Must include PLAYER
                if player_support is provided; must include ARTWORK if
                artwork_support is provided.
            device_info: Optional device information (product name, manufacturer,
                software version).
            player_support: Custom player capabilities. Required if PLAYER role
                is specified; raises ValueError if missing.
            artwork_support: Custom artwork capabilities. Required if ARTWORK
                role is specified; raises ValueError if missing.
            session: Optional aiohttp ClientSession. If None, a session is created
                and managed by this client.
            static_delay_ms: Static playback delay in milliseconds applied after
                clock synchronization. Defaults to 0.0.
            initial_volume: Initial volume level (0-100) for player role.
                Defaults to 100. Sent automatically after handshake if PLAYER
                role is supported.
            initial_muted: Initial mute state for player role. Defaults to False.
                Sent automatically after handshake if PLAYER role is supported.

        Raises:
            ValueError: If PLAYER in roles but player_support is None, or if
                ARTWORK in roles but artwork_support is None.
        """
        self._client_id = client_id
        self._client_name = client_name
        self._device_info = device_info
        self._roles = list(roles)

        # Validate and store player support
        if Roles.PLAYER in self._roles:
            if player_support is None:
                raise ValueError("player_support is required when PLAYER role is specified")
            self._player_support = player_support
        else:
            self._player_support = None

        # Validate and store artwork support
        if Roles.ARTWORK in self._roles:
            if artwork_support is None:
                raise ValueError("artwork_support is required when ARTWORK role is specified")
            self._artwork_support = artwork_support
        else:
            self._artwork_support = None
        self._session = session
        self._owns_session = session is None
        self._loop = asyncio.get_running_loop()
        self._send_lock = asyncio.Lock()
        self._time_filter = SendspinTimeFilter()
        self._initial_volume = initial_volume
        self._initial_muted = initial_muted
        self.set_static_delay_ms(static_delay_ms)

        # Initialize callback lists
        self._metadata_callbacks = []
        self._group_callbacks = []
        self._controller_callbacks = []
        self._stream_start_callbacks = []
        self._stream_end_callbacks = []
        self._stream_clear_callbacks = []
        self._audio_chunk_callbacks = []
        self._disconnect_callbacks = []
        self._server_command_callbacks = []

    @property
    def server_info(self) -> ServerInfo | None:
        """Return information about the connected server, if available."""
        return self._server_info

    @property
    def connected(self) -> bool:
        """Return True if the client currently has an active connection."""
        return self._connected and self._ws is not None and not self._ws.closed

    @property
    def static_delay_ms(self) -> float:
        """Return the currently configured static playback delay in milliseconds."""
        return self._static_delay_us / 1_000.0

    def set_static_delay_ms(self, delay_ms: float) -> None:
        """Update the static playback delay applied after clock synchronisation."""
        delay_us = round(delay_ms * 1_000.0)
        if delay_us == self._static_delay_us:
            return
        self._static_delay_us = delay_us
        logger.info("Set static playback delay to %.1f ms", self.static_delay_ms)

    async def connect(self, url: str) -> None:
        """Connect to a Sendspin server via WebSocket."""
        if self.connected:
            logger.debug("Already connected")
            return

        if self._session is None:
            self._session = ClientSession()
        self._server_hello_event = asyncio.Event()

        logger.info("Connecting to Sendspin server at %s", url)
        self._ws = await self._session.ws_connect(url, heartbeat=30)
        self._connected = True

        self._reader_task = self._loop.create_task(self._reader_loop())
        await self._send_client_hello()

        try:
            await asyncio.wait_for(self._server_hello_event.wait(), timeout=10)
        except TimeoutError as err:
            await self.disconnect()
            raise TimeoutError("Timed out waiting for server/hello response") from err

        # Send initial player state if player role is supported
        if Roles.PLAYER in self._roles:
            await self.send_player_state(
                state=PlayerStateType.SYNCHRONIZED,
                volume=self._initial_volume,
                muted=self._initial_muted,
            )

        await self._send_time_message()
        self._time_task = self._loop.create_task(self._time_sync_loop())
        logger.info("Handshake with server complete")

    async def disconnect(self) -> None:
        """Disconnect from the server and release resources."""
        self._connected = False
        current_task = asyncio.current_task(loop=self._loop)

        if self._time_task is not None and self._time_task is not current_task:
            self._time_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._time_task
            self._time_task = None
        if self._reader_task is not None:
            if self._reader_task is not current_task:
                self._reader_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._reader_task
            self._reader_task = None
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None
        self._time_filter.reset()
        self._server_info = None
        self._server_hello_event = None
        self._group_state = None
        self._server_state = None
        self._stream_active = False
        self._current_pcm_format = None
        self._current_player = None

        # Notify disconnect callback
        self._notify_disconnect_callback()

    async def send_player_state(
        self,
        *,
        state: PlayerStateType,
        volume: int,
        muted: bool,
    ) -> None:
        """Send the current player state to the server."""
        if not self.connected:
            raise RuntimeError("Client is not connected")
        message = ClientStateMessage(
            payload=ClientStatePayload(
                player=PlayerStatePayload(state=state, volume=volume, muted=muted)
            )
        )
        await self._send_message(message.to_json())

    async def send_group_command(
        self,
        command: MediaCommand,
        *,
        volume: int | None = None,
        mute: bool | None = None,
    ) -> None:
        """Send a group command (playback control) to the server."""
        if not self.connected:
            raise RuntimeError("Client is not connected")
        controller_payload = ControllerCommandPayload(command=command, volume=volume, mute=mute)
        payload = ClientCommandPayload(controller=controller_payload)
        message = ClientCommandMessage(payload=payload)
        await self._send_message(message.to_json())

    def add_metadata_listener(self, callback: MetadataCallback) -> Callable[[], None]:
        """Add a listener for server/state messages with metadata.

        Returns:
            A function that removes this listener when called.
        """
        self._metadata_callbacks.append(callback)
        return lambda: (
            self._metadata_callbacks.remove(callback)
            if callback in self._metadata_callbacks
            else None
        )

    def add_group_update_listener(self, callback: GroupUpdateCallback) -> Callable[[], None]:
        """Add a listener for group/update messages.

        Returns:
            A function that removes this listener when called.
        """
        self._group_callbacks.append(callback)
        return lambda: (
            self._group_callbacks.remove(callback) if callback in self._group_callbacks else None
        )

    def add_controller_state_listener(
        self, callback: ControllerStateCallback
    ) -> Callable[[], None]:
        """Add a listener for server/state messages.

        Returns:
            A function that removes this listener when called.
        """
        self._controller_callbacks.append(callback)
        return lambda: (
            self._controller_callbacks.remove(callback)
            if callback in self._controller_callbacks
            else None
        )

    def add_stream_start_listener(self, callback: StreamStartCallback) -> Callable[[], None]:
        """Add a listener for stream start events.

        Returns:
            A function that removes this listener when called.
        """
        self._stream_start_callbacks.append(callback)
        return lambda: (
            self._stream_start_callbacks.remove(callback)
            if callback in self._stream_start_callbacks
            else None
        )

    def add_stream_end_listener(self, callback: StreamEndCallback) -> Callable[[], None]:
        """Add a listener for stream end events.

        Returns:
            A function that removes this listener when called.
        """
        self._stream_end_callbacks.append(callback)
        return lambda: (
            self._stream_end_callbacks.remove(callback)
            if callback in self._stream_end_callbacks
            else None
        )

    def add_stream_clear_listener(self, callback: StreamClearCallback) -> Callable[[], None]:
        """Add a listener for stream clear events.

        Returns:
            A function that removes this listener when called.
        """
        self._stream_clear_callbacks.append(callback)
        return lambda: (
            self._stream_clear_callbacks.remove(callback)
            if callback in self._stream_clear_callbacks
            else None
        )

    def add_audio_chunk_listener(self, callback: AudioChunkCallback) -> Callable[[], None]:
        """Add a listener for audio chunk events.

        The callback receives:
        - server_timestamp_us: Server timestamp when this audio should play
        - audio_data: Raw PCM audio bytes
        - format: PCMFormat describing the audio format

        To convert server timestamps to client play time (monotonic loop time),
        use the compute_play_time() and compute_server_time() methods provided
        by this client instance. These handle time synchronization and static delay
        automatically.

        Returns:
            A function that removes this listener when called.
        """
        self._audio_chunk_callbacks.append(callback)
        return lambda: (
            self._audio_chunk_callbacks.remove(callback)
            if callback in self._audio_chunk_callbacks
            else None
        )

    def add_disconnect_listener(self, callback: DisconnectCallback) -> Callable[[], None]:
        """Add a listener for disconnect events.

        Returns:
            A function that removes this listener when called.
        """
        self._disconnect_callbacks.append(callback)
        return lambda: (
            self._disconnect_callbacks.remove(callback)
            if callback in self._disconnect_callbacks
            else None
        )

    def add_server_command_listener(self, callback: ServerCommandCallback) -> Callable[[], None]:
        """Add a listener for server command events.

        Returns:
            A function that removes this listener when called.
        """
        self._server_command_callbacks.append(callback)
        return lambda: (
            self._server_command_callbacks.remove(callback)
            if callback in self._server_command_callbacks
            else None
        )

    def is_time_synchronized(self) -> bool:
        """Return whether time synchronization with the server has converged."""
        return self._time_filter.is_synchronized

    def _build_client_hello(self) -> ClientHelloMessage:
        payload = ClientHelloPayload(
            client_id=self._client_id,
            name=self._client_name,
            version=1,
            supported_roles=self._roles,
            device_info=self._device_info,
            player_support=self._player_support,
            artwork_support=self._artwork_support,
        )
        return ClientHelloMessage(payload=payload)

    async def _send_client_hello(self) -> None:
        hello = self._build_client_hello()
        await self._send_message(hello.to_json())

    async def _send_time_message(self) -> None:
        if not self.connected:
            return
        now_us = self._now_us()
        message = ClientTimeMessage(payload=ClientTimePayload(client_transmitted=now_us))
        await self._send_message(message.to_json())

    async def _send_message(self, payload: str) -> None:
        if not self._ws:
            raise RuntimeError("WebSocket is not connected")
        async with self._send_lock:
            await self._ws.send_str(payload)

    async def _reader_loop(self) -> None:
        assert self._ws is not None
        try:
            async for msg in self._ws:
                await self._handle_ws_message(msg)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("WebSocket reader encountered an error")
        finally:
            if self._connected:
                await self.disconnect()

    async def _handle_ws_message(self, msg: WSMessage) -> None:
        if msg.type is WSMsgType.TEXT:
            await self._handle_json_message(msg.data)
        elif msg.type is WSMsgType.BINARY:
            self._handle_binary_message(msg.data)
        elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED):
            logger.info("WebSocket closed by server")
            await self.disconnect()
        elif msg.type is WSMsgType.ERROR:
            logger.error("WebSocket error: %s", self._ws.exception() if self._ws else "unknown")
            await self.disconnect()

    async def _handle_json_message(self, data: str) -> None:
        try:
            message = ServerMessage.from_json(data)
        except Exception:
            logger.exception("Failed to parse server message: %s", data)
            return

        match message:
            case ServerHelloMessage(payload=payload):
                self._handle_server_hello(payload)
            case ServerTimeMessage(payload=payload):
                self._handle_server_time(payload)
            case StreamStartMessage():
                await self._handle_stream_start(message)
            case StreamClearMessage():
                self._handle_stream_clear(message)
            case StreamEndMessage():
                self._handle_stream_end(message)
            case GroupUpdateServerMessage(payload=payload):
                self._handle_group_update(payload)
            case ServerStateMessage(payload=payload):
                self._handle_server_state(payload)
            case ServerCommandMessage(payload=payload):
                self._handle_server_command(payload)
            case _:
                logger.debug("Unhandled server message type: %s", type(message).__name__)

    def _handle_binary_message(self, payload: bytes) -> None:
        try:
            header = unpack_binary_header(payload)
        except Exception:
            logger.exception("Failed to unpack binary header")
            return

        try:
            message_type = BinaryMessageType(header.message_type)
        except ValueError:
            logger.warning("Unknown binary message type: %s", header.message_type)
            return

        if not self._stream_active:
            logger.warning(
                "Ignoring binary message of type %s since no stream is active", message_type
            )
            return

        if message_type is BinaryMessageType.AUDIO_CHUNK:
            self._handle_audio_chunk(header.timestamp_us, payload[BINARY_HEADER_SIZE:])
        else:
            logger.debug("Ignoring unsupported binary message type: %s", message_type)

    def _handle_server_hello(self, payload: ServerHelloPayload) -> None:
        self._server_info = ServerInfo(
            server_id=payload.server_id,
            name=payload.name,
            version=payload.version,
        )
        if self._server_hello_event:
            self._server_hello_event.set()
        logger.info(
            "Connected to server '%s' (%s) version %s",
            payload.name,
            payload.server_id,
            payload.version,
        )

    def _handle_server_time(self, payload: ServerTimePayload) -> None:
        now_us = self._now_us()
        offset = (
            (payload.server_received - payload.client_transmitted)
            + (payload.server_transmitted - now_us)
        ) / 2
        delay = (
            (now_us - payload.client_transmitted)
            - (payload.server_transmitted - payload.server_received)
        ) / 2
        self._time_filter.update(round(offset), round(delay), now_us)

    async def _handle_stream_start(self, message: StreamStartMessage) -> None:
        player = message.payload.player
        if player is None:
            # stream/start without player payload - may be for artwork/visualizer only
            logger.debug("Stream start message without player payload")
            return

        if player.codec != AudioCodec.PCM:
            logger.error("Unsupported codec '%s' - only PCM is supported", player.codec.value)
            return

        is_format_update = self._stream_active and self._current_player is not None
        if is_format_update:
            logger.info("Stream format updated")
        else:
            logger.info("Stream started")
            self._stream_active = True

        pcm_format = PCMFormat(
            sample_rate=player.sample_rate,
            channels=player.channels,
            bit_depth=player.bit_depth,
        )
        self._configure_audio_output(pcm_format)
        self._current_player = StreamStartPlayer(
            codec=player.codec,
            sample_rate=player.sample_rate,
            channels=player.channels,
            bit_depth=player.bit_depth,
            codec_header=player.codec_header,
        )

        if not is_format_update:
            self._notify_stream_start(message)
            await self._send_time_message()

    def _handle_stream_clear(self, message: StreamClearMessage) -> None:
        roles = message.payload.roles
        logger.info("Stream clear received for roles: %s", roles or "all")
        self._notify_stream_clear(roles)

    def _handle_stream_end(self, message: StreamEndMessage) -> None:
        roles = message.payload.roles
        logger.info("Stream ended for roles: %s", roles or "all")

        # If roles is None or includes player role, end the player stream
        if roles is None or Roles.PLAYER in roles:
            self._stream_active = False
            self._current_player = None
            self._current_pcm_format = None

        self._notify_stream_end(roles)

    def _handle_group_update(self, payload: GroupUpdateServerPayload) -> None:
        self._group_state = payload
        self._notify_group_callback(payload)

    def _handle_server_state(self, payload: ServerStatePayload) -> None:
        self._server_state = payload
        # Notify controller callback for controller state
        self._notify_controller_callback(payload)
        # Notify metadata callback when metadata is present
        if payload.metadata is not None:
            self._notify_metadata_callback(payload)

    def _handle_server_command(self, payload: ServerCommandPayload) -> None:
        """Handle server/command message."""
        self._notify_server_command_callback(payload)

    def _configure_audio_output(self, pcm_format: PCMFormat) -> None:
        """Store the current audio format for use in callbacks."""
        self._current_pcm_format = pcm_format

    def _handle_audio_chunk(self, timestamp_us: int, payload: bytes) -> None:
        """Handle incoming audio chunk and notify callbacks."""
        if not self._audio_chunk_callbacks:
            return
        if self._current_pcm_format is None:
            logger.debug("Dropping audio chunk without format")
            return

        # Pass server timestamp directly to callback - it handles time conversion
        # to allow for dynamic time base updates
        for callback in self._audio_chunk_callbacks:
            try:
                callback(timestamp_us, payload, self._current_pcm_format)
            except Exception:
                logger.exception("Error in audio chunk callback %s", callback)

    def compute_play_time(self, server_timestamp_us: int) -> int:
        """
        Convert server timestamp to client play time with static delay applied.

        This method converts a server timestamp to the equivalent client timestamp
        (based on monotonic loop time) and adds the configured static delay.
        Use this to determine when audio should be played on the client.

        Args:
            server_timestamp_us: Server timestamp in microseconds.

        Returns:
            Client play time in microseconds (monotonic loop time + static delay).
        """
        if self._time_filter.is_synchronized:
            client_time = self._time_filter.compute_client_time(server_timestamp_us)
            return client_time + self._static_delay_us
        # Fallback: add a conservative delay if time sync isn't ready yet
        return self._now_us() + 500_000 + self._static_delay_us

    def compute_server_time(self, client_timestamp_us: int) -> int:
        """
        Convert client timestamp to server timestamp with static delay removed.

        This is the inverse of compute_play_time. It converts a client timestamp
        (monotonic loop time) to the equivalent server timestamp, removing the
        static delay first.

        Args:
            client_timestamp_us: Client timestamp in microseconds (monotonic loop time).

        Returns:
            Server timestamp in microseconds.
        """
        # Remove static delay first, then convert to server time
        adjusted_client_time = client_timestamp_us - self._static_delay_us
        return self._time_filter.compute_server_time(adjusted_client_time)

    def _notify_metadata_callback(self, payload: ServerStatePayload) -> None:
        for callback in self._metadata_callbacks:
            try:
                callback(payload)
            except Exception:
                logger.exception("Error in metadata callback %s", callback)

    def _notify_group_callback(self, payload: GroupUpdateServerPayload) -> None:
        for callback in self._group_callbacks:
            try:
                callback(payload)
            except Exception:
                logger.exception("Error in group callback %s", callback)

    def _notify_controller_callback(self, payload: ServerStatePayload) -> None:
        for callback in self._controller_callbacks:
            try:
                callback(payload)
            except Exception:
                logger.exception("Error in controller callback %s", callback)

    def _notify_stream_start(self, message: StreamStartMessage) -> None:
        for callback in self._stream_start_callbacks:
            try:
                callback(message)
            except Exception:
                logger.exception("Error in stream start callback %s", callback)

    def _notify_stream_end(self, roles: list[Roles] | None) -> None:
        for callback in self._stream_end_callbacks:
            try:
                callback(roles)
            except Exception:
                logger.exception("Error in stream end callback %s", callback)

    def _notify_stream_clear(self, roles: list[Roles] | None) -> None:
        for callback in self._stream_clear_callbacks:
            try:
                callback(roles)
            except Exception:
                logger.exception("Error in stream clear callback %s", callback)

    def _notify_disconnect_callback(self) -> None:
        for callback in self._disconnect_callbacks:
            try:
                callback()
            except Exception:
                logger.exception("Error in disconnect callback %s", callback)

    def _notify_server_command_callback(self, payload: ServerCommandPayload) -> None:
        for callback in self._server_command_callbacks:
            try:
                callback(payload)
            except Exception:
                logger.exception("Error in server command callback %s", callback)

    async def _time_sync_loop(self) -> None:
        try:
            while self.connected:
                try:
                    await self._send_time_message()
                except Exception:
                    logger.exception("Failed to send time sync message")
                await asyncio.sleep(self._compute_time_sync_interval())
        except asyncio.CancelledError:
            pass

    def _compute_time_sync_interval(self) -> float:
        if not self._time_filter.is_synchronized:
            return 0.2
        error = self._time_filter.error
        if error < 1_000:
            return 3.0
        if error < 2_000:
            return 1.0
        if error < 5_000:
            return 0.5
        return 0.2

    def _now_us(self) -> int:
        return int(self._loop.time() * 1_000_000)
