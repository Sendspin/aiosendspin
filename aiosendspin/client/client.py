"""Sendspin Client implementation to connect to a Sendspin Server."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from contextlib import suppress

from aiohttp import ClientSession, web

from aiosendspin.clock import Clock, RawMonotonicClock
from aiosendspin.models.artwork import ClientHelloArtworkSupport
from aiosendspin.models.core import (
    DeviceInfo,
    GroupUpdateServerPayload,
    ServerCommandPayload,
    ServerStatePayload,
    StreamStartMessage,
)
from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.models.source import ClientHelloSourceSupport
from aiosendspin.models.types import (
    Activity,
    GoodbyeReason,
    MediaCommand,
    PairAbortReason,
    PairMethod,
    PlayerCommand,
    Roles,
)
from aiosendspin.models.visualizer import ClientHelloVisualizerSupport, VisualizerFrame
from aiosendspin.noise.driver import HandshakeAbortedError
from aiosendspin.noise.keys import Identity
from aiosendspin.noise.pairing import PairingError
from aiosendspin.noise.session import NoiseCipherSuite
from aiosendspin.noise.trust_store import ClientPairingStore, ResolvedPsk

from .connection import DECODABLE_CODECS, UNSYNCED_PLAY_LEAD_US, SendspinConnection
from .models import (
    AudioFormat,
    PairingCodeDisplay,
    PairingCodeSpeaker,
    PairingSupport,
    QRCodeDisplay,
    ServerInfo,
)
from .source import SourceCapture

logger = logging.getLogger(__name__)

_PAIRING_WINDOW_LIFETIME_S: float = 300.0


def _validate_decodable_formats(player_support: ClientHelloPlayerSupport) -> None:
    """Reject advertised codecs the SDK cannot decode."""
    undecodable = {f.codec for f in player_support.supported_formats}.difference(DECODABLE_CODECS)
    if undecodable:
        names = ", ".join(sorted(c.value for c in undecodable))
        raise ValueError(
            f"player_support advertises codecs the SDK cannot decode ({names}); "
            "only PCM and FLAC are supported"
        )


# Callback invoked when server state metadata updates are received.
MetadataCallback = Callable[[ServerStatePayload], None]

# Callback invoked when group state updates are received.
GroupUpdateCallback = Callable[[GroupUpdateServerPayload], None]

# Callback invoked when controller state updates are received.
ControllerStateCallback = Callable[[ServerStatePayload], None]

# Callback invoked when server state color updates are received.
ColorCallback = Callable[[ServerStatePayload], None]

# Callback invoked when audio streaming begins.
StreamStartCallback = Callable[[StreamStartMessage], None]

# Callback invoked when audio streaming ends.
# Receives list of roles to end, or None if all roles should be ended.
StreamEndCallback = Callable[[list[str] | None], None]

# Callback invoked when stream buffers should be cleared (e.g., seek operation).
# Receives list of roles to clear, or None if all roles should be cleared.
StreamClearCallback = Callable[[list[str] | None], None]

# Callback invoked with (server_timestamp_us, audio_data, format) when audio chunks arrive.
AudioChunkCallback = Callable[[int, bytes, AudioFormat], None]

# Callback invoked when the client disconnects from the server.
DisconnectCallback = Callable[[], None]

# Callback invoked with the abort reason when a non-closing pairing attempt ends.
PairingAbortCallback = Callable[[PairAbortReason], None]

# Callback invoked when the server sends a command.
ServerCommandCallback = Callable[[ServerCommandPayload], None]

# Callback invoked when visualizer frames are received. Beat events are
# delivered through the same callback as a `VisualizerFrame` carrying
# only `timestamp_us` + `is_downbeat`.
VisualizerCallback = Callable[[list[VisualizerFrame]], None]

# Callback invoked when artwork binary frames are received.
ArtworkCallback = Callable[[int, bytes], None]


class SendspinClient:
    """
    Async Sendspin client for handling playback and metadata.

    The client must be created within an async context and requires explicit
    role specification. Player and metadata support configs are required if
    their respective roles are enabled.
    """

    _identity: Identity
    """This client's static public X25519 identity."""
    _client_id: str
    """The client's wire identifier — derived from ``_identity``."""
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
    _visualizer_support: ClientHelloVisualizerSupport | None
    """Visualizer capabilities (only set if VISUALIZER role is supported)."""
    _source_support: ClientHelloSourceSupport | None
    """Source capabilities."""
    _session: ClientSession | None
    """Optional aiohttp ClientSession for WebSocket connection."""

    _loop: asyncio.AbstractEventLoop
    """Event loop for this client."""
    _pairing_store: ClientPairingStore
    """Trust store consulted to resolve a PSK by ``psk_id`` during the handshake."""
    _owns_session: bool
    """Whether this client owns and should close the session."""

    _output_delay_us: int = 0
    """Default output delay seeding new connections, in microseconds."""
    _required_lead_time_us: int = 250_000
    """Reported startup lead time in microseconds."""
    _min_buffer_us: int = 250_000
    """Reported minimum ongoing buffer duration in microseconds."""

    _admitted_connection: SendspinConnection | None = None
    """The currently-admitted connection (at most one), if any."""
    _provisional_connections: set[SendspinConnection]
    """Incoming connections still being brought up / awaiting admission."""
    _admission_lock: asyncio.Lock
    """Serializes the admit/displace decision across concurrent connections."""

    _pairing_support: PairingSupport | None
    """Operator wiring whose presence enables the pairing-code methods, if configured."""
    _pairing_window_opened: asyncio.Event
    """Set when a pairing window opens; wakes a gated attempt's wait."""
    _pairing_window_deadline: float | None = None
    """Loop-time deadline of the open pairing window, if one is open."""
    _pairing_window_waiters: int = 0
    """Gated attempts currently waiting for a window; refcounts the gesture prompt."""

    last_playback_server_id: str | None = None
    """server_id of the last server admitted with the playback activity; the discovery tiebreak."""

    _metadata_callbacks: list[MetadataCallback]
    """Callbacks invoked on server/state messages with metadata."""
    _group_callbacks: list[GroupUpdateCallback]
    """Callbacks invoked on group/update messages."""
    _controller_callbacks: list[ControllerStateCallback]
    """Callbacks invoked on server/state messages."""
    _color_callbacks: list[ColorCallback]
    """Callbacks invoked on server/state messages with color."""
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
    _pairing_abort_callbacks: list[PairingAbortCallback]
    """Callbacks invoked when a non-closing pairing attempt ends with an abort reason."""
    _server_command_callbacks: list[ServerCommandCallback]
    """Callbacks invoked when server sends player commands."""
    _visualizer_callbacks: list[VisualizerCallback]
    """Callbacks invoked when visualizer frames are received (beats included)."""
    _artwork_callbacks: list[ArtworkCallback]
    """Callbacks invoked when artwork frames are received."""

    _initial_volume: int
    """Initial volume level for player role (0-100)."""
    _initial_muted: bool
    """Initial mute state for player role."""
    _state_supported_commands: list[PlayerCommand]
    """Supported commands advertised in client/state messages."""

    def __init__(  # noqa: PLR0913, PLR0915
        self,
        identity: Identity,
        client_name: str,
        roles: Sequence[Roles],
        *,
        pairing_store: ClientPairingStore,
        device_info: DeviceInfo | None = None,
        player_support: ClientHelloPlayerSupport | None = None,
        artwork_support: ClientHelloArtworkSupport | None = None,
        visualizer_support: ClientHelloVisualizerSupport | None = None,
        source_support: ClientHelloSourceSupport | None = None,
        session: ClientSession | None = None,
        output_delay_ms: float = 0.0,
        required_lead_time_ms: float = 250.0,
        min_buffer_ms: float = 250.0,
        initial_volume: int = 100,
        initial_muted: bool = False,
        state_supported_commands: list[PlayerCommand] | None = None,
        pairing_support: PairingSupport | None = None,
        clock: Clock | None = None,
        cipher_suite: NoiseCipherSuite = NoiseCipherSuite.CHACHAPOLY,
    ) -> None:
        """Create a new Sendspin client instance."""
        self._identity = identity
        self._client_id = identity.peer_id
        self._client_name = client_name
        self._device_info = device_info
        self._roles = list(roles)
        self._pairing_store = pairing_store
        self._pairing_support = pairing_support
        self._clock: Clock = clock or RawMonotonicClock()
        self._cipher_suite = cipher_suite

        # Validate and store player support
        if Roles.PLAYER in self._roles:
            if player_support is None:
                raise ValueError("player_support is required when PLAYER role is specified")
            _validate_decodable_formats(player_support)
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

        # Validate and store visualizer support
        if Roles.VISUALIZER in self._roles:
            if visualizer_support is None:
                raise ValueError("visualizer_support is required when VISUALIZER role is specified")
            self._visualizer_support = visualizer_support
        else:
            self._visualizer_support = None

        if Roles.SOURCE in self._roles:
            if source_support is None:
                raise ValueError("source_support is required when SOURCE role is specified")
            self._source_support = source_support
        else:
            self._source_support = None
        self._session = session
        self._owns_session = session is None
        self._loop = asyncio.get_running_loop()
        self._initial_volume = initial_volume
        self._initial_muted = initial_muted
        self.set_output_delay_ms(output_delay_ms)
        self.set_required_lead_time_ms(required_lead_time_ms)
        self.set_min_buffer_ms(min_buffer_ms)
        self._state_supported_commands: list[PlayerCommand] = list(state_supported_commands or [])

        self._provisional_connections = set()
        self._admission_lock = asyncio.Lock()
        self._last_playback_loaded = False
        self._pairing_window_opened = asyncio.Event()

        # Initialize callback lists
        self._metadata_callbacks = []
        self._group_callbacks = []
        self._controller_callbacks = []
        self._color_callbacks = []
        self._stream_start_callbacks = []
        self._stream_end_callbacks = []
        self._stream_clear_callbacks = []
        self._audio_chunk_callbacks = []
        self._disconnect_callbacks = []
        self._pairing_abort_callbacks = []
        self._server_command_callbacks = []
        self._visualizer_callbacks = []
        self._artwork_callbacks = []

    # --- Configuration ---

    @property
    def identity(self) -> Identity:
        """This client's static public X25519 identity."""
        return self._identity

    @property
    def cipher_suite(self) -> NoiseCipherSuite:
        """Noise cipher suite this client picks for its handshakes."""
        return self._cipher_suite

    @property
    def client_name(self) -> str:
        """Human-readable name for this client."""
        return self._client_name

    @property
    def device_info(self) -> DeviceInfo | None:
        """Optional device information."""
        return self._device_info

    @property
    def roles(self) -> list[Roles]:
        """List of roles this client supports."""
        return self._roles

    @property
    def player_support(self) -> ClientHelloPlayerSupport | None:
        """Player capabilities (only set if PLAYER role is supported)."""
        return self._player_support

    @property
    def artwork_support(self) -> ClientHelloArtworkSupport | None:
        """Artwork capabilities (only set if ARTWORK role is supported)."""
        return self._artwork_support

    @property
    def visualizer_support(self) -> ClientHelloVisualizerSupport | None:
        """Visualizer capabilities (only set if VISUALIZER role is supported)."""
        return self._visualizer_support

    @property
    def source_support(self) -> ClientHelloSourceSupport | None:
        """Source capabilities."""
        return self._source_support

    def create_source_capture(self, audio_format: SupportedAudioFormat) -> SourceCapture:
        """Create a capture for PCM matching ``audio_format`` on the source connection."""
        if Roles.SOURCE not in self._roles:
            raise RuntimeError("Client does not have the source role")
        if self._admitted_connection is None:
            raise RuntimeError("Client is not connected")
        return SourceCapture(self, self._admitted_connection, audio_format)

    @property
    def pairing_store(self) -> ClientPairingStore:
        """Trust store holding the long-term records and Pairing PSKs."""
        return self._pairing_store

    @property
    def pairing_code_display(self) -> PairingCodeDisplay | None:
        """Out-channel that surfaces a derived pairing code, if configured.

        Called with the pairing code string when one is derived, and with ``None`` when the
        pairing exchange ends (success or failure) so the channel can clear.
        """
        return (
            self._pairing_support.pairing_code_display
            if self._pairing_support is not None
            else None
        )

    @property
    def pairing_code_speaker(self) -> PairingCodeSpeaker | None:
        """Spoken out-channel for a derived pairing code, if configured."""
        return (
            self._pairing_support.pairing_code_speaker
            if self._pairing_support is not None
            else None
        )

    @property
    def qr_code_display(self) -> QRCodeDisplay | None:
        """Display that renders the dynamic pairing token as a QR code, if configured."""
        return self._pairing_support.qr_code_display if self._pairing_support is not None else None

    @property
    def pairing_code_out_channels(self) -> tuple[str, ...]:
        """Channels the dynamic pairing code is conveyed through, in descriptor order."""
        channels = []
        if self.pairing_code_display is not None or self.qr_code_display is not None:
            channels.append("display")
        if self.pairing_code_speaker is not None:
            channels.append("speaker")
        return tuple(channels)

    @property
    def secret_locations(self) -> tuple[str, ...]:
        """Where the operator finds a configured static secret, empty when undeclared."""
        return self._pairing_support.secret_locations if self._pairing_support is not None else ()

    @property
    def pairing_window_open(self) -> bool:
        """Whether a pairing window is currently open."""
        deadline = self._pairing_window_deadline
        return deadline is not None and self._loop.time() < deadline

    async def await_pairing_window(self) -> None:
        """Claim a pairing window for the caller's attempt, prompting for a gesture meanwhile."""
        if self._claim_pairing_window():
            return
        support = self._pairing_support
        prompt = support.gesture_prompt if support is not None else None
        self._pairing_window_waiters += 1
        try:
            if self._pairing_window_waiters == 1 and prompt is not None:
                await prompt(True)  # noqa: FBT003
            while not self._claim_pairing_window():
                self._pairing_window_opened.clear()
                await self._pairing_window_opened.wait()
        finally:
            self._pairing_window_waiters -= 1
            if self._pairing_window_waiters == 0 and prompt is not None:
                await prompt(False)  # noqa: FBT003

    def _claim_pairing_window(self) -> bool:
        """Close an open pairing window for one attempt, reporting whether one was open."""
        if not self.pairing_window_open:
            return False
        self.consume_pairing_window()
        return True

    def consume_pairing_window(self) -> None:
        """Close the pairing window as a pairing attempt starts."""
        self._pairing_window_deadline = None
        self._pairing_window_opened.clear()

    def open_pairing_window(self) -> None:
        """Open a pairing window admitting one gesture-gated pairing attempt.

        Called on an operator gesture, or by a paired server through
        ``management/open-pairing-window``. A no-op while a window is already open.
        """
        if self.pairing_window_open:
            return
        self._pairing_window_deadline = self._loop.time() + _PAIRING_WINDOW_LIFETIME_S
        self._pairing_window_opened.set()

    @property
    def implemented_pair_methods(self) -> frozenset[PairMethod]:
        """Pairing methods this client implements: each pairing-code method needs its wiring."""
        methods = {PairMethod.PAIRING_PSK}
        if self._pairing_support is not None:
            if self._pairing_support.offer_static_pairing_code:
                methods.add(PairMethod.STATIC_PAIRING_CODE)
            if self.pairing_code_out_channels:
                methods.add(PairMethod.DYNAMIC_PAIRING_CODE)
        return frozenset(methods)

    @property
    def initial_volume(self) -> int:
        """Initial volume level for player role (0-100)."""
        return self._initial_volume

    @property
    def initial_muted(self) -> bool:
        """Initial mute state for player role."""
        return self._initial_muted

    @property
    def state_supported_commands(self) -> list[PlayerCommand]:
        """Supported commands advertised in client/state messages."""
        return self._state_supported_commands

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """Event loop for this client."""
        return self._loop

    @property
    def clock(self) -> Clock:
        """Monotonic clock used for time sync timestamps."""
        return self._clock

    @property
    def output_delay_us(self) -> int:
        """Default output delay seeding new connections, in microseconds."""
        return self._output_delay_us

    # --- Connection state ---

    @property
    def connected(self) -> bool:
        """Return True if the client currently has an active connection."""
        return self._admitted_connection is not None and self._admitted_connection.connected

    @property
    def server_info(self) -> ServerInfo | None:
        """Return information about the connected server, if available."""
        if self._admitted_connection is None:
            return None
        return self._admitted_connection.server_info

    @property
    def noise_psk(self) -> ResolvedPsk | None:
        """The PSK that admitted the current connection, or ``None`` if not connected."""
        if self._admitted_connection is None:
            return None
        return self._admitted_connection.noise_psk

    @property
    def activities(self) -> list[Activity]:
        """The server's currently-declared activities."""
        if self._admitted_connection is None:
            return []
        return self._admitted_connection.activities

    @property
    def output_delay_ms(self) -> float:
        """Return the currently configured output delay in milliseconds."""
        if self._admitted_connection is not None:
            return self._admitted_connection.output_delay_ms
        return self._output_delay_us / 1_000.0

    def set_output_delay_ms(self, delay_ms: float) -> None:
        """Update the output delay applied after clock synchronisation."""
        delay_ms = max(0.0, min(5000.0, delay_ms))
        delay_us = round(delay_ms * 1_000.0)
        if self._admitted_connection is not None:
            self._admitted_connection.set_output_delay_ms(delay_ms)
        if delay_us == self._output_delay_us:
            return
        self._output_delay_us = delay_us
        logger.info("Set output delay to %.1f ms", self.output_delay_ms)

    @property
    def required_lead_time_ms(self) -> float:
        """Return the currently reported startup lead time in milliseconds."""
        return self._required_lead_time_us / 1_000.0

    def set_required_lead_time_ms(self, lead_ms: float) -> None:
        """Update the startup lead time reported via client/state.

        If changing frequently, the caller must debounce to only report sustained shifts.
        """
        lead_ms = max(0.0, min(30000.0, lead_ms))
        lead_us = round(lead_ms * 1_000.0)
        if lead_us == self._required_lead_time_us:
            return
        self._required_lead_time_us = lead_us
        logger.info("Set required lead time to %.1f ms", self.required_lead_time_ms)

    @property
    def min_buffer_ms(self) -> float:
        """Return the currently reported minimum ongoing buffer duration in milliseconds."""
        return self._min_buffer_us / 1_000.0

    def set_min_buffer_ms(self, buffer_ms: float) -> None:
        """Update the minimum ongoing buffer duration reported via client/state.

        If changing frequently, the caller must debounce to only report sustained shifts.
        """
        buffer_ms = max(0.0, min(30000.0, buffer_ms))
        buffer_us = round(buffer_ms * 1_000.0)
        if buffer_us == self._min_buffer_us:
            return
        self._min_buffer_us = buffer_us
        logger.info("Set minimum ongoing buffer to %.1f ms", self.min_buffer_ms)

    # --- Connection lifecycle ---

    async def connect(self, url: str, *, expected_server_id: str | None = None) -> None:
        """Dial a server (client-initiated) and admit it, displacing any current connection.

        The user explicitly chose this server, so the new connection is admitted
        unconditionally — an explicit switch. Returns once admitted; the reader and
        time-sync loops then run in the background.
        """
        if self._session is None:
            self._session = ClientSession()

        logger.info("Connecting to Sendspin server at %s", url)

        raw_ws = await self._session.ws_connect(url, heartbeat=30)
        connection = SendspinConnection(self)
        self._provisional_connections.add(connection)
        try:
            await connection.connect(raw_ws, expected_server_id=expected_server_id)
        finally:
            self._provisional_connections.discard(connection)

        # Hold the lock only for the admit decision, not for start()/pairing.
        try:
            async with self._admission_lock:
                await self._admit_connection(connection)
        except BaseException:
            # Bring-up already dropped it from _provisional_connections, so nothing else owns it.
            await connection.disconnect()
            raise
        await connection.start()

    async def attach_websocket(
        self, ws: web.WebSocketResponse, *, expected_server_id: str | None = None
    ) -> None:
        """Bring up an incoming (server-initiated) connection, arbitrate, and serve it.

        The connection is provisional until its first ``server/activate``; it is
        then admitted (displacing a lower-or-equal-ranked holder) or rejected per
        the multi-server admission rules. If admitted, this blocks until the
        connection closes, so it can drive an incoming WebSocket handler.

        Args:
            ws: An already-prepared WebSocketResponse from an incoming connection.
            expected_server_id: If given, the server's advertised identity must
                match or the handshake is aborted.
        """
        connection = SendspinConnection(self)
        self._provisional_connections.add(connection)
        try:
            await connection.attach_websocket(ws, expected_server_id=expected_server_id)
        except (HandshakeAbortedError, OSError, RuntimeError, TimeoutError) as exc:
            # Bring-up failed; the connection/socket is already torn down.
            self._provisional_connections.discard(connection)
            logger.debug("Incoming connection failed bring-up: %s", exc)
            return
        except BaseException:
            # Failures outside the expected set may leave the transport half-open.
            await connection.disconnect()
            raise
        finally:
            self._provisional_connections.discard(connection)

        # Hold the lock only for the admit/reject decision, not for start()/pairing.
        try:
            async with self._admission_lock:
                await self._ensure_last_playback_loaded()
                if not self._should_admit_connection(connection):
                    await self._reject_connection(connection)
                    return
                await self._admit_connection(connection)
        except BaseException:
            # Bring-up already dropped it from _provisional_connections, so nothing else owns it.
            await connection.disconnect()
            raise
        try:
            await connection.start()
        except (PairingError, OSError, RuntimeError, TimeoutError) as exc:
            logger.debug("Admitted connection did not reach steady state: %s", exc)
            return
        await connection.wait_closed()

    async def disconnect(self, reason: GoodbyeReason = GoodbyeReason.SHUTDOWN) -> None:
        """Announce goodbye, disconnect all connections, and release the session."""
        for connection in list(self._provisional_connections):
            await connection.disconnect()
        admitted = self._admitted_connection
        if admitted is not None:
            await admitted.send_goodbye(reason)
            await admitted.disconnect()  # on_connection_closed fires the disconnect callback
        else:
            self.notify_disconnect_callback()
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    # --- Connection admission ---

    @staticmethod
    def _activity_rank(activities: list[Activity]) -> int:
        """Rank a connection by its highest activity: management > playback > pairing > none."""
        if Activity.MANAGEMENT in activities:
            return 3
        if Activity.PLAYBACK in activities:
            return 2
        if Activity.PAIRING in activities:
            return 1
        return 0

    def _should_admit_connection(self, incoming: SendspinConnection) -> bool:
        """Whether ``incoming`` should displace the current admitted connection."""
        admitted = self._admitted_connection
        if admitted is None or not admitted.connected:
            return True
        incoming_rank = self._activity_rank(incoming.activities)
        admitted_rank = self._activity_rank(admitted.activities)
        # A pairing attempt in progress is not displaced by an incoming playback or pairing.
        if admitted.pairing_attempt_in_progress and incoming_rank in (1, 2):
            return False
        if incoming_rank != admitted_rank:
            return incoming_rank > admitted_rank
        # Equal rank is accepted, except both-empty resolves by the last-playback server.
        if incoming_rank == 0:
            return (
                incoming.server_id == self.last_playback_server_id
                and admitted.server_id != self.last_playback_server_id
            )
        return True

    async def _ensure_last_playback_loaded(self) -> None:
        """Load the persisted last-playback server once, seeding the discovery tiebreak."""
        if self._last_playback_loaded:
            return
        if self.last_playback_server_id is None:
            self.last_playback_server_id = await self._pairing_store.get_last_playback_server_id()
        self._last_playback_loaded = True

    async def _admit_connection(self, connection: SendspinConnection) -> None:
        """Make ``connection`` the admitted one, displacing any prior holder."""
        previous = self._admitted_connection
        if previous is connection:
            return
        await self._record_last_playback(connection)
        self._admitted_connection = connection
        if previous is not None:
            await self._dismiss_connection(previous, GoodbyeReason.ANOTHER_SERVER)
            await previous.disconnect()

    async def note_playback_activity(self, connection: SendspinConnection) -> None:
        """Record the admitted server as last-playback when it carries the playback activity."""
        if connection is self._admitted_connection:
            await self._record_last_playback(connection)

    async def _record_last_playback(self, connection: SendspinConnection) -> None:
        """Persist the last-playback server before caching it, so a failed write retries."""
        if (
            Activity.PLAYBACK in connection.activities
            and connection.server_id is not None
            and connection.server_id != self.last_playback_server_id
        ):
            await self._pairing_store.set_last_playback_server_id(connection.server_id)
            self.last_playback_server_id = connection.server_id

    async def _reject_connection(self, connection: SendspinConnection) -> None:
        """Refuse an incoming connection that lost arbitration."""
        await self._dismiss_connection(connection, GoodbyeReason.CONCURRENT_ATTEMPT)
        await connection.disconnect()

    @staticmethod
    async def _dismiss_connection(connection: SendspinConnection, reason: GoodbyeReason) -> None:
        """Tell a connection it's being dropped: pair/abort if pairing, else client/goodbye."""
        with suppress(Exception):
            if connection.is_pairing:
                await connection.send_pair_abort(PairAbortReason.CONCURRENT_ATTEMPT)
            else:
                await connection.send_goodbye(reason)

    def on_connection_closed(self, connection: SendspinConnection) -> None:
        """Report a disconnect only when the admitted connection (not a provisional one) closes."""
        self._provisional_connections.discard(connection)
        if self._admitted_connection is connection:
            self._admitted_connection = None
            self.notify_disconnect_callback()

    # --- Outbound protocol ---

    async def send_player_state(
        self,
        *,
        available: bool,
        volume: int,
        muted: bool,
    ) -> None:
        """Send player state, including client availability.

        Player clients can report availability here. Use ``send_available()`` when no
        player fields changed.
        """
        if self._admitted_connection is None:
            raise RuntimeError("Client is not connected")
        await self._admitted_connection.send_player_state(
            available=available, volume=volume, muted=muted
        )

    async def send_available(self, *, available: bool) -> None:
        """Report whether this client can participate in Sendspin.

        Use this for non-player clients or when no player fields changed.
        An active source stream ends before the client reports unavailable.

        Args:
            available: True when operational and ready, False when unavailable.
        """
        if self._admitted_connection is None:
            raise RuntimeError("Client is not connected")
        await self._admitted_connection.send_available(available=available)

    async def send_group_command(
        self,
        command: MediaCommand,
        *,
        volume: int | None = None,
        mute: bool | None = None,
        position_ms: int | None = None,
        offset_ms: int | None = None,
    ) -> None:
        """Send a group command (playback control) to the server."""
        if self._admitted_connection is None:
            raise RuntimeError("Client is not connected")
        await self._admitted_connection.send_group_command(
            command,
            volume=volume,
            mute=mute,
            position_ms=position_ms,
            offset_ms=offset_ms,
        )

    # --- Time synchronization ---

    def is_time_synchronized(self) -> bool:
        """Return whether time synchronization with the server has converged."""
        if self._admitted_connection is None:
            return False
        return self._admitted_connection.is_time_synchronized()

    def compute_play_time(self, server_timestamp_us: int) -> int:
        """Convert a server timestamp to client play time, with output delay applied."""
        if self._admitted_connection is None:
            return self.now_us() + UNSYNCED_PLAY_LEAD_US - self._output_delay_us
        return self._admitted_connection.compute_play_time(server_timestamp_us)

    def compute_server_time(self, client_timestamp_us: int) -> int:
        """Convert a client timestamp to a server timestamp, with output delay removed."""
        if self._admitted_connection is None:
            return client_timestamp_us + self._output_delay_us
        return self._admitted_connection.compute_server_time(client_timestamp_us)

    def now_us(self) -> int:
        """Return current timestamp from the client's clock in microseconds."""
        return self._clock.now_us()

    # --- Listener registration ---

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

    def add_color_listener(self, callback: ColorCallback) -> Callable[[], None]:
        """Add a listener for server/state messages with color.

        Returns:
            A function that removes this listener when called.
        """
        self._color_callbacks.append(callback)
        return lambda: (
            self._color_callbacks.remove(callback) if callback in self._color_callbacks else None
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

        To convert server timestamps to client play time (monotonic client clock),
        use the compute_play_time() and compute_server_time() methods provided
        by this client instance. These handle time synchronization and output delay
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

    def add_pairing_abort_listener(self, callback: PairingAbortCallback) -> Callable[[], None]:
        """Add a listener for non-closing pairing aborts.

        Returns:
            A function that removes this listener when called.
        """
        self._pairing_abort_callbacks.append(callback)
        return lambda: (
            self._pairing_abort_callbacks.remove(callback)
            if callback in self._pairing_abort_callbacks
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

    def add_visualizer_listener(self, callback: VisualizerCallback) -> Callable[[], None]:
        """Add a listener for visualizer frame events.

        The callback receives a list of VisualizerFrame objects parsed from
        a single visualization data binary message.

        Returns:
            A function that removes this listener when called.
        """
        self._visualizer_callbacks.append(callback)
        return lambda: (
            self._visualizer_callbacks.remove(callback)
            if callback in self._visualizer_callbacks
            else None
        )

    def add_artwork_listener(self, callback: ArtworkCallback) -> Callable[[], None]:
        """Add a listener for artwork binary frame events."""
        self._artwork_callbacks.append(callback)
        return lambda: (
            self._artwork_callbacks.remove(callback)
            if callback in self._artwork_callbacks
            else None
        )

    # --- Listener dispatch ---

    def notify_metadata_callback(self, payload: ServerStatePayload) -> None:
        """Dispatch a server/state with metadata to the registered listeners."""
        for callback in list(self._metadata_callbacks):
            try:
                callback(payload)
            except Exception:
                logger.exception("Error in metadata callback %s", callback)

    def notify_group_callback(self, payload: GroupUpdateServerPayload) -> None:
        """Dispatch a group/update to the registered listeners."""
        for callback in list(self._group_callbacks):
            try:
                callback(payload)
            except Exception:
                logger.exception("Error in group callback %s", callback)

    def notify_controller_callback(self, payload: ServerStatePayload) -> None:
        """Dispatch a server/state to the registered controller listeners."""
        for callback in list(self._controller_callbacks):
            try:
                callback(payload)
            except Exception:
                logger.exception("Error in controller callback %s", callback)

    def notify_color_callback(self, payload: ServerStatePayload) -> None:
        """Dispatch a server/state with color to the registered listeners."""
        for callback in list(self._color_callbacks):
            try:
                callback(payload)
            except Exception:
                logger.exception("Error in color callback %s", callback)

    def notify_stream_start(self, message: StreamStartMessage) -> None:
        """Dispatch a stream/start to the registered listeners."""
        for callback in list(self._stream_start_callbacks):
            try:
                callback(message)
            except Exception:
                logger.exception("Error in stream start callback %s", callback)

    def notify_stream_end(self, roles: list[str] | None) -> None:
        """Dispatch a stream/end to the registered listeners."""
        for callback in list(self._stream_end_callbacks):
            try:
                callback(roles)
            except Exception:
                logger.exception("Error in stream end callback %s", callback)

    def notify_stream_clear(self, roles: list[str] | None) -> None:
        """Dispatch a stream/clear to the registered listeners."""
        for callback in list(self._stream_clear_callbacks):
            try:
                callback(roles)
            except Exception:
                logger.exception("Error in stream clear callback %s", callback)

    def notify_disconnect_callback(self) -> None:
        """Dispatch a disconnect to the registered listeners."""
        for callback in list(self._disconnect_callbacks):
            try:
                callback()
            except Exception:
                logger.exception("Error in disconnect callback %s", callback)

    def notify_pairing_abort_callback(self, reason: PairAbortReason) -> None:
        """Dispatch a non-closing pairing abort to the registered listeners."""
        for callback in list(self._pairing_abort_callbacks):
            try:
                callback(reason)
            except Exception:
                logger.exception("Error in pairing abort callback %s", callback)

    def notify_server_command_callback(self, payload: ServerCommandPayload) -> None:
        """Dispatch a server/command to the registered listeners."""
        for callback in list(self._server_command_callbacks):
            try:
                callback(payload)
            except Exception:
                logger.exception("Error in server command callback %s", callback)

    def notify_visualizer_callbacks(self, frames: list[VisualizerFrame]) -> None:
        """Dispatch visualizer frames to the registered listeners."""
        for callback in list(self._visualizer_callbacks):
            try:
                callback(frames)
            except Exception:
                logger.exception("Error in visualizer callback %s", callback)

    def notify_audio_chunk(
        self, timestamp_us: int, payload: bytes, audio_format: AudioFormat
    ) -> None:
        """Dispatch an audio chunk to the registered listeners."""
        for callback in list(self._audio_chunk_callbacks):
            try:
                callback(timestamp_us, payload, audio_format)
            except Exception:
                logger.exception("Error in audio chunk callback %s", callback)

    def notify_artwork(self, channel: int, payload: bytes) -> None:
        """Dispatch an artwork chunk to the registered listeners."""
        for callback in list(self._artwork_callbacks):
            try:
                callback(channel, payload)
            except Exception:
                logger.exception("Error in artwork callback %s", callback)
