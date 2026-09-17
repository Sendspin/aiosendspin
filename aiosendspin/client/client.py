"""Sendspin Client implementation to connect to a Sendspin Server."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import replace

from aiohttp import ClientSession, web

from aiosendspin.audio.codecs import opus_available
from aiosendspin.clock import Clock, RawMonotonicClock
from aiosendspin.models.artwork import ArtworkChannel, ClientStateArtwork
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
    AudioCodec,
    GoodbyeReason,
    MediaCommand,
    PairAbortReason,
    PairMethod,
    PlayerCommand,
    Roles,
)
from aiosendspin.models.visualizer import (
    ClientHelloVisualizerSupport,
    VisualizerFrame,
    VisualizerStatePayload,
)
from aiosendspin.noise.driver import HandshakeAbortedError
from aiosendspin.noise.keys import Identity
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
_PAIRING_WINDOW_MAX_FAILURES = 5

# Every server accepts these; a server predating the codec list in server/hello
# activates the source role without sending one.
_MANDATORY_SOURCE_CODECS = frozenset({AudioCodec.FLAC, AudioCodec.PCM})


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
# Output MUST stop and buffers MUST be cleared for those roles, also while unavailable.
StreamEndCallback = Callable[[list[str] | None], None]

# Callback invoked when stream buffers should be cleared (e.g., seek operation).
# Receives list of roles to clear, or None if all roles should be cleared.
# Buffered data MUST be cleared for those roles, also while unavailable.
StreamClearCallback = Callable[[list[str] | None], None]

# Callback invoked with (server_timestamp_us, audio_data, format, send_ahead) when audio
# chunks arrive.
AudioChunkCallback = Callable[[int, bytes, AudioFormat, int], None]

# Callback invoked when the client disconnects from the server.
DisconnectCallback = Callable[[], None]

# Callback invoked with the abort reason when a non-closing pairing attempt ends.
PairingAbortCallback = Callable[[PairAbortReason], None]

# Callback invoked when the server sends a command.
ServerCommandCallback = Callable[[ServerCommandPayload], None]

# Callback invoked with the new output delay in milliseconds whenever it changes.
OutputDelayCallback = Callable[[float], None]

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

    Player embedders MUST persist the output delay across restarts, using
    ``add_output_delay_listener()``, and pass it back as ``output_delay_ms``.
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
    _artwork_state: ClientStateArtwork | None
    """Artwork channels reported via client/state (only set if ARTWORK role is supported)."""
    _visualizer_support: ClientHelloVisualizerSupport | None
    """Visualizer capabilities (only set if VISUALIZER role is supported)."""
    _visualizer_state: VisualizerStatePayload | None
    """Visualizer stream configuration reported via client/state."""
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
    _preferred_format: SupportedAudioFormat | None = None
    """Player format preference reported via client/state."""

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
    _pairing_window_connection: SendspinConnection | None = None
    """Connection that carried the window's first attempt; only it starts further ones."""
    _pairing_window_failures: int = 0
    """Attempts under the window whose ``server_kc`` failed to verify."""
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
    _output_delay_callbacks: list[OutputDelayCallback]
    """Callbacks invoked when the output delay changes."""
    _visualizer_callbacks: list[VisualizerCallback]
    """Callbacks invoked when visualizer frames are received (beats included)."""
    _artwork_callbacks: list[ArtworkCallback]
    """Callbacks invoked when artwork frames are received."""

    _initial_volume: int
    """Initial volume level for player role (0-100)."""
    _initial_muted: bool
    """Initial mute state for player role."""
    _state_supported_commands: list[PlayerCommand]
    """Commands advertised in client/state messages."""

    def __init__(  # noqa: PLR0913, PLR0915
        self,
        identity: Identity,
        client_name: str,
        roles: Sequence[Roles],
        *,
        pairing_store: ClientPairingStore,
        device_info: DeviceInfo | None = None,
        player_support: ClientHelloPlayerSupport | None = None,
        artwork_channels: Sequence[ArtworkChannel] | None = None,
        visualizer_support: ClientHelloVisualizerSupport | None = None,
        visualizer_state: VisualizerStatePayload | None = None,
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

        # Validate and store artwork channels
        if Roles.ARTWORK in self._roles:
            if artwork_channels is None:
                raise ValueError("artwork_channels is required when ARTWORK role is specified")
            self._artwork_state = ClientStateArtwork(channels=list(artwork_channels))
        else:
            self._artwork_state = None

        # Validate and store visualizer support and requested configuration
        if Roles.VISUALIZER in self._roles:
            if visualizer_support is None:
                raise ValueError("visualizer_support is required when VISUALIZER role is specified")
            if visualizer_support.has_stream_config:
                raise ValueError(
                    "visualizer types, rate_max and spectrum belong in visualizer_state, "
                    "not visualizer_support"
                )
            if visualizer_state is None:
                raise ValueError("visualizer_state is required when VISUALIZER role is specified")
            self._visualizer_support = visualizer_support
            self._visualizer_state = visualizer_state
        else:
            self._visualizer_support = None
            self._visualizer_state = None

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
        # set_output_delay_ms() notifies these listeners.
        self._output_delay_callbacks = []
        self.set_output_delay_ms(output_delay_ms)
        self.set_required_lead_time_ms(required_lead_time_ms)
        self.set_min_buffer_ms(min_buffer_ms)
        # DEPRECATED(spec-pr-177): remove in aiosendspin <version>
        # Commands an embedder declared on player_support are sent in client/state instead.
        hello_commands = player_support.supported_commands if player_support else None
        self._state_supported_commands: list[PlayerCommand] = list(
            dict.fromkeys([*(hello_commands or []), *(state_supported_commands or [])])
        )

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
    def artwork_state(self) -> ClientStateArtwork | None:
        """Artwork channels reported via client/state (only set if ARTWORK role is supported)."""
        return self._artwork_state

    @property
    def visualizer_support(self) -> ClientHelloVisualizerSupport | None:
        """Visualizer capabilities (only set if VISUALIZER role is supported)."""
        return self._visualizer_support

    @property
    def visualizer_state(self) -> VisualizerStatePayload | None:
        """Visualizer stream configuration reported via client/state."""
        return self._visualizer_state

    @property
    def source_support(self) -> ClientHelloSourceSupport | None:
        """Source capabilities."""
        return self._source_support

    def create_source_capture(self, audio_format: SupportedAudioFormat) -> SourceCapture:
        """
        Create a capture for PCM matching ``audio_format`` on the source connection.

        The codec falls back to FLAC, or else PCM, when server/hello did not list the
        requested one or this client cannot encode it.
        """
        if Roles.SOURCE not in self._roles:
            raise RuntimeError("Client does not have the source role")
        if self._admitted_connection is None:
            raise RuntimeError("Client is not connected")
        server_info = self._admitted_connection.server_info
        accepted = (
            server_info.source_codecs
            if server_info is not None and server_info.source_codecs is not None
            else _MANDATORY_SOURCE_CODECS
        )
        codec = audio_format.codec
        if codec not in accepted or (codec is AudioCodec.OPUS and not opus_available()):
            fallback = AudioCodec.FLAC if AudioCodec.FLAC in accepted else AudioCodec.PCM
            logger.info(
                "%s is not available on this connection, streaming %s instead",
                codec.value,
                fallback.value,
            )
            audio_format = replace(audio_format, codec=fallback)
        return SourceCapture(self, self._admitted_connection, audio_format)

    @property
    def admission_lock(self) -> asyncio.Lock:
        """Lock serializing connection admission and pairing-config writes."""
        return self._admission_lock

    @property
    def pairing_store(self) -> ClientPairingStore:
        """Trust store holding the long-term records and Pairing PSKs."""
        return self._pairing_store

    @property
    def pairing_code_display(self) -> PairingCodeDisplay | None:
        """Out-channel that surfaces a derived pairing code, if configured.

        Called with the pairing code string and its grouped form when one is derived, and with
        ``None`` for both when the pairing exchange ends (success or failure) so the channel can
        clear.
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
    def out_channel_suspend(self) -> Callable[[bool], Awaitable[None]] | None:
        """Hook suspending a role output that doubles as the pairing-code out-channel."""
        return (
            self._pairing_support.out_channel_suspend if self._pairing_support is not None else None
        )

    @property
    def pair_pending_message(self) -> str | None:
        """Operator message sent in ``client/pair-pending``, if configured."""
        return (
            self._pairing_support.pair_pending_message
            if self._pairing_support is not None
            else None
        )

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

    def pairing_window_admits(self, connection: SendspinConnection) -> bool:
        """Whether an open pairing window admits an attempt on ``connection``."""
        owner = self._pairing_window_connection
        return self.pairing_window_open and (owner is None or owner is connection)

    async def await_pairing_window(self, connection: SendspinConnection) -> None:
        """Wait until a pairing window admits an attempt on ``connection``, binding it there.

        Prompts for a gesture meanwhile.
        """
        if self._claim_pairing_window(connection):
            return
        support = self._pairing_support
        prompt = support.gesture_prompt if support is not None else None
        self._pairing_window_waiters += 1
        try:
            if self._pairing_window_waiters == 1 and prompt is not None:
                await prompt(True)  # noqa: FBT003
            while not self._claim_pairing_window(connection):
                self._pairing_window_opened.clear()
                await self._pairing_window_opened.wait()
        finally:
            self._pairing_window_waiters -= 1
            if self._pairing_window_waiters == 0 and prompt is not None:
                await prompt(False)  # noqa: FBT003

    def _claim_pairing_window(self, connection: SendspinConnection) -> bool:
        """Bind an admitting pairing window to ``connection``, reporting whether one admits it."""
        if not self.pairing_window_admits(connection):
            return False
        self._pairing_window_connection = connection
        return True

    def record_pairing_window_attempt(
        self,
        connection: SendspinConnection,
        *,
        paired: bool,
    ) -> None:
        """Record a gated attempt on ``connection`` that paired or whose ``server_kc`` failed.

        A pairing, or the fifth failure, closes the window the attempt ran under.
        """
        if connection is not self._pairing_window_connection:
            return
        if not paired:
            self._pairing_window_failures += 1
        if paired or self._pairing_window_failures >= _PAIRING_WINDOW_MAX_FAILURES:
            self.close_pairing_window()

    def close_pairing_window(self) -> None:
        """Close the pairing window, as on operator cancellation.

        An attempt already in progress runs to its own end.
        """
        self._pairing_window_deadline = None
        self._pairing_window_connection = None
        self._pairing_window_failures = 0
        self._pairing_window_opened.clear()

    def open_pairing_window(self) -> None:
        """Open a pairing window admitting gesture-gated pairing attempts until it closes.

        Called on an operator gesture, or by a paired server through
        ``management/open-pairing-window``. A no-op while a window is already open.
        """
        if self.pairing_window_open:
            return
        self.close_pairing_window()
        self._pairing_window_deadline = self._loop.time() + _PAIRING_WINDOW_LIFETIME_S
        self._pairing_window_opened.set()

    async def cancel_pairing(self) -> None:
        """Cancel the pairing attempt on the admitted connection, as on operator cancellation.

        Sends ``pair/abort`` with reason ``user_cancelled``, closes the pairing window and
        clears the pairing-code out-channels; the connection stays open. Pairing-abort
        listeners are not called, since the caller initiated the cancellation. A no-op when
        no attempt is running or awaiting a pairing window, or once the attempt has sent
        ``client/pair-finalize``: it then completes. Must not be called from a
        ``PairingSupport`` callback, which runs inside the attempt.
        """
        if self._admitted_connection is not None:
            await self._admitted_connection.cancel_pairing()

    async def set_unpaired_access(self, *, enabled: bool) -> None:
        """Persist whether this client admits unpaired access.

        Disabling it closes the admitted connection with ``client/goodbye`` reason
        ``pairing_required`` when that connection relies on unpaired access. Enabling it
        closes nothing: the setting is advertised from the next ``client/hello``.
        A pairing-store error propagates, and then no connection is closed.
        """
        store = self._pairing_store
        # Admission re-checks the setting under this lock, so no connection is admitted
        # between the write and the close.
        async with self._admission_lock:
            config = await store.get_pairing_config()
            await store.store_pairing_config(replace(config, unpaired_access_enabled=enabled))
            connection = self._admitted_connection
            if enabled or connection is None:
                return
            await self._refuse_unpaired_access(connection)

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
        """Commands advertised in client/state, including any declared on player_support."""
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
        """Update the output delay applied after clock synchronisation, clamped to 0-5000 ms.

        The embedder MUST persist the delay across restarts and pass it back as
        ``output_delay_ms`` when creating the client; a server can also change it, so
        persist it from ``add_output_delay_listener()``.
        """
        delay_ms = max(0.0, min(5000.0, delay_ms))
        delay_us = round(delay_ms * 1_000.0)
        if self._admitted_connection is not None:
            self._admitted_connection.set_output_delay_ms(delay_ms)
        if delay_us == self._output_delay_us:
            return
        self._output_delay_us = delay_us
        logger.info("Set output delay to %.1f ms", self.output_delay_ms)
        self.notify_output_delay_callback(delay_us / 1_000.0)

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

    @property
    def preferred_format(self) -> SupportedAudioFormat | None:
        """Return the player format preference reported via client/state."""
        return self._preferred_format

    async def set_preferred_format(self, audio_format: SupportedAudioFormat | None) -> None:
        """Set or clear the player format preference and report it to the server.

        The server applies it to the current stream when it can produce it, or to the
        next stream otherwise. None leaves the choice to the `supported_formats` order.

        Raises ValueError when `audio_format` is not one of this client's
        `supported_formats` (`bit_depth` is not compared for opus).
        """
        supported = self._player_support.supported_formats if self._player_support else []
        if audio_format is not None and not any(audio_format.matches(fmt) for fmt in supported):
            raise ValueError(f"{audio_format} is not one of the client's supported_formats")
        self._preferred_format = audio_format
        connection = self._admitted_connection
        if connection is not None and connection.connected:
            await connection.send_full_player_state()

    async def set_artwork_channels(self, channels: Sequence[ArtworkChannel]) -> None:
        """Set the artwork channel configuration and report it to the server.

        Array index is the channel number; a channel past the end is not streamed.

        Raises ValueError when the ARTWORK role is not specified or `channels` does not
        have 1-4 entries.
        """
        if Roles.ARTWORK not in self._roles:
            raise ValueError("ARTWORK role is not specified")
        self._artwork_state = ClientStateArtwork(channels=list(channels))
        connection = self._admitted_connection
        if connection is not None and connection.connected:
            await connection.send_artwork_state()

    async def set_visualizer_state(self, state: VisualizerStatePayload) -> None:
        """Set the requested visualizer stream configuration and report it to the server.

        The server applies it to the current visualizer stream, or to the next one
        when none is active. `rate_max` caps periodic types only.

        Raises ValueError when this client does not have the VISUALIZER role.
        """
        if Roles.VISUALIZER not in self._roles:
            raise ValueError("visualizer_state requires the VISUALIZER role")
        self._visualizer_state = state
        connection = self._admitted_connection
        if connection is not None and connection.connected:
            await connection.send_visualizer_state()

    # --- Connection lifecycle ---

    async def connect(self, url: str, *, expected_server_id: str | None = None) -> None:
        """Dial a server (client-initiated) and admit it, displacing any current connection.

        The user explicitly chose this server, so the new connection is admitted
        unconditionally — an explicit switch — unless it relies on unpaired access the
        client no longer admits, which raises ``RuntimeError``. Returns once admitted; the
        reader and time-sync loops, and any pairing attempt the server requests, then run in
        the background.
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
                refused = await self._refuse_unpaired_access(connection)
                if not refused:
                    await self._admit_connection(connection)
        except BaseException:
            # Bring-up already dropped it from _provisional_connections, so nothing else owns it.
            await connection.disconnect()
            raise
        if refused:
            raise RuntimeError("server activation rejected (pairing_required)")
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
                if await self._refuse_unpaired_access(connection):
                    return
                await self._admit_connection(connection)
        except BaseException:
            # Bring-up already dropped it from _provisional_connections, so nothing else owns it.
            await connection.disconnect()
            raise
        try:
            await connection.start()
        except (OSError, RuntimeError, TimeoutError) as exc:
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

    async def _refuse_unpaired_access(self, connection: SendspinConnection) -> bool:
        """Close ``connection`` if it relies on unpaired access the client no longer admits.

        Returns whether it was closed, with ``client/goodbye`` reason ``pairing_required``.
        """
        if not connection.relies_on_unpaired_access:
            return False
        if (await self._pairing_store.get_pairing_config()).unpaired_access_enabled:
            return False
        await connection.goodbye_and_disconnect(GoodbyeReason.PAIRING_REQUIRED)
        return True

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
        if self._pairing_window_connection is connection:
            self.close_pairing_window()
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

        ``volume`` (0-100) is perceived loudness: the embedder SHOULD apply the gain
        ``(volume / 100) ** 1.5`` over a short ramp. ``muted`` is independent of
        ``volume``. Persisting both and passing them back as ``initial_volume`` and
        ``initial_muted`` is RECOMMENDED.
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

    async def send_leave(self) -> None:
        """Leave the current group, stopping playback on this client.

        The client stays available and lands in a solo group; a subsequent ``switch``
        group command rejoins the group it left.
        """
        if self._admitted_connection is None:
            raise RuntimeError("Client is not connected")
        await self._admitted_connection.send_leave()

    async def send_group_command(
        self,
        command: MediaCommand,
        *,
        volume: int | None = None,
        mute: bool | None = None,
        position_ms: int | None = None,
        offset_ms: int | None = None,
    ) -> None:
        """Send a group command (playback control) to the server.

        Commands are checked against the latest controller state received from the server.
        Raises ValueError if no controller state was received, if `command` is not in its
        `supported_commands`, or if a `seek` targets a position outside 0 to `seek_max_ms`.
        """
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

    def current_track_position(self) -> int | None:
        """Return the playback position in milliseconds as of now, or None when unknown.

        The position is extrapolated from the progress in the latest metadata received,
        including metadata whose timestamp is still in the future. Returns None while not
        connected, without progress, or before time synchronization has converged.
        """
        if self._admitted_connection is None:
            return None
        return self._admitted_connection.current_track_position()

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

        The callback receives the roles that ended, or ``None`` for all. For each, the
        embedder MUST stop output and clear its buffers. Stream end keeps arriving while
        the client reports unavailable and MUST be handled then too.

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

        The callback receives the roles to clear, or ``None`` for all. When the player
        role is included, the embedder MUST drop all buffered audio and continue with
        chunks received after the clear. Stream clear keeps arriving while the client
        reports unavailable and MUST be handled then too.

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
        - send_ahead: Microseconds from the server's transmission of the chunk to
          server_timestamp_us. 0 and 4294967295 are saturated values and carry no
          delay sample. Never affects when the chunk plays.

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

        A player ``volume`` command sets perceived loudness: the embedder SHOULD apply
        the gain ``(volume / 100) ** 1.5`` over a short ramp. A volume command MUST NOT
        clear the mute state.

        Returns:
            A function that removes this listener when called.
        """
        self._server_command_callbacks.append(callback)
        return lambda: (
            self._server_command_callbacks.remove(callback)
            if callback in self._server_command_callbacks
            else None
        )

    def add_output_delay_listener(self, callback: OutputDelayCallback) -> Callable[[], None]:
        """Add a listener for output delay changes, local or server-set.

        The callback receives the clamped delay in milliseconds. Persist it and pass it
        back as ``output_delay_ms`` on the next start.

        Returns:
            A function that removes this listener when called.
        """
        self._output_delay_callbacks.append(callback)
        return lambda: (
            self._output_delay_callbacks.remove(callback)
            if callback in self._output_delay_callbacks
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

    def notify_output_delay_callback(self, delay_ms: float) -> None:
        """Dispatch an output delay change to the registered listeners."""
        for callback in list(self._output_delay_callbacks):
            try:
                callback(delay_ms)
            except Exception:
                logger.exception("Error in output delay callback %s", callback)

    def notify_visualizer_callbacks(self, frames: list[VisualizerFrame]) -> None:
        """Dispatch visualizer frames to the registered listeners."""
        for callback in list(self._visualizer_callbacks):
            try:
                callback(frames)
            except Exception:
                logger.exception("Error in visualizer callback %s", callback)

    def notify_audio_chunk(
        self, timestamp_us: int, payload: bytes, audio_format: AudioFormat, send_ahead: int
    ) -> None:
        """Dispatch an audio chunk to the registered listeners."""
        for callback in list(self._audio_chunk_callbacks):
            try:
                callback(timestamp_us, payload, audio_format, send_ahead)
            except Exception:
                logger.exception("Error in audio chunk callback %s", callback)

    def notify_artwork(self, channel: int, payload: bytes) -> None:
        """Dispatch an artwork chunk to the registered listeners."""
        for callback in list(self._artwork_callbacks):
            try:
                callback(channel, payload)
            except Exception:
                logger.exception("Error in artwork callback %s", callback)
