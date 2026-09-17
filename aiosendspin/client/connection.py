"""A single connection from a ``SendspinClient`` to one Sendspin server."""

from __future__ import annotations

import asyncio
import base64
import logging
import struct
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager, suppress
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Literal, NoReturn, assert_never

import orjson
from aiohttp import ClientWebSocketResponse, WSMessage, WSMsgType, web

from aiosendspin.models import BinaryMessageType, pack_binary_header_raw
from aiosendspin.models.artwork import (
    ARTWORK_ANNOUNCE_SIZE,
    ARTWORK_FLAG_ANNOUNCE,
    ARTWORK_FLAG_CANCEL,
    ARTWORK_MAX_MESSAGE_SIZE,
    ARTWORK_PREFIX_SIZE,
    ARTWORK_RESERVED_FLAGS,
    ClientStateArtwork,
    StreamArtworkChannelConfig,
    StreamStartArtwork,
    unpack_artwork_announce,
)
from aiosendspin.models.controller import ControllerCommandPayload
from aiosendspin.models.core import (
    STREAM_END_ROLE_FAMILIES,
    ActivatePairing,
    ClientCommandMessage,
    ClientCommandPayload,
    ClientGoodbyeMessage,
    ClientGoodbyePayload,
    ClientHelloMessage,
    ClientHelloPayload,
    ClientLeaveMessage,
    ClientStateMessage,
    ClientStatePayload,
    ClientTimeMessage,
    ClientTimePayload,
    DynamicPairMethodDescriptor,
    GroupUpdateServerMessage,
    GroupUpdateServerPayload,
    PairMethodDescriptor,
    ServerActivateMessage,
    ServerActivatePayload,
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
    SupportedPairMethods,
    UnpairedAccess,
)
from aiosendspin.models.management import (
    ManagementAddRecordMessage,
    ManagementGetPairingConfigMessage,
    ManagementListRecordsMessage,
    ManagementOpenPairingWindowMessage,
    ManagementRemoveRecordMessage,
    ManagementResultMessage,
    ManagementResultPayload,
    ManagementSetPairingConfigMessage,
    ServerUnpairMessage,
)
from aiosendspin.models.player import (
    PLAYER_AUDIO_HEADER_SIZE,
    PlayerStatePayload,
    StreamStartPlayer,
    unpack_player_audio_header,
)
from aiosendspin.models.source import (
    ClientStreamEndMessage,
    ClientStreamStartMessage,
    ClientStreamStartPayload,
    ClientStreamStartSource,
    SourceStatePayload,
)
from aiosendspin.models.types import (
    CLOSING_ABORT_REASONS,
    Activity,
    ArtworkSource,
    AudioCodec,
    GoodbyeReason,
    ManagementResult,
    MediaCommand,
    PairAbortReason,
    PairingCodeFormat,
    PairMethod,
    PlayerCommand,
    Roles,
    ServerMessage,
    SignalState,
    UndefinedField,
    role_family,
    undefined_field,
)
from aiosendspin.models.visualizer import StreamStartVisualizer, VisualizerFrame
from aiosendspin.noise.constants import SENTINEL_PSK
from aiosendspin.noise.driver import (
    HandshakeAbortedError,
    InitRejectedError,
    run_handshake_client,
    run_rehandshake_client,
)
from aiosendspin.noise.keys import psk_id_for
from aiosendspin.noise.models import (
    ClientPairPendingMessage,
    ClientPairPendingPayload,
    NoiseHandshakeMessage,
    PairAbortMessage,
    PairAbortPayload,
)
from aiosendspin.noise.pairing import (
    LocalPairingAbortError,
    PairingAbortError,
    abort_pairing,
    receive_pairing_abort,
    run_dynamic_pairing_code_client,
    run_pairing_psk_client,
    run_static_pairing_code_client,
)
from aiosendspin.noise.pairing_code import format_pairing_code
from aiosendspin.noise.trust_store import PskCategory, ResolvedPsk
from aiosendspin.noise.wire import EncryptedWebSocket, QueuedEncryptedWebSocket

from .management import (
    ManagementEffect,
    handle_add_record,
    handle_get_pairing_config,
    handle_list_records,
    handle_open_pairing_window,
    handle_remove_record,
    handle_set_pairing_config,
    handle_unpair,
    with_storage,
)
from .models import AudioFormat, PCMFormat, ServerInfo
from .time_sync import SendspinTimeFilter

if TYPE_CHECKING:
    from .client import SendspinClient

logger = logging.getLogger(__name__)

# Codecs the SDK can decode. Anything else would be silently dropped.
DECODABLE_CODECS = (AudioCodec.PCM, AudioCodec.FLAC)

_ManagementRequest = (
    ManagementListRecordsMessage
    | ManagementAddRecordMessage
    | ManagementRemoveRecordMessage
    | ManagementGetPairingConfigMessage
    | ManagementSetPairingConfigMessage
    | ManagementOpenPairingWindowMessage
)

# A provisional connection must complete bring-up through its first server/activate
# within this window or be dropped (spec: multi-server admission).
PROVISIONAL_CONNECTION_TIMEOUT_S: float = 30.0

# Backstop for a server re-handshake (re-handshake → hello → activate); the per-message
# reads have their own tighter timeouts, this only guards a server that stalls entirely.
REHANDSHAKE_TIMEOUT_S: float = 60.0

# Lead time applied to play-time estimates before clock sync converges.
UNSYNCED_PLAY_LEAD_US: int = 500_000

# psk_id of the Sentinel PSK — the client matches it during pairing-code pairing / discovery.
_SENTINEL_PSK_ID: str = psk_id_for(SENTINEL_PSK)

# Pairing message types a server sends; the reader hands them to the attempt in progress.
_PAIRING_MESSAGE_TYPES: frozenset[str] = frozenset(
    {
        "server/pair-init",
        "server/pair-auth",
        "server/pair-confirm",
        "server/pair-finalize",
        "pair/abort",
    }
)

_ARTWORK_BINARY_TYPES: frozenset[BinaryMessageType] = frozenset(
    {
        BinaryMessageType.ARTWORK_CHANNEL_0,
        BinaryMessageType.ARTWORK_CHANNEL_1,
        BinaryMessageType.ARTWORK_CHANNEL_2,
        BinaryMessageType.ARTWORK_CHANNEL_3,
    }
)
_ARTWORK_CHANNEL_NONE = StreamArtworkChannelConfig(source=ArtworkSource.NONE)

_VISUALIZATION_BINARY_TYPES: frozenset[BinaryMessageType] = frozenset(
    {
        BinaryMessageType.VISUALIZATION_LOUDNESS,
        BinaryMessageType.VISUALIZATION_BEAT,
        BinaryMessageType.VISUALIZATION_F_PEAK,
        BinaryMessageType.VISUALIZATION_SPECTRUM,
        BinaryMessageType.VISUALIZATION_PEAK,
        BinaryMessageType.VISUALIZATION_PITCH,
    }
)

# Binary message IDs the spec leaves to application-specific roles.
_APPLICATION_BINARY_TYPES = range(192, 256)


@dataclass(slots=True)
class _PendingArtwork:
    """An announced image not yet shown on its channel."""

    timestamp_us: int
    total_size: int
    data: bytearray
    received: int = 0
    # Image data arrived while the client was unavailable; the image is never shown.
    discarded: bool = False
    show_handle: asyncio.TimerHandle | None = None


def _malformed_artwork_message(payload: bytes) -> str | None:
    """Return why `payload` is a malformed artwork message, or None when it is well formed."""
    length = len(payload)
    if not ARTWORK_PREFIX_SIZE <= length <= ARTWORK_MAX_MESSAGE_SIZE:
        return f"length {length} outside {ARTWORK_PREFIX_SIZE}-{ARTWORK_MAX_MESSAGE_SIZE}"
    flags = payload[1]
    if flags & ARTWORK_RESERVED_FLAGS:
        return f"flags 0x{flags:02x} set reserved bits"
    if flags & ARTWORK_FLAG_ANNOUNCE and flags & ARTWORK_FLAG_CANCEL:
        return "announce and cancel flags both set"
    if flags & ARTWORK_FLAG_ANNOUNCE and length != ARTWORK_ANNOUNCE_SIZE:
        return f"announce of {length} bytes"
    if flags & ARTWORK_FLAG_CANCEL and length != ARTWORK_PREFIX_SIZE:
        return f"cancel of {length} bytes"
    return None


def _artwork_channel_config(
    artwork: StreamStartArtwork | None, channel: int
) -> StreamArtworkChannelConfig:
    """Return the stream/start configuration of `channel`, uncovered ones as `none`."""
    if artwork is None or channel >= len(artwork.channels):
        return _ARTWORK_CHANNEL_NONE
    return artwork.channels[channel]


def _activities_allowed(
    category: PskCategory, activities: set[Activity], *, unpaired_access: bool
) -> bool:
    """Whether ``activities`` is an allowed set for the matched PSK."""
    if category is PskCategory.LONG_TERM:
        # ['pairing'] alone carries a server's re-verification of the existing pairing.
        return activities == {Activity.PAIRING} or activities <= {
            Activity.PLAYBACK,
            Activity.MANAGEMENT,
        }
    if unpaired_access:
        return activities <= {Activity.PLAYBACK, Activity.PAIRING}
    return activities <= {Activity.PAIRING}


def _admissible(
    category: PskCategory, activities: set[Activity], *, has_roles: bool, unpaired_access: bool
) -> bool:
    """Whether ``activities``/``active_roles`` satisfy the matched PSK's structural constraints."""
    return _activities_allowed(category, activities, unpaired_access=unpaired_access) and (
        # a non-empty active_roles requires a playback-capable connection
        not has_roles
        or _activities_allowed(
            category, activities | {Activity.PLAYBACK}, unpaired_access=unpaired_access
        )
    )


class SendspinConnection:
    """A single connection to one Sendspin server, owning its transport and machinery."""

    _ws: EncryptedWebSocket | None = None
    """Encrypted transport wrapper, installed once the Noise handshake completes."""
    _server_id: str | None = None
    """The server's ``server_id`` (static public key) learned during the handshake."""
    _noise_psk: ResolvedPsk | None = None
    """The PSK that admitted the current connection, with its trust metadata."""
    _handshake_hash: bytes | None = None
    """The Noise handshake hash of the current connection (set during the handshake)."""
    _connected: bool = False
    """Whether the connection is currently live."""
    _server_info: ServerInfo | None = None
    """Information about the connected server."""

    _reader_task: asyncio.Task[None] | None = None
    """Background task reading messages from server."""
    _time_task: asyncio.Task[None] | None = None
    """Background task for time synchronization."""
    _pairing_task: asyncio.Task[None] | None = None
    """Background task running the pairing attempt in progress."""
    _pairing_queue: asyncio.Queue[WSMessage] | None = None
    """Pairing messages the reader routes to the attempt in progress."""

    _output_delay_us: int = 0
    """Output delay in microseconds."""
    _send_lock: asyncio.Lock
    """Lock for serializing WebSocket message sends."""
    _time_filter: SendspinTimeFilter
    """Kalman filter for time synchronization."""

    _current_player: StreamStartPlayer | None = None
    """Current active player configuration."""
    _current_audio_format: AudioFormat | None = None
    """Current audio format for active stream."""
    _stream_active: bool = False
    """True if player stream is active."""
    _visualizer_stream_active: bool = False
    """True if visualizer stream is active."""
    _artwork_stream_active: bool = False
    """True if artwork stream is active."""
    _source_stream_active: bool = False
    """True between client-stream/start and client-stream/end for the source role."""
    _source_start_authorized: bool = False
    """True while a server source ``start`` awaits the client-stream/start it authorizes."""
    _current_visualizer_config: StreamStartVisualizer | None = None
    """Current visualizer config from stream/start."""
    _artwork_config: StreamStartArtwork | None = None
    """Artwork config from the latest stream/start of the active artwork stream."""
    _artwork_in_flight: int | None = None
    """Channel of the artwork transfer announced and not yet complete."""
    _artwork_pending: dict[int, _PendingArtwork]
    """Per channel, the latest announced image until it is shown."""
    _artwork_shown: set[int]
    """Channels currently showing a non-empty image."""
    _protocol_error_task: asyncio.Task[None] | None = None
    """Task closing the connection after a protocol error."""

    _group_state: GroupUpdateServerPayload | None = None
    """Latest group state received from server."""
    _server_state: ServerStatePayload | None = None
    """Latest state of each role object received from server."""

    def __init__(self, client: SendspinClient) -> None:
        """Create a connection owned by ``client``, seeding per-connection state."""
        self._client = client
        self._activities: list[Activity] = []
        self._active_roles: list[str] = []
        self._reported_available: bool = True
        self._initial_state_sent = False
        self._reported_volume = client.initial_volume
        self._reported_muted = client.initial_muted
        self._reported_supported_commands: frozenset[PlayerCommand] = frozenset()
        self._reported_source_signal: SignalState | None = None
        self._selected_pairing: ActivatePairing | None = None
        self._pairing_index = 0
        self._pairing_attempt_in_progress = False
        # Whether cancel_pairing may still end the attempt; cleared once it is finalizing or
        # releasing its out-channels.
        self._pairing_cancellable = False
        self._out_channel_suspended = False
        self._logged_static_pairing_code_dropped = False
        self._exchange_in_progress = False
        self._send_lock = asyncio.Lock()
        self._time_filter = SendspinTimeFilter()
        self._output_delay_us = client.output_delay_us
        self._closed = asyncio.Event()
        self._artwork_pending = {}
        self._artwork_shown = set()

    @property
    def connected(self) -> bool:
        """Return True if the connection is currently live."""
        return self._connected and self._ws is not None and not self._ws.closed

    @property
    def server_id(self) -> str | None:
        """The connected server's ``server_id``, or ``None`` before the handshake."""
        return self._server_id

    @property
    def noise_psk(self) -> ResolvedPsk | None:
        """The PSK that admitted the current connection, or ``None`` if not connected."""
        return self._noise_psk

    async def wait_closed(self) -> None:
        """Block until this connection has fully disconnected."""
        await self._closed.wait()

    @property
    def server_info(self) -> ServerInfo | None:
        """Return information about the connected server, if available."""
        return self._server_info

    @property
    def activities(self) -> list[Activity]:
        """The server's currently-declared activities."""
        return list(self._activities)

    @property
    def output_delay_ms(self) -> float:
        """Return the currently configured output delay in milliseconds."""
        return self._output_delay_us / 1_000.0

    def set_output_delay_ms(self, delay_ms: float) -> None:
        """Update the output delay applied after clock synchronisation."""
        delay_ms = max(0.0, min(5000.0, delay_ms))
        delay_us = round(delay_ms * 1_000.0)
        if delay_us == self._output_delay_us:
            return
        self._output_delay_us = delay_us
        logger.info("Set output delay to %.1f ms", self.output_delay_ms)

    async def connect(
        self, raw_ws: ClientWebSocketResponse, *, expected_server_id: str | None
    ) -> None:
        """Run the handshake over a client-initiated ``raw_ws`` and bring the connection up."""
        await self._bring_up(raw_ws, expected_server_id=expected_server_id)

    async def attach_websocket(
        self, ws: web.WebSocketResponse, *, expected_server_id: str | None
    ) -> None:
        """Run the handshake over an incoming ``ws`` and bring the connection up."""
        await self._bring_up(ws, expected_server_id=expected_server_id)

    async def _bring_up(
        self,
        ws: ClientWebSocketResponse | web.WebSocketResponse,
        *,
        expected_server_id: str | None,
    ) -> None:
        """Reach the first server/activate under a bring-up timeout, closing on stall."""
        try:
            async with asyncio.timeout(PROVISIONAL_CONNECTION_TIMEOUT_S):
                await self._run_noise_handshake(ws, expected_server_id=expected_server_id)
                await self._run_inner_handshake()
        except TimeoutError:
            # Close whatever transport bring-up reached: encrypted if up, else the raw socket.
            if self._connected:
                await self.disconnect()
            else:
                await ws.close()
            raise

    async def _run_noise_handshake(
        self,
        raw_ws: ClientWebSocketResponse | web.WebSocketResponse,
        *,
        expected_server_id: str | None,
    ) -> None:
        """Drive the Noise responder handshake and install the encrypted transport.

        On success ``self._ws`` is the ``EncryptedWebSocket`` and the
        connection is marked live. On failure the raw socket is closed silently
        (spec) and ``HandshakeAbortedError`` propagates to the caller; a server
        that answered with ``server/error`` raises ``InitRejectedError``.
        """
        try:
            result = await run_handshake_client(
                raw_ws,
                local_identity=self._client.identity,
                suite=self._client.cipher_suite,
                psk_resolver=self._resolve_psk,
                expected_server_id=expected_server_id,
            )
        except HandshakeAbortedError as exc:
            if isinstance(exc, InitRejectedError):
                logger.warning("Server rejected the connection: %s", exc)
            await raw_ws.close()
            raise
        self._ws = result.encrypted_ws
        self._server_id = result.peer_id
        self._noise_psk = result.psk
        self._handshake_hash = result.handshake_hash
        self._pairing_index = 0
        self._connected = True
        if result.credential_mismatch:
            logger.warning(
                "Server %s referenced a credential this client cannot use; continuing "
                "unpaired on the Sentinel PSK until re-paired",
                self._server_id,
            )
        if result.psk.category is PskCategory.LONG_TERM:
            await self._client.pairing_store.mark_record_used(result.psk.psk_id)

    async def _resolve_psk(self, psk_id: str, category: PskCategory) -> ResolvedPsk | None:
        """Resolve a ``psk_id`` to a PSK this client holds under ``category``."""
        store = self._client.pairing_store
        if category is PskCategory.SENTINEL:
            if psk_id != _SENTINEL_PSK_ID:
                return None
            return ResolvedPsk(psk_id, SENTINEL_PSK, PskCategory.SENTINEL)
        if category is PskCategory.LONG_TERM:
            record = await store.record_by_psk_id(psk_id)
            return record.as_resolved() if record is not None else None
        pairing = await store.pairing_psk()
        if pairing is None or pairing.psk_id != psk_id:
            return None
        return pairing.as_resolved()

    async def _run_inner_handshake(self) -> None:
        """Bring the connection up to its first server/activate, without pairing or I/O.

        Stops short of pairing and steady-state I/O so a provisional (not-yet-admitted)
        connection never drives the app or runs a pairing exchange; the owning client
        calls ``start`` once this connection is admitted.
        """
        activate = await self._exchange_hellos()
        if (reason := await self._apply_activation(activate)) is not None:
            await self.goodbye_and_disconnect(reason)
            raise RuntimeError(f"server activation rejected ({reason.value})")

    async def _exchange_hellos(self) -> ServerActivatePayload:
        """Exchange hellos with the server and return its server/activate."""
        hello = await self._receive_server_hello()
        assert self._server_id is not None
        self._server_info = ServerInfo(
            server_id=self._server_id,
            name=hello.name,
            languages=tuple(hello.languages or ()),
            source_codecs=(
                frozenset(hello.source_support.supported_codecs)
                if hello.source_support is not None
                else None
            ),
        )
        await self._send_client_hello()
        return await self._receive_server_activate()

    async def _receive_server_hello(self) -> ServerHelloPayload:
        """Read and parse the single ``server/hello`` reply."""
        assert self._ws is not None
        try:
            async with asyncio.timeout(10):
                msg = await self._ws.receive()
        except TimeoutError as err:
            await self.disconnect()
            raise RuntimeError("Timed out waiting for server/hello response") from err
        if msg.type is not WSMsgType.TEXT:
            await self.disconnect()
            raise RuntimeError("Connection closed or non-text frame while awaiting server/hello")
        message = ServerMessage.from_json(msg.data)
        if not isinstance(message, ServerHelloMessage):
            await self.disconnect()
            raise RuntimeError(  # noqa: TRY004 - protocol violation, not a type error
                f"Expected server/hello, got {type(message).__name__}"
            )
        return message.payload

    async def _receive_server_activate(self) -> ServerActivatePayload:
        """Read the ``server/activate`` message."""
        assert self._ws is not None
        msg = await self._ws.receive()
        if msg.type is not WSMsgType.TEXT:
            await self.disconnect()
            raise RuntimeError("Connection closed or non-text frame while awaiting server/activate")
        message = ServerMessage.from_json(msg.data)
        if not isinstance(message, ServerActivateMessage):
            await self.disconnect()
            raise RuntimeError(  # noqa: TRY004 - protocol violation, not a type error
                f"Expected server/activate, got {type(message).__name__}"
            )
        return message.payload

    async def _apply_activation(self, payload: ServerActivatePayload) -> GoodbyeReason | None:
        """Apply a ``server/activate``'s state, or return the goodbye reason that rejects it."""
        assert self._noise_psk is not None
        if payload.ignored_activities:
            # The server speaks a newer spec revision; the known activities still apply.
            logger.info(
                "Ignoring unrecognized server/activate activities: %s",
                ", ".join(payload.ignored_activities),
            )
        category = self._noise_psk.category
        activities = set(payload.activities)
        unpaired_access = await self._unpaired_access_enabled()
        # active_roles is sticky: an omitted set keeps the prior one, unless the connection is
        # no longer playback-capable, which empties it. Gate on this effective set.
        if payload.active_roles is not None:
            effective_roles = payload.active_roles
        elif _activities_allowed(
            category, activities | {Activity.PLAYBACK}, unpaired_access=unpaired_access
        ):
            effective_roles = self._active_roles
        else:
            effective_roles = []
        if category is not PskCategory.LONG_TERM and any(
            role_family(role_id) == "source" for role_id in effective_roles
        ):
            return GoodbyeReason.UNAUTHORIZED
        has_roles = bool(effective_roles)
        if not _admissible(
            category, activities, has_roles=has_roles, unpaired_access=unpaired_access
        ):
            # pairing_required when the session is unpaired and enabling unpaired access
            # would make it admissible.
            if (
                category is not PskCategory.LONG_TERM
                and not unpaired_access
                and _admissible(category, activities, has_roles=has_roles, unpaired_access=True)
            ):
                return GoodbyeReason.PAIRING_REQUIRED
            return GoodbyeReason.UNAUTHORIZED
        self._activities = payload.activities
        source_dropped = (
            Roles.SOURCE.value in self._active_roles and Roles.SOURCE.value not in effective_roles
        )
        self._discard_removed_role_state(effective_roles)
        self._end_removed_role_streams(effective_roles)
        self._active_roles = effective_roles
        if source_dropped:
            self._source_start_authorized = False
            if self._source_stream_active and self.connected:
                await self.send_client_stream_end()
        self._selected_pairing = payload.pairing
        await self._client.note_playback_activity(self)
        return None

    async def start(self) -> None:
        """Start steady-state I/O, and the pairing attempt if the server requested one.

        Called by the owning client only after this connection is admitted.
        """
        if self.is_pairing:
            self._start_pairing_attempt()
        self._reader_task = self._client.loop.create_task(self._reader_loop())
        self._time_task = self._client.loop.create_task(self._time_sync_loop())
        await self._send_full_client_state()

    def _is_role_active(self, family: str) -> bool:
        """Whether the server activated any role in ``family`` for this session."""
        return any(role_family(r) == family for r in self._active_roles)

    async def _send_full_client_state(self) -> None:
        """Push the client's full state to the server, (re)populating its role instances."""
        player_active = Roles.PLAYER in self._client.roles and self._is_role_active("player")
        source_active = self._is_role_active("source")
        if player_active:
            # The player state carries the source object too.
            await self.send_full_player_state()
            self._initial_state_sent = True
        elif source_active and self.is_time_synchronized():
            await self._send_source_state()
            self._initial_state_sent = True
        if player_active or source_active or not self._active_roles:
            return
        await self._send_message(self._client_state_message().to_json())
        self._initial_state_sent = True

    def _start_pairing_attempt(self) -> None:
        """Start the attempt the current pairing activation admits, alongside other traffic."""
        assert self._ws is not None
        self._pairing_index += 1
        self._pairing_cancellable = True
        queue: asyncio.Queue[WSMessage] = asyncio.Queue()
        self._pairing_queue = queue
        self._pairing_task = self._client.loop.create_task(
            self._pair(QueuedEncryptedWebSocket(self._ws, queue), self._pairing_index)
        )

    async def _cancel_pairing_attempt(self) -> None:
        """Abandon the attempt in progress, discarding its state without persisting anything."""
        task = self._pairing_task
        if task is None:
            return
        # A concurrent cancel_pairing must not abort whatever attempt follows this one.
        self._pairing_cancellable = False
        task.cancel()
        await asyncio.wait((task,))
        # A task cancelled before its first step never ran the attempt's own cleanup.
        if self._pairing_task is task:
            self._pairing_task = None
            self._pairing_queue = None

    async def _pair(self, ws: EncryptedWebSocket, pairing_index: int) -> None:
        """Run one pairing attempt; a non-closing abort leaves the connection in pairing.

        On finalize the server re-handshakes onto the new record, which the reader handles.
        """
        try:
            await self._run_pairing_protocol(ws, pairing_index)
        except PairingAbortError as err:
            if err.reason in CLOSING_ABORT_REASONS:
                await self.disconnect()
                return
            logger.info("Pairing attempt with %s ended: %s", self._server_id, err.reason.value)
            self._client.notify_pairing_abort_callback(err.reason)
        except Exception:
            logger.exception("Pairing attempt with %s failed", self._server_id)
            await self.disconnect()
        else:
            assert self._selected_pairing is not None
            logger.info(
                "Paired with server %s via %s", self._server_id, self._selected_pairing.method.value
            )
        finally:
            self._pairing_task = None
            self._pairing_queue = None

    async def _run_pairing_protocol(self, ws: EncryptedWebSocket, pairing_index: int) -> None:
        """Run the server-selected method's exchange over ``ws`` through finalize."""
        assert self._server_id is not None
        pairing = await self._validate_pairing(self._selected_pairing)
        method = pairing.method
        store = self._client.pairing_store
        if method is PairMethod.PAIRING_PSK:
            with self._attempt_in_progress():
                await run_pairing_psk_client(
                    ws,
                    pairing_index=pairing_index,
                    server_id=self._server_id,
                    store=store,
                    on_finalize=self._end_cancellability,
                )
            return
        assert self._handshake_hash is not None
        if method is PairMethod.STATIC_PAIRING_CODE:
            static_pairing_code = await store.static_pairing_code()
            assert static_pairing_code is not None  # offered only when configured
            # Every static-pairing-code attempt is gesture-gated.
            await self._gate_on_pairing_window(ws, pairing_index)
            try:
                with self._attempt_in_progress():
                    await run_static_pairing_code_client(
                        ws,
                        handshake_hash=self._handshake_hash,
                        pairing_index=pairing_index,
                        static_pairing_code=static_pairing_code,
                        server_id=self._server_id,
                        store=store,
                        on_finalize=self._end_cancellability,
                    )
            except LocalPairingAbortError as err:
                # The client aborts with pairing_code_mismatch only when server_kc fails.
                if err.reason is PairAbortReason.PAIRING_CODE_MISMATCH:
                    self._client.record_pairing_window_attempt(self, paired=False)
                raise
            self._client.record_pairing_window_attempt(self, paired=True)
            return
        pairing_format = await self._validate_pairing_format(pairing.format)
        try:
            # Dynamic pairing code is held back only at the round limit, until an operator action.
            if await store.is_pairing_round_limit_reached():
                await self._gate_on_pairing_window(ws, pairing_index)
                await store.reset_pairing_rounds()
                # The operator action is spent on lifting the hold-back.
                self._client.close_pairing_window()
            with self._attempt_in_progress():
                await run_dynamic_pairing_code_client(
                    ws,
                    handshake_hash=self._handshake_hash,
                    pairing_index=pairing_index,
                    pairing_format=pairing_format,
                    pairing_code_emitter=partial(
                        self._emit_pairing_code, pairing_format=pairing_format
                    ),
                    server_id=self._server_id,
                    store=store,
                    on_finalize=self._end_cancellability,
                )
        finally:
            self._end_cancellability()
            await self._emit_pairing_code(None, pairing_format=pairing_format)

    def _end_cancellability(self) -> None:
        """Let the attempt in progress run to its own end, past ``cancel_pairing``."""
        self._pairing_cancellable = False

    @contextmanager
    def _attempt_in_progress(self) -> Iterator[None]:
        """Mark a pairing attempt in-flight for the duration of the method exchange."""
        self._pairing_attempt_in_progress = True
        try:
            yield
        finally:
            self._pairing_attempt_in_progress = False

    async def _gate_on_pairing_window(self, ws: EncryptedWebSocket, pairing_index: int) -> None:
        """Hold a gesture-gated attempt until a pairing window admits it on this connection.

        With no window open, signals ``client/pair-pending`` first and waits, raising if the
        server aborts meanwhile.
        """
        if self._client.pairing_window_admits(self):
            # Claims the window without signalling pair-pending, since none is awaited.
            await self._client.await_pairing_window(self)
            return
        await ws.send_str(
            ClientPairPendingMessage(
                payload=ClientPairPendingPayload(
                    pairing_index=pairing_index, message=self._client.pair_pending_message
                ),
            ).to_json(),
        )
        window = asyncio.ensure_future(self._client.await_pairing_window(self))
        receive = asyncio.create_task(receive_pairing_abort(ws))
        try:
            done, _ = await asyncio.wait((window, receive), return_when=asyncio.FIRST_COMPLETED)
        finally:
            window.cancel()
            receive.cancel()
            # Let both finish unwinding: the window wait's cleanup clears the gesture prompt.
            await asyncio.wait((window, receive))
        # raise by awaiting if the task fails
        if window in done:
            await window
        if receive in done:
            await receive

    async def _supported_pair_methods(self) -> tuple[PairMethod, ...]:
        """Methods this client advertises: each implemented method that config enables.

        ``static_pairing_code`` additionally requires a configured pairing code.
        """
        implemented = self._client.implemented_pair_methods
        config = await self._client.pairing_store.get_pairing_config()
        methods: list[PairMethod] = []
        if config.pairing_psk_enabled:
            methods.append(PairMethod.PAIRING_PSK)
        if (
            PairMethod.STATIC_PAIRING_CODE in implemented
            and config.static_pairing_code_enabled
            and await self._client.pairing_store.static_pairing_code() is not None
        ):
            methods.append(PairMethod.STATIC_PAIRING_CODE)
        if PairMethod.DYNAMIC_PAIRING_CODE in implemented and config.dynamic_pairing_code_enabled:
            methods.append(PairMethod.DYNAMIC_PAIRING_CODE)
            methods = self._without_static_pairing_code(methods)
        return tuple(methods)

    def _without_static_pairing_code(self, methods: list[PairMethod]) -> list[PairMethod]:
        """Drop ``static_pairing_code``, so only the dynamic pairing-code method is offered."""
        if PairMethod.STATIC_PAIRING_CODE not in methods:
            return methods
        if not self._logged_static_pairing_code_dropped:
            self._logged_static_pairing_code_dropped = True
            logger.info(
                "Offering dynamic_pairing_code only: a client should offer one pairing-code "
                "method, so the configured static pairing code goes unused"
            )
        return [m for m in methods if m is not PairMethod.STATIC_PAIRING_CODE]

    async def _unpaired_access_enabled(self) -> bool:
        """Whether the client currently admits unpaired access (from pairing config)."""
        return (await self._client.pairing_store.get_pairing_config()).unpaired_access_enabled

    async def _validate_pairing(self, pairing: ActivatePairing | None) -> ActivatePairing:
        """Reject a pairing whose method the matched PSK disallows or the client did not offer."""
        assert self._noise_psk is not None
        method = pairing.method if pairing is not None else None
        # pairing_psk iff the matched PSK is the Pairing PSK; a pairing-code method otherwise.
        method_fits_psk = (method is PairMethod.PAIRING_PSK) == (
            self._noise_psk.category is PskCategory.PAIRING
        )
        supported = await self._supported_pair_methods()
        if pairing is None or not method_fits_psk or method not in supported:
            await self._abort_pairing(PairAbortReason.METHOD_NOT_SUPPORTED)
        return pairing

    async def _validate_pairing_format(self, pairing_format: str | None) -> PairingCodeFormat:
        """Validate that the activation selects a currently offered dynamic format.

        An identifier from a newer spec revision does not parse, so it aborts like any
        other format this client does not offer.
        """
        try:
            selected = PairingCodeFormat(pairing_format)
        except ValueError:
            await self._abort_pairing(PairAbortReason.METHOD_NOT_SUPPORTED)
        if selected not in await self._dynamic_pairing_formats():
            await self._abort_pairing(PairAbortReason.METHOD_NOT_SUPPORTED)
        return selected

    async def _dynamic_pairing_formats(self) -> tuple[PairingCodeFormat, ...]:
        """Return formats currently offered by this client."""
        formats = []
        if (
            self._client.pairing_code_display is not None
            or self._client.pairing_code_speaker is not None
        ):
            formats.append(PairingCodeFormat.DIGITS)
        if self._client.qr_code_display is not None:
            formats.append(PairingCodeFormat.QR_CODE)
        return tuple(formats)

    async def _abort_pairing(self, reason: PairAbortReason) -> NoReturn:
        """Send ``pair/abort``; never returns (the abort raises)."""
        assert self._ws is not None
        await abort_pairing(self._ws, reason)

    async def _emit_pairing_code(
        self, pairing_code: str | None, *, pairing_format: PairingCodeFormat
    ) -> None:
        """Hand ``pairing_code`` to the format's out-channels, or release them when ``None``.

        The out-channel is suspended before the first code is emitted and resumed on release.
        """
        suspend = self._client.out_channel_suspend
        if pairing_code is not None and not self._out_channel_suspended:
            self._out_channel_suspended = True
            if suspend is not None:
                await suspend(True)  # noqa: FBT003
        try:
            await self._emit_to_out_channels(pairing_code, pairing_format=pairing_format)
        finally:
            if pairing_code is None and self._out_channel_suspended:
                self._out_channel_suspended = False
                if suspend is not None:
                    await suspend(False)  # noqa: FBT003

    async def _emit_to_out_channels(
        self, pairing_code: str | None, *, pairing_format: PairingCodeFormat
    ) -> None:
        """Hand ``pairing_code`` to the format's out-channels."""
        if pairing_format is PairingCodeFormat.QR_CODE:
            assert self._client.qr_code_display is not None  # validated as offered
            await self._client.qr_code_display(pairing_code)
            return
        emissions = []
        if self._client.pairing_code_display is not None:
            grouped = format_pairing_code(pairing_code) if pairing_code is not None else None
            emissions.append(self._client.pairing_code_display(pairing_code, grouped=grouped))
        if self._client.pairing_code_speaker is not None:
            emissions.append(
                self._client.pairing_code_speaker(pairing_code, languages=self._server_languages())
            )
        await asyncio.gather(*emissions)

    def _server_languages(self) -> tuple[str, ...]:
        """Operator language preferences declared in the server/hello."""
        return self._server_info.languages if self._server_info is not None else ()

    async def _rehandshake(self, hs1_text: str) -> None:
        """Re-run the Noise handshake as responder from message 1 and swap the session."""
        assert self._ws is not None
        assert self._server_id is not None
        assert self._handshake_hash is not None
        result = await run_rehandshake_client(
            self._ws,
            local_identity=self._client.identity,
            server_id=self._server_id,
            suite=self._ws.session.suite,
            prologue=self._handshake_hash,
            psk_resolver=self._resolve_psk,
            hs1_text=hs1_text,
        )
        self._noise_psk = result.psk
        self._handshake_hash = result.handshake_hash
        self._pairing_index = 0

    async def goodbye_and_disconnect(self, reason: GoodbyeReason) -> None:
        """Send ``client/goodbye`` with ``reason`` and disconnect."""
        await self.send_goodbye(reason)
        await self.disconnect()

    async def send_goodbye(self, reason: GoodbyeReason) -> None:
        """Send a client/goodbye message to the server before disconnecting."""
        if not self.connected:
            return
        message = ClientGoodbyeMessage(
            payload=ClientGoodbyePayload(reason=reason),
        )
        # A goodbye precedes disconnect, so it must reach the wire even mid-exchange.
        await self._send_message(message.to_json(), force=True)

    async def cancel_pairing(self) -> None:
        """Cancel the pairing attempt in progress, as on operator cancellation.

        Ends the attempt, whether started or awaiting a pairing window, closes the pairing
        window and sends ``pair/abort`` with reason ``user_cancelled``. The connection stays
        open. A no-op when no attempt is in progress, or once it has sent
        ``client/pair-finalize`` or is releasing its out-channels: it then runs to its own end.
        Must not be called from a ``PairingSupport`` callback, which runs inside the attempt.
        """
        if self._pairing_task is None or not self._pairing_cancellable:
            return
        self._client.close_pairing_window()
        # Ending the attempt clears its out-channels and stops its sends before the abort.
        await self._cancel_pairing_attempt()
        await self.send_pair_abort(PairAbortReason.USER_CANCELLED)

    async def send_pair_abort(self, reason: PairAbortReason) -> None:
        """Send a pair/abort to the server, even while an in-band exchange owns the wire."""
        if not self.connected:
            return
        message = PairAbortMessage(payload=PairAbortPayload(reason=reason))
        # A displaced connection's own exchange flag is still set; send regardless (spec).
        await self._send_message(message.to_json(), force=True)

    @property
    def is_pairing(self) -> bool:
        """Whether this connection is currently a pairing connection."""
        return Activity.PAIRING in self._activities

    @property
    def relies_on_unpaired_access(self) -> bool:
        """Whether this connection is unpaired and its activities or roles need unpaired access."""
        psk = self._noise_psk
        if psk is None or psk.category is PskCategory.LONG_TERM:
            return False
        return not _admissible(
            psk.category,
            set(self._activities),
            has_roles=bool(self._active_roles),
            unpaired_access=False,
        )

    @property
    def pairing_attempt_in_progress(self) -> bool:
        """Whether a pairing attempt is mid-flight."""
        return self._pairing_attempt_in_progress

    async def disconnect(self) -> None:
        """Disconnect from the server and release resources (idempotent)."""
        if not self._connected:
            return
        self._connected = False

        current_task = asyncio.current_task(loop=self._client.loop)
        if self._pairing_task is not None and self._pairing_task is not current_task:
            self._pairing_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._pairing_task
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
        self._time_filter.reset()
        self._server_info = None
        self._server_id = None
        self._noise_psk = None
        self._group_state = None
        self._server_state = None
        self._stream_active = False
        self._current_audio_format = None
        self._current_player = None
        self._artwork_stream_active = False
        self._reset_artwork()
        self._artwork_shown.clear()
        self._visualizer_stream_active = False
        self._source_stream_active = False
        self._source_start_authorized = False
        self._current_visualizer_config = None
        self._activities = []
        self._active_roles = []

        self._closed.set()
        self._client.on_connection_closed(self)

    async def send_full_player_state(self) -> None:
        """Resend the last reported player state when the player role is active."""
        if Roles.PLAYER in self._client.roles and self._is_role_active("player"):
            await self.send_player_state(
                available=self._reported_available,
                volume=self._reported_volume,
                muted=self._reported_muted,
            )

    async def send_player_state(
        self,
        *,
        available: bool,
        volume: int,
        muted: bool,
    ) -> None:
        """Send the current player state to the server."""
        if not self.connected:
            raise RuntimeError("Client is not connected")
        await self._update_reported_available(available=available)
        self._reported_volume = volume
        self._reported_muted = muted
        self._reported_supported_commands = frozenset(self._client.state_supported_commands)
        message = self._client_state_message(
            player=PlayerStatePayload(
                volume=volume,
                muted=muted,
                output_delay_ms=round(self._output_delay_us / 1_000),
                required_lead_time_ms=round(self._client.required_lead_time_ms),
                min_buffer_ms=round(self._client.min_buffer_ms),
                supported_commands=list(self._client.state_supported_commands),
                format=self._client.preferred_format,
            ),
        )
        await self._send_message(message.to_json())

    async def send_available(self, *, available: bool) -> None:
        """Send the current client-level availability."""
        if not self.connected:
            raise RuntimeError("Client is not connected")
        await self._update_reported_available(available=available)
        if self._is_role_active("source"):
            if not self.is_time_synchronized():
                return
            await self._send_source_state()
            return
        await self._send_message(self._client_state_message().to_json())

    async def send_artwork_state(self) -> None:
        """
        Report the artwork channel configuration when the artwork role is active.

        An active source withholds client/state until its clock synchronizes; the
        configuration is then reported with that state.
        """
        if self._is_role_active("artwork"):
            await self.send_available(available=self._reported_available)

    async def send_visualizer_state(self) -> None:
        """
        Report the requested visualizer configuration when the visualizer role is active.

        An active source withholds client/state until its clock synchronizes; the
        configuration is then reported with that state.
        """
        if self._is_role_active("visualizer"):
            await self.send_available(available=self._reported_available)

    def _client_state_message(
        self, *, player: PlayerStatePayload | None = None
    ) -> ClientStateMessage:
        """Build a client/state with availability and the objects of the active roles."""
        visualizer = self._client.visualizer_state if self._is_role_active("visualizer") else None
        return ClientStateMessage(
            payload=ClientStatePayload(
                available=self._wire_available(),
                player=player,
                source=self._active_source_state(),
                artwork=self._active_artwork_state(),
                visualizer=visualizer,
            )
        )

    def _active_source_state(self) -> SourceStatePayload | None:
        """Return the source object every client/state carries while the role is active."""
        if not self._is_role_active("source"):
            return None
        return SourceStatePayload(signal=self._reported_source_signal)

    def _active_artwork_state(self) -> ClientStateArtwork | None:
        """Return the artwork object every client/state carries while the role is active."""
        return self._client.artwork_state if self._is_role_active("artwork") else None

    async def _update_reported_available(self, *, available: bool) -> None:
        if not available:
            self._source_start_authorized = False
            if self._source_stream_active:
                await self.send_client_stream_end()
        self._reported_available = available

    async def send_leave(self) -> None:
        """Send client/leave to leave the current group."""
        if not self.connected:
            raise RuntimeError("Client is not connected")
        await self._send_message(ClientLeaveMessage().to_json())

    def _wire_available(self) -> bool:
        """Return the availability to report; an active player is unavailable until synced."""
        return self._reported_available and (
            not self._is_role_active("player") or self.is_time_synchronized()
        )

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
        if not self.connected:
            raise RuntimeError("Client is not connected")
        controller = None if self._server_state is None else self._server_state.controller
        if controller is None or isinstance(controller, UndefinedField):
            raise ValueError("No controller state has been received from the server")
        if command not in controller.supported_commands:
            raise ValueError(f"Command '{command.value}' is not supported by the server")
        controller_payload = ControllerCommandPayload(
            command=command,
            volume=volume,
            mute=mute,
            position_ms=position_ms,
            offset_ms=offset_ms,
        )
        if (
            controller_payload.position_ms is not None
            and controller.seek_max_ms is not None
            and controller_payload.position_ms > controller.seek_max_ms
        ):
            raise ValueError(
                f"position_ms must be at most seek_max_ms ({controller.seek_max_ms}), "
                f"got {controller_payload.position_ms}"
            )
        payload = ClientCommandPayload(controller=controller_payload)
        message = ClientCommandMessage(payload=payload)
        await self._send_message(message.to_json())

    async def send_client_stream_start(
        self,
        *,
        codec: AudioCodec,
        sample_rate: int,
        channels: int,
        bit_depth: int,
        codec_header: str | None,
    ) -> None:
        """
        Start a source stream, consuming the pending server source ``start``.

        Raises RuntimeError unless a server ``start`` is pending (see
        ``is_source_start_authorized()``).
        """
        message = ClientStreamStartMessage(
            payload=ClientStreamStartPayload(
                source=ClientStreamStartSource(
                    codec=codec,
                    channels=channels,
                    sample_rate=sample_rate,
                    bit_depth=bit_depth,
                    codec_header=codec_header,
                )
            )
        )
        async with self._send_lock:
            self._ensure_source_authorized()
            if not self._source_start_authorized:
                raise RuntimeError("Source stream start requires a server start command")
            if not self.is_time_synchronized():
                raise RuntimeError("Source capture requires a synchronized clock")
            if self._exchange_in_progress:
                raise RuntimeError("Connection is busy with an in-band exchange")
            await self._send_message_locked(message.to_json())
            self._source_start_authorized = False
            self._source_stream_active = True

    async def send_client_stream_end(self) -> None:
        """End the source stream; a new server ``start`` is required to open another."""
        async with self._send_lock:
            self._source_start_authorized = False
            if not self.connected:
                raise RuntimeError("Client is not connected")
            if not self._source_stream_active:
                return
            if self._exchange_in_progress:
                raise RuntimeError("Connection is busy with an in-band exchange")
            await self._send_message_locked(ClientStreamEndMessage().to_json())
            self._source_stream_active = False

    async def send_source_chunk(self, frame: bytes, *, timestamp_us: int) -> None:
        """Send an encoded source audio frame."""
        header = pack_binary_header_raw(BinaryMessageType.SOURCE_AUDIO_CHUNK.value, timestamp_us)
        async with self._send_lock:
            self._ensure_source_authorized(require_stream=True)
            await self._send_bytes_locked(header + frame)

    async def send_source_signal(self, signal: SignalState) -> None:
        """
        Report source signal presence.

        Raises RuntimeError unless the client advertised the ``line_sense`` feature.
        """
        support = self._client.source_support
        if support is None or support.features is None or not support.features.line_sense:
            raise RuntimeError("Source signal requires the line_sense feature")
        self._ensure_source_authorized()
        self._reported_source_signal = signal
        if not self.is_time_synchronized():
            return
        await self._send_source_state()

    async def _send_source_state(self) -> None:
        """Send current source state."""
        await self._send_message(self._client_state_message().to_json())

    def _ensure_source_authorized(self, *, require_stream: bool = False) -> None:
        if not self.connected:
            raise RuntimeError("Client is not connected")
        if self._noise_psk is None or self._noise_psk.category is not PskCategory.LONG_TERM:
            raise RuntimeError("Source role requires a paired connection")
        if not self._is_role_active("source"):
            raise RuntimeError("Source role is not active")
        if require_stream and not self._source_stream_active:
            raise RuntimeError("Source stream is not active")

    def is_time_synchronized(self) -> bool:
        """Return whether time synchronization with the server has converged."""
        return self._time_filter.is_synchronized

    async def _build_client_hello(self) -> ClientHelloMessage:
        player_support = self._client.player_support
        if player_support is not None:
            # DEPRECATED(spec-pr-177): remove in aiosendspin <version>
            # Player commands are declared in client/state, never in the hello.
            player_support = replace(player_support, supported_commands=None)
        payload = ClientHelloPayload(
            name=self._client.client_name,
            supported_roles=[r.value for r in self._client.roles],
            device_info=self._client.device_info,
            player_support=player_support,
            visualizer_support=self._client.visualizer_support,
            source_support=self._client.source_support,
            supported_pair_methods=await self._build_supported_pair_methods(),
            unpaired_access=UnpairedAccess(enabled=await self._unpaired_access_enabled()),
        )
        return ClientHelloMessage(payload=payload)

    async def _build_supported_pair_methods(self) -> SupportedPairMethods:
        """Build the ``client/hello`` advertisement of the methods this client offers."""
        methods = await self._supported_pair_methods()
        offered = SupportedPairMethods()
        if PairMethod.PAIRING_PSK in methods:
            offered.pairing_psk = self._secret_method_descriptor()
        if PairMethod.STATIC_PAIRING_CODE in methods:
            offered.static_pairing_code = self._secret_method_descriptor()
        if PairMethod.DYNAMIC_PAIRING_CODE in methods:
            offered.dynamic_pairing_code = DynamicPairMethodDescriptor(
                out_channels=list(self._client.pairing_code_out_channels),
                formats=[f.value for f in await self._dynamic_pairing_formats()],
            )
        return offered

    def _secret_method_descriptor(self) -> PairMethodDescriptor:
        """Build the descriptor for a method whose secret the operator looks up."""
        locations = self._client.secret_locations
        return PairMethodDescriptor(locations=list(locations) if locations else None)

    async def _send_client_hello(self) -> None:
        assert self._ws is not None
        hello = await self._build_client_hello()
        await self._ws.send_str(hello.to_json())

    async def _send_time_message(self) -> None:
        if not self.connected:
            return
        now_us = self.now_us()
        message = ClientTimeMessage(payload=ClientTimePayload(client_transmitted=now_us))
        await self._send_message(message.to_json())

    async def _send_message(self, payload: str, *, force: bool = False) -> None:
        """Send a JSON frame; ``force`` bypasses the in-band-exchange suppression."""
        async with self._send_lock:
            await self._send_message_locked(payload, force=force)

    async def _send_message_locked(self, payload: str, *, force: bool = False) -> None:
        if self._ws is None:
            raise RuntimeError("WebSocket is not connected")
        if self._exchange_in_progress and not force:
            return
        await self._ws.send_str(payload)

    async def _send_bytes(self, payload: bytes) -> None:
        async with self._send_lock:
            await self._send_bytes_locked(payload)

    async def _send_bytes_locked(self, payload: bytes) -> None:
        if self._ws is None:
            raise RuntimeError("WebSocket is not connected")
        if self._exchange_in_progress:
            return
        await self._ws.send_bytes(payload)

    def is_source_stream_active(self) -> bool:
        """Return whether this connection has an open source stream."""
        return self._source_stream_active

    def is_source_start_authorized(self) -> bool:
        """Return whether a server source ``start`` is pending for ``SourceCapture.start()``."""
        return self._source_start_authorized

    @asynccontextmanager
    async def _exchange(self) -> AsyncIterator[None]:
        """Reserve the wire for an in-band exchange: suppress other sends, then drain in-flight."""
        self._exchange_in_progress = True
        async with self._send_lock:  # wait out any send already in flight
            pass
        try:
            yield
        finally:
            self._exchange_in_progress = False

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
            await self.disconnect()

    async def _handle_ws_message(self, msg: WSMessage) -> None:
        if msg.type is WSMsgType.TEXT:
            await self._handle_json_message(msg.data)
        elif msg.type is WSMsgType.BINARY:
            self._handle_binary_message(msg.data)
        elif msg.type is WSMsgType.ERROR:
            logger.error("WebSocket error: %s", self._ws.exception() if self._ws else "unknown")
            await self.disconnect()

    @staticmethod
    def _peek_message_type(data: str) -> str | None:
        """Return the envelope ``type`` of ``data``, or ``None`` if it has none."""
        try:
            decoded = orjson.loads(data)
        except orjson.JSONDecodeError:
            return None
        message_type = decoded.get("type") if isinstance(decoded, dict) else None
        return message_type if isinstance(message_type, str) else None

    async def _handle_json_message(self, data: str) -> None:
        try:
            message = ServerMessage.from_json(data)
        except Exception:
            message_type = self._peek_message_type(data)
            if message_type in _PAIRING_MESSAGE_TYPES:
                # A malformed one still reaches the attempt, which fails on it.
                await self._route_pairing_message(data, message_type)
                return
            logger.exception("Failed to parse server message: %s", data)
            return

        match message:
            case NoiseHandshakeMessage():
                await self._handle_handshake(data)
            case ServerActivateMessage(payload=payload):
                await self._handle_server_activate(payload)
            case ServerTimeMessage(payload=payload):
                await self._handle_server_time(payload)
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
                await self._handle_server_command(payload)
            case ServerUnpairMessage():
                await self._handle_unpair()
            case (
                ManagementListRecordsMessage()
                | ManagementAddRecordMessage()
                | ManagementRemoveRecordMessage()
                | ManagementGetPairingConfigMessage()
                | ManagementSetPairingConfigMessage()
                | ManagementOpenPairingWindowMessage()
            ):
                await self._handle_management_request(message)
            case _:
                logger.debug("Unhandled server message type: %s", type(message).__name__)

    async def _route_pairing_message(self, data: str, message_type: str) -> None:
        """Hand a pairing message to the attempt in progress, or discard it."""
        if self._pairing_queue is None or self._pairing_task is None:
            # In flight from before the server observed our pair/abort or leave.
            logger.debug("Discarding pairing message: no attempt in progress")
            return
        self._pairing_queue.put_nowait(WSMessage(WSMsgType.TEXT, data, ""))
        if message_type == "server/pair-finalize":
            # The re-handshake that follows needs the record the attempt persists on this ack.
            await asyncio.wait((self._pairing_task,))

    def _handle_binary_message(self, payload: bytes) -> None:
        if len(payload) < 1:
            logger.warning("Empty binary message")
            return

        raw_type = payload[0]
        if raw_type in _APPLICATION_BINARY_TYPES:
            self._client.notify_application_binary(raw_type, payload[1:])
            return
        try:
            message_type = BinaryMessageType(raw_type)
        except ValueError:
            logger.warning("Unknown binary message type: %s", raw_type)
            return

        if message_type is BinaryMessageType.AUDIO_CHUNK:
            role_active = self._stream_active
        elif message_type in _ARTWORK_BINARY_TYPES:
            role_active = self._artwork_stream_active
        elif message_type in _VISUALIZATION_BINARY_TYPES:
            role_active = self._visualizer_stream_active
        else:
            role_active = False

        if message_type in _ARTWORK_BINARY_TYPES and (
            reason := _malformed_artwork_message(payload)
        ):
            self._close_on_protocol_error(f"malformed artwork message: {reason}")
            return

        if not role_active:
            logger.debug(
                "Ignoring binary message of type %s since its role stream is inactive",
                message_type,
            )
            return

        if message_type is BinaryMessageType.AUDIO_CHUNK:
            if len(payload) < PLAYER_AUDIO_HEADER_SIZE:
                logger.warning("Dropping truncated audio chunk of %d bytes", len(payload))
                return
            header = unpack_player_audio_header(payload)
            self._handle_audio_chunk(
                header.timestamp_us, payload[PLAYER_AUDIO_HEADER_SIZE:], header.send_ahead
            )
        elif message_type in _ARTWORK_BINARY_TYPES:
            self._handle_artwork_chunk(message_type, payload)
        elif message_type is BinaryMessageType.VISUALIZATION_BEAT:
            self._handle_visualization_beat(payload[1:])
        elif message_type in _VISUALIZATION_BINARY_TYPES:
            self._handle_visualization_frame(message_type, payload[1:])
        else:
            logger.debug("Ignoring unsupported binary message type: %s", message_type)

    async def _handle_handshake(self, data: str) -> None:
        """Run a server-initiated re-handshake, ending any pairing attempt still in progress."""
        await self._cancel_pairing_attempt()
        async with self._exchange(), asyncio.timeout(REHANDSHAKE_TIMEOUT_S):
            await self._rehandshake(data)
            activate = await self._receive_server_activate()
        await self._handle_server_activate(activate, resync=True)

    async def _handle_server_activate(
        self, payload: ServerActivatePayload, *, resync: bool = False
    ) -> None:
        # Any server/activate ends the attempt in progress; a pairing one admits the next.
        await self._cancel_pairing_attempt()
        was_player_active = self._is_role_active("player")
        was_source_active = self._is_role_active("source")
        was_artwork_active = self._is_role_active("artwork")
        was_visualizer_active = self._is_role_active("visualizer")
        if (reason := await self._apply_activation(payload)) is not None:
            await self.goodbye_and_disconnect(reason)
            return
        if self.is_pairing:
            self._start_pairing_attempt()
        self._resume_time_sync()
        player_activated = not was_player_active and self._is_role_active("player")
        source_activated = not was_source_active and self._is_role_active("source")
        artwork_activated = not was_artwork_active and self._is_role_active("artwork")
        visualizer_activated = not was_visualizer_active and self._is_role_active("visualizer")
        initial_state_due = bool(self._active_roles) and not self._initial_state_sent
        if (
            resync
            or player_activated
            or source_activated
            or artwork_activated
            or visualizer_activated
            or initial_state_due
        ):
            await self._send_full_client_state()

    def _resume_time_sync(self) -> None:
        """Restart the time-sync loop if it is not running."""
        if self._time_task is None or self._time_task.done():
            self._time_task = self._client.loop.create_task(self._time_sync_loop())

    async def _handle_server_time(self, payload: ServerTimePayload) -> None:
        was_synchronized = self._time_filter.is_synchronized
        now_us = self.now_us()
        offset = (
            (payload.server_received - payload.client_transmitted)
            + (payload.server_transmitted - now_us)
        ) / 2
        delay = (
            (now_us - payload.client_transmitted)
            - (payload.server_transmitted - payload.server_received)
        ) / 2
        self._time_filter.update(round(offset), round(delay), now_us)
        if (
            not was_synchronized
            and self._time_filter.is_synchronized
            and (self._is_role_active("player") or self._is_role_active("source"))
        ):
            await self._send_full_client_state()

    async def _handle_stream_start(self, message: StreamStartMessage) -> None:
        if message.payload.visualizer is not None:
            self._current_visualizer_config = message.payload.visualizer
            self._visualizer_stream_active = True
        if message.payload.artwork is not None:
            self._artwork_stream_active = True
            for channel in list(self._artwork_pending):
                if _artwork_channel_config(message.payload.artwork, channel) != (
                    _artwork_channel_config(self._artwork_config, channel)
                ):
                    self._discard_pending_artwork(channel)
            self._artwork_config = message.payload.artwork

        player = message.payload.player
        if player is None:
            # stream/start without player payload - may be for other roles only
            if (
                message.payload.visualizer is not None
                or message.payload.artwork is not None
                or message.payload.application_objects
            ):
                self._client.notify_stream_start(message)
            else:
                logger.debug("Stream start message without player payload")
            return

        if player.codec not in DECODABLE_CODECS:
            logger.error(
                "Unsupported codec '%s' - only PCM and FLAC are supported", player.codec.value
            )
            return

        is_format_update = self._stream_active and self._current_player is not None
        if is_format_update:
            logger.info("Stream format updated to %s", player.codec.value)
        else:
            logger.info("Stream started with codec %s", player.codec.value)
            self._stream_active = True

        pcm_format = PCMFormat(
            sample_rate=player.sample_rate,
            channels=player.channels,
            bit_depth=player.bit_depth,
        )
        codec_header_bytes: bytes | None = None
        if player.codec_header:
            codec_header_bytes = base64.b64decode(player.codec_header)

        self._configure_audio_output(
            AudioFormat(
                codec=player.codec,
                pcm_format=pcm_format,
                codec_header=codec_header_bytes,
            )
        )
        self._current_player = StreamStartPlayer(
            codec=player.codec,
            sample_rate=player.sample_rate,
            channels=player.channels,
            bit_depth=player.bit_depth,
            codec_header=player.codec_header,
        )

        if not is_format_update:
            self._client.notify_stream_start(message)
            await self._send_time_message()

    def _handle_stream_clear(self, message: StreamClearMessage) -> None:
        roles = message.payload.roles
        logger.debug("Stream clear received for roles: %s", roles or "all")
        self._client.notify_stream_clear(roles)

    def _handle_stream_end(self, message: StreamEndMessage) -> None:
        roles = message.payload.roles
        logger.debug("Stream ended for roles: %s", roles or "all")

        self._end_streams(roles)
        self._client.notify_stream_end(roles)

    def _end_streams(self, roles: list[str] | None) -> None:
        """Mark the streams of `roles` (all when `None`) inactive and drop their state."""
        if roles is None or "player" in roles:
            self._stream_active = False
            self._current_player = None
            self._current_audio_format = None

        # If roles is None or includes visualizer role, end the visualizer stream
        if roles is None or "visualizer" in roles:
            self._visualizer_stream_active = False
            self._current_visualizer_config = None
        if roles is None or "artwork" in roles:
            self._artwork_stream_active = False
            self._reset_artwork()
            for channel in sorted(self._artwork_shown):
                self._client.notify_artwork(channel, b"")
            self._artwork_shown.clear()

    def _end_removed_role_streams(self, active_roles: list[str]) -> None:
        """End the server-to-client streams of every active role missing from `active_roles`."""
        # Dispatched even when the stream already ended: the embedder may still be draining
        # output or holding an effect such as ducking. Whether an application-specific role
        # carries a stream is unknown here, so those are always included.
        ended = sorted(
            {
                family
                for role_id in set(self._active_roles) - set(active_roles)
                if (family := role_family(role_id)) in STREAM_END_ROLE_FAMILIES
                or family.startswith("_")
            }
        )
        if not ended:
            return
        self._end_streams(ended)
        self._client.notify_stream_end(ended)

    def _handle_group_update(self, payload: GroupUpdateServerPayload) -> None:
        self._group_state = payload
        self._client.notify_group_callback(payload)

    def _discard_removed_role_state(self, active_roles: list[str]) -> None:
        """Discard the server/state object of every active role missing from `active_roles`."""
        state = self._server_state
        removed = {role_family(role_id) for role_id in set(self._active_roles) - set(active_roles)}
        if state is None or not removed:
            return
        notifiers = {
            "metadata": self._client.notify_metadata_callback,
            "controller": self._client.notify_controller_callback,
            "color": self._client.notify_color_callback,
        }
        discarded = [
            family
            for family in notifiers
            if family in removed and not isinstance(getattr(state, family), UndefinedField)
        ]
        self._server_state = replace(
            state,
            **dict.fromkeys(discarded, undefined_field()),
            application_objects={
                key: value for key, value in state.application_objects.items() if key not in removed
            },
        )
        for family in discarded:
            notifiers[family](None)

    def _handle_server_state(self, payload: ServerStatePayload) -> None:
        self._server_state = (
            payload if self._server_state is None else self._server_state.merge(payload)
        )
        if not isinstance(payload.controller, UndefinedField):
            self._client.notify_controller_callback(payload)
        if not isinstance(payload.metadata, UndefinedField):
            self._client.notify_metadata_callback(payload)
        if not isinstance(payload.color, UndefinedField):
            self._client.notify_color_callback(payload)

    async def _handle_server_command(self, payload: ServerCommandPayload) -> None:
        """Handle server/command message."""
        if payload.source is not None and not await self._apply_source_command(
            payload.source.command
        ):
            logger.debug("Ignoring source command without effect: %s", payload.source.command)
            if payload.player is None:
                return
            payload = replace(payload, source=None)
        if (
            payload.player is not None
            and payload.player.command not in self._reported_supported_commands
        ):
            logger.debug("Ignoring unsupported player command: %s", payload.player.command)
            if payload.source is None:
                return
            payload = replace(payload, player=None)
        if payload.player is not None:
            player_cmd = payload.player
            if (
                player_cmd.command
                in (PlayerCommand.SET_OUTPUT_DELAY, PlayerCommand.SET_STATIC_DELAY)
                and player_cmd.output_delay_ms is not None
            ):
                self._client.set_output_delay_ms(float(player_cmd.output_delay_ms))
        self._client.notify_server_command_callback(payload)

    async def _apply_source_command(self, command: Literal["start", "stop"]) -> bool:
        """Apply a server source command and return whether it had an effect."""
        if command == "start":
            if (
                self._source_start_authorized
                or self._source_stream_active
                or not self._wire_available()
                or not self._is_role_active("source")
            ):
                return False
            self._source_start_authorized = True
            return True
        # A stop also ends a stream whose client-stream/start is still in flight.
        if not self._source_start_authorized and not self._source_stream_active:
            return False
        self._source_start_authorized = False
        if self.connected:
            await self.send_client_stream_end()
        return True

    async def _handle_unpair(self) -> None:
        """Handle server/unpair: drop the matched record (unless shared) and close."""
        if self._noise_psk is None or self._noise_psk.category is not PskCategory.LONG_TERM:
            return  # Not a long-term session (pairing / unpaired): ignore and continue.
        await handle_unpair(self._client.pairing_store, matched_psk_id=self._noise_psk.psk_id)
        await self.goodbye_and_disconnect(GoodbyeReason.UNPAIRED)

    async def _handle_management_request(self, message: _ManagementRequest) -> None:
        """Handle a management/* request, gating on the management activity."""
        if Activity.MANAGEMENT not in self._activities:
            await self._send_message(
                ManagementResultMessage(
                    payload=ManagementResultPayload(result=ManagementResult.PERMISSION_DENIED)
                ).to_json()
            )
            return
        store = self._client.pairing_store
        match message:
            case ManagementListRecordsMessage():
                payload, effect = await handle_list_records(store)
            case ManagementAddRecordMessage(payload=request):
                payload, effect = await handle_add_record(store, request)
            case ManagementRemoveRecordMessage(payload=request):
                assert self._noise_psk is not None
                payload, effect = await handle_remove_record(
                    store,
                    request,
                    requester_psk_id=self._noise_psk.psk_id,
                )
            case ManagementGetPairingConfigMessage():
                payload, effect = await handle_get_pairing_config(
                    store, implemented_pair_methods=self._client.implemented_pair_methods
                )
            case ManagementSetPairingConfigMessage(payload=request):
                payload, effect = await handle_set_pairing_config(
                    store,
                    request,
                    implemented_pair_methods=self._client.implemented_pair_methods,
                    config_lock=self._client.admission_lock,
                )
            case ManagementOpenPairingWindowMessage():
                payload, effect = await handle_open_pairing_window(
                    store,
                    implemented_pair_methods=self._client.implemented_pair_methods,
                    open_window=self._client.open_pairing_window,
                )
            case _:
                assert_never(message)
        payload = await with_storage(
            payload,
            store,
            include_static=isinstance(
                message, ManagementListRecordsMessage | ManagementGetPairingConfigMessage
            ),
        )
        await self._send_message(ManagementResultMessage(payload=payload).to_json())
        if effect is ManagementEffect.GOODBYE_UNAUTHORIZED:
            await self.goodbye_and_disconnect(GoodbyeReason.UNAUTHORIZED)

    def _configure_audio_output(self, audio_format: AudioFormat) -> None:
        """Store the current audio format for use in callbacks."""
        self._current_audio_format = audio_format

    def _handle_audio_chunk(self, timestamp_us: int, payload: bytes, send_ahead: int) -> None:
        """Handle incoming audio chunk and notify callbacks."""
        if self._current_audio_format is None:
            logger.debug("Dropping audio chunk without format")
            return
        if not self._reported_available:
            return
        # Pass server timestamp directly to callback - it handles time conversion
        # to allow for dynamic time base updates
        self._client.notify_audio_chunk(
            timestamp_us, payload, self._current_audio_format, send_ahead
        )

    def _handle_artwork_chunk(self, message_type: BinaryMessageType, payload: bytes) -> None:
        """Apply a well-formed artwork announce, part or cancel of the active stream."""
        channel = message_type.value - BinaryMessageType.ARTWORK_CHANNEL_0.value
        flags = payload[1]
        if flags & ARTWORK_FLAG_CANCEL:
            self._discard_pending_artwork(channel)
            return
        if flags & ARTWORK_FLAG_ANNOUNCE:
            if self._artwork_in_flight is not None:
                self._close_on_protocol_error("artwork announce while a transfer is in flight")
                return
            announce = unpack_artwork_announce(payload)
            self._discard_pending_artwork(channel)
            pending = _PendingArtwork(
                timestamp_us=announce.timestamp_us,
                total_size=announce.total_size,
                data=bytearray(),
                # An empty image carries no data to discard, so a clear always applies.
                discarded=announce.total_size > 0 and not self._reported_available,
            )
            self._artwork_pending[channel] = pending
            self._artwork_in_flight = channel
        else:
            if self._artwork_in_flight != channel:
                self._close_on_protocol_error(
                    f"artwork part on channel {channel} with no transfer in flight there"
                )
                return
            pending = self._artwork_pending[channel]
            data = payload[ARTWORK_PREFIX_SIZE:]
            pending.received += len(data)
            if pending.received > pending.total_size:
                self._close_on_protocol_error("artwork part extends past total_size")
                return
            if pending.discarded or not self._reported_available:
                pending.discarded = True
                pending.data.clear()
            else:
                pending.data += data
        if pending.received == pending.total_size:
            self._artwork_in_flight = None
            self._schedule_artwork(channel, pending)

    def _schedule_artwork(self, channel: int, pending: _PendingArtwork) -> None:
        """Show a complete pending image once its timestamp is reached on the local clock."""
        if pending.discarded:
            del self._artwork_pending[channel]
            return
        delay_us = 0
        if self._time_filter.count > 0:
            delay_us = self._time_filter.compute_client_time(pending.timestamp_us) - self.now_us()
        if delay_us <= 0:
            self._show_artwork(channel)
            return
        pending.show_handle = self._client.loop.call_later(
            delay_us / 1_000_000, self._show_artwork, channel
        )

    def _show_artwork(self, channel: int) -> None:
        """Make the channel's pending image current and notify the listeners."""
        image = bytes(self._artwork_pending.pop(channel).data)
        if image:
            self._artwork_shown.add(channel)
        else:
            self._artwork_shown.discard(channel)
        self._client.notify_artwork(channel, image)

    def _discard_pending_artwork(self, channel: int) -> None:
        """Discard the channel's pending image, ending its transfer if in flight."""
        pending = self._artwork_pending.pop(channel, None)
        if pending is not None and pending.show_handle is not None:
            pending.show_handle.cancel()
        if self._artwork_in_flight == channel:
            self._artwork_in_flight = None

    def _reset_artwork(self) -> None:
        """Discard every pending image and the stream's artwork configuration."""
        for channel in list(self._artwork_pending):
            self._discard_pending_artwork(channel)
        self._artwork_in_flight = None
        self._artwork_config = None

    def _close_on_protocol_error(self, reason: str) -> None:
        """Close the connection because the server violated the protocol."""
        logger.error("Closing connection on protocol error: %s", reason)
        if self._protocol_error_task is None or self._protocol_error_task.done():
            self._protocol_error_task = self._client.loop.create_task(self.disconnect())

    def _handle_visualization_frame(self, message_type: BinaryMessageType, payload: bytes) -> None:
        """Parse a single-type visualization binary and notify callbacks."""
        if self._current_visualizer_config is None or not self._reported_available:
            return
        try:
            frame = self._parse_visualization_frame(
                message_type, payload, self._current_visualizer_config
            )
        except Exception:
            logger.exception("Failed to parse visualization frame")
            return
        if frame is not None and not self._is_visualization_late(frame.timestamp_us):
            self._client.notify_visualizer_callbacks([frame])

    @staticmethod
    def _parse_visualization_frame(
        message_type: BinaryMessageType,
        data: bytes,
        config: StreamStartVisualizer,
    ) -> VisualizerFrame | None:
        """Parse a `[ts:8][data]` payload for one of the v1 visualizer types."""
        if len(data) < 8:
            return None
        (timestamp_us,) = struct.unpack_from(">q", data, 0)
        rest = data[8:]

        if message_type is BinaryMessageType.VISUALIZATION_LOUDNESS:
            if len(rest) != 2:
                return None
            (value,) = struct.unpack(">H", rest)
            return VisualizerFrame(timestamp_us=timestamp_us, loudness=value)
        if message_type is BinaryMessageType.VISUALIZATION_F_PEAK:
            if len(rest) != 4:
                return None
            freq, amp = struct.unpack(">HH", rest)
            return VisualizerFrame(timestamp_us=timestamp_us, f_peak_freq=freq, f_peak_amp=amp)
        if message_type is BinaryMessageType.VISUALIZATION_SPECTRUM:
            n_disp_bins = config.spectrum.n_disp_bins if config.spectrum is not None else 0
            if n_disp_bins <= 0 or len(rest) != n_disp_bins * 2:
                return None
            bins = list(struct.unpack(f">{n_disp_bins}H", rest))
            return VisualizerFrame(timestamp_us=timestamp_us, spectrum=bins)
        if message_type is BinaryMessageType.VISUALIZATION_PEAK:
            if len(rest) != 1:
                return None
            return VisualizerFrame(timestamp_us=timestamp_us, peak_strength=rest[0])
        if message_type is BinaryMessageType.VISUALIZATION_PITCH:
            if len(rest) != 3:
                return None
            (midi_q88,) = struct.unpack(">H", rest[:2])
            return VisualizerFrame(
                timestamp_us=timestamp_us,
                pitch_midi_q88=midi_q88,
                pitch_confidence=rest[2],
            )
        return None

    def _handle_visualization_beat(self, payload: bytes) -> None:
        """Dispatch a `beat` binary (`[ts:8][flags:1]`) as a timestamp + is_downbeat frame."""
        if len(payload) != 9 or not self._reported_available:
            return
        try:
            (ts,) = struct.unpack_from(">q", payload, 0)
        except Exception:
            logger.exception("Failed to parse beat data")
            return
        if self._is_visualization_late(ts):
            return
        is_downbeat = bool(payload[8] & 0b0000_0001)
        self._client.notify_visualizer_callbacks(
            [VisualizerFrame(timestamp_us=ts, is_downbeat=is_downbeat)]
        )

    def _is_visualization_late(self, timestamp_us: int) -> bool:
        """Return whether visualization data is already past on the local clock."""
        return (
            self._time_filter.count > 0
            and self._time_filter.compute_client_time(timestamp_us) < self.now_us()
        )

    def compute_play_time(self, server_timestamp_us: int) -> int:
        """Convert a server timestamp to client play time, with output delay applied."""
        if self._time_filter.is_synchronized:
            client_time = self._time_filter.compute_client_time(server_timestamp_us)
            return client_time - self._output_delay_us
        return self.now_us() + UNSYNCED_PLAY_LEAD_US - self._output_delay_us

    def compute_server_time(self, client_timestamp_us: int) -> int:
        """Convert a client timestamp to a server timestamp, with output delay removed."""
        adjusted_client_time = client_timestamp_us + self._output_delay_us
        return self._time_filter.compute_server_time(adjusted_client_time)

    def current_track_position(self) -> int | None:
        """Return the playback position in milliseconds as of now, or None when unknown.

        The position is extrapolated from the progress in the latest metadata received,
        including metadata whose timestamp is still in the future. Returns None without
        progress or before time synchronization has converged.
        """
        metadata = None if self._server_state is None else self._server_state.metadata
        if (
            metadata is None
            or isinstance(metadata, UndefinedField)
            or metadata.progress is None
            or not self._time_filter.is_synchronized
        ):
            return None
        progress = metadata.progress
        server_now_us = self._time_filter.compute_server_time(self.now_us())
        elapsed_us = server_now_us - metadata.timestamp
        position = progress.track_progress + elapsed_us * progress.playback_speed // 1_000_000
        if progress.track_duration != 0:
            return max(min(position, progress.track_duration), 0)
        return max(position, 0)

    def compute_source_timestamp(self, capture_timestamp_us: int) -> int:
        """Convert a capture timestamp to server time without playback delay."""
        return self._time_filter.compute_server_time(capture_timestamp_us)

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

    def now_us(self) -> int:
        """Return current timestamp from the client's clock in microseconds."""
        return self._client.clock.now_us()
