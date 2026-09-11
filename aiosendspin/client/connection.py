"""A single connection from a ``SendspinClient`` to one Sendspin server."""

from __future__ import annotations

import asyncio
import base64
import logging
import struct
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager, suppress
from functools import partial
from typing import TYPE_CHECKING, NoReturn, assert_never

from aiohttp import ClientWebSocketResponse, WSMessage, WSMsgType, web

from aiosendspin.models import (
    BINARY_HEADER_SIZE,
    BinaryMessageType,
    pack_binary_header_raw,
    unpack_binary_header,
)
from aiosendspin.models.controller import ControllerCommandPayload
from aiosendspin.models.core import (
    ActivatePairing,
    ClientCommandMessage,
    ClientCommandPayload,
    ClientGoodbyeMessage,
    ClientGoodbyePayload,
    ClientHelloMessage,
    ClientHelloPayload,
    ClientStateMessage,
    ClientStatePayload,
    ClientTimeMessage,
    ClientTimePayload,
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
from aiosendspin.models.player import PlayerStatePayload, StreamStartPlayer
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
    TrustLevel,
    UndefinedField,
    role_family,
)
from aiosendspin.models.visualizer import StreamStartVisualizer, VisualizerFrame
from aiosendspin.noise.constants import SENTINEL_PSK
from aiosendspin.noise.driver import (
    HandshakeAbortedError,
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
    PairingMessage,
)
from aiosendspin.noise.pairing import (
    PairingAbortError,
    abort_pairing,
    receive_pairing_abort,
    run_dynamic_pairing_code_client,
    run_pairing_psk_client,
    run_static_pairing_code_client,
)
from aiosendspin.noise.trust_store import PskCategory, ResolvedPsk
from aiosendspin.noise.wire import EncryptedWebSocket

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

# Backstop for the post-pairing transition (re-handshake → hello → activate); the per-message
# reads have their own tighter timeouts, this only guards a server that stalls entirely.
POST_PAIRING_ACTIVATE_TIMEOUT_S: float = 60.0

# Lead time applied to play-time estimates before clock sync converges.
UNSYNCED_PLAY_LEAD_US: int = 500_000

# psk_id of the Sentinel PSK — the client matches it during pairing-code pairing / discovery.
_SENTINEL_PSK_ID: str = psk_id_for(SENTINEL_PSK)

_ARTWORK_BINARY_TYPES: frozenset[BinaryMessageType] = frozenset(
    {
        BinaryMessageType.ARTWORK_CHANNEL_0,
        BinaryMessageType.ARTWORK_CHANNEL_1,
        BinaryMessageType.ARTWORK_CHANNEL_2,
        BinaryMessageType.ARTWORK_CHANNEL_3,
    }
)
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


def _activities_allowed(
    category: PskCategory, activities: set[Activity], *, unpaired_access: bool
) -> bool:
    """Whether ``activities`` is an allowed set for the matched PSK."""
    if Activity.PAIRING in activities:
        return activities == {Activity.PAIRING}
    if category is PskCategory.LONG_TERM:
        return activities <= {Activity.PLAYBACK, Activity.MANAGEMENT}
    if category is PskCategory.SENTINEL:
        if not activities:
            return True
        return activities == {Activity.PLAYBACK} and unpaired_access
    return False  # Pairing PSK admits only {'pairing'}, handled above.


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

    _static_delay_us: int = 0
    """Static playback delay in microseconds."""
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
    _current_visualizer_config: StreamStartVisualizer | None = None
    """Current visualizer config from stream/start."""

    _group_state: GroupUpdateServerPayload | None = None
    """Latest group state received from server."""
    _server_state: ServerStatePayload | None = None
    """Latest server state received from server."""

    def __init__(self, client: SendspinClient) -> None:
        """Create a connection owned by ``client``, seeding per-connection state."""
        self._client = client
        self._activities: list[Activity] = []
        self._active_roles: list[str] = []
        self._reported_available: bool = True
        self._reported_volume = client.initial_volume
        self._reported_muted = client.initial_muted
        self._reported_source_signal: SignalState | None = None
        self._selected_pairing: ActivatePairing | None = None
        self._pairing_index = 0
        self._pairing_attempt_in_progress = False
        self._exchange_in_progress = False
        self._send_lock = asyncio.Lock()
        self._time_filter = SendspinTimeFilter()
        self._static_delay_us = client.static_delay_us
        self._closed = asyncio.Event()

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
    def static_delay_ms(self) -> float:
        """Return the currently configured static playback delay in milliseconds."""
        return self._static_delay_us / 1_000.0

    def set_static_delay_ms(self, delay_ms: float) -> None:
        """Update the static playback delay applied after clock synchronisation."""
        delay_ms = max(0.0, min(5000.0, delay_ms))
        delay_us = round(delay_ms * 1_000.0)
        if delay_us == self._static_delay_us:
            return
        self._static_delay_us = delay_us
        logger.info("Set static playback delay to %.1f ms", self.static_delay_ms)

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
        (spec) and ``HandshakeAbortedError`` propagates to the caller.
        """
        try:
            result = await run_handshake_client(
                raw_ws,
                local_identity=self._client.identity,
                suite=self._client.cipher_suite,
                psk_resolver=self._resolve_psk,
                expected_server_id=expected_server_id,
            )
        except HandshakeAbortedError:
            await raw_ws.close()
            raise
        self._ws = result.encrypted_ws
        self._server_id = result.peer_id
        self._noise_psk = result.psk
        self._handshake_hash = result.handshake_hash
        self._pairing_index = 0
        self._connected = True
        if result.psk.category is PskCategory.LONG_TERM:
            await self._client.pairing_store.mark_record_used(result.psk.psk_id)

    async def _resolve_psk(self, psk_id: str) -> ResolvedPsk | None:
        """Resolve a ``psk_id`` to its PSK, matching the Sentinel PSK or a stored credential."""
        if psk_id == _SENTINEL_PSK_ID:
            return ResolvedPsk(psk_id, SENTINEL_PSK, PskCategory.SENTINEL)
        return await self._client.pairing_store.resolve_by_psk_id(psk_id)

    async def _run_inner_handshake(self) -> None:
        """Bring the connection up to its first server/activate, without pairing or I/O.

        Stops short of pairing and steady-state I/O so a provisional (not-yet-admitted)
        connection never drives the app or runs a pairing exchange; the owning client
        calls ``start`` once this connection is admitted.
        """
        activate = await self._exchange_hellos()
        if (reason := await self._apply_activation(activate)) is not None:
            await self._goodbye_and_disconnect(reason)
            raise RuntimeError(f"server activation rejected ({reason.value})")

    async def _exchange_hellos(self) -> ServerActivatePayload:
        """Exchange hellos with the server and return its server/activate."""
        hello = await self._receive_server_hello()
        assert self._server_id is not None
        self._server_info = ServerInfo(
            server_id=self._server_id,
            name=hello.name,
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
        category = self._noise_psk.category
        activities = set(payload.activities)
        # active_roles is sticky: an omitted set keeps the prior one, so gate on the
        # effective set, not just what this message declares.
        effective_roles = (
            payload.active_roles if payload.active_roles is not None else self._active_roles
        )
        if category is not PskCategory.LONG_TERM and any(
            role_family(role_id) == "source" for role_id in effective_roles
        ):
            return GoodbyeReason.UNAUTHORIZED
        has_roles = bool(effective_roles)
        unpaired_access = await self._unpaired_access_enabled()
        if not _admissible(
            category, activities, has_roles=has_roles, unpaired_access=unpaired_access
        ):
            # pairing_required when enabling unpaired access would make it admissible.
            if (
                category is PskCategory.SENTINEL
                and not unpaired_access
                and _admissible(category, activities, has_roles=has_roles, unpaired_access=True)
            ):
                return GoodbyeReason.PAIRING_REQUIRED
            return GoodbyeReason.UNAUTHORIZED
        self._activities = payload.activities
        if payload.active_roles is not None:
            source_dropped = (
                Roles.SOURCE.value in self._active_roles
                and Roles.SOURCE.value not in payload.active_roles
            )
            self._active_roles = payload.active_roles
            if source_dropped and self._source_stream_active and self.connected:
                await self.send_client_stream_end()
        self._selected_pairing = payload.pairing
        await self._client.note_playback_activity(self)
        return None

    async def start(self) -> None:
        """Run pairing if the server requested it, then start steady-state I/O.

        Called by the owning client only after this connection is admitted.
        """
        if self.is_pairing:
            try:
                await self._pair()
            except BaseException:
                await self.disconnect()
                raise
            if not self.connected:
                return
            self._reader_task = self._client.loop.create_task(self._reader_loop())
        else:
            self._reader_task = self._client.loop.create_task(self._reader_loop())
            self._time_task = self._client.loop.create_task(self._time_sync_loop())
            await self._send_full_client_state()

    def _is_role_active(self, family: str) -> bool:
        """Whether the server activated any role in ``family`` for this session."""
        return any(role_family(r) == family for r in self._active_roles)

    async def _send_full_client_state(self) -> None:
        """Push the client's full state to the server, (re)populating its role instances."""
        if Roles.PLAYER in self._client.roles and self._is_role_active("player"):
            await self.send_player_state(
                available=self._reported_available,
                volume=self._reported_volume,
                muted=self._reported_muted,
            )
        if self._is_role_active("source") and self.is_time_synchronized():
            await self._send_source_state()

    async def _pair(self) -> None:
        """Run one pairing attempt; on a non-closing abort stay in pairing for a retry."""
        await self._pause_time_sync()
        while True:
            try:
                async with self._exchange():
                    leftover = await self._run_pairing_protocol()
                    activate = await self._resolve_pairing_activate(leftover)
            except PairingAbortError as err:
                if err.reason in CLOSING_ABORT_REASONS:
                    await self.disconnect()
                else:
                    logger.info(
                        "Pairing attempt with %s ended: %s", self._server_id, err.reason.value
                    )
                    self._client.notify_pairing_abort_callback(err.reason)
                return
            if (reason := await self._apply_activation(activate)) is not None:
                await self._goodbye_and_disconnect(reason)
                return
            if self.is_pairing:
                # The leave activate redeclared pairing: it admits the next attempt.
                continue
            self._resume_time_sync()
            await self._send_full_client_state()
            return

    async def _run_pairing_protocol(self) -> str | None:
        """Run the server-selected method's exchange.

        Returns ``None`` on finalize, else the raw ``server/activate`` leave frame
        (from the method run, or from the pairing window closing before an attempt).
        """
        assert self._ws is not None
        assert self._server_id is not None
        self._pairing_index += 1
        pairing_index = self._pairing_index
        pairing = await self._validate_pairing(self._selected_pairing)
        method = pairing.method
        store = self._client.pairing_store
        if method is PairMethod.PAIRING_PSK:
            with self._attempt_in_progress():
                return await run_pairing_psk_client(
                    self._ws, server_id=self._server_id, store=store
                )
        assert self._handshake_hash is not None
        if method is PairMethod.STATIC_PAIRING_CODE:
            static_pairing_code = await store.static_pairing_code()
            assert static_pairing_code is not None  # offered only when configured
            # Every static-pairing-code attempt is gesture-gated.
            if (leave := await self._gate_on_pairing_window(pairing_index)) is not None:
                return leave
            with self._attempt_in_progress():
                return await run_static_pairing_code_client(
                    self._ws,
                    handshake_hash=self._handshake_hash,
                    pairing_index=pairing_index,
                    static_pairing_code=static_pairing_code,
                    server_id=self._server_id,
                    store=store,
                )
        # Dynamic pairing code is gesture-gated only after the failure counter escalates.
        pairing_format = await self._validate_pairing_format(pairing.format)
        if (
            await store.is_pairing_code_escalated()
            and (leave := await self._gate_on_pairing_window(pairing_index)) is not None
        ):
            return leave
        self._client.consume_pairing_window()
        try:
            with self._attempt_in_progress():
                return await run_dynamic_pairing_code_client(
                    self._ws,
                    handshake_hash=self._handshake_hash,
                    pairing_index=pairing_index,
                    pairing_format=pairing_format,
                    pairing_code_emitter=partial(
                        self._emit_pairing_code, pairing_format=pairing_format
                    ),
                    server_id=self._server_id,
                    store=store,
                )
        finally:
            await self._emit_pairing_code(None, pairing_format=pairing_format)

    @contextmanager
    def _attempt_in_progress(self) -> Iterator[None]:
        """Mark a pairing attempt in-flight for the duration of the method exchange."""
        self._pairing_attempt_in_progress = True
        try:
            yield
        finally:
            self._pairing_attempt_in_progress = False

    async def _gate_on_pairing_window(self, pairing_index: int) -> str | None:
        """Hold a gesture-gated attempt until a pairing window is open.

        With no window open, signals ``client/pair-pending`` first and waits without
        leaving the socket unread. Returns the raw ``server/activate`` frame if the
        server leaves pairing first, else ``None``.
        """
        assert self._ws is not None
        if self._client.pairing_window_open:
            # Claims the window without signalling pair-pending, since none is awaited.
            await self._client.await_pairing_window()
            return None
        await self._ws.send_str(
            ClientPairPendingMessage(
                payload=ClientPairPendingPayload(pairing_index=pairing_index),
            ).to_json(),
        )
        window = asyncio.ensure_future(self._client.await_pairing_window())
        receive = asyncio.create_task(receive_pairing_abort(self._ws))
        try:
            done, _ = await asyncio.wait((window, receive), return_when=asyncio.FIRST_COMPLETED)
        finally:
            window.cancel()
            receive.cancel()
        if window not in done:
            # Let the window wait finish unwinding (its cleanup clears the gesture prompt).
            await asyncio.wait((window,))
        if receive not in done:
            # The cancelled receive must exit ws.receive() before anyone else may read.
            await asyncio.wait((receive,))
        # raise by awaiting if the task fails
        if window in done:
            await window
        if receive in done:
            return await receive
        return None

    async def _resolve_pairing_activate(self, leftover: str | None) -> ServerActivatePayload:
        """Resolve the server/activate that ends pairing, re-handshaking first if it finalized."""
        pairing = self._selected_pairing
        assert pairing is not None
        async with asyncio.timeout(POST_PAIRING_ACTIVATE_TIMEOUT_S):
            if leftover is None:
                # Server finalized and re-handshakes onto the new long-term PSK.
                logger.info("Paired with server %s via %s", self._server_id, pairing.method.value)
                await self._rehandshake()
                return await self._exchange_hellos()
            # Server left pairing without finalizing: apply the leave server/activate.
            logger.info("Server %s left pairing without finalizing", self._server_id)
            message = ServerMessage.from_json(leftover)
            if not isinstance(message, ServerActivateMessage):
                raise RuntimeError(  # noqa: TRY004 - protocol violation, not a type error
                    f"Expected server/activate ending pairing, got {type(message).__name__}"
                )
            return message.payload

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
        return tuple(methods)

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
        """Hand ``pairing_code`` to the format's out-channels, or release them when ``None``."""
        if pairing_format is PairingCodeFormat.QR_CODE:
            assert self._client.qr_code_display is not None  # validated as offered
            await self._client.qr_code_display(pairing_code)
            return
        emissions = []
        if self._client.pairing_code_display is not None:
            emissions.append(self._client.pairing_code_display(pairing_code))
        if self._client.pairing_code_speaker is not None:
            emissions.append(
                self._client.pairing_code_speaker(
                    pairing_code, languages=self._activation_languages()
                )
            )
        await asyncio.gather(*emissions)

    def _activation_languages(self) -> tuple[str, ...]:
        """Operator language preferences carried by the pairing activation."""
        pairing = self._selected_pairing
        if pairing is None or pairing.languages is None:
            return ()
        return tuple(pairing.languages)

    async def _rehandshake(self, hs1_text: str | None = None) -> None:
        """Re-run the Noise handshake as responder and swap the session (None reads msg 1)."""
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

    async def _goodbye_and_disconnect(self, reason: GoodbyeReason) -> None:
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

    async def send_pair_abort(self, reason: PairAbortReason) -> None:
        """Send a pair/abort to the server (for a pairing connection lost to arbitration)."""
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
    def pairing_attempt_in_progress(self) -> bool:
        """Whether a pairing attempt is mid-flight."""
        return self._pairing_attempt_in_progress

    async def disconnect(self) -> None:
        """Disconnect from the server and release resources (idempotent)."""
        if not self._connected:
            return
        self._connected = False

        current_task = asyncio.current_task(loop=self._client.loop)
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
        self._visualizer_stream_active = False
        self._source_stream_active = False
        self._current_visualizer_config = None
        self._activities = []
        self._active_roles = []

        self._closed.set()
        self._client.on_connection_closed(self)

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
        message = ClientStateMessage(
            payload=ClientStatePayload(
                available=available,
                player=PlayerStatePayload(
                    volume=volume,
                    muted=muted,
                    static_delay_ms=round(self._static_delay_us / 1_000),
                    required_lead_time_ms=round(self._client.required_lead_time_ms),
                    min_buffer_ms=round(self._client.min_buffer_ms),
                    supported_commands=self._client.state_supported_commands or None,
                ),
            )
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
        message = ClientStateMessage(payload=ClientStatePayload(available=available))
        await self._send_message(message.to_json())

    async def _update_reported_available(self, *, available: bool) -> None:
        if not available and self._source_stream_active:
            await self.send_client_stream_end()
        self._reported_available = available

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
        if not self.connected:
            raise RuntimeError("Client is not connected")
        controller_payload = ControllerCommandPayload(
            command=command,
            volume=volume,
            mute=mute,
            position_ms=position_ms,
            offset_ms=offset_ms,
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
        """Start a source stream."""
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
            if not self.is_time_synchronized():
                raise RuntimeError("Source capture requires a synchronized clock")
            if self._exchange_in_progress:
                raise RuntimeError("Connection is busy with an in-band exchange")
            await self._send_message_locked(message.to_json())
            self._source_stream_active = True

    async def send_client_stream_end(self) -> None:
        """End the source stream."""
        async with self._send_lock:
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
        """Report source signal presence."""
        self._ensure_source_authorized()
        self._reported_source_signal = signal
        if not self.is_time_synchronized():
            return
        await self._send_source_state()

    async def _send_source_state(self) -> None:
        """Send current source state."""
        message = ClientStateMessage(
            payload=ClientStatePayload(
                available=self._reported_available,
                source=SourceStatePayload(signal=self._reported_source_signal),
            )
        )
        await self._send_message(message.to_json())

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
        payload = ClientHelloPayload(
            name=self._client.client_name,
            supported_roles=[r.value for r in self._client.roles],
            device_info=self._client.device_info,
            player_support=self._client.player_support,
            artwork_support=self._client.artwork_support,
            visualizer_support=self._client.visualizer_support,
            source_support=self._client.source_support,
            trust_level=self._compute_trust(),
            supported_pair_methods=[
                await self._pair_method_descriptor(m) for m in await self._supported_pair_methods()
            ],
            unpaired_access=UnpairedAccess(enabled=await self._unpaired_access_enabled()),
        )
        return ClientHelloMessage(payload=payload)

    async def _pair_method_descriptor(self, method: PairMethod) -> PairMethodDescriptor:
        """Build the ``client/hello`` descriptor for ``method``."""
        if method is not PairMethod.DYNAMIC_PAIRING_CODE:
            locations = self._client.secret_locations
            return PairMethodDescriptor(
                method=method, locations=list(locations) if locations else None
            )
        out_channels = self._client.pairing_code_out_channels
        return PairMethodDescriptor(
            method=method,
            formats=[f.value for f in await self._dynamic_pairing_formats()],
            out_channels=list(out_channels) if out_channels else None,
        )

    def _compute_trust(self) -> TrustLevel:
        """Trust extended to the reached server: ``user`` when paired, else ``none``."""
        if self._noise_psk is not None and self._noise_psk.category is PskCategory.LONG_TERM:
            return TrustLevel.USER
        return TrustLevel.NONE

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
    def _is_pairing_frame(data: str) -> bool:
        """Whether ``data`` parses as a pairing message."""
        with suppress(Exception):
            PairingMessage.from_json(data)
            return True
        return False

    async def _handle_json_message(self, data: str) -> None:
        try:
            message = ServerMessage.from_json(data)
        except Exception:
            if self._is_pairing_frame(data):
                # In flight from before the server observed our pair/abort.
                logger.debug("Discarding pairing message: no attempt in progress")
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
                self._handle_server_command(payload)
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

    def _handle_binary_message(self, payload: bytes) -> None:
        if len(payload) < 1:
            logger.warning("Empty binary message")
            return

        raw_type = payload[0]
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

        if not role_active:
            logger.debug(
                "Ignoring binary message of type %s since its role stream is inactive",
                message_type,
            )
            return

        if message_type is BinaryMessageType.AUDIO_CHUNK:
            try:
                header = unpack_binary_header(payload)
            except Exception:
                logger.exception("Failed to unpack binary header")
                return
            self._handle_audio_chunk(header.timestamp_us, payload[BINARY_HEADER_SIZE:])
        elif message_type in _ARTWORK_BINARY_TYPES:
            try:
                unpack_binary_header(payload)
            except Exception:
                logger.exception("Failed to unpack binary header")
                return
            self._handle_artwork_chunk(message_type, payload[BINARY_HEADER_SIZE:])
        elif message_type is BinaryMessageType.VISUALIZATION_BEAT:
            self._handle_visualization_beat(payload[1:])
        elif message_type in _VISUALIZATION_BINARY_TYPES:
            self._handle_visualization_frame(message_type, payload[1:])
        else:
            logger.debug("Ignoring unsupported binary message type: %s", message_type)

    async def _handle_handshake(self, data: str) -> None:
        """Run a server-initiated re-handshake that arrives outside a pairing exchange."""
        async with self._exchange():
            await self._rehandshake(data)
            activate = await self._exchange_hellos()
        await self._handle_server_activate(activate, resync=True)

    async def _handle_server_activate(
        self, payload: ServerActivatePayload, *, resync: bool = False
    ) -> None:
        was_player_active = self._is_role_active("player")
        was_source_active = self._is_role_active("source")
        if (reason := await self._apply_activation(payload)) is not None:
            await self._goodbye_and_disconnect(reason)
            return
        if self.is_pairing:
            await self._pair()
            return
        self._resume_time_sync()
        player_activated = not was_player_active and self._is_role_active("player")
        source_activated = not was_source_active and self._is_role_active("source")
        if resync or player_activated or source_activated:
            await self._send_full_client_state()

    async def _pause_time_sync(self) -> None:
        if self._time_task is not None and not self._time_task.done():
            self._time_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._time_task
            self._time_task = None

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
            and self._is_role_active("source")
        ):
            await self._send_source_state()

    async def _handle_stream_start(self, message: StreamStartMessage) -> None:
        if message.payload.visualizer is not None:
            self._current_visualizer_config = message.payload.visualizer
            self._visualizer_stream_active = True
        if message.payload.artwork is not None:
            self._artwork_stream_active = True

        player = message.payload.player
        if player is None:
            # stream/start without player payload - may be for artwork/visualizer only
            if message.payload.visualizer is not None or message.payload.artwork is not None:
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

        self._client.notify_stream_end(roles)

    def _handle_group_update(self, payload: GroupUpdateServerPayload) -> None:
        self._group_state = payload
        self._client.notify_group_callback(payload)

    def _handle_server_state(self, payload: ServerStatePayload) -> None:
        self._server_state = payload
        if not isinstance(payload.controller, UndefinedField):
            self._client.notify_controller_callback(payload)
        if not isinstance(payload.metadata, UndefinedField):
            self._client.notify_metadata_callback(payload)
        if not isinstance(payload.color, UndefinedField):
            self._client.notify_color_callback(payload)

    def _handle_server_command(self, payload: ServerCommandPayload) -> None:
        """Handle server/command message."""
        if payload.player is not None:
            player_cmd = payload.player
            if (
                player_cmd.command == PlayerCommand.SET_STATIC_DELAY
                and player_cmd.static_delay_ms is not None
            ):
                self.set_static_delay_ms(float(player_cmd.static_delay_ms))
        self._client.notify_server_command_callback(payload)

    async def _handle_unpair(self) -> None:
        """Handle server/unpair: drop the matched record (unless shared) and close."""
        if self._noise_psk is None or self._noise_psk.category is not PskCategory.LONG_TERM:
            return  # trust_level 'none' (pairing / unpaired handshake): ignore and continue.
        await handle_unpair(self._client.pairing_store, matched_psk_id=self._noise_psk.psk_id)
        await self._goodbye_and_disconnect(GoodbyeReason.UNPAIRED)

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
                    store, request, implemented_pair_methods=self._client.implemented_pair_methods
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
            await self._goodbye_and_disconnect(GoodbyeReason.UNAUTHORIZED)

    def _configure_audio_output(self, audio_format: AudioFormat) -> None:
        """Store the current audio format for use in callbacks."""
        self._current_audio_format = audio_format

    def _handle_audio_chunk(self, timestamp_us: int, payload: bytes) -> None:
        """Handle incoming audio chunk and notify callbacks."""
        if self._current_audio_format is None:
            logger.debug("Dropping audio chunk without format")
            return
        # Pass server timestamp directly to callback - it handles time conversion
        # to allow for dynamic time base updates
        self._client.notify_audio_chunk(timestamp_us, payload, self._current_audio_format)

    def _handle_artwork_chunk(self, message_type: BinaryMessageType, payload: bytes) -> None:
        """Handle incoming artwork chunk and notify callbacks."""
        channel = int(message_type.value - BinaryMessageType.ARTWORK_CHANNEL_0.value)
        self._client.notify_artwork(channel, payload)

    def _handle_visualization_frame(self, message_type: BinaryMessageType, payload: bytes) -> None:
        """Parse a single-type visualization binary and notify callbacks."""
        if self._current_visualizer_config is None:
            return
        try:
            frame = self._parse_visualization_frame(
                message_type, payload, self._current_visualizer_config
            )
        except Exception:
            logger.exception("Failed to parse visualization frame")
            return
        if frame is not None:
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
        if len(payload) != 9:
            return
        try:
            (ts,) = struct.unpack_from(">q", payload, 0)
        except Exception:
            logger.exception("Failed to parse beat data")
            return
        is_downbeat = bool(payload[8] & 0b0000_0001)
        self._client.notify_visualizer_callbacks(
            [VisualizerFrame(timestamp_us=ts, is_downbeat=is_downbeat)]
        )

    def compute_play_time(self, server_timestamp_us: int) -> int:
        """Convert a server timestamp to client play time, with static delay applied."""
        if self._time_filter.is_synchronized:
            client_time = self._time_filter.compute_client_time(server_timestamp_us)
            return client_time - self._static_delay_us
        return self.now_us() + UNSYNCED_PLAY_LEAD_US - self._static_delay_us

    def compute_server_time(self, client_timestamp_us: int) -> int:
        """Convert a client timestamp to a server timestamp, with static delay removed."""
        adjusted_client_time = client_timestamp_us + self._static_delay_us
        return self._time_filter.compute_server_time(adjusted_client_time)

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
