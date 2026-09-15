"""WebSocket connection handling for a Sendspin client.

Message Sending Architecture
----------------------------
This module implements a priority-based message queue with timestamp ordering for sync.

**Queue Structure:**
- Priority messages: ServerHello, time sync - sent immediately (FIFO deque)
- Normal messages: Non-role JSON control messages - sent in FIFO order (deque)
- Role queues: Per-role min-heaps holding both binary and JSON messages, sorted by
  (timestamp, sequence). Binary messages use their playback timestamp; JSON messages
  inherit the timestamp of the previous message in that role's queue.

**Message Ordering:**
Messages are grouped by role (e.g., player, artwork). Within each role, binary and
JSON messages share the same min-heap, ensuring strict ordering. Binary messages sort
by playback timestamp for correct sequencing even when chunks are encoded out-of-order.
JSON messages inherit the previous message's timestamp so they stay in position relative
to surrounding binary data.

**Epoch-Based Invalidation:**
Each role has an epoch counter. When a stream is cleared or ends, the epoch increments,
causing binary entries with the old epoch to be silently discarded. JSON entries in the
same queue are NOT affected - they skip epoch validation and are always delivered.

**Backpressure:**
Roles can be "blocked" until a future time (e.g., waiting for client buffer space).
Blocked roles are tracked in a separate heap and promoted back when ready.
"""

from __future__ import annotations

import asyncio
import heapq
import logging
import time
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import orjson
from aiohttp import ClientWebSocketResponse, WSMessage, WSMsgType, web

from aiosendspin.models import BINARY_HEADER_SIZE, unpack_binary_header
from aiosendspin.models.core import (
    ActivatePairing,
    ClientCommandMessage,
    ClientGoodbyeMessage,
    ClientHelloMessage,
    ClientHelloPayload,
    ClientStateMessage,
    ClientStatePayload,
    ClientTimeMessage,
    LegacyServerHelloMessage,
    LegacyServerHelloPayload,
    ServerActivateMessage,
    ServerActivatePayload,
    ServerHelloMessage,
    ServerHelloPayload,
    ServerTimeMessage,
    ServerTimePayload,
    StreamClearMessage,
    StreamEndMessage,
    StreamRequestFormatMessage,
    StreamStartMessage,
)
from aiosendspin.models.management import (
    ManagementAddRecordMessage,
    ManagementAddRecordPayload,
    ManagementGetPairingConfigMessage,
    ManagementListRecordsMessage,
    ManagementOpenPairingWindowMessage,
    ManagementRemoveRecordMessage,
    ManagementRemoveRecordPayload,
    ManagementResultData,
    ManagementResultMessage,
    ManagementResultPayload,
    ManagementSetPairingConfigMessage,
    ManagementSetPairingConfigPayload,
    RecordSummary,
    ServerUnpairMessage,
    StorageAccounting,
)
from aiosendspin.models.source import (
    ClientStreamEndMessage,
    ClientStreamStartMessage,
)
from aiosendspin.models.types import (
    CLOSING_ABORT_REASONS,
    Activity,
    ClientMessage,
    ConnectionReason,
    GoodbyeReason,
    ManagementResult,
    PairAbortReason,
    PairingCodeFormat,
    PairMethod,
    PictureFormat,
    PlaybackStateType,
    Roles,
    ServerMessage,
    role_family,
)
from aiosendspin.noise.constants import SENTINEL_PSK
from aiosendspin.noise.driver import (
    HandshakeAbortedError,
    receive_text_frame,
    run_handshake_server,
    run_rehandshake_server,
)
from aiosendspin.noise.keys import b64url_encode, psk_id_for
from aiosendspin.noise.pairing import (
    LocalPairingAbortError,
    PairingAbortError,
    PairingAttempt,
    PairingError,
    PairingTimeoutError,
    abort_pairing,
    run_dynamic_pairing_code_server,
    run_pairing_psk_server,
    run_static_pairing_code_server,
)
from aiosendspin.noise.session import NoiseSession
from aiosendspin.noise.trust_store import PskCategory, ResolvedPsk, ServerPairingRecord
from aiosendspin.noise.wire import EncryptedWebSocket
from aiosendspin.util import create_task

from .client import SendspinClient
from .compliance import ClientComplianceError
from .events import ClientEvent, ClientGroupChangedEvent, GroupEvent, GroupStateChangedEvent
from .roles.negotiation import negotiate_roles
from .roles.registry import ROLE_FACTORIES, ROLE_SUPPORT_SPECS, role_requires_pairing

if TYPE_CHECKING:
    from aiosendspin.models.artwork import ClientHelloArtworkSupport

    from .audio import BufferTracker
    from .group import SendspinGroup
    from .roles.base import BinaryHandling, Role
    from .server import SendspinServer

# Transport used by the writer and message loop: the encrypted wrapper post-handshake,
# or the raw aiohttp socket for a legacy (transition-mode) connection.
Transport = EncryptedWebSocket | web.WebSocketResponse | ClientWebSocketResponse


logger = logging.getLogger(__name__)

MAX_PENDING_MSG = 4096  # Default queue cap (per role queues, and global control queues)

# Quiet period between repeats of a throttled warning. Each warning reports how
# many occurrences it stands for, so a sustained fault keeps its magnitude visible
# at default level without emitting one line per event.
_WARN_INTERVAL_S = 30.0

# Bound the wait for the writer to drain when quiescing.
QUIESCE_TIMEOUT_S: float = 30.0

_PAIRING_MESSAGE_TYPES: frozenset[str] = frozenset(
    {
        "client/pair-pending",
        "client/pair-init",
        "client/pair-finalize",
        "client/pair-auth",
        "client/pair-confirm",
        "pair/abort",
    }
)

_PAIR_TRANSITION_TYPES: frozenset[str] = frozenset(
    {
        "noise/handshake",
        "client/hello",
        "client/pair-pending",
        "client/pair-init",
        "client/pair-finalize",
        "client/pair-auth",
        "client/pair-confirm",
        "pair/abort",
    }
)


@dataclass(frozen=True, slots=True)
class _BinaryData:
    """Binary payload metadata for buffer tracking."""

    data: bytes
    message_type: int
    buffer_end_time_us: int | None = None
    buffer_byte_count: int | None = None
    duration_us: int | None = None


@dataclass(frozen=True, slots=True)
class _RoleQueueEntry:
    """Unified queue entry for binary or JSON messages within a role.

    Both binary and JSON messages for a role go through the same min-heap,
    sorted by (timestamp, sequence). JSON messages inherit the timestamp of the
    previous message in the role queue, ensuring they maintain their position
    relative to surrounding timed binary. If no previous message exists, timestamp is 0.
    """

    epoch: int
    timestamp_us: int
    # Exactly one of these is set
    binary: _BinaryData | None = None
    json_message: ServerMessage | None = None
    # Server clock at enqueue, for late-drop diagnostics (0 for JSON entries)
    enqueued_at_us: int = 0


class _QueuedTransport(EncryptedWebSocket):
    """``EncryptedWebSocket`` whose ``receive()`` pulls from a queue."""

    def __init__(self, base: EncryptedWebSocket, queue: asyncio.Queue[WSMessage]) -> None:
        super().__init__(base._ws, base._session)  # noqa: SLF001
        self._base = base
        self._queue = queue

    def swap_session(self, session: NoiseSession) -> None:
        super().swap_session(session)
        self._base.swap_session(session)

    async def receive(self) -> WSMessage:
        return await self._queue.get()


class SendspinConnection:
    """A single WebSocket connection to a Sendspin client device."""

    def __init__(  # noqa: PLR0915
        self,
        server: SendspinServer,
        *,
        request: web.Request | None = None,
        wsock_client: ClientWebSocketResponse | None = None,
        url: str | None = None,
        expected_client_id: str | None = None,
        pairing_attempt: PairingAttempt | None = None,
    ) -> None:
        """Initialize a SendspinConnection.

        Exactly one of `request` (client-initiated) or `wsock_client` (server-initiated)
        must be provided. For server-initiated connections, `url` should be provided
        for connection reason lookup and client URL registration, and
        ``expected_client_id`` may be set to bind the handshake to a known peer.
        ``pairing_attempt`` carries an operator-initiated pairing intent for this dial.
        """
        self._server = server
        self._wsock_client = wsock_client
        self._wsock_server: web.WebSocketResponse | None = None
        self._request = request
        self._url = url  # For server-initiated connections
        self._expected_client_id = expected_client_id
        self._in_pairing = False
        self._pairing_attempt = pairing_attempt
        self._pairing_task: asyncio.Task[bool] | None = None
        self._pairing_message_queue: asyncio.Queue[WSMessage] | None = None
        self._pairing_messages_started = False
        self._pairing_index = 0
        self._connection_done = asyncio.Event()
        self._transport: Transport | None = None
        self._pending_first_text: str | None = None  # legacy first frame held for the loop

        if request is not None:
            if wsock_client is not None:
                raise ValueError("Only one of request or wsock_client may be provided")
            self._wsock_server = web.WebSocketResponse(heartbeat=30, compress=False)
            self._logger = logger.getChild(f"unknown-{request.remote}")
        elif wsock_client is not None:
            self._logger = logger.getChild("unknown-client")
        else:
            raise ValueError("Either request or wsock_client must be provided")

        self._queue_sequence: int = 0  # FIFO tie-breaker across all queues
        self._queue_size: int = 0
        # Outgoing message queues
        self._priority_messages: deque[ServerMessage] = deque()
        self._normal_messages: deque[ServerMessage] = deque()
        # Role queues: per role min-heap of (sort_ts, seq, entry)
        # Both binary and JSON messages for a role go through the same heap.
        self._role_queues: dict[str, list[tuple[int, int, _RoleQueueEntry]]] = defaultdict(list)
        self._max_pending_msg_by_role: defaultdict[str, int] = defaultdict(lambda: MAX_PENDING_MSG)
        # Last timestamp per role for JSON inheritance (JSON gets previous message's timestamp)
        self._last_enqueued_ts_by_role: dict[str, int] = {}
        # Rate-limit state for already-late-at-enqueue and slow-send warnings
        self._late_at_enqueue_count: dict[str, int] = {}
        self._last_late_at_enqueue_log_s: dict[str, float] = {}
        self._last_slow_send_log_s = 0.0
        self._slow_send_count = 0
        # Global scheduler heaps for families
        self._ready_roles: list[tuple[int, int, str]] = []
        self._delayed_roles: list[tuple[int, int, str]] = []
        self._blocked_until_us: dict[str, int] = {}
        self._block_generation: defaultdict[str, int] = defaultdict(int)
        self._writer_wakeup = asyncio.Event()
        self._writer_idle = asyncio.Event()
        self._writer_task: asyncio.Task[None] | None = None
        self._message_loop_task: asyncio.Task[None] | None = None

        self._noise_psk: ResolvedPsk | None = None
        self._handshake_hash: bytes | None = None

        self._client_id: str | None = None
        self._client_info: ClientHelloPayload | None = None
        self._negotiated_roles: list[str] = []
        self._client: SendspinClient | None = None
        self._trusted_unpaired = False

        self._declared_activities: list[Activity] | None = None
        self._client_event_unsub: Callable[[], None] | None = None
        self._group_event_unsub: Callable[[], None] | None = None

        self._management_active = (
            url is not None and server.get_connection_reason(url) is ConnectionReason.MANAGEMENT
        )
        self._management_waiter: asyncio.Future[ManagementResultPayload] | None = None

        self._closing = False
        self._disconnecting = False

        self._initial_state_received = False
        self._initial_state_timeout_handle: asyncio.TimerHandle | None = None
        # Binary held while a role that receives binary awaits the initial client/state.
        # The spec forbids sending binary before it; flushed once the state arrives.
        # Each entry carries the role's epoch at buffer time so a stream boundary
        # in the meantime (which bumps the epoch) discards it instead of replaying.
        self._pending_binary: list[tuple[str, int, Callable[[], None]]] = []

        self._last_goodbye_reason: GoodbyeReason | None = None
        self._epoch_by_role: defaultdict[str, int] = defaultdict(int)

        # Timing tracking for binary frame logging (per role)
        self._last_send_time_us_by_role: dict[str, int] = {}
        self._last_timestamp_us_by_role: dict[str, int] = {}
        self._send_stats_by_role: dict[str, dict[str, float | int]] = {}
        self._send_summary_last_log_s = time.monotonic()

    @property
    def websocket_connection(self) -> web.WebSocketResponse | ClientWebSocketResponse:
        """Return the underlying aiohttp WebSocket connection object."""
        wsock = self._wsock_server or self._wsock_client
        assert wsock is not None
        return wsock

    @property
    def is_server_initiated(self) -> bool:
        """Return True if this connection was initiated by the server."""
        return self._wsock_client is not None

    @property
    def should_retry_server_initiated_connection(self) -> bool:
        """Whether the server should reconnect this URL after disconnect.

        Per client/goodbye reason: only ``restart`` (will reconnect) and
        ``concurrent_attempt`` (may retry later) warrant it. With no goodbye, assume a
        ``restart`` when the connection was idle or carried playback, else treat the drop
        as a session end.
        """
        if self._closing:
            return False
        reason = self._last_goodbye_reason
        if reason is None:
            activities = self._declared_activities or []
            return not activities or Activity.PLAYBACK in activities
        return reason in (GoodbyeReason.RESTART, GoodbyeReason.CONCURRENT_ATTEMPT)

    @property
    def goodbye_reason(self) -> GoodbyeReason | None:
        """Disconnect reason reported by client/goodbye, if available."""
        return self._last_goodbye_reason

    @property
    def psk_category(self) -> PskCategory | None:
        """Category of the PSK that cryptographically admitted this connection."""
        return None if self._noise_psk is None else self._noise_psk.category

    @property
    def is_encrypted(self) -> bool:
        """Whether this connection was admitted through the Noise handshake."""
        return self._noise_psk is not None

    def requires_initial_state(self) -> bool:
        """Whether this connection must receive initial client/state before being 'connected'."""
        if self._client is None:
            return False
        return any(role.requires_initial_state() for role in self._client.active_roles)

    def _flush_pending_binary(self) -> None:
        """Enqueue binary held until the initial client/state arrived, dropping stale entries."""
        pending, self._pending_binary = self._pending_binary, []
        for role, epoch, send in pending:
            # A stream boundary during the wait bumped the epoch; that data is stale.
            if epoch == self._epoch_by_role[role]:
                send()

    def _flag_initial_state_deviations(self, payload: ClientStatePayload) -> None:
        """Flag spec requirements the initial client/state must satisfy but does not."""
        reasons: list[str] = []
        if payload.available is None:
            reasons.append("omitted the required 'available' field")
        if self._client is not None:
            for role in self._client.active_roles:
                reasons.extend(role.initial_state_deviations(payload))
        for reason in reasons:
            self._flag_noncompliance(f"initial client/state {reason}")

    def _flag_inactive_role_payloads(self, kind: str, present: dict[str, object]) -> None:
        """Flag role objects sent for a family the client has no active role in."""
        if self._client is None:
            return
        active = {role.role_family for role in self._client.active_roles}
        for family, obj in present.items():
            if obj is not None and family not in active:
                self._flag_noncompliance(f"{kind} carried a {family} object for an inactive role")

    def drop_pending_binary(self, roles: list[str] | None) -> None:
        """Drop queued binary payloads for the specified roles.

        Uses epoch-based lazy invalidation: increments the epoch counter for each role,
        causing the writer loop to discard binary entries with the old epoch.
        JSON entries in the same queue are NOT affected (they skip epoch validation).
        """
        roles_to_drop = list(self._epoch_by_role.keys()) if roles is None else roles
        for role in roles_to_drop:
            self._epoch_by_role[role] += 1
            if role in self._blocked_until_us:
                # The backpressure deadline was computed against audio that is
                # now invalid; new-epoch work must not wait behind it.
                self._blocked_until_us.pop(role, None)
                self._block_generation[role] += 1
                self._schedule_role_head(role)
        self._wake_writer()

    def send_binary(
        self,
        data: bytes,
        *,
        role: str,
        timestamp_us: int,
        message_type: int,
        buffer_end_time_us: int | None = None,
        buffer_byte_count: int | None = None,
        duration_us: int | None = None,
    ) -> None:
        """Enqueue a binary message.

        Args:
            data: Binary data to send.
            role: Role for epoch tracking and queue routing.
            timestamp_us: Playback timestamp from binary header (cached to avoid unpacking).
            message_type: Binary message type for role lookup (cached).
            buffer_end_time_us: End timestamp for buffer tracking.
            buffer_byte_count: Byte count for buffer tracking.
            duration_us: Duration for buffer tracking.
        """
        if self.requires_initial_state() and not self._initial_state_received:
            # No binary before the client's initial state; replay once it arrives,
            # tagged with the current epoch so a stream boundary can invalidate it.
            self._pending_binary.append(
                (
                    role,
                    self._epoch_by_role[role],
                    partial(
                        self.send_binary,
                        data,
                        role=role,
                        timestamp_us=timestamp_us,
                        message_type=message_type,
                        buffer_end_time_us=buffer_end_time_us,
                        buffer_byte_count=buffer_byte_count,
                        duration_us=duration_us,
                    ),
                )
            )
            return

        if self._is_role_queue_full(role):
            self._disconnect_due_to_queue_overflow(
                f"Role queue full for {role} ({len(self._role_queues.get(role, []))}/"
                f"{self._max_pending_msg_by_role[role]}), client too slow"
            )
            return

        now_us = self._server.clock.now_us()
        # buffer_end_time_us already carries the role's output delay, which shifts the
        # deadline earlier than the raw timestamp. Fall back to the raw span only when
        # a caller does not supply it.
        deadline_us = (
            buffer_end_time_us
            if buffer_end_time_us is not None
            else timestamp_us + (duration_us or 0)
        )
        if timestamp_us != 0 and deadline_us <= now_us:
            self._warn_late_at_enqueue(role, message_type, timestamp_us, now_us)

        # Keep per-role queue ordering monotonic so role-scoped lifecycle JSON
        # (stream/start, stream/end, stream/clear) cannot be overtaken by binary
        # packets that carry an older playback timestamp (e.g. historical backfill).
        sort_ts = max(0, timestamp_us, self._last_enqueued_ts_by_role.get(role, 0))
        entry = _RoleQueueEntry(
            epoch=self._epoch_by_role[role],
            timestamp_us=timestamp_us,
            binary=_BinaryData(
                data=data,
                message_type=message_type,
                buffer_end_time_us=buffer_end_time_us,
                buffer_byte_count=buffer_byte_count,
                duration_us=duration_us,
            ),
            enqueued_at_us=now_us,
        )
        self._last_enqueued_ts_by_role[role] = sort_ts
        self._enqueue_role_entry(role, sort_ts, entry)

    def _warn_late_at_enqueue(
        self, role: str, message_type: int, timestamp_us: int, now_us: int
    ) -> None:
        """Warn when a binary message is already past its play deadline at enqueue.

        A late drop at send time only says a deadline was missed; this names the
        moment the unplayable data entered the queue, which points at the producer
        (stale timestamps, timeline reset) rather than the transport.
        """
        cached = None
        if self._client is not None:
            cached = self._client.get_binary_handling_cached(message_type)
        if cached is None or not cached[0].drop_late:
            return
        behind_by_us = now_us - (timestamp_us - cached[1].get_output_delay_us())
        self._late_at_enqueue_count[role] = self._late_at_enqueue_count.get(role, 0) + 1
        now_s = time.monotonic()
        if now_s - self._last_late_at_enqueue_log_s.get(role, 0.0) < _WARN_INTERVAL_S:
            return
        self._logger.warning(
            "Enqueued already-late binary type=%s role=%s: %s message(s); "
            "behind_by_us=%s ts_us=%s now_us=%s queue=%s",
            message_type,
            role,
            self._late_at_enqueue_count[role],
            behind_by_us,
            timestamp_us,
            now_us,
            len(self._role_queues.get(role, [])),
        )
        self._late_at_enqueue_count[role] = 0
        self._last_late_at_enqueue_log_s[role] = now_s

    def queue_status(self) -> tuple[int, int]:
        """Return (qsize, maxsize) for the outgoing queue."""
        maxsize = MAX_PENDING_MSG + (len(self._role_queues) * MAX_PENDING_MSG)
        return self._queue_size, maxsize

    def _disconnect_due_to_queue_overflow(self, message: str) -> None:
        if self._disconnecting:
            return
        self._logger.error("%s - disconnecting", message)
        create_task(self.disconnect(retry_connection=True))

    def _is_role_queue_full(self, role: str) -> bool:
        return len(self._role_queues.get(role, [])) >= self._max_pending_msg_by_role[role]

    def _enqueue_role_entry(self, role: str, sort_ts: int, entry: _RoleQueueEntry) -> None:
        """Push an entry into a role's heap and schedule it if it becomes the new head."""
        seq = self._queue_sequence
        self._queue_sequence += 1
        role_queue = self._role_queues[role]
        heapq.heappush(role_queue, (sort_ts, seq, entry))
        self._queue_size += 1

        if role not in self._blocked_until_us:
            head_sort_ts, head_seq, _ = role_queue[0]
            if head_sort_ts == sort_ts and head_seq == seq:
                heapq.heappush(self._ready_roles, (head_sort_ts, head_seq, role))

        self._wake_writer()

    def _wake_writer(self) -> None:
        """Signal the writer that new work is queued."""
        self._writer_idle.clear()
        self._writer_wakeup.set()

    def send_role_message(self, role: str, message: ServerMessage) -> None:
        """Enqueue a JSON message into a role's queue with inherited timestamp.

        The message inherits the timestamp of the last message enqueued for this role,
        so it maintains its position relative to surrounding timed binary. If no previous
        message exists, it uses timestamp 0 (sent before any timed binary).

        Exception: StreamEnd and StreamStart use current time instead of inheriting,
        ensuring they are ordered correctly across stream boundaries.
        """
        if isinstance(message, StreamClearMessage | StreamEndMessage):
            self.drop_pending_binary(message.payload.roles)

        if self._is_role_queue_full(role):
            self._disconnect_due_to_queue_overflow(
                f"Role queue full for {role} ({len(self._role_queues.get(role, []))}/"
                f"{self._max_pending_msg_by_role[role]}), client too slow"
            )
            return

        # Stream lifecycle messages use current time to ensure correct ordering
        # across stream boundaries (prevents old stream timestamps from affecting new stream)
        if isinstance(message, StreamEndMessage | StreamStartMessage):
            sort_ts = self._server.clock.now_us()
            # Update tracker so subsequent messages inherit this timestamp
            self._last_enqueued_ts_by_role[role] = sort_ts
        else:
            sort_ts = self._last_enqueued_ts_by_role.get(role, 0)

        entry = _RoleQueueEntry(
            epoch=self._epoch_by_role[role],
            timestamp_us=sort_ts,
            json_message=message,
        )
        self._enqueue_role_entry(role, sort_ts, entry)

        if not isinstance(message, ServerTimeMessage):
            self._logger.debug("Enqueueing role message: %s", type(message).__name__)

    def send_message(self, message: ServerMessage) -> None:
        """Enqueue a non-role JSON message (sent in FIFO order, not tied to any role)."""
        if isinstance(message, StreamClearMessage | StreamEndMessage):
            self.drop_pending_binary(message.payload.roles)

        if self._queue_size >= MAX_PENDING_MSG:
            self._disconnect_due_to_queue_overflow("Control message queue full, client too slow")
            return

        self._normal_messages.append(message)
        self._queue_size += 1
        self._wake_writer()

        if not isinstance(message, ServerTimeMessage):
            self._logger.debug("Enqueueing message: %s", type(message).__name__)

    def _merge_state_messages(
        self,
        existing: ServerMessage,
        incoming: ServerMessage,
    ) -> ServerMessage | None:
        """Merge consecutive state-like messages where safe."""
        return existing.merge(incoming)

    def send_priority_message(self, message: ServerMessage) -> None:
        """Enqueue a high-priority message (processed before regular queue)."""
        if len(self._priority_messages) >= MAX_PENDING_MSG:
            self._disconnect_due_to_queue_overflow("Priority message queue full, client too slow")
            return
        self._queue_sequence += 1
        self._priority_messages.append(message)
        self._queue_size += 1
        self._wake_writer()

    async def disconnect(self, *, retry_connection: bool = True) -> None:
        """Disconnect this connection and detach from its persistent client."""
        if not retry_connection:
            self._closing = True
        if self._disconnecting:
            return
        self._disconnecting = True

        if self._management_waiter is not None and not self._management_waiter.done():
            self._management_waiter.set_exception(RuntimeError("connection closed"))

        self._unsubscribe_activity_events()

        if self._initial_state_timeout_handle is not None:
            self._initial_state_timeout_handle.cancel()
            self._initial_state_timeout_handle = None

        if self._pairing_task and not self._pairing_task.done():
            # Ends like end_pairing: the attempt aborts instead of waiting out its timeout.
            self._pairing_task.cancel()
            with suppress(PairingError, OSError, asyncio.CancelledError):
                await self._pairing_task
        if self._writer_task and not self._writer_task.done():
            self._writer_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._writer_task
        if self._message_loop_task and not self._message_loop_task.done():
            self._message_loop_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._message_loop_task

        wsock = self._wsock_client or self._wsock_server
        if wsock is not None and not wsock.closed:
            with suppress(Exception):
                await wsock.close()

        if self._client is not None:
            # Only detach if this connection is still the active one.
            if self._client.connection is self:
                self._client.detach_connection(self._last_goodbye_reason)
            self._client = None

        self._logger.debug("Connection disconnected")

    def _initial_state_timeout_callback(self) -> None:
        if self._initial_state_received:
            return
        self._initial_state_timeout_handle = None
        try:
            self._flag_noncompliance("did not send the required initial client/state in time")
        except ClientComplianceError:
            # A timer callback can't propagate into the message loop, so tear down here.
            create_task(self.disconnect(retry_connection=False))
            return
        # Lenient: keep the connection and mark the client connected anyway.
        if self._client is not None:
            self._initial_state_received = True
            self._client.mark_connected()
            self._server.on_client_first_connect(self._client.client_id)
            self._flush_pending_binary()

    @staticmethod
    def _first_registered_role_id_in_family(
        supported_roles: list[str], *, family: str
    ) -> str | None:
        """Return first client-preferred, server-registered role id in a role family."""
        for role_id in supported_roles:
            if role_family(role_id) == family and role_id in ROLE_FACTORIES:
                return role_id
        return None

    @staticmethod
    def _unimplemented_roles(supported_roles: list[str]) -> list[str]:
        """Client-offered roles/versions this server does not implement.

        Excludes `_`-prefixed custom roles and versions. A non-empty result means the
        client likely speaks a newer spec revision than this server.
        """
        return [
            r
            for r in supported_roles
            if r not in ROLE_FACTORIES
            and not r.startswith("_")
            and not r.partition("@")[2].startswith("_")
        ]

    @classmethod
    def _primary_role_id_for_family(cls, family: str) -> str | None:
        """Return built-in primary role id for a role family, if defined."""
        for role in Roles:
            if role_family(role.value) == family:
                return role.value
        return None

    @classmethod
    def _extract_custom_role_supports(cls, message: dict[str, Any]) -> dict[str, tuple[str, Any]]:
        """Extract custom support objects from raw client/hello JSON without mutating payload."""
        payload = message.get("payload")
        if not isinstance(payload, dict):
            return {}
        supported_roles = payload.get("supported_roles")
        if not (
            isinstance(supported_roles, list) and all(isinstance(v, str) for v in supported_roles)
        ):
            return {}

        custom_supports: dict[str, tuple[str, Any]] = {}
        for family in ROLE_SUPPORT_SPECS:
            selected_role = cls._first_registered_role_id_in_family(supported_roles, family=family)
            if selected_role is None:
                # Restrict the fallback to spec-custom versions (those starting
                # with `_`). Unknown spec-versioned IDs like `v2` may carry a
                # schema-incompatible support payload, so parsing against the
                # family's registered schema would crash on field drift.
                selected_role = next(
                    (
                        r
                        for r in supported_roles
                        if role_family(r) == family and r.partition("@")[2].startswith("_")
                    ),
                    None,
                )
            if selected_role is None:
                # Skip support parsing for unknown roles.
                continue
            primary_role_id = cls._primary_role_id_for_family(family)
            if selected_role == primary_role_id:
                continue

            custom_support_key = f"{selected_role}_support"
            # If the role's support key has a mashumaro-aliased field on
            # ClientHelloPayload (e.g. legacy `visualizer@_draft_r1` running
            # alongside the primary `visualizer@v1`), let mashumaro parse it
            # via the alias and skip the custom-role path so the schema is
            # picked correctly for the role version.
            if custom_support_key in ClientHelloPayload._SUPPORT_KEY_ALIASES.values():  # noqa: SLF001
                continue
            custom_support = payload.get(custom_support_key)
            primary_support_key = (
                f"{primary_role_id}_support" if primary_role_id is not None else None
            )
            legacy_support_key = f"{family}_support"
            # Fall back to the legacy (unversioned) <family>_support key when
            # the client didn't include a per-version key. Versioned keys for
            # OTHER versions (e.g. primary_support_key) stay rejected because
            # their schema may differ from the selected role's.
            if custom_support is None and payload.get(legacy_support_key) is not None:
                custom_support = payload.get(legacy_support_key)
            elif custom_support is None and (
                primary_support_key is not None and payload.get(primary_support_key) is not None
            ):
                logger.warning(
                    "Ignoring %s for custom role %s; expected %s",
                    primary_support_key,
                    selected_role,
                    custom_support_key,
                )
            custom_supports[family] = (selected_role, custom_support)
        return custom_supports

    @classmethod
    def _apply_custom_role_support(
        cls, hello: ClientHelloPayload, custom_supports: dict[str, tuple[str, Any]]
    ) -> None:
        """Apply parsed custom role support objects onto ClientHelloPayload fields."""
        for family, spec in ROLE_SUPPORT_SPECS.items():
            custom = custom_supports.get(family)
            if custom is None:
                continue
            custom_role, raw_support = custom
            if raw_support is None:
                raise ValueError(
                    f"{custom_role}_support must be provided when "
                    f"'{custom_role}' is in supported_roles"
                )
            if not isinstance(raw_support, dict):
                raise TypeError(
                    f"{custom_role}_support must be an object for role family '{family}'"
                )
            setattr(hello, f"{family}_support", spec.parse_support(raw_support))

    @classmethod
    def _deserialize_client_message(cls, raw_message: str) -> ClientMessage:
        """Deserialize inbound client message with custom support-key normalization."""
        parsed = ClientMessage.from_json(raw_message)
        if isinstance(parsed, ClientHelloMessage):
            decoded = orjson.loads(raw_message)
            if not isinstance(decoded, dict):
                return parsed
            custom_supports = cls._extract_custom_role_supports(decoded)
            if isinstance(parsed, ClientHelloMessage):
                cls._apply_custom_role_support(parsed.payload, custom_supports)
            return parsed
        return parsed

    async def _setup_connection(self) -> None:
        """Prepare the socket and run the Noise handshake."""
        if self._wsock_server is not None:
            assert self._request is not None
            async with asyncio.timeout(10):
                await self._wsock_server.prepare(self._request)

        raw = self._wsock_server or self._wsock_client
        assert raw is not None
        if self._transport is None:
            self._transport = await self._establish_transport(raw)

        self._logger.debug("Connection established")

    def _is_pairing(self) -> bool:
        if self._pairing_attempt is not None:
            return True
        if self._noise_psk is None:
            return False
        # Pre-staged Pairing PSK (client-initiated admit via the store).
        return self._noise_psk.category is PskCategory.PAIRING

    @property
    def _pairing_in_progress(self) -> bool:
        """Whether operator-initiated pairing is currently being executed."""
        return self._pairing_message_queue is not None

    async def _establish_transport(
        self, raw: web.WebSocketResponse | ClientWebSocketResponse
    ) -> Transport:
        """Dispatch on the first frame: run the Noise handshake or accept a legacy client.

        A ``client/init`` first frame runs the Noise initiator handshake and yields
        an encrypted transport. In transition mode, a ``client/hello`` first frame
        is accepted unencrypted (the raw socket is the transport, and the frame is
        held for the message loop). Anything else raises
        ``HandshakeAbortedError``.
        """
        first_text = await receive_text_frame(raw, what="first frame")
        msg_type = self._peek_message_type(first_text)
        if msg_type == "client/init":
            result = await run_handshake_server(
                raw,
                local_identity=self._server.identity,
                psk_provider=self._psk_provider,
                client_init_text=first_text,
                expected_client_id=self._expected_client_id,
            )
            self._client_id = result.peer_id
            self._noise_psk = result.psk
            self._handshake_hash = result.handshake_hash
            self._pairing_index = 0
            self._logger = logger.getChild(result.peer_id)
            return result.encrypted_ws
        if msg_type == "client/hello" and self._server.allow_unencrypted:
            if self._pairing_attempt is not None:
                raise HandshakeAbortedError("pairing requires an encrypted connection")
            self._logger.warning("Accepting unencrypted legacy connection (transition mode)")
            self._pending_first_text = first_text
            return raw
        raise HandshakeAbortedError(f"unexpected first frame type {msg_type!r}")

    @staticmethod
    def _peek_message_type(text: str) -> str | None:
        try:
            decoded = orjson.loads(text)
        except orjson.JSONDecodeError:
            return None
        return decoded.get("type") if isinstance(decoded, dict) else None

    async def _psk_provider(self, client_id: str) -> ResolvedPsk | None:
        """Pick the PSK to admit ``client_id`` with."""
        if self._pairing_attempt is not None:
            attempt = self._pairing_attempt
            if attempt.method is PairMethod.PAIRING_PSK:
                assert attempt.pairing_psk is not None
                return ResolvedPsk(
                    psk_id_for(attempt.pairing_psk),
                    attempt.pairing_psk,
                    PskCategory.PAIRING,
                )
            return ResolvedPsk(psk_id_for(SENTINEL_PSK), SENTINEL_PSK, PskCategory.SENTINEL)

        store = self._server.pairing_store
        record = await store.record_by_client_id(client_id)
        if record is not None:
            return record.as_resolved()
        staged = await store.staged_pairing_psk(client_id)
        if staged is not None:
            return staged.as_resolved()
        return ResolvedPsk(psk_id_for(SENTINEL_PSK), SENTINEL_PSK, PskCategory.SENTINEL)

    async def _exchange_hellos(self) -> bool:
        """Exchange hellos and send the initial server/activate; False if hello rejected."""
        transport = self._transport
        assert transport is not None

        if not self.is_encrypted:
            # Non-spec transition path: the legacy hello replaces server/hello plus activate.
            assert self._pending_first_text is not None
            client_hello_text = self._pending_first_text
            self._pending_first_text = None
            if not await self._ingest_client_hello(client_hello_text):
                return False
            connection_reason = (
                self._server.get_connection_reason(self._url)
                if self._url is not None
                else ConnectionReason.DISCOVERY
            )
            if connection_reason not in (ConnectionReason.DISCOVERY, ConnectionReason.PLAYBACK):
                # Legacy clients parse the enum strictly and predate the other reasons.
                self._logger.debug(
                    "Clamping connection_reason %s to discovery for a legacy client",
                    connection_reason.value,
                )
                connection_reason = ConnectionReason.DISCOVERY
            await transport.send_str(
                LegacyServerHelloMessage(
                    payload=LegacyServerHelloPayload(
                        server_id=self._server.id,
                        name=self._server.name,
                        version=1,
                        active_roles=self._negotiated_roles,
                        connection_reason=connection_reason,
                    )
                ).to_json()
            )
        else:
            if not await self._send_server_hello_and_recv(transport):
                return False
            if self._is_pairing():
                assert isinstance(transport, EncryptedWebSocket)
                try:
                    if not await self._pair(transport):
                        return False
                except PairingTimeoutError as exc:
                    # Timeout waiting for client; the connection stays open for a retry.
                    self._logger.debug("Initial-connect pairing timed out: %s", exc)
                    self._pairing_attempt = None
                except PairingAbortError as exc:
                    if exc.reason in CLOSING_ABORT_REASONS:
                        raise
                    # Non-closing abort reason; the connection stays open for a retry.
                    self._logger.debug("Initial-connect pairing aborted: %s", exc)
                    self._pairing_attempt = None
            await self._activate()

        if self.requires_initial_state():
            self._initial_state_timeout_handle = self._server.loop.call_later(
                5.0, self._initial_state_timeout_callback
            )
        else:
            assert self._client is not None
            self._client.mark_connected()
            self._server.on_client_first_connect(self._client.client_id)
        return True

    async def _send_server_hello_and_recv(self, transport: Transport) -> bool:
        """Send ``server/hello`` and receive+ingest ``client/hello``."""
        await transport.send_str(
            ServerHelloMessage(payload=ServerHelloPayload(name=self._server.name)).to_json()
        )
        client_hello_text = await receive_text_frame(transport, what="client/hello")
        return await self._ingest_client_hello(client_hello_text)

    def _flag_noncompliance(self, reason: str) -> None:
        """Log a tolerated spec violation, or reject it when the server is strict.

        Usable during the hello exchange before a persistent client exists; once
        attached, delegates to the client so its logger carries the client_id.
        """
        if self._client is not None:
            self._client.flag_noncompliance(reason)
            return
        if not self._server.allow_noncompliant_clients:
            self._logger.error("rejecting non-compliant client: %s", reason)
            raise ClientComplianceError(reason)
        self._logger.warning("non-compliant client: %s", reason)

    async def _ingest_client_hello(self, text: str) -> bool:
        """Validate and record the client/hello, attaching the client; False if rejected."""
        try:
            return await self._ingest_client_hello_checked(text)
        except ClientComplianceError:
            await self.disconnect(retry_connection=False)
            return False

    async def _ingest_client_hello_checked(self, text: str) -> bool:
        """Body of the hello exchange; raises ClientComplianceError in strict mode."""
        try:
            message = self._deserialize_client_message(text)
        except (LookupError, TypeError, ValueError) as exc:
            self._logger.error("Malformed client/hello: %s", exc)
            await self.disconnect(retry_connection=False)
            return False
        if not isinstance(message, ClientHelloMessage):
            self._logger.error("Expected client/hello, got %s", type(message).__name__)
            await self.disconnect(retry_connection=False)
            return False

        client_info = message.payload
        # Encrypted clients omit version (it is in client/init); only a legacy
        # client carries it in the hello, so validate it only when present.
        if client_info.version is not None and client_info.version != 1:
            self._logger.error(
                "Incompatible protocol version %s (only '1' is supported)",
                client_info.version,
            )
            await self.disconnect(retry_connection=False)
            return False
        # Encrypted clients carry version in client/init, so only an unencrypted
        # hello is required to include it.
        if not self.is_encrypted and client_info.version is None:
            self._flag_noncompliance("unencrypted client/hello omitted required version")
        # The Noise handshake sets client_id (authenticated); a legacy client
        # instead carries it in the hello payload.
        client_id = self._client_id or client_info.client_id
        if client_id is None:
            self._logger.error("client/hello has no client_id and no handshake identity")
            await self.disconnect(retry_connection=False)
            return False

        if not self.is_encrypted and not await self._admit_legacy_client_id(client_id):
            await self.disconnect(retry_connection=False)
            return False

        self._client_info = client_info
        self._client_id = client_id
        self._negotiated_roles = negotiate_roles(
            client_info.supported_roles, strict=not self._server.allow_noncompliant_clients
        )
        self._logger = logger.getChild(client_id)
        self._logger.debug("Received client/hello: %s", client_info)
        if client_info.legacy_support_keys_used:
            self._flag_noncompliance(
                "client/hello used unversioned support keys: "
                + ", ".join(client_info.legacy_support_keys_used)
            )
        if client_info.unlisted_support_roles:
            self._flag_noncompliance(
                "client/hello sent support objects for unlisted roles: "
                + ", ".join(client_info.unlisted_support_roles)
            )
        if client_info.artwork_support is not None:
            self._flag_legacy_artwork_wire(client_info.artwork_support)
        if unimplemented := self._unimplemented_roles(client_info.supported_roles):
            self._logger.info(
                "Client offered roles/versions this server does not implement: %s", unimplemented
            )

        if self._noise_psk is not None and self._noise_psk.category is PskCategory.SENTINEL:
            self._trusted_unpaired = (
                await self._server.pairing_store.trusted_unpaired(client_id) is not None
            )

        if self._client is None:
            client = self._server.get_or_create_client(client_id)
            if not self.is_encrypted:
                # Legacy unencrypted is never paired, so drop pairing-required roles.
                initial_active = self._filter_pairing_roles(self._negotiated_roles)
            elif self._is_pairing():
                initial_active = []
            else:
                initial_active = self._roles_to_activate
            client.attach_connection(
                self,
                client_info=client_info,
                negotiated_roles=self._negotiated_roles,
                active_roles=initial_active,
            )
            self._client = client
            if self._url is not None:
                self._server.register_client_url(client_id, self._url)
        else:
            # Hello re-sent over the same connection after an in-band re-handshake.
            self._client.refresh_identity_from_hello(
                client_info, negotiated_roles=self._negotiated_roles
            )
        return True

    def _flag_legacy_artwork_wire(self, support: ClientHelloArtworkSupport) -> None:
        """Flag artwork channels declared on the wire the spec superseded."""
        legacy_keys = sorted(
            {key for channel in support.channels for key in channel.legacy_dimension_keys or ()}
        )
        if legacy_keys:
            self._flag_noncompliance(
                "client/hello artwork used pre-rename dimension keys: " + ", ".join(legacy_keys)
            )
        if any(channel.format is PictureFormat.BMP for channel in support.channels):
            self._flag_noncompliance("client/hello artwork declared the removed 'bmp' format")

    async def _admit_legacy_client_id(self, client_id: str) -> bool:
        """Whether an unauthenticated (legacy) hello may claim ``client_id``."""
        if self._expected_client_id is not None and client_id != self._expected_client_id:
            self._logger.error(
                "Unencrypted client/hello claims %r, expected %r",
                client_id,
                self._expected_client_id,
            )
            return False
        # A paired, pairing-staged, or trusted-unpaired client has proven it can
        # connect encrypted (its static key authenticated the Noise handshake);
        # never admit it unencrypted (downgrade protection).
        store = self._server.pairing_store
        if (
            await store.record_by_client_id(client_id) is not None
            or await store.staged_pairing_psk(client_id) is not None
            or await store.trusted_unpaired(client_id) is not None
        ):
            self._logger.error(
                "Rejecting unencrypted connection claiming known client %s", client_id
            )
            return False
        return True

    @property
    def _playback_capable(self) -> bool:
        """Whether this connection may ever carry playback."""
        assert self._noise_psk is not None
        assert self._client_info is not None
        if self._noise_psk.category is PskCategory.LONG_TERM:
            return True
        if self._noise_psk.category is PskCategory.SENTINEL:
            return self._client_info.unpaired_access.enabled and self._trusted_unpaired
        return False

    @property
    def _management_capable(self) -> bool:
        """Whether this connection may carry management."""
        return self._noise_psk is not None and self._noise_psk.category is PskCategory.LONG_TERM

    @property
    def _client_in_playback(self) -> bool:
        """Whether the client's group is in active/upcoming (non-stopped) playback."""
        return (
            self._client is not None and self._client.group.state is not PlaybackStateType.STOPPED
        )

    @property
    def _is_long_term_paired(self) -> bool:
        """Whether the connection was admitted by a long-term Sendspin PSK (trust ``user``)."""
        return self._noise_psk is not None and self._noise_psk.category is PskCategory.LONG_TERM

    def _filter_pairing_roles(self, role_ids: list[str]) -> list[str]:
        """Drop roles that require pairing unless the connection is long-term paired."""
        if self._is_long_term_paired:
            return role_ids
        return [rid for rid in role_ids if not role_requires_pairing(rid)]

    @property
    def _roles_to_activate(self) -> list[str]:
        """Active roles to advertise — the negotiated set when playback-capable, else empty."""
        if not self._playback_capable:
            return []
        return self._filter_pairing_roles(self._negotiated_roles)

    @property
    def _desired_activities(self) -> list[Activity]:
        """Activities the live group state warrants, plus management when enabled."""
        activities: list[Activity] = []
        if self._playback_capable and self._client_in_playback:
            activities.append(Activity.PLAYBACK)
        if self._management_active and self._management_capable:
            activities.append(Activity.MANAGEMENT)
        return activities

    @property
    def _initial_activities(self) -> list[Activity]:
        """Activities for the first server/activate, seeded by the dial intent."""
        activities: list[Activity] = []
        dialed_playback = (
            self._url is not None
            and self._server.get_connection_reason(self._url) is ConnectionReason.PLAYBACK
        )
        if self._playback_capable and (dialed_playback or self._client_in_playback):
            activities.append(Activity.PLAYBACK)
        if self._management_active and self._management_capable:
            activities.append(Activity.MANAGEMENT)
        return activities

    def _refresh_activities(self) -> None:
        """Re-send server/activate if the desired activity set changed (active_roles sticky)."""
        if self._pairing_in_progress:
            return
        if self._declared_activities is None:
            return  # not an activated encrypted connection
        desired = self._desired_activities
        if desired == self._declared_activities:
            return
        self._declared_activities = desired
        self.send_priority_message(
            ServerActivateMessage(payload=ServerActivatePayload(activities=desired))
        )

    def _subscribe_activity_events(self) -> None:
        """Watch the client's group/playback transitions to keep activities current."""
        assert self._client is not None
        self._client_event_unsub = self._client.add_event_listener(self._on_client_event)
        self._subscribe_group_events(self._client.group)

    def _subscribe_group_events(self, group: SendspinGroup) -> None:
        if self._group_event_unsub is not None:
            self._group_event_unsub()
        self._group_event_unsub = group.add_event_listener(self._on_group_event)

    def _on_client_event(self, _client: SendspinClient, event: ClientEvent) -> None:
        if isinstance(event, ClientGroupChangedEvent):
            self._subscribe_group_events(event.new_group)
            self._refresh_activities()

    def _on_group_event(self, _group: SendspinGroup, event: GroupEvent) -> None:
        if isinstance(event, GroupStateChangedEvent):
            self._refresh_activities()

    def _unsubscribe_activity_events(self) -> None:
        if self._client_event_unsub is not None:
            self._client_event_unsub()
            self._client_event_unsub = None
        if self._group_event_unsub is not None:
            self._group_event_unsub()
            self._group_event_unsub = None

    async def initiate_pairing(self, attempt: PairingAttempt) -> None:
        """Run a pairing attempt on a connection.

        A pair abort raises and leaves the connection for a retry or ``end_pairing``.
        A server-side timeout raises after leaving pairing, also keeping the connection.
        Any other failure propagates for the caller to disconnect.
        """
        if self._pairing_attempt is not None:
            raise PairingError("connection is already in a pairing attempt")
        transport = self._transport
        if not isinstance(transport, EncryptedWebSocket):
            raise PairingError("cannot pair over an unencrypted connection")
        if not self._in_pairing:
            await self._quiesce_for_pairing()
            await self._pause_writer()
            self._pairing_message_queue = asyncio.Queue()
            self._in_pairing = True
        assert self._pairing_message_queue is not None
        self._pairing_attempt = attempt
        self._pairing_messages_started = False
        dispatched = _QueuedTransport(transport, self._pairing_message_queue)
        task = create_task(self._pair(dispatched))
        self._pairing_task = task
        try:
            if not await task:
                raise PairingError("pairing failed")
        except LocalPairingAbortError:
            current_task = asyncio.current_task()
            if current_task is not None and current_task.cancelling():
                # Our own cancellation was forwarded into the child and converted; restore it.
                raise asyncio.CancelledError from None
            raise
        except PairingTimeoutError:
            # A server has no pair/abort reason for its own timeout: cancel the attempt in
            # band with the leave activate (best-effort; the client may be gone).
            with suppress(Exception):
                await self._leave_pairing()
            raise
        finally:
            self._pairing_attempt = None
            self._pairing_task = None
        await self._leave_pairing()

    async def end_pairing(self) -> None:
        """End pairing without finalizing, restoring the connection's activities and roles.

        No-op if not in pairing. Aborts any in-progress attempt with ``user_cancelled``, keeping
        the connection alive.
        If an attempt has already been finalized by the client, it completes as a success instead.
        """
        if not self._in_pairing:
            return
        task = self._pairing_task
        if task is not None and not task.done():
            task.cancel()
            with suppress(PairingError):
                await task
        await self._leave_pairing()

    async def _leave_pairing(self) -> None:
        """Exit the pairing state, returning the connection to normal service."""
        if not self._in_pairing:  # a success and a concurrent end_pairing
            return
        self._pairing_message_queue = None
        self._pairing_messages_started = False
        self._in_pairing = False
        await self._activate()
        self._resume_writer()

    async def _quiesce_for_pairing(self) -> None:
        """Quiesce playback and roles for pairing, then wait for the teardown to flush."""
        assert self._client is not None
        await self._client.quiesce_to_solo_stopped()
        self._client.set_active_roles([])
        if self._writer_task is not None and not self._writer_task.done():
            async with asyncio.timeout(QUIESCE_TIMEOUT_S):
                await self._writer_idle.wait()

    async def _pair(self, transport: EncryptedWebSocket) -> bool:
        """Run the pairing exchange."""
        try:
            if not await self._rehandshake_for_pairing_if_needed(transport):
                return False
            method = (
                self._pairing_attempt.method
                if self._pairing_attempt is not None
                else PairMethod.PAIRING_PSK
            )
            pairing_format: PairingCodeFormat | None = None
            languages: list[str] | None = None
            if method is PairMethod.DYNAMIC_PAIRING_CODE:
                assert self._pairing_attempt is not None
                pairing_format = self._negotiated_dynamic_pairing_format()
                # The language hint applies to spoken emission, so only the digits format.
                if pairing_format is PairingCodeFormat.DIGITS and self._pairing_attempt.languages:
                    languages = list(self._pairing_attempt.languages)
            # No gate on the hello-advertised methods: the advertisement may lag the client's
            # live pairing config (management can change it mid-connection). The client
            # arbitrates, aborting an unsupported method with ``method_not_supported``.
            await transport.send_str(
                ServerActivateMessage(
                    payload=ServerActivatePayload(
                        activities=[Activity.PAIRING],
                        active_roles=[],
                        pairing=ActivatePairing(
                            method=method,
                            format=pairing_format.value if pairing_format is not None else None,
                            languages=languages,
                        ),
                    )
                ).to_json()
            )
            record = await self._run_pairing_protocol(method, transport, pairing_format)
        except asyncio.CancelledError:
            # A cancelled attempt ends like any local abort: the task never reports
            # cancelled(), so awaiting callers see the abort rather than the cancel.
            await abort_pairing(transport, PairAbortReason.USER_CANCELLED)
        self._pairing_attempt = None
        if record is None:  # verified an existing pairing: no new record, no re-handshake
            self._logger.info(
                "Verified pairing with client %s via %s", self._client_id, method.value
            )
            return True
        self._logger.info("Paired with client %s via %s", self._client_id, method.value)
        # The client finalized, so the attempt has succeeded and both sides hold the record:
        # a late cancel must not abort it or corrupt the re-handshake. Complete the tail and
        # report the success; the one absorbed cancel ends with the pairing in effect.
        rehandshake = create_task(self._rehandshake_to(transport, record.as_resolved()))
        try:
            return await asyncio.shield(rehandshake)
        except asyncio.CancelledError:
            return await rehandshake

    async def _run_pairing_protocol(
        self,
        method: PairMethod,
        transport: EncryptedWebSocket,
        pairing_format: PairingCodeFormat | None,
    ) -> ServerPairingRecord | None:
        """Run ``method``'s exchange, returning the record (``None`` when verifying)."""
        assert self._client_id is not None
        self._pairing_index += 1
        pairing_index = self._pairing_index
        if method is PairMethod.PAIRING_PSK:
            # The attempt is absent when the client dialed in with a staged Pairing PSK.
            attempt = self._pairing_attempt
            return await run_pairing_psk_server(
                transport,
                client_id=self._client_id,
                store=self._server.pairing_store,
                owner=attempt.owner if attempt is not None else None,
            )
        assert self._pairing_attempt is not None
        assert self._pairing_attempt.pairing_code_provider is not None
        assert self._handshake_hash is not None
        assert self._noise_psk is not None
        verify = self._pairing_attempt.verify
        if verify and self._noise_psk.category is not PskCategory.LONG_TERM:
            raise PairingError("verification requires an existing pairing")
        if method is PairMethod.STATIC_PAIRING_CODE:
            return await run_static_pairing_code_server(
                transport,
                handshake_hash=self._handshake_hash,
                pairing_index=pairing_index,
                pairing_code_provider=self._pairing_attempt.pairing_code_provider,
                client_id=self._client_id,
                store=self._server.pairing_store,
                verify=verify,
                on_pair_pending=self._pairing_attempt.on_pair_pending,
                owner=self._pairing_attempt.owner,
            )
        assert pairing_format is not None
        return await run_dynamic_pairing_code_server(
            transport,
            handshake_hash=self._handshake_hash,
            pairing_index=pairing_index,
            pairing_code_provider=self._pairing_attempt.pairing_code_provider,
            pairing_format=pairing_format,
            client_id=self._client_id,
            store=self._server.pairing_store,
            verify=verify,
            on_pair_pending=self._pairing_attempt.on_pair_pending,
            owner=self._pairing_attempt.owner,
        )

    def _negotiated_dynamic_pairing_format(self) -> PairingCodeFormat:
        """Return the attempt's emission format, checked against the advertised descriptor."""
        assert self._client_info is not None
        assert self._pairing_attempt is not None
        requested = self._pairing_attempt.pairing_format
        assert requested is not None
        descriptor = next(
            (
                d
                for d in (self._client_info.supported_pair_methods or [])
                if d.method is PairMethod.DYNAMIC_PAIRING_CODE
            ),
            None,
        )
        if descriptor is None:
            # The advertisement lags a management enable; the client arbitrates.
            return requested
        if not descriptor.formats:
            raise PairingError("client's dynamic_pairing_code descriptor is missing formats")
        if requested.value not in descriptor.formats:
            raise PairingError(f"client does not offer the {requested.value} emission format")
        return requested

    async def _rehandshake_for_pairing_if_needed(self, transport: Transport) -> bool:
        """If the attempt needs a PSK other than the current one, rehandshake and redo hellos."""
        attempt = self._pairing_attempt
        if attempt is None:
            return True
        assert self._noise_psk is not None
        if attempt.method is PairMethod.PAIRING_PSK:
            assert attempt.pairing_psk is not None
            if (
                self._noise_psk.category is PskCategory.PAIRING
                and self._noise_psk.psk == attempt.pairing_psk
            ):
                return True
            target = ResolvedPsk(
                psk_id_for(attempt.pairing_psk), attempt.pairing_psk, PskCategory.PAIRING
            )
        else:
            if self._noise_psk.category in (PskCategory.SENTINEL, PskCategory.LONG_TERM):
                # Long-term: verification runs over the existing PSK.
                # Sentinel: a fresh pairing-code pairing.
                return True
            target = ResolvedPsk(psk_id_for(SENTINEL_PSK), SENTINEL_PSK, PskCategory.SENTINEL)
        assert isinstance(transport, EncryptedWebSocket)
        return await self._rehandshake_to(transport, target)

    async def _rehandshake_to(self, transport: EncryptedWebSocket, psk: ResolvedPsk) -> bool:
        """Re-handshake onto ``psk`` and redo the hello dance."""
        assert self._client_id is not None
        assert self._handshake_hash is not None
        result = await run_rehandshake_server(
            transport,
            local_identity=self._server.identity,
            client_id=self._client_id,
            suite=transport.session.suite,
            prologue=self._handshake_hash,
            psk=psk,
        )
        self._noise_psk = result.psk
        self._handshake_hash = result.handshake_hash
        self._pairing_index = 0
        return await self._send_server_hello_and_recv(transport)

    async def _activate(self) -> None:
        """Send ``server/activate`` and reconcile the client's active roles."""
        assert self._transport is not None
        if self._declared_activities is None:
            self._declared_activities = self._initial_activities
        else:
            self._declared_activities = self._desired_activities
        active_roles = self._roles_to_activate
        await self._transport.send_str(
            ServerActivateMessage(
                payload=ServerActivatePayload(
                    activities=self._declared_activities,
                    active_roles=active_roles,
                )
            ).to_json()
        )
        assert self._client is not None
        self._client.set_active_roles(active_roles)

    async def refresh_trusted_unpaired(self) -> None:
        """Re-read the trusted-unpaired approval and re-activate roles."""
        if self._noise_psk is None or self._noise_psk.category is not PskCategory.SENTINEL:
            return
        if self._client is None or self._declared_activities is None:
            return
        assert self._client_id is not None
        self._trusted_unpaired = (
            await self._server.pairing_store.trusted_unpaired(self._client_id) is not None
        )
        active_roles = self._roles_to_activate
        self._declared_activities = self._desired_activities
        self.send_priority_message(
            ServerActivateMessage(
                payload=ServerActivatePayload(
                    activities=self._declared_activities, active_roles=active_roles
                )
            )
        )
        self._client.set_active_roles(active_roles)

    def enable_management(self) -> None:
        """Add ``management`` to this connection's activities; requires a paired connection."""
        if not self._management_capable:
            msg = "management requires a paired (long-term Sendspin PSK) connection"
            raise RuntimeError(msg)
        if self._management_active:
            return
        self._management_active = True
        self._refresh_activities()

    def disable_management(self) -> None:
        """Drop ``management`` from this connection's activities, leaving playback intact."""
        if not self._management_active:
            return
        self._management_active = False
        self._refresh_activities()

    def _resolve_management(self, payload: ManagementResultPayload) -> None:
        """Deliver a management reply, draining the waiter slot."""
        waiter = self._management_waiter
        if waiter is None:
            self._flag_noncompliance("sent an unsolicited management/result")
            return
        # Clear even an abandoned waiter, so its late reply can't match the next request.
        self._management_waiter = None
        if not waiter.done():
            waiter.set_result(payload)

    async def _management_request[T: ManagementResultPayload](
        self, message: ServerMessage, expected: type[T]
    ) -> T:
        """Send a management request and await its single reply of type ``expected``."""
        # No timeout: replies are matched to requests by order, not id (one in flight).
        if self._management_waiter is not None:
            raise RuntimeError("a management request is already in flight")
        if self._transport is None or self._disconnecting:
            raise RuntimeError("connection is not active")
        waiter: asyncio.Future[ManagementResultPayload] = asyncio.get_running_loop().create_future()
        self._management_waiter = waiter
        self.send_priority_message(message)
        payload = await waiter
        if not isinstance(payload, expected):
            raise RuntimeError(  # noqa: TRY004 - protocol violation, not a type error
                f"expected a {expected.__name__} reply, got {type(payload).__name__}"
            )
        return payload

    def unpair(self) -> None:
        """Tell the client to drop this server's pairing record (it then closes)."""
        self.send_priority_message(ServerUnpairMessage())

    async def list_records(
        self,
    ) -> tuple[ManagementResult, list[RecordSummary], StorageAccounting | None]:
        """Return the result code, the client's pairing records, and its storage accounting."""
        payload = await self._management_request(
            ManagementListRecordsMessage(), ManagementResultPayload
        )
        records = payload.data.records if payload.data and payload.data.records else []
        return payload.result, records, payload.storage

    async def add_record(self, *, psk: bytes, server_id: str | None) -> ManagementResult:
        """Add a pairing record on the client."""
        payload = await self._management_request(
            ManagementAddRecordMessage(
                payload=ManagementAddRecordPayload(psk=b64url_encode(psk), server_id=server_id)
            ),
            ManagementResultPayload,
        )
        return payload.result

    async def remove_record(self, *, psk_id: str) -> ManagementResult:
        """Remove a pairing record from the client."""
        payload = await self._management_request(
            ManagementRemoveRecordMessage(payload=ManagementRemoveRecordPayload(psk_id=psk_id)),
            ManagementResultPayload,
        )
        return payload.result

    async def get_pairing_config(
        self,
    ) -> tuple[ManagementResult, ManagementResultData, StorageAccounting | None]:
        """Return the result code, the client's pairing configuration (no secrets), and storage."""
        payload = await self._management_request(
            ManagementGetPairingConfigMessage(), ManagementResultPayload
        )
        data = payload.data if payload.data is not None else ManagementResultData()
        return payload.result, data, payload.storage

    async def set_pairing_config(
        self, patch: ManagementSetPairingConfigPayload
    ) -> ManagementResult:
        """Apply a pairing-config patch on the client."""
        payload = await self._management_request(
            ManagementSetPairingConfigMessage(payload=patch), ManagementResultPayload
        )
        return payload.result

    async def open_pairing_window(self) -> ManagementResult:
        """Open a pairing window on the client in place of the operator gesture."""
        payload = await self._management_request(
            ManagementOpenPairingWindowMessage(), ManagementResultPayload
        )
        return payload.result

    def _start_message_loops(self) -> None:
        """Spawn the reader/writer tasks."""
        self._writer_task = create_task(self._writer())
        self._message_loop_task = create_task(self._run_message_loop())

    async def _pause_writer(self) -> None:
        """Stop the writer task, leaving the reader loop running."""
        # Cancelling mid-send is nonce-safe: no await separates encrypt() from the
        # transport write.
        if self._writer_task is not None and not self._writer_task.done():
            self._writer_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._writer_task
            self._writer_task = None

    def _resume_writer(self) -> None:
        """Restart the writer task, unless the connection is being torn down."""
        if self._disconnecting or self._closing:
            return
        if self._writer_task is None or self._writer_task.done():
            self._writer_task = create_task(self._writer())

    async def _cleanup_connection(self) -> None:
        wsock = self._wsock_client or self._wsock_server
        if wsock is not None and not wsock.closed:
            with suppress(Exception):
                await wsock.close()
        await self.disconnect(retry_connection=not self._closing)

    def _try_route_to_pairing_queue(self, msg: WSMessage) -> bool:
        """Forward a message to the pairing handler; return whether it was routed."""
        if self._pairing_message_queue is None:
            return False
        if (
            msg.type is WSMsgType.TEXT
            and self._peek_message_type(cast("str", msg.data)) in _PAIR_TRANSITION_TYPES
        ):
            self._pairing_messages_started = True
        if not self._pairing_messages_started:
            return False
        self._pairing_message_queue.put_nowait(msg)
        return True

    async def _run_message_loop(self) -> None:
        transport = self._transport
        assert transport is not None
        cancelled = False
        try:
            async for msg in transport:
                timestamp_us = self._server.clock.now_us()

                if self._try_route_to_pairing_queue(msg):
                    continue

                if msg.type == WSMsgType.ERROR:
                    self._logger.warning("WebSocket error: %s", transport.exception() or "unknown")
                    break

                if msg.type == WSMsgType.BINARY:
                    self._route_inbound_binary(cast("bytes", msg.data))
                    continue

                if msg.type != WSMsgType.TEXT:
                    self._logger.debug("Ignoring message type: %s", msg.type.name)
                    continue

                if self._pairing_in_progress:
                    continue

                text = cast("str", msg.data)
                try:
                    message = self._deserialize_client_message(text)
                except Exception:
                    if self._peek_message_type(text) in _PAIRING_MESSAGE_TYPES:
                        # In flight from before the client observed the leave activate.
                        self._logger.debug("Discarding pairing message: not in pairing")
                        continue
                    raise
                await self._handle_message(message, timestamp_us)
            else:
                # Loop exited normally (iterator exhausted) - connection closed
                close_code = transport.close_code
                log_func = (
                    self._logger.debug if close_code in (1000, 1001) else self._logger.warning
                )
                log_func(
                    "WebSocket closed, close_code=%s",
                    close_code,
                )
        except asyncio.CancelledError:
            cancelled = True
            self._logger.debug("Message loop cancelled")
        except ClientComplianceError:
            # Strict mode: hard-reject (no warm reconnect). Cleanup reads _closing.
            # flag_noncompliance already logged the reason at error before raising.
            self._closing = True
        except Exception:
            self._logger.exception("Unexpected error inside websocket API")
        finally:
            if self._pairing_message_queue is not None:
                self._pairing_message_queue.put_nowait(WSMessage(WSMsgType.CLOSE, None, ""))
            if self._writer_task and not self._writer_task.done():
                self._writer_task.cancel()
            if not cancelled:
                self._connection_done.set()

    def _route_inbound_binary(self, data: bytes) -> None:
        """Route an inbound binary chunk to the role that declares its message type."""
        if self._client is None:
            return
        if len(data) < BINARY_HEADER_SIZE:
            self._logger.warning("Inbound binary message shorter than header, dropping")
            return
        header = unpack_binary_header(data)
        payload = data[BINARY_HEADER_SIZE:]
        for role in self._client.active_roles:
            if role.handles_inbound_binary(header.message_type):
                role.on_binary_chunk(header.message_type, header.timestamp_us, payload)
                return
        self._logger.warning(
            "Received unhandled binary message type %s from client", header.message_type
        )

    async def _handle_message(self, message: ClientMessage, timestamp_us: int) -> None:
        """Handle a single client message, dispatching to roles or the connection."""
        if isinstance(message, ClientHelloMessage):
            self._flag_noncompliance("sent a second client/hello after the hello exchange")
            return

        if isinstance(message, ClientTimeMessage):
            client_time = message.payload
            self.send_priority_message(
                ServerTimeMessage(
                    payload=ServerTimePayload(
                        client_transmitted=client_time.client_transmitted,
                        server_received=timestamp_us,
                        server_transmitted=0,  # Set at actual send time
                    )
                )
            )
            return

        if isinstance(message, ClientStateMessage):
            await self._handle_client_state(message.payload)
            return

        if isinstance(message, StreamRequestFormatMessage):
            if self._client is None:
                return
            fmt = message.payload
            self._flag_inactive_role_payloads(
                "stream/request-format",
                {"player": fmt.player, "artwork": fmt.artwork, "visualizer": fmt.visualizer},
            )
            for role in self._client.active_roles:
                role.on_stream_request_format(fmt)
            return

        if isinstance(message, ClientCommandMessage):
            if self._client is None:
                return
            self._flag_inactive_role_payloads(
                "client/command", {"controller": message.payload.controller}
            )
            for role in self._client.active_roles:
                role.on_command(message.payload)
            return

        if isinstance(message, ClientStreamStartMessage):
            if self._client is None:
                return
            for role in self._client.active_roles:
                role.on_client_stream_start(message.payload)
            return

        if isinstance(message, ClientStreamEndMessage):
            if self._client is None:
                return
            for role in self._client.active_roles:
                role.on_client_stream_end()
            return

        if isinstance(message, ManagementResultMessage):
            self._resolve_management(message.payload)
            return

        if isinstance(message, ClientGoodbyeMessage):
            self._logger.debug(
                "Received client/goodbye with reason: %s",
                message.payload.reason,
            )
            self._last_goodbye_reason = message.payload.reason
            retry = message.payload.reason == GoodbyeReason.RESTART
            await self.disconnect(retry_connection=retry)
            return

    async def _handle_client_state(self, payload: ClientStatePayload) -> None:
        """Apply a client/state update: compliance checks, initial-state gate, dispatch."""
        if self._client is None:
            return

        # Validate before applying initial state.
        is_initial = self.requires_initial_state() and not self._initial_state_received
        if is_initial:
            self._flag_initial_state_deviations(payload)
        if payload.legacy_state_used:
            self._flag_noncompliance("client/state used the legacy top-level 'state' field")
        self._flag_inactive_role_payloads(
            "client/state", {"player": payload.player, "source": payload.source}
        )
        for role in self._client.active_roles:
            for reason in role.client_state_deviations(payload):
                self._flag_noncompliance(f"client/state {reason}")

        if is_initial:
            self._initial_state_received = True
            if self._initial_state_timeout_handle is not None:
                self._initial_state_timeout_handle.cancel()
                self._initial_state_timeout_handle = None
            self._client.mark_connected()
            self._server.on_client_first_connect(self._client.client_id)
            self._flush_pending_binary()

        if payload.available is not None and payload.available != self._client.available:
            await self._client.handle_availability_change(available=payload.available)
        for role in self._client.active_roles:
            role.on_client_state(payload)

    def _late_binary_diagnostics(
        self, role: Role, entry: _RoleQueueEntry, now_us: int, elapsed_us: int
    ) -> str:
        """Build the field list that tells the late-drop regimes apart.

        enq_lead is the margin the chunk had when it was queued and queue_age how
        long it then waited, which separates a thin producer lead from a chunk that
        aged in transit. buf reports the tracked device buffer, and stream_elapsed
        marks a startup or restart window. Fields with no value are omitted.
        """
        fields: list[str] = []
        if entry.enqueued_at_us:
            # Same effective play time the late-drop decision uses, so this field and
            # late_by_us in the surrounding line share one basis.
            effective_ts_us = entry.timestamp_us - role.get_output_delay_us()
            fields.append(f"enq_lead_ms={(effective_ts_us - entry.enqueued_at_us) / 1000:.0f}")
            fields.append(f"queue_age_ms={(now_us - entry.enqueued_at_us) / 1000:.0f}")
        if (tracker := role.get_buffer_tracker()) is not None:
            fields.append(f"buf_ms={tracker.buffered_horizon_us(now_us) / 1000:.0f}")
            fields.append(f"buf_bytes={tracker.buffered_bytes}/{tracker.capacity_bytes}")
        fields.append(f"stream_elapsed_s={elapsed_us / 1_000_000:.1f}")
        return " ".join(fields)

    def _check_late_binary(
        self,
        handling: BinaryHandling | None,
        role: Role | None,
        entry: _RoleQueueEntry,
        message_type: int = 0,
    ) -> bool:
        """Check if a binary message's playback time has passed and should be dropped.

        Compares the message's playback timestamp against the current clock. During the
        grace period (configurable per-role), late messages are allowed through to give
        clients time to build their initial buffer.
        """
        timestamp_us = entry.timestamp_us
        # timestamp_us=0 means "no playback semantics" - skip late detection
        if handling is None or role is None or not handling.drop_late or timestamp_us == 0:
            return False

        now = self._server.clock.now_us()
        if role._stream_start_time_us is None:  # noqa: SLF001
            role._stream_start_time_us = now  # noqa: SLF001
        elapsed = now - role._stream_start_time_us  # noqa: SLF001
        in_grace_period = elapsed < handling.grace_period_us
        late_by_us = now - (timestamp_us - role.get_output_delay_us())

        if late_by_us > 0 and not in_grace_period:
            role._late_skips_since_log += 1  # noqa: SLF001
            self._logger.debug(
                "Discarding late chunk type=%s role=%s: late_by=%.1fms, plays_in=%.1fms",
                message_type,
                role.role_family,
                late_by_us / 1000,
                -late_by_us / 1000,
            )
            now_s = time.monotonic()
            if now_s - role._last_late_log_s >= _WARN_INTERVAL_S:  # noqa: SLF001
                qsize, qmax = self.queue_status()
                self._logger.warning(
                    "Late binary type=%s role=%s: skipping %s chunk(s); "
                    "late_by_us=%s ts_us=%s now_us=%s queue=%s/%s %s",
                    message_type,
                    role.role_family,
                    role._late_skips_since_log,  # noqa: SLF001
                    late_by_us,
                    timestamp_us,
                    now,
                    qsize,
                    qmax,
                    self._late_binary_diagnostics(role, entry, now, elapsed),
                )
                role._late_skips_since_log = 0  # noqa: SLF001
                role._last_late_log_s = now_s  # noqa: SLF001
            return True
        return False

    async def _send_message(
        self,
        wsock: Transport,
        message: ServerMessage,
    ) -> None:
        """Send a single message, handling time message timestamps."""
        if isinstance(message, ServerTimeMessage):
            # Update timestamp to actual send time
            message = ServerTimeMessage(
                payload=ServerTimePayload(
                    client_transmitted=message.payload.client_transmitted,
                    server_received=message.payload.server_received,
                    server_transmitted=self._server.clock.now_us(),
                )
            )
        elif isinstance(message, StreamStartMessage | StreamClearMessage | StreamEndMessage):
            # Stamp send time on the dequeued (send-once) payload.
            message.payload.server_transmitted = self._server.clock.now_us()
        await wsock.send_str(message.to_json())

    async def _send_binary_data(
        self,
        wsock: Transport,
        role: str,
        entry: _RoleQueueEntry,
        buffer_tracker: BufferTracker | None,
    ) -> None:
        """Send a binary frame with buffer tracking."""
        assert entry.binary is not None
        binary = entry.binary
        start_s = time.monotonic()
        await wsock.send_bytes(binary.data)
        elapsed_ms = (time.monotonic() - start_s) * 1000
        if elapsed_ms >= 50.0:
            # Slow writes indicate transport/backpressure issues but are not fatal.
            # Extreme stalls surface at WARNING (rate-limited) so default-level
            # logs show transport backpressure alongside any late-binary drops.
            if elapsed_ms >= 500.0:
                self._slow_send_count += 1
            now_s = time.monotonic()
            if elapsed_ms >= 500.0 and now_s - self._last_slow_send_log_s >= _WARN_INTERVAL_S:
                self._logger.warning(
                    "Slow send_bytes: %.1fms size=%s ts_us=%s role=%s; "
                    "%s stall(s) over 500ms since last report",
                    elapsed_ms,
                    len(binary.data),
                    entry.timestamp_us,
                    role,
                    self._slow_send_count,
                )
                self._slow_send_count = 0
                self._last_slow_send_log_s = now_s
            else:
                self._logger.debug(
                    "Slow send_bytes: %.1fms size=%s ts_us=%s role=%s",
                    elapsed_ms,
                    len(binary.data),
                    entry.timestamp_us,
                    role,
                )

        # Buffer tracking via role's tracker (framework-managed)
        if (
            buffer_tracker is not None
            and binary.buffer_end_time_us is not None
            and binary.buffer_byte_count is not None
        ):
            buffer_tracker.register(
                binary.buffer_end_time_us,
                binary.buffer_byte_count,
                binary.duration_us or 0,
            )

    #### Role Queue Heap Management ####
    #
    # Two-level heap: per-role min-heaps hold entries sorted by (timestamp, seq).
    # A global _ready_roles heap tracks which role has the earliest head entry.
    # _delayed_roles tracks roles blocked by backpressure until a future time;
    # _promote_ready_roles moves them back to _ready_roles when their time comes.
    # Generation counters prevent stale delayed entries from unblocking a re-blocked role.

    def _schedule_role_head(self, role: str) -> None:
        if role in self._blocked_until_us:
            return
        if role_queue := self._role_queues.get(role):
            head_sort_ts, head_seq, _ = role_queue[0]
            heapq.heappush(self._ready_roles, (head_sort_ts, head_seq, role))

    def _discard_role_head(self, role: str) -> None:
        role_queue = self._role_queues.get(role)
        if not role_queue:
            return
        heapq.heappop(role_queue)
        self._queue_size = max(self._queue_size - 1, 0)
        if not role_queue:
            self._role_queues.pop(role, None)

    def _peek_ready_entry(self) -> tuple[str, _RoleQueueEntry, int, int] | None:
        # TODO: any reason why a peek method does a full pop and push operation?
        # TODO: or is it most of the time not pushing back? i mean does this peek
        # TODO: mutate anything or not?
        while self._ready_roles:
            sort_ts, seq, role = heapq.heappop(self._ready_roles)
            if role in self._blocked_until_us:
                continue
            role_queue = self._role_queues.get(role)
            if not role_queue:
                continue
            head_sort_ts, head_seq, head_entry = role_queue[0]
            if head_sort_ts != sort_ts or head_seq != seq:
                heapq.heappush(self._ready_roles, (head_sort_ts, head_seq, role))
                continue
            return role, head_entry, head_sort_ts, head_seq
        return None

    def _block_role(self, role: str, ready_at_us: int) -> None:
        self._blocked_until_us[role] = ready_at_us
        generation = self._block_generation[role] + 1
        self._block_generation[role] = generation
        heapq.heappush(self._delayed_roles, (ready_at_us, generation, role))

    def _promote_ready_roles(self, now_us: int) -> None:
        while self._delayed_roles and self._delayed_roles[0][0] <= now_us:
            ready_at_us, generation, role = heapq.heappop(self._delayed_roles)
            if self._block_generation.get(role, 0) != generation:
                continue
            blocked_until = self._blocked_until_us.get(role)
            if blocked_until is None or blocked_until != ready_at_us:
                continue
            self._blocked_until_us.pop(role, None)
            self._schedule_role_head(role)

    async def _process_priority_messages(
        self,
        wsock: Transport,
    ) -> bool:
        """Send one queued priority message if available."""
        if not self._priority_messages:
            return False
        message = self._priority_messages.popleft()
        self._queue_size = max(self._queue_size - 1, 0)
        await self._send_message(wsock, message)
        return True

    async def _process_normal_messages(
        self,
        wsock: Transport,
        ready_entry: tuple[str, _RoleQueueEntry, int, int] | None,
    ) -> bool:
        """Send one queued non-role message when no role entry is ready."""
        if ready_entry is not None or not self._normal_messages:
            return False
        message = self._normal_messages.popleft()
        self._queue_size = max(self._queue_size - 1, 0)
        await self._send_message(wsock, message)
        return True

    def _fresh_send_stats(self) -> dict[str, float | int]:
        return {
            "count": 0,
            "send_gap_sum_ms": 0.0,
            "send_gap_min_ms": 1e9,
            "send_gap_max_ms": 0.0,
            "ts_gap_sum_ms": 0.0,
            "ts_gap_min_ms": 1e9,
            "ts_gap_max_ms": 0.0,
            "buf_count": 0,
            "buf_sum_ms": 0.0,
            "buf_min_ms": 1e9,
            "buf_max_ms": 0.0,
        }

    def _update_send_stats(
        self,
        role: str,
        *,
        send_gap_ms: float,
        ts_gap_ms: float,
        buffer_tracker: BufferTracker | None,
        now_us: int,
    ) -> None:
        stats = self._send_stats_by_role.setdefault(role, self._fresh_send_stats())
        stats["count"] += 1
        stats["send_gap_sum_ms"] += send_gap_ms
        stats["send_gap_min_ms"] = min(stats["send_gap_min_ms"], send_gap_ms)
        stats["send_gap_max_ms"] = max(stats["send_gap_max_ms"], send_gap_ms)
        stats["ts_gap_sum_ms"] += ts_gap_ms
        stats["ts_gap_min_ms"] = min(stats["ts_gap_min_ms"], ts_gap_ms)
        stats["ts_gap_max_ms"] = max(stats["ts_gap_max_ms"], ts_gap_ms)
        if buffer_tracker is not None:
            buf_ms = buffer_tracker.buffered_horizon_us(now_us) / 1000
            stats["buf_count"] += 1
            stats["buf_sum_ms"] += buf_ms
            stats["buf_min_ms"] = min(stats["buf_min_ms"], buf_ms)
            stats["buf_max_ms"] = max(stats["buf_max_ms"], buf_ms)

    def _log_send_summaries_if_due(self) -> None:
        if not self._logger.isEnabledFor(logging.DEBUG):
            return
        now_s = time.monotonic()
        if now_s - self._send_summary_last_log_s < 5.0:
            return
        self._send_summary_last_log_s = now_s
        for role_name, role_stats in self._send_stats_by_role.items():
            count = int(role_stats["count"])
            if count <= 0:
                continue
            avg_send = role_stats["send_gap_sum_ms"] / count
            avg_ts = role_stats["ts_gap_sum_ms"] / count
            if role_stats["buf_count"] > 0:
                avg_buf = role_stats["buf_sum_ms"] / role_stats["buf_count"]
                self._logger.debug(
                    "Send summary role=%s samples=%s "
                    "send_gap_ms(avg=%.1f min=%.1f max=%.1f) "
                    "ts_gap_ms(avg=%.1f min=%.1f max=%.1f) "
                    "buf_ms(avg=%.1f min=%.1f max=%.1f)",
                    role_name,
                    count,
                    avg_send,
                    role_stats["send_gap_min_ms"],
                    role_stats["send_gap_max_ms"],
                    avg_ts,
                    role_stats["ts_gap_min_ms"],
                    role_stats["ts_gap_max_ms"],
                    avg_buf,
                    role_stats["buf_min_ms"],
                    role_stats["buf_max_ms"],
                )
            else:
                self._logger.debug(
                    "Send summary role=%s samples=%s "
                    "send_gap_ms(avg=%.1f min=%.1f max=%.1f) "
                    "ts_gap_ms(avg=%.1f min=%.1f max=%.1f)",
                    role_name,
                    count,
                    avg_send,
                    role_stats["send_gap_min_ms"],
                    role_stats["send_gap_max_ms"],
                    avg_ts,
                    role_stats["ts_gap_min_ms"],
                    role_stats["ts_gap_max_ms"],
                )
            self._send_stats_by_role[role_name] = self._fresh_send_stats()

    async def _process_binary_role_messages(
        self,
        wsock: Transport,
        role: str,
        entry: _RoleQueueEntry,
        now_us: int,
    ) -> tuple[bool, int]:
        assert entry.binary is not None

        # Look up handling info for late detection + buffer tracking
        cached = None
        if self._client is not None:
            cached = self._client.get_binary_handling_cached(entry.binary.message_type)
        handling = cached[0] if cached else None
        handling_role = cached[1] if cached else None

        # Drop late messages if role requests it
        if (
            handling is not None
            and handling_role is not None
            and self._check_late_binary(handling, handling_role, entry, entry.binary.message_type)
        ):
            self._discard_role_head(role)
            self._schedule_role_head(role)
            return False, now_us

        # Check backpressure from buffer tracker
        wait_us = 0
        buffer_tracker = None
        if handling is not None and handling_role is not None:
            if handling.buffer_track:
                buffer_tracker = handling_role.get_buffer_tracker()
            if buffer_tracker is not None:
                buffer_tracker.prune_consumed(now_us)
                bytes_needed = entry.binary.buffer_byte_count or 0
                duration_needed_us = entry.binary.duration_us or 0
                wait_us = max(
                    wait_us,
                    buffer_tracker.time_until_ready(
                        bytes_needed,
                        duration_needed_us,
                        end_time_us=entry.binary.buffer_end_time_us,
                    ),
                )

        if wait_us > 0:
            # Block this role until buffer has space
            self._block_role(role, now_us + wait_us)
            return False, now_us

        debug_enabled = self._logger.isEnabledFor(logging.DEBUG)
        last_send_us: int | None = None
        last_ts_us: int | None = None
        send_gap_ms = 0.0
        ts_gap_ms = 0.0
        if debug_enabled:
            timestamp_us = entry.timestamp_us
            last_send_us = self._last_send_time_us_by_role.get(role)
            last_ts_us = self._last_timestamp_us_by_role.get(role)
            send_gap_ms = (now_us - last_send_us) / 1000 if last_send_us is not None else 0
            ts_gap_ms = (timestamp_us - last_ts_us) / 1000 if last_ts_us is not None else 0
            self._last_send_time_us_by_role[role] = now_us
            self._last_timestamp_us_by_role[role] = timestamp_us

        self._discard_role_head(role)
        await self._send_binary_data(wsock, role, entry, buffer_tracker)

        if debug_enabled and last_send_us is not None and last_ts_us is not None:
            self._update_send_stats(
                role,
                send_gap_ms=send_gap_ms,
                ts_gap_ms=ts_gap_ms,
                buffer_tracker=buffer_tracker,
                now_us=now_us,
            )
        if debug_enabled:
            self._log_send_summaries_if_due()
        self._schedule_role_head(role)
        return True, self._server.clock.now_us()

    async def _process_role_messages(
        self,
        wsock: Transport,
        ready_entry: tuple[str, _RoleQueueEntry, int, int],
        now_us: int,
    ) -> tuple[bool, int]:
        """Process one ready role entry."""
        role, entry, _sort_ts, _seq = ready_entry

        # Binary entries with a stale epoch are discarded (stream was cleared/ended).
        # JSON entries skip this check - they are always delivered.
        if entry.binary is not None and entry.epoch != self._epoch_by_role[role]:
            self._discard_role_head(role)
            self._schedule_role_head(role)
            return False, now_us

        if entry.json_message is not None:
            self._discard_role_head(role)
            # Merge consecutive state-like messages at send time.
            message = entry.json_message
            while True:
                role_queue = self._role_queues.get(role)
                if not role_queue:
                    break
                _, _, next_entry = role_queue[0]
                if next_entry.json_message is None:
                    break
                merged = self._merge_state_messages(message, next_entry.json_message)
                if merged is None:
                    break
                message = merged
                self._discard_role_head(role)
            await self._send_message(wsock, message)
            self._schedule_role_head(role)
            return True, self._server.clock.now_us()

        return await self._process_binary_role_messages(wsock, role, entry, now_us)

    async def _wait_for_writer_work(self, now_us: int) -> None:
        """Sleep until new work arrives or next delayed role becomes ready."""
        self._writer_wakeup.clear()
        if self._priority_messages or self._normal_messages or self._ready_roles:
            return

        sleep_s = None
        if self._delayed_roles:
            next_ready_us = self._delayed_roles[0][0]
            sleep_s = max((next_ready_us - now_us) / 1_000_000, 0.0)
        else:
            self._writer_idle.set()

        try:
            if sleep_s is None:
                await self._writer_wakeup.wait()
            else:
                await asyncio.wait_for(self._writer_wakeup.wait(), timeout=sleep_s)
        except TimeoutError:
            pass

    async def _writer(self) -> None:
        """Send queued messages to the client, respecting role timing and backpressure."""
        wsock = self._transport
        assert wsock is not None

        clock_now_us = self._server.clock.now_us

        iterations_since_yield = 0
        now_us = clock_now_us()

        try:
            while not wsock.closed and not self._closing:
                # Periodic yield to prevent event loop starvation
                if iterations_since_yield >= 50:
                    await asyncio.sleep(0)
                    iterations_since_yield = 0
                    now_us = clock_now_us()

                if await self._process_priority_messages(wsock):
                    now_us = clock_now_us()
                    iterations_since_yield = 0
                    continue

                now_us = clock_now_us()
                self._promote_ready_roles(now_us)

                ready_entry = self._peek_ready_entry()
                has_normal = bool(self._normal_messages)

                if ready_entry is None and not has_normal:
                    await self._wait_for_writer_work(now_us)
                    continue

                if await self._process_normal_messages(wsock, ready_entry):
                    now_us = clock_now_us()
                    iterations_since_yield = 0
                    continue

                assert ready_entry is not None
                sent, now_us = await self._process_role_messages(wsock, ready_entry, now_us)
                if sent:
                    iterations_since_yield = 0
                    continue
                iterations_since_yield += 1
        except asyncio.CancelledError:
            self._logger.debug("Writer cancelled")
        except Exception:
            self._logger.exception("Writer failed")
            # Close the websocket to signal the message loop to exit
            if not wsock.closed:
                with suppress(Exception):
                    await wsock.close()
        finally:
            self._writer_idle.set()

    async def handle_client(self) -> None:
        """Run the complete websocket connection lifecycle."""
        try:
            await self._setup_connection()
            if not await self._exchange_hellos():
                return
            if self.is_encrypted:
                self._subscribe_activity_events()
            self._start_message_loops()
            await self._connection_done.wait()
        except HandshakeAbortedError as exc:
            self._logger.debug("Noise handshake aborted: %s", exc)
        except PairingError as exc:
            self._logger.debug("Pairing aborted: %s", exc)
        finally:
            await self._cleanup_connection()
