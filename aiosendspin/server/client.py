"""Persistent Sendspin client (device) state.

SendspinClient represents a client device across reconnects. It may have an active
WebSocket connection (SendspinConnection) or be disconnected while still retaining
its identity, group membership, and per-role persistent state (e.g. BufferTracker).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, TypeVar

from aiosendspin.models.core import (
    ClientHelloPayload,
    StreamStartMessage,
)
from aiosendspin.models.types import (
    BinaryMessageType,
    GoodbyeReason,
    PlaybackStateType,
    Roles,
    TrustLevel,
    has_role,
    has_role_family,
)
from aiosendspin.noise.trust_store import PskCategory
from aiosendspin.util import create_task

from .compliance import ClientComplianceError
from .events import ClientEvent, ClientGroupChangedEvent
from .roles import Role
from .roles.base import BinaryHandling
from .roles.negotiation import negotiate_roles, sort_role_ids
from .roles.registry import create_role

if TYPE_CHECKING:
    from aiosendspin.models.types import ServerMessage

    from .connection import SendspinConnection
    from .group import SendspinGroup
    from .server import SendspinServer


logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# Cleanup delay for reconnect-friendly disconnect reasons (seconds)
CLIENT_CLEANUP_DELAY = 180.0

# Reasons that trigger immediate client cleanup from the registry.
# Note: ANOTHER_SERVER is intentionally excluded and never auto-cleaned up.
IMMEDIATE_CLEANUP_REASONS: frozenset[GoodbyeReason] = frozenset(
    {
        GoodbyeReason.SHUTDOWN,
        GoodbyeReason.USER_REQUEST,
    }
)


class DisconnectBehaviour(Enum):
    """Enum for disconnect behaviour options."""

    UNGROUP = "ungroup"
    """
    The client will ungroup itself from its current group when it gets disconnected.

    Playback will continue on the remaining group members.
    """

    STOP = "stop"
    """
    The client will stop playback of the whole group when disconnecting.
    """


@dataclass(frozen=True, slots=True)
class ConnectionSecurity:
    """Read-only view of a live connection's security state."""

    psk_category: PskCategory
    """Category of the PSK that admitted the connection; a server-verified fact."""

    @property
    def trust_level(self) -> TrustLevel:
        """Trust this connection carries, derived from the PSK category that admitted it."""
        if self.psk_category is PskCategory.LONG_TERM:
            return TrustLevel.USER
        return TrustLevel.NONE


class SendspinClient:
    """Persistent client/device object."""

    def __init__(self, server: SendspinServer, client_id: str) -> None:
        """Create a new persistent client/device object."""
        self._server = server
        self._client_id = client_id
        self._name = client_id
        self._info: ClientHelloPayload | None = None
        self._negotiated_role_ids: list[str] = []
        self._roles: dict[str, Role] = {}
        # Cached tuple of active roles, rebuilt when the role set changes.
        self._active_roles: tuple[Role, ...] | None = None
        self._group: SendspinGroup | None = None

        self._connection: SendspinConnection | None = None
        self._connected: bool = False
        self._added_event_fired: bool = False
        self._roles_warm_disconnected: bool = False
        self._roles_cold_preinitialized: bool = False
        self._roles_attached: bool = False

        self.disconnect_behaviour = DisconnectBehaviour.UNGROUP

        self._event_cbs: list[Callable[[SendspinClient, ClientEvent], None]] = []
        self._logger = logger.getChild(client_id)

        # Client-level availability (reported by client/state). Persists across reconnects.
        self._available: bool = True

        # External-source recovery state (persists across reconnects).
        self._previous_group_id: str | None = None
        """Group ID to rejoin after external_source ends."""
        self._external_source_solo_group_id: str | None = None
        """Solo group ID created when entering external_source."""
        self._switch_lock: asyncio.Lock = asyncio.Lock()

        # Role-owned persistent state (per role family).
        self._role_state: dict[str, object] = {}

        # Cache for binary handling lookup: message_type -> (BinaryHandling, Role)
        # Built when roles are attached for cached lookup in _send_binary_frame().
        self._binary_handling_cache: dict[int, tuple[BinaryHandling, Role]] = {}

        # Reasons already logged, deduped for the client's lifetime.
        self._noncompliance_logged: set[str] = set()

        # Pending cleanup handle (scheduled via loop.call_soon/call_later on disconnect)
        self._cleanup_handle: asyncio.Handle | None = None
        # True when we intentionally retain a disconnected client after ANOTHER_SERVER goodbye.
        self._cleanup_on_mdns_removal: bool = False

    def flag_noncompliance(self, reason: str) -> None:
        """Log a tolerated spec violation once, or reject it when the server is strict."""
        if not self._server.allow_noncompliant_clients:
            self._logger.error("rejecting non-compliant client: %s", reason)
            raise ClientComplianceError(reason)
        # Recurring deviations (e.g. per client/state) would otherwise log every
        # message, so log each distinct reason only once.
        if reason in self._noncompliance_logged:
            return
        self._noncompliance_logged.add(reason)
        self._logger.warning("non-compliant client: %s", reason)

    @property
    def client_id(self) -> str:
        """Return the stable unique identifier for this device."""
        return self._client_id

    @property
    def name(self) -> str:
        """Return the human-readable device name."""
        return self._name

    @property
    def info(self) -> ClientHelloPayload:
        """Return the most recent `client/hello` payload.

        Raises RuntimeError before the first hello has been processed; use
        `info_or_none` in contexts where that state is reachable.
        """
        if self._info is None:
            raise RuntimeError("client/hello has not been processed yet")
        return self._info

    @property
    def info_or_none(self) -> ClientHelloPayload | None:
        """Return the most recent `client/hello` payload, or None before the first hello."""
        return self._info

    @property
    def negotiated_role_ids(self) -> list[str]:
        """Return the mutually-supported role set for this device (versioned role IDs)."""
        return self._negotiated_role_ids

    @property
    def active_roles(self) -> tuple[Role, ...]:
        """All active role instances for iteration."""
        if self._active_roles is None:
            self._active_roles = tuple(self._roles.values())
        return self._active_roles

    @property
    def active_role_ids(self) -> list[str]:
        """Versioned IDs of the currently-active roles."""
        return list(self._roles)

    def role(self, role_id: str) -> Role | None:
        """Get active role by versioned ID (e.g., 'player@v1')."""
        return self._roles.get(role_id)

    def roles_by_family(self, family: str) -> list[Role]:
        """Return all active roles for a role family."""
        return [role for role in self._roles.values() if role.role_family == family]

    def get_role_state(self, family: str, cls: type[_T]) -> _T | None:
        """Return persistent role state for a family, or None if unset."""
        state = self._role_state.get(family)
        if state is None:
            return None
        if not isinstance(state, cls):
            raise TypeError(
                f"Role state for {family} is {type(state).__name__}, expected {cls.__name__}"
            )
        return state

    def set_role_state(self, family: str, state: object) -> None:
        """Store persistent role state for a family."""
        self._role_state[family] = state

    def get_or_create_role_state(self, family: str, cls: type[_T]) -> _T:
        """Return persistent role state, creating a default if missing."""
        existing = self.get_role_state(family, cls)
        if existing is not None:
            return existing
        created = cls()
        self._role_state[family] = created
        return created

    @property
    def group(self) -> SendspinGroup:
        """Return the current group this client belongs to."""
        assert self._group is not None, "client group has not been initialized"
        return self._group

    @property
    def is_connected(self) -> bool:
        """Return True if this device currently has an active WebSocket connection."""
        return self._connected and self._connection is not None

    @property
    def connection(self) -> SendspinConnection | None:
        """Return the active connection for this device, if connected."""
        return self._connection

    @property
    def connection_security(self) -> ConnectionSecurity | None:
        """Security state of the active connection, or ``None`` when disconnected."""
        conn = self._connection
        if conn is None or conn.psk_category is None:
            return None
        return ConnectionSecurity(psk_category=conn.psk_category)

    @property
    def is_paired(self) -> bool:
        """Whether the active connection is authenticated by a long-term Sendspin PSK."""
        conn = self._connection
        return conn is not None and conn.psk_category is PskCategory.LONG_TERM

    @property
    def has_warm_disconnected_roles(self) -> bool:
        """Whether role instances are retained while transport is down."""
        return self._roles_warm_disconnected and self._connection is None

    @property
    def has_cold_preinitialized_roles(self) -> bool:
        """Whether role instances were preinitialized without any transport attach."""
        return self._roles_cold_preinitialized and self._connection is None

    @property
    def cleanup_on_mdns_removal(self) -> bool:
        """Whether this retained client should be removed when mDNS record disappears."""
        return self._cleanup_on_mdns_removal

    @property
    def available(self) -> bool:
        """Return whether the client is available to participate, per `client/state`."""
        return self._available

    async def handle_availability_change(self, available: bool) -> None:  # noqa: FBT001
        """Handle a client availability change by notifying all roles."""
        old_available = self._available
        self._available = available

        # Snapshot: an awaited hook can trigger a cold reconnect that rebuilds _roles.
        for role in list(self._roles.values()):
            coro = role.on_availability_changed(old_available, available)
            if coro is not None:
                await coro

        if not available:
            await self._handle_external_source_transition()

    async def _handle_external_source_transition(self) -> None:
        """Move the client out of any shared group when it switches to external_source.

        - Multi-client group: remember the previous group and move to a solo group.
        - Solo group: stop playback so the client is no longer streaming.
        """
        previous_group_id = await self.quiesce_to_solo_stopped()
        if previous_group_id is None:
            self._logger.debug("Client already in solo group, stopped playback for external_source")
            return
        self._previous_group_id = previous_group_id
        self._external_source_solo_group_id = self.group.group_id
        self._logger.debug("Stored previous group %s for external_source", previous_group_id)

    async def handle_switch_command(self) -> None:
        """Cycle this client through available groups."""
        if self._switch_lock.locked():
            self._logger.debug("Ignoring switch command; switch already in progress")
            return
        async with self._switch_lock:
            await self._handle_switch_command_locked()

    async def _handle_switch_command_locked(self) -> None:
        # Clients in external_source can't participate in playback.
        if not self._available:
            self._logger.debug("Ignoring switch command while client is in external_source state")
            return

        # External-source recovery takes priority over the normal cycle.
        if await self._try_rejoin_previous_group():
            return

        current_group = self.group
        all_groups = self._get_all_groups()
        has_player_role = has_role_family("player", self._negotiated_role_ids)
        cycle_groups = self._build_group_cycle(all_groups, current_group, has_player_role)

        if not cycle_groups:
            self._logger.debug("No groups available to switch to")
            return

        try:
            current_index = cycle_groups.index(current_group)
            next_index = (current_index + 1) % len(cycle_groups)
        except ValueError:
            next_index = 0

        next_group = cycle_groups[next_index]

        if next_group is None:
            self._logger.info("Switching client %s to solo group", self._client_id)
            await current_group.remove_client(self)
        elif next_group != current_group:
            self._logger.info(
                "Switching client %s to group %s", self._client_id, next_group.group_id
            )
            await current_group.remove_client(self)
            await next_group.add_client(self)

    def _get_all_groups(self) -> list[SendspinGroup]:
        """Return all unique groups across currently connected clients."""
        groups_seen: set[str] = set()
        unique_groups: list[SendspinGroup] = []
        for client in self._server.connected_clients:
            group = client.group
            if group.group_id not in groups_seen:
                groups_seen.add(group.group_id)
                unique_groups.append(group)
        return unique_groups

    def _build_group_cycle(
        self,
        all_groups: list[SendspinGroup],
        current_group: SendspinGroup,
        has_player_role: bool,  # noqa: FBT001
    ) -> list[SendspinGroup | None]:
        """Build the switch cycle list.

        ``None`` in the list represents "switch to a new solo group" for clients
        that hold the player role.
        """
        multi_client_playing: list[SendspinGroup] = []
        single_client: list[SendspinGroup] = []

        for group in all_groups:
            client_count = len(group.clients)
            is_playing = group.state == PlaybackStateType.PLAYING
            if client_count > 1 and is_playing:
                if any(has_role_family("player", c.negotiated_role_ids) for c in group.clients):
                    multi_client_playing.append(group)
            elif client_count == 1 and is_playing:
                single_client_obj = group.clients[0]
                if group != current_group and has_role_family(
                    "player", single_client_obj.negotiated_role_ids
                ):
                    single_client.append(group)

        multi_client_playing.sort(key=lambda g: g.group_id)
        single_client.sort(key=lambda g: g.group_id)

        if has_player_role:
            current_is_solo = len(current_group.clients) == 1
            solo_option: list[SendspinGroup | None] = [current_group] if current_is_solo else [None]
            return multi_client_playing + single_client + solo_option
        return [*multi_client_playing, *single_client]

    def _should_rejoin_previous_group(self) -> bool:
        """Return True when switch should rejoin the pre-external-source group.

        Per spec: if the client is still in the solo group created by its
        ``external_source`` transition, switch prioritizes rejoining that group.
        """
        return (
            self._previous_group_id is not None
            and self._available
            and self._external_source_solo_group_id == self.group.group_id
            and len(self.group.clients) == 1
        )

    async def _try_rejoin_previous_group(self) -> bool:
        if not self._should_rejoin_previous_group():
            return False

        previous_group_id = self._previous_group_id
        # Clear external_source tracking after attempt, regardless of outcome.
        self._previous_group_id = None
        self._external_source_solo_group_id = None

        previous_group = self._find_group_by_id(previous_group_id)
        if previous_group is not None and previous_group != self.group:
            self._logger.info(
                "Rejoining previous group %s after external_source", previous_group_id
            )
            await self.group.remove_client(self)
            await previous_group.add_client(self)
            return True

        self._logger.debug(
            "Previous group %s no longer exists or is current group, "
            "falling back to normal switch cycle",
            previous_group_id,
        )
        return False

    def _find_group_by_id(self, group_id: str | None) -> SendspinGroup | None:
        if group_id is None:
            return None
        for client in self._server.connected_clients:
            if client.group.group_id == group_id:
                return client.group
        return None

    def check_role(self, role: Roles) -> bool:
        """Check if the client has a role active (by role family)."""
        return has_role(role.value, self._negotiated_role_ids)

    def attach_connection(
        self,
        connection: SendspinConnection,
        *,
        client_info: ClientHelloPayload,
        negotiated_roles: list[str],
        active_roles: list[str],
    ) -> None:
        """Attach a new WebSocket connection to this client."""
        # Cancel pending cleanup if client reconnected before cleanup fired
        self._cancel_cleanup()

        if self._connection is not None and self._connection is not connection:
            # Replace an existing connection for the same device.
            self._logger.debug("Replacing existing connection for %s", self._client_id)
            create_task(self._connection.disconnect(retry_connection=False))
            # The replaced socket's disconnect() is guarded out of detach_connection, so
            # retire its still-connected roles here to reuse them on this attach.
            if (
                self._roles
                and not self._roles_warm_disconnected
                and not self._roles_cold_preinitialized
            ):
                self._retire_roles_warm()

        self._connection = connection
        self._connected = False  # set True once initial state is received (spec)
        self._cleanup_on_mdns_removal = False
        on_transport_attached = getattr(self._server, "on_client_transport_attached", None)
        if callable(on_transport_attached):
            on_transport_attached(self._client_id)

        previous_info = self._info
        self._set_identity_from_hello(client_info, negotiated_roles=negotiated_roles)
        self._server._signal_client_connected(self._client_id)  # noqa: SLF001
        if previous_info is not None and previous_info != client_info:
            self._server._signal_client_updated(self._client_id)  # noqa: SLF001
        self._logger = logger.getChild(self._client_id)
        if not self._roles_warm_disconnected and not self._roles_cold_preinitialized:
            # First attach for this device: clear stale state, then reconcile from empty.
            self._hard_detach_roles()
        self._connect_active_roles(active_roles)

        self._roles_warm_disconnected = False
        self._roles_cold_preinitialized = False
        self._roles_attached = True

        self._rebuild_binary_handling_cache()

        # Ensure group exists (server creates it on first sight).
        if self._group is None:
            raise RuntimeError("SendspinClient.group must be initialized by the server")

        # Re-register client events now that roles are negotiated.
        # The initial _set_group() call during get_or_create_client() runs before
        # attach_connection(), so negotiated_roles is empty at that point and
        # cross-role hooks (e.g. ControllerGroupRole subscribing to player volume
        # events) are skipped. Re-registering here ensures they are set up.
        # Guard: individual on_client_added() implementations are idempotent.
        self._group._register_client_events(self)  # noqa: SLF001

    def _connect_active_roles(self, active_role_ids: list[str]) -> None:
        """Connect the active role set, reusing existing instances and creating the rest."""
        # A dropped role needs no on_disconnect: warm ran it at detach, cold never
        # connected. Exception: cold roles attached without a transport
        # (attach_preinitialized_roles) are live — reused instances skip the second
        # on_connect and dropped ones get their disconnect hook below.
        active_role_ids = sort_role_ids(active_role_ids)
        roles: dict[str, Role] = {}
        for role_id in active_role_ids:
            role = self._roles.pop(role_id, None)
            if role is None:
                role = create_role(role_id, self)
                if role is None:
                    continue
                role.on_connect()
            elif not self._roles_attached:
                role.on_connect()
            roles[role.role_id] = role
        if self._roles_attached:
            for dropped_role in reversed(self._roles.values()):
                dropped_role.on_disconnect()
        self._roles = roles

    def set_active_roles(self, active_role_ids: list[str]) -> None:
        """Reconcile active role instances to match a server/activate declaration."""
        active_role_ids = sort_role_ids(active_role_ids)
        desired_set = set(active_role_ids)
        if desired_set == set(self._roles):
            # Same set: no hooks to run, but keep dict order canonical.
            if list(self._roles) != active_role_ids:
                self._roles = {role_id: self._roles[role_id] for role_id in active_role_ids}
                self._rebuild_binary_handling_cache()
            return

        # Tear down in reverse attach order: the controller unwinds before the player it reads.
        for role_id in reversed(list(self._roles)):
            if role_id not in desired_set:
                deactivated_role = self._roles.pop(role_id)
                deactivated_role.on_deactivate()
                self.group.on_role_deactivated(deactivated_role)

        roles: dict[str, Role] = {}
        for role_id in active_role_ids:
            role = self._roles.get(role_id)
            if role is None:
                role = create_role(role_id, self)
                if role is None:
                    continue
                role.on_connect()
                self.group.on_role_activated(role)
            roles[role.role_id] = role
        self._roles = roles

        self._rebuild_binary_handling_cache()

    def refresh_identity_from_hello(
        self, client_info: ClientHelloPayload, *, negotiated_roles: list[str]
    ) -> None:
        """Apply a hello re-sent over the existing connection (post re-handshake)."""
        previous_info = self._info
        self._set_identity_from_hello(client_info, negotiated_roles=negotiated_roles)
        if previous_info is not None and previous_info != client_info:
            self._server._signal_client_updated(self._client_id)  # noqa: SLF001

    def preload_hello(self, client_info: ClientHelloPayload) -> None:
        """Seed persistent client identity/capabilities without an active connection."""
        self._set_identity_from_hello(client_info)

    def preinitialize_client_from_hello(self, client_info: ClientHelloPayload) -> None:
        """Preinitialize almost all client state from hello while disconnected.

        This prepares a not-yet-connected client with negotiated identity and
        role setup so PushStream logic can reason about role capabilities before
        transport attach.

        Initializes:
        - persisted client identity/capabilities from hello
        - negotiated role list
        - role instances for negotiated role IDs
        - binary handling cache derived from those roles
        - cold-preinitialized role marker

        Deferred until websocket attach (or attach_preinitialized_roles()):
        - role on_connect() lifecycle hooks
        """
        if self._connection is not None:
            raise RuntimeError(
                f"Cannot cold-preinitialize roles for {self._client_id!r} while connected"
            )

        # Disarm a pending cleanup so a just-registered client isn't removed.
        self._cancel_cleanup()
        self._hard_detach_roles(call_disconnect_hooks=self._roles_attached)
        self._set_identity_from_hello(client_info)
        self._roles_warm_disconnected = False

        for role_id in self._negotiated_role_ids:
            role = create_role(role_id, self)
            if role is None:
                continue
            self._roles[role.role_id] = role

        self._roles_cold_preinitialized = True
        self._rebuild_binary_handling_cache()

    def attach_preinitialized_roles(self) -> None:
        """Run connect hooks for cold-preinitialized roles without a transport."""
        if not self._roles_cold_preinitialized or self._roles_attached:
            return
        for role in self._roles.values():
            role.on_connect()
        self._roles_attached = True

    def mark_connected(self) -> None:
        """Mark this client as fully connected (after initial client/state if required)."""
        if self._connection is None:
            return
        self._connected = True
        self.group.on_client_connected(self)

    def detach_connection(self, goodbye_reason: GoodbyeReason | None) -> None:
        """Detach the current connection and apply BufferTracker reset policy."""
        self._connected = False

        warm_disconnect = goodbye_reason in {None, GoodbyeReason.RESTART}
        if warm_disconnect:
            self._retire_roles_warm()
        else:
            self._hard_detach_roles()
            self._roles_warm_disconnected = False

        self._connection = None

        self._server._signal_client_disconnected(  # noqa: SLF001
            self._client_id, goodbye_reason
        )

        if goodbye_reason == GoodbyeReason.ANOTHER_SERVER:
            create_task(self._handle_takeover_disconnect())

        # Schedule client cleanup from registry
        self._schedule_cleanup(goodbye_reason)

    def _retire_roles_warm(self) -> None:
        """Run role disconnect hooks but keep instances alive for warm reuse on reconnect."""
        for role in reversed(self._roles.values()):
            role.on_disconnect()
        self._binary_handling_cache.clear()
        self._roles_warm_disconnected = True
        self._roles_cold_preinitialized = False
        self._roles_attached = False

    async def _handle_takeover_disconnect(self) -> None:
        """Handle ANOTHER_SERVER disconnect by ungrouping the client from its group."""
        old_group_id = self.group.group_id
        try:
            # ungroup() already leaves the client in a fresh stopped solo group.
            await self.ungroup()
        except Exception:
            self._logger.exception(
                "Takeover disconnect sequence failed for %s (old_group=%s)",
                self._client_id,
                old_group_id,
            )

    def _cancel_cleanup(self) -> None:
        """Disarm a pending registry-cleanup timer, if one is armed."""
        if self._cleanup_handle is not None:
            self._logger.debug("Cancelling pending cleanup")
            self._cleanup_handle.cancel()
            self._cleanup_handle = None

    def _schedule_cleanup(self, goodbye_reason: GoodbyeReason | None) -> None:
        """Schedule cleanup from server registry based on disconnect reason."""
        if self._server.is_external_player(self._client_id):
            self._logger.debug("Skipping cleanup scheduling for external player")
            self._cleanup_on_mdns_removal = False
            return

        if goodbye_reason == GoodbyeReason.ANOTHER_SERVER:
            self._logger.debug("Skipping cleanup scheduling (reason: %s)", goodbye_reason)
            self._cleanup_on_mdns_removal = True
            return

        self._cleanup_on_mdns_removal = False
        if goodbye_reason in IMMEDIATE_CLEANUP_REASONS:
            # Immediate cleanup for explicit disconnect reasons
            self._logger.debug("Scheduling immediate cleanup (reason: %s)", goodbye_reason)
            self._cleanup_handle = self._server.loop.call_soon(self._do_cleanup)
        else:
            # Delayed cleanup for reconnect-friendly scenarios
            self._logger.debug(
                "Scheduling delayed cleanup in %.0fs (reason: %s)",
                CLIENT_CLEANUP_DELAY,
                goodbye_reason,
            )
            self._cleanup_handle = self._server.loop.call_later(
                CLIENT_CLEANUP_DELAY, self._do_cleanup
            )

    def _do_cleanup(self) -> None:
        """Remove this client from the server registry."""
        self._cleanup_handle = None
        if self._connected:
            # Client reconnected, don't clean up
            return
        self._hard_detach_roles()
        self._roles_warm_disconnected = False
        self._logger.debug("Cleaning up client from registry")
        create_task(self._server.remove_client(self._client_id))

    def _hard_detach_roles(self, *, call_disconnect_hooks: bool = True) -> None:
        """Run role disconnect hooks and clear role-related caches."""
        if call_disconnect_hooks:
            for role in reversed(self._roles.values()):
                role.on_disconnect()
        self._roles.clear()
        self._active_roles = None
        self._binary_handling_cache.clear()
        self._roles_cold_preinitialized = False
        self._roles_attached = False

    def _set_identity_from_hello(
        self,
        client_info: ClientHelloPayload,
        *,
        negotiated_roles: list[str] | None = None,
    ) -> None:
        """Store hello identity/capabilities with optional explicit negotiated roles."""
        self._info = client_info
        # A group with no name of its own reports its founding member's, so this device
        # learning its own name can change what its whole group is called.
        group = self._group
        previous_group_name = group.group_name if group is not None else None
        self._name = client_info.name
        if group is not None and previous_group_name is not None:
            group._publish_if_name_changed(previous_group_name)  # noqa: SLF001
        if negotiated_roles is None:
            self._negotiated_role_ids = negotiate_roles(
                client_info.supported_roles, strict=not self._server.allow_noncompliant_clients
            )
        else:
            self._negotiated_role_ids = negotiated_roles

    def _rebuild_binary_handling_cache(self) -> None:
        """Build binary handling cache for fast lookup."""
        self._active_roles = None
        self._binary_handling_cache.clear()
        for msg_type in BinaryMessageType:
            for role in self._roles.values():
                handling = role.get_binary_handling(msg_type.value)
                if handling is not None:
                    self._binary_handling_cache[msg_type.value] = (handling, role)
                    break  # First role that handles it wins

    # ---- Messaging (delegates to connection) ----

    def send_message(self, message: ServerMessage) -> None:
        """Send a message if connected; otherwise no-op."""
        if self._connection is None:
            return
        if isinstance(message, StreamStartMessage):
            self._logger.debug("Sending stream/start: %s", message.payload)
        self._connection.send_message(message)

    def send_role_message(self, role: str, message: ServerMessage) -> None:
        """Send a role-scoped message if connected; otherwise no-op."""
        if self._connection is None:
            return
        if isinstance(message, StreamStartMessage):
            self._logger.debug("Sending stream/start: %s", message.payload)
        self._connection.send_role_message(role, message)

    def send_binary(
        self,
        data: bytes,
        *,
        role_family: str,
        timestamp_us: int,
        message_type: int,
        buffer_end_time_us: int | None = None,
        buffer_byte_count: int | None = None,
        duration_us: int | None = None,
    ) -> None:
        """Enqueue a binary payload for this client, or no-op when disconnected."""
        if self._connection is None:
            return
        self._connection.send_binary(
            data,
            role=role_family,
            timestamp_us=timestamp_us,
            message_type=message_type,
            buffer_end_time_us=buffer_end_time_us,
            buffer_byte_count=buffer_byte_count,
            duration_us=duration_us,
        )

    def drop_pending_binary(self, roles: list[str] | None) -> None:
        """Drop queued binary payloads for the given role families, if connected."""
        if self._connection is None:
            return
        self._connection.drop_pending_binary(roles)

    def get_binary_handling_cached(self, message_type: int) -> tuple[BinaryHandling, Role] | None:
        """Return cached binary handling for a message type."""
        return self._binary_handling_cache.get(message_type)

    # ---- Events + grouping ----

    def add_event_listener(
        self, callback: Callable[[SendspinClient, ClientEvent], None]
    ) -> Callable[[], None]:
        """Register a callback for client-scoped events.

        The second callback argument is ``ClientEvent`` to cover both client-core
        events and role-emitted client events.

        Returns an unsubscribe callable.
        """
        self._event_cbs.append(callback)

        def _remove() -> None:
            with suppress(ValueError):
                self._event_cbs.remove(callback)

        return _remove

    def _signal_event(self, event: ClientEvent) -> None:
        for cb in self._event_cbs:
            try:
                cb(self, event)
            except Exception:
                logger.exception("Error in event listener")

    def _set_group(self, group: SendspinGroup) -> None:
        """Set the group for this client. For internal use by SendspinGroup only."""
        if self._group is not None:
            self._group._unregister_client_events(self)  # noqa: SLF001
        self._group = group
        self._group._register_client_events(self)  # noqa: SLF001
        self._signal_event(ClientGroupChangedEvent(group))
        for role in self._roles.values():
            role.on_group_changed(group)

    async def ungroup(self) -> None:
        """Remove the client from its current group, placing it in a fresh solo group."""
        if self._group is None:
            return
        await self._group.remove_client(self)

    async def quiesce_to_solo_stopped(self) -> str | None:
        """Leave any multi-client group for a solo stopped group, ending active streams."""
        group = self.group
        if len(group.clients) > 1:
            previous_group_id = group.group_id
            await group.remove_client(self)
            return previous_group_id
        had_active_stream = group.has_active_stream
        await group.stop()
        if not had_active_stream:
            # A group can be PLAYING without a stream (track transition); stop() fires
            # on_stream_end only via the PushStream, so end the streams explicitly here.
            for role in self.active_roles:
                role.on_stream_end()
        return None
