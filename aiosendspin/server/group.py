"""Manages and synchronizes playback for a group of one or more clients."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable, Iterable
from contextlib import suppress
from typing import TYPE_CHECKING
from uuid import UUID

from aiosendspin.models.core import (
    GroupUpdateServerMessage,
    GroupUpdateServerPayload,
)
from aiosendspin.models.types import PlaybackStateType, has_role_family
from aiosendspin.server.events import (
    GroupDeletedEvent,
    GroupEvent,
    GroupMemberAddedEvent,
    GroupMemberRemovedEvent,
    GroupStateChangedEvent,
)
from aiosendspin.server.roles import GroupRole, MetadataGroupRole
from aiosendspin.server.roles.registry import create_group_roles

from .audio_transformers import TransformerPool
from .channels import ChannelResolver, default_channel_resolver
from .push_stream import PushStream

if TYPE_CHECKING:
    from .client import SendspinClient
    from .roles import Role
    from .server import SendspinServer

logger = logging.getLogger(__name__)


class SendspinGroup:
    """
    A group of one or more clients for synchronized playback.

    Handles synchronized audio streaming across multiple clients with automatic
    format conversion and buffer management. Every client is always assigned to
    a group to simplify grouping requests.
    """

    _clients: list[SendspinClient]
    """List of all clients in this group."""
    _server: SendspinServer
    """Reference to the SendspinServer instance."""
    _event_cbs: list[Callable[[SendspinGroup, GroupEvent], None]]
    """List of event callbacks for this group."""
    _current_state: PlaybackStateType = PlaybackStateType.STOPPED
    """Current playback state of the group."""
    _group_id: str
    """Unique identifier for this group."""
    _group_name: str | None
    """Friendly name for this group."""
    _play_start_time_us: int | None
    """Absolute timestamp in microseconds when playback started, None when not streaming."""
    _playback_lock: asyncio.Lock
    """Lock to serialize play_media() and stop() operations, preventing race conditions."""
    _push_stream: PushStream | None
    """Current PushStream for push-based streaming, None when not active."""
    _transformer_pool: TransformerPool
    """Pool for shared transformer instances (encoders, etc.) across roles."""
    _group_roles: dict[str, GroupRole]
    """Registry of GroupRole instances, keyed by role family."""
    _channel_resolver: ChannelResolver
    """Callback to determine which channel a player should receive audio from."""

    def __init__(self, server: SendspinServer, *args: SendspinClient) -> None:
        """
        DO NOT CALL THIS CONSTRUCTOR. INTERNAL USE ONLY.

        Groups are managed automatically by the server.

        Initialize a new SendspinGroup.

        Args:
            server: The SendspinServer instance this group belongs to.
            *args: Clients to add to this group.
        """
        self._clients = list(args)
        assert len(self._clients) > 0, "A group must have at least one client"
        self._server = server
        self._event_cbs = []
        self._group_id = str(uuid.uuid4())
        self._group_name: str | None = None
        self._play_start_time_us: int | None = None
        self._playback_lock = asyncio.Lock()
        self._push_stream: PushStream | None = None
        self._transformer_pool = TransformerPool()
        self._group_roles = create_group_roles(self)
        self._channel_resolver = default_channel_resolver

        # Set group reference for initial clients
        for client in self._clients:
            client._set_group(self)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

        logger.debug(
            "SendspinGroup initialized with %d client(s): %s",
            len(self._clients),
            [type(c).__name__ for c in self._clients],
        )

    def start_stream(
        self,
        *,
        channel_resolver: ChannelResolver | None = None,
    ) -> PushStream:
        """
        Create a new PushStream for push-based audio streaming.

        Args:
            channel_resolver: Optional callback to determine which channel a player
                should receive audio from. If not provided, all players receive
                audio from MAIN_CHANNEL.

        Returns:
            A new PushStream instance configured for this group.
        """
        if channel_resolver is not None:
            self._channel_resolver = channel_resolver
        else:
            self._channel_resolver = default_channel_resolver

        # Replace any existing active stream so stale handles cannot continue
        # committing audio after a new stream is started.
        if self._push_stream is not None and not self._push_stream.is_stopped:
            if self._server.allow_noncompliant_clients:
                self._push_stream.stop()
            else:
                # Clear buffered audio while preserving the protocol stream across replacement.
                # Gate the spec-compliant path while legacy clients still mishandle stream/clear.
                self._push_stream.clear()
                self._push_stream.stop(keep_stream=True)

        self._push_stream = PushStream(
            loop=self._server.loop,
            clock=self._server.clock,
            group=self,
        )

        # Reclaim any disconnected clients in the group (multi-server support).
        # This reconnects to clients that may have switched to another server.
        for client in self._clients:
            if not client.is_connected:
                self._server.request_client_playback_connection(client.client_id)

        self._set_playback_state(PlaybackStateType.PLAYING)
        return self._push_stream

    def stop_stream(self) -> None:
        """
        Stop only the current PushStream transport.

        This preserves group playback state as PLAYING. Use this when you are
        about to immediately start another stream and want clients to stay in a
        logical PLAYING state during the transition.

        To stop transport and also mark the group state as STOPPED, call stop().

        Does nothing if no stream is active.
        """
        if self._push_stream is not None:
            self._push_stream.stop()
            self._push_stream = None

    def on_role_format_changed(self, role: Role) -> None:
        """Notify PushStream that a role's audio format changed mid-stream."""
        if self._push_stream is not None and not self._push_stream.is_stopped:
            self._push_stream.on_role_format_changed(role)

    def on_role_activated(self, role: Role) -> None:
        """Join a role activated mid-connection to the active stream, if any."""
        if (
            self._push_stream is not None
            and not self._push_stream.is_stopped
            and role.get_audio_requirements() is not None
        ):
            self._push_stream.on_role_join(role)

    def on_role_deactivated(self, role: Role) -> None:
        """Drop a role deactivated mid-connection from the active stream, if any."""
        if self._push_stream is not None and not self._push_stream.is_stopped:
            self._push_stream.on_role_leave(role)

    def _group_update_message(self) -> GroupUpdateServerMessage:
        """Build a group/update carrying this group's current state."""
        return GroupUpdateServerMessage(
            GroupUpdateServerPayload(
                playback_state=self._current_state,
                group_id=self.group_id,
                group_name=self.group_name,
            )
        )

    def _send_group_update(self, client: SendspinClient, message: GroupUpdateServerMessage) -> None:
        """Send one group/update to a client that has finished coming up.

        A client still mid-bring-up is told the group's state by ``on_client_connected``
        once its first client/state lands, so anything sent before then is superseded.
        """
        if client.is_connected:
            client.send_message(message)

    def _send_group_update_to_clients(self) -> None:
        """Send group/update to every member that has finished coming up."""
        group_message = self._group_update_message()
        for client in self._clients:
            self._send_group_update(client, group_message)

    def _publish_if_name_changed(self, previous: str) -> None:
        """Publish group/update when the name the group reports is no longer ``previous``."""
        if self.group_name != previous:
            self._send_group_update_to_clients()

    def on_client_connected(self, client: SendspinClient) -> None:
        """Send current group state to a client that just finished handshaking."""
        if client not in self._clients:
            return

        client.send_message(self._group_update_message())

        if self._push_stream is not None and not self._push_stream.is_stopped:
            for role in client.active_roles:
                if role.get_audio_requirements() is not None:
                    self._push_stream.on_role_join(role)

    def _set_playback_state(self, new_state: PlaybackStateType) -> None:
        """Set playback state and notify listeners/clients when it changes."""
        if self._current_state == new_state:
            return

        self._current_state = new_state
        self._signal_event(GroupStateChangedEvent(new_state))
        self._send_group_update_to_clients()

    async def stop(self) -> bool:
        """
        Stop playback for the group and clean up resources.

        This stops any active PushStream and marks the group playback state as
        STOPPED.

        Returns:
            bool: True if an active stream was stopped,
            False if no stream was active and no cleanup was required.
        """
        if len(self._clients) == 0:
            # An empty group cannot have active playback
            return False

        async with self._playback_lock:
            active = self._push_stream is not None and not self._push_stream.is_stopped
            needs_cleanup = self._current_state != PlaybackStateType.STOPPED

            if not active and not needs_cleanup:
                return False

            logger.debug(
                "Stopping playback for group with clients: %s",
                [c.client_id for c in self._clients],
            )

            metadata_group_role = self.group_role("metadata")
            if isinstance(metadata_group_role, MetadataGroupRole):
                metadata_group_role.freeze_progress()

            # Stop the push stream if active
            if self._push_stream is not None:
                self._push_stream.stop()
                self._push_stream = None

            self._set_playback_state(PlaybackStateType.STOPPED)
            return True

    @property
    def clients(self) -> list[SendspinClient]:
        """All clients that are part of this group."""
        return self._clients

    @property
    def has_active_stream(self) -> bool:
        """Check if there is an active stream running."""
        return self._push_stream is not None and not self._push_stream.is_stopped

    @property
    def transformer_pool(self) -> TransformerPool:
        """Return the transformer pool for encoder deduplication."""
        return self._transformer_pool

    def get_channel_for_player(self, player_id: str) -> UUID:
        """
        Get the channel a player should receive audio from.

        Args:
            player_id: The player's client_id.

        Returns:
            The channel UUID for this player.
        """
        return self._channel_resolver(player_id)

    def group_role(self, family: str) -> GroupRole | None:
        """Get the GroupRole for a role family."""
        return self._group_roles.get(family)

    def add_event_listener(
        self, callback: Callable[[SendspinGroup, GroupEvent], None]
    ) -> Callable[[], None]:
        """
        Register a callback to listen for state changes of this group.

        State changes include:
        - The group started playing
        - The group stopped/finished playing

        Returns a function to remove the listener.
        """
        self._event_cbs.append(callback)

        def _remove() -> None:
            with suppress(ValueError):
                self._event_cbs.remove(callback)

        return _remove

    def _signal_event(self, event: GroupEvent) -> None:
        for cb in self._event_cbs:
            try:
                cb(self, event)
            except Exception:
                logger.exception("Error in event listener")

    def _register_client_events(self, client: SendspinClient) -> None:
        """Notify GroupRoles that a client was added."""
        for group_role in self._group_roles.values():
            group_role.on_client_added(client)

    def _unregister_client_events(self, client: SendspinClient) -> None:
        """Notify GroupRoles that a client was removed."""
        for group_role in self._group_roles.values():
            group_role.on_client_removed(client)

    @property
    def group_id(self) -> str:
        """Unique identifier for this group."""
        return self._group_id

    @property
    def group_name(self) -> str:
        """Friendly name for this group, its founding member's device name by default."""
        if self._group_name is not None:
            return self._group_name
        return self._clients[0].name if self._clients else ""

    def set_group_name(self, name: str | None) -> None:
        """Name this group, publishing the change; ``None`` restores the default."""
        previous = self.group_name
        self._group_name = name
        self._publish_if_name_changed(previous)

    @property
    def state(self) -> PlaybackStateType:
        """Current playback state of the group."""
        return self._current_state

    async def remove_client(self, client: SendspinClient) -> None:
        """
        Remove a client from this group.

        If a stream is active, the client receives a stream end message.
        The client is automatically moved to its own new group since every
        client must belong to a group.
        If the client is not part of this group, this will have no effect.

        Args:
            client: The client to remove from this group.
        """
        if client not in self._clients:
            return

        # Cancel any pending delayed join for this client
        logger.debug("removing %s from group with members: %s", client.client_id, self._clients)
        previous_name = self.group_name
        if len(self._clients) == 1:
            # Delete this group if that was the last client
            await self._stop_and_invalidate_stale_binary([client])
            self._clients = []
        else:
            self._clients.remove(client)
            # End the stream for the removed client via role hooks
            for role in client.active_roles:
                role.on_stream_end()
                if self._push_stream is not None and not self._push_stream.is_stopped:
                    self._push_stream.on_role_leave(role)
        if not self._clients:
            self._finalize_empty_group()
        else:
            # Stop a remnant with no player-role client left to source audio.
            if not any(has_role_family("player", c.negotiated_role_ids) for c in self._clients):
                await self._stop_and_invalidate_stale_binary(self._clients)
            # Emit event for client removal
            self._signal_event(GroupMemberRemovedEvent(client.client_id))
            # Losing the founding member changes the name the survivors report.
            self._publish_if_name_changed(previous_name)
        # Each client needs to be in a group, add it to a new one
        new_group = SendspinGroup(self._server, client)
        # Send group update to notify client of their new solo group
        new_group.on_client_connected(client)

    async def _stop_and_invalidate_stale_binary(self, clients: Iterable[SendspinClient]) -> None:
        """Stop the group, directly signaling roles when mid-track-transition.

        A group can be PLAYING without a PushStream during a track transition.
        stop() only emits stream/end through an active PushStream, so without one
        the given clients' roles are signaled directly to drop stale binary. A
        STOPPED group has nothing buffered, so no stream/end is sent (sending one
        would tear a client down right before it rejoins another group's stream).
        """
        was_playing = self._current_state != PlaybackStateType.STOPPED
        had_active_stream = self.has_active_stream
        await self.stop()
        if was_playing and not had_active_stream:
            for client in clients:
                for role in client.active_roles:
                    role.on_stream_end()

    def _finalize_empty_group(self) -> None:
        """Tear down a group with no remaining clients."""
        self._signal_event(GroupDeletedEvent())

    async def add_client(self, client: SendspinClient) -> None:
        """
        Add a client to this group.

        The client is first removed from any existing group. If a session is
        currently active, players are immediately joined to the session with
        an appropriate audio format.

        Args:
            client: The client to add to this group.
        """
        logger.debug("adding %s to group with members: %s", client.client_id, self._clients)
        if client in self._clients:
            return
        # Remove from the current group first. A remnant with a player keeps playing.
        await client.ungroup()

        # Check for and remove any stale client with the same client_id
        # This handles the case where a client disconnects and reconnects
        # while still being listed in _clients (e.g., solo client disconnect)
        stale_client = next((c for c in self._clients if c.client_id == client.client_id), None)
        if stale_client is not None:
            # Defensive fallback: normal server flow keeps one persistent client object
            # per client_id, but if a duplicate object appears, replace the stale one so
            # membership and role subscriptions stay coherent.
            logger.debug(
                "Removing stale client %s (object %s) before adding new client (object %s)",
                stale_client.client_id,
                id(stale_client),
                id(client),
            )
            self._clients.remove(stale_client)
            self._unregister_client_events(stale_client)

        # Add client to this group's client list
        self._clients.append(client)

        # Emit event for client addition
        self._signal_event(GroupMemberAddedEvent(client.client_id))

        # Then set the group (which will emit ClientGroupChangedEvent)
        client._set_group(self)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

        # Handle player joining/reconnecting with active PushStream
        if self._push_stream is not None and not self._push_stream.is_stopped:
            if not client.is_connected:
                # Defensive fallback for programmatic moves of retained/disconnected clients:
                # when joining an active group, try to reclaim the client for playback.
                self._server.request_client_playback_connection(client.client_id)
                # Disconnected roles that explicitly opt into preconnect audio must also
                # run through late-join catch-up on group add.
                for role in client.active_roles:
                    if role.get_audio_requirements() is None:
                        continue
                    if role.supports_preconnect_audio():
                        self._push_stream.on_role_join(role)
            else:
                # Call on_role_join for all roles with audio requirements (hook-based flow)
                for role in client.active_roles:
                    if role.get_audio_requirements() is not None:
                        self._push_stream.on_role_join(role)

        # Send current state to the new client
        logger.debug("Sending group update to new client %s", client.client_id)
        self._send_group_update(client, self._group_update_message())
