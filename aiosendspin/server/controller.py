"""Helpers for clients supporting the controller role."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiosendspin.models.controller import ControllerCommandPayload
from aiosendspin.models.types import ClientStateType, MediaCommand, PlaybackStateType, Roles

if TYPE_CHECKING:
    from .client import SendspinClient
    from .group import SendspinGroup
    from .server import SendspinServer


class ControllerClient:
    """Encapsulates controller role behaviour for a client."""

    def __init__(self, client: SendspinClient) -> None:
        """Attach to a client that exposes controller capabilities."""
        self.client = client
        self._logger = client._logger.getChild("controller")  # noqa: SLF001

    async def handle_command(self, payload: ControllerCommandPayload) -> None:
        """Handle controller commands."""
        # Get supported commands from the group
        supported_commands = self.client.group._get_supported_commands()  # noqa: SLF001

        # Validate command is supported
        if payload.command not in supported_commands:
            self._logger.warning(
                "Client %s sent unsupported command '%s'. Supported commands: %s",
                self.client.client_id,
                payload.command.value,
                [cmd.value for cmd in supported_commands],
            )
            # Silently ignore unsupported commands (spec doesn't define error responses)
            return

        if payload.command == MediaCommand.SWITCH:
            await self._handle_switch()
        else:
            # Forward other commands to the group
            self.client.group._handle_group_command(payload)  # noqa: SLF001

    async def _handle_switch(self) -> None:
        """Handle the switch command to cycle through groups."""
        # Clients in external_source can't participate in playback; don't allow switching groups
        # until they report a normal operational state again.
        if self.client._client_state == ClientStateType.EXTERNAL_SOURCE:  # noqa: SLF001
            self._logger.warning("Ignoring switch command while client is in external_source state")
            return

        # Check if client should rejoin previous group (external_source recovery priority)
        if await self._try_rejoin_previous_group():
            return

        server = self.client._server  # noqa: SLF001
        current_group = self.client.group

        # Get all unique groups from all connected clients
        all_groups = self._get_all_groups(server)

        # Build the cycle list based on client's player role
        has_player_role = self.client.player is not None
        cycle_groups = self._build_group_cycle(all_groups, current_group, has_player_role)

        if not cycle_groups:
            self._logger.debug("No groups available to switch to")
            return

        # Find current position in cycle and move to next
        try:
            current_index = cycle_groups.index(current_group)
            next_index = (current_index + 1) % len(cycle_groups)
        except ValueError:
            # Current group not in cycle, start from beginning
            next_index = 0

        next_group = cycle_groups[next_index]

        # Move client to the next group
        if next_group is None:
            # The group.remove_client will create a new solo group for the client
            self._logger.info(
                "Switching client %s to solo group",
                self.client.client_id,
            )
            await current_group.remove_client(self.client)
        elif next_group != current_group:
            self._logger.info(
                "Switching client %s to group %s",
                self.client.client_id,
                next_group.group_id,
            )
            await current_group.remove_client(self.client)
            await next_group.add_client(self.client)

    def _get_all_groups(self, server: SendspinServer) -> list[SendspinGroup]:
        """Get all unique groups from all connected clients."""
        groups_seen: set[str] = set()
        unique_groups: list[SendspinGroup] = []

        for client in server._clients:  # noqa: SLF001
            group = client.group
            group_id = group.group_id
            if group_id not in groups_seen:
                groups_seen.add(group_id)
                unique_groups.append(group)

        return unique_groups

    def _build_group_cycle(
        self,
        all_groups: list[SendspinGroup],
        current_group: SendspinGroup,
        has_player_role: bool,  # noqa: FBT001
    ) -> list[SendspinGroup | None]:
        """
        Build the cycle of groups based on the spec.

        Returns a list of groups to cycle through. For player clients, the list
        may contain None indicating to "go to a new solo group".
        """
        # Separate groups into categories
        multi_client_playing: list[SendspinGroup] = []
        single_client: list[SendspinGroup] = []

        for group in all_groups:
            client_count = len(group.clients)
            is_playing = group.state == PlaybackStateType.PLAYING

            if client_count > 1 and is_playing:
                # Verify the group has at least one player
                # (groups with only controllers/metadata can't actually be "playing")
                has_player = any(c.check_role(Roles.PLAYER) for c in group.clients)
                if has_player:
                    multi_client_playing.append(group)
            elif client_count == 1 and is_playing:
                # Get the single client in this group
                single_client_obj = group.clients[0]
                # Skip current group, it will be handled as solo option for player clients
                if group != current_group and single_client_obj.check_role(Roles.PLAYER):
                    # Only include single-client groups where the client has player role
                    single_client.append(group)

        # Sort for stable ordering (by group ID)
        multi_client_playing.sort(key=lambda g: g.group_id)
        single_client.sort(key=lambda g: g.group_id)

        # Build cycle based on client's player role
        if has_player_role:
            # With player role: multi-client playing -> single-client -> own solo
            current_is_solo = len(current_group.clients) == 1
            # Use current group if solo, otherwise switch to new solo group (None)
            solo_option: list[SendspinGroup | None] = [current_group] if current_is_solo else [None]
            return multi_client_playing + single_client + solo_option
        # Without player role: multi-client playing -> single-client (no own solo)
        return [*multi_client_playing, *single_client]

    def _should_rejoin_previous_group(self) -> bool:
        """
        Check if client should rejoin previous group (external_source recovery).

        Per spec: "If the client is still in the solo group from its 'external_source'
        transition, the switch command prioritizes rejoining the previous group."
        """
        return (
            self.client._previous_group_id is not None  # noqa: SLF001
            and self.client._client_state != ClientStateType.EXTERNAL_SOURCE  # noqa: SLF001
            and self.client._external_source_solo_group_id == self.client.group.group_id  # noqa: SLF001
            and len(self.client.group.clients) == 1  # Still in the solo group
        )

    async def _try_rejoin_previous_group(self) -> bool:
        """Try to rejoin the previous group after external_source ended."""
        if not self._should_rejoin_previous_group():
            return False

        previous_group_id = self.client._previous_group_id  # noqa: SLF001
        # Clear external_source tracking after attempt (regardless of success)
        self.client._previous_group_id = None  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        self.client._external_source_solo_group_id = None  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

        previous_group = self._find_group_by_id(previous_group_id)

        if previous_group is not None and previous_group != self.client.group:
            self._logger.info(
                "Rejoining previous group %s after external_source",
                previous_group_id,
            )
            await self.client.group.remove_client(self.client)
            await previous_group.add_client(self.client)
            return True
        self._logger.debug(
            "Previous group %s no longer exists or is current group, "
            "falling back to normal switch cycle",
            previous_group_id,
        )
        return False

    def _find_group_by_id(self, group_id: str | None) -> SendspinGroup | None:
        """Find a group by its ID from all connected clients."""
        if group_id is None:
            return None

        server = self.client._server  # noqa: SLF001
        for client in server._clients:  # noqa: SLF001
            if client.group.group_id == group_id:
                return client.group
        return None
