"""PlayerGroupRole - group-level player coordination."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from aiosendspin.models.types import has_role_family
from aiosendspin.server.events import ClientEvent
from aiosendspin.server.roles.base import GroupRole
from aiosendspin.server.roles.player.events import (
    PlayerGroupMuteChangedEvent,
    PlayerGroupVolumeChangedEvent,
    VolumeChangedEvent,
)
from aiosendspin.server.roles.player.types import PlayerRoleProtocol

if TYPE_CHECKING:
    from aiosendspin.server.client import SendspinClient
    from aiosendspin.server.group import SendspinGroup
    from aiosendspin.server.roles.base import Role


class PlayerGroupRole(GroupRole):
    """Coordinate player roles across a group."""

    role_family = "player"

    def __init__(self, group: SendspinGroup) -> None:
        """Initialize PlayerGroupRole."""
        super().__init__(group)
        self._last_emitted_volume: int | None = None
        self._last_emitted_muted: bool | None = None
        self._player_client_unsubs: dict[SendspinClient, Callable[[], None]] = {}

    def _player_roles(self) -> list[PlayerRoleProtocol]:
        """Return player role members.

        All members of PlayerGroupRole are PlayerV1Role instances since only
        roles with role_family="player" subscribe to this GroupRole.
        """
        return list(cast("list[PlayerRoleProtocol]", self._members))

    def get_group_volume(self) -> int | None:
        """Return the average volume of volume-supporting players, or None when none support it."""
        volumes = [v for p in self._player_roles() if (v := p.get_player_volume()) is not None]
        if not volumes:
            return None
        return round(sum(volumes) / len(volumes))

    def get_group_muted(self) -> bool | None:
        """Return whether all mute-supporting players are muted, or None when none support it."""
        mutes = [m for p in self._player_roles() if (m := p.get_player_muted()) is not None]
        if not mutes:
            return None
        return all(mutes)

    def set_group_volume(self, level: int) -> bool | None:
        """Set group volume on volume-supporting players using the redistribution algorithm."""
        level = max(0, min(100, level))

        # Build mapping of player -> current volume (only players with volume support)
        player_volumes: dict[PlayerRoleProtocol, float] = {}
        for p in self._player_roles():
            vol = p.get_player_volume()
            if vol is not None:
                player_volumes[p] = float(vol)

        if not player_volumes:
            return True

        # Calculate initial delta
        current_avg = sum(player_volumes.values()) / len(player_volumes)
        delta = level - current_avg

        # Redistribution iterations: a pass that clamps nobody has applied all delta;
        # any other pass drops at least one player, so this ends within one pass per player.
        active_players = list(player_volumes.keys())
        while active_players:
            lost_delta_sum = 0.0
            next_active: list[PlayerRoleProtocol] = []

            for player in active_players:
                current = player_volumes[player]
                proposed = current + delta

                if proposed > 100:
                    clamped = 100.0
                    lost_delta_sum += proposed - clamped
                elif proposed < 0:
                    clamped = 0.0
                    lost_delta_sum += proposed - clamped
                else:
                    clamped = proposed
                    next_active.append(player)

                player_volumes[player] = clamped

            if len(next_active) == len(active_players):
                break

            if next_active:
                delta = lost_delta_sum / len(next_active)
            active_players = next_active

        # Apply to players
        for player, final_vol in player_volumes.items():
            player.set_player_volume(round(final_vol))
        return True

    def set_group_muted(self, muted: bool) -> bool | None:  # noqa: FBT001
        """Set mute state on mute-supporting players."""
        for player in self._player_roles():
            if player.get_player_muted() is not None:
                player.set_player_mute(muted)
        return True

    @property
    def volume(self) -> int:
        """Return current group volume, 100 when no player supports volume."""
        volume = self.get_group_volume()
        return 100 if volume is None else volume

    @property
    def muted(self) -> bool:
        """Return current group mute state, False when no player supports mute."""
        return self.get_group_muted() is True

    def set_volume(self, level: int) -> None:
        """Set group volume using redistribution algorithm."""
        self.set_group_volume(level)

    def set_mute(self, muted: bool) -> None:  # noqa: FBT001
        """Set mute state on mute-supporting players."""
        self.set_group_muted(muted)

    def get_player_clients(self) -> list[SendspinClient]:
        """Return all clients in this group that have an active player role.

        Returns:
            Clients with player roles.
        """
        return [role._client for role in self._player_roles()]  # noqa: SLF001

    # --- Member and client hooks ---

    def on_member_join(self, role: Role) -> None:  # noqa: ARG002
        """Recompute group volume/mute and controller state for the new membership."""
        self._on_membership_changed()

    def on_member_leave(self, role: Role) -> None:  # noqa: ARG002
        """Recompute group volume/mute and controller state for the new membership."""
        self._on_membership_changed()

    def on_client_added(self, client: SendspinClient) -> None:
        """Subscribe to per-client volume events to aggregate group transitions."""
        if client in self._player_client_unsubs:
            return
        if not has_role_family("player", client.negotiated_role_ids):
            return

        def on_client_event(_client: SendspinClient, event: ClientEvent) -> None:
            if isinstance(event, VolumeChangedEvent):
                self._recompute_and_emit()

        unsub = client.add_event_listener(on_client_event)
        self._player_client_unsubs[client] = unsub
        # Prime cached state so the first real echo emits against a known baseline.
        if self._last_emitted_volume is None:
            self._last_emitted_volume = self.volume
        if self._last_emitted_muted is None:
            self._last_emitted_muted = self.muted

    def on_client_removed(self, client: SendspinClient) -> None:
        """Unsubscribe from per-client volume events.

        Membership still includes the leaver here (role unsubscribe runs later),
        so emission is left to on_member_leave.
        """
        if client in self._player_client_unsubs:
            self._player_client_unsubs[client]()
            del self._player_client_unsubs[client]

    def _on_membership_changed(self) -> None:
        """Recompute group volume/mute and push controller state for the current members."""
        # The controller package imports player events, so a module-level import is circular.
        from aiosendspin.server.roles.controller.group import ControllerGroupRole  # noqa: PLC0415

        self._recompute_and_emit()
        controller_group_role = self._group.group_role("controller")
        if isinstance(controller_group_role, ControllerGroupRole):
            controller_group_role.push_state()

    def _recompute_and_emit(self) -> None:
        """Recompute group volume/mute and emit on integer-average / bool transitions."""
        new_volume = self.volume
        previous_volume = self._last_emitted_volume
        self._last_emitted_volume = new_volume
        if previous_volume is not None and previous_volume != new_volume:
            self.emit_group_event(
                PlayerGroupVolumeChangedEvent(
                    previous_volume=previous_volume,
                    volume=new_volume,
                )
            )

        new_muted = self.muted
        previous_muted = self._last_emitted_muted
        self._last_emitted_muted = new_muted
        if previous_muted is not None and previous_muted != new_muted:
            self.emit_group_event(
                PlayerGroupMuteChangedEvent(previous_muted=previous_muted, muted=new_muted)
            )
