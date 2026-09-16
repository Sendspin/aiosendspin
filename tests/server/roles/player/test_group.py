"""Tests for PlayerGroupRole volume/mute coordination."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from aiosendspin.server.events import ClientEvent
from aiosendspin.server.roles.player.events import (
    PlayerGroupMuteChangedEvent,
    PlayerGroupVolumeChangedEvent,
    VolumeChangedEvent,
)
from aiosendspin.server.roles.player.group import PlayerGroupRole

if TYPE_CHECKING:
    from aiosendspin.server.client import SendspinClient


class _ClientStub:
    """Minimal SendspinClient stub with event listener wiring."""

    def __init__(self, *, negotiated_roles: list[str] | None = None) -> None:
        self.negotiated_role_ids = negotiated_roles or ["player@v1"]
        self._cbs: list[Callable[[SendspinClient, ClientEvent], None]] = []

    def add_event_listener(
        self, callback: Callable[[SendspinClient, ClientEvent], None]
    ) -> Callable[[], None]:
        self._cbs.append(callback)

        def _remove() -> None:
            if callback in self._cbs:
                self._cbs.remove(callback)

        return _remove

    def signal(self, event: ClientEvent) -> None:
        for cb in list(self._cbs):
            cb(self, event)  # type: ignore[arg-type]


class _PlayerStub:
    """Minimal player role with mutable volume/mute used to drive aggregation."""

    def __init__(self, client: _ClientStub, *, volume: int = 100, muted: bool = False) -> None:
        self._client = client
        self._volume = volume
        self._muted = muted

    def get_player_volume(self) -> int | None:
        return self._volume

    def get_player_muted(self) -> bool | None:
        return self._muted

    def set_player_volume(self, volume: int) -> None:  # noqa: ARG002
        # Production set_volume is send-only; the echoed `client/state` is what
        # mutates the role's volume. Tests drive that echo explicitly below.
        return

    def set_player_mute(self, muted: bool) -> None:  # noqa: ARG002, FBT001
        return

    def echo_state(self, *, volume: int | None = None, muted: bool | None = None) -> None:
        if volume is not None:
            self._volume = volume
        if muted is not None:
            self._muted = muted
        self._client.signal(VolumeChangedEvent(volume=self._volume, muted=self._muted))


def _build(*players: _PlayerStub) -> tuple[PlayerGroupRole, MagicMock]:
    group = MagicMock()
    pgr = PlayerGroupRole(group)
    pgr._members = list(players)  # type: ignore[assignment]  # noqa: SLF001
    for player in players:
        pgr.on_client_added(player._client)  # type: ignore[arg-type]  # noqa: SLF001
    return pgr, group


def test_player_group_role_volume_empty() -> None:
    """Return 100 when no players are subscribed."""
    group = MagicMock()
    pgr = PlayerGroupRole(group)

    assert pgr.volume == 100


def test_player_group_role_volume_single_player() -> None:
    """Return player volume when single player is subscribed."""
    group = MagicMock()
    pgr = PlayerGroupRole(group)

    player = MagicMock()
    player.get_player_volume.return_value = 75
    pgr._members = [player]  # noqa: SLF001

    assert pgr.volume == 75


def test_player_group_role_volume_average() -> None:
    """Return average of player volumes."""
    group = MagicMock()
    pgr = PlayerGroupRole(group)

    p1 = MagicMock()
    p1.get_player_volume.return_value = 80
    p2 = MagicMock()
    p2.get_player_volume.return_value = 60
    pgr._members = [p1, p2]  # noqa: SLF001

    assert pgr.volume == 70


def test_player_group_role_volume_skips_none() -> None:
    """Skip players that return None for volume."""
    group = MagicMock()
    pgr = PlayerGroupRole(group)

    p1 = MagicMock()
    p1.get_player_volume.return_value = 80
    p2 = MagicMock()
    p2.get_player_volume.return_value = None
    pgr._members = [p1, p2]  # noqa: SLF001

    assert pgr.volume == 80


def test_player_group_role_muted_empty() -> None:
    """Return False when no players are subscribed."""
    group = MagicMock()
    pgr = PlayerGroupRole(group)

    assert pgr.muted is False


def test_player_group_role_muted_all_muted() -> None:
    """Return True when all players are muted."""
    group = MagicMock()
    pgr = PlayerGroupRole(group)

    p1 = MagicMock()
    p1.get_player_muted.return_value = True
    p2 = MagicMock()
    p2.get_player_muted.return_value = True
    pgr._members = [p1, p2]  # noqa: SLF001

    assert pgr.muted is True


def test_player_group_role_muted_one_unmuted() -> None:
    """Return False when any player is unmuted."""
    group = MagicMock()
    pgr = PlayerGroupRole(group)

    p1 = MagicMock()
    p1.get_player_muted.return_value = True
    p2 = MagicMock()
    p2.get_player_muted.return_value = False
    pgr._members = [p1, p2]  # noqa: SLF001

    assert pgr.muted is False


def test_player_group_role_muted_ignores_unsupported_players() -> None:
    """Return True when every mute-supporting player is muted, ignoring players without mute."""
    group = MagicMock()
    pgr = PlayerGroupRole(group)

    p1 = MagicMock()
    p1.get_player_muted.return_value = True
    p2 = MagicMock()
    p2.get_player_muted.return_value = None
    pgr._members = [p1, p2]  # noqa: SLF001

    assert pgr.muted is True


def test_player_group_role_unsupported_when_no_player_supports() -> None:
    """Report no group volume/mute when no player supports them, falling back to 100/False."""
    group = MagicMock()
    pgr = PlayerGroupRole(group)

    player = MagicMock()
    player.get_player_volume.return_value = None
    player.get_player_muted.return_value = None
    pgr._members = [player]  # noqa: SLF001

    assert pgr.get_group_volume() is None
    assert pgr.get_group_muted() is None
    assert pgr.volume == 100
    assert pgr.muted is False


def test_player_group_role_volume_zero() -> None:
    """Report a group volume of 0 as 0."""
    group = MagicMock()
    pgr = PlayerGroupRole(group)

    player = MagicMock()
    player.get_player_volume.return_value = 0
    pgr._members = [player]  # noqa: SLF001

    assert pgr.volume == 0


def test_player_group_role_set_volume_empty() -> None:
    """Set volume on empty group is a no-op."""
    group = MagicMock()
    pgr = PlayerGroupRole(group)

    pgr.set_volume(50)  # Should not raise


def test_player_group_role_set_volume_does_not_emit_synchronously() -> None:
    """`set_volume` only dispatches commands; events wait for the client echo."""
    p1 = _PlayerStub(_ClientStub(), volume=100)
    p2 = _PlayerStub(_ClientStub(), volume=100)
    pgr, group = _build(p1, p2)

    pgr.set_volume(50)

    group._signal_event.assert_not_called()  # noqa: SLF001


def test_player_group_role_emits_volume_when_clients_echo_state() -> None:
    """Group event fires when player client/state echoes complete the transition."""
    p1 = _PlayerStub(_ClientStub(), volume=100)
    p2 = _PlayerStub(_ClientStub(), volume=100)
    pgr, group = _build(p1, p2)

    pgr.set_volume(50)
    p1.echo_state(volume=50)
    p2.echo_state(volume=50)

    volume_events = [
        call.args[0]
        for call in group._signal_event.call_args_list  # noqa: SLF001
        if isinstance(call.args[0], PlayerGroupVolumeChangedEvent)
    ]
    assert len(volume_events) >= 1
    final = volume_events[-1]
    assert final.volume == 50


def test_player_group_role_volume_event_carries_previous_value() -> None:
    """`previous_volume` reflects the last emitted group volume."""
    p1 = _PlayerStub(_ClientStub(), volume=100)
    p2 = _PlayerStub(_ClientStub(), volume=100)
    _pgr, group = _build(p1, p2)

    # Seed the cached last-emitted volume with the starting group volume.
    p1.echo_state(volume=100)

    group._signal_event.reset_mock()  # noqa: SLF001
    p1.echo_state(volume=60)
    p2.echo_state(volume=40)

    volume_events = [
        call.args[0]
        for call in group._signal_event.call_args_list  # noqa: SLF001
        if isinstance(call.args[0], PlayerGroupVolumeChangedEvent)
    ]
    assert volume_events
    final = volume_events[-1]
    assert final.previous_volume != final.volume
    assert final.volume == 50


def test_player_group_role_set_mute_does_not_emit_synchronously() -> None:
    """`set_mute` only dispatches commands; events wait for the client echo."""
    p1 = _PlayerStub(_ClientStub(), muted=False)
    p2 = _PlayerStub(_ClientStub(), muted=False)
    pgr, group = _build(p1, p2)

    pgr.set_mute(muted=True)

    group._signal_event.assert_not_called()  # noqa: SLF001


def test_player_group_role_emits_mute_when_all_clients_echo_state() -> None:
    """`PlayerGroupMuteChangedEvent` fires only once all players echo muted."""
    p1 = _PlayerStub(_ClientStub(), muted=False)
    p2 = _PlayerStub(_ClientStub(), muted=False)
    pgr, group = _build(p1, p2)

    pgr.set_mute(muted=True)
    p1.echo_state(muted=True)
    mute_events = [
        call.args[0]
        for call in group._signal_event.call_args_list  # noqa: SLF001
        if isinstance(call.args[0], PlayerGroupMuteChangedEvent)
    ]
    assert mute_events == []  # not yet — group muted only when ALL muted

    p2.echo_state(muted=True)
    mute_events = [
        call.args[0]
        for call in group._signal_event.call_args_list  # noqa: SLF001
        if isinstance(call.args[0], PlayerGroupMuteChangedEvent)
    ]
    assert len(mute_events) == 1
    assert mute_events[0].previous_muted is False
    assert mute_events[0].muted is True


def test_player_group_role_set_volume_redistributes() -> None:
    """Set volume redistributes across multiple players."""
    group = MagicMock()
    pgr = PlayerGroupRole(group)

    p1 = MagicMock()
    p1.get_player_volume.return_value = 100
    p2 = MagicMock()
    p2.get_player_volume.return_value = 100
    pgr._members = [p1, p2]  # noqa: SLF001

    pgr.set_volume(50)

    # Both should be set to 50 (average target is 50, both at 100 -> delta -50)
    p1.set_player_volume.assert_called_once_with(50)
    p2.set_player_volume.assert_called_once_with(50)


def test_player_group_role_set_volume_redistributes_through_iterative_clamp() -> None:
    """Asymmetric starting volumes still hit the target after iterative redistribution."""
    group = MagicMock()
    pgr = PlayerGroupRole(group)

    p1 = MagicMock()
    p1.get_player_volume.return_value = 90
    p2 = MagicMock()
    p2.get_player_volume.return_value = 70
    p3 = MagicMock()
    p3.get_player_volume.return_value = 30
    pgr._members = [p1, p2, p3]  # noqa: SLF001

    pgr.set_volume(100)

    p1.set_player_volume.assert_called_once_with(100)
    p2.set_player_volume.assert_called_once_with(100)
    p3.set_player_volume.assert_called_once_with(100)


def test_player_group_role_set_volume_clamps_to_zero() -> None:
    """Set volume clamps player volumes to 0 minimum."""
    group = MagicMock()
    pgr = PlayerGroupRole(group)

    player = MagicMock()
    player.get_player_volume.return_value = 10
    pgr._members = [player]  # noqa: SLF001

    pgr.set_volume(0)

    player.set_player_volume.assert_called_once_with(0)


def test_player_group_role_set_volume_clamps_to_100() -> None:
    """Set volume clamps player volumes to 100 maximum."""
    group = MagicMock()
    pgr = PlayerGroupRole(group)

    player = MagicMock()
    player.get_player_volume.return_value = 90
    pgr._members = [player]  # noqa: SLF001

    pgr.set_volume(100)

    player.set_player_volume.assert_called_once_with(100)


def test_player_group_role_set_volume_skips_none() -> None:
    """Skip players that return None for volume."""
    group = MagicMock()
    pgr = PlayerGroupRole(group)

    p1 = MagicMock()
    p1.get_player_volume.return_value = 50
    p2 = MagicMock()
    p2.get_player_volume.return_value = None
    pgr._members = [p1, p2]  # noqa: SLF001

    pgr.set_volume(75)

    p1.set_player_volume.assert_called_once_with(75)
    p2.set_player_volume.assert_not_called()


def test_player_group_role_set_volume_redistributes_without_pass_limit() -> None:
    """Redistribution keeps going until all delta is applied, past any fixed pass count."""
    group = MagicMock()
    pgr = PlayerGroupRole(group)

    # Reaching 2 from these volumes takes six passes, each clamping more players at 0.
    volumes = [90, 69, 68, 65, 62, 49, 45, 23, 20, 8]
    players = []
    for volume in volumes:
        player = MagicMock()
        player.get_player_volume.return_value = volume
        players.append(player)
    pgr._members = players  # noqa: SLF001

    pgr.set_volume(2)

    players[0].set_player_volume.assert_called_once_with(20)
    for player in players[1:]:
        player.set_player_volume.assert_called_once_with(0)


def test_player_group_role_set_mute_skips_none() -> None:
    """Skip players that return None for muted."""
    group = MagicMock()
    pgr = PlayerGroupRole(group)

    p1 = MagicMock()
    p1.get_player_muted.return_value = False
    p2 = MagicMock()
    p2.get_player_muted.return_value = None
    pgr._members = [p1, p2]  # noqa: SLF001

    pgr.set_mute(muted=True)

    p1.set_player_mute.assert_called_once_with(True)  # noqa: FBT003
    p2.set_player_mute.assert_not_called()


def test_player_group_role_emits_on_member_join_and_leave() -> None:
    """Group volume/mute events reflect the membership after a join or leave."""
    p1 = _PlayerStub(_ClientStub(), volume=100, muted=True)
    p2 = _PlayerStub(_ClientStub(), volume=50, muted=False)
    pgr, group = _build(p1)

    pgr.subscribe(p2)  # type: ignore[arg-type]

    assert group._signal_event.call_args_list[-2].args[0] == PlayerGroupVolumeChangedEvent(  # noqa: SLF001
        previous_volume=100, volume=75
    )
    assert group._signal_event.call_args_list[-1].args[0] == PlayerGroupMuteChangedEvent(  # noqa: SLF001
        previous_muted=True, muted=False
    )

    group._signal_event.reset_mock()  # noqa: SLF001
    pgr.unsubscribe(p2)  # type: ignore[arg-type]

    assert group._signal_event.call_args_list[0].args[0] == PlayerGroupVolumeChangedEvent(  # noqa: SLF001
        previous_volume=75, volume=100
    )
    assert group._signal_event.call_args_list[1].args[0] == PlayerGroupMuteChangedEvent(  # noqa: SLF001
        previous_muted=False, muted=True
    )


def test_player_group_role_set_volume_no_change_no_event() -> None:
    """No event is emitted when effective group volume is unchanged across echoes."""
    p1 = _PlayerStub(_ClientStub(), volume=50)
    _pgr, group = _build(p1)

    # Prime the last-emitted state to match current.
    p1.echo_state(volume=50)
    group._signal_event.reset_mock()  # noqa: SLF001

    # Re-echo same volume — no transition.
    p1.echo_state(volume=50)

    group._signal_event.assert_not_called()  # noqa: SLF001


def test_player_group_role_unsubscribes_on_client_removed() -> None:
    """`on_client_removed` stops aggregating that client's volume echoes."""
    client = _ClientStub()
    player = _PlayerStub(client, volume=50)
    pgr, group = _build(player)

    pgr.on_client_removed(client)  # type: ignore[arg-type]
    player.echo_state(volume=20)

    group._signal_event.assert_not_called()  # noqa: SLF001
