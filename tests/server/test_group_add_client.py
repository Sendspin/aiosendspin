"""Tests for SendspinGroup.add_client membership and playback preservation.

Adding a client must not disturb playback it should leave alone: re-adding an
existing member is a no-op, and moving one player out of a multi-player group
leaves the remaining players playing.
"""

from __future__ import annotations

import asyncio
import dataclasses
from dataclasses import dataclass

import pytest

from aiosendspin.models.core import ClientHelloPayload
from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.models.types import (
    AudioCodec,
    PlaybackStateType,
    PlayerCommand,
    Roles,
)
from aiosendspin.server.client import SendspinClient
from aiosendspin.server.clock import LoopClock
from aiosendspin.server.group import SendspinGroup


@dataclass(slots=True)
class _DummyServer:
    loop: asyncio.AbstractEventLoop
    clock: LoopClock
    id: str = "srv"
    name: str = "server"
    visualizer_pitch_enabled: bool = True
    _clients: dict[str, SendspinClient] = dataclasses.field(default_factory=dict)

    def is_external_player(self, client_id: str) -> bool:  # noqa: ARG002
        return False

    def _signal_client_updated(self, client_id: str) -> None:
        pass

    def _signal_client_connected(self, client_id: str) -> None:
        pass

    def _signal_client_disconnected(self, client_id: str, goodbye_reason: object) -> None:
        pass

    def register(self, client: SendspinClient) -> None:
        self._clients[client.client_id] = client

    @property
    def connected_clients(self) -> list[SendspinClient]:
        return [c for c in self._clients.values() if c.is_connected]

    def request_client_playback_connection(self, client_id: str) -> bool:  # noqa: ARG002
        return False


class _DummyConnection:
    receives_group_updates = True

    def __init__(self) -> None:
        self.role_messages: list[tuple[str, object]] = []

    async def disconnect(self, *, retry_connection: bool = True) -> None:  # noqa: ARG002
        return

    def send_message(self, message: object) -> None:
        pass

    def send_role_message(self, role: str, message: object) -> None:
        self.role_messages.append((role, message))

    def send_binary(self, data: bytes, **kwargs: object) -> bool:  # noqa: ARG002
        return True


def _hello(client_id: str) -> ClientHelloPayload:
    return ClientHelloPayload(
        client_id=client_id,
        name=client_id,
        version=1,
        supported_roles=[Roles.PLAYER.value],
        player_support=ClientHelloPlayerSupport(
            supported_formats=[
                SupportedAudioFormat(
                    codec=AudioCodec.PCM, channels=2, sample_rate=48000, bit_depth=16
                )
            ],
            buffer_capacity=100_000,
            supported_commands=[PlayerCommand.VOLUME, PlayerCommand.MUTE],
        ),
    )


def _make_player(server: _DummyServer, client_id: str) -> SendspinClient:
    client = SendspinClient(server, client_id=client_id)
    server.register(client)
    SendspinGroup(server, client)
    client.attach_connection(
        _DummyConnection(),
        client_info=_hello(client_id),
        negotiated_roles=[Roles.PLAYER.value],
        active_roles=[Roles.PLAYER.value],
    )
    client.mark_connected()
    return client


@pytest.mark.asyncio
async def test_readding_existing_member_preserves_playback() -> None:
    """Re-adding a client already in the group is a no-op that leaves it playing."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))

    player = _make_player(server, "web")
    group = player.group
    group._set_playback_state(PlaybackStateType.PLAYING)  # noqa: SLF001

    await group.add_client(player)

    assert group.state == PlaybackStateType.PLAYING


@pytest.mark.asyncio
async def test_moving_one_player_leaves_other_players_playing() -> None:
    """Moving one player out of a multi-player group keeps the remnant playing."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))

    stayer = _make_player(server, "stayer")
    mover = _make_player(server, "mover")
    other = _make_player(server, "other")

    group = stayer.group
    await group.add_client(mover)
    group._set_playback_state(PlaybackStateType.PLAYING)  # noqa: SLF001

    await other.group.add_client(mover)

    # A player (stayer) still sources audio for the old group, so it keeps playing.
    assert mover.group is other.group
    assert group.state == PlaybackStateType.PLAYING
