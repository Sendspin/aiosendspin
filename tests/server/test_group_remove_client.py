"""Tests for SendspinGroup.remove_client teardown behavior.

A group only makes sense as a playback session while it has a ``player``-role
client to source audio. When the last such client leaves, any visualizer- or
metadata-only remnant must not keep "playing" as a phantom-led group.
"""

from __future__ import annotations

import asyncio
import dataclasses
from dataclasses import dataclass

import pytest

from aiosendspin.models.core import ClientHelloPayload, ClientStatePayload, StreamEndMessage
from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.models.types import (
    AudioCodec,
    PlaybackStateType,
    PlayerCommand,
    Roles,
)
from aiosendspin.models.visualizer import ClientHelloVisualizerSupport, VisualizerStatePayload
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
    allow_noncompliant_clients: bool = True
    _clients: dict[str, SendspinClient] = dataclasses.field(default_factory=dict)

    def is_external_player(self, client_id: str) -> bool:  # noqa: ARG002
        return False

    def _signal_client_updated(self, client_id: str) -> None:
        pass

    def register(self, client: SendspinClient) -> None:
        self._clients[client.client_id] = client

    @property
    def connected_clients(self) -> list[SendspinClient]:
        return [c for c in self._clients.values() if c.is_connected]

    def request_client_playback_connection(self, client_id: str) -> bool:  # noqa: ARG002
        return False

    def _signal_client_connected(self, client_id: str) -> None:
        pass

    def _signal_client_disconnected(self, client_id: str, goodbye_reason: object = None) -> None:
        pass


class _DummyConnection:
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


def _hello(client_id: str, *, supported_roles: list[str]) -> ClientHelloPayload:
    player_support: ClientHelloPlayerSupport | None = None
    visualizer_support: ClientHelloVisualizerSupport | None = None
    if Roles.PLAYER.value in supported_roles:
        player_support = ClientHelloPlayerSupport(
            supported_formats=[
                SupportedAudioFormat(
                    codec=AudioCodec.PCM, channels=2, sample_rate=48000, bit_depth=16
                )
            ],
            buffer_capacity=100_000,
            supported_commands=[PlayerCommand.VOLUME, PlayerCommand.MUTE],
        )
    if Roles.VISUALIZER.value in supported_roles:
        visualizer_support = ClientHelloVisualizerSupport(buffer_capacity=65536)
    return ClientHelloPayload(
        client_id=client_id,
        name=client_id,
        version=1,
        supported_roles=supported_roles,
        player_support=player_support,
        visualizer_support=visualizer_support,
    )


def _make_client(
    server: _DummyServer, client_id: str, *, supported_roles: list[str]
) -> SendspinClient:
    client = SendspinClient(server, client_id=client_id)
    server.register(client)
    SendspinGroup(server, client)
    client.attach_connection(
        _DummyConnection(),
        client_info=_hello(client_id, supported_roles=supported_roles),
        negotiated_roles=supported_roles,
        active_roles=supported_roles,
    )
    client.mark_connected()
    return client


@pytest.mark.asyncio
async def test_group_stops_when_last_player_leaves_visualizer_behind() -> None:
    """Removing the only player-role client stops a visualizer-only remnant."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))

    player = _make_client(server, "web", supported_roles=[Roles.PLAYER.value])
    visualizer = _make_client(server, "hue", supported_roles=[Roles.VISUALIZER.value])

    group = player.group
    await group.add_client(visualizer)
    group._set_playback_state(PlaybackStateType.PLAYING)  # noqa: SLF001
    assert group.state == PlaybackStateType.PLAYING

    await group.remove_client(player)

    # The visualizer remnant has no audio source, so the group must go idle
    # rather than linger as a phantom-led, still-playing group.
    assert visualizer.group is group
    assert group.state == PlaybackStateType.STOPPED


@pytest.mark.asyncio
async def test_surviving_visualizer_gets_stream_end_without_active_stream() -> None:
    """Stopping a stream-less PLAYING remnant signals survivors' roles directly."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))

    player = _make_client(server, "web", supported_roles=[Roles.PLAYER.value])
    visualizer = _make_client(server, "hue", supported_roles=[Roles.VISUALIZER.value])

    group = player.group
    await group.add_client(visualizer)
    # The visualizer stream was announced before the track transition kept it open.
    role = visualizer.role(Roles.VISUALIZER.value)
    assert role is not None
    role.on_client_state(
        ClientStatePayload(visualizer=VisualizerStatePayload(types=["loudness"], rate_max=30))
    )
    role.on_stream_start()
    # PLAYING with no PushStream mirrors the track-transition window.
    group._set_playback_state(PlaybackStateType.PLAYING)  # noqa: SLF001
    assert not group.has_active_stream

    connection = visualizer.connection
    assert connection is not None
    connection.role_messages.clear()

    await group.remove_client(player)

    # stop() emits stream/end only via an active PushStream, so the survivor's
    # role must be signaled directly to invalidate stale binary.
    assert any(isinstance(msg, StreamEndMessage) for _, msg in connection.role_messages)


@pytest.mark.asyncio
async def test_moving_stopped_solo_player_sends_no_stream_end() -> None:
    """Moving a player out of a STOPPED solo group must not emit a spurious stream/end."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))

    mover = _make_client(server, "mover", supported_roles=[Roles.PLAYER.value])
    target = _make_client(server, "target", supported_roles=[Roles.PLAYER.value])

    # mover's solo group was never streamed, so there is no stale binary to drop.
    assert mover.group.state == PlaybackStateType.STOPPED
    connection = mover.connection
    assert connection is not None
    connection.role_messages.clear()

    await target.group.add_client(mover)

    # A stream/end here would tear down the device right before the new group's
    # stream/start, leaving playback dead.
    assert not any(isinstance(msg, StreamEndMessage) for _, msg in connection.role_messages)


@pytest.mark.asyncio
async def test_removing_sole_player_from_streamless_playing_group_sends_stream_end() -> None:
    """Ungrouping a PLAYING-without-stream solo player still invalidates stale binary."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))

    player = _make_client(server, "web", supported_roles=[Roles.PLAYER.value])
    group = player.group
    group._set_playback_state(PlaybackStateType.PLAYING)  # noqa: SLF001
    assert not group.has_active_stream

    connection = player.connection
    assert connection is not None
    connection.role_messages.clear()

    await group.remove_client(player)

    assert any(isinstance(msg, StreamEndMessage) for _, msg in connection.role_messages)
