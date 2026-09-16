"""Tests for client-level external_source handling and switch cycling.

These cover the state machine that was previously embedded in the controller
role: any client (regardless of negotiated roles) entering ``external_source``
must be pulled out of multi-client groups, and the switch command must respect
the previous-group rejoin priority and the spec's cycle ordering.
"""

from __future__ import annotations

import asyncio
import dataclasses
from dataclasses import dataclass

import pytest

from aiosendspin.models.controller import ControllerCommandPayload
from aiosendspin.models.core import (
    ClientCommandPayload,
    ClientHelloPayload,
    StreamEndMessage,
)
from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.models.types import (
    AudioCodec,
    MediaCommand,
    PlaybackStateType,
    PlayerCommand,
    Roles,
)
from aiosendspin.server.client import SendspinClient
from aiosendspin.server.clock import LoopClock
from aiosendspin.server.group import SendspinGroup
from aiosendspin.server.roles.controller.v1 import ControllerV1Role


@dataclass(slots=True)
class _DummyServer:
    loop: asyncio.AbstractEventLoop
    clock: LoopClock
    id: str = "srv"
    name: str = "server"
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
        self.sent_messages: list[object] = []

    async def disconnect(self, *, retry_connection: bool = True) -> None:  # noqa: ARG002
        return

    def send_message(self, message: object) -> None:
        self.sent_messages.append(message)

    def send_role_message(self, role: str, message: object) -> None:  # noqa: ARG002
        self.sent_messages.append(message)

    def send_binary(
        self,
        data: bytes,  # noqa: ARG002
        *,
        role: str,  # noqa: ARG002
        timestamp_us: int,  # noqa: ARG002
        message_type: int,  # noqa: ARG002
        buffer_end_time_us: int | None = None,  # noqa: ARG002
        buffer_byte_count: int | None = None,  # noqa: ARG002
        duration_us: int | None = None,  # noqa: ARG002
        player_audio_header: bool = False,  # noqa: ARG002
    ) -> bool:
        return True


def _hello(client_id: str, *, supported_roles: list[str]) -> ClientHelloPayload:
    player_support: ClientHelloPlayerSupport | None = None
    if Roles.PLAYER.value in supported_roles:
        player_support = ClientHelloPlayerSupport(
            supported_formats=[
                SupportedAudioFormat(
                    codec=AudioCodec.PCM,
                    channels=2,
                    sample_rate=48000,
                    bit_depth=16,
                )
            ],
            buffer_capacity=100_000,
            supported_commands=[PlayerCommand.VOLUME, PlayerCommand.MUTE],
        )
    return ClientHelloPayload(
        client_id=client_id,
        name=client_id,
        version=1,
        supported_roles=supported_roles,
        player_support=player_support,
    )


def _make_client(
    server: _DummyServer, client_id: str, *, supported_roles: list[str]
) -> tuple[SendspinClient, _DummyConnection]:
    client = SendspinClient(server, client_id=client_id)
    server.register(client)
    SendspinGroup(server, client)
    conn = _DummyConnection()
    client.attach_connection(
        conn,
        client_info=_hello(client_id, supported_roles=supported_roles),
        negotiated_roles=supported_roles,
        active_roles=supported_roles,
    )
    client.mark_connected()
    return client, conn


@pytest.mark.asyncio
async def test_external_source_moves_player_only_client_out_of_multi_group() -> None:
    """Player-only client entering external_source must leave the shared group."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))

    player_a, conn_a = _make_client(server, "player-a", supported_roles=[Roles.PLAYER.value])
    player_b, _ = _make_client(server, "player-b", supported_roles=[Roles.PLAYER.value])

    shared_group = player_a.group
    await shared_group.add_client(player_b)
    assert player_a.group is player_b.group
    assert len(shared_group.clients) == 2
    shared_group_id = shared_group.group_id

    conn_a.sent_messages.clear()
    await player_a.handle_availability_change(available=False)

    # Player A left the shared group, landed in a solo group.
    assert player_a not in shared_group.clients
    assert player_a.group is not shared_group
    assert len(player_a.group.clients) == 1

    # Previous group bookkeeping was recorded for later rejoin.
    assert player_a._previous_group_id == shared_group_id  # noqa: SLF001
    assert player_a._external_source_solo_group_id == player_a.group.group_id  # noqa: SLF001

    # The player role's on_stream_end fired stream/end on the connection.
    assert any(isinstance(msg, StreamEndMessage) for msg in conn_a.sent_messages)


@pytest.mark.asyncio
async def test_external_source_in_solo_group_stops_playback() -> None:
    """Already-solo client transitioning to external_source stops the group."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))

    player, _ = _make_client(server, "player-solo", supported_roles=[Roles.PLAYER.value])
    group = player.group
    group.start_stream()
    assert group.state == PlaybackStateType.PLAYING

    await player.handle_availability_change(available=False)

    # Same solo group, now stopped.
    assert player.group is group
    assert group.state == PlaybackStateType.STOPPED
    # No previous group recorded; we were already solo.
    assert player._previous_group_id is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_switch_after_external_source_rejoins_previous_group() -> None:
    """Switch command after leaving external_source rejoins the original group."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))

    client, _ = _make_client(
        server,
        "ctrl-player",
        supported_roles=[Roles.PLAYER.value, Roles.CONTROLLER.value],
    )
    other, _ = _make_client(server, "player-other", supported_roles=[Roles.PLAYER.value])

    shared_group = client.group
    await shared_group.add_client(other)
    assert client.group is other.group
    shared_group_id = shared_group.group_id

    # Enter external_source — moved to solo, previous group remembered.
    await client.handle_availability_change(available=False)
    assert client._previous_group_id == shared_group_id  # noqa: SLF001
    assert client.group is not shared_group

    # Leave external_source.
    await client.handle_availability_change(available=True)

    # Switch command should rejoin the previous (shared) group.
    await client.handle_switch_command()

    assert client.group is shared_group
    # Previous-group tracking is cleared after the rejoin.
    assert client._previous_group_id is None  # noqa: SLF001
    assert client._external_source_solo_group_id is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_switch_while_in_external_source_is_ignored() -> None:
    """Switch must not move a client that is still in external_source."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))

    client, _ = _make_client(
        server,
        "ctrl",
        supported_roles=[Roles.PLAYER.value, Roles.CONTROLLER.value],
    )
    other, _ = _make_client(server, "other-player", supported_roles=[Roles.PLAYER.value])
    other_group = other.group
    other_group.start_stream()

    await client.handle_availability_change(available=False)
    group_before = client.group

    await client.handle_switch_command()

    assert client.group is group_before


@pytest.mark.asyncio
async def test_switch_cycle_for_player_client_ends_with_solo_option() -> None:
    """Player clients see multi-client playing -> single playing -> solo option."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))

    # Client under test holds player + controller, starts solo+stopped.
    client, _ = _make_client(
        server,
        "ctrl-player",
        supported_roles=[Roles.PLAYER.value, Roles.CONTROLLER.value],
    )

    # Multi-client playing group: alpha + beta.
    alpha, _ = _make_client(server, "alpha", supported_roles=[Roles.PLAYER.value])
    beta, _ = _make_client(server, "beta", supported_roles=[Roles.PLAYER.value])
    multi_group = alpha.group
    await multi_group.add_client(beta)
    multi_group.start_stream()

    # Single-player playing group: gamma.
    gamma, _ = _make_client(server, "gamma", supported_roles=[Roles.PLAYER.value])
    gamma_group = gamma.group
    gamma_group.start_stream()

    cycle = client._build_group_cycle(  # noqa: SLF001
        client._get_all_groups(),  # noqa: SLF001
        client.group,
        has_player_role=True,
    )

    # current_group is solo+stopped, so the cycle ends with that same group as
    # the "land in solo" option (not None).
    assert cycle == [multi_group, gamma_group, client.group]
    # First switch step jumps from the solo current group to the multi group.
    await client.handle_switch_command()
    assert client.group is multi_group


@pytest.mark.asyncio
async def test_switch_cycle_from_active_solo_uses_none_solo_marker() -> None:
    """When the current group is not solo+stopped, the solo option is a new group."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))

    client, _ = _make_client(
        server,
        "ctrl-player",
        supported_roles=[Roles.PLAYER.value, Roles.CONTROLLER.value],
    )
    alpha, _ = _make_client(server, "alpha", supported_roles=[Roles.PLAYER.value])
    beta, _ = _make_client(server, "beta", supported_roles=[Roles.PLAYER.value])
    multi_group = alpha.group
    await multi_group.add_client(beta)
    multi_group.start_stream()
    await multi_group.add_client(client)
    assert client.group is multi_group

    cycle = client._build_group_cycle(  # noqa: SLF001
        client._get_all_groups(),  # noqa: SLF001
        client.group,
        has_player_role=True,
    )

    # Only one group exists now (multi_group with all three clients).
    assert cycle == [multi_group, None]


@pytest.mark.asyncio
async def test_switch_cycle_for_non_player_client_omits_solo_option() -> None:
    """Clients without the player role get no solo step in the switch cycle."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))

    # Controller-only client (no player role).
    client = SendspinClient(server, client_id="remote")
    server.register(client)
    SendspinGroup(server, client)
    client.attach_connection(
        _DummyConnection(),
        client_info=ClientHelloPayload(
            client_id="remote",
            name="remote",
            version=1,
            supported_roles=[Roles.CONTROLLER.value],
        ),
        negotiated_roles=[Roles.CONTROLLER.value],
        active_roles=[Roles.CONTROLLER.value],
    )
    client.mark_connected()

    alpha, _ = _make_client(server, "alpha", supported_roles=[Roles.PLAYER.value])
    beta, _ = _make_client(server, "beta", supported_roles=[Roles.PLAYER.value])
    multi_group = alpha.group
    await multi_group.add_client(beta)
    multi_group.start_stream()

    gamma, _ = _make_client(server, "gamma", supported_roles=[Roles.PLAYER.value])
    gamma_group = gamma.group
    gamma_group.start_stream()

    cycle = client._build_group_cycle(  # noqa: SLF001
        client._get_all_groups(),  # noqa: SLF001
        client.group,
        has_player_role=False,
    )

    assert cycle == [multi_group, gamma_group]


@pytest.mark.asyncio
async def test_controller_role_delegates_switch_command_to_client() -> None:
    """ControllerV1Role's on_command must hand SWITCH to the client state machine."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))

    client, _ = _make_client(
        server,
        "ctrl",
        supported_roles=[Roles.PLAYER.value, Roles.CONTROLLER.value],
    )
    other, _ = _make_client(server, "other", supported_roles=[Roles.PLAYER.value])
    other_group = other.group
    other_group.start_stream()

    controller_role = client.role(Roles.CONTROLLER.value)
    assert isinstance(controller_role, ControllerV1Role)

    controller_role.on_command(
        ClientCommandPayload(controller=ControllerCommandPayload(command=MediaCommand.SWITCH))
    )

    # on_command schedules an eager task; yield twice to let it complete.
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert client.group is other_group
