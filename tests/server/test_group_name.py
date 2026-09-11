"""The name a group reports in group/update, and when it publishes a change."""

from __future__ import annotations

import asyncio
import dataclasses
from dataclasses import dataclass

import pytest

from aiosendspin.models.core import (
    ClientHelloPayload,
    GroupUpdateServerMessage,
)
from aiosendspin.models.types import PlaybackStateType
from aiosendspin.server.client import SendspinClient
from aiosendspin.server.clock import LoopClock
from aiosendspin.server.group import SendspinGroup


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


class _RecordingConnection:
    def __init__(self) -> None:
        self.messages: list[object] = []

    async def disconnect(self, *, retry_connection: bool = True) -> None:  # noqa: ARG002
        return

    def send_message(self, message: object) -> None:
        self.messages.append(message)

    def send_role_message(self, role: str, message: object) -> None:
        pass

    def send_binary(self, data: bytes, **kwargs: object) -> bool:  # noqa: ARG002
        return True


def _connected_member(server: _DummyServer, client_id: str, name: str) -> SendspinClient:
    """Register a client under ``name``, group it, and bring it fully up."""
    client = SendspinClient(server, client_id=client_id)
    server.register(client)
    SendspinGroup(server, client)
    client.attach_connection(
        _RecordingConnection(),
        client_info=ClientHelloPayload(client_id=client_id, name=name, supported_roles=[]),
        negotiated_roles=[],
        active_roles=[],
    )
    client.mark_connected()
    return client


def _group_updates(client: SendspinClient) -> list[GroupUpdateServerMessage]:
    connection = client.connection
    assert isinstance(connection, _RecordingConnection)
    return [m for m in connection.messages if isinstance(m, GroupUpdateServerMessage)]


@pytest.mark.asyncio
async def test_group_reports_its_member_name_by_default() -> None:
    """An unnamed group is the device it contains, so it reports that device's name."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))
    client = _connected_member(server, "c1", "Kitchen Speaker")

    assert client.group.group_name == "Kitchen Speaker"


@pytest.mark.asyncio
async def test_the_first_update_carries_the_name() -> None:
    """The field must be present, not omitted as an absent optional.

    ``group/update`` omits None-valued fields, so a group with no name at all left the
    key off the wire entirely, against the spec's requirement to carry every field.
    """
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))
    client = _connected_member(server, "c1", "Kitchen Speaker")

    first = _group_updates(client)[0]
    assert '"group_name":"Kitchen Speaker"' in first.to_json()


@pytest.mark.asyncio
async def test_an_overridden_name_reaches_the_wire() -> None:
    """A name the embedder chose is what members are told, not the derived one."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))
    client = _connected_member(server, "c1", "Kitchen Speaker")

    client.group.set_group_name("Downstairs")

    updates = _group_updates(client)
    assert updates
    assert updates[-1].payload.group_name == "Downstairs"
    assert '"group_name":"Downstairs"' in updates[-1].to_json()


@pytest.mark.asyncio
async def test_setting_the_name_publishes_once_per_change() -> None:
    """Naming a group tells its members; naming it the same again tells no one."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))
    client = _connected_member(server, "c1", "Kitchen Speaker")
    before = len(_group_updates(client))

    client.group.set_group_name("Downstairs")
    client.group.set_group_name("Downstairs")

    assert len(_group_updates(client)) == before + 1


@pytest.mark.asyncio
async def test_clearing_the_name_restores_the_derived_default() -> None:
    """None is not a name of its own; it hands the group back to its member."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))
    client = _connected_member(server, "c1", "Kitchen Speaker")
    client.group.set_group_name("Downstairs")

    client.group.set_group_name(None)

    assert client.group.group_name == "Kitchen Speaker"
    assert _group_updates(client)[-1].payload.group_name == "Kitchen Speaker"


@pytest.mark.asyncio
async def test_no_update_reaches_a_client_still_coming_up() -> None:
    """A member mid-bring-up is not told anything before its own connect update.

    It is attached to its group during the hello exchange, so a group change in that
    window would otherwise reach it ahead of the state it is brought up with.
    """
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))
    client = SendspinClient(server, client_id="c1")
    server.register(client)
    SendspinGroup(server, client)
    client.attach_connection(
        _RecordingConnection(),
        client_info=ClientHelloPayload(client_id="c1", name="Kitchen Speaker", supported_roles=[]),
        negotiated_roles=[],
        active_roles=[],
    )

    client.group._set_playback_state(PlaybackStateType.PLAYING)  # noqa: SLF001
    assert _group_updates(client) == []

    client.mark_connected()

    updates = _group_updates(client)
    assert len(updates) == 1
    assert updates[0].payload.playback_state is PlaybackStateType.PLAYING


@pytest.mark.asyncio
async def test_losing_the_founder_republishes_the_derived_name() -> None:
    """The survivors were reporting the departed device's name, so they must be told."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))
    founder = _connected_member(server, "c1", "Kitchen Speaker")
    joiner = _connected_member(server, "c2", "Living Room")
    group = founder.group
    await group.add_client(joiner)
    assert group.group_name == "Kitchen Speaker"
    before = len(_group_updates(joiner))

    await group.remove_client(founder)

    assert group.group_name == "Living Room"
    updates = _group_updates(joiner)
    assert len(updates) > before
    assert updates[-1].payload.group_name == "Living Room"


@pytest.mark.asyncio
async def test_a_member_learning_its_own_name_tells_the_group() -> None:
    """A device is known by its client_id until its hello lands, and the group with it.

    A group founded before that hello reports the id, so the members it has gained since
    have to be told when the real name arrives.
    """
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))
    founder = SendspinClient(server, client_id="c1")
    server.register(founder)
    SendspinGroup(server, founder)
    joiner = _connected_member(server, "c2", "Living Room")
    group = founder.group
    await group.add_client(joiner)
    assert group.group_name == "c1"
    before = len(_group_updates(joiner))

    founder.attach_connection(
        _RecordingConnection(),
        client_info=ClientHelloPayload(client_id="c1", name="Kitchen Speaker", supported_roles=[]),
        negotiated_roles=[],
        active_roles=[],
    )

    assert group.group_name == "Kitchen Speaker"
    updates = _group_updates(joiner)
    assert len(updates) > before
    assert updates[-1].payload.group_name == "Kitchen Speaker"
