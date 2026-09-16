"""Tests for lifecycle events (ClientConnected, ClientDisconnected)."""

from __future__ import annotations

import asyncio
import dataclasses
from dataclasses import dataclass

import pytest

from aiosendspin.models.core import ClientHelloPayload
from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.models.types import (
    AudioCodec,
    GoodbyeReason,
    PlayerCommand,
    Roles,
)
from aiosendspin.server.client import (
    SendspinClient,
)
from aiosendspin.server.clock import LoopClock
from aiosendspin.server.group import SendspinGroup


@dataclass(slots=True)
class _DummyServer:
    loop: asyncio.AbstractEventLoop
    clock: LoopClock
    id: str = "srv"
    name: str = "server"

    def is_external_player(self, client_id: str) -> bool:  # noqa: ARG002
        return False

    events: list[object] = dataclasses.field(default_factory=list)

    def _signal_client_updated(self, client_id: str) -> None:
        self.events.append(("updated", client_id))

    def _signal_client_connected(self, client_id: str) -> None:
        self.events.append(("connected", client_id))

    def _signal_client_disconnected(self, client_id: str, goodbye_reason: object = None) -> None:
        self.events.append(("disconnected", client_id, goodbye_reason))

    def _signal_client_added(self, client_id: str) -> None:
        self.events.append(("added", client_id))

    def _signal_client_removed(self, client_id: str, reason: str) -> None:  # noqa: ARG002
        self.events.append(("removed", client_id))


class _DummyConnection:
    def __init__(self) -> None:
        self.sent_json: list[object] = []
        self.sent_binary: list[bytes] = []

    async def disconnect(self, *, retry_connection: bool = True) -> None:  # noqa: ARG002
        return

    def send_message(self, message: object) -> None:
        self.sent_json.append(message)

    def send_role_message(self, role: str, message: object) -> None:  # noqa: ARG002
        self.sent_json.append(message)

    def send_binary(
        self,
        data: bytes,
        *,
        role: str,  # noqa: ARG002
        timestamp_us: int,  # noqa: ARG002
        message_type: int,  # noqa: ARG002
        buffer_end_time_us: int | None = None,  # noqa: ARG002
        buffer_byte_count: int | None = None,  # noqa: ARG002
        duration_us: int | None = None,  # noqa: ARG002
        player_audio_header: bool = False,  # noqa: ARG002
    ) -> bool:
        self.sent_binary.append(data)
        return True


def _player_hello(client_id: str) -> ClientHelloPayload:
    return ClientHelloPayload(
        client_id=client_id,
        name=client_id,
        version=1,
        supported_roles=[Roles.PLAYER.value],
        player_support=ClientHelloPlayerSupport(
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
        ),
    )


@pytest.mark.asyncio
async def test_connected_event_fires_on_first_connect() -> None:
    """ClientConnectedEvent fires on the first-ever attach_connection."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))
    client = SendspinClient(server, client_id="player-1")
    SendspinGroup(server, client)

    client.attach_connection(
        _DummyConnection(),
        client_info=_player_hello("player-1"),
        negotiated_roles=[Roles.PLAYER.value],
        active_roles=[Roles.PLAYER.value],
    )

    connected_events = [e for e in server.events if e[0] == "connected"]
    assert len(connected_events) == 1, (
        f"Expected exactly 1 ClientConnectedEvent, got {len(connected_events)}"
    )
    assert connected_events[0][1] == "player-1"


@pytest.mark.asyncio
async def test_connected_event_fires_on_reconnect() -> None:
    """ClientConnectedEvent fires on every reconnect, not just the first connect."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))
    client = SendspinClient(server, client_id="player-1")
    SendspinGroup(server, client)
    hello = _player_hello("player-1")

    # First connect
    client.attach_connection(
        _DummyConnection(),
        client_info=hello,
        negotiated_roles=[Roles.PLAYER.value],
        active_roles=[Roles.PLAYER.value],
    )
    client.mark_connected()
    server.events.clear()

    # Disconnect
    client.detach_connection(GoodbyeReason.RESTART)
    server.events.clear()

    # Reconnect with same hello
    client.attach_connection(
        _DummyConnection(),
        client_info=hello,
        negotiated_roles=[Roles.PLAYER.value],
        active_roles=[Roles.PLAYER.value],
    )

    connected_events = [e for e in server.events if e[0] == "connected"]
    assert len(connected_events) == 1, (
        f"Expected 1 ClientConnectedEvent on reconnect, got {len(connected_events)}"
    )
    assert connected_events[0][1] == "player-1"


@pytest.mark.asyncio
async def test_disconnected_event_fires_on_goodbye() -> None:
    """ClientDisconnectedEvent fires with goodbye reason on intentional disconnect."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))
    client = SendspinClient(server, client_id="player-1")
    SendspinGroup(server, client)

    client.attach_connection(
        _DummyConnection(),
        client_info=_player_hello("player-1"),
        negotiated_roles=[Roles.PLAYER.value],
        active_roles=[Roles.PLAYER.value],
    )
    client.mark_connected()
    server.events.clear()

    client.detach_connection(GoodbyeReason.RESTART)

    disconnected_events = [e for e in server.events if e[0] == "disconnected"]
    assert len(disconnected_events) == 1, (
        f"Expected 1 ClientDisconnectedEvent, got {len(disconnected_events)}"
    )
    assert disconnected_events[0][1] == "player-1"
    assert disconnected_events[0][2] == GoodbyeReason.RESTART


@pytest.mark.asyncio
async def test_disconnected_event_fires_on_connection_drop() -> None:
    """ClientDisconnectedEvent fires with None reason on unexpected drop."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))
    client = SendspinClient(server, client_id="player-1")
    SendspinGroup(server, client)

    client.attach_connection(
        _DummyConnection(),
        client_info=_player_hello("player-1"),
        negotiated_roles=[Roles.PLAYER.value],
        active_roles=[Roles.PLAYER.value],
    )
    client.mark_connected()
    server.events.clear()

    # Simulate unexpected disconnect (no goodbye)
    client.detach_connection(None)

    disconnected_events = [e for e in server.events if e[0] == "disconnected"]
    assert len(disconnected_events) == 1, (
        f"Expected 1 ClientDisconnectedEvent, got {len(disconnected_events)}"
    )
    assert disconnected_events[0][2] is None
