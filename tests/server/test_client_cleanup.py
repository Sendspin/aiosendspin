"""Tests for client cleanup on disconnect."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from unittest.mock import AsyncMock

import pytest

from aiosendspin.models.core import ClientHelloPayload
from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.models.types import AudioCodec, GoodbyeReason, PlayerCommand, Roles
from aiosendspin.server import client as client_module
from aiosendspin.server.client import SendspinClient
from aiosendspin.server.clock import LoopClock
from aiosendspin.server.group import SendspinGroup


@dataclass
class _MockServer:
    """Mock server with remove_client tracking."""

    loop: asyncio.AbstractEventLoop
    clock: LoopClock
    id: str = "srv"
    name: str = "server"
    remove_client: AsyncMock = field(default_factory=AsyncMock)

    def is_external_player(self, client_id: str) -> bool:  # noqa: ARG002
        return False

    def _signal_client_connected(self, client_id: str) -> None:
        pass

    def _signal_client_disconnected(self, client_id: str, goodbye_reason: object = None) -> None:
        pass


class _DummyConnection:
    async def disconnect(self, *, retry_connection: bool = True) -> None:  # noqa: ARG002
        return

    def send_message(self, message: object) -> None:  # noqa: ARG002
        return

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


@pytest.fixture
async def mock_server() -> _MockServer:
    """Create a mock server with remove_client tracking."""
    loop = asyncio.get_running_loop()
    return _MockServer(loop=loop, clock=LoopClock(loop))


@pytest.fixture(autouse=True)
def _fast_cleanup_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client_module, "CLIENT_CLEANUP_DELAY", 0.2)


@pytest.fixture
async def client(mock_server: _MockServer) -> SendspinClient:
    """Create a connected client attached to the mock server."""
    client = SendspinClient(mock_server, client_id="player-1")
    SendspinGroup(mock_server, client)
    client.attach_connection(
        _DummyConnection(),
        client_info=_player_hello("player-1"),
        negotiated_roles=[Roles.PLAYER.value],
        active_roles=[Roles.PLAYER.value],
    )
    client.mark_connected()
    return client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason",
    [
        GoodbyeReason.SHUTDOWN,
        GoodbyeReason.USER_REQUEST,
    ],
)
async def test_immediate_cleanup_on_explicit_disconnect(
    mock_server: _MockServer, client: SendspinClient, reason: GoodbyeReason
) -> None:
    """Client is removed from registry immediately on SHUTDOWN/USER_REQUEST."""
    client.detach_connection(reason)

    # Allow scheduled callback and resulting task to run
    # (call_soon schedules _do_cleanup, which creates a task)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    mock_server.remove_client.assert_awaited_once_with("player-1")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason",
    [
        GoodbyeReason.RESTART,
        None,  # Ungraceful disconnect
    ],
)
async def test_delayed_cleanup_on_reconnectable_disconnect(
    mock_server: _MockServer, client: SendspinClient, reason: GoodbyeReason | None
) -> None:
    """Client cleanup is delayed for RESTART and ungraceful disconnects."""
    client.detach_connection(reason)

    # Allow immediate callbacks to run
    await asyncio.sleep(0)

    # Should not be cleaned up yet
    mock_server.remove_client.assert_not_awaited()

    # Wait for the delayed cleanup
    await asyncio.sleep(client_module.CLIENT_CLEANUP_DELAY + 0.1)

    mock_server.remove_client.assert_awaited_once_with("player-1")


@pytest.mark.asyncio
async def test_no_cleanup_on_another_server_disconnect(
    mock_server: _MockServer, client: SendspinClient
) -> None:
    """Client remains in registry indefinitely after ANOTHER_SERVER disconnect."""
    client.detach_connection(GoodbyeReason.ANOTHER_SERVER)

    await asyncio.sleep(0)
    await asyncio.sleep(client_module.CLIENT_CLEANUP_DELAY + 0.1)

    mock_server.remove_client.assert_not_awaited()


@pytest.mark.asyncio
async def test_another_server_disconnect_ungroups_client(
    mock_server: _MockServer, client: SendspinClient
) -> None:
    """Takeover disconnect ungroups the client out of its shared group."""
    other = SendspinClient(mock_server, client_id="player-2")
    SendspinGroup(mock_server, other)
    await client.group.add_client(other)
    shared_group = client.group

    ungroup_calls: list[str] = []
    original_ungroup = client.ungroup

    async def _record_ungroup() -> None:
        ungroup_calls.append("ungroup")
        await original_ungroup()

    client.ungroup = _record_ungroup  # type: ignore[method-assign]

    client.detach_connection(GoodbyeReason.ANOTHER_SERVER)
    await asyncio.sleep(0)

    assert ungroup_calls == ["ungroup"]
    assert client.group is not shared_group
    assert client.group.clients == [client]


@pytest.mark.asyncio
async def test_another_server_disconnect_solo_client_is_stopped(
    client: SendspinClient,
) -> None:
    """Takeover disconnect is safe for solo clients and results in STOPPED state."""
    client.group.start_stream()
    assert client.group.has_active_stream

    client.detach_connection(GoodbyeReason.ANOTHER_SERVER)
    await asyncio.sleep(0)

    assert len(client.group.clients) == 1
    assert client.group.clients[0] is client
    assert not client.group.has_active_stream


@pytest.mark.asyncio
async def test_another_server_disconnect_already_stopped_remains_stopped(
    client: SendspinClient,
) -> None:
    """Takeover disconnect remains safe when playback is already stopped."""
    assert not client.group.has_active_stream

    client.detach_connection(GoodbyeReason.ANOTHER_SERVER)
    await asyncio.sleep(0)

    assert len(client.group.clients) == 1
    assert client.group.clients[0] is client
    assert not client.group.has_active_stream


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason",
    [
        GoodbyeReason.SHUTDOWN,
        GoodbyeReason.USER_REQUEST,
        GoodbyeReason.RESTART,
        None,
    ],
)
async def test_non_takeover_disconnect_does_not_ungroup_or_stop(
    client: SendspinClient, reason: GoodbyeReason | None
) -> None:
    """Non-takeover disconnect reasons keep existing behavior."""
    client.group.start_stream()
    assert client.group.has_active_stream
    initial_group = client.group

    client.detach_connection(reason)
    await asyncio.sleep(0)

    assert client.group is initial_group
    assert client.group.has_active_stream


@pytest.mark.asyncio
async def test_cleanup_cancelled_on_reconnect(
    mock_server: _MockServer, client: SendspinClient
) -> None:
    """Pending cleanup is cancelled if client reconnects."""
    client.detach_connection(GoodbyeReason.RESTART)

    # Wait some time but not until cleanup fires
    await asyncio.sleep(client_module.CLIENT_CLEANUP_DELAY / 2)
    mock_server.remove_client.assert_not_awaited()

    # Client reconnects
    client.attach_connection(
        _DummyConnection(),
        client_info=_player_hello("player-1"),
        negotiated_roles=[Roles.PLAYER.value],
        active_roles=[Roles.PLAYER.value],
    )
    client.mark_connected()

    # Wait past the original cleanup time
    await asyncio.sleep(client_module.CLIENT_CLEANUP_DELAY)

    # Should not have been cleaned up
    mock_server.remove_client.assert_not_awaited()


@pytest.mark.asyncio
async def test_cleanup_skipped_if_reconnected_before_callback(
    mock_server: _MockServer, client: SendspinClient
) -> None:
    """Cleanup callback is a no-op if client reconnected (double-check via _connected flag)."""
    client.detach_connection(GoodbyeReason.SHUTDOWN)

    # Reconnect immediately (before call_soon callback runs)
    client.attach_connection(
        _DummyConnection(),
        client_info=_player_hello("player-1"),
        negotiated_roles=[Roles.PLAYER.value],
        active_roles=[Roles.PLAYER.value],
    )
    client.mark_connected()

    # Allow callbacks to run
    await asyncio.sleep(0)

    # Cleanup should have been cancelled by attach_connection
    mock_server.remove_client.assert_not_awaited()
