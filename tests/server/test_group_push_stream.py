"""Tests for SendspinGroup integration with PushStream."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from aiosendspin.models.core import StreamClearMessage, StreamEndMessage, StreamStartMessage
from aiosendspin.models.types import PlaybackStateType
from aiosendspin.server.audio import AudioFormat
from aiosendspin.server.audio_transformers import TransformerPool
from aiosendspin.server.channels import MAIN_CHANNEL
from aiosendspin.server.clock import LoopClock
from aiosendspin.server.events import GroupStateChangedEvent
from aiosendspin.server.group import SendspinGroup
from aiosendspin.server.push_stream import PushStream, StreamStoppedError
from aiosendspin.server.roles.metadata import Metadata
from aiosendspin.server.roles.metadata.group import MetadataGroupRole
from tests.server.test_group_add_client import _DummyConnection, _DummyServer, _make_player

if TYPE_CHECKING:
    from aiosendspin.server.client import SendspinClient


class TestGroupStartStream:
    """Tests for SendspinGroup.start_stream() integration."""

    @pytest.fixture
    def mock_loop(self) -> MagicMock:
        """Create a mock event loop."""
        loop = MagicMock()
        loop.time.return_value = 1000.0
        return loop

    @pytest.fixture
    def mock_server(self, mock_loop: MagicMock) -> MagicMock:
        """Create a mock server."""
        server = MagicMock()
        server.loop = mock_loop
        server.clock = LoopClock(mock_loop)
        server.allow_noncompliant_clients = True
        return server

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock client for the group."""
        client = MagicMock()
        client.client_id = "test-client"
        return client

    def test_start_stream_returns_push_stream(
        self,
        mock_server: MagicMock,
        mock_client: MagicMock,
    ) -> None:
        """start_stream() should return a PushStream instance."""
        group = SendspinGroup(mock_server, mock_client)
        stream = group.start_stream()

        assert isinstance(stream, PushStream)

    def test_start_stream_uses_group(
        self,
        mock_server: MagicMock,
        mock_client: MagicMock,
    ) -> None:
        """start_stream() should bind the PushStream to the group."""
        group = SendspinGroup(mock_server, mock_client)
        stream = group.start_stream()

        assert stream._group is group  # noqa: SLF001

    def test_start_stream_uses_server_loop(
        self,
        mock_server: MagicMock,
        mock_client: MagicMock,
    ) -> None:
        """start_stream() should use the server's event loop."""
        group = SendspinGroup(mock_server, mock_client)
        stream = group.start_stream()

        # The stream should use the server's loop
        assert stream._loop is mock_server.loop  # noqa: SLF001

    def test_group_stop_stops_stream(
        self,
        mock_server: MagicMock,
        mock_client: MagicMock,
    ) -> None:
        """Group stop should call stream stop."""
        group = SendspinGroup(mock_server, mock_client)
        stream = group.start_stream()

        assert not stream.is_stopped

        group.stop_stream()

        assert stream.is_stopped

    def test_group_stop_stream_clears_stream_reference(
        self,
        mock_server: MagicMock,
        mock_client: MagicMock,
    ) -> None:
        """stop_stream() should clear the group's PushStream reference."""
        group = SendspinGroup(mock_server, mock_client)
        group.start_stream()

        group.stop_stream()

        assert group._push_stream is None  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_group_stop_clears_stream_reference(
        self,
        mock_server: MagicMock,
        mock_client: MagicMock,
    ) -> None:
        """stop() should clear the group's PushStream reference."""
        group = SendspinGroup(mock_server, mock_client)
        group.start_stream()

        await group.stop()

        assert group._push_stream is None  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_group_stop_updates_state_to_stopped(
        self,
        mock_server: MagicMock,
        mock_client: MagicMock,
    ) -> None:
        """stop() should set logical playback state to STOPPED."""
        group = SendspinGroup(mock_server, mock_client)
        group.start_stream()

        assert group.state == PlaybackStateType.PLAYING

        await group.stop()

        assert group.state == PlaybackStateType.STOPPED

    @pytest.mark.asyncio
    async def test_group_stop_resets_metadata_progress(
        self,
        mock_server: MagicMock,
        mock_client: MagicMock,
    ) -> None:
        """stop() should report position 0 at the stop, leaving other fields intact."""
        group = SendspinGroup(mock_server, mock_client)
        metadata_role = group.group_role("metadata")
        assert isinstance(metadata_role, MetadataGroupRole)

        group.start_stream()
        metadata_role.set_metadata(
            Metadata(
                title="Test",
                track_progress=30_000,
                track_duration=180_000,
                playback_speed=1000,
            )
        )

        mock_server.loop.time.return_value = 1010.0

        await group.stop()

        assert metadata_role.metadata is not None
        assert metadata_role.metadata.title == "Test"
        assert metadata_role.metadata.track_duration == 180_000
        assert metadata_role.metadata.track_progress == 0
        assert metadata_role.metadata.playback_speed == 0
        assert metadata_role.metadata.timestamp_us == 1_010_000_000

    @pytest.mark.asyncio
    async def test_group_stop_while_paused_resets_metadata_progress(
        self,
        mock_server: MagicMock,
        mock_client: MagicMock,
    ) -> None:
        """stop() should reset a paused position, which does not advance on its own."""
        group = SendspinGroup(mock_server, mock_client)
        metadata_role = group.group_role("metadata")
        assert isinstance(metadata_role, MetadataGroupRole)

        group.start_stream()
        metadata_role.set_metadata(
            Metadata(
                title="Test",
                track_progress=30_000,
                track_duration=180_000,
                playback_speed=0,
            )
        )

        mock_server.loop.time.return_value = 1010.0

        await group.stop()

        assert metadata_role.metadata is not None
        assert metadata_role.metadata.track_progress == 0
        assert metadata_role.metadata.playback_speed == 0

    @pytest.mark.asyncio
    async def test_group_stop_without_progress_leaves_metadata_untouched(
        self,
        mock_server: MagicMock,
        mock_client: MagicMock,
    ) -> None:
        """stop() should publish no metadata state when there is no position to reset."""
        group = SendspinGroup(mock_server, mock_client)
        metadata_role = group.group_role("metadata")
        assert isinstance(metadata_role, MetadataGroupRole)

        group.start_stream()
        metadata_role.set_metadata(Metadata(title="Test"))
        member = MagicMock()
        metadata_role._members = [member]  # noqa: SLF001

        await group.stop()

        member.send_message.assert_not_called()
        assert metadata_role.metadata is not None
        assert metadata_role.metadata.title == "Test"
        assert metadata_role.metadata.track_progress is None

    @pytest.mark.asyncio
    async def test_second_group_stop_publishes_nothing(
        self,
        mock_server: MagicMock,
        mock_client: MagicMock,
    ) -> None:
        """A second stop() is a no-op and publishes no further metadata state."""
        group = SendspinGroup(mock_server, mock_client)
        metadata_role = group.group_role("metadata")
        assert isinstance(metadata_role, MetadataGroupRole)

        group.start_stream()
        metadata_role.set_metadata(
            Metadata(
                title="Test",
                track_progress=30_000,
                track_duration=180_000,
                playback_speed=1000,
            )
        )
        await group.stop()

        member = MagicMock()
        metadata_role._members = [member]  # noqa: SLF001
        mock_server.loop.time.return_value = 1020.0

        assert await group.stop() is False

        member.send_message.assert_not_called()
        assert metadata_role.metadata is not None
        assert metadata_role.metadata.track_progress == 0
        assert metadata_role.metadata.playback_speed == 0

    def test_multiple_start_stream_returns_new_instances(
        self,
        mock_server: MagicMock,
        mock_client: MagicMock,
    ) -> None:
        """Each call to start_stream() should return a new PushStream."""
        group = SendspinGroup(mock_server, mock_client)
        stream1 = group.start_stream()
        stream2 = group.start_stream()

        assert stream1 is not stream2

    @pytest.mark.asyncio
    async def test_replaced_stream_cannot_commit_audio(
        self,
        mock_server: MagicMock,
        mock_client: MagicMock,
    ) -> None:
        """A superseded stream handle must not be able to commit audio."""
        group = SendspinGroup(mock_server, mock_client)
        stream1 = group.start_stream()
        group.start_stream()

        with pytest.raises(StreamStoppedError):
            await stream1.commit_audio()

    def test_start_stream_with_channel_resolver(
        self,
        mock_server: MagicMock,
        mock_client: MagicMock,
    ) -> None:
        """start_stream() can accept a custom channel resolver callback."""
        left_channel = UUID("11111111-1111-1111-1111-111111111111")

        def custom_resolver(player_id: str) -> UUID:
            if player_id == "left-speaker":
                return left_channel
            return MAIN_CHANNEL

        group = SendspinGroup(mock_server, mock_client)
        group.start_stream(channel_resolver=custom_resolver)

        assert group.get_channel_for_player("left-speaker") == left_channel
        assert group.get_channel_for_player("other") == MAIN_CHANNEL

    def test_start_stream_default_updates_state_to_playing(
        self,
        mock_server: MagicMock,
        mock_client: MagicMock,
    ) -> None:
        """start_stream() should update group state to PLAYING by default."""
        group = SendspinGroup(mock_server, mock_client)

        assert group.state == PlaybackStateType.STOPPED

        group.start_stream()

        assert group.state == PlaybackStateType.PLAYING

    def test_stop_stream_preserves_playback_state(
        self,
        mock_server: MagicMock,
        mock_client: MagicMock,
    ) -> None:
        """stop_stream() should not modify logical playback state."""
        group = SendspinGroup(mock_server, mock_client)
        group.start_stream()

        assert group.state == PlaybackStateType.PLAYING

        group.stop_stream()

        assert group.state == PlaybackStateType.PLAYING

    def test_track_transition_no_state_flap(
        self,
        mock_server: MagicMock,
        mock_client: MagicMock,
    ) -> None:
        """stop_stream() + start_stream() should not emit STOPPED during transitions."""
        group = SendspinGroup(mock_server, mock_client)
        state_events: list[PlaybackStateType] = []

        def on_group_event(_: SendspinGroup, event: object) -> None:
            if isinstance(event, GroupStateChangedEvent):
                state_events.append(event.state)

        group.add_event_listener(on_group_event)

        group.start_stream()
        group.stop_stream()
        group.start_stream()

        assert group.state == PlaybackStateType.PLAYING
        assert state_events == [PlaybackStateType.PLAYING]


class _ReplacementConnection(_DummyConnection):
    def __init__(self, *, legacy: bool) -> None:
        super().__init__()
        self.uses_pre_spec_177_wire = legacy


def _replacement_player(
    server: _DummyServer, client_id: str, *, legacy: bool
) -> tuple[SendspinClient, _ReplacementConnection]:
    client = _make_player(server, client_id)
    connection = _ReplacementConnection(legacy=legacy)
    client._connection = connection  # type: ignore[assignment]  # noqa: SLF001
    return client, connection


def _lifecycle_types(connection: _DummyConnection) -> list[type]:
    return [
        type(message)
        for _, message in connection.role_messages
        if isinstance(message, StreamStartMessage | StreamClearMessage | StreamEndMessage)
    ]


async def _start_and_replace(*clients: SendspinClient) -> None:
    group = clients[0].group
    for client in clients[1:]:
        await group.add_client(client)
    stream = group.start_stream()
    stream.prepare_audio(bytes(4800 * 4), AudioFormat(sample_rate=48000, bit_depth=16, channels=2))
    await stream.commit_audio()
    group.start_stream().stop(keep_stream=True)


@pytest.mark.asyncio
async def test_start_stream_replacement_clears_and_keeps_the_stream() -> None:
    """Replacing an active stream sends stream/clear and never stream/end."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))
    player, connection = _replacement_player(server, "player", legacy=False)

    await _start_and_replace(player)

    assert _lifecycle_types(connection) == [StreamStartMessage, StreamClearMessage]


# DEPRECATED(spec-pr-218): remove in aiosendspin <version>
@pytest.mark.asyncio
async def test_start_stream_replacement_ends_the_stream_only_for_legacy_clients() -> None:
    """A pre-#177 client gets stream/end on replacement; a current one in the group does not."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))
    current, current_connection = _replacement_player(server, "current", legacy=False)
    legacy, legacy_connection = _replacement_player(server, "legacy", legacy=True)

    await _start_and_replace(current, legacy)

    assert _lifecycle_types(current_connection) == [StreamStartMessage, StreamClearMessage]
    assert _lifecycle_types(legacy_connection) == [StreamStartMessage, StreamEndMessage]


class TestRoleJoinWithActiveStream:
    """Tests for roles joining a group with an active PushStream."""

    @pytest.fixture
    def mock_loop(self) -> MagicMock:
        """Create a mock event loop."""
        loop = MagicMock()
        loop.time.return_value = 1000.0
        return loop

    @pytest.fixture
    def mock_server(self, mock_loop: MagicMock) -> MagicMock:
        """Create a mock server."""
        server = MagicMock()
        server.loop = mock_loop
        server.clock = LoopClock(mock_loop)
        return server

    @pytest.fixture
    def mock_owner_client(self) -> MagicMock:
        """Create a mock owner client for the group."""
        client = MagicMock()
        client.client_id = "owner-client"
        client.group = MagicMock()
        client.group.stop = AsyncMock()
        return client

    @pytest.fixture
    def mock_player_client(self) -> MagicMock:
        """Create a mock player client to join the group."""
        client = MagicMock()
        client.client_id = "player-client"
        # Mock the group property for ungroup() call
        client.group = MagicMock()
        client.group.stop = AsyncMock()
        client.group._clients = []  # noqa: SLF001
        client.ungroup = AsyncMock()
        # Set up a mock role with audio requirements for role-based join
        mock_role = MagicMock()
        mock_role.role_family = "player"
        mock_role.get_audio_requirements.return_value = MagicMock()  # Has audio requirements
        mock_role.get_player_volume.return_value = 100
        mock_role.get_player_muted.return_value = False
        mock_role.set_player_volume = MagicMock()
        mock_role.set_player_mute = MagicMock()
        client.active_roles = [mock_role]
        return client

    @pytest.mark.asyncio
    async def test_role_join_triggers_on_role_join(
        self,
        mock_server: MagicMock,
        mock_owner_client: MagicMock,
        mock_player_client: MagicMock,
    ) -> None:
        """Adding a client with roles to a group with active stream calls on_role_join."""
        group = SendspinGroup(mock_server, mock_owner_client)
        stream = group.start_stream()

        # Mock on_role_join to track calls
        with patch.object(stream, "on_role_join") as mock_on_role_join:
            await group.add_client(mock_player_client)

            # Should be called once for each role with audio requirements
            mock_on_role_join.assert_called_once()

    @pytest.mark.asyncio
    async def test_client_without_audio_roles_does_not_trigger_on_role_join(
        self,
        mock_server: MagicMock,
        mock_owner_client: MagicMock,
    ) -> None:
        """Adding a client without audio roles does not call on_role_join."""
        group = SendspinGroup(mock_server, mock_owner_client)
        stream = group.start_stream()

        # Create a client with no audio-capable roles
        visualizer_client = MagicMock()
        visualizer_client.client_id = "visualizer-client"
        visualizer_client.group = MagicMock()
        visualizer_client.group.stop = AsyncMock()
        visualizer_client.group._clients = []  # noqa: SLF001
        visualizer_client.ungroup = AsyncMock()
        # No roles with audio requirements
        mock_role = MagicMock()
        mock_role.get_audio_requirements.return_value = None
        visualizer_client.active_roles = [mock_role]

        with patch.object(stream, "on_role_join") as mock_on_role_join:
            await group.add_client(visualizer_client)

            mock_on_role_join.assert_not_called()

    @pytest.mark.asyncio
    async def test_client_join_without_active_stream(
        self,
        mock_server: MagicMock,
        mock_owner_client: MagicMock,
        mock_player_client: MagicMock,
    ) -> None:
        """Adding a client without an active stream does not crash."""
        group = SendspinGroup(mock_server, mock_owner_client)
        # No stream started

        # Should not raise
        await group.add_client(mock_player_client)

    @pytest.mark.asyncio
    async def test_client_join_with_stopped_stream(
        self,
        mock_server: MagicMock,
        mock_owner_client: MagicMock,
        mock_player_client: MagicMock,
    ) -> None:
        """Adding a client to a stopped stream does not call on_role_join."""
        group = SendspinGroup(mock_server, mock_owner_client)
        stream = group.start_stream()
        stream.stop()  # Stop the stream

        with patch.object(stream, "on_role_join") as mock_on_role_join:
            await group.add_client(mock_player_client)

            mock_on_role_join.assert_not_called()

    @pytest.mark.asyncio
    async def test_remove_client_uses_role_stream_end_hook(
        self,
        mock_server: MagicMock,
        mock_owner_client: MagicMock,
        mock_player_client: MagicMock,
    ) -> None:
        """remove_client() should call role.on_stream_end() for removed client roles."""
        group = SendspinGroup(mock_server, mock_owner_client)
        await group.add_client(mock_player_client)
        group.start_stream()

        role = mock_player_client.active_roles[0]
        role.on_stream_end.reset_mock()

        await group.remove_client(mock_player_client)

        role.on_stream_end.assert_called_once()

    @pytest.mark.asyncio
    async def test_remove_last_client_without_active_stream_still_ends_role_streams(
        self,
        mock_server: MagicMock,
    ) -> None:
        """Removing a solo client from a no-stream PLAYING group should still emit stream/end."""
        owner_client = MagicMock()
        owner_client.client_id = "owner-client"
        role = MagicMock()
        role.get_audio_requirements.return_value = MagicMock()
        owner_client.active_roles = [role]

        group = SendspinGroup(mock_server, owner_client)
        group.start_stream()
        group.stop_stream()
        assert group.state == PlaybackStateType.PLAYING
        assert not group.has_active_stream

        role.on_stream_end.reset_mock()
        await group.remove_client(owner_client)

        role.on_stream_end.assert_called_once()


class TestGroupTransformerPool:
    """Tests for SendspinGroup.transformer_pool property."""

    @pytest.fixture
    def mock_loop(self) -> MagicMock:
        """Create a mock event loop."""
        loop = MagicMock()
        loop.time.return_value = 1000.0
        return loop

    @pytest.fixture
    def mock_server(self, mock_loop: MagicMock) -> MagicMock:
        """Create a mock server."""
        server = MagicMock()
        server.loop = mock_loop
        server.clock = LoopClock(mock_loop)
        return server

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock client for the group."""
        client = MagicMock()
        client.client_id = "test-client"
        return client

    def test_group_has_transformer_pool(
        self,
        mock_server: MagicMock,
        mock_client: MagicMock,
    ) -> None:
        """SendspinGroup exposes a transformer_pool."""
        group = SendspinGroup(mock_server, mock_client)

        assert isinstance(group.transformer_pool, TransformerPool)

    def test_transformer_pool_is_same_instance(
        self,
        mock_server: MagicMock,
        mock_client: MagicMock,
    ) -> None:
        """transformer_pool returns the same instance on multiple accesses."""
        group = SendspinGroup(mock_server, mock_client)

        pool1 = group.transformer_pool
        pool2 = group.transformer_pool

        assert pool1 is pool2
