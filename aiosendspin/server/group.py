"""Manages and synchronizes playback for a group of one or more clients."""

from __future__ import annotations

import asyncio
import logging
import uuid
from asyncio import Task
from collections.abc import AsyncGenerator, Callable, Coroutine
from contextlib import suppress
from dataclasses import dataclass, replace
from io import BytesIO
from typing import TYPE_CHECKING
from uuid import UUID

from PIL import Image

from aiosendspin.models import (
    BinaryMessageType,
    pack_binary_header_raw,
)
from aiosendspin.models.artwork import (
    ArtworkChannel,
    StreamArtworkChannelConfig,
    StreamStartArtwork,
)
from aiosendspin.models.controller import (
    ControllerCommandPayload,
    ControllerStatePayload,
)
from aiosendspin.models.core import (
    GroupUpdateServerMessage,
    GroupUpdateServerPayload,
    ServerStateMessage,
    ServerStatePayload,
    StreamEndMessage,
    StreamEndPayload,
    StreamRequestFormatPayload,
    StreamStartMessage,
    StreamStartPayload,
)
from aiosendspin.models.metadata import Progress
from aiosendspin.models.player import (
    StreamStartPlayer,
)
from aiosendspin.models.types import (
    ArtworkSource,
    MediaCommand,
    PictureFormat,
    PlaybackStateType,
    Roles,
)
from aiosendspin.models.visualizer import StreamStartVisualizer

from .events import ClientEvent, VolumeChangedEvent
from .metadata import Metadata
from .stream import AudioCodec, AudioFormat, ClientStreamConfig, MediaStream, Streamer

# The cyclic import is not an issue during runtime, so hide it
# pyright: reportImportCycles=none
if TYPE_CHECKING:
    import av

    from .client import SendspinClient
    from .player import PlayerClient
    from .server import SendspinServer

INITIAL_PLAYBACK_DELAY_US = 1_000_000
# Maximum time to spend prefilling buffers before starting playback.
# This allows live/radio streams (which arrive at real-time rate) to start
# with a partial buffer rather than blocking indefinitely.
MAX_PREFILL_DURATION_S = 3.0

logger = logging.getLogger(__name__)


class GroupEvent:
    """Base event type used by SendspinGroup.add_event_listener()."""


# TODO: make types more fancy
@dataclass
class GroupCommandEvent(GroupEvent):
    """A command was sent to the group."""

    command: MediaCommand
    """The command that was sent."""
    volume: int | None = None
    """For MediaCommand.VOLUME, the target volume (0-100)."""
    mute: bool | None = None
    """For MediaCommand.MUTE, the target mute status."""


@dataclass
class GroupStateChangedEvent(GroupEvent):
    """Group state has changed."""

    state: PlaybackStateType
    """The new group state."""


@dataclass
class GroupMemberAddedEvent(GroupEvent):
    """A client was added to the group."""

    client_id: str
    """The ID of the client that was added."""


@dataclass
class GroupMemberRemovedEvent(GroupEvent):
    """A client was removed from the group."""

    client_id: str
    """The ID of the client that was removed."""


@dataclass
class GroupDeletedEvent(GroupEvent):
    """This group has no more members and has been deleted."""


@dataclass
class _StreamerReconfigureCommand:
    """Signal to reconfigure the running streamer with new player topology."""

    all_player_configs: list[ClientStreamConfig]
    """List of ClientStreamConfig for all players (existing and new)."""


def _build_artwork_stream_info(
    client_state: dict[int, ArtworkChannel],
) -> StreamStartArtwork:
    """Build StreamStartArtwork from client artwork channel state."""
    stream_channels = []
    for channel_num in sorted(client_state.keys()):
        channel = client_state[channel_num]
        stream_channels.append(
            StreamArtworkChannelConfig(
                source=channel.source,
                format=channel.format,
                width=channel.media_width,
                height=channel.media_height,
            )
        )
    return StreamStartArtwork(channels=stream_channels)


class SendspinGroup:
    """
    A group of one or more clients for synchronized playback.

    Handles synchronized audio streaming across multiple clients with automatic
    format conversion and buffer management. Every client is always assigned to
    a group to simplify grouping requests.
    """

    _clients: list[SendspinClient]
    """List of all clients in this group."""
    _client_artwork_state: dict[str, dict[int, ArtworkChannel]]
    """Mapping of client IDs to their per-channel artwork state (channel 0-3)."""
    _server: SendspinServer
    """Reference to the SendspinServer instance."""
    _stream_task: Task[int] | None = None
    """Task handling the audio streaming loop, None when not streaming."""
    _current_metadata: Metadata | None = None
    """Current metadata for the group, None if no metadata set."""
    _current_media_art: dict[ArtworkSource, Image.Image]
    """Current media art images for the group, keyed by source type."""
    _audio_encoders: dict[AudioFormat, av.AudioCodecContext]
    """Mapping of audio formats to their base64 encoded headers."""
    _preferred_stream_codec: AudioCodec = AudioCodec.OPUS
    """Preferred codec used by the current stream."""
    _event_cbs: list[Callable[[SendspinGroup, GroupEvent], Coroutine[None, None, None]]]
    """List of event callbacks for this group."""
    _current_state: PlaybackStateType = PlaybackStateType.STOPPED
    """Current playback state of the group."""
    _group_id: str
    """Unique identifier for this group."""
    _group_name: str | None
    """Friendly name for this group."""
    _streamer: Streamer | None
    """Active Streamer instance for the current stream, None when not streaming."""
    _media_stream: MediaStream | None
    """Current MediaStream being played, None when not streaming."""
    _stream_commands: asyncio.Queue[_StreamerReconfigureCommand] | None
    """Command queue for the active streamer task, None when not streaming."""
    _play_start_time_us: int | None
    """Absolute timestamp in microseconds when playback started, None when not streaming."""
    _track_progress_timestamp_us: int | None
    """Timestamp in microseconds when track_progress was last updated, for progress calculation."""
    _scheduled_stop_handle: asyncio.TimerHandle | None
    """Timer handle for scheduled stop, None when no stop is scheduled."""
    _last_sent_volume: int | None
    """Last volume sent to controller clients, for change detection."""
    _last_sent_muted: bool | None
    """Last muted state sent to controller clients, for change detection."""
    _last_sent_supported_commands: list[MediaCommand] | None
    """Last computed supported commands sent to clients (output of _get_supported_commands())."""
    _supported_commands: list[MediaCommand]
    """Commands supported by the application (input to _get_supported_commands())."""
    _playback_lock: asyncio.Lock
    """Lock to serialize play_media() and stop() operations, preventing race conditions."""

    def __init__(self, server: SendspinServer, *args: SendspinClient) -> None:
        """
        DO NOT CALL THIS CONSTRUCTOR. INTERNAL USE ONLY.

        Groups are managed automatically by the server.

        Initialize a new SendspinGroup.

        Args:
            server: The SendspinServer instance this group belongs to.
            *args: Clients to add to this group.
        """
        self._clients = list(args)
        assert len(self._clients) > 0, "A group must have at least one client"
        self._client_artwork_state = {}
        self._server = server
        self._stream_task: Task[int] | None = None
        self._current_metadata = None
        self._current_media_art = {}
        self._audio_encoders = {}
        self._event_cbs = []
        self._group_id = str(uuid.uuid4())
        self._group_name: str | None = None
        self._streamer: Streamer | None = None
        self._media_stream: MediaStream | None = None
        self._stream_commands: asyncio.Queue[_StreamerReconfigureCommand] | None = None
        self._play_start_time_us: int | None = None
        self._track_progress_timestamp_us: int | None = None
        self._scheduled_stop_handle: asyncio.TimerHandle | None = None
        self._last_sent_volume: int | None = None
        self._last_sent_muted: bool | None = None
        self._last_sent_supported_commands: list[MediaCommand] | None = None
        self._supported_commands: list[MediaCommand] = []
        self._client_event_unsubs: dict[SendspinClient, Callable[[], None]] = {}
        self._playback_lock = asyncio.Lock()
        logger.debug(
            "SendspinGroup initialized with %d client(s): %s",
            len(self._clients),
            [type(c).__name__ for c in self._clients],
        )

    async def play_media(
        self,
        media_stream: MediaStream,
        *,
        play_start_time_us: int | None = None,
    ) -> int:
        """Start synchronized playback for the current group using a MediaStream."""
        logger.debug(
            "Starting play_media with play_start_time_us=%s",
            play_start_time_us,
        )

        # Hold lock during setup to prevent concurrent stop() from interfering
        async with self._playback_lock:
            # Cancel any previously scheduled stop to prevent race conditions
            if self._scheduled_stop_handle is not None:
                logger.debug("Canceling previously scheduled stop")
                self._scheduled_stop_handle.cancel()
                self._scheduled_stop_handle = None

            self._media_stream = media_stream
            self._streamer = None

            start_time_us = (
                play_start_time_us
                if play_start_time_us is not None
                else int(self._server.loop.time() * 1_000_000) + INITIAL_PLAYBACK_DELAY_US
            )
            self._play_start_time_us = start_time_us

            group_players = self.players()
            if not group_players:
                logger.info("No player clients in group; skipping playback")
                self._current_state = PlaybackStateType.STOPPED
                return start_time_us

            streamer = Streamer(
                loop=self._server.loop,
                play_start_time_us=start_time_us,
            )
            self._streamer = streamer
            self._media_stream = media_stream

            # Build configs for all players (all are new for initial setup)
            all_player_configs: list[ClientStreamConfig] = []
            for player in group_players:
                assert player.support
                target_format = player.determine_optimal_format(media_stream.main_channel[1])
                all_player_configs.append(
                    ClientStreamConfig(
                        client_id=player.client.client_id,
                        target_format=target_format,
                        buffer_capacity_bytes=player.support.buffer_capacity,
                        send=player.client.send_message,
                    )
                )

            start_payloads, channel_sources = await streamer.configure(
                all_player_configs, media_stream
            )
            self._stream_commands = asyncio.Queue()
            self._stream_task = self._server.loop.create_task(
                self._run_streamer(streamer, media_stream, channel_sources)
            )

            # Notify clients about the upcoming stream configuration
            for player in group_players:
                player_payload = start_payloads.get(player.client.client_id)
                assert player_payload is not None
                self._send_stream_start_msg(
                    player.client,
                    player_payload,
                )

            for client in self._clients:
                if client.check_role(Roles.PLAYER):
                    continue
                if client.check_role(Roles.VISUALIZER) or client.check_role(Roles.ARTWORK):
                    self._send_stream_start_msg(client, None)

            # Send any pre-existing artwork to artwork clients
            await self._send_existing_artwork_to_clients()

            self._current_state = PlaybackStateType.PLAYING
            self._signal_event(GroupStateChangedEvent(PlaybackStateType.PLAYING))
            self._send_group_update_to_clients()

        # Release lock during the blocking await on stream task
        # This allows stop() to cancel the stream
        end_time_us = start_time_us
        stream_task = self._stream_task
        current_media_stream = media_stream
        if stream_task is not None:
            end_time_us = await stream_task
            # Only clear resources if they're still ours (not replaced by a new play_media call)
            if self._stream_task is stream_task:
                self._stream_task = None
            if self._media_stream is current_media_stream:
                self._streamer = None
                self._media_stream = None
                self._stream_commands = None

        return end_time_us

    def _send_group_update_to_clients(self) -> None:
        """Send group/update and server/state messages to all clients."""
        group_message = GroupUpdateServerMessage(
            GroupUpdateServerPayload(
                playback_state=self._current_state,
                group_id=self.group_id,
                group_name=self.group_name,
            )
        )
        supported_commands = self._get_supported_commands()
        controller_state = ControllerStatePayload(
            supported_commands=supported_commands,
            volume=self.volume,
            muted=self.muted,
        )
        # Update tracking variables
        self._last_sent_volume = self.volume
        self._last_sent_muted = self.muted
        self._last_sent_supported_commands = supported_commands

        for client in self._clients:
            # Send group/update to all clients
            client.send_message(group_message)

            # Build server/state payload with relevant fields for this client
            metadata_for_client = None
            if client.check_role(Roles.METADATA):
                if self._current_metadata is not None:
                    metadata_update = self._current_metadata.snapshot_update(
                        int(self._server.loop.time() * 1_000_000)
                    )
                else:
                    metadata_update = Metadata.cleared_update(
                        int(self._server.loop.time() * 1_000_000)
                    )
                # Use calculated track progress for actively playing content
                if self._current_metadata is not None:
                    current_progress = self._get_current_track_progress()
                    # Update the progress object with current calculated progress
                    if (
                        current_progress is not None
                        and self._current_metadata.track_duration is not None
                        and self._current_metadata.playback_speed is not None
                    ):
                        metadata_update.progress = Progress(
                            track_progress=current_progress,
                            track_duration=self._current_metadata.track_duration,
                            playback_speed=self._current_metadata.playback_speed,
                        )
                metadata_for_client = metadata_update

            controller_for_client = None
            if client.check_role(Roles.CONTROLLER):
                controller_for_client = controller_state

            # Send single server/state message with all relevant payloads
            if metadata_for_client is not None or controller_for_client is not None:
                state_message = ServerStateMessage(
                    ServerStatePayload(
                        metadata=metadata_for_client, controller=controller_for_client
                    )
                )
                client.send_message(state_message)

    def _send_controller_state_to_clients(self) -> None:
        """Send server/state with controller payload to all controller clients."""
        current_volume = self.volume
        current_muted = self.muted
        current_supported_commands = self._get_supported_commands()

        # Only send if any field changed
        if (
            self._last_sent_volume == current_volume
            and self._last_sent_muted == current_muted
            and self._last_sent_supported_commands == current_supported_commands
        ):
            return

        self._last_sent_volume = current_volume
        self._last_sent_muted = current_muted
        self._last_sent_supported_commands = current_supported_commands
        controller_state = ControllerStatePayload(
            supported_commands=current_supported_commands,
            volume=current_volume,
            muted=current_muted,
        )
        for client in self._clients:
            if client.check_role(Roles.CONTROLLER):
                state_message = ServerStateMessage(ServerStatePayload(controller=controller_state))
                client.send_message(state_message)

    async def _handle_reconfiguration_command(
        self,
        command: _StreamerReconfigureCommand,
        streamer: Streamer,
        media_stream: MediaStream,
        active_channels: dict[UUID, AsyncGenerator[bytes, None]],
        just_started_channels: set[UUID],
    ) -> None:
        """Handle a streamer reconfiguration command by updating topology and notifying clients."""
        # Reconfigure with current player topology
        start_payloads, new_sources = await streamer.configure(
            command.all_player_configs, media_stream
        )

        # Add new channel sources to active channels
        for channel_id, source in new_sources.items():
            if channel_id not in active_channels:
                active_channels[channel_id] = source
                just_started_channels.add(channel_id)

        # Drop channel sources that were removed by configure()
        removed_channel_ids = set(active_channels) - streamer.get_channel_ids()
        for removed_id in removed_channel_ids:
            source = active_channels.pop(removed_id)
            just_started_channels.discard(removed_id)
            with suppress(Exception):
                await source.aclose()

        # Send stream/start messages to affected players
        player_lookup = {player.client.client_id: player for player in self.players()}
        for client_id, player_payload in start_payloads.items():
            player_obj = player_lookup.get(client_id)
            if player_obj is not None:
                self._send_stream_start_msg(
                    player_obj.client,
                    player_stream_info=player_payload,
                )
        # Send group/update and server/state to all clients
        # TODO: only send to clients that were affected by the change!
        self._send_group_update_to_clients()
        logger.debug("streamer reconfigured")

    async def _prefill_channel_buffers(
        self,
        streamer: Streamer,
        active_channels: dict[UUID, AsyncGenerator[bytes, None]],
        just_started_channels: set[UUID],
    ) -> None:
        """Pre-fill buffers for channels that just started before beginning playback."""
        channels_to_check = list(just_started_channels)
        for channel_id in channels_to_check:
            if channel_id not in active_channels:
                just_started_channels.discard(channel_id)
                continue
            # Pre-fill this channel's buffer before starting playback
            prefill_start = self._server.loop.time()
            while streamer.channel_needs_data(channel_id):
                # Check if we've spent too long prefilling (so we don't end up blocking forever
                # on live/radio streams)
                if self._server.loop.time() - prefill_start > MAX_PREFILL_DURATION_S:
                    logger.debug(
                        "Channel %s prefill timeout after %.1fs, continuing with partial buffer",
                        channel_id,
                        MAX_PREFILL_DURATION_S,
                    )
                    break
                source = active_channels[channel_id]
                try:
                    chunk = await asyncio.wait_for(anext(source), timeout=30.0)

                    streamer.prepare(channel_id, chunk, during_initial_buffering=True)
                    continue  # Continue filling buffer
                except StopAsyncIteration:
                    pass  # Channel exhausted (normal completion)
                except TimeoutError:
                    logger.error("Channel %s timed out during prefill, removing", channel_id)
                except Exception:
                    logger.exception("Channel %s failed during prefill, removing", channel_id)
                # Channel done (exhausted, timed out, or failed) - clean up and exit
                del active_channels[channel_id]
                with suppress(Exception):
                    await source.aclose()
                break
            # Channel is now pre-filled, remove from just_started set
            just_started_channels.discard(channel_id)

    async def _read_pending_chunks(
        self,
        streamer: Streamer,
        active_channels: dict[UUID, AsyncGenerator[bytes, None]],
    ) -> bool:
        """Read chunks from channels that need data until buffers are full.

        Returns:
            True if there are still active channels, False if all channels are exhausted.
        """
        fill_start = self._server.loop.time()
        any_channel_needs_data = True
        while any_channel_needs_data and active_channels:
            # Avoid blocking indefinitely on live/radio streams that can never
            # build the full target buffer ahead of real-time.
            if self._server.loop.time() - fill_start > MAX_PREFILL_DURATION_S:
                logger.debug(
                    "Pending read timeout after %.1fs, continuing with partial buffer",
                    MAX_PREFILL_DURATION_S,
                )
                break
            any_channel_needs_data = False
            for channel_id in list(active_channels.keys()):
                if not streamer.channel_needs_data(channel_id):
                    continue
                any_channel_needs_data = True
                source = active_channels[channel_id]
                try:
                    chunk = await asyncio.wait_for(anext(source), timeout=30.0)
                    streamer.prepare(channel_id, chunk)
                    continue  # Done, continue with next channel
                except StopAsyncIteration:
                    pass  # Channel exhausted (normal completion)
                except TimeoutError:
                    logger.error("Channel %s timed out during read, removing", channel_id)
                except Exception:
                    logger.exception("Channel %s failed during read, removing", channel_id)
                # Channel done (exhausted, timed out, or failed) - clean up
                del active_channels[channel_id]
                with suppress(Exception):
                    await source.aclose()

        return bool(active_channels)

    async def _run_streamer(
        self,
        streamer: Streamer,
        media_stream: MediaStream,
        active_channels: dict[UUID, AsyncGenerator[bytes, None]],
    ) -> int:
        """Consume media channels, distribute via streamer, and return end timestamp."""
        last_end_us = self._play_start_time_us or int(self._server.loop.time() * 1_000_000)
        just_started_channels: set[UUID] = set(active_channels.keys())

        try:
            while True:
                # Check for commands before processing chunks
                if self._stream_commands is not None and not self._stream_commands.empty():
                    command = self._stream_commands.get_nowait()
                    await self._handle_reconfiguration_command(
                        command, streamer, media_stream, active_channels, just_started_channels
                    )
                    continue

                # Pre-fill buffers for channels that just started
                if just_started_channels:
                    await self._prefill_channel_buffers(
                        streamer, active_channels, just_started_channels
                    )

                # Read chunks from channels that need data until buffers are full
                if not await self._read_pending_chunks(streamer, active_channels):
                    break  # All channels exhausted

                # Send prepared chunks after buffers are full
                await streamer.send()

            # Normal completion - flush and send remaining chunks
            streamer.flush()
            await streamer.send()
            if streamer.last_chunk_end_time_us is not None:
                last_end_us = streamer.last_chunk_end_time_us
        except asyncio.CancelledError:
            # Cancellation - flush and send remaining chunks before cleanup
            streamer.flush()
            await streamer.send()
            raise
        else:
            return last_end_us
        finally:
            # Always close all remaining active channels to prevent resource leaks
            for source in list(active_channels.values()):
                with suppress(Exception):
                    await source.aclose()
            active_channels.clear()

    def _reconfigure_streamer(self) -> None:
        """Reconfigure the running streamer with current client topology."""
        if (
            self._streamer is None
            or self._stream_commands is None
            or self._stream_task is None
            or self._media_stream is None
        ):
            raise RuntimeError("Streamer is not running")

        # Build configs for all current players
        all_player_configs: list[ClientStreamConfig] = []
        for player in self.players():
            assert player.support
            target_format = player.determine_optimal_format(self._media_stream.main_channel[1])
            all_player_configs.append(
                ClientStreamConfig(
                    client_id=player.client.client_id,
                    target_format=target_format,
                    buffer_capacity_bytes=player.support.buffer_capacity,
                    send=player.client.send_message,
                )
            )

        # Signal the streamer to reconfigure on next iteration with the new topology
        self._stream_commands.put_nowait(
            _StreamerReconfigureCommand(
                all_player_configs=all_player_configs,
            )
        )

    def suggest_optimal_sample_rate(self, source_sample_rate: int) -> int:
        """
        Suggest an optimal sample rate for the next track.

        Analyzes all player clients in this group and returns the best sample rate that
        minimizes resampling across group members. Preference order:
        - If there is a common supported rate across all players, choose the one closest
          to the source sample rate (tie-breaker: higher rate).
        - Otherwise, choose the rate supported by the most players; among those, pick the
          closest to the source (tie-breaker: higher rate).

        Args:
            source_sample_rate: The sample rate of the upcoming source media.

        Returns:
            The recommended sample rate in Hz.
        """
        supported_sets: list[set[int]] = [
            {fmt.sample_rate for fmt in client.info.player_support.supported_formats}
            for client in self._clients
            if client.check_role(Roles.PLAYER) and client.info.player_support
        ]

        if not supported_sets:
            return source_sample_rate

        # Helper for choosing the closest candidate, biasing towards higher rates on ties
        def choose(candidates: set[int]) -> int:
            # Compute the minimal absolute distance to the source sample rate
            best_distance = min(abs(r - source_sample_rate) for r in candidates)
            # Keep all candidates at that distance and pick the highest rate on a tie
            best_rates = [r for r in candidates if abs(r - source_sample_rate) == best_distance]
            return max(best_rates)

        # 1) Intersection across all players
        if (supported_sets) and (intersection := set.intersection(*supported_sets)):
            return choose(intersection)

        # 2) No common rate; pick the rate supported by the most players, then closest to source
        counts: dict[int, int] = {}
        for s in supported_sets:
            for r in s:
                counts[r] = counts.get(r, 0) + 1
        max_count = max(counts.values())
        top_rates = {r for r, c in counts.items() if c == max_count}
        return choose(top_rates)

    def _send_stream_start_msg(
        self,
        client: SendspinClient,
        player_stream_info: StreamStartPlayer | None = None,
    ) -> None:
        """Send a stream start message to a client with the specified audio format for players."""
        assert client.check_role(Roles.PLAYER) == (player_stream_info is not None)
        artwork_stream_info: StreamStartArtwork | None = None
        if client.check_role(Roles.ARTWORK) and client.info.artwork_support:
            # Initialize artwork state for all channels
            channels = client.info.artwork_support.channels
            if channels:
                client_channel_state = dict(enumerate(channels))
                self._client_artwork_state[client.client_id] = client_channel_state
                artwork_stream_info = _build_artwork_stream_info(client_channel_state)

        # TODO: finish once spec is finalized
        visualizer_stream_info = (
            StreamStartVisualizer() if client.check_role(Roles.VISUALIZER) else None
        )

        stream_info = StreamStartPayload(
            player=player_stream_info,
            artwork=artwork_stream_info,
            visualizer=visualizer_stream_info,
        )
        logger.debug(
            "Sending stream start message to client %s",
            client.client_id,
        )
        client.send_message(StreamStartMessage(stream_info))

    def _send_stream_end_msg(
        self, client: SendspinClient, roles: list[Roles] | None = None
    ) -> None:
        """Send a stream end message to a client.

        Args:
            client: The client to send the message to.
            roles: Optional list of roles to end streams for. If None, ends all streams.
        """
        logger.debug("ending stream for %s (%s), roles=%s", client.name, client.client_id, roles)
        # Lifetime of artwork state is bound to the stream
        if roles is None or Roles.ARTWORK in roles:
            self._client_artwork_state.pop(client.client_id, None)
        client.send_message(StreamEndMessage(payload=StreamEndPayload(roles=roles)))

    def _schedule_delayed_stop(self, stop_time_us: int, active: bool, needs_cleanup: bool) -> bool:  # noqa: FBT001
        """Schedule a delayed stop at the specified timestamp.

        Args:
            stop_time_us: Absolute timestamp when stop should occur
            active: Whether stream task is currently active
            needs_cleanup: Whether cleanup is needed

        Returns:
            True if stop was scheduled, False if nothing to do
        """
        now_us = int(self._server.loop.time() * 1_000_000)
        if stop_time_us <= now_us:
            return False

        # Only schedule if there's something to stop or cleanup
        if not active and not needs_cleanup:
            return False

        delay = (stop_time_us - now_us) / 1_000_000

        async def _delayed_stop() -> None:
            # Store handle locally to detect if it's been replaced
            handle = self._scheduled_stop_handle
            try:
                await self.stop()  # This will clear _scheduled_stop_handle
            except Exception:
                logger.exception("Scheduled stop failed")
            finally:
                # Only clear if this handle is still current (e.g., stop() was interrupted
                # or a new stop was scheduled during the stop() call)
                if self._scheduled_stop_handle is handle:
                    self._scheduled_stop_handle = None

        def _schedule_stop() -> None:
            task = self._server.loop.create_task(_delayed_stop())
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

        self._scheduled_stop_handle = self._server.loop.call_later(delay, _schedule_stop)
        return True

    async def _cancel_stream_task(self) -> None:
        """Cancel the active stream task and wait for it to complete."""
        if self._stream_task is None:
            return

        stream_task = self._stream_task
        stream_task.cancel()
        try:
            await stream_task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Unhandled exception while stopping stream task")
        # Only clear if it's still the same task (not replaced by a new play_media call)
        if self._stream_task is stream_task:
            self._stream_task = None

    async def _cleanup_streaming_resources(self) -> None:
        """Clean up all streaming-related resources."""
        if self._streamer is not None:
            self._streamer.reset()
            self._streamer = None

        if self._media_stream is not None:
            with suppress(Exception):
                await self._media_stream.main_channel[0].aclose()
        self._media_stream = None
        self._stream_commands = None

        for client in self._clients:
            if (
                client.check_role(Roles.PLAYER)
                or client.check_role(Roles.VISUALIZER)
                or client.check_role(Roles.ARTWORK)
            ):
                self._send_stream_end_msg(client)

        self._audio_encoders.clear()
        self._current_media_art.clear()
        self._play_start_time_us = None
        self._track_progress_timestamp_us = None

    def _send_stopped_state_to_clients(self) -> None:
        """Send stopped state to all clients."""
        group_message = GroupUpdateServerMessage(
            GroupUpdateServerPayload(
                playback_state=PlaybackStateType.STOPPED,
                group_id=self.group_id,
                group_name=self.group_name,
            )
        )
        supported_commands = self._get_supported_commands()
        controller_state = ControllerStatePayload(
            supported_commands=supported_commands,
            volume=self.volume,
            muted=self.muted,
        )
        # Update tracking variables
        self._last_sent_volume = self.volume
        self._last_sent_muted = self.muted
        self._last_sent_supported_commands = supported_commands

        for client in self._clients:
            # Send group/update to all clients
            client.send_message(group_message)

            # Send controller state to controller clients
            if client.check_role(Roles.CONTROLLER):
                state_message = ServerStateMessage(ServerStatePayload(controller=controller_state))
                client.send_message(state_message)

    async def stop(self, stop_time_us: int | None = None) -> bool:
        """
        Stop playback for the group and clean up resources.

        Compared to pause(), this also:
        - Cancels the audio streaming task
        - Sends stream end messages to all clients
        - Clears all buffers and format mappings
        - Cleans up all audio encoders

        Args:
            stop_time_us: Optional absolute timestamp (microseconds) when playback should
                stop. When provided and in the future, the stop request is scheduled and
                this method returns immediately.

        Returns:
            bool: True if an active stream was stopped (or scheduled to stop),
            False if no stream was active and no cleanup was required.
        """
        if len(self._clients) == 0:
            # An empty group cannot have active playback
            return False

        async with self._playback_lock:
            # Cancel any existing scheduled stop first to prevent race conditions
            if self._scheduled_stop_handle is not None:
                logger.debug("Canceling previously scheduled stop in stop()")
                self._scheduled_stop_handle.cancel()
                self._scheduled_stop_handle = None

            active = self._stream_task is not None
            needs_cleanup = self._current_state != PlaybackStateType.STOPPED

            # Handle delayed stop if requested
            if stop_time_us is not None and self._schedule_delayed_stop(
                stop_time_us, active, needs_cleanup
            ):
                return active or needs_cleanup

            if not active and not needs_cleanup:
                return False

            logger.debug(
                "Stopping playback for group with clients: %s",
                [c.client_id for c in self._clients],
            )

            # Capture resources to clean up before any await points
            # This prevents cleaning up resources from a new play_media call
            media_stream = self._media_stream

            try:
                await self._cancel_stream_task()
            finally:
                # Only clean up if resources haven't been replaced by a new play_media
                if self._media_stream is media_stream:
                    await self._cleanup_streaming_resources()

            if self._current_state != PlaybackStateType.STOPPED:
                self._signal_event(GroupStateChangedEvent(PlaybackStateType.STOPPED))
                self._current_state = PlaybackStateType.STOPPED

            self._send_stopped_state_to_clients()
            return True

    def _get_current_track_progress(self) -> int | None:
        """
        Calculate the current track progress in milliseconds.

        Returns the calculated progress based on playback time if actively playing,
        otherwise returns the stored progress value.
        """
        if self._current_metadata is None or self._current_metadata.track_progress is None:
            return None

        # If we have a stored timestamp and we're actively playing, calculate current progress
        if (
            self._track_progress_timestamp_us is not None
            and self.has_active_stream
            and self._current_metadata.playback_speed is not None
        ):
            current_time_us = int(self._server.loop.time() * 1_000_000)
            elapsed_us = current_time_us - self._track_progress_timestamp_us
            # playback_speed is stored as int * 1000 (e.g., 1000 = normal speed)
            # Convert elapsed microseconds to milliseconds, accounting for playback speed
            elapsed_ms = (elapsed_us * self._current_metadata.playback_speed) // 1_000_000
            calculated_progress = self._current_metadata.track_progress + elapsed_ms

            # Clamp to valid range
            # If track_duration is 0, it indicates unlimited/unknown duration (e.g., live streams)
            # In this case, only clamp to >= 0
            if (
                self._current_metadata.track_duration is not None
                and self._current_metadata.track_duration > 0
            ):
                # Normal track with known duration: clamp to [0, track_duration]
                calculated_progress = max(
                    0, min(calculated_progress, self._current_metadata.track_duration)
                )
            else:
                # Live stream (track_duration == 0) or unknown duration: only clamp to >= 0
                calculated_progress = max(0, calculated_progress)

            return calculated_progress

        # Otherwise return the stored value
        return self._current_metadata.track_progress

    def set_metadata(self, metadata: Metadata | None) -> None:
        """
        Set metadata for the group and send to all clients.

        Only sends updates for fields that have changed since the last call.

        Args:
            metadata: The new metadata to send to clients.
        """
        # TODO: integrate this more closely with play_media?
        timestamp = int(self._server.loop.time() * 1_000_000)

        if metadata is not None:
            if metadata.timestamp_us is None:
                metadata = replace(metadata, timestamp_us=timestamp)
            else:
                timestamp = metadata.timestamp_us

        if metadata is not None and metadata.equals(self._current_metadata):
            # No meaningful change, skip this update
            return
        last_metadata = self._current_metadata
        if metadata is None:
            # Clear all metadata fields when metadata is None
            metadata_update = Metadata.cleared_update(timestamp)
        else:
            # Only include fields that have changed since the last metadata update
            metadata_update = metadata.diff_update(last_metadata, timestamp)

        # Send server/state for metadata only to metadata clients
        for client in self._clients:
            if client.check_role(Roles.METADATA):
                state_message = ServerStateMessage(ServerStatePayload(metadata=metadata_update))
                logger.debug(
                    "Sending server state to client %s",
                    client.client_id,
                )
                client.send_message(state_message)

        # Update current metadata
        self._current_metadata = metadata

        # Store timestamp when track_progress is updated for progress calculation
        if metadata is not None and metadata.track_progress is not None:
            self._track_progress_timestamp_us = timestamp

    async def set_media_art(
        self, image: Image.Image | None, source: ArtworkSource = ArtworkSource.ALBUM
    ) -> None:
        """Set or clear artwork image for the current media.

        Args:
            image: The artwork image to set, or None to clear artwork for this source
            source: Source type (ALBUM or ARTIST), NONE is not allowed
        """
        if source == ArtworkSource.NONE:
            raise ValueError("Cannot set artwork with source NONE")

        if image is None:
            self._current_media_art.pop(source, None)
        else:
            self._current_media_art[source] = image

        # Gather all send tasks for matching channels
        send_tasks = []
        for client in self._clients:
            client_state = self._client_artwork_state.get(client.client_id)
            if client_state:
                for channel_num, channel_config in client_state.items():
                    if channel_config.source == source:
                        send_tasks.append(
                            self._send_media_art_to_client(client, image, channel_num)
                        )

        if send_tasks:
            await asyncio.gather(*send_tasks, return_exceptions=True)

    def _letterbox_image(
        self, image: Image.Image, target_width: int, target_height: int
    ) -> Image.Image:
        """
        Resize image to fit within target dimensions while preserving aspect ratio.

        Uses letterboxing (black bars) to fill any remaining space.

        Args:
            image: Source image to resize
            target_width: Target width in pixels
            target_height: Target height in pixels

        Returns:
            Resized image with letterboxing if needed
        """
        # Calculate aspect ratios
        image_aspect = image.width / image.height
        target_aspect = target_width / target_height

        if image_aspect > target_aspect:
            # Image is wider than target - fit by width, letterbox on top/bottom
            new_width = target_width
            new_height = int(target_width / image_aspect)
        else:
            # Image is taller than target - fit by height, letterbox on left/right
            new_height = target_height
            new_width = int(target_height * image_aspect)

        # Resize the image to the calculated size
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create a new image with the target size and black background
        letterboxed = Image.new("RGB", (target_width, target_height), (0, 0, 0))

        # Calculate position to center the resized image
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2

        # Paste the resized image onto the letterboxed background
        letterboxed.paste(resized, (x_offset, y_offset))

        return letterboxed

    async def _send_existing_artwork_to_clients(self) -> None:
        """Send any pre-existing artwork images to all artwork clients."""
        for client in self._clients:
            client_state = self._client_artwork_state.get(client.client_id)
            if client_state:
                send_tasks = []
                for channel_num, channel_config in client_state.items():
                    if channel_config.source == ArtworkSource.NONE:
                        continue
                    artwork = self._current_media_art.get(channel_config.source)
                    if artwork:
                        send_tasks.append(
                            self._send_media_art_to_client(client, artwork, channel_num)
                        )
                if send_tasks:
                    await asyncio.gather(*send_tasks, return_exceptions=True)

    async def _send_media_art_to_client(
        self, client: SendspinClient, image: Image.Image | None, channel: int
    ) -> None:
        """Send or clear media art to a specific client channel.

        Args:
            client: Client to send to
            image: Image to send, or None to clear artwork on this channel
            channel: Channel number (0-3)
        """
        if not client.check_role(Roles.ARTWORK):
            return

        client_state = self._client_artwork_state.get(client.client_id)
        if client_state is None:
            logger.warning(
                "Cannot send artwork to client %s channel %d: no active stream",
                client.client_id,
                channel,
            )
            return
        if channel not in client_state:
            logger.warning(
                "Cannot send artwork to client %s channel %d: channel not configured",
                client.client_id,
                channel,
            )
            return

        message_type = BinaryMessageType.ARTWORK_CHANNEL_0.value + channel
        header = pack_binary_header_raw(message_type, int(self._server.loop.time() * 1_000_000))

        if image is None:
            client.send_message(header)
        else:
            channel_state = client_state[channel]
            # Process and encode image in thread to avoid blocking event loop
            img_data = await asyncio.to_thread(
                self._process_and_encode_image,
                image,
                channel_state.media_width,
                channel_state.media_height,
                channel_state.format,
            )
            client.send_message(header + img_data)

    def _process_and_encode_image(
        self,
        image: Image.Image,
        width: int,
        height: int,
        art_format: PictureFormat,
    ) -> bytes:
        """
        Process and encode image for client.

        NOTE: This method is not async friendly.
        """
        # Use letterboxing to preserve aspect ratio
        resized_image = self._letterbox_image(image, width, height)

        with BytesIO() as img_bytes:
            if art_format == PictureFormat.JPEG:
                resized_image.save(img_bytes, format="JPEG", quality=85)
            elif art_format == PictureFormat.PNG:
                resized_image.save(img_bytes, format="PNG", compress_level=6)
            elif art_format == PictureFormat.BMP:
                resized_image.save(img_bytes, format="BMP")
            else:
                raise NotImplementedError(f"Unsupported artwork format: {art_format}")
            img_bytes.seek(0)
            return img_bytes.read()

    @property
    def clients(self) -> list[SendspinClient]:
        """All clients that are part of this group."""
        return self._clients

    @property
    def has_active_stream(self) -> bool:
        """Check if there is an active stream running."""
        return self._stream_task is not None

    def players(self) -> list[PlayerClient]:
        """Return player helpers for all members that support the role."""
        return [client.player for client in self._clients if client.player is not None]

    def _get_supported_commands(self) -> list[MediaCommand]:
        """Get list of commands supported based on application capabilities."""
        # Commands handled internally by this library (always supported)
        # TODO: differentiate between protocol and application supported commands?
        # Now it's not clear if MediaCommand.SWITCH or VOLUME needs to be handled by the app
        protocol_commands = [
            MediaCommand.VOLUME,
            MediaCommand.MUTE,
            MediaCommand.SWITCH,
        ]

        if self._supported_commands:
            # Return union of protocol commands and app-declared commands
            return list(set(protocol_commands) | set(self._supported_commands))

        # If app didn't declare any commands, only protocol commands are supported
        return protocol_commands

    def _handle_group_command(self, cmd: ControllerCommandPayload) -> None:
        # Handle volume and mute commands directly
        if cmd.command == MediaCommand.VOLUME and cmd.volume is not None:
            self.set_volume(cmd.volume)
            return
        if cmd.command == MediaCommand.MUTE and cmd.mute is not None:
            self.set_mute(cmd.mute)
            return

        # Signal the event for application commands (PLAY, PAUSE, STOP, etc.)
        event = GroupCommandEvent(
            command=cmd.command,
            volume=cmd.volume,
            mute=cmd.mute,
        )
        self._signal_event(event)

    def add_event_listener(
        self, callback: Callable[[SendspinGroup, GroupEvent], Coroutine[None, None, None]]
    ) -> Callable[[], None]:
        """
        Register a callback to listen for state changes of this group.

        State changes include:
        - The group started playing
        - The group stopped/finished playing

        Returns a function to remove the listener.
        """
        self._event_cbs.append(callback)

        def _remove() -> None:
            with suppress(ValueError):
                self._event_cbs.remove(callback)

        return _remove

    def _signal_event(self, event: GroupEvent) -> None:
        for cb in self._event_cbs:
            task = self._server.loop.create_task(cb(self, event))
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

    def _register_client_events(self, client: SendspinClient) -> None:
        """Register event listeners for client events like volume changes."""

        # Inline function to capture self
        async def on_client_event(_client: SendspinClient, event: ClientEvent) -> None:
            if isinstance(event, VolumeChangedEvent):
                # When any player's volume changes, update controller clients
                self._send_controller_state_to_clients()

        unsub = client.add_event_listener(on_client_event)
        self._client_event_unsubs[client] = unsub

    def _unregister_client_events(self, client: SendspinClient) -> None:
        """Unregister event listeners for a client."""
        if client in self._client_event_unsubs:
            self._client_event_unsubs[client]()
            del self._client_event_unsubs[client]

    @property
    def group_id(self) -> str:
        """Unique identifier for this group."""
        return self._group_id

    @property
    def group_name(self) -> str | None:
        """Friendly name for this group."""
        return self._group_name

    @property
    def state(self) -> PlaybackStateType:
        """Current playback state of the group."""
        return self._current_state

    @property
    def volume(self) -> int:
        """Current group volume (0-100), calculated as average of player volumes."""
        players = self.players()
        if not players:
            return 100
        # Calculate average volume from all players
        total_volume = sum(player.volume for player in players)
        return round(total_volume / len(players))

    @property
    def muted(self) -> bool:
        """Current group mute state - true only when ALL players are muted."""
        players = self.players()
        if not players:
            return False
        return all(player.muted for player in players)

    def set_volume(self, volume_level: int) -> None:
        """Set group volume using redistribution algorithm from spec."""
        volume_level = max(0, min(100, volume_level))
        players = self.players()
        if not players:
            return

        # Initialize working state with current volumes
        # We work entirely on this dict until the end
        player_volumes = {p: float(p.volume) for p in players}

        # Calculate initial target delta
        current_avg = sum(player_volumes.values()) / len(player_volumes)
        delta = volume_level - current_avg

        # Track who is still participating in redistribution
        active_players = list(players)

        for _ in range(5):
            # Apply delta to all active players and calculate lost delta (overflow)
            lost_delta_sum = 0.0
            next_active_players = []

            for player in active_players:
                current_vol = player_volumes[player]
                proposed = current_vol + delta

                # Clamp and calculate loss
                if proposed > 100:
                    clamped = 100.0
                    lost_delta_sum += proposed - clamped
                elif proposed < 0:
                    clamped = 0.0
                    lost_delta_sum += proposed - clamped
                else:
                    clamped = proposed
                    next_active_players.append(player)

                # Update our working state
                player_volumes[player] = clamped

            # If everyone is clamped or no delta lost, we are done
            if not next_active_players or abs(lost_delta_sum) < 0.01:
                break

            # Prepare for next iteration
            # Redistribute the lost delta among the remaining active players
            delta = lost_delta_sum / len(next_active_players)
            active_players = next_active_players

        # Apply final calculated volumes to the actual players
        for player, volume in player_volumes.items():
            player.set_volume(round(volume))

        # Send state update to controller clients
        self._send_controller_state_to_clients()

    def set_mute(self, muted: bool) -> None:  # noqa: FBT001
        """Set group mute state and propagate to all players."""
        # Propagate to all player clients
        for player in self.players():
            if muted:
                player.mute()
            else:
                player.unmute()
        # Send state update to controller clients
        self._send_controller_state_to_clients()

    def set_supported_commands(self, commands: list[MediaCommand]) -> None:
        """
        Set the media commands supported by the application.

        Args:
            commands: List of MediaCommand values that the application can handle.
                Empty list means no commands are supported.
        """
        self._supported_commands = commands
        self._send_controller_state_to_clients()

    async def remove_client(self, client: SendspinClient) -> None:
        """
        Remove a client from this group.

        If a stream is active, the client receives a stream end message.
        The client is automatically moved to its own new group since every
        client must belong to a group.
        If the client is not part of this group, this will have no effect.

        Args:
            client: The client to remove from this group.
        """
        if client not in self._clients:
            return
        logger.debug("removing %s from group with members: %s", client.client_id, self._clients)
        if len(self._clients) == 1:
            # Delete this group if that was the last client
            await self.stop()
            self._clients = []
        else:
            self._clients.remove(client)
            self._send_stream_end_msg(client)

            # Reconfigure streamer if actively streaming
            if (
                self._stream_task is not None
                and self._media_stream is not None
                and client.check_role(Roles.PLAYER)
            ):
                self._reconfigure_streamer()
        if not self._clients:
            # Emit event for group deletion, no clients left
            self._signal_event(GroupDeletedEvent())
        else:
            # Emit event for client removal
            self._signal_event(GroupMemberRemovedEvent(client.client_id))
        # Each client needs to be in a group, add it to a new one
        new_group = SendspinGroup(self._server, client)
        client._set_group(new_group)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        # Send group update to notify client of their new solo group
        new_group._send_group_update_to_clients()

    async def add_client(self, client: SendspinClient) -> None:
        """
        Add a client to this group.

        The client is first removed from any existing group. If a session is
        currently active, players are immediately joined to the session with
        an appropriate audio format.

        Args:
            client: The client to add to this group.
        """
        logger.debug("adding %s to group with members: %s", client.client_id, self._clients)
        await client.group.stop()
        if client in self._clients:
            return
        # Remove it from any existing group first
        await client.ungroup()

        # Add client to this group's client list
        self._clients.append(client)

        # Emit event for client addition
        self._signal_event(GroupMemberAddedEvent(client.client_id))

        # Then set the group (which will emit ClientGroupChangedEvent)
        client._set_group(self)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        if self._stream_task is not None and self._media_stream:
            logger.debug("Joining client %s to current stream", client.client_id)
            if client.check_role(Roles.PLAYER):
                # This will also send a stream start message
                self._reconfigure_streamer()
            elif client.check_role(Roles.VISUALIZER) or client.check_role(Roles.ARTWORK):
                self._send_stream_start_msg(client, None)

        # Send current state to the new client
        group_message = GroupUpdateServerMessage(
            GroupUpdateServerPayload(
                playback_state=self._current_state,
                group_id=self.group_id,
                group_name=self.group_name,
            )
        )
        logger.debug("Sending group update to new client %s", client.client_id)
        client.send_message(group_message)

        # Build server/state payload with relevant fields for this client
        metadata_for_client = None
        if client.check_role(Roles.METADATA):
            if self._current_metadata is not None:
                metadata_update = self._current_metadata.snapshot_update(
                    int(self._server.loop.time() * 1_000_000)
                )
                # Use calculated track progress for actively playing content
                current_progress = self._get_current_track_progress()
                # Update the progress object with current calculated progress
                if (
                    current_progress is not None
                    and self._current_metadata.track_duration is not None
                    and self._current_metadata.playback_speed is not None
                ):
                    metadata_update.progress = Progress(
                        track_progress=current_progress,
                        track_duration=self._current_metadata.track_duration,
                        playback_speed=self._current_metadata.playback_speed,
                    )
                metadata_for_client = metadata_update
            else:
                # Explicitly clear metadata for clients joining a group without existing metadata
                metadata_for_client = Metadata.cleared_update(
                    int(self._server.loop.time() * 1_000_000)
                )

        controller_for_client = None
        if client.check_role(Roles.CONTROLLER):
            controller_for_client = ControllerStatePayload(
                supported_commands=self._get_supported_commands(),
                volume=self.volume,
                muted=self.muted,
            )

        # Send single server/state message with all relevant payloads
        if metadata_for_client is not None or controller_for_client is not None:
            state_message = ServerStateMessage(
                ServerStatePayload(metadata=metadata_for_client, controller=controller_for_client)
            )
            logger.debug("Sending server state to new client %s", client.client_id)
            client.send_message(state_message)

        # Send current media art to the new client if available
        client_state = self._client_artwork_state.get(client.client_id)
        if client_state:
            send_tasks = []
            for channel_num, channel_config in client_state.items():
                if channel_config.source == ArtworkSource.NONE:
                    continue
                artwork = self._current_media_art.get(channel_config.source)
                if artwork:
                    send_tasks.append(self._send_media_art_to_client(client, artwork, channel_num))
            if send_tasks:
                await asyncio.gather(*send_tasks, return_exceptions=True)

    async def handle_stream_format_request(
        self,
        client: SendspinClient,
        request: StreamRequestFormatPayload,
    ) -> None:
        """Handle stream/request-format from a client and send stream/start."""
        if request.artwork:
            if not client.check_role(Roles.ARTWORK):
                raise ValueError(
                    f"Client {client.client_id} sent artwork format request "
                    "but does not have artwork role"
                )

            artwork_request = request.artwork

            if not client.info.artwork_support:
                raise ValueError(
                    f"Client {client.client_id} sent artwork format request "
                    "but has no artwork support"
                )

            client_state = self._client_artwork_state.get(client.client_id)
            if client_state is None:
                return

            if artwork_request.channel not in client_state:
                raise ValueError(
                    f"Invalid channel {artwork_request.channel} from client {client.client_id} "
                    f"(client declared {len(client.info.artwork_support.channels)} channels)"
                )

            current_channel = client_state[artwork_request.channel]

            updated_channel = ArtworkChannel(
                source=artwork_request.source
                if artwork_request.source is not None
                else current_channel.source,
                format=artwork_request.format
                if artwork_request.format is not None
                else current_channel.format,
                media_width=artwork_request.media_width
                if artwork_request.media_width is not None
                else current_channel.media_width,
                media_height=artwork_request.media_height
                if artwork_request.media_height is not None
                else current_channel.media_height,
            )

            client_state[artwork_request.channel] = updated_channel

            stream_start = StreamStartPayload(
                artwork=_build_artwork_stream_info(client_state),
            )

            logger.debug(
                "Sending stream/start to client %s for artwork format change on channel %d",
                client.client_id,
                artwork_request.channel,
            )
            client.send_message(StreamStartMessage(stream_start))

            if updated_channel.source != ArtworkSource.NONE:
                artwork = self._current_media_art.get(updated_channel.source)
                if artwork:
                    await self._send_media_art_to_client(client, artwork, artwork_request.channel)

        if request.player:
            if not client.check_role(Roles.PLAYER):
                raise ValueError(
                    f"Client {client.client_id} sent player format request "
                    "but does not have player role"
                )
            raise NotImplementedError("Player format changes are not yet implemented")
