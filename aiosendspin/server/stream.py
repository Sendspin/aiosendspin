"""High-level streaming pipeline primitives."""

from __future__ import annotations

import asyncio
import base64
import logging
from collections import deque
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from typing import NamedTuple, cast
from uuid import UUID, uuid4

import av
from av.logging import Capture

from aiosendspin.models import AudioCodec, BinaryMessageType, pack_binary_header_raw
from aiosendspin.models.player import StreamStartPlayer

logger = logging.getLogger(__name__)

# Universal main channel ID for the primary audio source.
# Used as the canonical source for visualization and as a fallback when
# player_channel() returns None.
MAIN_CHANNEL_ID: UUID = UUID("00000000-0000-0000-0000-000000000000")


@dataclass(frozen=True)
class AudioFormat:
    """Audio format of a stream."""

    sample_rate: int
    """Sample rate in Hz (e.g., 44100, 48000)."""
    bit_depth: int
    """Bit depth in bits per sample (16 or 24)."""
    channels: int
    """Number of audio channels (1 for mono, 2 for stereo)."""
    codec: AudioCodec = AudioCodec.PCM
    """Audio codec of the stream."""


@dataclass
class SourceChunk:
    """Raw PCM chunk received from the source."""

    pcm_data: bytes
    """Raw PCM audio data."""
    start_time_us: int
    """Absolute timestamp when this chunk starts playing."""
    end_time_us: int
    """Absolute timestamp when this chunk finishes playing."""
    sample_count: int
    """Number of audio samples in this chunk."""


class BufferedChunk(NamedTuple):
    """Buffered chunk metadata tracked by BufferTracker for backpressure control."""

    end_time_us: int
    """Absolute timestamp when these bytes should be fully consumed."""
    byte_count: int
    """Compressed byte count occupying the device buffer."""


class BufferTracker:
    """
    Track buffered compressed audio for a client and apply backpressure when needed.

    This class monitors the amount of compressed audio data buffered on a client device
    and ensures the server doesn't exceed the client's buffer capacity by applying
    backpressure when necessary.
    """

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        client_id: str,
        capacity_bytes: int,
    ) -> None:
        """
        Initialize the buffer tracker for a client.

        Args:
            loop: The event loop for timing calculations.
            client_id: Identifier for the client being tracked.
            capacity_bytes: Maximum buffer capacity in bytes reported by the client.
        """
        self._loop = loop
        self.client_id = client_id
        self.capacity_bytes = capacity_bytes
        self.buffered_chunks: deque[BufferedChunk] = deque()
        self.buffered_bytes = 0

    def prune_consumed(self, now_us: int | None = None) -> int:
        """Drop finished chunks and return the timestamp used for the calculation."""
        if now_us is None:
            now_us = int(self._loop.time() * 1_000_000)
        while self.buffered_chunks and self.buffered_chunks[0].end_time_us <= now_us:
            self.buffered_bytes -= self.buffered_chunks.popleft().byte_count
        self.buffered_bytes = max(self.buffered_bytes, 0)
        return now_us

    def has_capacity_now(self, bytes_needed: int) -> bool:
        """
        Check if buffer can accept bytes_needed without waiting.

        This is a non-blocking version of wait_for_capacity that returns immediately.

        Args:
            bytes_needed: Number of bytes to check capacity for.

        Returns:
            True if the buffer has capacity for bytes_needed, False otherwise.
        """
        if bytes_needed <= 0:
            return True
        if bytes_needed >= self.capacity_bytes:
            # Chunk exceeds capacity, but allow it through
            logger.warning(
                "Chunk size %s exceeds reported buffer capacity %s for client %s",
                bytes_needed,
                self.capacity_bytes,
                self.client_id,
            )
            return True

        self.prune_consumed()
        projected_usage = self.buffered_bytes + bytes_needed
        return projected_usage <= self.capacity_bytes

    def time_until_capacity(self, bytes_needed: int) -> int:
        """
        Calculate time in microseconds until the buffer can accept bytes_needed more bytes.

        Returns 0 if bytes_needed <= 0 (immediate capacity) or bytes_needed >= capacity_bytes
        (chunk exceeds capacity but is allowed through anyway).
        """
        if bytes_needed <= 0:
            return 0
        if bytes_needed >= self.capacity_bytes:
            # TODO: raise exception instead?
            logger.warning(
                "Chunk size %s exceeds reported buffer capacity %s for client %s",
                bytes_needed,
                self.capacity_bytes,
                self.client_id,
            )
            return 0

        # Prune consumed chunks once at the start
        cursor_time_us = self.prune_consumed()
        time_needed_us = 0

        # Simulate state without modifying it to find when capacity is available
        virtual_buffered_bytes = self.buffered_bytes
        cursor_index = 0

        while cursor_index < len(self.buffered_chunks):
            projected_usage = virtual_buffered_bytes + bytes_needed
            if projected_usage <= self.capacity_bytes:
                # We have enough capacity at this point
                break

            chunk = self.buffered_chunks[cursor_index]
            cursor_end_time_us = chunk.end_time_us
            time_needed_us += max(cursor_end_time_us - cursor_time_us, 0)

            # Advance cursor to the next chunk
            cursor_index += 1
            cursor_time_us = cursor_end_time_us
            virtual_buffered_bytes -= chunk.byte_count
        return time_needed_us

    async def wait_for_capacity(self, bytes_needed: int) -> None:
        """Block until the device buffer can accept bytes_needed more bytes."""
        if sleep_time_us := self.time_until_capacity(bytes_needed):
            await asyncio.sleep(sleep_time_us / 1_000_000)

    def register(self, end_time_us: int, byte_count: int) -> None:
        """Record bytes added to the buffer finishing at end_time_us."""
        if byte_count <= 0:
            return
        self.buffered_chunks.append(BufferedChunk(end_time_us, byte_count))
        self.buffered_bytes += byte_count


def _resolve_audio_format(audio_format: AudioFormat) -> tuple[int, str, str]:
    """Resolve helper data for an audio format."""
    if audio_format.bit_depth == 16:
        bytes_per_sample = 2
        av_format = "s16"
    elif audio_format.bit_depth == 24:
        bytes_per_sample = 3
        av_format = "s24"
    else:
        raise ValueError("Only 16-bit and 24-bit PCM are supported")

    if audio_format.channels == 1:
        layout = "mono"
    elif audio_format.channels == 2:
        layout = "stereo"
    else:
        raise ValueError("Only mono and stereo layouts are supported")

    return bytes_per_sample, av_format, layout


def build_encoder_for_format(
    audio_format: AudioFormat,
    *,
    input_audio_layout: str,
    input_audio_format: str,
) -> tuple[av.AudioCodecContext | None, str | None, int]:
    """Create and configure an encoder for the target audio format."""
    if audio_format.codec == AudioCodec.PCM:
        samples_per_chunk = int(audio_format.sample_rate * 0.025)
        return None, None, samples_per_chunk

    codec = "libopus" if audio_format.codec == AudioCodec.OPUS else audio_format.codec.value

    encoder = cast("av.AudioCodecContext", av.AudioCodecContext.create(codec, "w"))
    encoder.sample_rate = audio_format.sample_rate
    encoder.layout = input_audio_layout
    encoder.format = input_audio_format
    if audio_format.codec == AudioCodec.FLAC:
        encoder.options = {"compression_level": "5"}

    with Capture() as logs:
        encoder.open()
    for log in logs:
        logger.debug("Opening AudioCodecContext log from av: %s", log)

    header = bytes(encoder.extradata) if encoder.extradata else b""
    if audio_format.codec == AudioCodec.FLAC and header:
        # For FLAC, we need to construct a proper FLAC stream header ourselves
        # since ffmpeg only provides the StreamInfo metadata block in extradata:
        # See https://datatracker.ietf.org/doc/rfc9639/ Section 8.1

        # FLAC stream signature (4 bytes): "fLaC"
        # Metadata block header (4 bytes):
        # - Bit 0: last metadata block (1 since we only have one)
        # - Bits 1-7: block type (0 for StreamInfo)
        # - Next 3 bytes: block length of the next metadata block in bytes
        # StreamInfo block (34 bytes): as provided by ffmpeg
        header = b"fLaC\x80" + len(header).to_bytes(3, "big") + header

    codec_header_b64 = base64.b64encode(header).decode()

    # Calculate samples per chunk
    if audio_format.codec == AudioCodec.FLAC:
        # FLAC: Use 25ms chunks regardless of encoder frame_size
        samples_per_chunk = int(audio_format.sample_rate * 0.025)
    elif encoder.frame_size and encoder.frame_size > 0:
        # Use recommended frame size for other codecs (e.g., OPUS)
        samples_per_chunk = int(encoder.frame_size)
    else:
        raise ValueError(
            f"Codec {audio_format.codec.value} encoder has invalid frame_size: {encoder.frame_size}"
        )
    return encoder, codec_header_b64, samples_per_chunk


@dataclass(frozen=True)
class AudioFormatParams:
    """Audio format parameters with computed PyAV values for processing."""

    audio_format: AudioFormat
    """Source audio format."""
    bytes_per_sample: int
    """Bytes per sample (derived from bit depth)."""
    frame_stride: int
    """Bytes per frame (bytes_per_sample * channels)."""
    av_format: str
    """PyAV format string (e.g., 's16', 's24')."""
    av_layout: str
    """PyAV channel layout (e.g., 'mono', 'stereo')."""


@dataclass
class ClientStreamConfig:
    """Configuration for delivering audio to a player."""

    client_id: str
    """Unique client identifier."""
    target_format: AudioFormat
    """Target audio format for this client."""
    buffer_capacity_bytes: int
    """Client's buffer capacity in bytes."""
    send: Callable[[bytes], None]
    """Function to send data to client."""


@dataclass
class PreparedChunkState:
    """Prepared chunk shared between all subscribers of a pipeline."""

    data: bytes
    """Prepared/encoded audio data."""
    start_time_us: int
    """Chunk playback start time in microseconds."""
    end_time_us: int
    """Chunk playback end time in microseconds."""
    sample_count: int
    """Number of samples in this chunk."""
    byte_count: int
    """Size of chunk data in bytes."""


@dataclass
class PipelineState:
    """Holds state for a pipeline of a channel/format/chunk-size/encoding combination."""

    source_format_params: AudioFormatParams
    """Source audio format parameters."""
    channel_id: UUID
    """Channel this pipeline consumes from."""
    target_format: AudioFormat
    """Target output format."""
    target_frame_stride: int
    """Target bytes per frame."""
    target_av_format: str
    """Target PyAV format string."""
    target_layout: str
    """Target PyAV channel layout."""
    chunk_samples: int
    """Target samples per chunk."""
    resampler: av.AudioResampler
    """PyAV audio resampler."""
    encoder: av.AudioCodecContext | None
    """PyAV encoder (None for PCM)."""
    codec_header_b64: str | None
    """Base64 encoded codec header."""
    buffer: bytearray = field(default_factory=bytearray)
    """Resampled PCM buffer awaiting encoding."""
    prepared: deque[PreparedChunkState] = field(default_factory=deque)
    """Prepared chunks ready for delivery."""
    subscribers: list[str] = field(default_factory=list)
    """Client IDs subscribed to this pipeline."""
    samples_produced: int = 0
    """Total samples published from this pipeline."""
    flushed: bool = False
    """Whether pipeline has been flushed."""
    source_read_position: int = 0
    """Read position in this pipeline's source channel buffer."""
    next_chunk_start_us: int | None = None
    """Next output chunk start timestamp, initialized from first source chunk."""


@dataclass
class ChannelState:
    """State for a single time-synchronized audio channel."""

    source_format_params: AudioFormatParams
    """Audio format parameters for this channel."""
    source_buffer: deque[SourceChunk] = field(default_factory=deque)
    """Buffer of raw PCM chunks scheduled for playback."""
    samples_produced: int = 0
    """Total samples added to this channel's buffer."""


@dataclass
class PlayerState:
    """Tracks delivery state for a single player."""

    config: ClientStreamConfig
    """Client streaming configuration."""
    audio_format: AudioFormat
    """Format key for pipeline lookup."""
    channel_id: UUID
    """Channel this player consumes from."""
    queue: deque[PreparedChunkState] = field(default_factory=deque)
    """Chunks queued for delivery."""
    buffer_tracker: BufferTracker | None = None
    """Tracks client buffer state."""
    join_wall_time_us: int | None = None
    """Wall-clock time when player joined (for grace period tracking)."""


class MediaStream:
    """
    Container for audio stream with optional per-device DSP support.

    Provides a main audio source used for visualization and playback. Optionally,
    implementations can override player_channel() to provide device-specific channels
    for individual DSP processing chains. If player_channel returns None, the main
    channel is used as fallback.
    """

    _main_channel_source: AsyncGenerator[bytes, None]
    """
    Main audio source generator yielding PCM bytes.

    Used for visualization, and as fallback when player_channel() returns None.
    """
    _main_channel_format: AudioFormat
    """Audio format of the main_channel()."""

    def __init__(
        self,
        *,
        main_channel_source: AsyncGenerator[bytes, None],
        main_channel_format: AudioFormat,
    ) -> None:
        """Initialise the media stream with audio source and format for main_channel()."""
        self._main_channel_source = main_channel_source
        self._main_channel_format = main_channel_format

    @property
    def main_channel(self) -> tuple[AsyncGenerator[bytes, None], AudioFormat]:
        """Return the main audio source generator and its audio format."""
        return self._main_channel_source, self._main_channel_format

    async def player_channel(
        self,
        player_id: str,
        preferred_format: AudioFormat | None = None,
        position_us: int = 0,
    ) -> tuple[AsyncGenerator[bytes, None], AudioFormat, int] | None:
        """
        Get a player-specific audio channel (time-synchronized with main channel).

        The returned audio stream should ideally start at position_us relative to
        the main channel's start. But implementations can return any position if
        they can only start at specific boundaries (e.g., codec frame sizes, internal buffers).
        If the returned stream and actual_position_us are equal, they will play in perfect sync.

        Args:
            player_id: Identifier for the player requesting the channel.
            preferred_format: The player's preferred native format.
            position_us: Requested position in microseconds relative to main_channel start.

        Returns:
            Tuple of (generator, format, actual_position_us) or None for fallback.
            The actual_position_us may differ from requested position_us if the
            implementation can only provide channels at specific boundaries.
        """
        _ = (player_id, preferred_format, position_us)
        return None


class Streamer:
    """Adapts incoming channel data to player-specific formats."""

    _loop: asyncio.AbstractEventLoop
    """Event loop used for time calculations and task scheduling."""
    _play_start_time_us: int
    """Playback start time in microseconds, may be adjusted forward to prevent chunk skipping."""
    _channels: dict[UUID, ChannelState]
    """Mapping of channel IDs to their state."""
    _pipelines: dict[tuple[UUID, AudioFormat], PipelineState]
    """Mapping of (channel_id, target_format) to pipeline state."""
    _players: dict[str, PlayerState]
    """Mapping of client IDs to their player delivery state."""
    _last_chunk_end_us: int | None = None
    """End timestamp of the most recently prepared chunk, None if no chunks prepared yet."""
    _source_buffer_target_duration_us: int = 5_000_000
    """Target duration for source buffer in microseconds."""
    _prepare_buffer_margin_us: int = 2_500_000
    """Time margin for stale chunk detection during prepare() (2.5 seconds)."""

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        play_start_time_us: int,
    ) -> None:
        """Create a streamer bound to the event loop and playback start time."""
        self._loop = loop
        self._play_start_time_us = play_start_time_us
        self._channels = {}
        self._pipelines = {}
        self._players = {}

    def _now_us(self) -> int:
        """Get current time in microseconds."""
        return int(self._loop.time() * 1_000_000)

    def _create_or_update_player(
        self,
        client_cfg: ClientStreamConfig,
        pipeline: PipelineState,
        channel_id: UUID,
        audio_format: AudioFormat,
    ) -> PlayerState | None:
        """Create new player or update existing one, handling format changes.

        Returns:
            PlayerState if a new player was created, None if existing player was reused.
        """
        old_player = self._players.get(client_cfg.client_id)

        # Reuse existing player if format unchanged
        if old_player and old_player.audio_format == audio_format:
            old_player.config = client_cfg
            return None  # Signal that existing player should be reused

        # Format changed - clean up old queue
        if old_player and old_player.audio_format != audio_format:
            old_player.queue.clear()

        # Create new player or reconfigure existing one
        buffer_tracker = (
            old_player.buffer_tracker
            if old_player
            else BufferTracker(
                loop=self._loop,
                client_id=client_cfg.client_id,
                capacity_bytes=client_cfg.buffer_capacity_bytes,
            )
        )

        # Find synchronized join point based on SOURCE CHANNEL
        # Use the earliest prepared chunk across ALL pipelines on this channel
        # This ensures late joiners start from the beginning of available audio,
        # not from where existing players' queues currently are (which would skip
        # already-sent chunks that are still playing on those clients)
        sync_point_start_time_us: int | None = None

        # Check all pipelines consuming from this channel for their earliest prepared chunk
        for (pipe_channel_id, _), pipe in self._pipelines.items():
            if pipe_channel_id == channel_id and pipe.prepared:
                chunk_start = pipe.prepared[0].start_time_us
                if sync_point_start_time_us is None or chunk_start < sync_point_start_time_us:
                    sync_point_start_time_us = chunk_start

        # Fallback to play_start_time_us if no prepared chunks exist yet (initial startup)
        # This ensures chunks timestamped relative to play_start_time_us will be queued
        if sync_point_start_time_us is None:
            sync_point_start_time_us = self._play_start_time_us

        player_state = PlayerState(
            config=client_cfg,
            audio_format=audio_format,
            channel_id=channel_id,
            buffer_tracker=buffer_tracker,
            join_wall_time_us=self._now_us(),
        )

        # Queue chunks starting from the sync point
        for chunk in pipeline.prepared:
            if chunk.start_time_us >= sync_point_start_time_us:
                player_state.queue.append(chunk)

        return player_state

    async def _query_player_channel(
        self,
        player_id: str,
        player_config: ClientStreamConfig,
        media_stream: MediaStream,
        channel_formats: dict[UUID, AudioFormat],
        new_channel_sources: dict[UUID, AsyncGenerator[bytes, None]],
        player_channel_assignments: dict[str, UUID],
        channel_initial_samples: dict[UUID, int],
    ) -> None:
        """Query player channel from MediaStream and update topology dictionaries.

        Args:
            player_id: ID of the player to query.
            player_config: Configuration for the player.
            media_stream: Media stream to query for player channel.
            channel_formats: Dict to update with channel formats.
            new_channel_sources: Dict to update with channel sources.
            player_channel_assignments: Dict to update with player assignments.
            channel_initial_samples: Dict to update with initial sample counts.
        """
        # Try to get player-specific channel with error handling
        try:
            player_channel_result = await media_stream.player_channel(
                player_id=player_id,
                preferred_format=player_config.target_format,
                position_us=self._now_us() - self._play_start_time_us,
            )
        except Exception:
            logger.exception(
                "Failed to query player_channel for %s, falling back to main channel",
                player_id,
            )
            player_channel_result = None

        if player_channel_result is not None:
            source, channel_format, actual_pos_us = player_channel_result
            channel_id = uuid4()

            # Add new channel
            channel_formats[channel_id] = channel_format
            new_channel_sources[channel_id] = source
            player_channel_assignments[player_id] = channel_id

            # Calculate and store position offset
            initial_samples = round(actual_pos_us * channel_format.sample_rate / 1_000_000)
            channel_initial_samples[channel_id] = initial_samples

            # Calculate when the first chunk from this channel will play
            first_chunk_start_us = self._play_start_time_us + int(
                initial_samples * 1_000_000 / channel_format.sample_rate
            )
            now_us = self._now_us()
            delay_s = (first_chunk_start_us - now_us) / 1_000_000

            logger.info(
                "Player %s assigned to dedicated channel with offset %d us (%.3f s). "
                "First chunk will play in %.3f seconds from now.",
                player_id,
                actual_pos_us,
                actual_pos_us / 1_000_000,
                delay_s,
            )
        else:
            # Fallback to main channel
            player_channel_assignments[player_id] = MAIN_CHANNEL_ID
            logger.info("Player %s assigned to main channel", player_id)

    def _get_or_create_pipeline(self, channel_id: UUID, audio_format: AudioFormat) -> PipelineState:
        """Get existing pipeline or create new one for channel/format combination."""
        pipeline_key = (channel_id, audio_format)
        pipeline = self._pipelines.get(pipeline_key)
        if pipeline is not None:
            return pipeline

        # Create new pipeline for this channel/format
        channel_state = self._channels[channel_id]
        source_format_params = channel_state.source_format_params

        target_bytes_per_sample, target_av_format, target_layout = _resolve_audio_format(
            audio_format
        )

        resampler = av.AudioResampler(
            format=target_av_format,
            layout=target_layout,
            rate=audio_format.sample_rate,
        )
        encoder, codec_header_b64, chunk_samples = build_encoder_for_format(
            audio_format,
            input_audio_layout=target_layout,
            input_audio_format=target_av_format,
        )
        pipeline = PipelineState(
            source_format_params=source_format_params,
            channel_id=channel_id,
            target_format=audio_format,
            target_frame_stride=target_bytes_per_sample * audio_format.channels,
            target_av_format=target_av_format,
            target_layout=target_layout,
            chunk_samples=chunk_samples,
            resampler=resampler,
            encoder=encoder,
            codec_header_b64=codec_header_b64,
        )
        self._pipelines[pipeline_key] = pipeline
        return pipeline

    def _build_start_payload(
        self, client_cfg: ClientStreamConfig, pipeline: PipelineState
    ) -> StreamStartPlayer:
        """Build StreamStartPlayer message for client."""
        return StreamStartPlayer(
            codec=client_cfg.target_format.codec,
            sample_rate=client_cfg.target_format.sample_rate,
            channels=client_cfg.target_format.channels,
            bit_depth=client_cfg.target_format.bit_depth,
            codec_header=pipeline.codec_header_b64,
        )

    async def configure(
        self,
        all_player_configs: list[ClientStreamConfig],
        media_stream: MediaStream,
    ) -> tuple[dict[str, StreamStartPlayer], dict[UUID, AsyncGenerator[bytes, None]]]:
        """
        Configure or reconfigure pipelines for the provided players.

        Resolves topology (which players get which channels) by querying MediaStream
        for player-specific channels and calculating synchronization offsets.

        Args:
            all_player_configs: List of ClientStreamConfig for all players (existing and new).
            media_stream: Media stream providing audio sources and player channels.

        Returns:
            Tuple of (start_payloads, new_channel_sources):
            - start_payloads: Dict mapping client IDs to StreamStartPlayer messages
            - new_channel_sources: Dict mapping channel IDs to source generators for new channels
        """
        # RESOLVE TOPOLOGY
        # Build topology: determine which players are new, query channels
        main_source, main_format = media_stream.main_channel
        channel_formats: dict[UUID, AudioFormat] = {MAIN_CHANNEL_ID: main_format}
        new_channel_sources: dict[UUID, AsyncGenerator[bytes, None]] = {
            MAIN_CHANNEL_ID: main_source
        }
        player_channel_assignments: dict[str, UUID] = {}
        channel_initial_samples: dict[UUID, int] = {MAIN_CHANNEL_ID: 0}

        # Calculate which players are new by comparing with existing players
        new_config_player_ids = {cfg.client_id for cfg in all_player_configs}
        new_player_ids = new_config_player_ids - set(self._players.keys())

        # Preserve existing channel assignments and add their formats to channel_formats
        # Only preserve for players that are still in the new configuration
        for player_id, player_state in self._players.items():
            if player_id not in new_config_player_ids:
                continue  # Skip players being removed
            player_channel_assignments[player_id] = (channel_id := player_state.channel_id)
            if channel_id not in channel_formats and (
                channel_state := self._channels.get(channel_id)
            ):
                channel_formats[channel_id] = channel_state.source_format_params.audio_format

        # Query channels for new players
        for player_id in new_player_ids:
            if not (
                player_config := next(
                    (c for c in all_player_configs if c.client_id == player_id), None
                )
            ):
                logger.warning("Config not found for new player %s", player_id)
                player_channel_assignments[player_id] = MAIN_CHANNEL_ID
                continue
            await self._query_player_channel(
                player_id,
                player_config,
                media_stream,
                channel_formats,
                new_channel_sources,
                player_channel_assignments,
                channel_initial_samples,
            )

        # APPLY TOPOLOGY TO INTERNAL STATE
        # Update or create channel states
        for channel_id, audio_format in channel_formats.items():
            if channel_id not in self._channels:
                bytes_per_sample, av_format, av_layout = _resolve_audio_format(audio_format)
                self._channels[channel_id] = ChannelState(
                    source_format_params=AudioFormatParams(
                        audio_format=audio_format,
                        bytes_per_sample=bytes_per_sample,
                        frame_stride=bytes_per_sample * audio_format.channels,
                        av_format=av_format,
                        av_layout=av_layout,
                    ),
                )
                if initial_sample_count := channel_initial_samples.get(channel_id):
                    self._channels[channel_id].samples_produced = initial_sample_count

        # Remove channels that are no longer needed
        for channel_id in set(self._channels) - set(channel_formats):
            self._channels.pop(channel_id)

        # Clear subscriber lists to rebuild them
        for existing_pipeline in self._pipelines.values():
            existing_pipeline.subscribers.clear()

        # Build new player and subscription configuration
        new_players: dict[str, PlayerState] = {}
        start_payloads: dict[str, StreamStartPlayer] = {}

        for client_cfg in all_player_configs:
            channel_id = player_channel_assignments[client_cfg.client_id]
            pipeline = self._get_or_create_pipeline(channel_id, client_cfg.target_format)
            pipeline.subscribers.append(client_cfg.client_id)

            # Create or update player state
            if (
                result := self._create_or_update_player(
                    client_cfg, pipeline, channel_id, client_cfg.target_format
                )
            ) is None:
                new_players[client_cfg.client_id] = self._players[client_cfg.client_id]
                continue

            new_players[client_cfg.client_id] = result
            start_payloads[client_cfg.client_id] = self._build_start_payload(client_cfg, pipeline)

        # Remove pipelines with no subscribers
        for key in [k for k, p in self._pipelines.items() if not p.subscribers]:
            if (pipeline := self._pipelines.pop(key)).encoder:
                pipeline.encoder = None

        # Clean up queues for players being removed
        for old_client_id, old_player in self._players.items():
            if old_client_id not in new_players:
                old_player.queue.clear()

        # Replace players dict
        self._players = new_players

        return start_payloads, new_channel_sources

    def _channel_wait_time_us(self, channel_state: ChannelState, now_us: int) -> int:
        """
        Calculate time in microseconds until a channel needs data.

        Args:
            channel_state: Channel state to check.
            now_us: Current time in microseconds.

        Returns:
            Wait time in microseconds (0 if immediate).
        """
        if not channel_state.source_buffer:
            return 0

        # Calculate when buffer will drop below target duration from now
        buffer_end = channel_state.source_buffer[-1].end_time_us
        return max(0, buffer_end - self._source_buffer_target_duration_us - now_us)

    def channel_needs_data(self, channel_id: UUID) -> bool:
        """
        Check if a channel's buffer is below the target duration from now.

        Args:
            channel_id: ID of the channel to check.

        Returns:
            True if buffer depth from now < target, False otherwise.
        """
        channel_state = self._channels.get(channel_id)
        if channel_state is None:
            raise ValueError(f"Channel {channel_id} not found")
        if not channel_state.source_buffer:
            return True

        # Check if wait time is zero, means that it needs data now
        return self._channel_wait_time_us(channel_state, self._now_us()) == 0

    def _get_earliest_channel_wait_time_us(self) -> int | None:
        """
        Calculate the earliest wait time across all channels.

        Returns:
            Earliest wait time in microseconds, or None if any channel is empty.
        """
        now_us = self._now_us()
        earliest_wait_us = None

        for channel in self._channels.values():
            wait_us = self._channel_wait_time_us(channel, now_us)
            if earliest_wait_us is None or wait_us < earliest_wait_us:
                earliest_wait_us = wait_us

        return earliest_wait_us

    def prepare(
        self, channel_id: UUID, chunk: bytes, *, during_initial_buffering: bool = False
    ) -> None:
        """
        Buffer raw PCM data and process through pipelines.

        Args:
            channel_id: ID of the channel this chunk belongs to.
            chunk: Raw PCM audio data to buffer.
            during_initial_buffering: True when filling initial buffer on startup,
                which skips building full 5-second buffer during timing adjustments.
        """
        channel_state = self._channels[channel_id]
        if len(chunk) % channel_state.source_format_params.frame_stride:
            raise ValueError("Chunk must be aligned to whole samples")
        sample_count = len(chunk) // channel_state.source_format_params.frame_stride
        if sample_count == 0:
            return

        # Calculate timestamps for this chunk
        start_samples = channel_state.samples_produced

        # Check and adjust for stale chunks (skip during initial buffering)
        if not during_initial_buffering:
            start_us, end_us = self._check_and_adjust_for_stale_chunk(
                channel_state, start_samples, sample_count
            )
        else:
            # During initial buffering, just calculate timestamps without stale detection
            start_us = self._play_start_time_us + int(
                start_samples
                * 1_000_000
                / channel_state.source_format_params.audio_format.sample_rate
            )
            end_us = self._play_start_time_us + int(
                (start_samples + sample_count)
                * 1_000_000
                / channel_state.source_format_params.audio_format.sample_rate
            )

        # Create and buffer the source chunk
        source_chunk = SourceChunk(
            pcm_data=chunk,
            start_time_us=start_us,
            end_time_us=end_us,
            sample_count=sample_count,
        )
        channel_state.source_buffer.append(source_chunk)
        channel_state.samples_produced += sample_count

        # Process through pipelines that consume from this channel
        for pipeline in self._pipelines.values():
            if pipeline.channel_id == channel_id:
                self._process_pipeline_from_source(pipeline, channel_state)

    def _check_and_adjust_for_stale_chunk(
        self,
        channel_state: ChannelState,
        start_samples: int,
        sample_count: int,
    ) -> tuple[int, int]:
        """
        Check if the next chunk would be stale and adjust timing if needed.

        Args:
            channel_state: The channel state for this chunk.
            start_samples: Sample position where the chunk starts.
            sample_count: Number of samples in the chunk.

        Returns:
            Tuple of (start_us, end_us) timestamps after any adjustments.
        """
        # Calculate initial timestamps
        start_us = self._play_start_time_us + int(
            start_samples * 1_000_000 / channel_state.source_format_params.audio_format.sample_rate
        )

        # Check if this chunk would be stale
        now_us = self._now_us()

        if start_us < now_us + self._prepare_buffer_margin_us:
            # Adjust timing globally (checks all channels)
            self._adjust_timing_for_stale_chunk(now_us, start_us)
            # Recalculate timestamps after adjustment
            start_us = self._play_start_time_us + int(
                start_samples
                * 1_000_000
                / channel_state.source_format_params.audio_format.sample_rate
            )

        end_us = self._play_start_time_us + int(
            (start_samples + sample_count)
            * 1_000_000
            / channel_state.source_format_params.audio_format.sample_rate
        )

        return start_us, end_us

    def _adjust_timing_for_stale_chunk(self, now_us: int, chunk_start_us: int) -> None:
        """
        Adjust timing when a stale chunk is detected.

        Checks all channels and uses minimum buffer depth for conservative adjustment
        that keeps all channels in sync.

        Args:
            now_us: Current time in microseconds.
            chunk_start_us: Start time of the stale chunk.
        """
        target_buffer_us = self._source_buffer_target_duration_us

        # Calculate minimum buffer depth across all channel source buffers
        min_buffer_us = None
        for channel_state in self._channels.values():
            if channel_state.source_buffer:
                last_chunk_end = channel_state.source_buffer[-1].end_time_us
                current_buffer_us = max(0, last_chunk_end - now_us)
                if min_buffer_us is None or current_buffer_us < min_buffer_us:
                    min_buffer_us = current_buffer_us

        current_buffer_us = min_buffer_us if min_buffer_us is not None else 0

        # Calculate minimum adjustment needed to give this chunk proper headroom
        headroom_shortfall_us = (now_us + self._prepare_buffer_margin_us) - chunk_start_us

        # Determine total adjustment based on buffer status
        if current_buffer_us >= target_buffer_us:
            # We already have enough buffer, just ensure headroom
            timing_adjustment_us = headroom_shortfall_us
        else:
            # Need to build buffer to target level
            buffer_shortfall_us = target_buffer_us - current_buffer_us
            # Use the larger of headroom need and buffer need
            timing_adjustment_us = max(headroom_shortfall_us, buffer_shortfall_us)

        logger.info("Detected slow source, adjusting timing to prevent skipping.")
        logger.debug(
            "Adjusting timing: needs %.3fs headroom, have %.3fs buffer (adjusting %.3fs)",
            headroom_shortfall_us / 1_000_000,
            current_buffer_us / 1_000_000,
            timing_adjustment_us / 1_000_000,
        )

        # Adjust global timing forward
        self._play_start_time_us += timing_adjustment_us

        # Update source buffers chunk timestamps
        for ch_state in self._channels.values():
            for source_chunk in ch_state.source_buffer:
                source_chunk.start_time_us += timing_adjustment_us
                source_chunk.end_time_us += timing_adjustment_us

        # Update pipelines buffer and prepared chunks timestamps
        for pipeline in self._pipelines.values():
            if pipeline.next_chunk_start_us is not None:
                pipeline.next_chunk_start_us += timing_adjustment_us
            # Update timestamps of already-prepared chunks to prevent cascading adjustments
            for prepared_chunk in pipeline.prepared:
                prepared_chunk.start_time_us += timing_adjustment_us
                prepared_chunk.end_time_us += timing_adjustment_us

        # Update chunks in player queues that are no longer in pipeline.prepared
        # Track by object id() to avoid double-updating the same chunk object
        updated_chunks: set[int] = {id(c) for p in self._pipelines.values() for c in p.prepared}
        for player_state in self._players.values():
            for queued_chunk in player_state.queue:
                if id(queued_chunk) not in updated_chunks:
                    queued_chunk.start_time_us += timing_adjustment_us
                    queued_chunk.end_time_us += timing_adjustment_us
                    updated_chunks.add(id(queued_chunk))

    async def send(self) -> None:
        """
        Send prepared audio to all clients with perfect group synchronization.

        This method performs stages in a loop:
        1. Check for stale chunks and adjust timing globally if needed (prevents skipping)
        2. Send chunks to players with backpressure control (per-player throughput)
        3. Prune old data
        4. Check exit conditions and apply source buffer backpressure

        Global timing adjustments prevent chunk skipping, ensuring all players
        receive identical audio content for perfect synchronization. Players can
        have different queue depths (due to buffer capacity differences), but all
        receive the same chunks with the same timestamps.

        Continues until all pending audio has been delivered and source buffer is below target.
        """
        while True:
            # Stage 1: Check for stale chunks across all channels
            # Newly joined players are excluded via grace period to avoid false positives
            # from sync-point chunks with past timestamps
            now_us = self._now_us()
            send_transmission_margin_us = 100_000  # 100ms for network + client processing
            join_grace_period_us = 2_000_000  # 2s grace period for newly joined players

            # Find the earliest chunk across all channels, excluding newly joined players
            # Newly joined players receive sync-point chunks with past timestamps by design
            earliest_chunk_start = min(
                (
                    ps.queue[0].start_time_us
                    for ps in self._players.values()
                    if ps.queue
                    and (
                        ps.join_wall_time_us is None  # Old player (no join time tracked)
                        or now_us - ps.join_wall_time_us
                        > join_grace_period_us  # Grace period expired
                    )
                ),
                default=None,
            )

            # If any chunk is stale, adjust timing globally
            if (
                earliest_chunk_start is not None
                and earliest_chunk_start < now_us + send_transmission_margin_us
            ):
                logger.warning(
                    "Audio chunk is stale (starts at %d us, now is %d us). "
                    "Adjusting timing globally.",
                    earliest_chunk_start,
                    now_us,
                )
                self._adjust_timing_for_stale_chunk(now_us, earliest_chunk_start)
                # After adjustment, continue to next iteration with updated timing
                continue

            # Stage 2: Send chunks to players with backpressure control
            earliest_blocked_wait_time_us = self._send_chunks_to_players()

            # Stage 2b: Handle backpressure - compare client buffer wait vs source buffer urgency
            if earliest_blocked_wait_time_us > 0:
                channel_wait_us = self._get_earliest_channel_wait_time_us() or 0

                if channel_wait_us < earliest_blocked_wait_time_us:
                    # Source buffer more urgent - wait then exit to refill
                    if channel_wait_us:
                        await asyncio.sleep(channel_wait_us / 1_000_000)
                    break

                # Client buffer more urgent - wait then continue sending
                await asyncio.sleep(earliest_blocked_wait_time_us / 1_000_000)
                continue

            # Stage 3: Cleanup (prune old source data and stale prepared chunks)
            self._prune_old_data()
            self._prune_stale_prepared_chunks()

            # Stage 4: Check exit conditions and apply source buffer backpressure
            has_pending_deliveries = any(
                player_state.queue for player_state in self._players.values()
            )

            if has_pending_deliveries:
                # If client work pending, continue immediately
                continue

            # Stage 4b: Wait for source buffer to drain below target
            channel_wait_us = self._get_earliest_channel_wait_time_us() or 0

            # If no channels or immediate need, exit to refill
            if channel_wait_us == 0:
                break

            # Wait for source buffer to drain
            await asyncio.sleep(channel_wait_us / 1_000_000)

    def flush(self) -> None:
        """Flush all pipelines, preparing any buffered data for sending."""
        for pipeline in self._pipelines.values():
            if pipeline.flushed:
                continue
            if pipeline.buffer:
                self._drain_pipeline_buffer(pipeline, force_flush=True)
            if pipeline.encoder is not None:
                packets = pipeline.encoder.encode(None)
                for packet in packets:
                    if not packet.duration or packet.duration <= 0:
                        logger.warning(
                            "Skipping packet with invalid duration %r during encoder flush",
                            packet.duration,
                        )
                        continue
                    # Calculate timestamps for each flushed packet from its duration
                    start_us, end_us = self._calculate_chunk_timestamps(pipeline, packet.duration)
                    self._handle_encoded_packet(pipeline, packet, start_us, end_us)
                    # Advance next_chunk_start_us for each flushed packet
                    pipeline.next_chunk_start_us = end_us
            pipeline.flushed = True

    def reset(self) -> None:
        """Reset state, releasing encoders and resamplers."""
        for pipeline in self._pipelines.values():
            pipeline.encoder = None
        self._channels.clear()
        self._pipelines.clear()
        self._players.clear()

    def _prune_stale_prepared_chunks(self) -> None:
        """Prune prepared chunks past their playback time (keeps chunks for late joiners)."""
        now_us = self._now_us()
        for pipeline in self._pipelines.values():
            while pipeline.prepared and pipeline.prepared[0].end_time_us < now_us:
                pipeline.prepared.popleft()

    def _send_chunks_to_players(self) -> int:
        """Send chunks to all players with backpressure control.

        Returns:
            Earliest blocked wait time in microseconds (0 if no players blocked).
        """
        earliest_blocked_wait_time_us = 0
        players_to_remove = []

        for player_state in self._players.values():
            tracker = player_state.buffer_tracker
            if tracker is None:
                continue
            queue = player_state.queue

            # Send as many chunks as possible for this player
            while queue:
                chunk = queue[0]

                # Check if we can send without waiting
                if requested_wait := tracker.time_until_capacity(chunk.byte_count):
                    # This player is blocked, track the earliest unblock time
                    if (
                        earliest_blocked_wait_time_us == 0
                        or requested_wait < earliest_blocked_wait_time_us
                    ):
                        earliest_blocked_wait_time_us = requested_wait
                    break

                # We have capacity - send immediately
                header = pack_binary_header_raw(
                    BinaryMessageType.AUDIO_CHUNK.value, chunk.start_time_us
                )
                try:
                    player_state.config.send(header + chunk.data)
                except Exception:
                    logger.exception(
                        "Failed to send chunk to player %s. Removing player.",
                        player_state.config.client_id,
                    )
                    players_to_remove.append(player_state.config.client_id)
                    break
                tracker.register(chunk.end_time_us, chunk.byte_count)
                player_state.queue.popleft()

        # Remove players that failed to send
        for player_id in players_to_remove:
            player_state = self._players.pop(player_id)
            player_state.queue.clear()

            pipeline_key = (player_state.channel_id, player_state.audio_format)
            if pipeline := self._pipelines.get(pipeline_key):
                if player_id in pipeline.subscribers:
                    pipeline.subscribers.remove(player_id)

                # Clean up pipeline if no subscribers remain
                if not pipeline.subscribers:
                    self._pipelines.pop(pipeline_key)
                    pipeline.encoder = None
                    logger.debug("Removed empty pipeline for channel %s", player_state.channel_id)

        return earliest_blocked_wait_time_us

    def _prune_old_data(self) -> None:
        """
        Prune old source chunks to free memory.

        Removes source chunks that have finished playing (end_time_us <= now).
        Prepared chunks are managed separately by _prune_stale_prepared_chunks().
        """
        # Prune source buffer based on playback time
        now_us = self._now_us()

        for channel_id, channel_state in self._channels.items():
            chunks_removed = 0
            while (
                channel_state.source_buffer and channel_state.source_buffer[0].end_time_us <= now_us
            ):
                channel_state.source_buffer.popleft()
                chunks_removed += 1

            # Update pipeline read positions for pipelines consuming from this channel
            if chunks_removed > 0:
                for pipeline in self._pipelines.values():
                    if pipeline.channel_id == channel_id:
                        pipeline.source_read_position = max(
                            0, pipeline.source_read_position - chunks_removed
                        )

    def _process_pipeline_from_source(
        self, pipeline: PipelineState, channel_state: ChannelState
    ) -> None:
        """
        Process available source chunks through this pipeline.

        Args:
            pipeline: The pipeline to process.
            channel_state: The channel state to read from.
        """
        if not pipeline.subscribers:
            return

        # Process all available source chunks that haven't been processed yet
        while pipeline.source_read_position < len(channel_state.source_buffer):
            source_chunk = channel_state.source_buffer[pipeline.source_read_position]
            self._process_source_pcm(
                pipeline,
                channel_state,
                source_chunk,
            )
            pipeline.source_read_position += 1

    def _process_source_pcm(
        self,
        pipeline: PipelineState,
        channel_state: ChannelState,
        source_chunk: SourceChunk,
    ) -> None:
        """
        Process source PCM data through the pipeline's resampler.

        Args:
            pipeline: The pipeline to process through.
            channel_state: The channel state for source format data.
            source_chunk: The source PCM chunk to process.
        """
        # Initialize next_chunk_start_us from first source chunk
        if pipeline.next_chunk_start_us is None and not pipeline.buffer:
            pipeline.next_chunk_start_us = source_chunk.start_time_us

        frame = av.AudioFrame(
            format=channel_state.source_format_params.av_format,
            layout=channel_state.source_format_params.av_layout,
            samples=source_chunk.sample_count,
        )
        frame.sample_rate = channel_state.source_format_params.audio_format.sample_rate
        frame.planes[0].update(source_chunk.pcm_data)
        out_frames = pipeline.resampler.resample(frame)
        for out_frame in out_frames:
            expected = pipeline.target_frame_stride * out_frame.samples
            pcm_bytes = bytes(out_frame.planes[0])[:expected]
            pipeline.buffer.extend(pcm_bytes)
        self._drain_pipeline_buffer(pipeline, force_flush=False)

    def _calculate_chunk_timestamps(
        self,
        pipeline: PipelineState,
        sample_count: int,
    ) -> tuple[int, int]:
        """
        Calculate start and end timestamps for a chunk.

        Uses the pipeline's next_chunk_start_us to maintain alignment with source timestamps.

        Args:
            pipeline: The pipeline producing the chunk.
            sample_count: Number of samples in the chunk.

        Returns:
            Tuple of (start_us, end_us) timestamps.
        """
        if pipeline.next_chunk_start_us is None:
            raise RuntimeError("Pipeline next_chunk_start_us not initialized")

        start_us = pipeline.next_chunk_start_us
        duration_us = int(sample_count * 1_000_000 / pipeline.target_format.sample_rate)
        end_us = start_us + duration_us
        return start_us, end_us

    def _drain_pipeline_buffer(
        self,
        pipeline: PipelineState,
        *,
        force_flush: bool,
    ) -> None:
        """
        Drain the pipeline buffer by creating and publishing chunks.

        Extracts complete chunks from the pipeline buffer and either publishes them
        directly (for PCM) or encodes them first (for compressed codecs).
        Calculates timestamps based on the pipeline's current sample position.

        Args:
            pipeline: The pipeline whose buffer to drain.
            force_flush: If True, publish all available samples even if less than chunk_samples.
        """
        if not pipeline.subscribers:
            pipeline.buffer.clear()
            return

        frame_stride = pipeline.target_frame_stride
        while len(pipeline.buffer) >= frame_stride:
            available_samples = len(pipeline.buffer) // frame_stride
            if not force_flush and available_samples < pipeline.chunk_samples:
                break

            # Extract data to fit sample count
            sample_count = pipeline.chunk_samples
            if force_flush and available_samples < pipeline.chunk_samples:
                # Pad incomplete chunk with zeros to reach full chunk_samples
                audio_data_bytes = available_samples * frame_stride
                padding_bytes = (sample_count - available_samples) * frame_stride
                chunk = bytes(pipeline.buffer[:audio_data_bytes]) + bytes(padding_bytes)
                del pipeline.buffer[:audio_data_bytes]
            else:
                chunk_size = sample_count * frame_stride
                chunk = bytes(pipeline.buffer[:chunk_size])
                del pipeline.buffer[:chunk_size]

            if pipeline.encoder is None:
                # PCM path: calculate timestamps from input sample count
                start_us, end_us = self._calculate_chunk_timestamps(pipeline, sample_count)
                self._publish_chunk(pipeline, chunk, sample_count, start_us, end_us)
                # Advance next_chunk_start_us for the next chunk
                pipeline.next_chunk_start_us = end_us
            else:
                # Encoder path: let encoder calculate timestamps from output packets
                self._encode_and_publish(pipeline, chunk, sample_count)

    def _encode_and_publish(
        self,
        pipeline: PipelineState,
        chunk: bytes,
        sample_count: int,
    ) -> None:
        """
        Encode a PCM chunk and publish the resulting packets.

        The encoder may buffer input and produce 0, 1, or multiple output packets.
        Timestamps are calculated from each output packet's duration.

        Args:
            pipeline: The pipeline containing the encoder.
            chunk: Raw PCM audio data to encode.
            sample_count: Number of samples in the chunk.
        """
        if pipeline.encoder is None:
            raise RuntimeError("Encoder not configured for this pipeline")
        frame = av.AudioFrame(
            format=pipeline.target_av_format,
            layout=pipeline.target_layout,
            samples=sample_count,
        )
        frame.sample_rate = pipeline.target_format.sample_rate
        frame.planes[0].update(chunk)
        packets = pipeline.encoder.encode(frame)

        # Encoder may produce 0 or more packets
        for packet in packets:
            if not packet.duration or packet.duration <= 0:
                raise ValueError(f"Invalid packet duration: {packet.duration!r}")
            # Calculate timestamps from output packet duration
            start_us, end_us = self._calculate_chunk_timestamps(pipeline, packet.duration)
            self._handle_encoded_packet(pipeline, packet, start_us, end_us)
            # Advance next_chunk_start_us for each packet produced
            pipeline.next_chunk_start_us = end_us

    def _handle_encoded_packet(
        self,
        pipeline: PipelineState,
        packet: av.Packet,
        start_us: int,
        end_us: int,
    ) -> None:
        """
        Handle an encoded packet by publishing it as a chunk.

        Args:
            pipeline: The pipeline that produced the packet.
            packet: The encoded audio packet from the encoder.
            start_us: Start timestamp in microseconds.
            end_us: End timestamp in microseconds.
        """
        assert packet.duration is not None  # For type checking
        chunk_data = bytes(packet)
        self._publish_chunk(pipeline, chunk_data, packet.duration, start_us, end_us)

    def _publish_chunk(
        self,
        pipeline: PipelineState,
        audio_data: bytes,
        sample_count: int,
        start_us: int,
        end_us: int,
    ) -> None:
        """
        Create a PreparedChunkState and queue it for all subscribers.

        Queues the chunk for delivery to all clients subscribed to this pipeline.

        Args:
            pipeline: The pipeline publishing the chunk.
            audio_data: The encoded or PCM audio data.
            sample_count: Number of samples in the chunk.
            start_us: Start timestamp in microseconds.
            end_us: End timestamp in microseconds.
        """
        if not pipeline.subscribers or sample_count <= 0:
            return

        chunk = PreparedChunkState(
            data=audio_data,
            start_time_us=start_us,
            end_time_us=end_us,
            sample_count=sample_count,
            byte_count=len(audio_data),
        )
        pipeline.prepared.append(chunk)
        pipeline.samples_produced += sample_count
        self._last_chunk_end_us = end_us

        for client_id in pipeline.subscribers:
            if player_state := self._players.get(client_id):
                player_state.queue.append(chunk)

    def get_channel_ids(self) -> set[UUID]:
        """
        Get the set of active channel IDs.

        Returns:
            Set of currently active channel IDs.
        """
        return set(self._channels.keys())

    def get_player_ids(self) -> set[str]:
        """
        Get the set of active player IDs.

        Returns:
            Set of currently active player client IDs.
        """
        return set(self._players.keys())

    def remove_player(self, client_id: str) -> None:
        """
        Remove a player from the streamer.

        Cleans up the player's queue and removes it from pipeline subscribers.
        This is used when a stale client is being replaced by a reconnecting client.

        Args:
            client_id: The client ID to remove.
        """
        player_state = self._players.pop(client_id, None)
        if player_state is None:
            return

        player_state.queue.clear()

        # Remove from pipeline subscribers
        pipeline_key = (player_state.channel_id, player_state.audio_format)
        if pipeline := self._pipelines.get(pipeline_key):
            if client_id in pipeline.subscribers:
                pipeline.subscribers.remove(client_id)

            # Clean up pipeline if no subscribers remain
            if not pipeline.subscribers:
                self._pipelines.pop(pipeline_key)
                pipeline.encoder = None
                logger.debug("Removed empty pipeline for channel %s", player_state.channel_id)

    @property
    def last_chunk_end_time_us(self) -> int | None:
        """Return the end timestamp of the most recently prepared chunk."""
        return self._last_chunk_end_us


__all__ = [
    "MAIN_CHANNEL_ID",
    "AudioCodec",
    "AudioFormat",
    "AudioFormatParams",
    "ClientStreamConfig",
    "MediaStream",
    "Streamer",
]
