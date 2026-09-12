"""PlayerV1Role implementation for audio playback (v1).

This PlayerV1Role implementation uses hook-based streaming:
- on_stream_start(): Send stream/start message
- on_audio_chunk(): Pack and send binary audio
- on_stream_clear(): Send stream/clear message
- on_stream_end(): Send stream/end message
"""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from aiosendspin.models import AudioCodec, BinaryMessageType
from aiosendspin.models.core import (
    ClientStatePayload,
    ServerCommandMessage,
    ServerCommandPayload,
    StreamClearMessage,
    StreamClearPayload,
    StreamEndMessage,
    StreamEndPayload,
    StreamRequestFormatPayload,
    StreamStartMessage,
    StreamStartPayload,
)
from aiosendspin.models.player import PlayerCommandPayload, StreamStartPlayer, SupportedAudioFormat
from aiosendspin.models.types import PlayerCommand
from aiosendspin.server.audio import AudioFormat, BufferTracker
from aiosendspin.server.roles.base import (
    AudioChunk,
    AudioRequirements,
    BinaryHandling,
    Role,
    StreamRequirements,
)
from aiosendspin.server.roles.player.audio_transformers import (
    FlacEncoder,
    OpusEncoder,
    PcmPassthrough,
)
from aiosendspin.server.roles.player.capabilities import can_encode_format, filter_encodable_formats
from aiosendspin.server.roles.player.events import (
    MinBufferChangedEvent,
    OutputDelayChangedEvent,
    RequiredLeadTimeChangedEvent,
    VolumeChangedEvent,
)
from aiosendspin.util import create_task

if TYPE_CHECKING:
    from aiosendspin.server.client import SendspinClient


BUFFER_TRACKER_RESET_DELAY_S = 2.0


@dataclass
class PlayerPersistentState:
    """Persistent player state stored on the SendspinClient."""

    volume: int = 100
    muted: bool = False
    buffer_tracker: BufferTracker | None = None
    buffer_capacity_scale: float = 1.0
    max_duration_us: int = 30_000_000
    disconnect_time_us: int | None = None
    buffer_reset_handle: asyncio.TimerHandle | None = None
    output_delay_ms: int = 0
    required_lead_time_ms: int = 250
    min_buffer_ms: int = 1000
    state_supported_commands: list[PlayerCommand] = field(default_factory=list)
    preferred_format_override: AudioFormat | None = None
    preferred_codec_override: AudioCodec | None = None


class PlayerV1Role(Role):
    """Role implementation for audio playback.

    Hook-based streaming:
    - on_stream_start(): Send stream/start message
    - on_audio_chunk(): Pack and send binary audio
    - on_stream_clear(): Send stream/clear message
    - on_stream_end(): Send stream/end message
    """

    def __init__(
        self,
        client: SendspinClient | None = None,
        *,
        preferred_format: AudioFormat | None = None,
        audio_requirements: AudioRequirements | None = None,
    ) -> None:
        """Initialize PlayerV1Role.

        Args:
            client: The owning SendspinClient.
            preferred_format: Preferred audio format for this player.
            audio_requirements: Audio requirements for hook-based streaming.
        """
        if client is None:
            msg = "PlayerV1Role requires a client"
            raise ValueError(msg)
        self._client = client
        self._preferred_format_override = preferred_format
        self._preferred_format: AudioFormat | None = None
        self._preferred_codec: AudioCodec | None = None
        self._audio_requirements = audio_requirements
        self._stream_started = False
        self._buffer_tracker = None
        # Initialize timing state for binary handling
        self._stream_start_time_us = None
        self._last_late_log_s = 0.0
        self._late_skips_since_log = 0
        # Cached state reference (avoids repeated dict lookup + isinstance check)
        self._cached_state: PlayerPersistentState | None = None
        # Deferred stream start: set True by on_stream_start(), sent on first audio chunk
        self._pending_stream_start = False
        # Last format announced to the client via stream/start.
        self._last_sent_format: tuple[AudioCodec, int, int, int, str | None] | None = None

    @property
    def role_id(self) -> str:
        """Versioned role identifier."""
        return "player@v1"

    @property
    def role_family(self) -> str:
        """Role family name for protocol messages."""
        return "player"

    @property
    def preferred_format(self) -> AudioFormat | None:
        """Return the preferred audio format for this player."""
        return self._preferred_format or self._preferred_format_override

    @preferred_format.setter
    def preferred_format(self, value: AudioFormat | None) -> None:
        self._preferred_format = value

    @property
    def preferred_codec(self) -> AudioCodec | None:
        """Return the preferred audio codec for this player."""
        return self._preferred_codec

    @preferred_codec.setter
    def preferred_codec(self, value: AudioCodec | None) -> None:
        self._preferred_codec = value

    # --- Declarations ---

    def get_stream_requirements(self) -> StreamRequirements:
        """Player role sends binary audio streams."""
        return StreamRequirements()

    def get_audio_requirements(self) -> AudioRequirements | None:
        """Return audio requirements for hook-based streaming."""
        req = self._audio_requirements
        if req is None:
            return None

        # Legacy/manually-injected requirements may omit channel assignment.
        # In that case, preserve existing behavior.
        if req.channel_id is None:
            return req

        # Channel routing may change at stream start when a custom channel_resolver
        # is installed. Refresh cached requirements if the resolved channel changed.
        channel_id = self._client.group.get_channel_for_player(self._client.client_id)

        if channel_id != req.channel_id:
            self._ensure_audio_requirements(force=True)

        return self._audio_requirements

    def get_binary_handling(self, message_type: int) -> BinaryHandling | None:
        """Return handling policy for AUDIO_CHUNK messages."""
        if message_type == BinaryMessageType.AUDIO_CHUNK.value:
            return BinaryHandling(
                drop_late=True,
                grace_period_us=2_000_000,  # 2 seconds grace for initial buffering
                buffer_track=True,
            )
        return None

    def get_buffer_tracker(self) -> BufferTracker | None:
        """Return the role-owned buffer tracker."""
        return self._state().buffer_tracker

    def get_join_delay_s(self) -> float:
        """Delay joins briefly to allow time sync to stabilize."""
        return 1.0

    # --- Lifecycle hooks ---

    def on_connect(self) -> None:
        """Reset stream state and subscribe to PlayerGroupRole."""
        self._subscribe_to_group_role()
        self._stream_started = False
        self._last_sent_format = None
        state = self._state()
        if state.buffer_reset_handle is not None:
            state.buffer_reset_handle.cancel()
            state.buffer_reset_handle = None
        state.disconnect_time_us = None
        self._ensure_buffer_tracker(state)
        # Reset buffer tracker on (re)connect - client buffer is empty after reconnect
        if state.buffer_tracker is not None:
            state.buffer_tracker.reset()
        self._ensure_preferred_format()
        self._ensure_audio_requirements(force=True)

    def on_deactivate(self) -> None:
        """End the player stream when the role is deactivated while still connected."""
        if self._stream_started:
            self.on_stream_end()
        super().on_deactivate()

    def on_disconnect(self) -> None:
        """Clean up, apply delayed buffer reset policy, and unsubscribe from PlayerGroupRole."""
        self._unsubscribe_from_group_role()
        self._stream_started = False
        self._last_sent_format = None

        state = self._state()
        state.disconnect_time_us = self._client._server.clock.now_us()  # noqa: SLF001
        if state.buffer_tracker is None:
            return

        disconnect_time_us = state.disconnect_time_us

        def _maybe_reset() -> None:
            state.buffer_reset_handle = None
            if self._client.connection is not None:
                return
            if disconnect_time_us != state.disconnect_time_us:
                return
            if state.buffer_tracker is None:
                return
            state.buffer_tracker.reset()

        if state.buffer_reset_handle is not None:
            state.buffer_reset_handle.cancel()
        state.buffer_reset_handle = self._client._server.loop.call_later(  # noqa: SLF001
            BUFFER_TRACKER_RESET_DELAY_S, _maybe_reset
        )

    def requires_initial_state(self) -> bool:
        """Player role requires initial state with volume/mute info."""
        return True

    def on_group_changed(self, group: object) -> None:
        """Refresh transformer selection when group changes."""
        super().on_group_changed(group)
        state = self._state()
        self._ensure_buffer_tracker(state)
        if state.buffer_tracker is not None:
            # Group switches imply a stream boundary for this player; any previously
            # tracked buffered audio belongs to the old group timeline.
            state.buffer_tracker.reset()
        self._stream_started = False
        self._pending_stream_start = False
        self._last_sent_format = None
        self.reset_binary_timing()
        self._ensure_audio_requirements(force=True)

    # --- Stream lifecycle hooks ---

    def on_stream_start(self) -> None:
        """Mark stream start as pending - actual message sent on first audio chunk.

        This defers the stream/start message until the first audio chunk arrives,
        ensuring the codec header is available (FLAC generates header on first encode).
        """
        req = self.get_audio_requirements()
        if req is None:
            self._ensure_audio_requirements()
            req = self.get_audio_requirements()
        if req is None:
            return

        if not self.has_connection():
            return

        # New stream boundary: clear prior stream timing/log state so
        # late-drop grace period is measured from this stream's first chunk.
        self.reset_binary_timing()
        self._pending_stream_start = True

    def _send_stream_start_message(self) -> None:
        """Send stream/start message with codec header from transformer.

        Skips the send when the client already has an active stream with
        an identical format.
        """
        req = self.get_audio_requirements()
        if req is None or not self.has_connection():
            return

        transformer = req.transformer
        header = transformer.get_header() if isinstance(transformer, FlacEncoder) else None
        header_b64 = base64.b64encode(header).decode() if header else None

        # Determine codec from transformer type
        if isinstance(transformer, FlacEncoder):
            codec = AudioCodec.FLAC
        elif isinstance(transformer, OpusEncoder):
            codec = AudioCodec.OPUS
        else:
            codec = AudioCodec.PCM

        current_format = (codec, req.sample_rate, req.channels, req.bit_depth, header_b64)
        if self._stream_started and self._last_sent_format == current_format:
            # Client already configured for this exact format
            return

        stream_start = StreamStartMessage(
            payload=StreamStartPayload(
                player=StreamStartPlayer(
                    codec=codec,
                    sample_rate=req.sample_rate,
                    channels=req.channels,
                    bit_depth=req.bit_depth,
                    codec_header=header_b64,
                )
            )
        )
        self.send_message(stream_start)
        self._stream_started = True
        self._last_sent_format = current_format

    def on_audio_chunk(self, chunk: AudioChunk) -> None:
        """Pack and send binary audio. Late audio is discarded by connection."""
        # Send deferred stream/start on first chunk (ensures encoder header is available)
        if self._pending_stream_start:
            self._send_stream_start_message()
            self._pending_stream_start = False

        # Guard against stale delivery after stream/end.
        if not self._stream_started:
            if self.has_connection():
                self._client._logger.debug(  # noqa: SLF001
                    "Dropping stale player audio chunk without active stream for %s",
                    self._client.client_id,
                )
            return

        # Reuse the frame packed once and shared across subscribers.
        message_type = BinaryMessageType.AUDIO_CHUNK.value
        # Compute the wall-clock buffer horizon (effective play time) by shifting
        # the chunk's end time earlier by the configured output delay.
        output_delay_us = self.output_delay_ms * 1_000
        chunk_end_us = chunk.timestamp_us + chunk.duration_us - output_delay_us

        self._client.send_binary(
            chunk.packed,
            role_family=self.role_family,
            timestamp_us=chunk.timestamp_us,
            message_type=message_type,
            buffer_end_time_us=chunk_end_us,
            buffer_byte_count=chunk.byte_count,
            duration_us=chunk.duration_us,
        )

    def on_stream_clear(self) -> None:
        """Send stream/clear and reset buffer-tracking state."""
        if not self.has_connection():
            return

        stream_clear = StreamClearMessage(payload=StreamClearPayload(roles=["player"]))
        self.send_message(stream_clear)
        self._pending_stream_start = False
        self.reset_binary_timing()

        if self._buffer_tracker is not None:
            self._buffer_tracker.reset()

    def on_stream_end(self) -> None:
        """Send stream/end and reset state."""
        if not self.has_connection():
            return

        stream_end = StreamEndMessage(payload=StreamEndPayload(roles=["player"]))
        self.send_message(stream_end)
        self._stream_started = False
        self._pending_stream_start = False
        self._last_sent_format = None
        self.reset_binary_timing()

        if self._buffer_tracker is not None:
            self._buffer_tracker.reset()

    @property
    def stream_started(self) -> bool:
        """Whether stream/start has been sent."""
        return self._stream_started

    # ---- Volume/mute state and commands ----

    @property
    def volume(self) -> int:
        """Current volume of this player (0-100)."""
        return self._state().volume

    @volume.setter
    def volume(self, value: int) -> None:
        self._state().volume = value

    @property
    def muted(self) -> bool:
        """Current mute state of this player."""
        return self._state().muted

    @muted.setter
    def muted(self, value: bool) -> None:
        self._state().muted = value

    @property
    def output_delay_ms(self) -> int:
        """Current output delay of this player in milliseconds (0-5000)."""
        return self._state().output_delay_ms

    @output_delay_ms.setter
    def output_delay_ms(self, value: int) -> None:
        self._state().output_delay_ms = value

    @property
    def required_lead_time_ms(self) -> int:
        """Startup lead time reported by this player in milliseconds."""
        return self._state().required_lead_time_ms

    @required_lead_time_ms.setter
    def required_lead_time_ms(self, value: int) -> None:
        self._state().required_lead_time_ms = value

    @property
    def min_buffer_ms(self) -> int:
        """Minimum ongoing buffer duration reported by this player in milliseconds."""
        return self._state().min_buffer_ms

    @min_buffer_ms.setter
    def min_buffer_ms(self, value: int) -> None:
        self._state().min_buffer_ms = value

    @property
    def state_supported_commands(self) -> list[PlayerCommand]:
        """Commands supported via client/state (e.g., set_output_delay)."""
        return self._state().state_supported_commands

    @state_supported_commands.setter
    def state_supported_commands(self, value: list[PlayerCommand]) -> None:
        self._state().state_supported_commands = value

    def get_player_volume(self) -> int | None:
        """Return current volume for group aggregation."""
        return self.volume

    def get_player_muted(self) -> bool | None:
        """Return current mute state for group aggregation."""
        return self.muted

    def set_player_volume(self, volume: int) -> None:
        """Set player volume via role API."""
        self.set_volume(volume)

    def set_player_mute(self, muted: bool) -> None:  # noqa: FBT001
        """Set player mute via role API."""
        self.set_mute(muted)

    def get_output_delay_us(self) -> int:
        """Return transport delay in microseconds for timestamp offsetting."""
        return max(self.output_delay_ms, 0) * 1_000

    def get_required_lead_time_us(self) -> int:
        """Return reported startup lead time in microseconds."""
        return max(self.required_lead_time_ms, 0) * 1_000

    def get_min_buffer_us(self) -> int:
        """Return reported minimum ongoing buffer duration in microseconds."""
        return max(self.min_buffer_ms, 0) * 1_000

    def get_output_delay_ms(self) -> int:
        """Return output delay for protocol API."""
        return self.output_delay_ms

    def set_output_delay(self, delay_ms: int) -> None:
        """Send set_output_delay command to client.

        Addresses the client using whichever spelling it declared support for
        — a client that only declared the pre-rename 'set_static_delay' still
        receives a delay command it can act on.
        """
        if PlayerCommand.SET_OUTPUT_DELAY in self.state_supported_commands:
            command = PlayerCommand.SET_OUTPUT_DELAY
        elif PlayerCommand.SET_STATIC_DELAY in self.state_supported_commands:
            command = PlayerCommand.SET_STATIC_DELAY
        else:
            return

        self._client.send_message(
            ServerCommandMessage(
                payload=ServerCommandPayload(
                    player=PlayerCommandPayload(
                        command=command,
                        output_delay_ms=delay_ms,
                    )
                )
            )
        )

    def get_supported_formats(self) -> list[SupportedAudioFormat] | None:
        """Return formats both client and server support, in client priority order."""
        support = self._client.info.player_support
        if support is None:
            return None
        return filter_encodable_formats(support.supported_formats)

    def set_preferred_format(
        self,
        audio_format: AudioFormat | None,
        codec: AudioCodec | None = None,
    ) -> bool:
        """Set or clear preferred format override.

        Args:
            audio_format: The audio format to set, or None to clear the override.
            codec: The codec to use when a format is provided. If audio_format is
                None and codec is provided, the first compatible format for that
                codec (in client priority order) is used.

        Returns:
            True if the override was set/cleared, False if incompatible/invalid.
        """
        if audio_format is None:
            if codec is not None:
                support = self._client.info.player_support
                if support is None:
                    return False
                compatible = filter_encodable_formats(support.supported_formats)
                matched = next((fmt for fmt in compatible if fmt.codec == codec), None)
                if matched is None:
                    return False
                audio_format = AudioFormat(
                    sample_rate=matched.sample_rate,
                    bit_depth=matched.bit_depth,
                    channels=matched.channels,
                )
            else:
                state = self._state()
                state.preferred_format_override = None
                state.preferred_codec_override = None
                before = self._effective_format()
                self._ensure_preferred_format()
                self._ensure_audio_requirements(force=True)
                if self._client.group.has_active_stream and self._effective_format() != before:
                    self._begin_format_transition()
                return True

        if codec is None:
            return False

        support = self._client.info.player_support
        if support is None:
            return False

        # Check if format is in client's supported list
        client_format = SupportedAudioFormat(
            codec=codec,
            sample_rate=audio_format.sample_rate,
            bit_depth=audio_format.bit_depth,
            channels=audio_format.channels,
        )
        is_client_supported = any(
            fmt.codec == codec
            and fmt.sample_rate == audio_format.sample_rate
            and fmt.bit_depth == audio_format.bit_depth
            and fmt.channels == audio_format.channels
            for fmt in support.supported_formats
        )
        if not is_client_supported:
            return False

        # Check if server can encode this format
        if not can_encode_format(client_format):
            return False

        # Persist the server-side override across reconnects and role recreation.
        state = self._state()
        state.preferred_format_override = audio_format
        state.preferred_codec_override = codec

        # Set the preferred format for current session.
        self._preferred_format = audio_format
        self._preferred_codec = codec

        before = self._effective_format()

        # Rebuild audio requirements with the new format
        self._ensure_audio_requirements(force=True)

        if self._client.group.has_active_stream and self._effective_format() != before:
            self._begin_format_transition()

        return True

    def set_volume(self, volume: int) -> None:
        """Set the volume of this player."""
        support = self._client.info.player_support
        if not support or PlayerCommand.VOLUME not in support.supported_commands:
            return

        self._client.send_message(
            ServerCommandMessage(
                payload=ServerCommandPayload(
                    player=PlayerCommandPayload(
                        command=PlayerCommand.VOLUME,
                        volume=volume,
                    )
                )
            )
        )

    def set_mute(self, muted: bool) -> None:  # noqa: FBT001
        """Set the mute state of this player."""
        support = self._client.info.player_support
        if not support or PlayerCommand.MUTE not in support.supported_commands:
            return

        self._client.send_message(
            ServerCommandMessage(
                payload=ServerCommandPayload(
                    player=PlayerCommandPayload(
                        command=PlayerCommand.MUTE,
                        mute=muted,
                    )
                )
            )
        )

    # ---- Client message handling ----

    def initial_state_deviations(self, payload: ClientStatePayload) -> list[str]:
        """Report required player fields missing from the initial client/state."""
        player = payload.player
        if player is None:
            return ["has an active player role but no player state"]
        reasons: list[str] = []
        if (
            player.output_delay_ms is None
            or player.required_lead_time_ms is None
            or player.min_buffer_ms is None
        ):
            reasons.append("omitted required player timing fields")
        support = self._client.info.player_support
        commands = support.supported_commands if support else []
        if PlayerCommand.VOLUME in commands and player.volume is None:
            reasons.append("omitted volume despite declaring the volume command")
        if PlayerCommand.MUTE in commands and player.muted is None:
            reasons.append("omitted muted despite declaring the mute command")
        return reasons

    def client_state_deviations(self, payload: ClientStatePayload) -> list[str]:
        """Report player fields in a client/state that violate the spec."""
        state = payload.player
        if state is None:
            return []
        support = self._client.info.player_support
        commands = support.supported_commands if support else []
        reasons: list[str] = []
        if state.state is not None:
            reasons.append("used legacy player.state instead of top-level available")
        if state.volume is not None and PlayerCommand.VOLUME not in commands:
            reasons.append("sent volume without declaring the volume command")
        if state.muted is not None and PlayerCommand.MUTE not in commands:
            reasons.append("sent muted without declaring the mute command")
        if state.legacy_delay_key:
            reasons.append(f"used the pre-rename '{state.legacy_delay_key}' key")
        if state.supported_commands and PlayerCommand.SET_STATIC_DELAY in state.supported_commands:
            reasons.append("declared the pre-rename 'set_static_delay' command")
        return reasons

    def on_client_state(self, payload: ClientStatePayload) -> None:
        """Apply player-specific fields from client/state (validation runs earlier)."""
        state = payload.player
        if state is None:
            return

        # Fall back to legacy player.state for availability only when the compliant
        # top-level field is absent.
        if state.state is not None and payload.available is None:
            create_task(
                self._client.handle_availability_change(available=state.state != "external_source")
            )

        support = self._client.info.player_support
        commands = support.supported_commands if support else []
        changed = False

        if (
            state.volume is not None
            and PlayerCommand.VOLUME in commands
            and self.volume != state.volume
        ):
            self.volume = state.volume
            changed = True

        if state.muted is not None and PlayerCommand.MUTE in commands and self.muted != state.muted:
            self.muted = state.muted
            changed = True

        if changed:
            self.emit_client_event(VolumeChangedEvent(volume=self.volume, muted=self.muted))

        if state.supported_commands is not None:
            self.state_supported_commands = state.supported_commands

        if state.output_delay_ms is not None and self.output_delay_ms != state.output_delay_ms:
            self.output_delay_ms = state.output_delay_ms
            self.emit_client_event(OutputDelayChangedEvent(output_delay_ms=state.output_delay_ms))

        if (
            state.required_lead_time_ms is not None
            and self.required_lead_time_ms != state.required_lead_time_ms
        ):
            self.required_lead_time_ms = state.required_lead_time_ms
            self.emit_client_event(
                RequiredLeadTimeChangedEvent(required_lead_time_ms=state.required_lead_time_ms)
            )

        if state.min_buffer_ms is not None and self.min_buffer_ms != state.min_buffer_ms:
            self.min_buffer_ms = state.min_buffer_ms
            self.emit_client_event(MinBufferChangedEvent(min_buffer_ms=state.min_buffer_ms))

    def on_stream_request_format(self, payload: StreamRequestFormatPayload) -> None:
        """Handle stream/request-format for player role."""
        player_req = payload.player
        if player_req is None:
            return

        support = self._client.info.player_support
        if support is None:
            raise ValueError(
                f"Client {self._client.client_id} sent player format request "
                "but has no player support"
            )

        # A requested format must be one the client declared in its hello
        # supported_formats; only the fields the request actually carries are compared.
        if not any(
            (player_req.codec is None or fmt.codec == player_req.codec)
            and (player_req.sample_rate is None or fmt.sample_rate == player_req.sample_rate)
            and (player_req.bit_depth is None or fmt.bit_depth == player_req.bit_depth)
            and (player_req.channels is None or fmt.channels == player_req.channels)
            for fmt in support.supported_formats
        ):
            self._client.flag_noncompliance(
                "stream/request-format requested a format not in the client's "
                "declared supported_formats"
            )

        supported = filter_encodable_formats(support.supported_formats)
        if not supported:
            self._client._logger.warning(  # noqa: SLF001
                "Client %s requested format change but has no server-compatible formats",
                self._client.client_id,
            )
            return

        preferred_supported = supported[0]
        base_format = self.preferred_format or AudioFormat(
            sample_rate=preferred_supported.sample_rate,
            bit_depth=preferred_supported.bit_depth,
            channels=preferred_supported.channels,
        )
        base_codec = self.preferred_codec or preferred_supported.codec

        requested_codec = player_req.codec or base_codec
        requested_format = AudioFormat(
            sample_rate=player_req.sample_rate or base_format.sample_rate,
            bit_depth=player_req.bit_depth or base_format.bit_depth,
            channels=player_req.channels or base_format.channels,
        )

        if not any(
            fmt.codec == requested_codec
            and fmt.sample_rate == requested_format.sample_rate
            and fmt.bit_depth == requested_format.bit_depth
            and fmt.channels == requested_format.channels
            for fmt in supported
        ):
            self._client._logger.warning(  # noqa: SLF001
                "Client %s requested unsupported format %s codec=%s, falling back to %s",
                self._client.client_id,
                requested_format,
                requested_codec,
                base_format,
            )
            requested_format = base_format
            requested_codec = base_codec

        self.preferred_format = requested_format
        self.preferred_codec = requested_codec

        current_format = self._effective_format()
        if current_format is not None and current_format == (requested_codec, requested_format):
            # Already on the requested format. Running the boundary would evict
            # valid audio while the announcement's identity guard suppresses
            # the stream/start that would justify it.
            return

        stream_active = self._client.group.has_active_stream
        if stream_active:
            self._ensure_audio_requirements(force=True)
            self._begin_format_transition()
        else:
            # No active stream: also defer stream/start via _pending_stream_start
            # so codec header is included when the first chunk arrives.
            self._ensure_audio_requirements(force=True)
            self._pending_stream_start = True

    def _effective_format(self) -> tuple[AudioCodec, AudioFormat] | None:
        """Return the current negotiated (codec, format), or None without requirements."""
        req = self.get_audio_requirements()
        if req is None:
            return None
        transformer = req.transformer
        if isinstance(transformer, FlacEncoder):
            codec = AudioCodec.FLAC
        elif isinstance(transformer, OpusEncoder):
            codec = AudioCodec.OPUS
        else:
            codec = AudioCodec.PCM
        return (
            codec,
            AudioFormat(
                sample_rate=req.sample_rate,
                bit_depth=req.bit_depth,
                channels=req.channels,
            ),
        )

    def _begin_format_transition(self) -> None:
        """Start a mid-stream format change at a clean boundary.

        Frames queued under the old format must never reach the client after
        the new stream/start; the announcement itself waits for the next
        chunk so it can carry the new codec header.
        """
        self._client.drop_pending_binary([self.role_family])
        # Everything the buffer tracker and binary timing know about was just
        # invalidated with the old format; without resetting them the writer
        # paces the replacement audio against a buffer the client flushed.
        self.reset_binary_timing()
        if self._buffer_tracker is not None:
            self._buffer_tracker.reset()
        self._pending_stream_start = True
        # The client flushes its invalidated buffer only on a stream/start, so a
        # flip-flop back to the announced format must not suppress one.
        self._last_sent_format = None
        self._client.group.on_role_format_changed(self)

    # ---- Internal helpers ----

    def _state(self) -> PlayerPersistentState:
        if self._cached_state is None:
            self._cached_state = self._client.get_or_create_role_state(
                "player", PlayerPersistentState
            )
        return self._cached_state

    def _ensure_buffer_tracker(self, state: PlayerPersistentState) -> None:
        support = self._client.info.player_support
        if support is None:
            self._buffer_tracker = None
            return

        capacity = int(support.buffer_capacity * state.buffer_capacity_scale)
        capacity = max(1, capacity)
        max_duration_us = state.max_duration_us

        if state.buffer_tracker is None:
            state.buffer_tracker = BufferTracker(
                clock=self._client._server.clock,  # noqa: SLF001
                client_id=self._client.client_id,
                capacity_bytes=capacity,
                max_duration_us=max_duration_us,
            )
        else:
            state.buffer_tracker.capacity_bytes = capacity
            state.buffer_tracker.max_duration_us = max_duration_us
        self._buffer_tracker = state.buffer_tracker

    def _ensure_preferred_format(self) -> None:
        support = self._client.info.player_support
        if support is None:
            return

        # Filter to formats the server can actually encode
        compatible = filter_encodable_formats(support.supported_formats)
        if not compatible:
            self._client._logger.warning(  # noqa: SLF001
                "Client %s has no server-compatible formats",
                self._client.client_id,
            )
            return

        # The spec defines supported_formats as "in priority order (first is preferred)".
        # On every (re)connect the client sends a fresh client/hello with its current
        # priority, so compatible[0] represents the client's authoritative preference
        # for this connection.
        # If a server-side override was explicitly set, keep it sticky across reconnects
        # while still validating it against the latest client capabilities.
        preferred_supported = compatible[0]
        state = self._state()
        persistent_format = state.preferred_format_override
        persistent_codec = state.preferred_codec_override
        if persistent_format is not None and persistent_codec is not None:
            matched_persistent = next(
                (
                    fmt
                    for fmt in compatible
                    if fmt.codec == persistent_codec
                    and fmt.sample_rate == persistent_format.sample_rate
                    and fmt.bit_depth == persistent_format.bit_depth
                    and fmt.channels == persistent_format.channels
                ),
                None,
            )
            if matched_persistent is not None:
                preferred_supported = matched_persistent
            else:
                self._client._logger.warning(  # noqa: SLF001
                    "Clearing incompatible preferred format override for client %s",
                    self._client.client_id,
                )
                state.preferred_format_override = None
                state.preferred_codec_override = None

        self._preferred_format = AudioFormat(
            sample_rate=preferred_supported.sample_rate,
            bit_depth=preferred_supported.bit_depth,
            channels=preferred_supported.channels,
        )
        self._preferred_codec = preferred_supported.codec

    def _ensure_audio_requirements(self, *, force: bool = False) -> None:
        if self._audio_requirements is not None and not force:
            return

        support = self._client.info.player_support
        if support is None:
            self._audio_requirements = None
            return

        audio_format = self._preferred_format
        audio_codec = self._preferred_codec
        if audio_format is None or audio_codec is None:
            self._audio_requirements = None
            return

        group = self._client.group
        frame_duration_us = 25_000
        channel_id = group.get_channel_for_player(self._client.client_id)
        channel_id_int = channel_id.int
        transformer: FlacEncoder | OpusEncoder | PcmPassthrough
        if audio_codec == AudioCodec.FLAC:
            transformer = group.transformer_pool.get_or_create(
                FlacEncoder,
                channel_id=channel_id_int,
                sample_rate=audio_format.sample_rate,
                bit_depth=audio_format.bit_depth,
                channels=audio_format.channels,
                frame_duration_us=frame_duration_us,
            )
        elif audio_codec == AudioCodec.OPUS:
            transformer = group.transformer_pool.get_or_create(
                OpusEncoder,
                channel_id=channel_id_int,
                sample_rate=audio_format.sample_rate,
                bit_depth=audio_format.bit_depth,
                channels=audio_format.channels,
                frame_duration_us=frame_duration_us,
            )
        else:
            transformer = group.transformer_pool.get_or_create(
                PcmPassthrough,
                channel_id=channel_id_int,
                sample_rate=audio_format.sample_rate,
                bit_depth=audio_format.bit_depth,
                channels=audio_format.channels,
                frame_duration_us=frame_duration_us,
            )

        self._audio_requirements = AudioRequirements(
            sample_rate=audio_format.sample_rate,
            bit_depth=audio_format.bit_depth,
            channels=audio_format.channels,
            transformer=transformer,
            channel_id=channel_id,
            frame_duration_us=frame_duration_us,
        )
