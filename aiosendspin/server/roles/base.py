"""Base role classes and dataclasses.

This module contains:
- StreamRequirements: Declaration that a role sends binary streams
- AudioChunk: Audio data delivered to roles
- AudioRequirements: Declaration that a role needs audio chunks
- Role: Abstract base class for all roles
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any
from uuid import UUID

from aiosendspin.models import BinaryMessageType, pack_binary_header_raw

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from aiosendspin.models import AudioCodec
    from aiosendspin.models.core import (
        ClientCommandPayload,
        ClientStatePayload,
        StreamRequestFormatPayload,
    )
    from aiosendspin.models.source import ClientStreamStartPayload
    from aiosendspin.models.types import ServerMessage
    from aiosendspin.server.audio import AudioFormat, BufferTracker
    from aiosendspin.server.audio_transformers import AudioTransformer
    from aiosendspin.server.client import SendspinClient
    from aiosendspin.server.events import ClientRoleEvent, GroupRoleEvent
    from aiosendspin.server.group import SendspinGroup


# Default startup lead time used when a role does not report its own. Matches the
# push stream's no-roles fallback so default behavior is unchanged.
DEFAULT_REQUIRED_LEAD_TIME_US = 250_000


@dataclass(frozen=True)
class BinaryHandling:
    """Policy for how binary messages should be handled by connection.

    Roles return this from get_binary_handling() to declare how the connection
    should handle their binary messages (late detection, etc).
    """

    drop_late: bool = False
    """Drop binary messages whose timestamp is in the past."""

    grace_period_us: int = 0
    """Grace period after stream start before dropping late messages."""

    buffer_track: bool = False
    """Track sent bytes in the role's buffer tracker."""


@dataclass(frozen=True)
class StreamRequirements:
    """Declaration that a role sends binary streams.

    Roles that return this from get_stream_requirements() will have a
    BufferTracker injected by the framework.
    """


@dataclass(frozen=True)
class AudioChunk:
    """Audio chunk delivered to roles."""

    data: bytes
    """Transformed audio bytes (PCM or encoded, depending on transformer)."""

    timestamp_us: int
    """Playback timestamp in microseconds."""

    duration_us: int
    """Duration of this chunk in microseconds."""

    byte_count: int
    """Size of data (for buffer tracking)."""

    @cached_property
    def packed(self) -> bytes:
        """Binary AUDIO_CHUNK frame, memoized so subscribers reuse one copy."""
        header = pack_binary_header_raw(BinaryMessageType.AUDIO_CHUNK.value, self.timestamp_us)
        return header + self.data


class GroupRole(ABC):
    """Group-level role coordination.

    GroupRole is the group-level API for a role family. Client Role instances
    subscribe when they connect and unsubscribe when they disconnect.

    GroupRole can:
    - Coordinate operations across all members (e.g., volume redistribution)
    - Own group-level state (e.g., current metadata)
    - Provide computed properties from member state (e.g., average volume)
    """

    role_family: str
    """Role family name this GroupRole coordinates (e.g., 'player')."""

    def __init__(self, group: SendspinGroup) -> None:
        """Initialize with reference to the owning group."""
        self._group = group
        self._members: list[Role] = []

    def subscribe(self, role: Role) -> None:
        """Add a client role as a member of this group role."""
        if role in self._members:
            return
        self._members.append(role)
        self.on_member_join(role)

    def unsubscribe(self, role: Role) -> None:
        """Remove a client role from this group role."""
        if role in self._members:
            self._members.remove(role)
            self.on_member_leave(role)

    def on_member_join(self, role: Role) -> None:  # noqa: B027
        """Handle member subscription (override for catch-up logic)."""

    def on_member_leave(self, role: Role) -> None:  # noqa: B027
        """Handle member unsubscription."""

    def on_client_added(self, client: SendspinClient) -> None:  # noqa: B027
        """Handle a client being added to this group.

        Called for ALL clients, not just those with matching roles.
        Use for cross-role coordination (e.g., controller subscribing to player volume).

        May fire multiple times for the same client (initial group assignment,
        then again after role negotiation on connect/reconnect). Implementations
        must be idempotent; ``negotiated_roles`` may be empty on the first call.
        """

    def on_client_removed(self, client: SendspinClient) -> None:  # noqa: B027
        """Handle a client being removed from this group."""

    def emit_group_event(self, event: GroupRoleEvent) -> None:
        """Emit a GroupRole event on the owning group's event stream."""
        self._group._signal_event(event)  # noqa: SLF001

    def get_group_volume(self) -> int | None:
        """Return group volume (0-100) if supported."""
        return None

    def get_group_muted(self) -> bool | None:
        """Return group mute state if supported."""
        return None

    def set_group_volume(self, _volume_level: int) -> bool | None:
        """Set group volume if supported, return True/False or None if unsupported."""
        return None

    def set_group_muted(self, _muted: bool) -> bool | None:  # noqa: FBT001
        """Set group mute state if supported, return True/False or None if unsupported."""
        return None


@dataclass(frozen=True)
class AudioRequirements:
    """Declaration that a role needs audio chunks.

    Roles that return this from get_audio_requirements() will receive
    audio via on_audio_chunk() calls from PushStream.
    """

    sample_rate: int
    """Target sample rate in Hz."""

    bit_depth: int
    """Target bit depth (8, 16, 24, 32)."""

    channels: int
    """Number of audio channels."""

    transformer: AudioTransformer | None = None
    """Optional transformer for encoding. None means raw PCM."""

    channel_id: UUID | None = None
    """Channel to receive audio from. None means main channel."""

    frame_duration_us: int | None = None
    """Requested output frame duration in microseconds (optional)."""

    transform_options: Mapping[str, str] | None = None
    """Optional transformer-specific options (e.g., codec settings)."""


class Role(ABC):
    """Base class for all roles.

    Roles encapsulate per-connection behavior for different client capabilities.
    Each role can declare its streaming requirements and receive framework-injected
    resources like BufferTracker.
    """

    _client: SendspinClient
    """Reference to the owning client."""

    _buffer_tracker: BufferTracker | None = None
    """Framework-injected buffer tracker for roles that stream binary data."""

    _stream_started: bool = False
    """Whether stream/start has been sent for this role."""

    _group_role: GroupRole | None = None
    """Reference to the subscribed GroupRole, if any."""

    # Timing state for binary handling (used by connection)
    _stream_start_time_us: int | None = None
    """Timestamp when stream started, for grace period calculation."""

    _last_late_log_s: float = 0.0
    """Monotonic time of last late-message log (for rate limiting logs)."""

    _late_skips_since_log: int = 0
    """Count of skipped late messages since last log."""

    @property
    @abstractmethod
    def role_id(self) -> str:
        """Versioned role identifier (e.g., 'player@v1')."""
        ...

    @property
    @abstractmethod
    def role_family(self) -> str:
        """Role family name for protocol messages (e.g., 'player', 'artwork')."""
        ...

    # --- Declarations ---

    def get_stream_requirements(self) -> StreamRequirements | None:
        """Return StreamRequirements if role sends binary streams, else None.

        Roles that return StreamRequirements will have a BufferTracker injected
        by the framework.
        """
        return None

    def get_audio_requirements(self) -> AudioRequirements | None:
        """Return AudioRequirements if role needs audio, else None.

        Roles that return AudioRequirements will receive audio chunks via
        on_audio_chunk() calls from PushStream.
        """
        return None

    def supports_preconnect_audio(self) -> bool:
        """Whether this role may receive audio before any transport has connected.

        Default is False and should remain False in almost all cases.
        Set this to True only for specialized roles that must continue receiving
        on_stream_start()/on_audio_chunk() while the owning client is intentionally
        disconnected and registered through SendspinServer.register_external_player().
        """
        return False

    def replay_from_pcm_cache(self) -> bool:
        """Whether late-join should replay cached PCM even when a live resampler exists.

        Audio-playback roles skip historical replay when another live role
        already drives a resampler at their target PCM shape, because
        re-running that resampler would shift the live role's samples across
        the hand-off. Analysis-only roles (e.g. visualizer) do not feed a
        synchronized output, so the seam is inaudible; they override this to
        True so a mid-stream join is fed the buffered PCM immediately instead
        of waiting for the next live commit (which may sit far ahead of the
        playhead behind a producer buffer).
        """
        return False

    def get_binary_handling(self, message_type: int) -> BinaryHandling | None:  # noqa: ARG002
        """Return handling policy for a binary message type, or None if not handled.

        The connection calls this to determine how to handle binary messages:
        - Whether to drop late messages
        - Whether to rate-limit delivery
        - Whether to track in buffer tracker
        """
        return None

    def handles_inbound_binary(self, message_type: int) -> bool:  # noqa: ARG002
        """Return whether this role consumes an inbound binary message type."""
        return False

    def get_buffer_tracker(self) -> BufferTracker | None:
        """Return the role-owned buffer tracker, if any."""
        return self._buffer_tracker

    def get_output_delay_us(self) -> int:
        """Return transport delay in microseconds applied by this role (default: 0)."""
        return 0

    def get_required_lead_time_us(self) -> int:
        """Return the startup lead time this role needs before the first audio chunk."""
        return DEFAULT_REQUIRED_LEAD_TIME_US

    def get_min_buffer_us(self) -> int:
        """Return the minimum ongoing buffer duration this role wants during playback."""
        return 0

    def get_join_delay_s(self) -> float:
        """Return the join delay in seconds for reconnects (default: 0)."""
        return 0.0

    def get_player_volume(self) -> int | None:
        """Return player volume if supported by this role."""
        return None

    def get_player_muted(self) -> bool | None:
        """Return player mute state if supported by this role."""
        return None

    def set_player_volume(self, volume: int) -> None:  # noqa: ARG002
        """Set player volume if supported by this role."""
        return

    def set_player_mute(self, muted: bool) -> None:  # noqa: ARG002, FBT001
        """Set player mute if supported by this role."""
        return

    def get_supported_formats(self) -> list[Any] | None:
        """Return formats both client and server support, in client priority order."""
        return None

    def set_preferred_format(
        self,
        audio_format: AudioFormat | None,  # noqa: ARG002
        codec: AudioCodec | None = None,  # noqa: ARG002
    ) -> bool:
        """Set or clear preferred format override. Returns True on success."""
        return False

    def reset_binary_timing(self) -> None:
        """Reset timing/log state for binary handling at stream boundaries."""
        self._stream_start_time_us = None
        self._last_late_log_s = 0.0
        self._late_skips_since_log = 0

    # --- Framework-provided send methods ---

    def has_connection(self) -> bool:
        """Return True when the client currently has an active transport."""
        return self._client.connection is not None

    def send_message(self, message: ServerMessage) -> None:
        """Send JSON message to the client. Drop silently if no transport."""
        if not self.has_connection():
            return
        self._client.send_role_message(self.role_family, message)

    def emit_client_event(self, event: ClientRoleEvent) -> None:
        """Emit a role event on the owning client's event stream."""
        self._client._signal_event(event)  # noqa: SLF001

    # --- Stream lifecycle hooks (optional) ---

    def on_stream_start(self) -> None:  # noqa: B027
        """Handle stream start before first audio chunk."""

    def on_audio_chunk(self, chunk: AudioChunk) -> None:  # noqa: B027
        """Receive audio chunk."""

    def on_stream_clear(self) -> None:  # noqa: B027
        """Handle seek/clear by discarding buffered audio."""

    def on_stream_end(self) -> None:  # noqa: B027
        """Handle stream stop."""

    def on_binary_chunk(self, message_type: int, timestamp_us: int, data: bytes) -> None:  # noqa: B027
        """Handle a declared inbound binary message."""

    # --- Lifecycle hooks ---

    @abstractmethod
    def on_connect(self) -> None:
        """Handle connection establishment.

        Implementations should call _subscribe_to_group_role() to subscribe
        to the corresponding GroupRole.
        """

    @abstractmethod
    def on_disconnect(self) -> None:
        """Handle connection close.

        Implementations should call _unsubscribe_from_group_role() to unsubscribe
        from the corresponding GroupRole.
        """

    def on_deactivate(self) -> None:
        """Handle the role leaving active_roles while the client stays connected."""
        self._unsubscribe_from_group_role()

    def _subscribe_to_group_role(self) -> None:
        """Subscribe to the corresponding GroupRole (call from on_connect)."""
        if group_role := self._client.group.group_role(self.role_family):
            group_role.subscribe(self)
            self._group_role = group_role

    def _unsubscribe_from_group_role(self) -> None:
        """Unsubscribe from the GroupRole (call from on_disconnect / on_deactivate)."""
        if self._group_role:
            self._group_role.unsubscribe(self)
            self._group_role = None

    def requires_initial_state(self) -> bool:
        """Whether this role requires initial client/state before being 'connected'.

        Roles that return True will block the connection's "connected" status
        until their initial state subobject is received in client/state.
        """
        return False

    def initial_state_deviations(self, payload: ClientStatePayload) -> list[str]:  # noqa: ARG002
        """Spec requirements this role's part of the initial client/state does not meet."""
        return []

    def client_state_deviations(self, payload: ClientStatePayload) -> list[str]:  # noqa: ARG002
        """Spec violations in any client/state for this role, checked before it is applied."""
        return []

    def on_group_changed(self, group: object) -> None:  # noqa: ARG002
        """Handle group changes by re-subscribing to the new GroupRole."""
        self._unsubscribe_from_group_role()
        # Subscribing fires on_member_join (a state push); defer to on_connect when
        # there is no transport so the push lands on the live connection, not nowhere.
        if self.has_connection():
            self._subscribe_to_group_role()

    def on_availability_changed(
        self,
        old_available: bool,  # noqa: ARG002, FBT001
        new_available: bool,  # noqa: ARG002, FBT001
    ) -> Coroutine[Any, Any, None] | None:
        """Handle a client availability change.

        Return a coroutine if async work is needed, else None.
        Called when client/state reports a new availability.
        """
        return None

    # --- Message hooks ---

    def on_client_state(self, payload: ClientStatePayload) -> None:  # noqa: B027
        """Handle client/state payload."""

    def on_stream_request_format(  # noqa: B027
        self,
        payload: StreamRequestFormatPayload,
    ) -> None:
        """Handle stream/request-format payload."""

    def on_command(self, payload: ClientCommandPayload) -> None:  # noqa: B027
        """Handle client/command payload.

        Handlers must be synchronous. For async operations, launch eager tasks.
        """

    def on_client_stream_start(self, payload: ClientStreamStartPayload) -> None:  # noqa: B027
        """Handle client_stream/start."""

    def on_client_stream_end(self) -> None:  # noqa: B027
        """Handle client_stream/end."""
