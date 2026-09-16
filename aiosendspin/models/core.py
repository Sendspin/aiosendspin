"""
Core messages for the Sendspin protocol.

This module contains the fundamental messages that establish communication between
clients and the server. These messages handle initial handshakes, ongoing clock
synchronization, stream lifecycle management, and role-based state updates and commands.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Annotated, Any, ClassVar, Literal

from mashumaro.types import Alias

from .artwork import (
    ClientHelloArtworkSupport,
    StreamRequestFormatArtwork,
    StreamStartArtwork,
)
from .base import SendspinConfig, SendspinModel
from .color import SessionUpdateColor
from .controller import ControllerCommandPayload, ControllerStatePayload
from .metadata import SessionUpdateMetadata
from .player import (
    ClientHelloPlayerSupport,
    PlayerCommandPayload,
    PlayerStatePayload,
    StreamRequestFormatPlayer,
    StreamStartPlayer,
)
from .source import (
    ClientHelloSourceSupport,
    SourceCommandServerPayload,
    SourceStatePayload,
)
from .types import (
    Activity,
    ClientMessage,
    ConnectionReason,
    GoodbyeReason,
    PairMethod,
    PlaybackStateType,
    Roles,
    ServerMessage,
    UndefinedField,
    undefined_field,
)
from .visualizer import (
    ClientHelloVisualizerSupport,
    StreamRequestFormatVisualizer,
    StreamStartVisualizer,
)
from .visualizer_draft_r1 import (
    ClientHelloVisualizerSupport as ClientHelloVisualizerSupportDraftR1,
)
from .visualizer_draft_r1 import (
    StreamStartVisualizer as StreamStartVisualizerDraftR1,
)


def _has_merge_value(value: Any) -> bool:
    """Return whether a field value should overwrite the existing value during merge."""
    return not isinstance(value, UndefinedField)


def _merge_optional_field_value(existing: Any, incoming: Any) -> Any:
    """Merge one field value, recursively merging nested dataclasses when both are present."""
    if not _has_merge_value(incoming):
        return existing
    if (
        incoming is not None
        and _has_merge_value(existing)
        and is_dataclass(existing)
        and is_dataclass(incoming)
    ):
        return _merge_optional_dataclass_fields(existing, incoming)
    return incoming


def _merge_optional_dataclass_fields(existing: Any, incoming: Any) -> Any:
    """Merge dataclass instances by preferring incoming values that are actually present."""
    merged_values = {
        field.name: _merge_optional_field_value(
            getattr(existing, field.name),
            getattr(incoming, field.name),
        )
        for field in fields(existing)
    }
    return type(existing)(**merged_values)


@dataclass
class DeviceInfo(SendspinModel):
    """Optional information about the device."""

    product_name: str | None = None
    """Device model/product name."""
    manufacturer: str | None = None
    """Device manufacturer name."""
    software_version: str | None = None
    """Software version of the client (not the Sendspin version)."""
    mac_address: str | None = None
    """MAC address of the connection's network interface, lowercase colon-separated."""

    class Config(SendspinConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class PairMethodDescriptor(SendspinModel):
    """A pairing method a client offers in client/hello."""

    method: PairMethod
    """The pairing method identifier."""
    formats: list[str] | None = None
    """For dynamic_pairing_code only: emission formats offered by the client."""
    out_channels: list[str] | None = None
    """For dynamic_pairing_code only: channels through which the code is conveyed."""
    locations: list[str] | None = None
    """For static_pairing_code and pairing_psk only: where the operator finds the secret."""

    class Config(SendspinConfig):
        """Omit method-specific fields where they do not apply."""

        omit_none = True


@dataclass
class UnpairedAccess(SendspinModel):
    """Whether the client currently admits unpaired access."""

    enabled: bool = False


# Client -> Server: client/hello
@dataclass
class ClientHelloPayload(SendspinModel):
    """Information about a connected client."""

    name: str
    """Friendly name of the client."""
    supported_roles: list[str]
    """List of versioned role IDs the client supports (e.g., 'player@v1')."""
    device_info: DeviceInfo | None = None
    """Optional information about the device."""
    client_id: str | None = None
    """Client identifier. Omitted under encryption (taken from client/init); sent by
    legacy unencrypted clients."""
    version: int | None = None
    """Core protocol version. Omitted under encryption (taken from client/init)."""
    player_support: Annotated[ClientHelloPlayerSupport | None, Alias("player@v1_support")] = None
    """Player support configuration - only if player role is in supported_roles."""
    artwork_support: Annotated[ClientHelloArtworkSupport | None, Alias("artwork@v1_support")] = None
    """Artwork support configuration - only if artwork role is in supported_roles."""
    visualizer_support: Annotated[
        ClientHelloVisualizerSupport | None, Alias("visualizer@v1_support")
    ] = None
    """Visualizer support configuration - only if visualizer@v1 role is in supported_roles."""
    visualizer_draft_r1_support: Annotated[
        ClientHelloVisualizerSupportDraftR1 | None, Alias("visualizer@_draft_r1_support")
    ] = None
    """Visualizer support for clients on the legacy `visualizer@_draft_r1` wire."""
    supported_pair_methods: list[PairMethodDescriptor] | None = None
    """Pairing methods this client offers."""
    unpaired_access: UnpairedAccess = field(default_factory=UnpairedAccess)
    """Whether this client currently admits unpaired access."""
    legacy_support_keys_used: list[str] | None = None
    """Unversioned support keys the parser rewrote to versioned aliases, recorded for
    the server to flag. Not part of the wire schema (omitted when None)."""
    unlisted_support_roles: list[str] | None = None
    """Roles whose support object was provided without listing the role in
    ``supported_roles`` (dropped during parse), recorded for the server to flag.
    Not part of the wire schema (omitted when None)."""
    source_support: Annotated[ClientHelloSourceSupport | None, Alias("source@v1_support")] = None
    """Source support configuration."""

    # Static mapping: unversioned support key -> actual alias key.
    _SUPPORT_KEY_ALIASES: ClassVar[dict[str, str]] = {
        "player_support": "player@v1_support",
        "artwork_support": "artwork@v1_support",
        "visualizer_support": "visualizer@v1_support",
        "visualizer_draft_r1_support": "visualizer@_draft_r1_support",
    }

    @classmethod
    def __pre_deserialize__(cls, d: dict[str, Any]) -> dict[str, Any]:
        """Rewrite legacy unversioned support keys to versioned aliases, recording which."""
        normalized = dict(d)
        legacy_keys: list[str] = []
        for legacy_key, versioned_key in cls._SUPPORT_KEY_ALIASES.items():
            if legacy_key not in normalized:
                continue
            legacy_keys.append(legacy_key)
            value = normalized.pop(legacy_key)
            # Rewrite to the versioned alias only when the client didn't also send it.
            if versioned_key not in normalized:
                normalized[versioned_key] = value
        # Always overwrite so a client cannot spoof the record via the wire.
        normalized["legacy_support_keys_used"] = legacy_keys or None
        return normalized

    def __post_init__(self) -> None:
        """Enforce that support configs match supported roles."""
        # Validate player role and support configuration
        # Require support objects only for the exact role version we parse (e.g. "player@v1").
        # Clients may advertise newer versions (e.g. "player@v2") which this server may not
        # implement. Those must not trigger v1 support requirements.
        unlisted: list[str] = []
        player_role_supported = Roles.PLAYER.value in self.supported_roles
        if player_role_supported and self.player_support is None:
            raise ValueError(
                "player@v1_support (player_support alias) must be provided when "
                "'player@v1' is in supported_roles"
            )
        if not player_role_supported:
            if self.player_support is not None:
                unlisted.append(Roles.PLAYER.value)
            self.player_support = None

        # Validate artwork role and support configuration
        artwork_role_supported = Roles.ARTWORK.value in self.supported_roles
        if artwork_role_supported and self.artwork_support is None:
            raise ValueError(
                "artwork@v1_support (artwork_support alias) must be provided when "
                "'artwork@v1' is in supported_roles"
            )
        if not artwork_role_supported:
            if self.artwork_support is not None:
                unlisted.append(Roles.ARTWORK.value)
            self.artwork_support = None

        # Validate visualizer role and support configuration.
        visualizer_role_supported = Roles.VISUALIZER.value in self.supported_roles
        if visualizer_role_supported and self.visualizer_support is None:
            raise ValueError(
                "visualizer@v1_support (visualizer_support alias) must be "
                "provided when 'visualizer@v1' is in supported_roles"
            )
        if not visualizer_role_supported:
            if self.visualizer_support is not None:
                unlisted.append(Roles.VISUALIZER.value)
            self.visualizer_support = None

        # Validate legacy `visualizer@_draft_r1` support configuration.
        visualizer_draft_supported = "visualizer@_draft_r1" in self.supported_roles
        if visualizer_draft_supported and self.visualizer_draft_r1_support is None:
            raise ValueError(
                "visualizer@_draft_r1_support must be provided when "
                "'visualizer@_draft_r1' is in supported_roles"
            )
        if not visualizer_draft_supported:
            if self.visualizer_draft_r1_support is not None:
                unlisted.append("visualizer@_draft_r1")
            self.visualizer_draft_r1_support = None

        source_role_supported = Roles.SOURCE.value in self.supported_roles
        if source_role_supported and self.source_support is None:
            raise ValueError(
                "source@v1_support (source_support alias) must be provided when "
                "'source@v1' is in supported_roles"
            )
        if not source_role_supported:
            if self.source_support is not None:
                unlisted.append(Roles.SOURCE.value)
            self.source_support = None

        # Overwrite so a client cannot spoof the record via the wire.
        self.unlisted_support_roles = unlisted or None

    class Config(SendspinConfig):
        """Config for parsing json messages."""

        omit_none = True
        serialize_by_alias = True


@dataclass
class ClientHelloMessage(ClientMessage):
    """Message sent by the client to identify itself."""

    payload: ClientHelloPayload
    type: Literal["client/hello"] = "client/hello"


# Client -> Server: client/time
@dataclass
class ClientTimePayload(SendspinModel):
    """Timing information from the client."""

    client_transmitted: int
    """Client's internal clock timestamp in microseconds."""


@dataclass
class ClientTimeMessage(ClientMessage):
    """Message sent by the client for time synchronization."""

    payload: ClientTimePayload
    type: Literal["client/time"] = "client/time"


# Client -> Server: client/state
@dataclass
class ClientStatePayload(SendspinModel):
    """Client sends state updates to the server."""

    available: bool | None = None
    """
    Whether the client is available to participate in Sendspin playback.

    - true: operational and ready; for a player, its clock is synchronized.
    - false: output is in use by an external system, not currently participating.
    """
    player: PlayerStatePayload | None = None
    """Player state - only if client has player role."""
    legacy_state_used: bool | None = None
    """Set when the parser read a legacy top-level `state` field, recorded for the server
    to flag. Not part of the wire schema (omitted when None)."""
    source: SourceStatePayload | None = None
    """Source state."""

    @classmethod
    def __pre_deserialize__(cls, d: dict[str, Any]) -> dict[str, Any]:
        """Normalize a legacy `state` enum to `available`, recording that it was used."""
        d = dict(d)
        legacy_state = "state" in d
        if d.get("available") is None and legacy_state:
            d["available"] = d["state"] != "external_source"
        # Always overwrite so a client cannot spoof the record via the wire.
        d["legacy_state_used"] = legacy_state or None
        return d

    class Config(SendspinConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class ClientStateMessage(ClientMessage):
    """Message sent by the client to report state changes."""

    payload: ClientStatePayload
    type: Literal["client/state"] = "client/state"


# Client -> Server: client/command
@dataclass
class ClientCommandPayload(SendspinModel):
    """Client sends commands to the server."""

    controller: ControllerCommandPayload | None = None
    """Controller commands - only if client has controller role."""

    class Config(SendspinConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class ClientCommandMessage(ClientMessage):
    """Message sent by the client to send commands."""

    payload: ClientCommandPayload
    type: Literal["client/command"] = "client/command"


# Client -> Server: client/goodbye
@dataclass
class ClientGoodbyePayload(SendspinModel):
    """Payload for client goodbye message."""

    reason: GoodbyeReason
    """Reason for disconnecting."""


@dataclass
class ClientGoodbyeMessage(ClientMessage):
    """Message sent by the client before gracefully closing the connection."""

    payload: ClientGoodbyePayload
    type: Literal["client/goodbye"] = "client/goodbye"


# Server -> Client: server/hello
@dataclass
class ServerHelloPayload(SendspinModel):
    """Information about the server."""

    name: str
    """Friendly name of the server"""


@dataclass
class ServerHelloMessage(ServerMessage):
    """Message sent by the server to identify itself."""

    payload: ServerHelloPayload
    type: Literal["server/hello"] = "server/hello"


# Legacy (transition-mode) server/hello, for unencrypted clients that predate
# server/activate. Standalone (NOT a ServerMessage subtype): it serializes to the
# same ``type: "server/hello"`` as the encrypted-path message above, so keeping it
# out of the discriminated union avoids an ambiguous dispatch. The server only ever
# serializes and sends it; our own client always speaks the encrypted path and so
# never deserializes it.
@dataclass
class LegacyServerHelloPayload(SendspinModel):
    """Server identity for a legacy unencrypted connection (no server/activate)."""

    server_id: str
    """Identifier of the server."""
    name: str
    """Friendly name of the server."""
    version: int
    """Version of the core message format (independent of role versions)."""
    connection_reason: ConnectionReason
    """Reason for this connection (relevant for multi-server environments)."""
    active_roles: list[str]
    """Versioned role IDs active for this client (e.g., 'player@v1')."""
    selected_pair_method: PairMethod | None = None
    """Pairing method the server picked; present when connection_reason is 'pairing'."""

    class Config(SendspinConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class LegacyServerHelloMessage(SendspinModel):
    """Legacy server/hello for transition-mode (unencrypted) clients."""

    payload: LegacyServerHelloPayload
    type: Literal["server/hello"] = "server/hello"


# Server -> Client: server/activate
@dataclass
class ActivatePairing(SendspinModel):
    """Parameters of the pairing attempt a server/activate admits."""

    method: PairMethod
    """Pairing method the server picked, drawn from the client's supported_pair_methods."""
    format: str | None = None
    """The dynamic pairing-code emission format; required for dynamic_pairing_code."""
    languages: list[str] | None = None
    """BCP 47 tags in descending operator preference, for spoken pairing-code emission."""

    class Config(SendspinConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class ServerActivatePayload(SendspinModel):
    """Declares the server's current purpose on this connection."""

    activities: list[Activity]
    """The set of currently-active purposes on this connection. May be empty."""
    active_roles: list[str] | None = None
    """Versioned role IDs active for this client (e.g., 'player@v1'). Required on
    connections capable of playback; absent otherwise. Persists across subsequent
    server/activate messages that omit it."""
    pairing: ActivatePairing | None = None
    """Parameters of the admitted pairing attempt. Required when 'pairing' is in activities."""

    class Config(SendspinConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class ServerActivateMessage(ServerMessage):
    """Message sent by the server to declare its active purpose on this connection."""

    payload: ServerActivatePayload
    type: Literal["server/activate"] = "server/activate"


# Server -> Client: server/time
@dataclass
class ServerTimePayload(SendspinModel):
    """Timing information from the server."""

    client_transmitted: int
    """Client's internal clock timestamp received in the client/time message"""
    server_received: int
    """Timestamp that the server received the client/time message in microseconds"""
    server_transmitted: int
    """Timestamp that the server transmitted this message in microseconds"""


@dataclass
class ServerTimeMessage(ServerMessage):
    """Message sent by the server for time synchronization."""

    payload: ServerTimePayload
    type: Literal["server/time"] = "server/time"


# Server -> Client: server/state
@dataclass
class ServerStatePayload(SendspinModel):
    """Server sends state updates to the client."""

    metadata: SessionUpdateMetadata | None | UndefinedField = field(default_factory=undefined_field)
    """Metadata state - only sent to clients with metadata role."""
    controller: ControllerStatePayload | None | UndefinedField = field(
        default_factory=undefined_field
    )
    """Controller state - only sent to clients with controller role."""
    color: SessionUpdateColor | None | UndefinedField = field(default_factory=undefined_field)
    """Color state - only sent to clients with color role."""

    class Config(SendspinConfig):
        """Config for parsing json messages."""

        omit_default = True


@dataclass
class ServerStateMessage(ServerMessage):
    """Message sent by the server to send state updates."""

    payload: ServerStatePayload
    type: Literal["server/state"] = "server/state"

    def merge(self, other: ServerMessage) -> ServerMessage | None:
        """Merge with another server/state message, preferring non-null incoming fields."""
        if not isinstance(other, ServerStateMessage):
            return None

        return ServerStateMessage(_merge_optional_dataclass_fields(self.payload, other.payload))


# Server -> Client: group/update
@dataclass
class GroupUpdateServerPayload(SendspinModel):
    """State update of the group this client is part of."""

    playback_state: PlaybackStateType | None = None
    """Playback state of the group."""
    group_id: str | None = None
    """Group identifier."""
    group_name: str | None = None
    """Friendly name of the group."""

    class Config(SendspinConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class GroupUpdateServerMessage(ServerMessage):
    """Message sent by the server to update group state."""

    payload: GroupUpdateServerPayload
    type: Literal["group/update"] = "group/update"

    def merge(self, other: ServerMessage) -> ServerMessage | None:
        """Merge with another group/update message, preferring defined incoming fields."""
        if not isinstance(other, GroupUpdateServerMessage):
            return None

        merged_payload = _merge_optional_dataclass_fields(self.payload, other.payload)
        return GroupUpdateServerMessage(merged_payload)


# Server -> Client: server/command
@dataclass
class ServerCommandPayload(SendspinModel):
    """Server sends commands to the client."""

    player: PlayerCommandPayload | None = None
    """Player commands - only sent to clients with player role."""
    source: SourceCommandServerPayload | None = None
    """Source command - only sent to clients with source role."""

    class Config(SendspinConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class ServerCommandMessage(ServerMessage):
    """Message sent by the server to send commands to the client."""

    payload: ServerCommandPayload
    type: Literal["server/command"] = "server/command"


# Shape carried by `StreamStartPayload.visualizer`. The field is typed `Any`
# so mashumaro defers to the dispatch hooks below, but callers should annotate
# against this alias for static checking.
StreamStartVisualizerLike = StreamStartVisualizer | StreamStartVisualizerDraftR1 | None


def _serialize_stream_start_visualizer(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, (StreamStartVisualizer, StreamStartVisualizerDraftR1)):
        raise TypeError(
            "StreamStartPayload.visualizer must be a StreamStartVisualizer, "
            f"StreamStartVisualizerDraftR1, or None; got {type(value).__name__}"
        )
    return value.to_dict()


def _deserialize_stream_start_visualizer(
    value: Any,
) -> StreamStartVisualizer | StreamStartVisualizerDraftR1 | None:
    """Pick the visualizer wire schema by its discriminating field.

    The v1 and draft schemas overlap on `types`/`spectrum` and share the
    class name `StreamStartVisualizer`, so mashumaro's bare-union resolution
    cannot tell them apart (it yields None / raises for a draft payload).
    They are distinguished by the required `rate_max` (v1) vs `batch_max`
    (draft); dispatch explicitly. The field is annotated `Any` so mashumaro
    defers to these hooks instead of generating union codec.
    """
    if not isinstance(value, dict):
        return None
    if "batch_max" in value and "rate_max" not in value:
        return StreamStartVisualizerDraftR1.from_dict(value)
    return StreamStartVisualizer.from_dict(value)


# Server -> Client: stream/start
@dataclass
class StreamStartPayload(SendspinModel):
    """Information about an active streaming session."""

    server_transmitted: int = 0
    """Timestamp the server transmitted this message in microseconds. Stamped at send."""
    player: StreamStartPlayer | None = None
    """Information about the player."""
    artwork: StreamStartArtwork | None = None
    """Artwork information (sent to clients with artwork role)."""
    # Typed `Any` (rather than `StreamStartVisualizerLike`) so mashumaro defers
    # to the explicit serialize/deserialize hooks; the bare union cannot
    # disambiguate the two same-named schemas. The serialize hook rejects
    # anything other than the alias's members at runtime.
    visualizer: Any = field(
        default=None,
        metadata={
            "serialize": _serialize_stream_start_visualizer,
            "deserialize": _deserialize_stream_start_visualizer,
        },
    )
    """Visualizer information (sent to clients with visualizer role).

    Carries the v1 schema by default; legacy clients on `visualizer@_draft_r1`
    get the draft schema. Roles emit whichever matches their negotiated wire.
    """

    class Config(SendspinConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class StreamStartMessage(ServerMessage):
    """Message sent by the server to start a stream."""

    payload: StreamStartPayload
    type: Literal["stream/start"] = "stream/start"


# Role family names that support stream/clear (have buffers to clear).
STREAM_CLEAR_ROLE_FAMILIES = frozenset({"player", "visualizer"})

# Role family names that support stream/end.
STREAM_END_ROLE_FAMILIES = frozenset({"player", "artwork", "visualizer"})


# Server -> Client: stream/clear
@dataclass
class StreamClearPayload(SendspinModel):
    """Instructs clients to clear buffers without ending the stream."""

    server_transmitted: int = 0
    """Timestamp the server transmitted this message in microseconds. Stamped at send."""
    roles: list[str] | None = None
    """Roles to clear: player, visualizer, or both. If omitted, clears both roles."""

    def __post_init__(self) -> None:
        """Validate role names. Permits known families and `_`-prefixed app roles."""
        if self.roles is not None:
            invalid_roles = {
                role
                for role in self.roles
                if role not in STREAM_CLEAR_ROLE_FAMILIES and not role.startswith("_")
            }
            if invalid_roles:
                supported = sorted(STREAM_CLEAR_ROLE_FAMILIES)
                invalid = sorted(invalid_roles)
                raise ValueError(
                    f"stream/clear only supports roles {supported} or `_`-prefixed "
                    f"application roles, got invalid roles: {invalid}"
                )

    class Config(SendspinConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class StreamClearMessage(ServerMessage):
    """Message sent by the server to clear stream buffers (e.g., for seek operations)."""

    payload: StreamClearPayload
    type: Literal["stream/clear"] = "stream/clear"


# Client -> Server: stream/request-format
@dataclass
class StreamRequestFormatPayload(SendspinModel):
    """Request different stream format (upgrade or downgrade)."""

    player: StreamRequestFormatPlayer | None = None
    """Player format request (only for clients with player role)."""
    artwork: StreamRequestFormatArtwork | None = None
    """Artwork format request (only for clients with artwork role)."""
    visualizer: StreamRequestFormatVisualizer | None = None
    """Visualizer format request (only for clients with visualizer role)."""

    class Config(SendspinConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class StreamRequestFormatMessage(ClientMessage):
    """Message sent by the client to request different stream format."""

    payload: StreamRequestFormatPayload
    type: Literal["stream/request-format"] = "stream/request-format"


# Server -> Client: stream/end
@dataclass
class StreamEndPayload(SendspinModel):
    """Payload for stream/end message."""

    server_transmitted: int = 0
    """Timestamp the server transmitted this message in microseconds. Stamped at send."""
    roles: list[str] | None = None
    """Roles to end streams for. If omitted, ends all active streams."""

    def __post_init__(self) -> None:
        """Validate role names. Permits known families and `_`-prefixed app roles."""
        if self.roles is not None:
            invalid_roles = {
                role
                for role in self.roles
                if role not in STREAM_END_ROLE_FAMILIES and not role.startswith("_")
            }
            if invalid_roles:
                supported = sorted(STREAM_END_ROLE_FAMILIES)
                invalid = sorted(invalid_roles)
                raise ValueError(
                    f"stream/end only supports roles {supported} or `_`-prefixed "
                    f"application roles, got invalid roles: {invalid}"
                )

    class Config(SendspinConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class StreamEndMessage(ServerMessage):
    """Message sent by the server to end a stream."""

    payload: StreamEndPayload
    type: Literal["stream/end"] = "stream/end"
