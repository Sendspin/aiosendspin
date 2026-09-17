"""Role implementations for connection-specific behavior.

This package contains the role implementations:
- Base role classes (Role ABC, dataclasses)
- Specific role implementations (PlayerV1Role, ControllerV1Role, etc.)

Roles encapsulate per-connection behavior for different client capabilities.
"""

# Import submodules to trigger auto-registration of roles
from aiosendspin.server.roles import (
    artwork,  # noqa: F401
    color,  # noqa: F401
    controller,  # noqa: F401
    metadata,  # noqa: F401
    player,  # noqa: F401
    source,  # noqa: F401
    visualizer,  # noqa: F401
)

# Re-export role classes for convenience
from aiosendspin.server.roles.artwork import (
    ArtworkClearedEvent,
    ArtworkEvent,
    ArtworkGroupRole,
    ArtworkUpdatedEvent,
    ArtworkV1Role,
)
from aiosendspin.server.roles.base import (
    AudioChunk,
    AudioRequirements,
    GroupRole,
    Role,
    StreamRequirements,
)
from aiosendspin.server.roles.color import (
    ColorClearedEvent,
    ColorEvent,
    ColorGroupRole,
    ColorUpdatedEvent,
    ColorV1Role,
)
from aiosendspin.server.roles.controller import (
    ControllerEvent,
    ControllerGroupRole,
    ControllerMuteEvent,
    ControllerNextEvent,
    ControllerPauseEvent,
    ControllerPlayEvent,
    ControllerPreviousEvent,
    ControllerRepeatEvent,
    ControllerSeekEvent,
    ControllerSeekRelativeEvent,
    ControllerShuffleEvent,
    ControllerStopEvent,
    ControllerSwitchEvent,
    ControllerV1Role,
    ControllerVolumeEvent,
)
from aiosendspin.server.roles.metadata import (
    MetadataClearedEvent,
    MetadataEvent,
    MetadataGroupRole,
    MetadataUpdatedEvent,
    MetadataV1Role,
)
from aiosendspin.server.roles.player import (
    PlayerGroupEvent,
    PlayerGroupMuteChangedEvent,
    PlayerGroupRole,
    PlayerGroupVolumeChangedEvent,
    PlayerV1Role,
)
from aiosendspin.server.roles.registry import register_role
from aiosendspin.server.roles.source import (
    SourceSignalChangedEvent,
    SourceStream,
    SourceStreamEndedEvent,
    SourceStreamStartedEvent,
    SourceV1Role,
)
from aiosendspin.server.roles.visualizer import (
    VisualizerGroupRole,
    VisualizerV1Role,
)

# DEPRECATED(spec-pr-86): remove in aiosendspin <version>
from aiosendspin.server.roles.visualizer_draft_r1 import VisualizerDraftR1Role

__all__ = [
    "ArtworkClearedEvent",
    "ArtworkEvent",
    "ArtworkGroupRole",
    "ArtworkUpdatedEvent",
    "ArtworkV1Role",
    "AudioChunk",
    "AudioRequirements",
    "ColorClearedEvent",
    "ColorEvent",
    "ColorGroupRole",
    "ColorUpdatedEvent",
    "ColorV1Role",
    "ControllerEvent",
    "ControllerGroupRole",
    "ControllerMuteEvent",
    "ControllerNextEvent",
    "ControllerPauseEvent",
    "ControllerPlayEvent",
    "ControllerPreviousEvent",
    "ControllerRepeatEvent",
    "ControllerSeekEvent",
    "ControllerSeekRelativeEvent",
    "ControllerShuffleEvent",
    "ControllerStopEvent",
    "ControllerSwitchEvent",
    "ControllerV1Role",
    "ControllerVolumeEvent",
    "GroupRole",
    "MetadataClearedEvent",
    "MetadataEvent",
    "MetadataGroupRole",
    "MetadataUpdatedEvent",
    "MetadataV1Role",
    "PlayerGroupEvent",
    "PlayerGroupMuteChangedEvent",
    "PlayerGroupRole",
    "PlayerGroupVolumeChangedEvent",
    "PlayerV1Role",
    "Role",
    "SourceSignalChangedEvent",
    "SourceStream",
    "SourceStreamEndedEvent",
    "SourceStreamStartedEvent",
    "SourceV1Role",
    "StreamRequirements",
    # DEPRECATED(spec-pr-86): remove in aiosendspin <version>
    "VisualizerDraftR1Role",
    "VisualizerGroupRole",
    "VisualizerV1Role",
    "register_role",
]
