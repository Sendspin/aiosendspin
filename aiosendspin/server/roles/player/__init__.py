"""Player role - client and group level."""

from aiosendspin.models.player import ClientHelloPlayerSupport
from aiosendspin.server.roles.player.audio_transformers import FlacEncoder, PcmPassthrough
from aiosendspin.server.roles.player.events import (
    MinBufferChangedEvent,
    OutputDelayChangedEvent,
    PlayerGroupEvent,
    PlayerGroupMuteChangedEvent,
    PlayerGroupVolumeChangedEvent,
    RequiredLeadTimeChangedEvent,
    VolumeChangedEvent,
)
from aiosendspin.server.roles.player.group import PlayerGroupRole
from aiosendspin.server.roles.player.types import PlayerRoleProtocol
from aiosendspin.server.roles.player.v1 import PlayerV1Role
from aiosendspin.server.roles.registry import (
    RoleSupportSpec,
    register_group_role,
    register_role,
    register_role_support_spec,
)

register_group_role("player", PlayerGroupRole)
register_role("player@v1", lambda client: PlayerV1Role(client=client))
register_role_support_spec(
    "player",
    RoleSupportSpec(
        parse_support=ClientHelloPlayerSupport.from_dict,
    ),
)

__all__ = [
    "FlacEncoder",
    "MinBufferChangedEvent",
    "OutputDelayChangedEvent",
    "PcmPassthrough",
    "PlayerGroupEvent",
    "PlayerGroupMuteChangedEvent",
    "PlayerGroupRole",
    "PlayerGroupVolumeChangedEvent",
    "PlayerRoleProtocol",
    "PlayerV1Role",
    "RequiredLeadTimeChangedEvent",
    "VolumeChangedEvent",
]
