"""Role registry for server-side role factories."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from aiosendspin.server.roles.base import GroupRole

if TYPE_CHECKING:
    from aiosendspin.server.client import SendspinClient
    from aiosendspin.server.group import SendspinGroup
    from aiosendspin.server.roles.base import Role

RoleFactory = Callable[["SendspinClient"], "Role"]
GroupRoleFactory = Callable[["SendspinGroup"], GroupRole]
SupportParser = Callable[[dict[str, Any]], object]

ROLE_FACTORIES: dict[str, RoleFactory] = {}
GROUP_ROLE_FACTORIES: dict[str, GroupRoleFactory] = {}
PAIRING_REQUIRED_ROLE_IDS: set[str] = set()


@dataclass(frozen=True, slots=True)
class RoleSupportSpec:
    """Parser for role-family support objects in client/hello."""

    parse_support: SupportParser


ROLE_SUPPORT_SPECS: dict[str, RoleSupportSpec] = {}


def register_role(role_id: str, factory: RoleFactory, *, requires_pairing: bool = False) -> None:
    """Register a role factory with optional pairing enforcement."""
    ROLE_FACTORIES[role_id] = factory
    if requires_pairing:
        PAIRING_REQUIRED_ROLE_IDS.add(role_id)
    else:
        PAIRING_REQUIRED_ROLE_IDS.discard(role_id)


def role_requires_pairing(role_id: str) -> bool:
    """Whether the role with this versioned ID needs pairing or explicit unpaired approval."""
    return role_id in PAIRING_REQUIRED_ROLE_IDS


def create_role(role_id: str, client: SendspinClient) -> Role | None:
    """Create a role instance for the given role ID, if registered."""
    factory = ROLE_FACTORIES.get(role_id)
    if factory is None:
        return None
    return factory(client)


def register_role_support_spec(role_family: str, spec: RoleSupportSpec) -> None:
    """Register support-object parsing metadata for a role family."""
    ROLE_SUPPORT_SPECS[role_family] = spec


def register_group_role(role_family: str, factory: GroupRoleFactory) -> None:
    """Register a group role factory for a role family."""
    GROUP_ROLE_FACTORIES[role_family] = factory


def create_group_roles(group: SendspinGroup) -> dict[str, GroupRole]:
    """Create group roles for a new group from registered factories."""
    return {family: factory(group) for family, factory in GROUP_ROLE_FACTORIES.items()}
