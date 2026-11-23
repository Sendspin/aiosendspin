"""Shared event base types for server, group, client, and role event streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from aiosendspin.models.types import PlaybackStateType

if TYPE_CHECKING:
    from .group import SendspinGroup


class ClientEvent:
    """Base event type used by SendspinClient.add_event_listener()."""


class ClientRoleEvent(ClientEvent):
    """Base type for role-emitted client events."""


@dataclass
class ClientGroupChangedEvent(ClientEvent):
    """The client was moved to a different group."""

    new_group: SendspinGroup
    """The new group the client is now part of."""


@dataclass
class ClientAvailabilityChangedEvent(ClientEvent):
    """The availability reported in the client's `client/state` changed."""

    available: bool
    """The new availability."""


@dataclass
class ClientLeftGroupEvent(ClientEvent):
    """The client left a shared group through `client/leave` or by becoming unavailable."""

    previous_group_id: str
    """ID of the group the client left, which may no longer exist."""


class GroupEvent:
    """Base event type used by SendspinGroup.add_event_listener()."""


class GroupRoleEvent(GroupEvent):
    """Base type for GroupRole-emitted group events."""


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
