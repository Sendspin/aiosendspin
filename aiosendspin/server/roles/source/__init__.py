"""Server-side source role: receive and decode audio captured by source clients."""

from __future__ import annotations

from aiosendspin.server.roles.registry import register_role

from .events import (
    SourceSignalChangedEvent,
    SourceStreamEndedEvent,
    SourceStreamStartedEvent,
)
from .stream import SourceStream
from .v1 import SourceV1Role

# Source capture requires a paired connection or explicit unpaired approval.
register_role(
    "source@v1",
    lambda client: SourceV1Role(client=client),
    requires_pairing=True,
)

__all__ = [
    "SourceSignalChangedEvent",
    "SourceStream",
    "SourceStreamEndedEvent",
    "SourceStreamStartedEvent",
    "SourceV1Role",
]
