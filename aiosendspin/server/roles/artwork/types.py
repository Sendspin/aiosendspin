"""Shared artwork role protocols."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from aiosendspin.models.artwork import ArtworkChannel


@runtime_checkable
class ArtworkRoleProtocol(Protocol):
    """Protocol for artwork role implementations."""

    @property
    def role_id(self) -> str:
        """Return the versioned role identifier."""
        ...

    def get_channel_configs(self) -> dict[int, ArtworkChannel]:
        """Return channel configuration mapping."""
        ...

    def send_artwork(self, channel: int, image_data: bytes, timestamp_us: int) -> None:
        """Send artwork bytes for a channel."""
        ...

    def send_artwork_cleared(self, channel: int, timestamp_us: int) -> None:
        """Clear artwork for a channel."""
        ...

    def cancel_scheduled_artwork(self, channel: int) -> bool:
        """Discard a channel's scheduled image, returning False if the current must be resent."""
        ...

    # DEPRECATED(spec-pr-188): remove in aiosendspin <version>
    def uses_single_message_framing(self) -> bool:
        """Whether the client gets each image as one message, without transfers."""
        ...
