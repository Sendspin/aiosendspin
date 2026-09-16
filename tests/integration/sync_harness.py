"""Shared integration harness for group sync tests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from aiosendspin.models.player import (
    ClientHelloPlayerSupport,
    SupportedAudioFormat,
    pack_player_audio_header,
)
from aiosendspin.models.types import AudioCodec, PlayerCommand, Roles
from aiosendspin.server.channels import MAIN_CHANNEL, ChannelResolver
from aiosendspin.server.client import SendspinClient
from aiosendspin.server.clock import ManualClock
from aiosendspin.server.group import SendspinGroup
from aiosendspin.server.roles import AudioRequirements
from aiosendspin.server.roles.player.audio_transformers import FlacEncoder, PcmPassthrough


@dataclass(slots=True)
class DummyServer:
    """Minimal server surface used by integration harness clients."""

    loop: asyncio.AbstractEventLoop
    clock: ManualClock
    id: str = "srv"
    name: str = "server"

    def _signal_client_connected(self, client_id: str) -> None:
        pass

    def _signal_client_disconnected(self, client_id: str, goodbye_reason: object = None) -> None:
        pass


EventKind = Literal["json", "bin"]


@dataclass(slots=True)
class Event:
    """Captured outbound event from a test connection."""

    kind: EventKind
    payload: object


class CaptureConnection:
    """Capture JSON + binary messages in order."""

    def __init__(self) -> None:
        """Initialize an empty event capture buffer."""
        self.events: list[Event] = []
        self.buffer_tracker = None

    async def disconnect(self, *, retry_connection: bool = True) -> None:  # noqa: ARG002
        """No-op disconnect used by tests."""
        return

    def send_message(self, message: object) -> None:
        """Record outbound JSON messages."""
        self.events.append(Event(kind="json", payload=message))

    def send_role_message(self, role: str, message: object) -> None:  # noqa: ARG002
        """Record outbound role-scoped JSON messages."""
        self.events.append(Event(kind="json", payload=message))

    def send_binary(
        self,
        data: bytes,
        *,
        role: str,  # noqa: ARG002
        timestamp_us: int,
        message_type: int,  # noqa: ARG002
        buffer_end_time_us: int | None = None,
        buffer_byte_count: int | None = None,
        duration_us: int | None = None,
        player_audio_header: bool = False,
        epoch_exempt: bool = False,  # noqa: ARG002
    ) -> bool:
        """Record outbound binary and optionally update buffered-byte accounting."""
        if player_audio_header:
            data = pack_player_audio_header(timestamp_us, 0) + data
        self.events.append(Event(kind="bin", payload=data))
        if (
            self.buffer_tracker is not None
            and buffer_end_time_us is not None
            and buffer_byte_count is not None
        ):
            self.buffer_tracker.register(buffer_end_time_us, buffer_byte_count, duration_us or 0)
        return True


def _transformer_for(format_: SupportedAudioFormat) -> FlacEncoder | PcmPassthrough:
    if format_.codec == AudioCodec.FLAC:
        return FlacEncoder(
            sample_rate=format_.sample_rate,
            channels=format_.channels,
            bit_depth=format_.bit_depth,
            chunk_duration_us=25_000,
        )

    return PcmPassthrough(
        sample_rate=format_.sample_rate,
        bit_depth=format_.bit_depth,
        channels=format_.channels,
        chunk_duration_us=25_000,
    )


def channel_resolver_for(mapping: dict[str, UUID]) -> ChannelResolver:
    """Build a channel resolver that falls back to MAIN_CHANNEL."""

    def resolve(player_id: str) -> UUID:
        return mapping.get(player_id, MAIN_CHANNEL)

    return resolve


def make_player(
    server: DummyServer,
    client_id: str,
    *,
    supported_formats: list[SupportedAudioFormat],
    buffer_capacity: int,
    channel_id: UUID = MAIN_CHANNEL,
) -> tuple[SendspinClient, SendspinGroup, CaptureConnection]:
    """Create a connected player with deterministic audio requirements."""
    client = SendspinClient(server, client_id=client_id)
    group = SendspinGroup(server, client)

    conn = CaptureConnection()
    hello = type("Hello", (), {})()
    hello.client_id = client_id
    hello.name = client_id
    hello.player_support = ClientHelloPlayerSupport(
        supported_formats=supported_formats,
        buffer_capacity=buffer_capacity,
        supported_commands=[PlayerCommand.VOLUME, PlayerCommand.MUTE],
    )
    hello.artwork_support = None
    hello.visualizer_support = None

    client.attach_connection(
        conn,
        client_info=hello,
        negotiated_roles=[Roles.PLAYER.value],
        active_roles=[Roles.PLAYER.value],
    )
    client.mark_connected()

    role = client.role("player@v1")
    if role is not None:
        conn.buffer_tracker = role.get_buffer_tracker()

    if role is not None and supported_formats:
        preferred = supported_formats[0]
        role._audio_requirements = AudioRequirements(  # noqa: SLF001
            sample_rate=preferred.sample_rate,
            bit_depth=preferred.bit_depth,
            channels=preferred.channels,
            transformer=_transformer_for(preferred),
            channel_id=channel_id,
            frame_duration_us=25_000,
        )

    return client, group, conn
