"""In-flight source responses across stop, unavailability and role removal."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from aiosendspin.models import pack_binary_header_raw
from aiosendspin.models.core import ClientStatePayload
from aiosendspin.models.source import (
    ClientStreamEndMessage,
    ClientStreamStartMessage,
    ClientStreamStartPayload,
    ClientStreamStartSource,
    SourceStatePayload,
)
from aiosendspin.models.types import AudioCodec, BinaryMessageType, Roles
from aiosendspin.noise.trust_store import PskCategory
from aiosendspin.server.compliance import ClientComplianceError
from aiosendspin.server.roles.source import SourceStreamEndedEvent, SourceStreamStartedEvent
from aiosendspin.server.roles.source.v1 import SourceV1Role
from tests.server.test_role_activation import _PLAYER_STATE, _client, _connect, _hello

if TYPE_CHECKING:
    from aiosendspin.server.connection import SendspinConnection

_ROLES = [Roles.PLAYER.value, "source@v1"]
_PCM_START = ClientStreamStartMessage(
    payload=ClientStreamStartPayload(
        source=ClientStreamStartSource(
            codec=AudioCodec.PCM, channels=2, sample_rate=48000, bit_depth=16
        )
    )
)
_CHUNK = pack_binary_header_raw(BinaryMessageType.SOURCE_AUDIO_CHUNK.value, 1) + bytes(4)


def _state(*, available: bool = True) -> ClientStatePayload:
    return ClientStatePayload(
        available=available, player=_PLAYER_STATE, source=SourceStatePayload()
    )


class _Source:
    """A connected player+source client and the source events it emits."""

    def __init__(self, conn: SendspinConnection) -> None:
        self.conn = conn
        self.client = _client(conn)
        self.events: list[Any] = []
        self.client.add_event_listener(lambda _client, event: self.events.append(event))

    @property
    def role(self) -> SourceV1Role:
        role = self.client.role("source@v1")
        assert isinstance(role, SourceV1Role)
        return role

    def stream_events(self) -> list[type]:
        return [
            type(e)
            for e in self.events
            if isinstance(e, SourceStreamStartedEvent | SourceStreamEndedEvent)
        ]

    def strict(self) -> None:
        self.conn._server.allow_noncompliant_clients = False  # type: ignore[misc]  # noqa: SLF001

    async def stream_start(self) -> None:
        await self.conn._handle_message(_PCM_START, timestamp_us=0)  # noqa: SLF001

    async def stream_end(self) -> None:
        await self.conn._handle_message(ClientStreamEndMessage(), timestamp_us=0)  # noqa: SLF001

    def chunk(self) -> None:
        self.conn._route_inbound_binary(_CHUNK)  # noqa: SLF001

    async def state(self, *, available: bool = True) -> None:
        await self.conn._handle_client_state(_state(available=available))  # noqa: SLF001

    async def activate(self, roles: list[str]) -> None:
        self.conn._send_activation(roles)  # noqa: SLF001
        if "source@v1" in roles:
            await self.state()


async def _source() -> _Source:
    conn, _fake = await _connect(_hello(_ROLES), send_state=False, category=PskCategory.LONG_TERM)
    source = _Source(conn)
    await source.state()
    return source


@pytest.mark.asyncio
async def test_start_crossing_a_stop_opens_a_quiet_discard_stream() -> None:
    """The response to a start that crossed a stop is tolerated until its end."""
    source = await _source()
    source.role.request_start()
    source.role.request_stop()

    with patch.object(source.conn, "_flag_noncompliance") as flag:
        await source.stream_start()
        source.chunk()
        await source.stream_end()
    flag.assert_not_called()
    assert source.stream_events() == []

    source.strict()
    with pytest.raises(ClientComplianceError):
        source.chunk()


@pytest.mark.asyncio
async def test_start_crossing_an_availability_round_trip_opens_the_stream() -> None:
    """A start sent before available: false and true may still be answered afterwards."""
    source = await _source()
    source.role.request_start()
    await source.state(available=False)
    await source.state(available=True)
    source.role.request_start()

    with patch.object(source.conn, "_flag_noncompliance") as flag:
        await source.stream_start()
        source.chunk()
    flag.assert_not_called()
    assert source.stream_events() == [SourceStreamStartedEvent]


@pytest.mark.asyncio
async def test_start_crossing_role_removal_is_tolerated_until_its_end() -> None:
    """After removing the role, its outstanding start's stream is dropped until it ends."""
    source = await _source()
    source.role.request_start()
    await source.activate([Roles.PLAYER.value])
    source.strict()

    await source.stream_start()
    source.chunk()
    await source.stream_end()
    assert source.stream_events() == []

    with pytest.raises(ClientComplianceError):
        source.chunk()


@pytest.mark.asyncio
async def test_start_without_authorization_after_role_removal_is_rejected() -> None:
    """With no start outstanding, an opening after role removal is still unsolicited."""
    source = await _source()
    await source.activate([Roles.PLAYER.value])
    source.strict()

    with pytest.raises(ClientComplianceError):
        await source.stream_start()


@pytest.mark.asyncio
async def test_reactivated_role_ignores_the_previous_roles_stream() -> None:
    """A stream left open by the removed role neither reaches nor ends the new role's stream."""
    source = await _source()
    source.role.request_start()
    await source.stream_start()
    await source.activate([Roles.PLAYER.value])
    assert source.stream_events() == [SourceStreamStartedEvent, SourceStreamEndedEvent]

    with patch.object(source.conn, "_flag_noncompliance") as flag:
        source.chunk()
        await source.activate(_ROLES)
        source.chunk()
        await source.stream_end()

        source.role.request_start()
        await source.stream_start()
        source.chunk()
    flag.assert_not_called()
    assert source.stream_events() == [
        SourceStreamStartedEvent,
        SourceStreamEndedEvent,
        SourceStreamStartedEvent,
    ]
    assert source.role.stream_active


@pytest.mark.asyncio
async def test_reactivated_role_discards_the_previous_roles_outstanding_start() -> None:
    """The removed role's start opens only a discard stream in the reactivated role."""
    source = await _source()
    source.role.request_start()
    await source.activate([Roles.PLAYER.value])
    await source.activate(_ROLES)

    with patch.object(source.conn, "_flag_noncompliance") as flag:
        await source.stream_start()
        source.chunk()
        await source.stream_end()
    flag.assert_not_called()
    assert source.stream_events() == []
    assert not source.role.stream_active


@pytest.mark.asyncio
async def test_undecodable_authorized_stream_drops_its_audio_quietly() -> None:
    """A stream whose decoder cannot be built stays open on the wire, so its audio is no error."""
    source = await _source()
    source.role.request_start()
    source.strict()

    with patch(
        "aiosendspin.server.roles.source.v1.create_decoder", side_effect=ValueError("no decoder")
    ):
        await source.stream_start()
    source.chunk()
    await source.stream_end()

    assert source.stream_events() == []
