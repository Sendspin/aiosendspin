"""The connection routes inbound source binary + client_stream messages to roles."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from aiosendspin.models import pack_binary_header_raw
from aiosendspin.models.source import (
    ClientStreamEndMessage,
    ClientStreamStartMessage,
    ClientStreamStartPayload,
    ClientStreamStartSource,
)
from aiosendspin.models.types import AudioCodec, BinaryMessageType, ClientMessage
from aiosendspin.server.compliance import ClientComplianceError
from aiosendspin.server.connection import SendspinConnection


class _RecordingRole:
    role_family = "source"

    def __init__(self, *, consume: bool = True) -> None:
        self.consume = consume
        self.binary: list[tuple[int, int, bytes]] = []
        self.starts: list[ClientStreamStartPayload] = []
        self.ends = 0

    def handles_inbound_binary(self, message_type: int) -> bool:
        return self.consume and message_type == BinaryMessageType.SOURCE_AUDIO_CHUNK.value

    def on_binary_chunk(self, message_type: int, timestamp_us: int, data: bytes) -> None:
        self.binary.append((message_type, timestamp_us, data))

    def on_client_stream_start(self, payload: ClientStreamStartPayload) -> None:
        self.starts.append(payload)

    def on_client_stream_end(self) -> None:
        self.ends += 1


class _FakeClient:
    def __init__(self, roles: list[Any], *, strict: bool = False) -> None:
        self._roles = roles
        self._strict = strict
        self.noncompliance: list[str] = []

    def flag_noncompliance(self, reason: str) -> None:
        self.noncompliance.append(reason)
        if self._strict:
            raise ClientComplianceError(reason)

    @property
    def active_roles(self) -> list[Any]:
        return self._roles


def _bare_connection(roles: list[Any], *, strict: bool = False) -> SendspinConnection:
    conn = SendspinConnection.__new__(SendspinConnection)
    conn._client = _FakeClient(roles, strict=strict)  # noqa: SLF001
    conn._logger = logging.getLogger("test.source.dispatch")  # noqa: SLF001
    return conn


def test_inbound_binary_routed_to_source_role() -> None:
    """A type-12 binary frame is parsed and delivered with header ts + payload."""
    role = _RecordingRole()
    conn = _bare_connection([role])
    frame = pack_binary_header_raw(BinaryMessageType.SOURCE_AUDIO_CHUNK.value, 42_000) + b"audio"
    conn._route_inbound_binary(frame)  # noqa: SLF001
    assert role.binary == [(BinaryMessageType.SOURCE_AUDIO_CHUNK.value, 42_000, b"audio")]


def test_inbound_binary_stops_at_first_consuming_role() -> None:
    """Routing stops at the first role declaring the chunk's type."""
    first = _RecordingRole(consume=True)
    second = _RecordingRole(consume=True)
    conn = _bare_connection([first, second])
    conn._route_inbound_binary(  # noqa: SLF001
        pack_binary_header_raw(BinaryMessageType.SOURCE_AUDIO_CHUNK.value, 1) + b"x"
    )
    assert len(first.binary) == 1
    assert second.binary == []


def test_unhandled_binary_warns(caplog: Any) -> None:
    """A binary type no role claims is logged as unhandled rather than crashing."""
    role = _RecordingRole(consume=False)
    conn = _bare_connection([role])
    with caplog.at_level(logging.WARNING):
        conn._route_inbound_binary(  # noqa: SLF001
            pack_binary_header_raw(BinaryMessageType.SOURCE_AUDIO_CHUNK.value, 1) + b"x"
        )
    assert any("unhandled binary" in r.message.lower() for r in caplog.records)


def test_short_binary_payload_is_dropped_safely(caplog: Any) -> None:
    """A payload shorter than the 9-byte header is dropped with a warning, no exception."""
    conn = _bare_connection([_RecordingRole()])
    with caplog.at_level(logging.WARNING):
        conn._route_inbound_binary(b"\x0c\x00")  # noqa: SLF001
    assert any("shorter than header" in r.message.lower() for r in caplog.records)


async def test_client_stream_start_and_end_dispatched_to_roles() -> None:
    """client-stream/start and client-stream/end reach role hooks via _handle_message."""
    role = _RecordingRole()
    conn = _bare_connection([role])
    start = ClientStreamStartMessage(
        payload=ClientStreamStartPayload(
            source=ClientStreamStartSource(
                codec=AudioCodec.PCM, channels=2, sample_rate=48000, bit_depth=16
            )
        )
    )
    await conn._handle_message(start, timestamp_us=0)  # noqa: SLF001
    await conn._handle_message(ClientStreamEndMessage(), timestamp_us=0)  # noqa: SLF001
    assert len(role.starts) == 1
    assert role.ends == 1


async def test_superseded_stream_message_names_are_dispatched_and_flagged() -> None:
    """A source on the pre-rename wire is still served, and the deviation recorded."""
    role = _RecordingRole()
    conn = _bare_connection([role])
    start = ClientMessage.from_json(
        '{"type":"client_stream/start","payload":{"source":'
        '{"codec":"pcm","sample_rate":48000,"bit_depth":16,"channels":2}}}'
    )
    end = ClientMessage.from_json('{"type":"client_stream/end"}')

    await conn._handle_message(start, timestamp_us=0)  # noqa: SLF001
    await conn._handle_message(end, timestamp_us=0)  # noqa: SLF001

    assert len(role.starts) == 1
    assert role.ends == 1
    assert conn._client.noncompliance == [  # noqa: SLF001
        "client sent client_stream/start, superseded by client-stream/start",
        "client sent client_stream/end, superseded by client-stream/end",
    ]


async def test_current_stream_message_names_are_not_flagged() -> None:
    """The current spelling raises nothing with the server."""
    role = _RecordingRole()
    conn = _bare_connection([role])

    await conn._handle_message(ClientStreamEndMessage(), timestamp_us=0)  # noqa: SLF001

    assert role.ends == 1
    assert conn._client.noncompliance == []  # noqa: SLF001


async def test_superseded_stream_message_name_is_rejected_by_a_strict_server() -> None:
    """The flag is not cosmetic: a strict server drops a source on the old spelling."""
    conn = _bare_connection([_RecordingRole()], strict=True)

    with pytest.raises(ClientComplianceError):
        await conn._handle_message(  # noqa: SLF001
            ClientMessage.from_json('{"type":"client_stream/end"}'), timestamp_us=0
        )
