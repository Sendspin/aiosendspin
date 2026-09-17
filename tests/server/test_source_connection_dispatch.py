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
        self.available = True

    def flag_noncompliance(self, reason: str) -> None:
        self.noncompliance.append(reason)
        if self._strict:
            raise ClientComplianceError(reason)

    @property
    def active_roles(self) -> list[Any]:
        return self._roles


def _bare_connection(
    roles: list[Any], *, strict: bool = False, starts: int = 0, input_open: bool = False
) -> SendspinConnection:
    conn = SendspinConnection.__new__(SendspinConnection)
    conn._client = _FakeClient(roles, strict=strict)  # noqa: SLF001
    conn._logger = logging.getLogger("test.source.dispatch")  # noqa: SLF001
    conn._source_starts_pending = starts  # noqa: SLF001
    conn._source_input_open = input_open  # noqa: SLF001
    return conn


def _chunk(timestamp_us: int = 1) -> bytes:
    return pack_binary_header_raw(BinaryMessageType.SOURCE_AUDIO_CHUNK.value, timestamp_us) + b"x"


_PCM_START = ClientStreamStartMessage(
    payload=ClientStreamStartPayload(
        source=ClientStreamStartSource(
            codec=AudioCodec.PCM, channels=2, sample_rate=48000, bit_depth=16
        )
    )
)


def test_inbound_binary_routed_to_source_role() -> None:
    """A type-12 binary frame is parsed and delivered with header ts + payload."""
    role = _RecordingRole()
    conn = _bare_connection([role], input_open=True)
    frame = pack_binary_header_raw(BinaryMessageType.SOURCE_AUDIO_CHUNK.value, 42_000) + b"audio"
    conn._route_inbound_binary(frame)  # noqa: SLF001
    assert role.binary == [(BinaryMessageType.SOURCE_AUDIO_CHUNK.value, 42_000, b"audio")]


def test_inbound_binary_stops_at_first_consuming_role() -> None:
    """Routing stops at the first role declaring the chunk's type."""
    first = _RecordingRole(consume=True)
    second = _RecordingRole(consume=True)
    conn = _bare_connection([first, second], input_open=True)
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
            pack_binary_header_raw(BinaryMessageType.AUDIO_CHUNK.value, 1) + b"x"
        )
    assert any("unhandled binary" in r.message.lower() for r in caplog.records)


@pytest.mark.parametrize("message_type", [24, 100, 191, 200])
def test_unimplemented_binary_type_is_ignored(message_type: int) -> None:
    """A binary type the server does not implement is ignored, even for a strict server."""
    role = _RecordingRole()
    conn = _bare_connection([role], strict=True, input_open=True)
    conn._route_inbound_binary(pack_binary_header_raw(message_type, 1) + b"x")  # noqa: SLF001
    assert role.binary == []
    assert conn._client.noncompliance == []  # type: ignore[union-attr]  # noqa: SLF001


def test_short_binary_payload_is_dropped_safely(caplog: Any) -> None:
    """A payload shorter than the 9-byte header is dropped with a warning, no exception."""
    conn = _bare_connection([_RecordingRole()])
    with caplog.at_level(logging.WARNING):
        conn._route_inbound_binary(b"\x0c\x00")  # noqa: SLF001
    assert any("shorter than header" in r.message.lower() for r in caplog.records)


async def test_client_stream_start_and_end_dispatched_to_roles() -> None:
    """client-stream/start and client-stream/end reach role hooks via _handle_message."""
    role = _RecordingRole()
    conn = _bare_connection([role], starts=1)
    await conn._handle_message(_PCM_START, timestamp_us=0)  # noqa: SLF001
    await conn._handle_message(ClientStreamEndMessage(), timestamp_us=0)  # noqa: SLF001
    assert len(role.starts) == 1
    assert role.ends == 1


async def test_superseded_stream_message_names_are_dispatched_and_flagged() -> None:
    """A source on the pre-rename wire is still served, and the deviation recorded."""
    role = _RecordingRole()
    conn = _bare_connection([role], starts=1)
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


async def test_unsolicited_stream_start_is_flagged_and_not_dispatched() -> None:
    """A client-stream/start with no start outstanding opens nothing."""
    role = _RecordingRole()
    conn = _bare_connection([role])

    await conn._handle_message(_PCM_START, timestamp_us=0)  # noqa: SLF001
    conn._route_inbound_binary(_chunk())  # noqa: SLF001

    assert role.starts == []
    assert role.binary == []
    assert conn._client.noncompliance == [  # noqa: SLF001
        "client-stream/start sent without a preceding source start command",
        "sent source audio without an open input stream",
    ]


async def test_each_start_authorizes_one_opening() -> None:
    """A second opening after client-stream/end needs a new start and does not reopen."""
    role = _RecordingRole()
    conn = _bare_connection([role], starts=1)

    await conn._handle_message(_PCM_START, timestamp_us=0)  # noqa: SLF001
    await conn._handle_message(ClientStreamEndMessage(), timestamp_us=0)  # noqa: SLF001
    await conn._handle_message(_PCM_START, timestamp_us=0)  # noqa: SLF001
    conn._route_inbound_binary(_chunk())  # noqa: SLF001

    assert len(role.starts) == 1
    assert role.binary == []
    assert conn._client.noncompliance == [  # noqa: SLF001
        "client-stream/start sent without a preceding source start command",
        "sent source audio without an open input stream",
    ]


async def test_stream_start_on_an_open_stream_replaces_it_without_a_start() -> None:
    """A format replacement needs no start and leaves an outstanding start unused."""
    role = _RecordingRole()
    conn = _bare_connection([role], starts=2)

    for _ in range(2):
        await conn._handle_message(_PCM_START, timestamp_us=0)  # noqa: SLF001
    await conn._handle_message(ClientStreamEndMessage(), timestamp_us=0)  # noqa: SLF001
    await conn._handle_message(_PCM_START, timestamp_us=0)  # noqa: SLF001

    assert len(role.starts) == 3
    assert conn._client.noncompliance == []  # noqa: SLF001


async def test_crossing_starts_each_open_a_stream() -> None:
    """start, stop, start may cross the client's end, so two openings are both valid."""
    role = _RecordingRole()
    conn = _bare_connection([role])
    conn.record_source_start()
    conn.record_source_start()

    await conn._handle_message(_PCM_START, timestamp_us=0)  # noqa: SLF001
    await conn._handle_message(ClientStreamEndMessage(), timestamp_us=0)  # noqa: SLF001
    await conn._handle_message(_PCM_START, timestamp_us=0)  # noqa: SLF001

    assert len(role.starts) == 2
    assert conn._client.noncompliance == []  # noqa: SLF001


def test_source_audio_without_an_open_stream_closes_a_strict_server() -> None:
    """Source audio outside an input stream is a protocol error the server rejects."""
    role = _RecordingRole()
    conn = _bare_connection([role], strict=True)

    with pytest.raises(ClientComplianceError):
        conn._route_inbound_binary(_chunk())  # noqa: SLF001
    assert role.binary == []


def test_source_audio_while_unavailable_is_flagged() -> None:
    """Source audio after the client reported available: false is flagged, not decoded."""
    role = _RecordingRole()
    conn = _bare_connection([role], input_open=True)
    conn._client.available = False  # type: ignore[union-attr]  # noqa: SLF001

    conn._route_inbound_binary(_chunk())  # noqa: SLF001

    assert role.binary == []
    assert conn._client.noncompliance == [  # noqa: SLF001
        "sent source audio while reporting available: false"
    ]


def test_source_audio_without_a_source_role_is_dropped_quietly(caplog: Any) -> None:
    """In-flight audio on a stream still open after the role's removal is discarded."""
    conn = _bare_connection([], input_open=True)

    with caplog.at_level(logging.WARNING):
        conn._route_inbound_binary(_chunk())  # noqa: SLF001

    assert caplog.records == []
    assert conn._client.noncompliance == []  # noqa: SLF001
