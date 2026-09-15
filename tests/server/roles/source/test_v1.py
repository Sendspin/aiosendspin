"""Behavioural tests for SourceV1Role: decode path, lifecycle, and events."""

from __future__ import annotations

from typing import Any

import pytest

from aiosendspin.audio.codecs import PcmPassthrough
from aiosendspin.audio.format import AudioFormat
from aiosendspin.models.core import ClientStatePayload
from aiosendspin.models.source import (
    ClientHelloSourceFeatures,
    ClientHelloSourceSupport,
    ClientStreamStartPayload,
    ClientStreamStartSource,
    SourceStatePayload,
)
from aiosendspin.models.types import AudioCodec, BinaryMessageType, SignalState
from aiosendspin.server.roles.source import (
    SourceSignalChangedEvent,
    SourceStreamEndedEvent,
    SourceStreamStartedEvent,
)
from aiosendspin.server.roles.source.stream import SourceStream
from aiosendspin.server.roles.source.v1 import SourceV1Role
from tests.conftest import sine_pcm_16bit


class _FakeInfo:
    def __init__(self, *, line_sense: bool) -> None:
        self.source_support = ClientHelloSourceSupport(
            features=ClientHelloSourceFeatures(line_sense=line_sense)
        )


class _FakeClient:
    def __init__(self, *, line_sense: bool = True) -> None:
        self.events: list[Any] = []
        self._info = _FakeInfo(line_sense=line_sense)
        self.connection: Any = object()
        self.sent: list[Any] = []
        self.available = True
        self.noncompliance: list[str] = []

    @property
    def info(self) -> _FakeInfo:
        return self._info

    def flag_noncompliance(self, reason: str) -> None:
        self.noncompliance.append(reason)

    def _signal_event(self, event: Any) -> None:
        self.events.append(event)

    def send_role_message(self, _family: str, message: Any) -> None:
        self.sent.append(message)


def _pcm_start_payload() -> ClientStreamStartPayload:
    return ClientStreamStartPayload(
        source=ClientStreamStartSource(
            codec=AudioCodec.PCM, channels=2, sample_rate=48000, bit_depth=16, codec_header=None
        )
    )


def _make_role() -> tuple[SourceV1Role, _FakeClient]:
    """Build a connected role the server has already asked to stream."""
    client = _FakeClient()
    role = SourceV1Role(client=client)  # type: ignore[arg-type]
    role.on_connect()
    role.request_start()
    return role, client


def test_client_stream_start_emits_event_with_native_format() -> None:
    """on_client_stream_start announces the decoded handle and its PCM format."""
    role, client = _make_role()
    role.on_client_stream_start(_pcm_start_payload())
    started = [e for e in client.events if isinstance(e, SourceStreamStartedEvent)]
    assert len(started) == 1
    assert started[0].audio_format.sample_rate == 48000
    assert started[0].audio_format.channels == 2


async def test_pcm_loopback_is_bit_exact_through_role() -> None:
    """PCM streamed up as type-12 chunks decodes bit-exact out of the handle."""
    role, client = _make_role()
    role.on_client_state(ClientStatePayload(available=True))
    role.on_client_stream_start(_pcm_start_payload())
    handle = next(e for e in client.events if isinstance(e, SourceStreamStartedEvent)).handle

    pcm = sine_pcm_16bit(48000)
    encoder = PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2)
    ts = 1_000_000
    for frame, dur in encoder.process(pcm, ts, 0):
        role.on_binary_chunk(BinaryMessageType.SOURCE_AUDIO_CHUNK.value, ts, frame)
        ts += dur
    role.on_client_stream_end()

    received = bytearray()
    async for chunk, _chunk_ts in handle:
        received += chunk
    assert bytes(received) == pcm


def test_role_declares_only_the_source_binary_type() -> None:
    """The source role consumes only the source audio binary type."""
    role, _client = _make_role()
    assert role.handles_inbound_binary(BinaryMessageType.SOURCE_AUDIO_CHUNK.value)
    assert not role.handles_inbound_binary(BinaryMessageType.AUDIO_CHUNK.value)


def test_binary_chunk_dropped_when_inactive() -> None:
    """A chunk with no active stream is dropped without starting one or erroring."""
    role, client = _make_role()
    role.on_binary_chunk(BinaryMessageType.SOURCE_AUDIO_CHUNK.value, 0, b"\x00\x00\x00\x00")
    assert not any(isinstance(e, SourceStreamStartedEvent) for e in client.events)


async def test_binary_chunk_dropped_before_initial_state() -> None:
    """Audio is dropped until the source sends its initial state."""
    role, client = _make_role()
    role.on_client_stream_start(_pcm_start_payload())
    handle = next(e for e in client.events if isinstance(e, SourceStreamStartedEvent)).handle

    role.on_binary_chunk(
        BinaryMessageType.SOURCE_AUDIO_CHUNK.value,
        1_000_000,
        sine_pcm_16bit(480),
    )
    role.on_client_stream_end()

    assert [chunk async for chunk, _ in handle] == []


@pytest.mark.parametrize(
    "bad_format",
    [
        {"bit_depth": 17},
        {"channels": 0},
        {"sample_rate": 0},
    ],
)
def test_impossible_declared_format_opens_no_stream(bad_format: dict[str, int]) -> None:
    """An unusable declared format is rejected at start, not inside the consumer."""
    role, client = _make_role()
    fields = {"codec": AudioCodec.PCM, "channels": 2, "sample_rate": 48000, "bit_depth": 16}
    fields.update(bad_format)
    role.on_client_stream_start(
        ClientStreamStartPayload(source=ClientStreamStartSource(**fields))  # type: ignore[arg-type]
    )
    assert not role.stream_active
    assert [e for e in client.events if isinstance(e, SourceStreamStartedEvent)] == []


def test_opus_start_ignores_declared_bit_depth() -> None:
    """An opus stream opens at 16-bit no matter what bit_depth the client declared."""
    role, client = _make_role()
    role.on_client_stream_start(
        ClientStreamStartPayload(
            source=ClientStreamStartSource(
                codec=AudioCodec.OPUS, channels=2, sample_rate=48000, bit_depth=17
            )
        )
    )
    handle = next(e for e in client.events if isinstance(e, SourceStreamStartedEvent)).handle
    assert handle.audio_format.bit_depth == 16


def test_stream_buffer_drops_oldest_beyond_byte_budget() -> None:
    """A stalled consumer's buffer is bounded by bytes, not just chunk count."""
    stream = SourceStream(AudioFormat(sample_rate=100, bit_depth=16, channels=1))
    for i in range(3):
        stream._push(bytes([i]) * 1500, i)  # noqa: SLF001

    assert [ts for _, ts in stream._queue] == [2]  # noqa: SLF001
    assert stream._buffered_bytes == 1500  # noqa: SLF001


def test_flac_start_requires_streaminfo_header() -> None:
    """A FLAC stream cannot open without its required STREAMINFO header."""
    role, client = _make_role()
    role.on_client_stream_start(
        ClientStreamStartPayload(
            source=ClientStreamStartSource(
                codec=AudioCodec.FLAC,
                channels=2,
                sample_rate=48000,
                bit_depth=16,
            )
        )
    )

    assert not role.stream_active
    assert client.noncompliance == ["client-stream/start FLAC codec_header must contain STREAMINFO"]


async def test_stream_replacement_announces_the_end_of_the_old_handle() -> None:
    """Replacing a stream tells listeners the previous handle is finished."""
    role, client = _make_role()
    role.on_client_stream_start(_pcm_start_payload())
    role.on_client_stream_start(_pcm_start_payload())
    assert len([e for e in client.events if isinstance(e, SourceStreamEndedEvent)]) == 1


def test_teardown_announces_stream_end() -> None:
    """Disconnect ends the stream visibly rather than dropping the handle silently."""
    role, client = _make_role()
    role.on_client_stream_start(_pcm_start_payload())
    role.on_disconnect()
    assert len([e for e in client.events if isinstance(e, SourceStreamEndedEvent)]) == 1


def test_source_requires_initial_state() -> None:
    """The source role participates in the initial state gate."""
    role, _client = _make_role()
    assert role.requires_initial_state() is True


def test_unsolicited_stream_start_opens_no_stream() -> None:
    """A start the server never asked for is ignored rather than opening a stream."""
    client = _FakeClient()
    role = SourceV1Role(client=client)  # type: ignore[arg-type]
    role.on_connect()
    role.on_client_stream_start(_pcm_start_payload())
    assert not role.stream_active
    assert [e for e in client.events if isinstance(e, SourceStreamStartedEvent)] == []
    assert client.noncompliance  # flagged, so a strict server can reject the client


def test_stream_start_after_stop_opens_no_stream() -> None:
    """Once stopped, a late start from the client does not reopen the stream."""
    role, client = _make_role()
    role.request_stop()
    role.on_client_stream_start(_pcm_start_payload())
    assert not role.stream_active
    assert [e for e in client.events if isinstance(e, SourceStreamStartedEvent)] == []


def test_start_request_does_not_survive_disconnect() -> None:
    """Streaming state is per-connection, so a reconnect needs a fresh start."""
    role, client = _make_role()
    role.on_disconnect()
    role.on_connect()
    role.on_client_stream_start(_pcm_start_payload())
    assert not role.stream_active
    assert [e for e in client.events if isinstance(e, SourceStreamStartedEvent)] == []


async def test_binary_chunk_dropped_when_client_not_available() -> None:
    """An open stream still drops chunks while the client is not available."""
    role, client = _make_role()
    role.on_client_state(ClientStatePayload(available=True))
    role.on_client_stream_start(_pcm_start_payload())
    handle = next(e for e in client.events if isinstance(e, SourceStreamStartedEvent)).handle
    client.available = False

    frame = sine_pcm_16bit(480)
    role.on_binary_chunk(BinaryMessageType.SOURCE_AUDIO_CHUNK.value, 1_000_000, frame)
    role.on_client_stream_end()

    drained = [chunk async for chunk, _ in handle]
    assert drained == []


def test_becoming_unavailable_implicitly_stops_stream() -> None:
    """Becoming unavailable closes the stream and clears start permission."""
    role, client = _make_role()
    role.on_client_stream_start(_pcm_start_payload())

    role.on_availability_changed(old_available=True, new_available=False)
    role.on_client_stream_start(_pcm_start_payload())

    assert not role.stream_active
    assert len([e for e in client.events if isinstance(e, SourceStreamEndedEvent)]) == 1
    assert client.noncompliance == [
        "client-stream/start sent without a preceding source start command"
    ]


async def test_second_start_restarts_stream() -> None:
    """A second client-stream/start ends the prior handle and announces a new one."""
    role, client = _make_role()
    role.on_client_stream_start(_pcm_start_payload())
    first = next(e for e in client.events if isinstance(e, SourceStreamStartedEvent)).handle
    role.on_client_stream_start(_pcm_start_payload())
    # The first handle must terminate so its consumer is released.
    drained = [chunk async for chunk, _ in first]
    assert drained == []
    starts = [e for e in client.events if isinstance(e, SourceStreamStartedEvent)]
    assert len(starts) == 2
    assert starts[1].handle is not first


async def test_client_stream_end_emits_event_and_closes_handle() -> None:
    """client-stream/end surfaces SourceStreamEndedEvent AND terminates the handle."""
    role, client = _make_role()
    role.on_client_stream_start(_pcm_start_payload())
    handle = next(e for e in client.events if isinstance(e, SourceStreamStartedEvent)).handle
    role.on_client_stream_end()
    assert any(isinstance(e, SourceStreamEndedEvent) for e in client.events)
    # The handle must be exhausted so a consumer's `async for` exits rather than hangs.
    drained = [chunk async for chunk, _ in handle]
    assert drained == []


@pytest.mark.parametrize("teardown", ["on_deactivate", "on_disconnect"])
async def test_teardown_ends_active_stream(teardown: str) -> None:
    """Role teardown (deactivation or disconnect) releases a waiting stream consumer."""
    role, client = _make_role()
    role.on_client_stream_start(_pcm_start_payload())
    handle = next(e for e in client.events if isinstance(e, SourceStreamStartedEvent)).handle
    getattr(role, teardown)()
    drained = [chunk async for chunk, _ in handle]
    assert drained == []
    # State is reset, so a later chunk is a safe no-op rather than pushed to the dead handle.
    role.on_binary_chunk(BinaryMessageType.SOURCE_AUDIO_CHUNK.value, 0, b"\x00\x00\x00\x00")


def test_client_state_surfaces_signal_only_when_advertised() -> None:
    """Signal is surfaced only when the source advertised the line_sense feature."""
    advertised = _FakeClient(line_sense=True)
    role = SourceV1Role(client=advertised)  # type: ignore[arg-type]
    role.on_connect()
    role.on_client_state(ClientStatePayload(source=SourceStatePayload(signal=SignalState.PRESENT)))
    event = next(e for e in advertised.events if isinstance(e, SourceSignalChangedEvent))
    assert event.signal is SignalState.PRESENT

    unadvertised = _FakeClient(line_sense=False)
    role = SourceV1Role(client=unadvertised)  # type: ignore[arg-type]
    role.on_connect()
    role.on_client_state(ClientStatePayload(source=SourceStatePayload(signal=SignalState.PRESENT)))
    assert not any(isinstance(e, SourceSignalChangedEvent) for e in unadvertised.events)


def test_request_start_and_stop_send_server_command() -> None:
    """request_start/request_stop emit server/command with the right verb."""
    client = _FakeClient()
    role = SourceV1Role(client=client)  # type: ignore[arg-type]
    role.on_connect()
    role.request_start()
    role.request_stop()
    commands = [m.payload.source.command for m in client.sent]
    assert commands == ["start", "stop"]


def test_client_state_signal_event_only_fires_on_change() -> None:
    """Clients repeat the signal in every state, so only transitions are surfaced."""
    client = _FakeClient(line_sense=True)
    role = SourceV1Role(client=client)  # type: ignore[arg-type]
    role.on_connect()

    for _ in range(3):
        role.on_client_state(
            ClientStatePayload(source=SourceStatePayload(signal=SignalState.PRESENT))
        )
    role.on_client_state(ClientStatePayload(source=SourceStatePayload(signal=SignalState.ABSENT)))
    role.on_client_state(ClientStatePayload(source=SourceStatePayload(signal=SignalState.ABSENT)))

    signals = [e.signal for e in client.events if isinstance(e, SourceSignalChangedEvent)]
    assert signals == [SignalState.PRESENT, SignalState.ABSENT]


def test_reconnect_resurfaces_the_current_signal() -> None:
    """A disconnect forgets the signal, so the client's next report is surfaced again."""
    client = _FakeClient(line_sense=True)
    role = SourceV1Role(client=client)  # type: ignore[arg-type]
    role.on_connect()
    role.on_client_state(ClientStatePayload(source=SourceStatePayload(signal=SignalState.PRESENT)))
    role.on_disconnect()
    role.on_client_state(ClientStatePayload(source=SourceStatePayload(signal=SignalState.PRESENT)))

    signals = [e.signal for e in client.events if isinstance(e, SourceSignalChangedEvent)]
    assert signals == [SignalState.PRESENT, SignalState.PRESENT]
