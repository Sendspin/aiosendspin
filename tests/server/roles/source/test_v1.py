"""Behavioural tests for SourceV1Role: decode path, lifecycle, and events."""

from __future__ import annotations

import base64
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
        self.held = False

    @property
    def info(self) -> _FakeInfo:
        return self._info

    def flag_noncompliance(self, reason: str) -> None:
        self.noncompliance.append(reason)

    def awaits_role_state(self, _role_family: str) -> bool:
        return self.held

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


_SOURCE_STATE = ClientStatePayload(available=True, source=SourceStatePayload())


def _connected_role(client: _FakeClient | None = None) -> tuple[SourceV1Role, _FakeClient]:
    """Build a connected role whose client has reported its source state."""
    client = client or _FakeClient()
    role = SourceV1Role(client=client)  # type: ignore[arg-type]
    role.on_connect()
    role.on_initial_client_state(_SOURCE_STATE)
    role.on_client_state(_SOURCE_STATE)
    return role, client


def _make_role() -> tuple[SourceV1Role, _FakeClient]:
    """Build a connected role the server has already asked to stream."""
    role, client = _connected_role()
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
    role, client = _connected_role()
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


def test_accepted_codecs_track_opus_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opus joins the mandatory codecs only when PyAV can handle it."""
    monkeypatch.setattr("aiosendspin.server.roles.source.v1.opus_available", lambda: False)
    assert SourceV1Role.accepted_codecs() == [AudioCodec.FLAC, AudioCodec.PCM]

    monkeypatch.setattr("aiosendspin.server.roles.source.v1.opus_available", lambda: True)
    assert AudioCodec.OPUS in SourceV1Role.accepted_codecs()


def test_unlisted_codec_is_flagged_and_opens_no_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """A codec server/hello did not list is rejected as non-compliant."""
    monkeypatch.setattr("aiosendspin.server.roles.source.v1.opus_available", lambda: False)
    role, client = _make_role()

    role.on_client_stream_start(
        ClientStreamStartPayload(
            source=ClientStreamStartSource(
                codec=AudioCodec.OPUS, channels=2, sample_rate=48000, bit_depth=16
            )
        )
    )

    assert any("server/hello did not list" in reason for reason in client.noncompliance)
    assert not role.stream_active
    assert [e for e in client.events if isinstance(e, SourceStreamStartedEvent)] == []


def test_state_without_source_object_is_flagged() -> None:
    """A client/state required by the activation must carry the source object."""
    role = SourceV1Role(client=_FakeClient())  # type: ignore[arg-type]

    assert role.initial_state_deviations(ClientStatePayload(available=True)) == [
        "has an active source role but no source state"
    ]
    assert role.initial_state_deviations(_SOURCE_STATE) == []


def _commands(client: _FakeClient) -> list[str]:
    return [m.payload.source.command for m in client.sent]


def _assert_start_queued(role: SourceV1Role, client: _FakeClient) -> None:
    """Request a start that can_start holds back: nothing is sent and no stream opens."""
    assert not role.can_start
    role.request_start()
    assert client.sent == []
    role.on_client_stream_start(_pcm_start_payload())
    assert not role.stream_active


def test_start_queued_until_source_state() -> None:
    """A start requested before any client/state is sent once the source state arrives."""
    client = _FakeClient()
    role = SourceV1Role(client=client)  # type: ignore[arg-type]
    role.on_connect()

    _assert_start_queued(role, client)
    role.on_initial_client_state(_SOURCE_STATE)
    role.on_client_state(_SOURCE_STATE)

    assert _commands(client) == ["start"]
    role.on_client_stream_start(_pcm_start_payload())
    assert role.stream_active


def test_start_queued_until_a_state_carries_the_source_object() -> None:
    """A later client/state without the source object does not release a queued start."""
    client = _FakeClient()
    role = SourceV1Role(client=client)  # type: ignore[arg-type]
    role.on_connect()

    _assert_start_queued(role, client)
    role.on_client_state(ClientStatePayload(available=True))
    assert client.sent == []

    role.on_client_state(_SOURCE_STATE)
    assert _commands(client) == ["start"]


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_initial_state_without_source_object_allows_start() -> None:
    """A tolerated pre-#195 client that never sends the source object can still be started."""
    client = _FakeClient()
    role = SourceV1Role(client=client)  # type: ignore[arg-type]
    role.on_connect()
    role.request_start()

    role.on_initial_client_state(ClientStatePayload(available=True))
    role.on_client_state(ClientStatePayload(available=True))

    assert _commands(client) == ["start"]


def test_start_queued_while_unavailable() -> None:
    """A source that reports available: false is started once it is available again."""
    client = _FakeClient()
    client.available = False
    role, _ = _connected_role(client)

    _assert_start_queued(role, client)
    client.available = True
    role.on_client_state(ClientStatePayload(available=True, source=SourceStatePayload()))

    assert _commands(client) == ["start"]


def test_start_sent_immediately_when_startable() -> None:
    """An available client with a source object gets its start command at once."""
    role, client = _connected_role()

    assert role.can_start
    role.request_start()
    role.on_client_stream_start(_pcm_start_payload())

    assert _commands(client) == ["start"]
    assert role.stream_active


def test_queued_start_is_sent_once() -> None:
    """A queued start is consumed when sent, so later states send nothing more."""
    client = _FakeClient()
    role = SourceV1Role(client=client)  # type: ignore[arg-type]
    role.on_connect()
    role.request_start()

    for _ in range(3):
        role.on_client_state(_SOURCE_STATE)

    assert _commands(client) == ["start"]


def test_start_queued_while_role_awaits_activation_state() -> None:
    """A role held for its activation client/state starts when the hold is released."""
    client = _FakeClient()
    client.held = True
    role, _ = _connected_role(client)

    _assert_start_queued(role, client)
    client.held = False
    role.on_hold_released()

    assert _commands(client) == ["start"]


def test_start_queued_after_reactivation_until_new_source_state() -> None:
    """The source object must be reported again for each activation."""
    role, client = _connected_role()
    role.on_deactivate()

    _assert_start_queued(role, client)
    role.on_client_state(_SOURCE_STATE)

    assert _commands(client) == ["start"]


def test_timeout_release_without_source_state_keeps_start_queued() -> None:
    """A hold released by the activation timeout still needs the source object."""
    client = _FakeClient()
    client.held = True
    role = SourceV1Role(client=client)  # type: ignore[arg-type]
    role.on_connect()

    _assert_start_queued(role, client)
    client.held = False
    role.on_hold_released()
    assert client.sent == []

    role.on_client_state(_SOURCE_STATE)
    assert _commands(client) == ["start"]


def test_request_stop_cancels_a_queued_start() -> None:
    """A stop withdraws a start that was still waiting to be sent."""
    client = _FakeClient()
    role = SourceV1Role(client=client)  # type: ignore[arg-type]
    role.on_connect()
    role.request_start()

    role.request_stop()
    role.on_client_state(_SOURCE_STATE)

    assert _commands(client) == ["stop"]


@pytest.mark.parametrize("teardown", ["on_deactivate", "on_disconnect"])
def test_teardown_cancels_a_queued_start(teardown: str) -> None:
    """Deactivation or disconnect withdraws a start that was still waiting to be sent."""
    client = _FakeClient()
    role = SourceV1Role(client=client)  # type: ignore[arg-type]
    role.on_connect()
    role.request_start()

    getattr(role, teardown)()
    role.on_connect()
    role.on_initial_client_state(_SOURCE_STATE)
    role.on_client_state(_SOURCE_STATE)

    assert client.sent == []


def _start_payload(
    codec: AudioCodec, *, channels: int = 2, bit_depth: int = 16, sample_rate: int = 48000
) -> ClientStreamStartPayload:
    header = None
    if codec is AudioCodec.FLAC:
        fields = (sample_rate << 44) | ((channels - 1) << 41) | ((bit_depth - 1) << 36)
        streaminfo = bytes([16, 0, 16, 0]) + bytes(6) + fields.to_bytes(8, "big") + bytes(16)
        header = base64.b64encode(b"fLaC\x80\x00\x00\x22" + streaminfo).decode()
    return ClientStreamStartPayload(
        source=ClientStreamStartSource(
            codec=codec,
            channels=channels,
            sample_rate=sample_rate,
            bit_depth=bit_depth,
            codec_header=header,
        )
    )


@pytest.mark.parametrize(
    ("codec", "channels", "bit_depth", "decoded_bit_depth"),
    [
        (AudioCodec.PCM, 9, 16, 16),
        (AudioCodec.PCM, 11, 24, 24),
        (AudioCodec.PCM, 12, 32, 32),
        (AudioCodec.PCM, 2, 8, 16),
        (AudioCodec.FLAC, 2, 8, 16),
        (AudioCodec.FLAC, 2, 12, 16),
        (AudioCodec.FLAC, 2, 20, 24),
    ],
)
def test_any_decodable_format_opens_at_its_decoded_format(
    codec: AudioCodec, channels: int, bit_depth: int, decoded_bit_depth: int
) -> None:
    """The stream opens for any channel count and announces the decoded PCM format."""
    pytest.importorskip("av")
    role, client = _make_role()

    role.on_client_stream_start(_start_payload(codec, channels=channels, bit_depth=bit_depth))

    assert role.stream_active
    assert client.noncompliance == []
    event = next(e for e in client.events if isinstance(e, SourceStreamStartedEvent))
    assert event.audio_format == AudioFormat(
        sample_rate=48000, bit_depth=decoded_bit_depth, channels=channels
    )
    assert event.handle.audio_format == event.audio_format


async def test_8_bit_pcm_is_streamed_as_16_bit() -> None:
    """8-bit PCM chunks come out of the handle widened to 16-bit samples."""
    role, client = _make_role()
    role.on_client_stream_start(_start_payload(AudioCodec.PCM, channels=1, bit_depth=8))
    handle = next(e for e in client.events if isinstance(e, SourceStreamStartedEvent)).handle

    role.on_binary_chunk(BinaryMessageType.SOURCE_AUDIO_CHUNK.value, 1, bytes([0x40, 0xC0]))
    role.on_client_stream_end()

    assert [chunk async for chunk, _ in handle] == [bytes([0x00, 0x40, 0x00, 0xC0])]


@pytest.mark.parametrize(
    ("codec", "bit_depth", "reason"),
    [
        (AudioCodec.PCM, 12, "pcm bit_depth 12, which is not a whole number of bytes"),
        (AudioCodec.PCM, 0, "unsupported bit_depth 0"),
        (AudioCodec.PCM, 40, "unsupported bit_depth 40"),
        (AudioCodec.FLAC, 33, "unsupported bit_depth 33"),
    ],
)
def test_bit_depth_the_codec_cannot_carry_is_flagged(
    codec: AudioCodec, bit_depth: int, reason: str
) -> None:
    """A depth the codec cannot express is flagged and opens no stream."""
    role, client = _make_role()

    role.on_client_stream_start(_start_payload(codec, bit_depth=bit_depth))

    assert not role.stream_active
    assert client.noncompliance == [f"client-stream/start announced {reason}"]
