"""Tests for VisualizerV1Role (the visualizer@v1 wire)."""

from __future__ import annotations

import asyncio
import struct
from unittest.mock import MagicMock

from aiosendspin.models.core import (
    ClientStatePayload,
    StreamClearMessage,
    StreamEndMessage,
    StreamRequestFormatPayload,
    StreamStartMessage,
)
from aiosendspin.models.types import BinaryMessageType
from aiosendspin.models.visualizer import (
    BeatAvailability,
    BeatTiming,
    ClientHelloVisualizerSpectrum,
    ClientHelloVisualizerSupport,
    StreamRequestFormatVisualizer,
    VisualizerStatePayload,
)
from aiosendspin.noise.keys import Identity
from aiosendspin.server.roles.base import AudioChunk
from aiosendspin.server.roles.visualizer.group import VisualizerGroupRole
from aiosendspin.server.roles.visualizer.packing import FLAG_DOWNBEAT
from aiosendspin.server.roles.visualizer.v1 import VisualizerV1Role
from aiosendspin.server.server import SendspinServer
from tests.server.roles.visualizer.conftest import sine_pcm_16bit


def _make_client_stub() -> MagicMock:
    """Create a mock client preconfigured for visualizer@v1 tests."""
    client = MagicMock()
    client.client_id = "client-1"
    client.group = MagicMock()
    client.group.group_role.return_value = None
    client.info = MagicMock()
    client.info.visualizer_support = ClientHelloVisualizerSupport(buffer_capacity=65536)
    client.visualizer_state = {
        "types": ["loudness", "f_peak", "spectrum"],
        "rate_max": 60,
        "spectrum": {
            "n_disp_bins": 8,
            "scale": "lin",
            "f_min": 20,
            "f_max": 16_000,
        },
    }
    client.send_role_message = MagicMock()
    client.send_binary = MagicMock()
    client._server = MagicMock()  # noqa: SLF001
    client._server.clock.now_us.return_value = 0  # noqa: SLF001
    client._server.visualizer_pitch_enabled = True  # noqa: SLF001
    client.connection = MagicMock()
    return client


def _make_pitch_client_stub() -> MagicMock:
    """Client stub negotiating loudness + pitch."""
    client = _make_client_stub()
    client.visualizer_state = {
        "types": ["loudness", "pitch"],
        "rate_max": 60,
    }
    return client


def _make_beat_client_stub() -> MagicMock:
    """Client stub negotiating both loudness and beat."""
    client = _make_client_stub()
    client.visualizer_state = {
        "types": ["loudness", "beat"],
        "rate_max": 60,
    }
    return client


def _connect(role: VisualizerV1Role) -> None:
    """Connect the role and deliver the stub client's visualizer state, as the connection does."""
    role.on_connect()
    state = VisualizerStatePayload.from_dict(role._client.visualizer_state)  # noqa: SLF001
    payload = ClientStatePayload(visualizer=state)
    role.on_initial_client_state(payload)
    role.on_client_state(payload)


def _audio_chunk(timestamp_us: int = 1_000_000) -> AudioChunk:
    pcm = sine_pcm_16bit(sample_rate=48_000, channels=2, hz=1000.0, duration_s=0.025)
    return AudioChunk(
        data=pcm,
        timestamp_us=timestamp_us,
        duration_us=25_000,
        byte_count=len(pcm),
    )


def _last_stream_start(client: MagicMock) -> StreamStartMessage:
    for call in reversed(client.send_role_message.call_args_list):
        msg = call.args[1]
        if isinstance(msg, StreamStartMessage):
            return msg
    raise AssertionError("no StreamStartMessage was sent")


def _stream_start_count(client: MagicMock) -> int:
    return sum(
        1
        for call in client.send_role_message.call_args_list
        if isinstance(call.args[1], StreamStartMessage)
    )


def _beat_calls(client: MagicMock) -> list:
    return [
        call
        for call in client.send_binary.call_args_list
        if call.kwargs["message_type"] == BinaryMessageType.VISUALIZATION_BEAT.value
    ]


# ---------------------------------------------------------------------------
# Role identity + lifecycle
# ---------------------------------------------------------------------------


def test_role_id_is_v1() -> None:
    """Role id is v1."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    assert role.role_id == "visualizer@v1"
    assert role.role_family == "visualizer"


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_visualizer_role_flags_nonpositive_request_fields() -> None:
    """A stream/request-format with a non-positive field is flagged."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    role.on_stream_request_format(
        StreamRequestFormatPayload(visualizer=StreamRequestFormatVisualizer(rate_max=0))
    )
    client.flag_noncompliance.assert_called_once()


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_visualizer_role_no_flag_for_valid_request_fields() -> None:
    """A stream/request-format with positive fields is not flagged."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    role.on_stream_request_format(
        StreamRequestFormatPayload(visualizer=StreamRequestFormatVisualizer(rate_max=30))
    )
    client.flag_noncompliance.assert_not_called()


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_visualizer_role_flags_request_buffer_capacity() -> None:
    """buffer_capacity is not a visualizer stream/request-format field and is flagged."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    role.on_stream_request_format(
        StreamRequestFormatPayload(visualizer=StreamRequestFormatVisualizer(buffer_capacity=32_768))
    )
    client.flag_noncompliance.assert_called_once()


def test_on_connect_subscribes_to_group_role() -> None:
    """On connect subscribes to group role."""
    client = _make_client_stub()
    group_role = MagicMock()
    client.group.group_role.return_value = group_role

    role = VisualizerV1Role(client=client)
    _connect(role)

    client.group.group_role.assert_called_with("visualizer")
    group_role.subscribe.assert_called_once_with(role)


def test_on_disconnect_unsubscribes() -> None:
    """On disconnect unsubscribes."""
    client = _make_client_stub()
    group_role = MagicMock()
    client.group.group_role.return_value = group_role

    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_disconnect()

    group_role.unsubscribe.assert_called_once_with(role)


def test_wants_beats_true_when_beat_negotiated() -> None:
    """wants_beats is True once the client negotiated `beat` (PENDING default)."""
    role = VisualizerV1Role(client=_make_beat_client_stub())
    _connect(role)
    assert role.wants_beats is True


def test_wants_beats_false_when_beat_not_negotiated() -> None:
    """wants_beats is False when the client never requested `beat`."""
    role = VisualizerV1Role(client=_make_client_stub())
    _connect(role)
    assert role.wants_beats is False


def test_wants_beats_false_when_unavailable() -> None:
    """UNAVAILABLE locks beats out even when the client requested `beat`."""
    role = VisualizerV1Role(client=_make_beat_client_stub())
    _connect(role)
    role.set_beat_availability(BeatAvailability.UNAVAILABLE)
    assert role.wants_beats is False


def test_on_stream_start_sends_stream_start_with_negotiated_config() -> None:
    """on_stream_start emits stream/start with the negotiated config."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()

    message = _last_stream_start(client)
    assert message.payload.visualizer is not None
    assert list(message.payload.visualizer.types) == ["loudness", "f_peak", "spectrum"]
    assert message.payload.visualizer.rate_max == 60
    assert message.payload.visualizer.spectrum is not None
    assert message.payload.visualizer.spectrum.n_disp_bins == 8


def test_on_stream_start_resent_after_stream_end() -> None:
    """stream/start is re-sent after stream/end → on_stream_start cycle."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()
    role.on_stream_end()
    role.on_stream_start()

    assert _stream_start_count(client) == 2


def test_on_stream_start_not_resent_during_active_stream() -> None:
    """A second on_stream_start with no stream/end in between is a no-op (per #239)."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()
    role.on_stream_start()

    assert _stream_start_count(client) == 1


def test_on_deactivate_ends_active_stream() -> None:
    """on_deactivate emits stream/end while a visualizer stream is active."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()
    role.on_deactivate()

    assert any(
        isinstance(call.args[1], StreamEndMessage)
        for call in client.send_role_message.call_args_list
    )


def test_on_deactivate_without_stream_sends_nothing() -> None:
    """on_deactivate is a no-op when no visualizer stream was started."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_deactivate()

    assert not any(
        isinstance(call.args[1], StreamEndMessage)
        for call in client.send_role_message.call_args_list
    )


def test_on_stream_clear_sends_clear_message() -> None:
    """on_stream_clear emits stream/clear."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()
    role.on_stream_clear()

    last = client.send_role_message.call_args.args[1]
    assert isinstance(last, StreamClearMessage)
    assert last.payload.roles == ["visualizer"]


def test_on_stream_end_sends_end_message() -> None:
    """on_stream_end emits stream/end."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()
    role.on_stream_end()

    last = client.send_role_message.call_args.args[1]
    assert isinstance(last, StreamEndMessage)
    assert last.payload.roles == ["visualizer"]


# ---------------------------------------------------------------------------
# Periodic per-type emission from on_audio_chunk
# ---------------------------------------------------------------------------


def test_on_audio_chunk_emits_one_binary_per_periodic_type() -> None:
    """on_audio_chunk emits one binary per negotiated periodic type."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()
    client.send_binary.reset_mock()

    role.on_audio_chunk(_audio_chunk(1_000_000))

    msg_types = [call.kwargs["message_type"] for call in client.send_binary.call_args_list]
    assert sorted(msg_types) == sorted(
        [
            BinaryMessageType.VISUALIZATION_LOUDNESS.value,
            BinaryMessageType.VISUALIZATION_F_PEAK.value,
            BinaryMessageType.VISUALIZATION_SPECTRUM.value,
        ]
    )


def test_loudness_binary_layout_is_type_ts_value() -> None:
    """`loudness`: [16][ts:8][uint16] = 11 bytes."""
    client = _make_client_stub()
    client.visualizer_state = {
        "types": ["loudness"],
        "rate_max": 60,
    }
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()
    client.send_binary.reset_mock()

    role.on_audio_chunk(_audio_chunk(1_234_000))

    client.send_binary.assert_called_once()
    payload = client.send_binary.call_args.args[0]
    kwargs = client.send_binary.call_args.kwargs
    assert payload[0] == BinaryMessageType.VISUALIZATION_LOUDNESS.value
    expected_ts = 1_234_000 + 25_000
    assert struct.unpack(">q", payload[1:9])[0] == expected_ts
    assert len(payload) == 11
    assert kwargs["timestamp_us"] == expected_ts


def test_f_peak_binary_carries_freq_and_amp() -> None:
    """`f_peak`: [18][ts:8][uint16 freq][uint16 amp] = 13 bytes."""
    client = _make_client_stub()
    client.visualizer_state = {
        "types": ["f_peak"],
        "rate_max": 60,
    }
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()
    client.send_binary.reset_mock()

    role.on_audio_chunk(_audio_chunk(2_000_000))

    client.send_binary.assert_called_once()
    payload = client.send_binary.call_args.args[0]
    assert payload[0] == BinaryMessageType.VISUALIZATION_F_PEAK.value
    assert struct.unpack(">q", payload[1:9])[0] == 2_000_000 + 25_000
    freq, amp = struct.unpack(">HH", payload[9:13])
    assert freq > 0
    assert amp > 0


def test_on_audio_chunk_emits_no_periodic_when_only_beat_negotiated() -> None:
    """Beat-only client: no periodic binaries on audio chunks (no FFT extractor)."""
    client = _make_client_stub()
    client.visualizer_state = {
        "types": ["beat"],
        "rate_max": 30,
    }
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()
    client.send_binary.reset_mock()

    role.on_audio_chunk(_audio_chunk())

    client.send_binary.assert_not_called()
    assert role._extractor is None  # noqa: SLF001


def test_audio_chunk_without_stream_start_is_noop() -> None:
    """on_audio_chunk before on_stream_start is a no-op."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_audio_chunk(_audio_chunk())
    client.send_binary.assert_not_called()
    assert role._extractor is None  # noqa: SLF001


# ---------------------------------------------------------------------------
# Beats — deferred type, drained via on_audio_chunk
# ---------------------------------------------------------------------------


def test_initial_stream_start_omits_beat_until_schedule_lands() -> None:
    """`beat` is deferred from the negotiated types until the first schedule arrives."""
    client = _make_beat_client_stub()
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()

    initial = _last_stream_start(client).payload.visualizer
    assert initial is not None
    assert "beat" not in initial.types
    # tracks_downbeats only meaningful when `beat` is negotiated.
    assert initial.tracks_downbeats is None


def test_first_beats_landing_reissues_stream_start_with_beat() -> None:
    """The first non-empty append_beats activates `beat` in the negotiated types."""
    client = _make_beat_client_stub()
    role = VisualizerV1Role(client=client)
    role.set_tracks_downbeats(tracks=True)
    _connect(role)
    role.on_stream_start()
    assert "beat" not in _last_stream_start(client).payload.visualizer.types
    starts_before = _stream_start_count(client)

    role.append_beats([BeatTiming(1_000_000)])

    assert _stream_start_count(client) == starts_before + 1
    after = _last_stream_start(client).payload.visualizer
    assert after is not None
    assert "beat" in after.types
    assert after.tracks_downbeats is True


def test_subsequent_beats_landings_do_not_reissue_stream_start() -> None:
    """Only the first batch flips `_has_beats_landed`; later batches are no-ops on the config."""
    client = _make_beat_client_stub()
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()
    role.append_beats([BeatTiming(1_000_000)])  # first landing — re-emits
    starts_after_first = _stream_start_count(client)

    role.append_beats([BeatTiming(2_000_000)])
    role.append_beats([BeatTiming(3_000_000)])

    assert _stream_start_count(client) == starts_after_first


def test_beats_drain_on_next_audio_chunk() -> None:
    """Beats sit in the pending queue and emit on the next on_audio_chunk drain."""
    client = _make_beat_client_stub()
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()

    role.append_beats([BeatTiming(500_000), BeatTiming(1_500_000)])
    # No beats on the wire yet — drain is audio-driven.
    assert _beat_calls(client) == []

    role.on_audio_chunk(_audio_chunk(1_000_000))

    beat_ts = [c.kwargs["timestamp_us"] for c in _beat_calls(client)]
    assert beat_ts == [500_000]

    role.on_audio_chunk(_audio_chunk(2_000_000))
    beat_ts = [c.kwargs["timestamp_us"] for c in _beat_calls(client)]
    assert beat_ts == [500_000, 1_500_000]


def test_downbeat_flag_masked_when_not_tracking_downbeats() -> None:
    """The downbeat bit must be 0 when the role does not track downbeats."""
    client = _make_beat_client_stub()
    role = VisualizerV1Role(client=client)
    role.set_tracks_downbeats(tracks=False)
    _connect(role)
    role.on_stream_start()
    role.append_beats([BeatTiming(500_000, is_downbeat=True)])
    role.on_audio_chunk(_audio_chunk(1_000_000))

    calls = _beat_calls(client)
    assert len(calls) == 1
    assert calls[0].args[0][-1] & FLAG_DOWNBEAT == 0


def test_downbeat_flag_set_when_tracking_downbeats() -> None:
    """A downbeat sets the bit when the role tracks downbeats."""
    client = _make_beat_client_stub()
    role = VisualizerV1Role(client=client)
    role.set_tracks_downbeats(tracks=True)
    _connect(role)
    role.on_stream_start()
    role.append_beats([BeatTiming(500_000, is_downbeat=True)])
    role.on_audio_chunk(_audio_chunk(1_000_000))

    calls = _beat_calls(client)
    assert len(calls) == 1
    assert calls[0].args[0][-1] & FLAG_DOWNBEAT == FLAG_DOWNBEAT


def test_beats_interleave_with_periodic_frames_in_ts_order() -> None:
    """All wire timestamps stay non-decreasing across periodic + beat frames."""
    client = _make_beat_client_stub()
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()
    role.append_beats([BeatTiming(500_000), BeatTiming(1_500_000)])
    client.send_binary.reset_mock()

    role.on_audio_chunk(_audio_chunk(1_000_000))
    role.on_audio_chunk(_audio_chunk(2_000_000))

    ts_values = [c.kwargs["timestamp_us"] for c in client.send_binary.call_args_list]
    assert ts_values == sorted(ts_values)


def test_beat_binary_layout() -> None:
    """`beat` binary: [17][ts:8][flags:1] = 10 bytes."""
    client = _make_beat_client_stub()
    role = VisualizerV1Role(client=client)
    role.set_tracks_downbeats(tracks=True)  # so the downbeat bit is exposed
    _connect(role)
    role.on_stream_start()
    role.append_beats(
        [
            BeatTiming(100, is_downbeat=True),
            BeatTiming(200),
        ]
    )
    role.on_audio_chunk(_audio_chunk(1_000_000))

    payloads = [call.args[0] for call in _beat_calls(client)]
    assert len(payloads) == 2
    for payload in payloads:
        assert len(payload) == 10
        assert payload[0] == BinaryMessageType.VISUALIZATION_BEAT.value
    assert struct.unpack(">q", payloads[0][1:9])[0] == 100
    assert payloads[0][9] == 0b0000_0001
    assert struct.unpack(">q", payloads[1][1:9])[0] == 200
    assert payloads[1][9] == 0


def test_append_beats_empty_is_noop() -> None:
    """Empty append_beats neither queues nor reissues stream/start."""
    client = _make_beat_client_stub()
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()
    starts_before = _stream_start_count(client)

    role.append_beats([])

    assert _stream_start_count(client) == starts_before
    assert list(role._pending_beats) == []  # noqa: SLF001


def test_append_beats_noop_without_beat_type() -> None:
    """Client without `beat` in supported types: append_beats has no effect."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()
    client.send_binary.reset_mock()
    starts_before = _stream_start_count(client)

    role.append_beats([BeatTiming(1), BeatTiming(2)])
    role.on_audio_chunk(_audio_chunk())

    assert _beat_calls(client) == []
    assert _stream_start_count(client) == starts_before


def test_clear_beats_drops_pending() -> None:
    """clear_beats empties the pending schedule."""
    client = _make_beat_client_stub()
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()
    role.append_beats([BeatTiming(1_000_000), BeatTiming(60_000_000)])
    assert list(role._pending_beats) != []  # noqa: SLF001

    role.clear_beats()

    assert list(role._pending_beats) == []  # noqa: SLF001


def test_clear_beats_reissues_stream_start_to_drop_beat() -> None:
    """After clear_beats the negotiated types no longer expose `beat` until next landing."""
    client = _make_beat_client_stub()
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()
    role.append_beats([BeatTiming(1_000_000)])
    assert "beat" in _last_stream_start(client).payload.visualizer.types

    role.clear_beats()

    assert "beat" not in _last_stream_start(client).payload.visualizer.types


def test_emit_beats_drops_duplicate_ts() -> None:
    """Beats whose ts equals the most-recently-emitted one are dropped (`<=` guard)."""
    client = _make_beat_client_stub()
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()
    role.append_beats([BeatTiming(500_000)])
    role.on_audio_chunk(_audio_chunk(1_000_000))  # emits 500_000
    client.send_binary.reset_mock()

    # Same ts again — should be dropped by the `<=` wire-ts guard.
    role._pending_beats.append(BeatTiming(500_000))  # noqa: SLF001
    role.on_audio_chunk(_audio_chunk(1_025_000))

    assert _beat_calls(client) == []


def test_join_ordering_beats_before_stream_start_drains_after_start() -> None:
    """Beats appended before on_stream_start (mid-stream join) drain after start."""
    client = _make_beat_client_stub()
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.append_beats([BeatTiming(500_000)])
    # Before on_stream_start: nothing on the wire (no stream/start emitted yet
    # either, so the client wouldn't know the config).
    assert _beat_calls(client) == []

    role.on_stream_start()
    # The first batch arrived during warmup → `beat` is already in the
    # initial stream/start (it landed before the start), so no second
    # stream/start is needed.
    assert "beat" in _last_stream_start(client).payload.visualizer.types
    assert _beat_calls(client) == []  # still waiting for a chunk to drain

    role.on_audio_chunk(_audio_chunk(1_000_000))
    beat_ts = [c.kwargs["timestamp_us"] for c in _beat_calls(client)]
    assert beat_ts == [500_000]


# ---------------------------------------------------------------------------
# Availability transitions
# ---------------------------------------------------------------------------


def test_unavailable_blocks_beat_activation() -> None:
    """UNAVAILABLE makes append_beats a no-op and keeps `beat` out of types."""
    client = _make_beat_client_stub()
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.set_beat_availability(BeatAvailability.UNAVAILABLE)
    role.on_stream_start()

    role.append_beats([BeatTiming(1_000_000)])
    role.on_audio_chunk(_audio_chunk(2_000_000))

    assert "beat" not in _last_stream_start(client).payload.visualizer.types
    assert _beat_calls(client) == []


def test_unavailable_after_landing_clears_beats_and_reissues() -> None:
    """Flipping to UNAVAILABLE drops the pending schedule and re-emits stream/start."""
    client = _make_beat_client_stub()
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()
    role.append_beats([BeatTiming(1_000_000)])
    assert "beat" in _last_stream_start(client).payload.visualizer.types

    role.set_beat_availability(BeatAvailability.UNAVAILABLE)

    assert "beat" not in _last_stream_start(client).payload.visualizer.types
    assert list(role._pending_beats) == []  # noqa: SLF001


def test_unavailable_then_pending_requires_fresh_beats_for_reactivation() -> None:
    """PENDING after UNAVAILABLE does not re-add `beat`; a fresh schedule must land."""
    client = _make_beat_client_stub()
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()
    role.append_beats([BeatTiming(1_000_000)])
    role.set_beat_availability(BeatAvailability.UNAVAILABLE)
    assert "beat" not in _last_stream_start(client).payload.visualizer.types

    role.set_beat_availability(BeatAvailability.PENDING)
    # Flipping back to PENDING alone is not enough — the schedule was wiped.
    assert "beat" not in _last_stream_start(client).payload.visualizer.types

    role.append_beats([BeatTiming(2_000_000)])
    assert "beat" in _last_stream_start(client).payload.visualizer.types


def test_beat_only_stream_initial_includes_beat() -> None:
    """Beat-only clients see `beat` from the start (no FFT type to fall back to)."""
    client = _make_client_stub()
    client.visualizer_state = {
        "types": ["beat"],
        "rate_max": 30,
    }
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()

    initial = _last_stream_start(client).payload.visualizer
    assert list(initial.types) == ["beat"]


# ---------------------------------------------------------------------------
# stream/request-format renegotiation
# ---------------------------------------------------------------------------


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_request_format_replaces_spectrum() -> None:
    """request-format replaces the spectrum config and rebuilds the extractor."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()
    original_extractor = role._extractor  # noqa: SLF001

    new_spectrum = ClientHelloVisualizerSpectrum(
        n_disp_bins=24, scale="mel", f_min=40, f_max=14_000
    )
    payload = StreamRequestFormatPayload(
        visualizer=StreamRequestFormatVisualizer(spectrum=new_spectrum)
    )
    role.on_stream_request_format(payload)

    assert role._stream_config is not None  # noqa: SLF001
    assert role._stream_config.spectrum is not None  # noqa: SLF001
    assert role._stream_config.spectrum.n_disp_bins == 24  # noqa: SLF001
    assert role._extractor is not original_extractor  # noqa: SLF001


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_request_format_replaces_rate_max() -> None:
    """request-format replaces rate_max."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()

    payload = StreamRequestFormatPayload(visualizer=StreamRequestFormatVisualizer(rate_max=15))
    role.on_stream_request_format(payload)

    assert role._stream_config is not None  # noqa: SLF001
    assert role._stream_config.rate_max == 15  # noqa: SLF001


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_request_format_with_no_active_stream_does_not_start_stream() -> None:
    """The server MUST NOT start a stream in response to request-format when none is active."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    _connect(role)  # no on_stream_start yet -> no active stream
    client.send_role_message.reset_mock()

    role.on_stream_request_format(
        StreamRequestFormatPayload(visualizer=StreamRequestFormatVisualizer(rate_max=15))
    )

    starts = [
        c
        for c in client.send_role_message.call_args_list
        if isinstance(c.args[1], StreamStartMessage)
    ]
    assert starts == []
    assert role._stream_started is False  # noqa: SLF001


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_request_format_with_no_active_stream_is_remembered_for_next_stream() -> None:
    """With no active stream, the request is remembered and applied to the next stream/start."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    _connect(role)  # no on_stream_start yet -> no active stream

    role.on_stream_request_format(
        StreamRequestFormatPayload(visualizer=StreamRequestFormatVisualizer(rate_max=15))
    )
    role.on_stream_start()

    cfg = _last_stream_start(client).payload.visualizer
    assert cfg is not None
    assert cfg.rate_max == 15


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_request_format_replaces_types() -> None:
    """request-format replaces the negotiated types."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()

    payload = StreamRequestFormatPayload(
        visualizer=StreamRequestFormatVisualizer(types=["loudness"])
    )
    role.on_stream_request_format(payload)

    assert role._stream_config is not None  # noqa: SLF001
    assert list(role._stream_config.types) == ["loudness"]  # noqa: SLF001


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_request_format_emits_new_stream_start() -> None:
    """A request-format that changes the config emits a fresh stream/start."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()
    starts_before = _stream_start_count(client)

    payload = StreamRequestFormatPayload(visualizer=StreamRequestFormatVisualizer(rate_max=20))
    role.on_stream_request_format(payload)

    assert _stream_start_count(client) == starts_before + 1


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_request_format_without_change_sends_no_stream_start() -> None:
    """A request-format that leaves the derived config unchanged sends no stream/start."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()
    starts_before = _stream_start_count(client)

    role.on_stream_request_format(
        StreamRequestFormatPayload(visualizer=StreamRequestFormatVisualizer(rate_max=60))
    )

    assert _stream_start_count(client) == starts_before


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_request_format_adding_spectrum_without_spectrum_object_falls_back() -> None:
    """A `types` change adding `spectrum` without a `spectrum` object is normalized away."""
    client = _make_client_stub()
    client.visualizer_state = {
        "types": ["loudness"],
        "rate_max": 60,
    }
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()

    payload = StreamRequestFormatPayload(
        visualizer=StreamRequestFormatVisualizer(types=["loudness", "spectrum"])
    )
    role.on_stream_request_format(payload)

    # No spectrum object available → `spectrum` is filtered out by normalization.
    assert "spectrum" not in role._stream_config.types  # noqa: SLF001
    # And no crash in pack_spectrum on the next chunk.
    role.on_audio_chunk(_audio_chunk(1_000_000))


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_request_format_clears_pending_beats_and_re_defers() -> None:
    """request-format drops pending beats and re-defers `beat` until next landing."""
    client = _make_beat_client_stub()
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()
    role.append_beats([BeatTiming(1_000_000)])
    assert "beat" in _last_stream_start(client).payload.visualizer.types
    assert list(role._pending_beats) != []  # noqa: SLF001

    role.on_stream_request_format(
        StreamRequestFormatPayload(visualizer=StreamRequestFormatVisualizer(rate_max=15))
    )

    assert "beat" not in _last_stream_start(client).payload.visualizer.types
    assert list(role._pending_beats) == []  # noqa: SLF001


# ---------------------------------------------------------------------------
# Buffer tracker + binary handling
# ---------------------------------------------------------------------------


def test_buffer_tracker_uses_hello_capacity_with_state_config() -> None:
    """buffer_capacity comes from the hello while the stream config comes from client/state."""
    client = _make_client_stub()
    client.info.visualizer_support = ClientHelloVisualizerSupport(buffer_capacity=1234)
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()
    tracker = role.get_buffer_tracker()
    assert tracker is not None
    assert tracker.capacity_bytes == 1234


def test_buffer_tracker_resets_on_stream_clear() -> None:
    """Buffer tracker resets on stream clear."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()
    role.on_stream_clear()

    tracker = role.get_buffer_tracker()
    assert tracker is not None
    assert tracker.buffered_bytes == 0


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_buffer_tracker_capacity_shrink_does_not_reset_buffered_bytes() -> None:
    """Mid-stream capacity changes update the limit but keep buffered_bytes intact."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()
    tracker = role.get_buffer_tracker()
    assert tracker is not None
    # Simulate already-sent bytes the client still holds.
    tracker.register(end_time_us=1_000_000, byte_count=4096, duration_us=25_000)
    assert tracker.buffered_bytes == 4096

    payload = StreamRequestFormatPayload(
        visualizer=StreamRequestFormatVisualizer(buffer_capacity=32_768)
    )
    role.on_stream_request_format(payload)

    assert tracker.capacity_bytes == 32_768
    assert tracker.buffered_bytes == 4096


def test_binary_handling_for_all_visualizer_types() -> None:
    """get_binary_handling covers all per-type wire bytes."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    for member in (
        BinaryMessageType.VISUALIZATION_LOUDNESS,
        BinaryMessageType.VISUALIZATION_BEAT,
        BinaryMessageType.VISUALIZATION_F_PEAK,
        BinaryMessageType.VISUALIZATION_SPECTRUM,
        BinaryMessageType.VISUALIZATION_PEAK,
        BinaryMessageType.VISUALIZATION_PITCH,
    ):
        handling = role.get_binary_handling(member.value)
        assert handling is not None
        assert handling.drop_late is True
        assert handling.buffer_track is True


# ---------------------------------------------------------------------------
# client/state visualizer object
# ---------------------------------------------------------------------------


def _state(**fields: object) -> ClientStatePayload:
    return ClientStatePayload(visualizer=VisualizerStatePayload.from_dict(fields))


_SPECTRUM = {"n_disp_bins": 8, "scale": "lin", "f_min": 20, "f_max": 16_000}


def test_no_stream_before_state_object() -> None:
    """A current hello starts no stream and sends no binary before the state object."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    role.on_connect()
    role.on_stream_start()
    role.on_audio_chunk(_audio_chunk())

    assert _stream_start_count(client) == 0
    client.send_binary.assert_not_called()


def test_clear_and_end_before_state_object_send_nothing() -> None:
    """A stream cleared or ended before the state object arrives was never announced."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    role.on_connect()
    role.on_stream_start()
    role.on_stream_clear()
    role.on_stream_end()

    client.send_role_message.assert_not_called()

    role.on_client_state(_state(types=["loudness"], rate_max=30))
    role.on_stream_start()
    role.on_stream_clear()
    role.on_stream_end()

    sent = [type(call.args[1]) for call in client.send_role_message.call_args_list]
    assert sent == [StreamStartMessage, StreamClearMessage, StreamEndMessage]


def test_first_state_requesting_beats_replays_group_schedule() -> None:
    """A state object that first requests `beat` receives the group's current schedule."""
    client = _make_client_stub()
    client.group._server.clock.now_us.return_value = 0  # noqa: SLF001
    group_role = VisualizerGroupRole(client.group)
    client.group.group_role.return_value = group_role
    group_role.append_beat_schedule([BeatTiming(1_000_000), BeatTiming(1_500_000)])
    role = VisualizerV1Role(client=client)
    role.on_connect()

    role.on_client_state(_state(types=["loudness", "beat"], rate_max=30))
    role.on_stream_start()

    assert "beat" in _last_stream_start(client).payload.visualizer.types
    assert [b.timestamp_us for b in role._pending_beats] == [1_000_000, 1_500_000]  # noqa: SLF001


def test_first_state_object_joins_running_stream() -> None:
    """The first state object asks the client to join the group's running stream."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    role.on_connect()
    role.on_stream_start()

    role.on_client_state(_state(types=["loudness"], rate_max=30))
    role.on_client_state(_state(types=["loudness"], rate_max=20))

    client.join_active_stream.assert_called_once_with(role)


def test_stream_start_follows_state_request() -> None:
    """stream/start carries the requested types it can stream now and the requested rate_max."""
    client = _make_client_stub()
    client.visualizer_state = {"types": ["loudness", "beat"], "rate_max": 12}
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()

    config = _last_stream_start(client).payload.visualizer
    assert config.types == ("loudness",)
    assert config.rate_max == 12


def test_empty_types_stream_sends_no_frames() -> None:
    """`types: []` is accepted without flagging and yields no data frames."""
    client = _make_client_stub()
    client.visualizer_state = {"types": [], "rate_max": 30}
    role = VisualizerV1Role(client=client)
    assert role.client_state_deviations(_state(types=[], rate_max=30)) == []
    _connect(role)
    role.on_stream_start()
    role.on_audio_chunk(_audio_chunk())

    assert _last_stream_start(client).payload.visualizer.types == ()
    client.send_binary.assert_not_called()
    client.flag_noncompliance.assert_not_called()


def test_identical_state_sends_no_stream_start() -> None:
    """Repeating the same state object during a stream sends no new stream/start."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()

    role.on_client_state(_state(**client.visualizer_state))

    assert _stream_start_count(client) == 1


def test_changed_state_during_stream_sends_one_stream_start() -> None:
    """A changed state object during a stream sends exactly one new stream/start."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()

    role.on_client_state(_state(types=["loudness"], rate_max=15))

    assert _stream_start_count(client) == 2
    config = _last_stream_start(client).payload.visualizer
    assert config.types == ("loudness",)
    assert config.rate_max == 15
    assert config.spectrum is None


def test_changed_state_without_stream_applies_to_next_stream() -> None:
    """A changed state object with no active stream sends nothing until the next stream."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    _connect(role)

    role.on_client_state(_state(types=["loudness"], rate_max=15))
    assert _stream_start_count(client) == 0

    role.on_stream_start()
    config = _last_stream_start(client).payload.visualizer
    assert config.types == ("loudness",)
    assert config.rate_max == 15


def test_state_change_not_affecting_config_sends_no_stream_start() -> None:
    """Requesting `beat` before any schedule landed leaves stream/start unchanged."""
    client = _make_client_stub()
    client.visualizer_state = {"types": ["loudness"], "rate_max": 30}
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()

    role.on_client_state(_state(types=["loudness", "beat"], rate_max=30))

    assert _stream_start_count(client) == 1
    assert role.wants_beats is True
    assert role._holdback_active is True  # noqa: SLF001


def test_spectrum_without_config_is_flagged_and_not_streamed() -> None:
    """`spectrum` without a spectrum object is a deviation; leniently the rest streams."""
    client = _make_client_stub()
    client.visualizer_state = {"types": ["loudness", "spectrum"], "rate_max": 30}
    role = VisualizerV1Role(client=client)
    reasons = role.client_state_deviations(_state(**client.visualizer_state))
    assert reasons == ["requested visualizer 'spectrum' without a spectrum configuration"]

    _connect(role)
    role.on_stream_start()

    assert _last_stream_start(client).payload.visualizer.types == ("loudness",)


def test_nonpositive_rate_max_is_flagged_and_ignored() -> None:
    """A non-positive rate_max is a deviation; leniently the prior request is kept."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    bad = _state(types=["loudness"], rate_max=0)
    assert role.client_state_deviations(bad) == ["sent a non-positive visualizer rate_max: 0"]

    _connect(role)
    role.on_stream_start()
    role.on_client_state(bad)

    assert _stream_start_count(client) == 1
    assert _last_stream_start(client).payload.visualizer.rate_max == 60


def test_state_without_visualizer_object_is_ignored() -> None:
    """A client/state without a visualizer object leaves the request unchanged."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()

    role.on_client_state(ClientStatePayload(available=True))

    assert _stream_start_count(client) == 1
    assert role.client_state_deviations(ClientStatePayload(available=True)) == []


def test_initial_state_without_visualizer_object_is_a_deviation() -> None:
    """A current client's initial client/state must carry the visualizer object."""
    role = VisualizerV1Role(client=_make_client_stub())
    assert role.initial_state_deviations(ClientStatePayload(available=True)) == [
        "has an active visualizer role but no visualizer state"
    ]
    assert role.initial_state_deviations(_state(types=[], rate_max=30)) == []


def test_reconnect_waits_for_fresh_state_object() -> None:
    """A reconnect drops the previous request instead of streaming from stale state."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_disconnect()
    role.on_connect()
    role.on_stream_start()

    assert _stream_start_count(client) == 0


# ---------------------------------------------------------------------------
# Pre-#195 hello stream configuration
# ---------------------------------------------------------------------------


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_legacy_hello_streams_without_state_object() -> None:
    """A hello carrying stream configuration streams with it without a state object."""
    client = _make_client_stub()
    client.info.visualizer_support = ClientHelloVisualizerSupport(
        types=["loudness", "f_peak"],
        buffer_capacity=65536,
        rate_max=30,
    )
    role = VisualizerV1Role(client=client)
    role.on_connect()
    role.on_stream_start()

    message = _last_stream_start(client)
    assert message.payload.visualizer is not None
    assert list(message.payload.visualizer.types) == ["loudness", "f_peak"]
    assert message.payload.visualizer.rate_max == 30
    assert role.initial_state_deviations(ClientStatePayload(available=True)) == []


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_legacy_hello_defaults_and_spectrum_without_config() -> None:
    """A legacy hello keeps the old defaults and drops `spectrum` without a config."""
    client = _make_client_stub()
    client.info.visualizer_support = ClientHelloVisualizerSupport(
        buffer_capacity=65536, rate_max=20
    )
    role = VisualizerV1Role(client=client)
    role.on_connect()
    role.on_stream_start()
    assert list(_last_stream_start(client).payload.visualizer.types) == ["loudness", "f_peak"]

    client = _make_client_stub()
    client.info.visualizer_support = ClientHelloVisualizerSupport(
        buffer_capacity=65536, types=["loudness", "spectrum"]
    )
    role = VisualizerV1Role(client=client)
    role.on_connect()
    role.on_stream_start()
    config = _last_stream_start(client).payload.visualizer
    assert list(config.types) == ["loudness"]
    assert config.rate_max == 30


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_state_object_replaces_legacy_hello_config() -> None:
    """A state object after a legacy hello replaces its configuration."""
    client = _make_client_stub()
    client.info.visualizer_support = ClientHelloVisualizerSupport(
        types=["loudness", "f_peak"],
        buffer_capacity=65536,
        rate_max=30,
    )
    role = VisualizerV1Role(client=client)
    role.on_connect()
    role.on_stream_start()

    role.on_client_state(_state(types=["spectrum"], rate_max=10, spectrum=_SPECTRUM))

    config = _last_stream_start(client).payload.visualizer
    assert config.types == ("spectrum",)
    assert config.rate_max == 10
    client.join_active_stream.assert_not_called()


# ---------------------------------------------------------------------------
# Type filtering
# ---------------------------------------------------------------------------


def test_unknown_requested_types_are_omitted() -> None:
    """Unknown requested types never reach stream/start."""
    client = _make_client_stub()
    client.visualizer_state = {
        "types": ["loudness", "_not_a_real_type"],
        "rate_max": 30,
    }
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()

    message = _last_stream_start(client)
    assert message.payload.visualizer is not None
    assert list(message.payload.visualizer.types) == ["loudness"]


def test_pitch_emits_msg_21_with_midi_and_confidence() -> None:
    """Pure-tone audio yields a confident pitch frame in MIDI 8.8 fixed-point."""
    client = _make_client_stub()
    client.visualizer_state = {
        "types": ["pitch"],
        "rate_max": 30,
    }
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()
    client.send_binary.reset_mock()

    pcm = sine_pcm_16bit(sample_rate=48_000, channels=2, hz=440.0, duration_s=0.05)
    role.on_audio_chunk(
        AudioChunk(data=pcm, timestamp_us=1_000_000, duration_us=50_000, byte_count=len(pcm))
    )

    assert client.send_binary.call_count == 1
    payload = client.send_binary.call_args.args[0]
    assert payload[0] == BinaryMessageType.VISUALIZATION_PITCH.value
    assert struct.unpack(">q", payload[1:9])[0] == 1_050_000
    (midi_q88,) = struct.unpack(">H", payload[9:11])
    confidence = payload[11]
    midi_float = midi_q88 / 256.0
    assert abs(midi_float - 69.0) < 1.0
    assert confidence > 128


# ---------------------------------------------------------------------------
# Warmup holdback: cap periodic send-ahead while beats are pending
# ---------------------------------------------------------------------------


def _periodic_calls(client: MagicMock) -> list:
    """send_binary calls for periodic (non-beat) frames."""
    return [
        call
        for call in client.send_binary.call_args_list
        if call.kwargs["message_type"] != BinaryMessageType.VISUALIZATION_BEAT.value
    ]


async def test_warmup_holds_periodic_frames_beyond_lead() -> None:
    """While beats are pending, periodic frames past the warmup lead are held."""
    client = _make_beat_client_stub()  # loudness + beat, now_us=0 → cutoff 3s
    role = VisualizerV1Role(client)
    _connect(role)
    role.on_stream_start()
    client.send_binary.reset_mock()
    role.on_audio_chunk(_audio_chunk(timestamp_us=5_000_000))  # frame ~5.025s > lead
    assert _periodic_calls(client) == []
    role._cancel_release_timer()  # noqa: SLF001


async def test_warmup_passes_frames_within_lead() -> None:
    """Periodic frames within the warmup lead send immediately while pending."""
    client = _make_beat_client_stub()
    role = VisualizerV1Role(client)
    _connect(role)
    role.on_stream_start()
    client.send_binary.reset_mock()
    role.on_audio_chunk(_audio_chunk(timestamp_us=1_000_000))  # frame ~1.025s < lead
    assert _periodic_calls(client)


async def test_no_holdback_when_beats_not_wanted() -> None:
    """Without beat negotiated, far-ahead frames are never held."""
    client = _make_client_stub()  # loudness/f_peak/spectrum, no beat
    role = VisualizerV1Role(client)
    _connect(role)
    role.on_stream_start()
    client.send_binary.reset_mock()
    role.on_audio_chunk(_audio_chunk(timestamp_us=5_000_000))
    assert _periodic_calls(client)


async def test_first_beats_keep_cap_release_in_ts_order() -> None:
    """Landing beats keeps the cap; the held frame and beat release in ts order on advance."""
    client = _make_beat_client_stub()
    role = VisualizerV1Role(client)
    _connect(role)
    role.on_stream_start()
    client.send_binary.reset_mock()
    role.on_audio_chunk(_audio_chunk(timestamp_us=5_000_000))  # frame ~5.025s held (cutoff 3s)
    assert _periodic_calls(client) == []
    role.append_beats([BeatTiming(5_010_000)])  # beyond the cap → pending, cap not lifted
    assert _beat_calls(client) == [], "beat beyond the cap must not emit on landing"
    assert _periodic_calls(client) == [], "held frame must stay parked while beats are wanted"
    # Playhead advances past the held frame/beat: both release, in ts order.
    client._server.clock.now_us.return_value = 5_100_000  # cutoff 8.1s  # noqa: SLF001
    role._run_release_scheduler()  # noqa: SLF001
    sent_ts = [call.kwargs["timestamp_us"] for call in client.send_binary.call_args_list]
    assert sent_ts == sorted(sent_ts), "wire timestamps must stay non-decreasing"
    assert _beat_calls(client), "beat should emit once within the cap"
    assert _periodic_calls(client), "held frame should release once within the cap"
    role._cancel_release_timer()  # noqa: SLF001


async def test_unavailable_flushes_held_frames() -> None:
    """Declaring beats UNAVAILABLE lifts the cap and releases held frames."""
    client = _make_beat_client_stub()
    role = VisualizerV1Role(client)
    _connect(role)
    role.on_stream_start()
    client.send_binary.reset_mock()
    role.on_audio_chunk(_audio_chunk(timestamp_us=5_000_000))
    assert _periodic_calls(client) == []
    role.set_beat_availability(BeatAvailability.UNAVAILABLE)
    assert _periodic_calls(client)


async def test_release_scheduler_sends_frames_as_playhead_advances() -> None:
    """Held frames release once the playhead advances within the warmup lead."""
    client = _make_beat_client_stub()
    role = VisualizerV1Role(client)
    _connect(role)
    role.on_stream_start()
    client.send_binary.reset_mock()
    role.on_audio_chunk(_audio_chunk(timestamp_us=5_000_000))  # held ~5.025s
    assert _periodic_calls(client) == []
    client._server.clock.now_us.return_value = 4_000_000  # cutoff 7s  # noqa: SLF001
    role._run_release_scheduler()  # noqa: SLF001
    assert _periodic_calls(client)


# ---------------------------------------------------------------------------
# Server-wide pitch shed toggle
# ---------------------------------------------------------------------------


def test_pitch_in_types_when_server_enabled() -> None:
    """Pitch stays in the negotiated types while the server flag is on (default)."""
    client = _make_pitch_client_stub()
    role = VisualizerV1Role(client)
    _connect(role)
    role.on_stream_start()
    assert "pitch" in _last_stream_start(client).payload.visualizer.types


def test_pitch_dropped_when_server_disabled() -> None:
    """Pitch is excluded from negotiated types when the server flag is off."""
    client = _make_pitch_client_stub()
    client._server.visualizer_pitch_enabled = False  # noqa: SLF001
    role = VisualizerV1Role(client)
    _connect(role)
    role.on_stream_start()
    types = _last_stream_start(client).payload.visualizer.types
    assert "pitch" not in types
    assert "loudness" in types


def test_pitch_dropped_when_clients_must_be_compliant() -> None:
    """The pitch toggle is ignored (pitch shed) when non-compliant clients are disallowed."""
    client = _make_pitch_client_stub()
    client._server.visualizer_pitch_enabled = True  # noqa: SLF001
    client._server.allow_noncompliant_clients = False  # noqa: SLF001
    role = VisualizerV1Role(client)
    _connect(role)
    role.on_stream_start()
    types = _last_stream_start(client).payload.visualizer.types
    assert "pitch" not in types
    assert "loudness" in types


def test_disabled_pitch_emits_no_pitch_binary() -> None:
    """With pitch disabled, no PITCH binary is produced from an audio chunk."""
    client = _make_pitch_client_stub()
    client._server.visualizer_pitch_enabled = False  # noqa: SLF001
    role = VisualizerV1Role(client)
    _connect(role)
    role.on_stream_start()
    client.send_binary.reset_mock()
    role.on_audio_chunk(_audio_chunk(timestamp_us=1_000_000))
    msg_types = [call.kwargs["message_type"] for call in client.send_binary.call_args_list]
    assert BinaryMessageType.VISUALIZATION_PITCH.value not in msg_types


def test_pitch_kept_when_sole_type_even_if_disabled() -> None:
    """A pitch-only client keeps pitch — types must not be emptied."""
    client = _make_client_stub()
    client.visualizer_state = {
        "types": ["pitch"],
        "rate_max": 60,
    }
    client._server.visualizer_pitch_enabled = False  # noqa: SLF001
    role = VisualizerV1Role(client)
    _connect(role)
    role.on_stream_start()
    assert _last_stream_start(client).payload.visualizer.types == ("pitch",)


def test_pitch_only_client_is_inert_when_clients_must_be_compliant() -> None:
    """A pitch-only client gets an empty stream in strict mode, which omits `pitch`."""
    client = _make_client_stub()
    client.visualizer_state = {
        "types": ["pitch"],
        "rate_max": 60,
    }
    client._server.allow_noncompliant_clients = False  # noqa: SLF001
    role = VisualizerV1Role(client)
    _connect(role)
    role.on_stream_start()
    assert _last_stream_start(client).payload.visualizer.types == ()
    client.send_binary.reset_mock()
    role.on_audio_chunk(_audio_chunk(timestamp_us=1_000_000))
    client.send_binary.assert_not_called()


def test_refresh_pitch_setting_reissues_stream_start_on_change() -> None:
    """Flipping the server flag live re-emits stream/start without pitch."""
    client = _make_pitch_client_stub()
    role = VisualizerV1Role(client)
    _connect(role)
    role.on_stream_start()
    before = _stream_start_count(client)
    client._server.visualizer_pitch_enabled = False  # noqa: SLF001
    role.refresh_pitch_setting()
    assert _stream_start_count(client) == before + 1
    assert "pitch" not in _last_stream_start(client).payload.visualizer.types


def test_refresh_pitch_setting_noop_when_unchanged() -> None:
    """refresh_pitch_setting does not re-emit when the resolved types are unchanged."""
    client = _make_pitch_client_stub()  # flag stays enabled
    role = VisualizerV1Role(client)
    _connect(role)
    role.on_stream_start()
    before = _stream_start_count(client)
    role.refresh_pitch_setting()
    assert _stream_start_count(client) == before


async def test_visualizer_pitch_disabled_by_default() -> None:
    """Pitch is off by default; emitting reserved type 21 is opt-in and non-spec."""
    server = SendspinServer(
        asyncio.get_running_loop(),
        Identity.generate(),
        "Srv",
        MagicMock(),
        pairing_store=MagicMock(),
    )
    assert server.visualizer_pitch_enabled is False


async def test_server_set_pitch_enabled_fans_out_to_roles() -> None:
    """SendspinServer.set_visualizer_pitch_enabled refreshes every active role once."""
    server = SendspinServer(
        asyncio.get_running_loop(),
        Identity.generate(),
        "Srv",
        MagicMock(),
        pairing_store=MagicMock(),
    )
    role = MagicMock()
    role.refresh_pitch_setting = MagicMock()
    other = MagicMock(spec=[])  # no refresh_pitch_setting attr → skipped
    client = MagicMock()
    client.active_roles = [role, other]
    server._clients["c1"] = client  # noqa: SLF001

    server.set_visualizer_pitch_enabled(enabled=True)
    assert server.visualizer_pitch_enabled is True
    role.refresh_pitch_setting.assert_called_once()

    role.refresh_pitch_setting.reset_mock()
    server.set_visualizer_pitch_enabled(enabled=True)  # idempotent
    role.refresh_pitch_setting.assert_not_called()


# ---------------------------------------------------------------------------
# Warmup re-arm on beat-schedule clear / availability flip
# ---------------------------------------------------------------------------


async def test_clear_beats_reholds_far_future_frames_after_schedule_landed() -> None:
    """Dropping a landed schedule re-arms warmup so the next schedule is protected."""
    client = _make_beat_client_stub()
    role = VisualizerV1Role(client)
    _connect(role)
    role.on_stream_start()
    role.append_beats([BeatTiming(1_000_000)])  # first landing → holdback lifted
    role.clear_beats()  # schedule dropped while stream continues → re-arm
    client.send_binary.reset_mock()
    role.on_audio_chunk(_audio_chunk(timestamp_us=5_000_000))  # > lead → must be held
    assert _periodic_calls(client) == []
    role._cancel_release_timer()  # noqa: SLF001


async def test_pending_after_unavailable_reholds_far_future_frames() -> None:
    """UNAVAILABLE → PENDING re-arms warmup (beats wanted again on a new source)."""
    client = _make_beat_client_stub()
    role = VisualizerV1Role(client)
    _connect(role)
    role.on_stream_start()
    role.set_beat_availability(BeatAvailability.UNAVAILABLE)  # holdback lifted
    role.set_beat_availability(BeatAvailability.PENDING)  # beats wanted again → re-arm
    client.send_binary.reset_mock()
    role.on_audio_chunk(_audio_chunk(timestamp_us=5_000_000))
    assert _periodic_calls(client) == []
    role._cancel_release_timer()  # noqa: SLF001


async def test_cap_keeps_cursor_near_playhead_so_track_change_beats_deliver() -> None:
    """A far-ahead chunk is capped, so a track-change re-push lands at the cursor, not behind it."""
    client = _make_beat_client_stub()
    role = VisualizerV1Role(client)
    _connect(role)
    role.on_stream_start()
    role.append_beats([BeatTiming(1_000_000)])  # track 1
    role.on_audio_chunk(_audio_chunk(timestamp_us=1_000_000))  # beat 1s emits, cursor ~1s
    # Far-ahead audio (send-ahead) must NOT push the cursor 30s ahead: frame is parked.
    role.on_audio_chunk(_audio_chunk(timestamp_us=30_000_000))
    role.clear_beats()  # track change
    role.append_beats([BeatTiming(2_000_000)])  # track 2 beat near the playhead
    role.on_audio_chunk(_audio_chunk(timestamp_us=2_000_000))
    sent_ts = [call.kwargs["timestamp_us"] for call in client.send_binary.call_args_list]
    assert sent_ts == sorted(sent_ts), f"wire timestamps regressed: {sent_ts}"
    beat_ts = [c.kwargs["timestamp_us"] for c in _beat_calls(client)]
    assert 2_000_000 in beat_ts, "track-change beat must deliver, not drop behind the cursor"
    role._cancel_release_timer()  # noqa: SLF001


async def test_beat_below_cursor_is_dropped_no_regression() -> None:
    """A beat at or below the wire cursor is dropped so the wire stays non-decreasing."""
    client = _make_beat_client_stub()
    role = VisualizerV1Role(client)
    _connect(role)
    role.on_stream_start()
    role.append_beats([BeatTiming(2_500_000)])  # within the cap (cutoff 3s)
    role.on_audio_chunk(_audio_chunk(timestamp_us=2_500_000))  # beat 2.5s emits → cursor ~2.5s
    assert 2_500_000 in [c.kwargs["timestamp_us"] for c in _beat_calls(client)]
    # A stale/replayed beat below the cursor must be dropped, not sent out of order.
    role.append_beats([BeatTiming(2_000_000), BeatTiming(2_800_000)])
    role.on_audio_chunk(_audio_chunk(timestamp_us=2_800_000))
    sent_ts = [call.kwargs["timestamp_us"] for call in client.send_binary.call_args_list]
    assert sent_ts == sorted(sent_ts), f"wire regressed: {sent_ts}"
    beat_ts = [c.kwargs["timestamp_us"] for c in _beat_calls(client)]
    assert 2_000_000 not in beat_ts, "beat below the cursor must be dropped"
    assert 2_800_000 in beat_ts, "beat ahead of the cursor must emit"
    role._cancel_release_timer()  # noqa: SLF001


async def test_track_change_keeps_parked_periodic_frames() -> None:
    """A flow-mode track change keeps parked periodic frames (continuous audio)."""
    client = _make_beat_client_stub()
    role = VisualizerV1Role(client)
    _connect(role)
    role.on_stream_start()
    role.append_beats([BeatTiming(1_000_000)])  # track 1
    role.on_audio_chunk(_audio_chunk(timestamp_us=5_000_000))  # frame ~5.025s parked (cutoff 3s)
    assert role._pending_frames, "frame beyond the cap should be parked"  # noqa: SLF001
    parked_before = len(role._pending_frames)  # noqa: SLF001
    role.clear_beats()  # track change — parked frames must survive
    assert len(role._pending_frames) == parked_before, (  # noqa: SLF001
        "track change must keep parked periodic frames"
    )
    # Playhead advances past the parked frame → it releases.
    client.send_binary.reset_mock()
    client._server.clock.now_us.return_value = 5_000_000  # cutoff 8s  # noqa: SLF001
    role._run_release_scheduler()  # noqa: SLF001
    assert _periodic_calls(client), "parked frame should release after the track change"
    role._cancel_release_timer()  # noqa: SLF001


# ---------------------------------------------------------------------------
# Wire cursor across in-stream stream/start
# ---------------------------------------------------------------------------


def _sent_ts(client: MagicMock) -> list[int]:
    return [call.kwargs["timestamp_us"] for call in client.send_binary.call_args_list]


def _assert_no_regression(client: MagicMock, sent_before: int, periodic_before: int) -> None:
    sent = _sent_ts(client)
    assert len(_periodic_calls(client)) > periodic_before, "the new config must take effect"
    assert min(sent[sent_before:]) >= max(sent[:sent_before]), f"wire regressed: {sent}"


async def test_state_change_mid_stream_keeps_wire_cursor() -> None:
    """A client/state change mid-stream never sends below the highest timestamp already sent."""
    client = _make_client_stub()
    client.visualizer_state = {"types": ["loudness"], "rate_max": 60}
    group_role = VisualizerGroupRole(client.group)
    client.group.group_role.return_value = group_role
    client.group._server.clock.now_us.return_value = 0  # noqa: SLF001
    group_role.append_beat_schedule([BeatTiming(1_000_000), BeatTiming(2_100_000)])
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()
    role.on_audio_chunk(_audio_chunk(2_000_000))
    sent_before = len(client.send_binary.call_args_list)
    periodic_before = len(_periodic_calls(client))

    # Requesting beats rejoins the group, which replays beats already behind the cursor.
    role.on_client_state(_state(types=["loudness", "beat"], rate_max=30))
    for timestamp_us in range(2_025_000, 2_125_000, 25_000):
        role.on_audio_chunk(_audio_chunk(timestamp_us))

    _assert_no_regression(client, sent_before, periodic_before)
    assert 2_100_000 in [c.kwargs["timestamp_us"] for c in _beat_calls(client)]
    role._cancel_release_timer()  # noqa: SLF001


async def test_pitch_toggle_mid_stream_keeps_wire_cursor() -> None:
    """A pitch toggle mid-stream never sends below the highest timestamp already sent."""
    client = _make_client_stub()
    client.visualizer_state = {"types": ["loudness", "pitch", "beat"], "rate_max": 60}
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()
    role.append_beats([BeatTiming(1_000_000)])
    role.on_audio_chunk(_audio_chunk(2_000_000))
    sent_before = len(client.send_binary.call_args_list)
    periodic_before = len(_periodic_calls(client))

    # A re-pushed schedule starts behind the cursor.
    role.append_beats([BeatTiming(1_500_000), BeatTiming(2_040_000)])
    client._server.visualizer_pitch_enabled = False  # noqa: SLF001
    role.refresh_pitch_setting()
    role.on_audio_chunk(_audio_chunk(2_025_000))

    _assert_no_regression(client, sent_before, periodic_before)
    assert 2_040_000 in [c.kwargs["timestamp_us"] for c in _beat_calls(client)]
    role._cancel_release_timer()  # noqa: SLF001


def test_repeated_stream_start_keeps_wire_cursor() -> None:
    """A repeated on_stream_start within an announced stream never lowers the cursor."""
    client = _make_beat_client_stub()
    # The playhead sits below every beat, so only the cursor can drop the stale one.
    client._server.clock.now_us.return_value = 0  # noqa: SLF001
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()
    role.append_beats([BeatTiming(2_000_000)])
    role.on_audio_chunk(_audio_chunk(2_000_000))

    role.on_stream_start()
    role.append_beats([BeatTiming(1_000_000), BeatTiming(2_500_000)])
    role.on_audio_chunk(_audio_chunk(2_500_000))

    beat_ts = [c.kwargs["timestamp_us"] for c in _beat_calls(client)]
    assert beat_ts == [2_000_000, 2_500_000]
    assert _stream_start_count(client) == 2


def test_stream_clear_and_new_stream_reset_wire_cursor() -> None:
    """After stream/clear, and on a new stream, timestamps may go below those already sent."""
    client = _make_beat_client_stub()
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()
    role.append_beats([BeatTiming(2_000_000)])
    role.on_audio_chunk(_audio_chunk(2_000_000))

    role.on_stream_clear()
    role.append_beats([BeatTiming(1_000_000)])
    role.on_audio_chunk(_audio_chunk(1_000_000))
    role.on_stream_end()
    role.on_stream_start()
    role.append_beats([BeatTiming(500_000)])
    role.on_audio_chunk(_audio_chunk(500_000))

    beat_ts = [c.kwargs["timestamp_us"] for c in _beat_calls(client)]
    assert beat_ts == [2_000_000, 1_000_000, 500_000]


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_request_format_partial_preserves_unchanged_fields() -> None:
    """A partial stream/request-format keeps prior values for omitted fields."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    _connect(role)
    role.on_stream_start()

    original_types = list(role._stream_config.types)  # noqa: SLF001
    original_buffer = role.get_buffer_tracker().capacity_bytes
    original_spectrum = role._stream_config.spectrum  # noqa: SLF001

    payload = StreamRequestFormatPayload(visualizer=StreamRequestFormatVisualizer(rate_max=15))
    role.on_stream_request_format(payload)

    assert role._stream_config.rate_max == 15  # noqa: SLF001
    assert list(role._stream_config.types) == original_types  # noqa: SLF001
    assert role.get_buffer_tracker().capacity_bytes == original_buffer
    assert role._stream_config.spectrum == original_spectrum  # noqa: SLF001
