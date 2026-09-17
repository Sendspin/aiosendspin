"""SourceBridge behaviour: latency priming, gap policy, drift correction, formats."""

from __future__ import annotations

import logging
import struct

import pytest

from aiosendspin.audio.format import AudioFormat
from aiosendspin.audio.source_bridge import AsrcSourceBridge, SourceBridge

FMT = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)
STRIDE = 4


def _pattern(frames: int) -> bytes:
    """Non-silent PCM whose content survives a passthrough conversion bit-exact."""
    return (b"\x01\x02\x03\x04" * frames)[: frames * STRIDE]


def _indexed(frames: int, start: int = 0) -> bytes:
    """PCM whose frames encode their index, to track positions through the bridge."""
    return b"".join(struct.pack(">I", start + i) for i in range(frames))


def _silence(frames: int) -> bytes:
    return bytes(frames * STRIDE)


def _bridge(target_ms: int = 100, max_ms: int = 300) -> SourceBridge:
    return SourceBridge(
        input_format=FMT, output_format=FMT, target_latency_ms=target_ms, max_latency_ms=max_ms
    )


def test_max_latency_must_exceed_target() -> None:
    """A max latency at or below target is rejected."""
    with pytest.raises(ValueError, match="max_latency_ms"):
        _bridge(target_ms=100, max_ms=100)


def test_read_returns_silence_until_target_reached() -> None:
    """read() serves silence while the buffer is still priming to the target."""
    bridge = _bridge()
    bridge.feed(_pattern(2400), 0)  # 50ms of the 100ms target
    assert bridge.read(480) == _silence(480)


def test_read_serves_audio_once_target_reached() -> None:
    """read() switches from silence to buffered audio when the target is reached."""
    bridge = _bridge()
    bridge.feed(_pattern(4800), 0)  # exactly the 100ms target
    assert bridge.read(480) == _pattern(480)


def test_prime_trims_startup_surplus_to_target() -> None:
    """Surplus accumulated before the first read is dropped inaudibly at priming."""
    bridge = _bridge(target_ms=100, max_ms=500)
    bridge.feed(_indexed(12000), 0)  # 250ms accumulates before the consumer's first read
    out = bridge.read(480)
    # The oldest 150ms (7200 frames) were trimmed; playback starts at the target.
    assert out[:4] == struct.pack(">I", 7200)
    assert bridge.occupancy_us == 90_000


def test_steady_state_passthrough_is_bit_exact() -> None:
    """Identical input and output formats pass audio through unmodified."""
    bridge = _bridge()
    data = _pattern(4800)
    bridge.feed(data, 0)
    assert bridge.read(4800) == data


def test_forward_gap_becomes_silence() -> None:
    """A capture-timestamp gap is served as silence so latency holds."""
    bridge = _bridge()
    bridge.feed(_pattern(4800), 0)  # exactly the target, primes without a trim
    assert bridge.read(4800) == _pattern(4800)
    # Next chunk arrives 20ms after the expected 100_000us continuation point.
    bridge.feed(_pattern(480), 120_000)
    out = bridge.read(960 + 480)
    assert out[: 960 * STRIDE] == _silence(960)
    assert out[960 * STRIDE :] == _pattern(480)


def test_backward_chunk_is_dropped() -> None:
    """A chunk stamped before already-buffered audio is discarded."""
    bridge = _bridge()
    bridge.feed(_pattern(4800), 0)
    bridge.feed(_pattern(480), 50_000)
    assert bridge.occupancy_us == 100_000


def test_jump_beyond_max_resets_and_reprimes() -> None:
    """A timestamp jump larger than max latency resets the buffer and re-primes."""
    bridge = _bridge()
    bridge.feed(_pattern(4800), 0)
    assert bridge.read(480) == _pattern(480)  # primed
    bridge.feed(_pattern(480), 10_000_000)
    assert bridge.read(480) == _silence(480)


def test_timestamp_reset_discards_resampler_tail() -> None:
    """A timestamp reset clears audio retained by the old resampler."""
    input_format = AudioFormat(sample_rate=44100, bit_depth=16, channels=1)
    output_format = AudioFormat(sample_rate=48000, bit_depth=16, channels=1)
    bridge = SourceBridge(
        input_format=input_format,
        output_format=output_format,
        target_latency_ms=5,
        max_latency_ms=20,
    )
    bridge.feed(struct.pack("<441h", *([32767] * 441)), 0)

    bridge.feed(bytes(441 * 2), 1_000_000)

    assert bridge.read(240) == bytes(240 * 2)


def test_underrun_pads_with_silence() -> None:
    """A drained buffer keeps serving exact-length silence instead of blocking."""
    bridge = _bridge()
    bridge.feed(_pattern(4800), 0)
    bridge.read(4800)
    assert bridge.read(480) == _silence(480)


def test_overflow_beyond_max_drops_oldest() -> None:
    """Occupancy above max latency sheds the oldest audio, keeping the newest."""
    bridge = _bridge()
    bridge.feed(_silence(4800), 0)  # exactly the target, primes without a trim
    assert bridge.read(480) == _silence(480)
    # The source keeps feeding with no reads until the 300ms cap forces a drop.
    bridge.feed(_silence(9600), 100_000)
    bridge.feed(_pattern(4800), 300_000)
    assert bridge.occupancy_us == 300_000
    out = bridge.read(14400)
    # The newest fed audio is intact at the tail of the buffered region.
    assert out[9600 * STRIDE :] == _pattern(4800)


def test_persistent_surplus_is_trimmed_to_target() -> None:
    """Occupancy sitting above target for a full window is corrected back down."""
    bridge = SourceBridge(
        input_format=FMT, output_format=FMT, target_latency_ms=100, max_latency_ms=500
    )
    bridge.feed(_pattern(4800), 0)  # exactly the target, primes without a trim
    assert bridge.read(480) == _pattern(480)
    bridge.feed(_pattern(9600), 100_000)  # 200ms surplus forms after priming
    ts = 300_000
    # Balanced feed/read holds the surplus. The first window's minimum still spans
    # the pre-surplus read, so the correction fires when the second window closes.
    for _ in range(2100):
        bridge.feed(_pattern(480), ts)
        bridge.read(480)
        ts += 10_000
    assert abs(bridge.occupancy_us - 100_000) <= 50_000


def test_subwatermark_surplus_converges_after_persisting() -> None:
    """A surplus below the watermark is still trimmed once it persists across windows."""
    bridge = SourceBridge(
        input_format=FMT, output_format=FMT, target_latency_ms=100, max_latency_ms=500
    )
    bridge.feed(_pattern(4800), 0)  # exactly the target, primes without a trim
    assert bridge.read(480) == _pattern(480)
    bridge.feed(_pattern(1920), 100_000)  # 40ms surplus, below the 50ms watermark
    ts = 140_000
    # Balanced feed/read across more than the persistence window count.
    for _ in range(4300):
        bridge.feed(_pattern(480), ts)
        bridge.read(480)
        ts += 10_000
    assert bridge.occupancy_us - 100_000 <= 15_000


def test_subwatermark_surplus_untouched_before_persistence() -> None:
    """A sub-watermark surplus is left alone until it proves persistent."""
    bridge = SourceBridge(
        input_format=FMT, output_format=FMT, target_latency_ms=100, max_latency_ms=500
    )
    bridge.feed(_pattern(4800), 0)  # exactly the target, primes without a trim
    assert bridge.read(480) == _pattern(480)
    bridge.feed(_pattern(1920), 100_000)  # 40ms surplus, below the 50ms watermark
    ts = 140_000
    for _ in range(2000):  # only two windows close, under the persistence count
        bridge.feed(_pattern(480), ts)
        bridge.read(480)
        ts += 10_000
    assert bridge.occupancy_us >= 120_000


def test_consumer_stall_bulge_is_served_intact() -> None:
    """A transient consumer stall is not trimmed; catch-up reads get the real audio."""
    bridge = _bridge(target_ms=100, max_ms=500)
    bridge.feed(_pattern(4800), 0)  # exactly the target, primes without a trim
    assert bridge.read(480) == _pattern(480)
    ts = 100_000
    # A 200ms consumer stall while the source keeps feeding.
    for _ in range(20):
        bridge.feed(_pattern(480), ts)
        ts += 10_000
    # Catch-up read: the leftover 90ms and the whole bulge, nothing trimmed.
    assert bridge.read(4320 + 9600) == _pattern(4320 + 9600)


def test_rate_conversion_preserves_duration() -> None:
    """Resampling 44100 to 48000 keeps the buffered duration intact."""
    fmt_in = AudioFormat(sample_rate=44100, bit_depth=16, channels=2)
    bridge = SourceBridge(input_format=fmt_in, output_format=FMT, target_latency_ms=2000)
    bridge.feed(b"\x01\x02\x03\x04" * 44100, 0)  # 1s at the input rate
    assert abs(bridge.occupancy_us - 1_000_000) <= 10_000


def test_bit_depth_conversion_to_24bit_output() -> None:
    """A 16-bit input is served as packed 24-bit output frames."""
    fmt_out = AudioFormat(sample_rate=48000, bit_depth=24, channels=2)
    bridge = SourceBridge(input_format=FMT, output_format=fmt_out, target_latency_ms=50)
    bridge.feed(_pattern(4800), 0)
    out = bridge.read(480)
    assert len(out) == 480 * 6
    assert out != bytes(480 * 6)


def test_unnamed_channel_layout_converts_at_the_same_channel_count() -> None:
    """A source without a named channel layout still converts rate and depth."""
    channels = 9
    bridge = SourceBridge(
        input_format=AudioFormat(sample_rate=48000, bit_depth=24, channels=channels),
        output_format=AudioFormat(sample_rate=44100, bit_depth=16, channels=channels),
        target_latency_ms=100,
        max_latency_ms=300,
    )
    bridge.feed(bytes(4800 * 3 * channels), 0)
    bridge.flush()
    assert len(bridge.read(4410)) == 4410 * 2 * channels


def test_remix_needs_a_layout_pyav_can_map() -> None:
    """Remixing a channel count PyAV has no layout for fails when the bridge is built."""
    with pytest.raises(ValueError, match="Cannot remix 9 channels to 2 channels"):
        SourceBridge(
            input_format=AudioFormat(sample_rate=48000, bit_depth=16, channels=9),
            output_format=FMT,
        )
    SourceBridge(
        input_format=AudioFormat(sample_rate=48000, bit_depth=16, channels=12),
        output_format=FMT,
    )


def test_asrc_bridge_preserves_duration() -> None:
    """The ASRC path converts rate without changing the buffered duration."""
    pytest.importorskip("soxr")
    fmt_in = AudioFormat(sample_rate=44100, bit_depth=16, channels=2)
    bridge = AsrcSourceBridge(input_format=fmt_in, output_format=FMT, target_latency_ms=2000)
    bridge.feed(b"\x01\x02\x03\x04" * 44100, 0)
    assert abs(bridge.occupancy_us - 1_000_000) <= 20_000


def test_asrc_ratio_pulls_surplus_toward_target() -> None:
    """The variable-rate loop drains a persistent surplus without drop events."""
    pytest.importorskip("soxr")
    bridge = AsrcSourceBridge(
        input_format=FMT, output_format=FMT, target_latency_ms=100, max_latency_ms=500
    )
    bridge.feed(_pattern(4800), 0)  # exactly the target, primes without a trim
    bridge.read(480)
    bridge.feed(_pattern(9600), 100_000)  # 200ms surplus forms after priming
    start = bridge.occupancy_us
    ts = 300_000
    # 20s of balanced feed/read: at the 0.1% ratio cap this trims ~20ms.
    for _ in range(2000):
        bridge.feed(_pattern(480), ts)
        bridge.read(480)
        ts += 10_000
    assert start - bridge.occupancy_us >= 10_000


def test_asrc_tracks_fast_source_rate() -> None:
    """A percent-level fast source converges to a rate match with occupancy near target."""
    pytest.importorskip("soxr")
    bridge = AsrcSourceBridge(
        input_format=FMT, output_format=FMT, target_latency_ms=100, max_latency_ms=500
    )
    bridge.feed(_pattern(4800), 0)  # exactly the target, primes without a trim
    bridge.read(480)
    ts = 100_000
    # The source supplies 2% more audio than the consumer pulls (fast capture clock);
    # capture timestamps advance at the true pace, like a real fast ADC.
    for _ in range(5500):
        bridge.feed(_pattern(612), ts)
        bridge.read(600)
        ts += 12_500
    # The estimator locked onto the ~2% skew and occupancy is recovering off the
    # 500ms rail toward target (the remaining tail decays at the slow trim rate).
    assert 0.015 <= bridge._rate_estimate <= 0.025  # noqa: SLF001
    assert bridge.occupancy_us <= 300_000


def test_asrc_relocks_quickly_after_skew_flip() -> None:
    """A railed window corroborates its measurement, so a skew flip re-locks fast."""
    pytest.importorskip("soxr")
    bridge = AsrcSourceBridge(
        input_format=FMT, output_format=FMT, target_latency_ms=100, max_latency_ms=500
    )
    bridge.feed(_pattern(4800), 0)  # exactly the target, primes without a trim
    bridge.read(480)
    ts = 100_000
    for _ in range(1700):  # lock onto a 2% fast source first
        bridge.feed(_pattern(612), ts)
        bridge.read(600)
        ts += 12_500
    for _ in range(2100):  # the source flips to 2% slow
        bridge.feed(_pattern(588), ts)
        bridge.read(600)
        ts += 12_500
    assert bridge._rate_estimate <= -0.015  # noqa: SLF001


def test_asrc_rate_estimate_skips_implausible_window() -> None:
    """A consumer-stalled window must not corrupt the measured rate estimate."""
    pytest.importorskip("soxr")
    bridge = AsrcSourceBridge(
        input_format=FMT, output_format=FMT, target_latency_ms=100, max_latency_ms=2000
    )
    bridge.feed(_pattern(4800), 0)  # exactly the target, primes without a trim
    bridge.read(480)
    ts = 100_000
    for _ in range(1100):  # a full window of supply with (almost) no consumption
        bridge.feed(_pattern(480), ts)
        ts += 10_000
    assert bridge._rate_estimate == 0.0  # noqa: SLF001


def test_underrun_logs_once_while_starved(caplog: pytest.LogCaptureFixture) -> None:
    """Only a window's first underrun logs; the rest aggregate into the heartbeat."""
    bridge = _bridge()
    bridge.feed(_pattern(4800), 0)
    bridge.read(4800)  # drains the buffer
    with caplog.at_level(logging.DEBUG, logger="aiosendspin.audio.source_bridge"):
        bridge.read(480)
        bridge.read(480)
    assert sum("Underrun" in r.message for r in caplog.records) == 1


def test_drift_heartbeat_reports_occupancy_over_window(caplog: pytest.LogCaptureFixture) -> None:
    """A periodic heartbeat summarises occupancy against target for evaluation."""
    bridge = _bridge(target_ms=100, max_ms=500)
    bridge.feed(_pattern(9600), 0)
    ts = 200_000
    with caplog.at_level(logging.DEBUG, logger="aiosendspin.audio.source_bridge"):
        for _ in range(1100):  # crosses the 10s correction window
            bridge.feed(_pattern(480), ts)
            bridge.read(480)
            ts += 10_000
    assert any("occupancy over last window" in r.message for r in caplog.records)


def test_asrc_heartbeat_reports_resample_ratio(caplog: pytest.LogCaptureFixture) -> None:
    """The ASRC heartbeat surfaces the ppm the variable-rate loop is applying."""
    pytest.importorskip("soxr")
    bridge = AsrcSourceBridge(
        input_format=FMT, output_format=FMT, target_latency_ms=100, max_latency_ms=500
    )
    bridge.feed(_pattern(9600), 0)
    ts = 200_000
    with caplog.at_level(logging.DEBUG, logger="aiosendspin.audio.source_bridge"):
        for _ in range(1100):
            bridge.feed(_pattern(480), ts)
            bridge.read(480)
            ts += 10_000
    assert any("ppm" in r.message for r in caplog.records if "occupancy" in r.message)


def test_flush_preserves_resampler_tail() -> None:
    """A rate-converting bridge gains occupancy when its PyAV tail is flushed."""
    fmt_in = AudioFormat(sample_rate=44100, bit_depth=16, channels=2)
    bridge = SourceBridge(
        input_format=fmt_in, output_format=FMT, target_latency_ms=5, max_latency_ms=20_000
    )
    bridge.feed(b"\x01\x02\x03\x04" * 44100, 0)
    before = bridge.occupancy_us
    bridge.flush()
    assert bridge.occupancy_us > before


def test_asrc_flush_preserves_soxr_tail() -> None:
    """An ASRC bridge gains occupancy when its PyAV and soxr tails are flushed."""
    pytest.importorskip("soxr")
    fmt_in = AudioFormat(sample_rate=44100, bit_depth=16, channels=2)
    bridge = AsrcSourceBridge(
        input_format=fmt_in, output_format=FMT, target_latency_ms=5, max_latency_ms=20_000
    )
    bridge.feed(b"\x01\x02\x03\x04" * 44100, 0)
    before = bridge.occupancy_us
    bridge.flush()
    assert bridge.occupancy_us > before
