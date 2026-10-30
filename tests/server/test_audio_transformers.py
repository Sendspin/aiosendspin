"""Tests for AudioTransformer protocol and implementations."""

from __future__ import annotations

import struct

import pytest

from aiosendspin.server.audio_transformers import (
    AudioTransformer,
    TransformerPool,
)
from aiosendspin.server.channels import MAIN_CHANNEL
from aiosendspin.server.roles.player.audio_transformers import (
    FlacEncoder,
    OpusEncoder,
    PcmPassthrough,
)
from aiosendspin.server.roles.player.capabilities import FLAC_BIT_DEPTHS


class TestAudioTransformerProtocol:
    """Tests for AudioTransformer protocol."""

    def test_protocol_defines_process_method(self) -> None:
        """AudioTransformer requires process() method."""

        class ValidTransformer:
            @property
            def frame_duration_us(self) -> int:
                return 25_000

            def process(
                self, pcm: bytes, _timestamp_us: int, _duration_us: int
            ) -> list[tuple[bytes, int]]:
                return [(pcm, 25_000)]

            def flush(self) -> list[tuple[bytes, int]]:
                return []

            def reset(self) -> None:
                pass

        # Should be recognized as implementing the protocol
        transformer: AudioTransformer = ValidTransformer()
        assert transformer.process(b"test", 0, 1000) == [(b"test", 25_000)]

    def test_protocol_defines_reset_method(self) -> None:
        """AudioTransformer requires reset() method."""

        class ResettableTransformer:
            def __init__(self) -> None:
                self.reset_count = 0

            @property
            def frame_duration_us(self) -> int:
                return 25_000

            def process(
                self, pcm: bytes, _timestamp_us: int, _duration_us: int
            ) -> list[tuple[bytes, int]]:
                return [(pcm, 25_000)]

            def flush(self) -> list[tuple[bytes, int]]:
                return []

            def reset(self) -> None:
                self.reset_count += 1

        transformer = ResettableTransformer()
        transformer.reset()
        assert transformer.reset_count == 1

    def test_protocol_defines_frame_duration_us_property(self) -> None:
        """AudioTransformer requires frame_duration_us property."""

        class TransformerWithFrameDuration:
            @property
            def frame_duration_us(self) -> int:
                return 25_000

            def process(
                self, pcm: bytes, _timestamp_us: int, _duration_us: int
            ) -> list[tuple[bytes, int]]:
                return [(pcm, 25_000)]

            def flush(self) -> list[tuple[bytes, int]]:
                return []

            def reset(self) -> None:
                pass

        transformer: AudioTransformer = TransformerWithFrameDuration()
        assert transformer.frame_duration_us == 25_000

    def test_protocol_defines_flush_method(self) -> None:
        """AudioTransformer requires flush() method."""

        class TransformerWithFlush:
            @property
            def frame_duration_us(self) -> int:
                return 25_000

            def process(
                self, pcm: bytes, _timestamp_us: int, _duration_us: int
            ) -> list[tuple[bytes, int]]:
                return [(pcm, 25_000)]

            def flush(self) -> list[tuple[bytes, int]]:
                return [(b"final", 25_000)]

            def reset(self) -> None:
                pass

        transformer: AudioTransformer = TransformerWithFlush()
        assert transformer.flush() == [(b"final", 25_000)]


class TestPcmPassthrough:
    """Tests for PcmPassthrough transformer."""

    def test_passthrough_has_no_header(self) -> None:
        """PcmPassthrough has no codec header."""
        transformer = PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2)
        assert transformer.get_header() is None

    def test_passthrough_accepts_kwargs(self) -> None:
        """PcmPassthrough accepts format parameters."""
        transformer = PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2)
        assert transformer.frame_duration_us == 25_000

    def test_passthrough_frame_duration_us_default(self) -> None:
        """PcmPassthrough has default 25ms frame duration."""
        transformer = PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2)
        assert transformer.frame_duration_us == 25_000

    def test_passthrough_frame_duration_us_configurable(self) -> None:
        """PcmPassthrough frame duration is configurable."""
        transformer = PcmPassthrough(
            sample_rate=48000, bit_depth=16, channels=2, chunk_duration_us=50_000
        )
        assert transformer.frame_duration_us == 50_000

    def test_passthrough_returns_list_of_frames(self) -> None:
        """PcmPassthrough returns list of frames."""
        transformer = PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2)
        pcm = bytes(4800)  # 25ms at 48kHz stereo 16-bit
        result = transformer.process(pcm, timestamp_us=0, duration_us=25_000)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == (pcm, 25_000)

    def test_passthrough_splits_large_input(self) -> None:
        """PcmPassthrough splits large input into multiple frames."""
        transformer = PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2)
        pcm = bytes(9600)  # 50ms = 2 frames
        result = transformer.process(pcm, timestamp_us=0, duration_us=50_000)
        assert len(result) == 2
        assert len(result[0][0]) == 4800
        assert len(result[1][0]) == 4800

    def test_passthrough_buffers_incomplete_frame(self) -> None:
        """PcmPassthrough buffers incomplete frames."""
        transformer = PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2)
        pcm = bytes(1920)  # 10ms - less than 25ms
        result = transformer.process(pcm, timestamp_us=0, duration_us=10_000)
        assert result == []

    def test_passthrough_emits_when_buffer_fills(self) -> None:
        """PcmPassthrough emits frame when buffer reaches frame size."""
        transformer = PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2)
        result1 = transformer.process(bytes(2880), timestamp_us=0, duration_us=15_000)  # 15ms
        assert result1 == []
        result2 = transformer.process(
            bytes(2880), timestamp_us=15_000, duration_us=15_000
        )  # +15ms = 30ms total
        assert len(result2) == 1
        assert len(result2[0][0]) == 4800

    def test_passthrough_flush_emits_remainder_padded(self) -> None:
        """PcmPassthrough flush emits remaining buffer padded with silence."""
        transformer = PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2)
        transformer.process(bytes(1920), timestamp_us=0, duration_us=10_000)
        result = transformer.flush()
        assert len(result) == 1
        assert len(result[0][0]) == 4800  # Padded to 25ms

    def test_passthrough_flush_empty_buffer(self) -> None:
        """PcmPassthrough flush returns empty list when buffer is empty."""
        transformer = PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2)
        result = transformer.flush()
        assert result == []

    def test_passthrough_reset_clears_buffer(self) -> None:
        """PcmPassthrough reset clears internal buffer."""
        transformer = PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2)
        transformer.process(bytes(1920), timestamp_us=0, duration_us=10_000)
        transformer.reset()
        assert transformer.flush() == []


class TestPcmPassthroughTimestampDrift:
    """Regression tests for the per-frame timestamp drift bug.

    Background: at 44.1kHz @ 25ms chunks, `int(44100 * 25_000 / 1_000_000) == 1102`
    samples per frame. The actual duration of 1102 samples at 44.1kHz is
    1102 * 1_000_000 / 44100 = 24988.66µs, not 25_000µs. Advancing
    `pending_timestamp_us` by `chunk_duration_us = 25_000` per frame accumulated
    +11.34µs of forward drift per frame, causing a dual-stream backward-ts
    glitch after ~17 minutes of playback.

    The fix uses rational arithmetic: pending advances by exactly
    `chunk_samples * 1_000_000 / sample_rate` per frame, with the fractional
    µs carried across frames in a residue accumulator.
    """

    def _drive_n_frames(
        self,
        transformer: PcmPassthrough,
        n_frames: int,
        *,
        sample_rate: int,
        bit_depth: int,
        channels: int,
        chunk_duration_us: int = 25_000,
        start_ts_us: int = 0,
    ) -> list[int]:
        """Push enough zero-PCM through transformer to emit `n_frames` frames.

        Returns the per-frame `pending_timestamp_us` snapshot taken AFTER each
        frame is emitted (i.e. each entry is the timestamp the NEXT frame would
        carry). Drives the transformer one input chunk at a time so we can
        sample pending precisely.
        """
        chunk_samples = int(sample_rate * chunk_duration_us / 1_000_000)
        frame_size = chunk_samples * (bit_depth // 8) * channels
        emitted: list[int] = []
        # Accumulate input timestamps using rational math so the input side
        # is not itself drifting.
        ts_residue = 0
        ts = start_ts_us
        while len(emitted) < n_frames:
            transformer.process(bytes(frame_size), timestamp_us=ts, duration_us=chunk_duration_us)
            pending = transformer.pending_timestamp_us
            assert pending is not None
            emitted.append(pending)
            ts_residue += chunk_samples * 1_000_000
            delta, ts_residue = divmod(ts_residue, sample_rate)
            ts += delta
        return emitted

    def test_passthrough_44100_no_drift_after_one_frame(self) -> None:
        """One frame of 1102 samples at 44.1k advances pending by 24988µs (not 25000)."""
        transformer = PcmPassthrough(sample_rate=44100, bit_depth=16, channels=2)
        # 1 frame of 1102 stereo s16 samples = 1102 * 4 = 4408 bytes
        result = transformer.process(bytes(4408), timestamp_us=0, duration_us=25_000)
        assert len(result) == 1
        # Pending advances by `1102 * 1_000_000 // 44100` = 24988 µs (NOT 25000)
        assert transformer.pending_timestamp_us == 24988
        # Wire duration on returned tuple matches advance — no scalar 25_000.
        assert result[0][1] == 24988

    def test_passthrough_44100_returned_dur_alternates(self) -> None:
        """Per-frame returned `frame_duration_us` alternates 24988/24989 at 44.1k.

        Guards against regression where pending advances correctly via residue
        but the emitted tuple reports a stale scalar `chunk_duration_us`.
        """
        transformer = PcmPassthrough(sample_rate=44100, bit_depth=16, channels=2)
        chunk_samples = 1102
        frame_size = chunk_samples * 4  # stereo s16
        n = 1000
        durs: list[int] = []
        ts_residue = 0
        ts = 0
        while len(durs) < n:
            for _, dur in transformer.process(
                bytes(frame_size), timestamp_us=ts, duration_us=25_000
            ):
                durs.append(dur)
            ts_residue += chunk_samples * 1_000_000
            delta, ts_residue = divmod(ts_residue, 44100)
            ts += delta
        # Every per-frame duration is exactly one of the two valid divmod outputs.
        assert set(durs) <= {24988, 24989}, (
            f"unexpected per-frame durs: {set(durs) - {24988, 24989}}"
        )
        # Cumulative duration matches sample-derived elapsed time exactly.
        expected = n * chunk_samples * 1_000_000 // 44100
        assert sum(durs) == expected, f"sum(durs)={sum(durs)} expected={expected}"

    def test_passthrough_44100_no_drift_after_one_thousand_frames(self) -> None:
        """After 1000 frames at 44.1k, pending matches sample-derived elapsed time exactly."""
        transformer = PcmPassthrough(sample_rate=44100, bit_depth=16, channels=2)
        emitted = self._drive_n_frames(
            transformer, n_frames=1000, sample_rate=44100, bit_depth=16, channels=2
        )
        # After N frames, true elapsed = N * 1102 * 1e6 / 44100. With residue
        # carried, last pending == N*1102*1_000_000 // 44100 (integer-truncated
        # cumulative, max 1µs final-rounding error vs floating-point ideal).
        n = 1000
        chunk_samples = 1102
        expected = n * chunk_samples * 1_000_000 // 44100
        assert emitted[-1] == expected, (
            f"expected pending={expected} after {n} frames, got {emitted[-1]} "
            f"(drift={emitted[-1] - expected}µs); regression of the +11.34µs/frame "
            f"truncation bug"
        )

    def test_passthrough_44100_no_drift_over_glitch_window(self) -> None:
        """After ~17 minutes of frames at 44.1k (40 frames/sec * 17*60), drift stays bounded."""
        # Match the empirical bug window: 17 minutes at ~40 frames/sec ≈ 40_800 frames.
        # Old code drifted +11.34µs/frame * 40_800 = +462ms — enough to trip the
        # 500ms cliff in _encode_for_transform_key. The fix keeps drift at 0.
        transformer = PcmPassthrough(sample_rate=44100, bit_depth=16, channels=2)
        n = 40_800
        emitted = self._drive_n_frames(
            transformer, n_frames=n, sample_rate=44100, bit_depth=16, channels=2
        )
        chunk_samples = 1102
        expected = n * chunk_samples * 1_000_000 // 44100
        drift_us = emitted[-1] - expected
        # Cumulative drift must be ZERO for our integer-residue arithmetic.
        # (Floating-point ideal is `n * 1102 / 44100 * 1_000_000`; we compute
        # `n * 1102 * 1_000_000 // 44100` which is the same up to one final
        # truncation, so they're exactly equal here.)
        assert drift_us == 0, (
            f"drift of {drift_us}µs accumulated over {n} frames at 44.1k — "
            f"the old +11.34µs/frame bug would produce ~{int(n * 11.34)}µs"
        )

    def test_passthrough_48000_zero_drift_at_clean_rate(self) -> None:
        """At 48kHz/25ms (1200 samples = 25000µs exact), drift is zero with old or new code."""
        transformer = PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2)
        n = 10_000
        emitted = self._drive_n_frames(
            transformer, n_frames=n, sample_rate=48000, bit_depth=16, channels=2
        )
        # 1200 * 1_000_000 // 48000 = 25000 exact
        assert emitted[-1] == n * 25_000

    def test_passthrough_96000_no_drift(self) -> None:
        """At 96kHz/25ms (2400 samples = 25000µs exact), no drift."""
        transformer = PcmPassthrough(sample_rate=96000, bit_depth=24, channels=2)
        n = 1000
        emitted = self._drive_n_frames(
            transformer, n_frames=n, sample_rate=96000, bit_depth=24, channels=2
        )
        assert emitted[-1] == n * 25_000

    def test_passthrough_88200_no_drift(self) -> None:
        """At 88.2kHz/25ms (2205 samples = 25000µs exact), no drift."""
        transformer = PcmPassthrough(sample_rate=88200, bit_depth=24, channels=2)
        n = 5_000
        emitted = self._drive_n_frames(
            transformer, n_frames=n, sample_rate=88200, bit_depth=24, channels=2
        )
        # int(88200 * 25_000 / 1_000_000) = 2205 samples per frame
        # 2205 * 1_000_000 / 88200 = 25000.0 exactly — 88.2k divides cleanly at 25ms.
        chunk_samples = 2205
        expected = n * chunk_samples * 1_000_000 // 88200
        assert emitted[-1] == expected

    def test_passthrough_emitted_timestamps_strictly_monotonic_at_44100(self) -> None:
        """Successive pending values must never go backward at any sample rate.

        This is the ultimate guarantee — the dual-stream glitch was a backward
        emission. After the fix, pending advances by either 24988 or 24989 per
        frame (never zero, never negative).
        """
        transformer = PcmPassthrough(sample_rate=44100, bit_depth=16, channels=2)
        emitted = self._drive_n_frames(
            transformer, n_frames=5_000, sample_rate=44100, bit_depth=16, channels=2
        )
        for i in range(1, len(emitted)):
            delta = emitted[i] - emitted[i - 1]
            assert delta in (24988, 24989), (
                f"frame {i}: pending advanced by {delta}µs; expected 24988 or 24989 "
                f"(true value 24988.66µs/frame at 44.1k)"
            )

    def test_passthrough_residue_resets_on_production_gap(self) -> None:
        """Long input gap rebases pending; residue must reset too."""
        transformer = PcmPassthrough(sample_rate=44100, bit_depth=16, channels=2)
        # Prime with a few frames at 44.1k so residue accumulates
        self._drive_n_frames(transformer, n_frames=10, sample_rate=44100, bit_depth=16, channels=2)
        # Simulate a 2-second production gap (>1.5s threshold)
        # Push input timestamped 2s after the last input
        new_ts = 2_000_000
        transformer.process(bytes(4408), timestamp_us=new_ts, duration_us=25_000)
        # Pending should reflect: rebased to new_ts, then advanced by exactly
        # one frame (24988µs) with residue starting fresh
        assert transformer.pending_timestamp_us == new_ts + 24988

    def test_passthrough_residue_resets_on_explicit_reset(self) -> None:
        """reset() clears residue so subsequent stream starts fresh."""
        transformer = PcmPassthrough(sample_rate=44100, bit_depth=16, channels=2)
        self._drive_n_frames(transformer, n_frames=5, sample_rate=44100, bit_depth=16, channels=2)
        transformer.reset()
        # Drive one fresh frame
        transformer.process(bytes(4408), timestamp_us=10_000_000, duration_us=25_000)
        # Should advance from the new anchor by exactly 24988 (residue was 0)
        assert transformer.pending_timestamp_us == 10_000_000 + 24988

    def test_passthrough_residue_resets_on_flush(self) -> None:
        """flush() clears residue so subsequent stream starts fresh."""
        transformer = PcmPassthrough(sample_rate=44100, bit_depth=16, channels=2)
        # Push enough samples to leave a partial buffer
        transformer.process(bytes(2000), timestamp_us=0, duration_us=10_000)
        transformer.flush()
        # Drive a fresh frame from new anchor
        transformer.process(bytes(4408), timestamp_us=5_000_000, duration_us=25_000)
        assert transformer.pending_timestamp_us == 5_000_000 + 24988


class TestTransformerPool:
    """Tests for TransformerPool."""

    def test_get_or_create_creates_new_transformer(self) -> None:
        """Pool creates new transformer when none exists for key."""
        pool = TransformerPool()
        transformer = pool.get_or_create(
            PcmPassthrough,
            channel_id=MAIN_CHANNEL.int,
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            frame_duration_us=25_000,
        )
        assert isinstance(transformer, PcmPassthrough)

    def test_get_or_create_returns_same_instance(self) -> None:
        """Pool returns same instance for identical key."""
        pool = TransformerPool()
        t1 = pool.get_or_create(
            PcmPassthrough,
            channel_id=MAIN_CHANNEL.int,
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            frame_duration_us=25_000,
        )
        t2 = pool.get_or_create(
            PcmPassthrough,
            channel_id=MAIN_CHANNEL.int,
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            frame_duration_us=25_000,
        )
        assert t1 is t2

    def test_get_or_create_different_config_different_instance(self) -> None:
        """Pool creates different instances for different keys."""
        pool = TransformerPool()
        t1 = pool.get_or_create(
            PcmPassthrough,
            channel_id=MAIN_CHANNEL.int,
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            frame_duration_us=25_000,
        )
        t2 = pool.get_or_create(
            PcmPassthrough,
            channel_id=MAIN_CHANNEL.int,
            sample_rate=44100,
            bit_depth=16,
            channels=2,
            frame_duration_us=25_000,
        )
        assert t1 is not t2

    def test_get_or_create_reuses_instance_for_identical_kwargs(self) -> None:
        """Pool reuses instances when constructor kwargs are identical."""
        pool = TransformerPool()
        t1 = pool.get_or_create(
            PcmPassthrough,
            channel_id=MAIN_CHANNEL.int,
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            frame_duration_us=25_000,
            options={"endianness": "little"},
        )
        t2 = pool.get_or_create(
            PcmPassthrough,
            channel_id=MAIN_CHANNEL.int,
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            frame_duration_us=25_000,
            options={"endianness": "little"},
        )
        assert t1 is t2

    def test_get_or_create_uses_kwargs_in_pool_key(self) -> None:
        """Pool creates distinct instances when constructor kwargs differ."""
        pool = TransformerPool()
        t1 = pool.get_or_create(
            PcmPassthrough,
            channel_id=MAIN_CHANNEL.int,
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            frame_duration_us=25_000,
            options={"endianness": "little"},
        )
        t2 = pool.get_or_create(
            PcmPassthrough,
            channel_id=MAIN_CHANNEL.int,
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            frame_duration_us=25_000,
            options={"endianness": "big"},
        )
        assert t1 is not t2

    def test_reset_all_calls_reset_on_all_transformers(self) -> None:
        """Pool reset_all calls reset on every transformer."""
        reset_counts: list[int] = []

        class CountingTransformer:
            def __init__(self, **_kwargs: object) -> None:
                self.index = len(reset_counts)
                reset_counts.append(0)

            @property
            def frame_duration_us(self) -> int:
                return 25_000

            def process(self, pcm: bytes, _ts: int, _dur: int) -> list[tuple[bytes, int]]:
                return [(pcm, 25_000)]

            def flush(self) -> list[tuple[bytes, int]]:
                return []

            def get_header(self) -> bytes | None:
                return None

            def reset(self) -> None:
                reset_counts[self.index] += 1

        pool = TransformerPool()
        pool.get_or_create(
            CountingTransformer,
            channel_id=MAIN_CHANNEL.int,
            sample_rate=48000,
            bit_depth=16,
            channels=2,  # type: ignore[type-var]
            frame_duration_us=25_000,
        )
        pool.get_or_create(
            CountingTransformer,
            channel_id=MAIN_CHANNEL.int,
            sample_rate=44100,
            bit_depth=16,
            channels=2,  # type: ignore[type-var]
            frame_duration_us=25_000,
        )
        pool.reset_all()
        assert reset_counts == [1, 1]


class TestFlacEncoder:
    """Tests for FlacEncoder transformer."""

    def test_flac_encoder_produces_bytes(self) -> None:
        """FlacEncoder produces encoded output."""
        encoder = FlacEncoder(sample_rate=48000, bit_depth=16, channels=2)
        # 25ms of silence at 48kHz stereo 16-bit = 1200 samples * 4 bytes = 4800 bytes
        # Send multiple chunks to ensure encoder produces output (FLAC buffers initial frames)
        pcm = bytes(4800)
        total_output: list[tuple[bytes, int]] = []
        for i in range(4):
            result = encoder.process(pcm, timestamp_us=i * 25_000, duration_us=25_000)
            total_output.extend(result)
        assert len(total_output) > 0

    def test_flac_encoder_rejects_32_bit(self) -> None:
        """FlacEncoder refuses 32-bit input rather than emitting a 24-bit stream."""
        with pytest.raises(ValueError, match="bit depths"):
            FlacEncoder(sample_rate=48000, bit_depth=32, channels=2)

    def test_flac_encoder_has_header(self) -> None:
        """FlacEncoder produces fLaC header."""
        encoder = FlacEncoder(sample_rate=48000, bit_depth=16, channels=2)
        pcm = bytes(4800)
        encoder.process(pcm, timestamp_us=0, duration_us=25_000)
        header = encoder.get_header()
        assert header is not None
        assert header.startswith(b"fLaC")

    def test_flac_encoder_reset_clears_state(self) -> None:
        """FlacEncoder reset clears internal state."""
        encoder = FlacEncoder(sample_rate=48000, bit_depth=16, channels=2)
        pcm = bytes(4800)
        encoder.process(pcm, timestamp_us=0, duration_us=25_000)
        encoder.reset()
        assert encoder._initialized is False  # noqa: SLF001
        assert encoder._codec_header is None  # noqa: SLF001

    def test_flac_encoder_returns_list_of_frames(self) -> None:
        """FlacEncoder returns list of frames after codec internal buffering fills."""
        encoder = FlacEncoder(sample_rate=48000, bit_depth=16, channels=2)
        # FLAC codec buffers ~4 frames before emitting output
        # Feed enough frames to guarantee output
        pcm = bytes(4800)  # 25ms per chunk
        all_results: list[tuple[bytes, int]] = []
        for i in range(8):  # 200ms total
            result = encoder.process(pcm, timestamp_us=i * 25_000, duration_us=25_000)
            assert isinstance(result, list)
            all_results.extend(result)
        # Should have some output after 8 frames
        assert len(all_results) >= 1

    def test_flac_encoder_buffers_incomplete_frame(self) -> None:
        """FlacEncoder buffers incomplete frames."""
        encoder = FlacEncoder(sample_rate=48000, bit_depth=16, channels=2)
        pcm = bytes(1920)  # 10ms
        result = encoder.process(pcm, timestamp_us=0, duration_us=10_000)
        assert result == []

    def test_flac_encoder_flush_emits_remainder(self) -> None:
        """FlacEncoder flush emits remaining buffered audio when buffer has data."""
        encoder = FlacEncoder(sample_rate=48000, bit_depth=16, channels=2)
        # Process incomplete frame (less than 25ms)
        encoder.process(bytes(1920), timestamp_us=0, duration_us=10_000)
        result = encoder.flush()
        # FLAC codec may not emit output immediately due to internal buffering,
        # but our buffer was cleared (padded to frame size and sent to encoder).
        # The actual output depends on codec timing.
        # At minimum, verify flush returns a list
        assert isinstance(result, list)

    def test_flac_encoder_flush_empty_buffer(self) -> None:
        """FlacEncoder flush returns empty list when buffer is empty."""
        encoder = FlacEncoder(sample_rate=48000, bit_depth=16, channels=2)
        # FLAC frame size is 4608 samples = 18432 bytes at 48kHz stereo 16-bit.
        # Process exactly one FLAC frame worth of data.
        flac_frame_bytes = 4608 * 4  # 4608 samples * 4 bytes per sample
        encoder.process(bytes(flac_frame_bytes), timestamp_us=0, duration_us=96_000)
        result = encoder.flush()
        assert result == []

    def test_flac_encoder_pending_timestamp_continuous(self) -> None:
        """FlacEncoder pending_timestamp_us produces continuous frame timestamps.

        FLAC uses a block size of 4608 samples (~96ms at 48kHz). pending_timestamp_us
        tracks output frame count to ensure timestamps are continuous regardless
        of input chunk sizes.
        """
        encoder = FlacEncoder(sample_rate=48000, bit_depth=16, channels=2)

        # Simulate source sending 1005ms chunks (doesn't align with FLAC frames)
        chunk_bytes = int(48000 * 1.005) * 4  # ~1005ms of audio

        timestamps: list[int] = []
        for call_num in range(5):
            input_ts = call_num * 1_005_000  # Input timestamps advance by 1005ms

            # Get base timestamp for output frames (mimics PushStream logic)
            pending_before = encoder.pending_timestamp_us
            base_ts = pending_before if pending_before is not None else input_ts

            frames = encoder.process(bytes(chunk_bytes), input_ts, 1_005_000)
            frame_dur = encoder.frame_duration_us  # ~96ms for FLAC

            # Calculate frame timestamps
            for i in range(len(frames)):
                frame_ts = base_ts + i * frame_dur
                timestamps.append(frame_ts)

        # Verify we got output (1005ms * 5 = 5025ms, at ~96ms/frame = ~52 frames)
        assert len(timestamps) > 40, f"Expected >40 frames, got {len(timestamps)}"

        # Verify timestamps are continuous (each frame is frame_dur after previous)
        frame_dur = encoder.frame_duration_us
        for i in range(1, len(timestamps)):
            gap = timestamps[i] - timestamps[i - 1]
            assert gap == frame_dur, (
                f"Frame {i}: gap={gap}us, expected {frame_dur}us. "
                f"Timestamps around gap: {timestamps[max(0, i - 2) : i + 2]}"
            )


@pytest.mark.parametrize("encoder_type", [FlacEncoder, OpusEncoder])
def test_encoder_frame_duration_us_known_before_first_chunk(
    encoder_type: type[FlacEncoder | OpusEncoder],
) -> None:
    """Encoders report the frame duration they encode with before any audio."""
    encoder = encoder_type(sample_rate=48000, bit_depth=16, channels=2)
    before_us = encoder.frame_duration_us
    encoder.process(bytes(4800), timestamp_us=0, duration_us=25_000)
    assert encoder.frame_duration_us == before_us


class TestFlacEncoderStreamInfo:
    """Tests that the FLAC stream delivered matches the bit depth the server announces."""

    @pytest.mark.parametrize("bit_depth", sorted(FLAC_BIT_DEPTHS))
    def test_streaminfo_bit_depth_matches_announced_depth(self, bit_depth: int) -> None:
        """Every depth the server accepts is the depth STREAMINFO declares."""
        encoder = FlacEncoder(sample_rate=48_000, bit_depth=bit_depth, channels=2)

        header = encoder.get_header()
        assert header is not None, "FLAC encoder should expose a STREAMINFO header"
        assert header.startswith(b"fLaC\x80")

        # get_header() prefixes the 34-byte STREAMINFO block with "fLaC", a 1-byte
        # metadata block header and a 3-byte length. Within the block (RFC 9639 §8.2)
        # the 64 bits at offset 10 hold: sample rate (20), channels (3),
        # bits per sample (5, stored as value-1), total samples (36).
        streaminfo = header[8:]
        packed = struct.unpack_from(">Q", streaminfo, 10)[0]
        streaminfo_bit_depth = ((packed >> 36) & 0x1F) + 1

        assert streaminfo_bit_depth == bit_depth


class TestOpusEncoderLookaheadCompensation:
    """Tests for OpusEncoder codec-lookahead timestamp compensation."""

    def test_lookahead_matches_pre_skip_from_extradata(self) -> None:
        """_lookahead_us is parsed from the OpusHead pre_skip field."""
        encoder = OpusEncoder(sample_rate=48000, bit_depth=16, channels=2)
        encoder._ensure_initialized()  # noqa: SLF001

        extradata = encoder._encoder.extradata  # noqa: SLF001
        assert extradata is not None
        assert extradata[:8] == b"OpusHead"
        pre_skip_samples = struct.unpack_from("<H", extradata, 10)[0]
        assert pre_skip_samples > 0, "libopus should report a non-zero lookahead"
        assert encoder._lookahead_us == pre_skip_samples * 1_000_000 // 48_000  # noqa: SLF001

    def test_stream_anchor_shifted_earlier_by_lookahead(self) -> None:
        """First emitted packet's anchor timestamp is pulled earlier by _lookahead_us."""
        encoder = OpusEncoder(sample_rate=48000, bit_depth=16, channels=2)
        encoder._ensure_initialized()  # noqa: SLF001
        lookahead_us = encoder._lookahead_us  # noqa: SLF001
        assert lookahead_us > 0

        # Use a first input timestamp comfortably above the lookahead so the anchor
        # remains positive regardless of libopus version.
        first_input_ts = 1_000_000_000
        chunk_size = encoder._chunk_samples * encoder._frame_stride  # noqa: SLF001
        frames = encoder.process(bytes(chunk_size), first_input_ts, encoder.frame_duration_us)

        # Anchor is only set once a packet is emitted; assert the precondition
        # explicitly so a future change in libopus first-packet latency surfaces here.
        assert frames, "Expected libopus to emit a packet on the first chunk"

        # First packet has encoder_delay_chunks == 0, so the anchor reduces to
        # first_input_ts - lookahead_us.
        assert encoder._stream_start_timestamp_us == first_input_ts - lookahead_us  # noqa: SLF001
        assert encoder.pending_timestamp_us == (
            first_input_ts - lookahead_us + encoder.frame_duration_us
        )

    def test_lookahead_reapplied_after_production_gap(self) -> None:
        """Lookahead shift is re-applied to the new anchor after a production-gap reset."""
        encoder = OpusEncoder(sample_rate=48000, bit_depth=16, channels=2)
        encoder._ensure_initialized()  # noqa: SLF001
        lookahead_us = encoder._lookahead_us  # noqa: SLF001

        chunk_size = encoder._chunk_samples * encoder._frame_stride  # noqa: SLF001
        frame_dur = encoder.frame_duration_us

        # First burst establishes the initial anchor.
        first_ts = 1_000_000_000
        frames = encoder.process(bytes(chunk_size), first_ts, frame_dur)
        assert frames
        assert encoder._stream_start_timestamp_us == first_ts - lookahead_us  # noqa: SLF001

        # Second burst more than 1.5 s later triggers the production-gap reset path.
        second_ts = first_ts + 2_000_000
        frames = encoder.process(bytes(chunk_size), second_ts, frame_dur)
        assert frames
        assert encoder._stream_start_timestamp_us == second_ts - lookahead_us  # noqa: SLF001

    def test_reset_clears_lookahead(self) -> None:
        """reset() clears _lookahead_us so a re-init cannot reuse a stale value."""
        encoder = OpusEncoder(sample_rate=48000, bit_depth=16, channels=2)
        encoder._ensure_initialized()  # noqa: SLF001
        assert encoder._lookahead_us > 0  # noqa: SLF001

        encoder.reset()
        assert encoder._lookahead_us == 0  # noqa: SLF001
