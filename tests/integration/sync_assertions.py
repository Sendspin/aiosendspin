"""Shared audio sync assertions for integration tests."""

from __future__ import annotations

import base64
import io
import math
from array import array
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from aiosendspin.models.core import StreamClearMessage, StreamEndMessage, StreamStartMessage
from aiosendspin.models.player import PLAYER_AUDIO_HEADER_SIZE, unpack_player_audio_header
from aiosendspin.models.types import AudioCodec


@dataclass(slots=True)
class DecodedSegment:
    """Decoded audio segment in normalized PCM form."""

    sample_rate: int
    channels: int
    start_timestamp_us: int
    pcm_s16le: bytes


@dataclass(slots=True)
class EncodedSegment:
    """Encoded segment boundaries and packets between stream control events."""

    codec: AudioCodec
    sample_rate: int
    channels: int
    start_timestamp_us: int
    codec_header_b64: str | None
    packets: list[EncodedPacket]


@dataclass(slots=True)
class EncodedPacket:
    """Encoded packet payload with its transport timestamp."""

    timestamp_us: int
    payload: bytes


def chirp(t: float, *, f0: float, k: float) -> float:
    """Frequency-swept sine with time-varying instantaneous frequency."""
    return math.sin(2.0 * math.pi * (f0 * t + 0.5 * k * t * t))


def signal_left(t: float) -> float:
    """Deterministic continuous-time test signal (left channel)."""
    return 0.55 * chirp(t, f0=233.0, k=1137.0) + 0.35 * chirp(t, f0=911.0, k=271.0)


def pcm_s16le_stereo_for_range(
    start_timestamp_us: int,
    *,
    sample_rate: int,
    frame_count: int,
) -> bytes:
    """Generate deterministic stereo PCM for a given absolute time range."""
    out = array("h")
    out_extend = out.extend

    for i in range(frame_count):
        t = (start_timestamp_us + int(i * 1_000_000 / sample_rate)) / 1_000_000.0
        left = max(-1.0, min(1.0, signal_left(t)))
        right = max(-1.0, min(1.0, signal_left(t + 0.0013)))
        out_extend((int(left * 32767.0), int(right * 32767.0)))

    return out.tobytes()


def extract_left_channel_s16le(pcm_s16le: bytes, channels: int) -> list[int]:
    """Extract left channel samples from packed s16le PCM bytes."""
    samples = array("h")
    samples.frombytes(pcm_s16le)
    return list(samples[0::channels])


def best_lag_samples(
    received: list[int],
    expected: list[int],
    *,
    max_lag_samples: int,
) -> tuple[int, float]:
    """Return lag (samples) with the best normalized correlation."""
    if not received or not expected:
        raise ValueError("signals must be non-empty")

    import numpy as np  # noqa: PLC0415

    n = min(len(received), len(expected))
    rec = np.array(received[:n], dtype=np.float64)
    exp_arr = np.array(expected[:n], dtype=np.float64)

    # FFT-based cross-correlation: O(n log n) instead of O(n * max_lag)
    size = 2 * n - 1
    fft_size = 1 << (size - 1).bit_length()
    corr = np.fft.irfft(
        np.conj(np.fft.rfft(rec, fft_size)) * np.fft.rfft(exp_arr, fft_size),
        fft_size,
    )

    # Cumulative sum of squares for per-overlap normalization (matches original per-lag norm)
    rec_css = np.empty(n + 1, dtype=np.float64)
    rec_css[0] = 0.0
    np.cumsum(rec**2, out=rec_css[1:])
    exp_css = np.empty(n + 1, dtype=np.float64)
    exp_css[0] = 0.0
    np.cumsum(exp_arr**2, out=exp_css[1:])

    # Positive lags k = 0..min(n-1, max_lag_samples): rec[0..n-k-1] vs exp[k..n-1]
    pk = np.arange(0, min(n, max_lag_samples + 1), dtype=np.intp)
    p_denom = np.sqrt(rec_css[n - pk] * (exp_css[n] - exp_css[pk]))
    p_valid = p_denom > 0
    p_score = np.where(p_valid, corr[pk] / np.where(p_valid, p_denom, 1.0), -2.0)

    # Negative lags: k=1..min(n-1, max_lag_samples), lag=-k: rec[k..n-1] vs exp[0..n-k-1]
    nk = np.arange(1, min(n, max_lag_samples + 1), dtype=np.intp)
    n_denom = np.sqrt((rec_css[n] - rec_css[nk]) * exp_css[n - nk])
    n_valid = n_denom > 0
    n_score = np.where(n_valid, corr[fft_size - nk] / np.where(n_valid, n_denom, 1.0), -2.0)

    p_best_i = int(np.argmax(p_score))
    if n_score.size == 0:
        return int(pk[p_best_i]), float(p_score[p_best_i])
    n_best_i = int(np.argmax(n_score))
    if float(p_score[p_best_i]) >= float(n_score[n_best_i]):
        return int(pk[p_best_i]), float(p_score[p_best_i])
    return -int(nk[n_best_i]), float(n_score[n_best_i])


def _decode_frames_to_pcm_s16le(
    frames: list[Any],
    *,
    sample_rate: int,
    channels: int,
) -> bytes:
    import av  # noqa: PLC0415

    layout = "stereo" if channels == 2 else "mono"
    resampler = av.AudioResampler(format="s16", layout=layout, rate=sample_rate)
    out = bytearray()
    for frame in frames:
        for out_frame in resampler.resample(frame):
            expected = out_frame.samples * channels * 2
            out.extend(bytes(out_frame.planes[0])[:expected])
    return bytes(out)


def encoded_segments_from_events(events: Sequence[Any]) -> list[EncodedSegment]:
    """Extract encoded segments from event streams."""
    segments: list[EncodedSegment] = []
    current_start_msg: StreamStartMessage | None = None
    current_packets: list[EncodedPacket] = []

    def _flush() -> None:
        nonlocal current_start_msg, current_packets
        if current_start_msg is None or not current_packets:
            current_start_msg = None
            current_packets = []
            return

        player = current_start_msg.payload.player
        segments.append(
            EncodedSegment(
                codec=player.codec,
                sample_rate=int(player.sample_rate),
                channels=int(player.channels),
                start_timestamp_us=current_packets[0].timestamp_us,
                codec_header_b64=player.codec_header,
                packets=current_packets,
            )
        )
        current_start_msg = None
        current_packets = []

    for ev in events:
        kind = ev.kind
        payload = ev.payload
        if kind == "json":
            if isinstance(payload, StreamStartMessage):
                _flush()
                current_start_msg = payload
                continue
            if isinstance(payload, (StreamClearMessage, StreamEndMessage)):
                _flush()
                continue
            continue

        if kind != "bin":
            continue

        assert isinstance(payload, (bytes, bytearray))
        header = unpack_player_audio_header(bytes(payload))
        if current_start_msg is None:
            continue
        current_packets.append(
            EncodedPacket(
                timestamp_us=header.timestamp_us,
                payload=bytes(payload)[PLAYER_AUDIO_HEADER_SIZE:],
            )
        )

    _flush()
    return segments


def decode_segment(seg: EncodedSegment) -> DecodedSegment:
    """Decode encoded segment into PCM s16le."""
    decoded_packets: list[tuple[int, bytes]] = []

    if seg.codec == AudioCodec.PCM:
        decoded_packets = [(packet.timestamp_us, packet.payload) for packet in seg.packets]
    elif seg.codec == AudioCodec.FLAC:
        header = base64.b64decode(seg.codec_header_b64) if seg.codec_header_b64 else b""
        for packet in seg.packets:
            decoded_packets.append(
                (
                    packet.timestamp_us,
                    _decode_flac_packet(
                        packet.payload,
                        codec_header=header,
                        sample_rate=seg.sample_rate,
                        channels=seg.channels,
                    ),
                )
            )

    start_timestamp_us, pcm = _rebuild_timeline_pcm(
        decoded_packets,
        sample_rate=seg.sample_rate,
        channels=seg.channels,
        fallback_start_us=seg.start_timestamp_us,
    )
    return DecodedSegment(
        sample_rate=seg.sample_rate,
        channels=seg.channels,
        start_timestamp_us=start_timestamp_us,
        pcm_s16le=pcm,
    )


def _decode_flac_packet(
    packet_payload: bytes,
    *,
    codec_header: bytes,
    sample_rate: int,
    channels: int,
) -> bytes:
    """Decode one FLAC packet payload into s16le PCM."""
    import av  # noqa: PLC0415

    bitstream = codec_header + packet_payload
    container = av.open(io.BytesIO(bitstream), format="flac")
    stream = container.streams.audio[0]
    decoded_frames: list[Any] = []
    for packet in container.demux(stream):
        decoded_frames.extend(packet.decode())

    if not decoded_frames:
        return b""
    return _decode_frames_to_pcm_s16le(
        decoded_frames,
        sample_rate=sample_rate,
        channels=channels,
    )


def _rebuild_timeline_pcm(
    packets: list[tuple[int, bytes]],
    *,
    sample_rate: int,
    channels: int,
    fallback_start_us: int,
) -> tuple[int, bytes]:
    """Rebuild PCM timeline from packet timestamps, preserving gaps/overlaps."""
    non_empty = [(ts, pcm) for ts, pcm in packets if pcm]
    if not non_empty:
        return fallback_start_us, b""

    frame_bytes = channels * 2
    first_ts = non_empty[0][0]
    out = bytearray()
    out_frames = 0

    for ts, packet_pcm in non_empty:
        pkt_frames = len(packet_pcm) // frame_bytes
        if pkt_frames <= 0:
            continue

        trimmed_pcm = packet_pcm
        start_frame = round((ts - first_ts) * sample_rate / 1_000_000)
        if start_frame > out_frames:
            gap_frames = start_frame - out_frames
            out.extend(bytes(gap_frames * frame_bytes))
            out_frames = start_frame
        elif start_frame < out_frames:
            overlap_frames = out_frames - start_frame
            if overlap_frames >= pkt_frames:
                continue
            trim = overlap_frames * frame_bytes
            trimmed_pcm = packet_pcm[trim:]
            pkt_frames -= overlap_frames

        out.extend(trimmed_pcm[: pkt_frames * frame_bytes])
        out_frames += pkt_frames

    return first_ts, bytes(out)


def decode_segments_from_events(events: Sequence[Any]) -> list[DecodedSegment]:
    """Decode all stream segments from an event list."""
    return [decode_segment(seg) for seg in encoded_segments_from_events(events)]


def choose_common_window(
    segments_by_player: dict[str, Sequence[DecodedSegment]],
    *,
    window_duration_us: int,
    warmup_us: int,
) -> int:
    """Pick a window start timestamp that exists in all players' latest segments."""
    starts: list[int] = []
    ends: list[int] = []

    for player_id, segments in segments_by_player.items():
        if not segments:
            raise AssertionError(f"expected at least one segment for {player_id}")
        seg = segments[-1]
        frame_count = len(seg.pcm_s16le) // (2 * seg.channels)
        dur_us = int(frame_count * 1_000_000 / seg.sample_rate)
        starts.append(seg.start_timestamp_us + warmup_us)
        ends.append(seg.start_timestamp_us + dur_us)

    start_us = max(starts)
    end_us = min(ends)
    if end_us - start_us < window_duration_us:
        raise AssertionError("not enough common audio coverage for comparison window")
    return start_us


def choose_common_tail_window(
    segments_by_player: dict[str, Sequence[DecodedSegment]],
    *,
    window_duration_us: int,
    tail_padding_us: int,
) -> int:
    """Pick a comparison window anchored near the shared tail of all players."""
    starts: list[int] = []
    ends: list[int] = []

    for player_id, segments in segments_by_player.items():
        if not segments:
            raise AssertionError(f"expected at least one segment for {player_id}")
        seg = segments[-1]
        frame_count = len(seg.pcm_s16le) // (2 * seg.channels)
        dur_us = int(frame_count * 1_000_000 / seg.sample_rate)
        starts.append(seg.start_timestamp_us)
        ends.append(seg.start_timestamp_us + dur_us)

    shared_end_us = min(ends) - max(0, tail_padding_us)
    shared_start_us = max(starts)
    start_us = shared_end_us - window_duration_us
    if start_us < shared_start_us:
        raise AssertionError("not enough shared tail audio coverage for comparison window")
    return start_us


def samples_for_window(
    seg: DecodedSegment,
    window_start_us: int,
    window_duration_us: int,
) -> list[int]:
    """Extract left-channel samples for a timestamp window."""
    frame_count_total = len(seg.pcm_s16le) // (2 * seg.channels)
    offset_frames = round((window_start_us - seg.start_timestamp_us) * seg.sample_rate / 1_000_000)
    window_frames = round(window_duration_us * seg.sample_rate / 1_000_000)
    offset_frames = max(0, min(offset_frames, frame_count_total))
    end_frames = max(0, min(offset_frames + window_frames, frame_count_total))

    start_byte = offset_frames * seg.channels * 2
    end_byte = end_frames * seg.channels * 2
    return extract_left_channel_s16le(seg.pcm_s16le[start_byte:end_byte], seg.channels)


def expected_left_for_window(
    window_start_us: int,
    *,
    sample_rate: int,
    frame_count: int,
) -> list[int]:
    """Generate expected left-channel samples for a timestamp window."""
    pcm = pcm_s16le_stereo_for_range(
        window_start_us,
        sample_rate=sample_rate,
        frame_count=frame_count,
    )
    return extract_left_channel_s16le(pcm, 2)


def resample_mono_s16(
    samples: list[int],
    *,
    src_rate: int,
    dst_rate: int,
) -> list[int]:
    """Resample mono s16 samples using the same PyAV path as server resampling."""
    if src_rate == dst_rate or not samples:
        return samples

    import av  # noqa: PLC0415

    mono = array("h", samples)
    in_frame = av.AudioFrame(format="s16", layout="mono", samples=len(samples))
    in_frame.sample_rate = src_rate
    in_frame.planes[0].update(mono.tobytes())

    resampler = av.AudioResampler(format="s16", layout="mono", rate=dst_rate)
    out_samples = array("h")

    for out_frame in resampler.resample(in_frame):
        expected = out_frame.samples * 2
        out_samples.frombytes(bytes(out_frame.planes[0])[:expected])

    return out_samples.tolist()


def first_audio_timestamp_after(events: Sequence[Any], *, start_index: int) -> int | None:
    """Return first binary audio timestamp in events[start_index:]."""
    for ev in events[start_index:]:
        if ev.kind != "bin":
            continue
        payload = ev.payload
        assert isinstance(payload, (bytes, bytearray))
        header = unpack_player_audio_header(bytes(payload))
        return header.timestamp_us
    return None


def assert_pcm_chunks_continuous(events: Sequence[Any], *, max_gap_us: int) -> None:
    """Assert PCM chunk timestamps are continuous within tolerance."""
    current_format: StreamStartMessage | None = None
    last_end_us: int | None = None
    for ev in events:
        kind = ev.kind
        payload = ev.payload
        if kind == "json":
            if isinstance(payload, StreamStartMessage):
                current_format = payload
                last_end_us = None
            if isinstance(payload, (StreamClearMessage, StreamEndMessage)):
                current_format = None
                last_end_us = None
            continue

        if kind != "bin":
            continue
        if current_format is None or current_format.payload.player is None:
            continue

        fmt = current_format.payload.player
        if fmt.codec != AudioCodec.PCM:
            continue

        assert isinstance(payload, (bytes, bytearray))
        header = unpack_player_audio_header(bytes(payload))
        data = bytes(payload)[PLAYER_AUDIO_HEADER_SIZE:]
        frame_count = len(data) // (fmt.channels * 2)
        dur_us = int(frame_count * 1_000_000 / fmt.sample_rate)

        if last_end_us is not None:
            gap_us = header.timestamp_us - last_end_us
            assert 0 <= gap_us <= max_gap_us
        last_end_us = header.timestamp_us + dur_us


def assert_audible_sync(
    segments_by_player: dict[str, Sequence[DecodedSegment]],
    *,
    max_skew_us: int = 5_000,
    min_corr: float = 0.85,
    enforce_corr: bool = True,
    window_duration_us: int = 500_000,
    warmup_us: int = 500_000,
    window_anchor: Literal["head", "tail"] = "head",
    tail_padding_us: int = 0,
    compare_to: Literal["signal", "reference"] = "signal",
    reference_player_id: str | None = None,
) -> None:
    """Assert players are synchronized to expected signal and each other."""
    if window_anchor == "tail":
        window_start_us = choose_common_tail_window(
            segments_by_player,
            window_duration_us=window_duration_us,
            tail_padding_us=tail_padding_us,
        )
    else:
        window_start_us = choose_common_window(
            segments_by_player,
            window_duration_us=window_duration_us,
            warmup_us=warmup_us,
        )

    window_samples_by_player: dict[str, list[int]] = {}
    sample_rate_by_player: dict[str, int] = {}
    for player_id, segments in segments_by_player.items():
        seg = segments[-1]
        window_samples_by_player[player_id] = samples_for_window(
            seg, window_start_us, window_duration_us
        )
        sample_rate_by_player[player_id] = seg.sample_rate

    if compare_to == "reference":
        chosen_reference = reference_player_id
        if chosen_reference is None:
            chosen_reference = min(segments_by_player)
        if chosen_reference not in segments_by_player:
            raise AssertionError(f"unknown reference player_id: {chosen_reference}")
        reference_samples = window_samples_by_player[chosen_reference]
        reference_rate = sample_rate_by_player[chosen_reference]

    signed_lag_us_by_player: dict[str, float] = {}
    for player_id, segments in segments_by_player.items():
        seg = segments[-1]
        received = window_samples_by_player[player_id]
        if compare_to == "reference":
            if player_id == chosen_reference:
                signed_lag_us_by_player[player_id] = 0.0
                continue
            expected = resample_mono_s16(
                reference_samples,
                src_rate=reference_rate,
                dst_rate=seg.sample_rate,
            )
        else:
            frame_count = round(window_duration_us * seg.sample_rate / 1_000_000)
            expected = expected_left_for_window(
                window_start_us,
                sample_rate=seg.sample_rate,
                frame_count=frame_count,
            )

        max_lag_samples = int(seg.sample_rate * (max_skew_us / 1_000_000))
        lag_samples, score = best_lag_samples(
            received,
            expected,
            max_lag_samples=max_lag_samples,
        )
        signed_lag_us = lag_samples * 1_000_000 / seg.sample_rate
        lag_us = abs(signed_lag_us)

        assert lag_us <= max_skew_us, f"{player_id} lag {lag_us:.2f}us > {max_skew_us}us"
        if enforce_corr:
            assert score >= min_corr, f"{player_id} correlation {score:.3f} < {min_corr}"
        signed_lag_us_by_player[player_id] = signed_lag_us

    spread = max(signed_lag_us_by_player.values()) - min(signed_lag_us_by_player.values())
    assert spread <= max_skew_us, f"player signed lag spread {spread:.2f}us > {max_skew_us}us"
