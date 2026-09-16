"""Push-based audio streaming API."""

from __future__ import annotations

__all__ = ["MAIN_CHANNEL", "PushStream"]

import asyncio
import contextlib
import logging
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass
from errno import EAGAIN
from functools import lru_cache
from itertools import islice
from typing import TYPE_CHECKING, Literal, NamedTuple, Protocol, cast
from uuid import UUID

from aiosendspin.server.audio import (
    AudioFormat,
    _convert_s24_to_s32,
    _convert_s32_to_s24,
    _get_av,
    _validate_pcm_buffer_length,
)
from aiosendspin.server.audio_transformers import TransformKey, normalize_options
from aiosendspin.server.channels import MAIN_CHANNEL
from aiosendspin.server.roles import AudioChunk
from aiosendspin.server.roles.player.audio_transformers import PcmPassthrough
from aiosendspin.util import create_task

if TYPE_CHECKING:
    import av

    from aiosendspin.clock import Clock
    from aiosendspin.server.audio_transformers import AudioTransformer
    from aiosendspin.server.client import SendspinClient
    from aiosendspin.server.group import SendspinGroup
    from aiosendspin.server.roles import AudioRequirements, Role

_LOGGER = logging.getLogger(__name__)

# TODO: test if still required, since I fixed double stream start messages
# Default initial delay before first audio plays (microseconds)
DEFAULT_INITIAL_DELAY_US = 250_000  # 250ms
# Pre-roll amount for catch-up encoding to absorb codec startup delay.
ENCODER_CATCHUP_WARMUP_US = 120_000
# Source audio encoded between yields while replaying the PCM cache. The cache holds
# everything the producer has buffered ahead (tens of seconds), so encoding it in one
# step would stall the loop and starve the live commit that keeps the timeline moving.
_PCM_SEQUENCE_YIELD_INTERVAL_US = 500_000
# Dithering policy when reducing to 16-bit integer PCM.
_DITHER_METHOD_TRIANGULAR_HP = "triangular_hp"
# Maximum allowed drift between transformer's internal timeline and the expected output
# timestamp before we discard the transformer's timeline. Normal codec buffering delays are
# < 100ms; this threshold catches the accumulated drift from timeline rebasing.
_TRANSFORMER_DRIFT_THRESHOLD_US = 500_000  # 500ms


class _AudioFilterGraph(Protocol):
    """Subset of PyAV filter graph API used by PushStream."""

    def push(self, frame: av.AudioFrame | None) -> None:
        """Push one audio frame into the graph, or None to signal EOF/flush."""
        ...

    def pull(self) -> av.AudioFrame:
        """Pull one audio frame from the graph sink."""
        ...


def _drain_audio_graph(graph: _AudioFilterGraph) -> list[av.AudioFrame]:
    """Pull all currently available audio frames from a configured filter graph."""
    out_frames: list[av.AudioFrame] = []
    while True:
        try:
            out_frames.append(graph.pull())
        except EOFError:
            break
        except OSError as exc:  # pragma: no cover - depends on FFmpeg/PyAV build details
            if exc.errno == EAGAIN:
                break
            raise
    return out_frames


@lru_cache(maxsize=1)
def _supports_soxr_resampler() -> bool:
    """Return True when this runtime FFmpeg build supports libsoxr in aresample."""
    av = _get_av()

    graph = av.filter.Graph()
    try:
        graph.link_nodes(
            graph.add_abuffer(format="s16", sample_rate=48_000, layout="stereo"),
            graph.add("aresample", "resampler=soxr:precision=30"),
            graph.add("abuffersink"),
        ).configure()
    except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover
        _LOGGER.debug("libsoxr not available for aresample; using swr fallback: %s", exc)
        return False
    return True


# Tracks format combos where soxr graph construction failed, to avoid retrying.
_soxr_failed_configs: set[tuple[str, str, int, str, str, int]] = set()


def _assemble_resample_graph(
    *,
    source_av_format: str,
    source_layout: str,
    source_sample_rate: int,
    target_av_format: str,
    target_layout: str,
    target_sample_rate: int,
    aresample_args: str,
) -> _AudioFilterGraph:
    """Build and configure a PyAV graph for one resampling configuration."""
    av = _get_av()

    graph = av.filter.Graph()
    graph.link_nodes(
        graph.add_abuffer(
            format=source_av_format,
            sample_rate=source_sample_rate,
            layout=source_layout,
        ),
        graph.add("aresample", aresample_args),
        graph.add(
            "aformat",
            (
                f"sample_fmts={target_av_format}:"
                f"sample_rates={target_sample_rate}:"
                f"channel_layouts={target_layout}"
            ),
        ),
        graph.add("abuffersink"),
    ).configure()
    return cast("_AudioFilterGraph", graph)


def _build_resample_graph(
    *,
    source_av_format: str,
    source_layout: str,
    source_sample_rate: int,
    target_av_format: str,
    target_layout: str,
    target_sample_rate: int,
    dither_method: str | None = None,
) -> _AudioFilterGraph:
    """Create an audio filter graph for resampling with soxr (or swr fallback)."""
    config_key = (
        source_av_format,
        source_layout,
        source_sample_rate,
        target_av_format,
        target_layout,
        target_sample_rate,
    )
    use_soxr = _supports_soxr_resampler() and config_key not in _soxr_failed_configs

    preferred_resampler = "resampler=soxr:precision=30" if use_soxr else "resampler=swr"
    preferred_aresample = (
        preferred_resampler
        if dither_method is None
        else f"{preferred_resampler}:osf={target_av_format}:dither_method={dither_method}"
    )

    try:
        return _assemble_resample_graph(
            source_av_format=source_av_format,
            source_layout=source_layout,
            source_sample_rate=source_sample_rate,
            target_av_format=target_av_format,
            target_layout=target_layout,
            target_sample_rate=target_sample_rate,
            aresample_args=preferred_aresample,
        )
    except (OSError, RuntimeError, ValueError):
        if not use_soxr:
            raise

    _soxr_failed_configs.add(config_key)
    _LOGGER.warning("Falling back to swr resampler after soxr graph setup failure")
    fallback_aresample = (
        "resampler=swr"
        if dither_method is None
        else f"resampler=swr:osf={target_av_format}:dither_method={dither_method}"
    )
    return _assemble_resample_graph(
        source_av_format=source_av_format,
        source_layout=source_layout,
        source_sample_rate=source_sample_rate,
        target_av_format=target_av_format,
        target_layout=target_layout,
        target_sample_rate=target_sample_rate,
        aresample_args=fallback_aresample,
    )


def _encode_for_transform_key(
    transformer: AudioTransformer | None,
    pcm_data: bytes,
    output_ts: int,
    duration_us: int,
) -> list[tuple[bytes, int, int]]:
    """Encode PCM for a single TransformKey."""
    if transformer is None:
        return [(pcm_data, output_ts, duration_us)]

    frames = transformer.process(pcm_data, output_ts, duration_us)
    if not frames:
        return []

    total_dur = sum(dur for _, dur in frames)
    base_ts = output_ts
    pending = transformer.pending_timestamp_us
    if pending is not None:
        candidate_base = pending - total_dur
        if abs(candidate_base - output_ts) <= _TRANSFORMER_DRIFT_THRESHOLD_US:
            base_ts = candidate_base

    result: list[tuple[bytes, int, int]] = []
    ts = base_ts
    for data, dur in frames:
        result.append((data, ts, dur))
        ts += dur
    return result


class _ResamplerKey(NamedTuple):
    """Key for sharing resamplers: (channel_id, source_format, target PCM params).

    Resamplers convert PCM from source format to target sample rate/channels/bit_depth.
    The codec is irrelevant for resampling, so multiple target formats with different
    codecs but the same PCM parameters can share a resampler.
    """

    channel_id: UUID
    source_format: AudioFormat
    target_sample_rate: int
    target_channels: int
    target_bit_depth: int
    target_sample_type: Literal["int", "float"] = "int"
    dither_method: str | None = None


@dataclass
class _ResamplerState:
    """Shared resampler state keyed by _ResamplerKey."""

    key: _ResamplerKey
    """Resampler key for identification."""
    graph: _AudioFilterGraph | None
    """PyAV audio filter graph used for resampling. None for passthrough."""
    source_av_format: str
    """PyAV format string for source."""
    source_av_layout: str
    """PyAV channel layout for source."""
    source_sample_rate: int
    """Source sample rate used when configuring the filter graph."""
    source_av_frame_stride: int
    """Bytes per frame in PyAV representation for source PCM."""
    target_av_format: str
    """PyAV format string for target (after resampling)."""
    target_layout: str
    """PyAV channel layout for target."""
    target_wire_frame_stride: int
    """Bytes per frame for wire PCM representation."""
    target_av_frame_stride: int
    """Bytes per frame in PyAV representation."""
    needs_s32_to_s24_conversion: bool
    """True when resampler output is s32 but wire PCM is packed s24."""
    pending_timestamp_us: int | None = None
    """Timestamp of the earliest audio sample not yet emitted by this resampler."""
    pending_ts_residue: int = 0
    """Drift-free residue accumulator for `pending_timestamp_us`. Each call advances
    pending by `output_samples * 1_000_000 / target_sample_rate`. For target rates
    where this isn't a whole number (e.g. 44.1kHz), `int(...)` truncation accumulates
    backward drift over time. We track the unconsumed numerator across calls so
    cumulative pending exactly matches sample-derived elapsed time. Reset to 0
    whenever pending is rebased onto a fresh anchor (drift_reset, init)."""
    pending_input_timestamp_us: int | None = None
    """Expected timestamp of the next input frame. Advances by input duration per call
    and is used to detect genuine input-timeline gaps independent of resampler FIR
    latency, which can make the output-side `pending_timestamp_us` lag the input by
    tens of ms even during steady-state operation (notably with soxr precision=30)."""
    pending_input_ts_residue: int = 0
    """Input-side counterpart to `pending_ts_residue`. Reset on init and drift rebase."""
    is_passthrough: bool = False
    """True when source and target formats are identical — skip graph processing."""

    @property
    def target_sample_type(self) -> Literal["int", "float"]:
        """PCM sample type produced by this resampler."""
        return "float" if self.target_av_format == "flt" else "int"


@dataclass(frozen=True)
class _ResampledPCM:
    """Resampler output with timestamp and sample metadata."""

    pcm_data: bytes
    output_start_ts: int
    sample_count: int
    needs_s32_to_s24_conversion: bool
    sample_type: Literal["int", "float"]


def _create_resampler_state(
    key: _ResamplerKey,
    source_format: AudioFormat,
    target_format: AudioFormat,
) -> _ResamplerState:
    """Create a new resampler state. Thread-safe (no shared state)."""
    _source_wire_bytes, source_av_format, source_layout, _source_av_bytes = (
        source_format.resolve_av_format()
    )
    target_wire_bytes, target_av_format, target_layout, target_av_bytes = (
        target_format.resolve_av_format()
    )

    needs_s32_to_s24 = (
        target_format.sample_type == "int"
        and target_format.bit_depth == 24
        and target_av_bytes != target_wire_bytes
    )

    is_passthrough = (
        source_format.sample_rate == target_format.sample_rate
        and source_format.channels == target_format.channels
        and source_format.bit_depth == target_format.bit_depth
        and source_format.sample_type == target_format.sample_type
        and key.dither_method is None
    )

    graph: _AudioFilterGraph | None = None
    if not is_passthrough:
        graph = _build_resample_graph(
            source_av_format=source_av_format,
            source_layout=source_layout,
            source_sample_rate=source_format.sample_rate,
            target_av_format=target_av_format,
            target_layout=target_layout,
            target_sample_rate=target_format.sample_rate,
            dither_method=key.dither_method,
        )

    return _ResamplerState(
        key=key,
        graph=graph,
        source_av_format=source_av_format,
        source_av_layout=source_layout,
        source_sample_rate=source_format.sample_rate,
        source_av_frame_stride=_source_av_bytes * source_format.channels,
        target_av_format=target_av_format,
        target_layout=target_layout,
        target_wire_frame_stride=target_wire_bytes * target_format.channels,
        target_av_frame_stride=target_av_bytes * target_format.channels,
        needs_s32_to_s24_conversion=needs_s32_to_s24,
        is_passthrough=is_passthrough,
    )


def _resample_pcm_standalone(  # noqa: PLR0915
    resampler_state: _ResamplerState,
    source_pcm: bytes,
    source_format: AudioFormat,
    input_timestamp_us: int,
) -> _ResampledPCM:
    """Resample PCM data to the target format.

    Thread-safe per resampler_state instance (each call should use its own state).

    Args:
        resampler_state: The resampler state to use.
        source_pcm: Source PCM bytes.
        source_format: Source audio format.
        input_timestamp_us: Timestamp for the input audio.

    Returns:
        Resampled PCM with timestamp and sample metadata.
    """
    av = _get_av()

    # Handle timestamp tracking.
    # Drift detection compares the *input* timeline — `pending_input_timestamp_us`
    # tracks where the next input sample is expected to land based on previously
    # consumed input durations. Using the output timeline would mis-fire on every
    # call because long-FIR resamplers (e.g. soxr precision=30) emit fewer samples
    # than the rate-conversion ratio until the graph is warmed up.
    if resampler_state.pending_input_timestamp_us is None:
        resampler_state.pending_timestamp_us = input_timestamp_us
        resampler_state.pending_ts_residue = 0
        resampler_state.pending_input_timestamp_us = input_timestamp_us
        resampler_state.pending_input_ts_residue = 0
    else:
        drift_us = abs(resampler_state.pending_input_timestamp_us - input_timestamp_us)
        if drift_us > 20_000:
            # Flush the old graph to release FIR filter tails cleanly.
            # Flushed samples are discarded — they belong to the stale timeline.
            if resampler_state.graph is not None:
                try:
                    resampler_state.graph.push(None)
                    _drain_audio_graph(resampler_state.graph)
                except (EOFError, OSError):
                    pass
                resampler_state.graph = _build_resample_graph(
                    source_av_format=resampler_state.source_av_format,
                    source_layout=resampler_state.source_av_layout,
                    source_sample_rate=resampler_state.source_sample_rate,
                    target_av_format=resampler_state.target_av_format,
                    target_layout=resampler_state.target_layout,
                    target_sample_rate=resampler_state.key.target_sample_rate,
                    dither_method=resampler_state.key.dither_method,
                )
            resampler_state.pending_timestamp_us = input_timestamp_us
            # Reset the residue accumulator: pending has been rebased onto a fresh
            # anchor and any carried fractional µs from the prior segment are stale.
            resampler_state.pending_ts_residue = 0
            resampler_state.pending_input_timestamp_us = input_timestamp_us
            resampler_state.pending_input_ts_residue = 0

    # Both cursors are guaranteed initialized above — narrow for mypy.
    assert resampler_state.pending_timestamp_us is not None
    assert resampler_state.pending_input_timestamp_us is not None

    # Calculate sample count from input
    bytes_per_sample = source_format.bit_depth // 8
    frame_stride = bytes_per_sample * source_format.channels
    if len(source_pcm) % frame_stride != 0:
        msg = (
            f"source PCM buffer length {len(source_pcm)} does not align to "
            f"{frame_stride}-byte frames"
        )
        raise ValueError(msg)
    sample_count = len(source_pcm) // frame_stride
    av_input_pcm = (
        _convert_s24_to_s32(source_pcm)
        if source_format.sample_type == "int"
        and source_format.bit_depth == 24
        and resampler_state.source_av_format == "s32"
        else source_pcm
    )

    if sample_count == 0:
        return _ResampledPCM(
            pcm_data=b"",
            output_start_ts=resampler_state.pending_timestamp_us,
            sample_count=0,
            needs_s32_to_s24_conversion=resampler_state.needs_s32_to_s24_conversion,
            sample_type=resampler_state.target_sample_type,
        )

    # Drift-free input-cursor advance; mirrors `pending_ts_residue` on the output side.
    resampler_state.pending_input_ts_residue += sample_count * 1_000_000
    input_duration_us, resampler_state.pending_input_ts_residue = divmod(
        resampler_state.pending_input_ts_residue,
        source_format.sample_rate,
    )
    resampler_state.pending_input_timestamp_us += input_duration_us

    # Fast path: no conversion needed — return input PCM with timestamp tracking
    if resampler_state.is_passthrough:
        output_start_ts = resampler_state.pending_timestamp_us
        # Drift-free advance: track the fractional µs across calls. Plain
        # `int(samples * 1e6 / sample_rate)` accumulates per-call truncation
        # error for rates that don't divide cleanly into 1e6 (e.g. 44.1k).
        resampler_state.pending_ts_residue += sample_count * 1_000_000
        duration_us, resampler_state.pending_ts_residue = divmod(
            resampler_state.pending_ts_residue,
            resampler_state.key.target_sample_rate,
        )
        resampler_state.pending_timestamp_us += duration_us
        return _ResampledPCM(
            pcm_data=av_input_pcm,
            output_start_ts=output_start_ts,
            sample_count=sample_count,
            needs_s32_to_s24_conversion=resampler_state.needs_s32_to_s24_conversion,
            sample_type=resampler_state.target_sample_type,
        )

    assert resampler_state.graph is not None  # guaranteed: not passthrough → graph was built

    # Create input frame
    _validate_pcm_buffer_length(
        av_input_pcm,
        expected=sample_count * resampler_state.source_av_frame_stride,
        context="resampler input",
    )
    frame = av.AudioFrame(
        format=resampler_state.source_av_format,
        layout=resampler_state.source_av_layout,
        samples=sample_count,
    )
    frame.sample_rate = source_format.sample_rate
    frame.planes[0].update(av_input_pcm)

    # Resample
    resampler_state.graph.push(frame)
    out_frames = _drain_audio_graph(resampler_state.graph)
    out_pcm = bytearray()
    output_sample_count = 0
    for out_frame in out_frames:
        expected = resampler_state.target_av_frame_stride * out_frame.samples
        out_pcm.extend(memoryview(out_frame.planes[0])[:expected])
        output_sample_count += out_frame.samples

    output_start_ts = resampler_state.pending_timestamp_us

    # Update pending timestamp based on output samples — same drift-free
    # accumulator as the passthrough fast path. See note above.
    resampler_state.pending_ts_residue += output_sample_count * 1_000_000
    duration_us, resampler_state.pending_ts_residue = divmod(
        resampler_state.pending_ts_residue,
        resampler_state.key.target_sample_rate,
    )
    resampler_state.pending_timestamp_us += duration_us

    return _ResampledPCM(
        pcm_data=bytes(out_pcm),
        output_start_ts=output_start_ts,
        sample_count=output_sample_count,
        needs_s32_to_s24_conversion=resampler_state.needs_s32_to_s24_conversion,
        sample_type=resampler_state.target_sample_type,
    )


def _flush_resampler(resampler_state: _ResamplerState) -> _ResampledPCM:
    """Push EOF into the resampler graph and capture drained output PCM.

    Returns an empty result when the state has no graph (passthrough or already
    flushed). Advances `pending_timestamp_us` for the drained samples so the
    caller can timestamp them in line with the running output timeline. The
    graph is dropped after flush — the state is single-use after this call.
    """
    if (
        resampler_state.graph is None
        or resampler_state.is_passthrough
        or resampler_state.pending_timestamp_us is None
    ):
        return _ResampledPCM(
            pcm_data=b"",
            output_start_ts=resampler_state.pending_timestamp_us or 0,
            sample_count=0,
            needs_s32_to_s24_conversion=resampler_state.needs_s32_to_s24_conversion,
            sample_type=resampler_state.target_sample_type,
        )

    output_start_ts = resampler_state.pending_timestamp_us
    with contextlib.suppress(EOFError, OSError):
        resampler_state.graph.push(None)
    out_frames = _drain_audio_graph(resampler_state.graph)
    out_pcm = bytearray()
    output_sample_count = 0
    for out_frame in out_frames:
        expected = resampler_state.target_av_frame_stride * out_frame.samples
        out_pcm.extend(memoryview(out_frame.planes[0])[:expected])
        output_sample_count += out_frame.samples

    if output_sample_count > 0:
        resampler_state.pending_ts_residue += output_sample_count * 1_000_000
        duration_us, resampler_state.pending_ts_residue = divmod(
            resampler_state.pending_ts_residue,
            resampler_state.key.target_sample_rate,
        )
        resampler_state.pending_timestamp_us += duration_us

    resampler_state.graph = None
    return _ResampledPCM(
        pcm_data=bytes(out_pcm),
        output_start_ts=output_start_ts,
        sample_count=output_sample_count,
        needs_s32_to_s24_conversion=resampler_state.needs_s32_to_s24_conversion,
        sample_type=resampler_state.target_sample_type,
    )


def _quantize_float_pcm(
    *,
    channel_id: UUID,
    pcm_data: bytes,
    output_ts: int,
    sample_rate: int,
    channels: int,
    target_bit_depth: int,
    resampler_cache: dict[_ResamplerKey, _ResamplerState],
) -> _ResampledPCM:
    """Convert float32 PCM to integer output format using the provided cache."""
    source_format = AudioFormat(
        sample_rate=sample_rate,
        bit_depth=32,
        channels=channels,
        sample_type="float",
    )
    target_format = AudioFormat(
        sample_rate=sample_rate,
        bit_depth=target_bit_depth,
        channels=channels,
        sample_type="int",
    )
    dither_method = _DITHER_METHOD_TRIANGULAR_HP if target_bit_depth == 16 else None

    resampler_key = _ResamplerKey(
        channel_id=channel_id,
        source_format=source_format,
        target_sample_rate=sample_rate,
        target_channels=channels,
        target_bit_depth=target_bit_depth,
        target_sample_type="int",
        dither_method=dither_method,
    )
    state = resampler_cache.get(resampler_key)
    if state is None:
        state = _create_resampler_state(resampler_key, source_format, target_format)
        resampler_cache[resampler_key] = state
    return _resample_pcm_standalone(state, pcm_data, source_format, output_ts)


def _processing_format_for_roles(
    source_format: AudioFormat,
    *,
    target_sample_rate: int,
    target_bit_depth: int,
    target_channels: int,
) -> AudioFormat:
    """Select processing format used during shared per-role resampling."""
    if source_format.sample_type == "float":
        return AudioFormat(
            sample_rate=target_sample_rate,
            bit_depth=32,
            channels=target_channels,
            sample_type="float",
        )
    return AudioFormat(
        sample_rate=target_sample_rate,
        bit_depth=target_bit_depth,
        channels=target_channels,
    )


# Minimum lead time (from now) for sending catch-up audio to late joiners.
# This must be lower than DEFAULT_INITIAL_DELAY_US, otherwise a steady-state low-latency
# stream may have no chunks whose *start* timestamp is >= now + DEFAULT_INITIAL_DELAY_US.
LATE_JOINER_MIN_LEAD_US = 100_000  # 100ms


@dataclass(frozen=True)
class CachedChunk:
    """Cached chunk for late joiner catch-up."""

    timestamp_us: int
    """Start timestamp for this chunk."""
    duration_us: int
    """Duration of this chunk in microseconds."""
    payload: bytes
    """Encoded audio payload bytes (without binary header)."""
    byte_count: int
    """Size of encoded audio data (without header)."""


@dataclass(frozen=True, slots=True)
class CachedPCMChunk:
    """Raw PCM cache entry for catch-up encoding."""

    timestamp_us: int
    duration_us: int
    pcm_data: bytes
    sample_rate: int
    bit_depth: int
    channels: int
    sample_type: Literal["int", "float"] = "int"


class StreamStoppedError(Exception):
    """Raised when trying to commit audio on a stopped stream."""


class PushStream:
    """
    Push-based audio streaming API.

    This class provides a push-based interface for streaming audio to players.
    Audio is prepared via prepare_audio(), then committed and sent via commit_audio().
    Late audio is handled by the connection layer (dropped if past playback time).
    """

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        clock: Clock,
        group: SendspinGroup,
    ) -> None:
        """
        Create a new PushStream.

        Args:
            loop: Event loop for timing and async operations.
            clock: Time source used for timestamping.
            group: Group this stream belongs to.
        """
        self._loop = loop
        self._clock = clock
        self._group = group
        self._is_stopped = False
        # Whether the audio source is realtime (live) vs buffered. Both honor
        # min_buffer_ms at startup; buffered sources may raise that floor to
        # required_lead_time_ms, since only a buffered queue can keep growing
        # after playback begins and absorb the extra lead.
        # Default to buffered; callers opt into live via set_live_source(True).
        self._is_live: bool = False
        # Monotonic lifecycle token used to invalidate in-flight commit work on stop().
        self._stream_generation = 0
        # Pending audio per channel: channel_id -> (pcm_bytes, audio_format)
        self._channel_buffers: dict[UUID, tuple[bytes, AudioFormat]] = {}
        # Per-channel timing: channel_id -> next_chunk_start_us
        self._channel_timing: dict[UUID, int] = {}
        # Unconsumed numerator per channel for drift-free _channel_timing advancement.
        # Reset to 0 whenever _channel_timing[channel_id] is rebased to a new absolute value.
        self._channel_timing_residue: dict[UUID, int] = {}
        # Sample rate of the last _advance_channel_timing call per channel. Residue is
        # modulo this rate, so changes invalidate it and force a residue reset.
        self._channel_timing_rate: dict[UUID, int] = {}
        # Channels that have committed real audio (live or historical), not just synthetic timing.
        self._channels_with_committed_audio: set[UUID] = set()
        # Role-based streaming tracking (for hook-based flow)
        self._started_roles: set[Role] = set()
        # Inline resamplers for role-based audio delivery
        self._resamplers: dict[_ResamplerKey, _ResamplerState] = {}
        # Role-based chunk cache: TransformKey -> list of cached chunks
        self._role_chunk_cache: defaultdict[TransformKey, deque[CachedChunk]] = defaultdict(deque)
        # PCM chunk cache: channel_id.int -> deque of cached PCM chunks
        self._pcm_chunk_cache: dict[int, deque[CachedPCMChunk]] = {}
        # Channels with PCM caching enabled
        self._pcm_cache_enabled_channels: set[int] = {MAIN_CHANNEL.int}
        # Catch-up encoding state per TransformKey
        self._catchup_state: dict[TransformKey, Literal["catching_up", "live"]] = {}
        self._catchup_roles: dict[TransformKey, set[Role]] = {}
        self._catchup_tasks: dict[TransformKey, asyncio.Task[None]] = {}
        # Set whenever a new PCM chunk lands in `_pcm_chunk_cache`. Catch-up tasks
        # await this so they keep up with quick commits.
        self._pcm_cache_signal: asyncio.Event | None = None
        # TransformKey cache by (role_id, channel_id_int) - avoids rebuilding keys each frame
        self._transform_key_cache: dict[tuple[int, int], TransformKey] = {}
        # Last encoded input end timestamp per TransformKey for long-gap reset handling.
        self._transform_last_input_end_us: dict[TransformKey, int] = {}
        # Roles awaiting delayed join; excluded from live delivery until join executes.
        self._pending_join_roles: weakref.WeakSet[Role] = weakref.WeakSet()
        # >0 while commit_audio() is between the _channel_timing advance and _role_chunk_cache
        # update, joiners should be delayed during that
        self._commit_in_flight: int = 0
        # Historical audio buffers: channel_id -> list of (pcm_bytes, audio_format)
        self._historical_buffers: dict[UUID, list[tuple[bytes, AudioFormat]]] = {}
        # Optional start timestamps for historical channels (set on first historical chunk).
        self._historical_start_us: dict[UUID, int] = {}

    def now_us(self) -> int:
        """Return current timestamp from the stream's clock in microseconds."""
        return self._clock.now_us()

    def _signal_pcm_cache_update(self) -> None:
        """Wake any catch-up tasks waiting for new PCM chunks."""
        signal = self._pcm_cache_signal
        if signal is not None:
            signal.set()

    @property
    def is_stopped(self) -> bool:
        """Whether this stream has been stopped."""
        return self._is_stopped

    def _is_generation_active(self, generation: int | None) -> bool:
        """Return True when stream lifecycle still matches the captured generation."""
        if generation is None:
            return not self._is_stopped
        return not self._is_stopped and self._stream_generation == generation

    def _stopped_commit_return_value(self) -> int:
        """Return a stable timestamp for commits interrupted by stop()."""
        if self._channel_timing:
            return min(self._channel_timing.values())
        return self._clock.now_us() + self._min_send_ahead_us()

    @staticmethod
    def _client_in_audio_pipeline(client: SendspinClient) -> bool:
        """Whether a client should participate in transform/delivery processing."""
        # Keep warm-disconnected roles in the pipeline: if audio processing
        # stops during a temporary disconnect, we stop generating/caching role
        # outputs. On reconnect that leaves no backlog for catch-up and creates
        # an audible gap. Warm roles keep push processing running while the
        # transport is down so reconnect can resume from cached output.
        return client.is_connected or client.has_warm_disconnected_roles

    @classmethod
    def _role_in_audio_pipeline(cls, client: SendspinClient, role: Role) -> bool:
        """Whether a specific role should participate in transform/delivery processing."""
        if cls._client_in_audio_pipeline(client):
            return True
        if not client.has_cold_preinitialized_roles:
            return False
        return role.supports_preconnect_audio()

    def _get_audio_roles(self) -> list[tuple[SendspinClient, Role]]:
        """Get all roles that need audio from connected/warm/cold-opted-in clients."""
        result: list[tuple[SendspinClient, Role]] = []
        for client in self._group.clients:
            for role in client.active_roles:
                if role.get_audio_requirements() is None:
                    continue
                if role in self._pending_join_roles:
                    continue
                if not self._role_in_audio_pipeline(client, role):
                    continue
                result.append((client, role))
        return result

    def _max_active_output_delay_us(self) -> int:
        """Return the largest output delay among active audio roles."""
        roles = self._get_audio_roles()
        if not roles:
            return 0
        return max(role.get_output_delay_us() for _, role in roles)

    def _role_send_ahead_us(self, role: Role) -> int:
        """Per-role send-ahead floor.

        Both cases floor at min_buffer + output_delay. Buffered streams extend the
        lead toward required_lead (it adds no latency once the queue grows past
        min_buffer); live streams do not, since a realtime queue cannot grow after
        playback begins and extra lead would only add latency.
        """
        min_buffer_us = role.get_min_buffer_us()
        output_delay_us = role.get_output_delay_us()
        if self._is_live:
            return min_buffer_us + output_delay_us
        return max(min_buffer_us, role.get_required_lead_time_us()) + output_delay_us

    def _min_send_ahead_us(self) -> int:
        """Return the common send-ahead floor across active audio roles."""
        roles = self._get_audio_roles()
        if not roles:
            return DEFAULT_INITIAL_DELAY_US
        return max(self._role_send_ahead_us(role) for _, role in roles)

    @property
    def is_live(self) -> bool:
        """Whether the audio source is treated as realtime/live."""
        return self._is_live

    def set_live_source(self, is_live: bool) -> None:  # noqa: FBT001
        """Configure whether subsequent stream startups treat audio as live or buffered.

        Both wait for min_buffer_ms at startup so the jitter buffer is filled
        before playback. Buffered (default) raises that floor to
        required_lead_time_ms when it is larger, which costs no lasting latency
        because the queue keeps growing after playback begins; a live source
        cannot grow its queue, so the extra lead would be permanent.

        Call before the first commit_audio of a new stream session so the startup
        anchor uses the right floor.
        """
        self._is_live = is_live

    def _get_cached_resampler(self, key: _ResamplerKey) -> _ResamplerState | None:
        """Get existing resampler from cache, or None if not cached."""
        return self._resamplers.get(key)

    def _cache_resampler(self, state: _ResamplerState) -> None:
        """Store a resampler state in the cache."""
        self._resamplers[state.key] = state

    def has_pending_audio(self) -> bool:
        """Return True if there is pending audio to commit."""
        return len(self._channel_buffers) > 0

    def enable_pcm_cache_for_channel(self, channel_id: UUID) -> None:
        """Enable raw PCM caching for a channel."""
        self._pcm_cache_enabled_channels.add(channel_id.int)

    def disable_pcm_cache_for_channel(self, channel_id: UUID) -> None:
        """Disable raw PCM caching for a channel."""
        channel_int = channel_id.int
        self._pcm_cache_enabled_channels.discard(channel_int)
        self._pcm_chunk_cache.pop(channel_int, None)

    def _has_pcm_cache(self, channel_id: UUID) -> bool:
        """Return True if PCM caching is enabled and cached chunks exist."""
        channel_int = channel_id.int
        if channel_int not in self._pcm_cache_enabled_channels:
            return False
        return bool(self._pcm_chunk_cache.get(channel_int))

    def get_pending_audio(self) -> dict[UUID, tuple[bytes, AudioFormat]]:
        """Return the pending audio buffers (for testing/inspection)."""
        return self._channel_buffers

    async def sleep_to_limit_buffer(self, max_buffer_us: int) -> None:
        """Sleep until the furthest-ahead active channel is at most max_buffer_us ahead of now.

        Only considers channels with active audio roles; stale channels from ungrouped
        players are ignored. Falls back to all channel timings when no roles are active
        to prevent a tight loop that starves the event loop.

        :param max_buffer_us: Maximum allowed buffer depth in microseconds.
        """
        if not self._channel_timing:
            return
        active_channels = self._get_active_audio_channels()
        active_timings = [t for ch, t in self._channel_timing.items() if ch in active_channels]
        if not active_timings:
            # Fall back to all channel timings when no audio roles are active yet
            # (e.g. client connected but handshake not complete). Without this,
            # the caller's commit loop never sleeps, starving the event loop and
            # preventing the handshake from completing.
            active_timings = list(self._channel_timing.values())
        if not active_timings:
            return
        max_timing_us = max(active_timings)
        now_us = self._clock.now_us()
        ahead_us = max_timing_us - now_us
        effective_limit_us = max_buffer_us + self._max_active_output_delay_us()
        if ahead_us > effective_limit_us:
            await asyncio.sleep(min((ahead_us - effective_limit_us) / 1_000_000, 1.0))

    def prepare_audio(
        self,
        pcm: bytes,
        audio_format: AudioFormat,
        *,
        channel_id: UUID = MAIN_CHANNEL,
    ) -> None:
        """
        Prepare PCM audio for the next commit.

        This is a synchronous method that stores the PCM data for encoding
        during commit_audio(). Calling twice for the same channel replaces
        the previous data (does not append).

        Args:
            pcm: Raw PCM audio data.
            audio_format: Format of the PCM data.
            channel_id: Channel to prepare audio for (default: MAIN_CHANNEL).
        """
        self._channel_buffers[channel_id] = (pcm, audio_format)

    def prepare_historical_audio(
        self,
        pcm: bytes,
        audio_format: AudioFormat,
        *,
        channel_id: UUID = MAIN_CHANNEL,
        start_time_us: int | None = None,
    ) -> None:
        """Queue historical PCM audio for a new channel.

        Called multiple times to accumulate historical chunks (oldest first).
        On commit_audio(), timestamps are assigned so historical chunks play
        consecutively starting at "now + lead_time", and the live chunk
        (from prepare_audio) continues seamlessly after.

        :param pcm: Raw PCM audio data.
        :param audio_format: Format of the PCM data.
        :param channel_id: Channel to inject historical audio into.
        :param start_time_us: Optional explicit timestamp for the first historical
            chunk on this channel. If omitted, commit_audio() uses now+initial_delay.

        Raises:
            ValueError: If the channel already has active timing.
        """
        if channel_id in self._channels_with_committed_audio:
            raise ValueError(
                f"Cannot add historical audio to channel {channel_id} - "
                "channel already has active timing"
            )
        if start_time_us is not None and channel_id not in self._historical_start_us:
            self._historical_start_us[channel_id] = start_time_us
        self._historical_buffers.setdefault(channel_id, []).append((pcm, audio_format))

    def get_cached_pcm_chunks(self, channel_id: UUID = MAIN_CHANNEL) -> list[CachedPCMChunk]:
        """Retrieve cached PCM chunks for a channel.

        :param channel_id: Channel to retrieve cache from.

        Returns:
            List of CachedPCMChunk objects in chronological order.
        """
        channel_int = channel_id.int
        if channel_int not in self._pcm_chunk_cache:
            return []
        return list(self._pcm_chunk_cache[channel_int])

    def get_late_join_target_timestamp_us(
        self,
        *,
        role: Role | None = None,
        channel_id: UUID | None = None,
        align_to_channel_tail: bool = False,
        min_lead_us: int = LATE_JOINER_MIN_LEAD_US,
    ) -> int:
        """Return a safe minimum playback timestamp for late-join replay."""
        now_us = self._clock.now_us()
        if role is not None:
            effective_lead_us = max(min_lead_us, self._role_send_ahead_us(role))
        else:
            effective_lead_us = max(0, min_lead_us)
        target_us = now_us + effective_lead_us
        if align_to_channel_tail and channel_id is not None and channel_id in self._channel_timing:
            # For channels that currently have no other subscribers, anchor catch-up
            # to that channel's own live tail when it is near real time. If that tail
            # drifted far ahead (e.g., reconnect with large server-side buffering),
            # use the standard near-now target to avoid long audible startup delays.
            channel_tail_us = max(now_us, self._channel_timing[channel_id])
            align_ceiling_us = now_us + (
                self._role_send_ahead_us(role) if role is not None else DEFAULT_INITIAL_DELAY_US
            )
            if channel_tail_us <= align_ceiling_us:
                return max(channel_tail_us, target_us)
        return target_us

    async def commit_audio(self, *, play_start_us: int | None = None) -> int:  # noqa: PLR0915
        """
        Encode and send all prepared audio to players.

        This is an asynchronous method that:
        1. Encodes prepared PCM for each required format
        2. Assigns timestamps to encoded chunks
        3. Sends chunks to connected players via role hooks

        Args:
            play_start_us: If provided, use this timestamp directly for all channels
                instead of auto-calculating from clock. Useful for multi-server sync
                where all servers share a clock and need identical timestamps.

        Returns:
            The earliest play_start_us timestamp across all channels.
            If stop() is called while this commit is already in-flight, the
            commit aborts remaining work and returns a stable timestamp without
            delivering any additional audio after the stop.

        Raises:
            StreamStoppedError: If the stream is already stopped when commit
                begins.
        """
        # Check if stopped
        if self._is_stopped:
            raise StreamStoppedError("Cannot commit audio on a stopped stream")
        commit_generation = self._stream_generation

        self._commit_in_flight += 1
        try:
            # Drain historical buffers
            historical = dict(self._historical_buffers)
            self._historical_buffers.clear()
            historical_start_us = dict(self._historical_start_us)
            self._historical_start_us.clear()

            # If no pending audio (live or historical), return earliest channel timing
            if not self._channel_buffers and not historical:
                now_us = self._clock.now_us()
                if not self._channel_timing:
                    self._channel_timing[MAIN_CHANNEL] = now_us + self._min_send_ahead_us()
                    self._channel_timing_residue[MAIN_CHANNEL] = 0
                return min(self._channel_timing.values())

            # Process historical buffers first: assign timestamps and inject into caches.
            # This initializes _channel_timing for historical channels so the live chunk
            # (if any) continues seamlessly after.
            if historical:
                await self._process_historical_buffers(
                    historical,
                    historical_start_us,
                    commit_generation=commit_generation,
                )
                if not self._is_generation_active(commit_generation):
                    return self._stopped_commit_return_value()

            # Drain live channel buffers
            prepared = dict(self._channel_buffers)
            self._channel_buffers.clear()

            if not prepared:
                # Historical-only commit: cache updated by _process_historical_buffers().
                self._prune_role_chunk_cache()
                return min(self._channel_timing.values())

            # Calculate duration for each channel and warn on misalignment
            durations_us = self._calculate_channel_durations(prepared)
            self._warn_duration_misalignment(durations_us)

            # Capture play_start_us for each channel
            channel_play_start = self._resolve_channel_play_start(
                prepared,
                play_start_us=play_start_us,
            )

            # Advance channel timing by overwriting durations_us with drift-free values.
            for channel_id, (pcm, fmt) in prepared.items():
                bytes_per_sample = fmt.bit_depth // 8
                frame_stride = bytes_per_sample * fmt.channels
                sample_count = len(pcm) // frame_stride
                durations_us[channel_id] = self._advance_channel_timing(
                    channel_id, sample_count, fmt.sample_rate
                )
                self._channels_with_committed_audio.add(channel_id)

            # Keep non-prepared active channels on the shared timeline.
            #
            # This avoids channel drift when an upstream channel (e.g., per-device DSP)
            # times out and we commit only a subset of channels for one or more cycles.
            # Those channels skip audio for this commit but should remain clock-aligned
            # when they resume.
            reference_duration_us = max(durations_us.values(), default=0)
            if reference_duration_us > 0:
                base_start_us = min(channel_play_start.values())
                for channel_id in self._get_active_audio_channels():
                    if channel_id in prepared:
                        continue
                    if channel_id not in self._channel_timing:
                        self._channel_timing[channel_id] = base_start_us
                        self._channel_timing_residue[channel_id] = 0
                    self._channel_timing[channel_id] += reference_duration_us

            # Cache PCM chunks before encoding (if enabled)
            cached_any_pcm = False
            for channel_id, (pcm_bytes, fmt) in prepared.items():
                channel_int = channel_id.int
                if channel_int not in self._pcm_cache_enabled_channels:
                    continue
                pcm_chunk = CachedPCMChunk(
                    timestamp_us=channel_play_start[channel_id],
                    duration_us=durations_us[channel_id],
                    pcm_data=pcm_bytes,
                    sample_rate=fmt.sample_rate,
                    bit_depth=fmt.bit_depth,
                    channels=fmt.channels,
                    sample_type=fmt.sample_type,
                )
                self._pcm_chunk_cache.setdefault(channel_int, deque()).append(pcm_chunk)
                cached_any_pcm = True
            if cached_any_pcm:
                self._signal_pcm_cache_update()
                if self._catchup_tasks:
                    # Yield so an active catch-up task can consume the new PCM before delivery.
                    await asyncio.sleep(0)

            # Role-based audio delivery via hooks
            role_cache_results = await self._deliver_audio_to_roles(
                prepared,
                channel_play_start,
                commit_generation=commit_generation,
            )
            if not self._is_generation_active(commit_generation):
                return self._stopped_commit_return_value()
            # Merge role-based cache results into the cache
            for cache_key, chunks in role_cache_results.items():
                self._role_chunk_cache[cache_key].extend(chunks)

            # Prune old chunks from cache
            self._prune_role_chunk_cache()
            self._prune_stale_channel_timing()

            # Return earliest play_start_us
            return min(channel_play_start.values())
        finally:
            self._commit_in_flight -= 1
            # Flush deferred joins with last commit (in case there ever were multiple
            # simultaneous commit calls).
            if self._commit_in_flight == 0 and self._pending_join_roles:
                pending = list(self._pending_join_roles)
                self._pending_join_roles.clear()
                for role in pending:
                    self._do_role_join(role)

    def _resolve_channel_play_start(
        self,
        prepared: dict[UUID, tuple[bytes, AudioFormat]],
        *,
        play_start_us: int | None,
    ) -> dict[UUID, int]:
        """Resolve play-start timestamp per prepared channel and initialize timing."""
        channel_play_start: dict[UUID, int] = {}

        if play_start_us is not None:
            # Explicit timestamp mode: caller-provided value is authoritative
            # across mode switches, so overwrite any stale advanced timing.
            for channel_id in prepared:
                channel_play_start[channel_id] = play_start_us
                self._channel_timing[channel_id] = play_start_us
                self._channel_timing_residue[channel_id] = 0
            return channel_play_start

        # Auto-calculate mode (existing behavior).
        now_us = self._clock.now_us()
        target_min_us = now_us + self._min_send_ahead_us()
        # Limit timeline sharing/rebase inputs to channels participating in this commit
        # (active subscribers + prepared payloads). This excludes stale timing entries
        # from inactive channels.
        active_or_prepared_channels = self._get_active_audio_channels() | set(prepared)
        for channel_id in prepared:
            if channel_id not in self._channel_timing:
                shared_candidates = [
                    self._channel_timing[cid]
                    for cid in active_or_prepared_channels
                    if cid in self._channel_timing
                ]
                if shared_candidates:
                    # Late-introduced channels should join the shared timeline
                    # instead of restarting from now+delay behind active channels.
                    # The rebase below shifts every channel up together until the
                    # earliest reaches the send-ahead floor, so clamping here would
                    # advance this one twice and leave it permanently behind the
                    # members it just joined.
                    self._channel_timing[channel_id] = min(shared_candidates)
                    self._channel_timing_residue[channel_id] = 0
                else:
                    self._channel_timing[channel_id] = target_min_us
                    self._channel_timing_residue[channel_id] = 0

        # If audio production stalls (e.g., the upstream source blocks), the scheduled
        # play timeline can drift into the past. Rebase the timeline so new audio is
        # always scheduled at least `_min_send_ahead_us()` from "now".
        rebase_candidates = [
            self._channel_timing[cid]
            for cid in active_or_prepared_channels
            if cid in self._channel_timing
        ]
        if not rebase_candidates:
            # During reconnect/handshake edges there may be no active/prepared channels.
            # Fall back to all timings to keep existing behavior in that case.
            rebase_candidates = list(self._channel_timing.values())
        min_timing_us = min(rebase_candidates)
        if min_timing_us < target_min_us:
            shift_us = target_min_us - min_timing_us
            for channel_id in self._channel_timing:
                self._channel_timing[channel_id] += shift_us

        for channel_id in prepared:
            channel_play_start[channel_id] = self._channel_timing[channel_id]
        return channel_play_start

    async def _process_historical_buffers(  # noqa: PLR0915
        self,
        historical: dict[UUID, list[tuple[bytes, AudioFormat]]],
        historical_start_us: dict[UUID, int] | None = None,
        *,
        commit_generation: int | None = None,
    ) -> None:
        """Process historical audio buffers, assigning timestamps consecutively.

        For each channel with historical data:
        1. Initialize timing from explicit start, shared timeline alignment, or default delay
        2. Assign timestamps so chunks play consecutively
        3. Cache PCM and encode for active roles
        4. Advance channel timing to end of last historical chunk

        :param historical: Channel ID -> list of (pcm, format) chunks (oldest first).
        """
        now_us = self._clock.now_us()
        min_delivery_timestamp_us = now_us + self._min_send_ahead_us()

        for channel_id, chunks in historical.items():
            if not self._is_generation_active(commit_generation):
                return
            # Avoid drifting by keeping track of the fraction, must match _advance_channel_timing
            chunk_durations: list[int] = []
            residue = 0
            last_rate: int | None = None
            for pcm_bytes, fmt in chunks:
                bytes_per_sample = fmt.bit_depth // 8
                frame_stride = bytes_per_sample * fmt.channels
                sample_count = len(pcm_bytes) // frame_stride
                if last_rate != fmt.sample_rate:
                    residue = 0
                    last_rate = fmt.sample_rate
                numerator = residue + sample_count * 1_000_000
                chunk_duration_us, residue = divmod(numerator, fmt.sample_rate)
                chunk_durations.append(chunk_duration_us)
            total_duration_us = sum(chunk_durations)

            if historical_start_us is not None and channel_id in historical_start_us:
                self._channel_timing[channel_id] = historical_start_us[channel_id]
                self._channel_timing_residue[channel_id] = 0
            elif channel_id in self._channel_timing:
                if channel_id not in self._channels_with_committed_audio:
                    # Synthetic timing (from alignment of missing channels) should not block
                    # history injection. Backfill from current channel tail so history ends
                    # exactly at the existing shared timeline position.
                    self._channel_timing[channel_id] = max(
                        0, self._channel_timing[channel_id] - total_duration_us
                    )
                    self._channel_timing_residue[channel_id] = 0
            elif self._channel_timing:
                # Align injected history so it ends at the current shared timeline.
                # This avoids leaving newly injected channels permanently behind live.
                # Prefer active-channel timings so inactive stale entries do not
                # pull historical injection behind the live timeline.
                active_channels = self._get_active_audio_channels()
                anchor_candidates = [
                    self._channel_timing[cid]
                    for cid in active_channels
                    if cid in self._channel_timing
                ]
                if not anchor_candidates:
                    # Fallback preserves historical-only behavior when no roles are active.
                    anchor_candidates = list(self._channel_timing.values())
                anchor_timing_us = min(anchor_candidates)
                self._channel_timing[channel_id] = max(0, anchor_timing_us - total_duration_us)
                self._channel_timing_residue[channel_id] = 0
            else:
                self._channel_timing[channel_id] = now_us + self._min_send_ahead_us()
                self._channel_timing_residue[channel_id] = 0

            for pcm_bytes, fmt in chunks:
                chunk_start_us = self._channel_timing[channel_id]

                bytes_per_sample = fmt.bit_depth // 8
                frame_stride = bytes_per_sample * fmt.channels
                sample_count = len(pcm_bytes) // frame_stride
                duration_us = self._advance_channel_timing(
                    channel_id, sample_count, fmt.sample_rate
                )

                # Cache PCM (if enabled)
                channel_int = channel_id.int
                if channel_int in self._pcm_cache_enabled_channels:
                    pcm_chunk = CachedPCMChunk(
                        timestamp_us=chunk_start_us,
                        duration_us=duration_us,
                        pcm_data=pcm_bytes,
                        sample_rate=fmt.sample_rate,
                        bit_depth=fmt.bit_depth,
                        channels=fmt.channels,
                        sample_type=fmt.sample_type,
                    )
                    self._pcm_chunk_cache.setdefault(channel_int, deque()).append(pcm_chunk)
                    self._signal_pcm_cache_update()

                # For late-join injection, historical chunks may already be too old by the
                # time they're processed. Keep timeline/cache continuity, but don't deliver
                # chunks that would be immediately dropped as late by the connection layer.
                if chunk_start_us + duration_us <= min_delivery_timestamp_us:
                    continue

                # Encode and deliver to roles
                prepared = {channel_id: (pcm_bytes, fmt)}
                play_start = {channel_id: chunk_start_us}
                role_cache_results = await self._deliver_audio_to_roles(
                    prepared,
                    play_start,
                    commit_generation=commit_generation,
                )
                if not self._is_generation_active(commit_generation):
                    return
                for cache_key, cached_chunks in role_cache_results.items():
                    self._role_chunk_cache[cache_key].extend(cached_chunks)
                # Yield so historical injection doesn't starve the event loop.
                await asyncio.sleep(0)
            self._channels_with_committed_audio.add(channel_id)

    def _advance_channel_timing(self, channel_id: UUID, sample_count: int, sample_rate: int) -> int:
        """Advance _channel_timing drift-free via residue accumulator. Return µs added."""
        assert channel_id in self._channel_timing, f"channel {channel_id} not initialised"
        if self._channel_timing_rate.get(channel_id) != sample_rate:
            # Residue is modulo the previous rate; reset on rate change to keep units consistent.
            self._channel_timing_residue[channel_id] = 0
            self._channel_timing_rate[channel_id] = sample_rate
        numerator = self._channel_timing_residue.get(channel_id, 0) + sample_count * 1_000_000
        delta_us, self._channel_timing_residue[channel_id] = divmod(numerator, sample_rate)
        self._channel_timing[channel_id] += delta_us
        return delta_us

    def _calculate_channel_durations(
        self,
        prepared: dict[UUID, tuple[bytes, AudioFormat]],
    ) -> dict[UUID, int]:
        """Calculate duration in microseconds for each prepared channel."""
        durations: dict[UUID, int] = {}
        for channel_id, (pcm, fmt) in prepared.items():
            bytes_per_sample = fmt.bit_depth // 8
            frame_stride = bytes_per_sample * fmt.channels
            sample_count = len(pcm) // frame_stride
            duration_us = int(sample_count * 1_000_000 / fmt.sample_rate)
            durations[channel_id] = duration_us
        return durations

    def _warn_duration_misalignment(self, durations_us: dict[UUID, int]) -> None:
        """Log a warning if channel durations differ significantly."""
        if len(durations_us) <= 1:
            return

        values = list(durations_us.values())
        min_dur = min(values)
        max_dur = max(values)

        # Warn if durations differ by more than 5ms
        tolerance_us = 5000
        if max_dur - min_dur > tolerance_us:
            _LOGGER.warning(
                "Channel durations differ by %dus (tolerance: %dus)",
                max_dur - min_dur,
                tolerance_us,
            )

    def _group_roles_by_pcm_requirements(
        self,
        prepared: dict[UUID, tuple[bytes, AudioFormat]],
    ) -> dict[tuple[UUID, int, int, int], list[tuple[SendspinClient, Role, AudioRequirements]]]:
        """Group roles by their PCM requirements (channel_id, sample_rate, bit_depth, channels)."""
        # Key type: (channel_id, sample_rate, bit_depth, channels)
        roles_by_pcm: defaultdict[
            tuple[UUID, int, int, int], list[tuple[SendspinClient, Role, AudioRequirements]]
        ] = defaultdict(list)

        for client, role in self._get_audio_roles():
            req = role.get_audio_requirements()
            if req is None:
                continue

            channel_id = req.channel_id or MAIN_CHANNEL
            if channel_id not in prepared:
                continue

            pcm_key = (channel_id, req.sample_rate, req.bit_depth, req.channels)
            roles_by_pcm[pcm_key].append((client, role, req))

        return roles_by_pcm

    def _get_active_audio_channels(self) -> set[UUID]:
        """Return channels currently used by connected audio roles."""
        channels: set[UUID] = set()
        for _client, role in self._get_audio_roles():
            req = role.get_audio_requirements()
            if req is None:
                continue
            channels.add(req.channel_id or MAIN_CHANNEL)
        return channels

    async def _resample_for_roles(
        self,
        roles_by_pcm: dict[
            tuple[UUID, int, int, int], list[tuple[SendspinClient, Role, AudioRequirements]]
        ],
        prepared: dict[UUID, tuple[bytes, AudioFormat]],
        channel_play_start: dict[UUID, int],
    ) -> dict[tuple[UUID, int, int, int], _ResampledPCM]:
        """Resample PCM once per unique PCM key. Returns (channel, rate, depth, ch) -> (pcm, ts).

        Resampler state (PyAV objects) is cached and must stay on a single thread.
        Running cached resamplers across a worker pool can deadlock depending on
        thread scheduling, so resampling runs synchronously on the loop thread.
        """
        if not roles_by_pcm:
            return {}

        results: dict[tuple[UUID, int, int, int], _ResampledPCM] = {}
        shared_results_by_resampler: dict[_ResamplerKey, _ResampledPCM] = {}
        for pcm_key in roles_by_pcm:
            channel_id, target_sample_rate, target_bit_depth, target_channels = pcm_key
            source_pcm, source_format = prepared[channel_id]
            input_timestamp_us = channel_play_start[channel_id]

            target_format = _processing_format_for_roles(
                source_format,
                target_sample_rate=target_sample_rate,
                target_bit_depth=target_bit_depth,
                target_channels=target_channels,
            )

            resampler_key = _ResamplerKey(
                channel_id=channel_id,
                source_format=source_format,
                target_sample_rate=target_sample_rate,
                target_channels=target_channels,
                target_bit_depth=target_format.bit_depth,
                target_sample_type=target_format.sample_type,
            )

            if resampler_key in shared_results_by_resampler:
                results[pcm_key] = shared_results_by_resampler[resampler_key]
                continue

            state = self._get_cached_resampler(resampler_key)
            if state is None:
                state = _create_resampler_state(resampler_key, source_format, target_format)
                self._cache_resampler(state)
            elif (
                state.pending_input_timestamp_us is not None
                and state.pending_input_timestamp_us > input_timestamp_us
            ):
                # Resampler already consumed this chunk (e.g. catch-up task
                # pre-processed the PCM cache and promoted the warmed graph).
                # Re-pushing the same input would duplicate samples in the
                # encoder buffer and shift content vs labels. Emit no output.
                empty = _ResampledPCM(
                    pcm_data=b"",
                    output_start_ts=state.pending_timestamp_us or input_timestamp_us,
                    sample_count=0,
                    needs_s32_to_s24_conversion=state.needs_s32_to_s24_conversion,
                    sample_type=state.target_sample_type,
                )
                shared_results_by_resampler[resampler_key] = empty
                results[pcm_key] = empty
                continue

            resampled = _resample_pcm_standalone(
                state, source_pcm, source_format, input_timestamp_us
            )
            shared_results_by_resampler[resampler_key] = resampled
            results[pcm_key] = resampled
        return results

    def _resolve_frame_duration_us(self, req: AudioRequirements) -> int:
        if req.frame_duration_us is not None:
            return req.frame_duration_us
        if req.transformer is not None:
            return req.transformer.frame_duration_us
        return 25_000

    def _build_transform_key(
        self, req: AudioRequirements, channel_id: UUID, role: Role | None = None
    ) -> TransformKey:
        # Cache by (role_id, channel_id_int); invalidated by on_role_format_changed()
        transformer_type = type(req.transformer) if req.transformer is not None else type(None)
        frame_duration_us = self._resolve_frame_duration_us(req)
        options = normalize_options(req.transform_options)

        if role is not None:
            cache_key = (id(role), channel_id.int)
            cached = self._transform_key_cache.get(cache_key)
            if (
                cached is not None
                and cached.transformer_type == transformer_type
                and cached.sample_rate == req.sample_rate
                and cached.bit_depth == req.bit_depth
                and cached.channels == req.channels
                and cached.frame_duration_us == frame_duration_us
                and cached.options == options
            ):
                return cached

        tkey = TransformKey(
            channel_id=channel_id.int,  # Use int for faster hashing
            transformer_type=transformer_type,
            sample_rate=req.sample_rate,
            bit_depth=req.bit_depth,
            channels=req.channels,
            frame_duration_us=frame_duration_us,
            options=options,
        )

        if role is not None:
            self._transform_key_cache[cache_key] = tkey

        return tkey

    def _encode_transform_for_key(
        self,
        tkey: TransformKey,
        transformer: AudioTransformer | None,
        pcm_data: bytes,
        output_ts: int,
        duration_us: int,
    ) -> list[tuple[bytes, int, int]]:
        """Encode PCM while resetting transformer state after long production gaps."""
        last_input_end_us = self._transform_last_input_end_us.get(tkey)
        if (
            transformer is not None
            and last_input_end_us is not None
            and output_ts - last_input_end_us > 1_500_000
        ):
            transformer.reset()

        # Reset transformer if its internal timeline has drifted too far from the
        # expected output position. This catches gradual drift from timeline rebasing
        # (when audio production is slower than real-time, each commit's rebase shifts
        # timestamps forward, but the transformer's internal timeline only advances by
        # the actual audio duration).
        if (
            transformer is not None
            and transformer.pending_timestamp_us is not None
            and abs(transformer.pending_timestamp_us - (output_ts + duration_us))
            > _TRANSFORMER_DRIFT_THRESHOLD_US
        ):
            transformer.reset()

        encoded = _encode_for_transform_key(
            transformer,
            pcm_data,
            output_ts,
            duration_us,
        )
        self._transform_last_input_end_us[tkey] = output_ts + duration_us
        return encoded

    async def _transform_and_deliver(  # noqa: PLR0915
        self,
        roles_by_pcm: dict[
            tuple[UUID, int, int, int], list[tuple[SendspinClient, Role, AudioRequirements]]
        ],
        resampled_pcm: dict[tuple[UUID, int, int, int], _ResampledPCM],
        *,
        commit_generation: int | None = None,
    ) -> dict[TransformKey, list[CachedChunk]]:
        """Transform PCM, deliver live chunks to roles, and return cache results.

        Encoding runs sequentially on the event loop, yielding every few keys.
        """
        # Collect unique encoding tasks: tkey -> (transformer, pcm_data, output_ts, duration_us)
        encode_tasks: dict[TransformKey, tuple[AudioTransformer | None, bytes, int, int]] = {}
        roles_by_transform: defaultdict[TransformKey, list[Role]] = defaultdict(list)

        for pcm_key, roles_list in roles_by_pcm.items():
            channel_id, rate, depth, _channels = pcm_key
            resampled = resampled_pcm[pcm_key]
            pcm_data = resampled.pcm_data
            output_ts = resampled.output_start_ts
            duration_us = int(resampled.sample_count * 1_000_000 / rate) if rate > 0 else 0
            needs_s32_to_s24_conversion = resampled.needs_s32_to_s24_conversion

            # Quantize float PCM once per pcm_key — all TransformKeys sharing this
            # pcm_key produce the same quantizer _ResamplerKey. Using the quantizer's
            # output_start_ts and sample_count ensures downstream timestamps reflect
            # any buffering or trimming introduced by the quantizer graph.
            if resampled.sample_type == "float":
                edge_quantized = _quantize_float_pcm(
                    channel_id=channel_id,
                    pcm_data=pcm_data,
                    output_ts=output_ts,
                    sample_rate=rate,
                    channels=_channels,
                    target_bit_depth=depth,
                    resampler_cache=self._resamplers,
                )
                pcm_data = edge_quantized.pcm_data
                output_ts = edge_quantized.output_start_ts
                duration_us = int(edge_quantized.sample_count * 1_000_000 / rate) if rate > 0 else 0
                needs_s32_to_s24_conversion = edge_quantized.needs_s32_to_s24_conversion

            grouped_by_key: defaultdict[
                TransformKey, list[tuple[SendspinClient, Role, AudioRequirements]]
            ] = defaultdict(list)
            for client, role, req in roles_list:
                tkey = self._build_transform_key(req, channel_id, role)
                grouped_by_key[tkey].append((client, role, req))

            for tkey, grouped in grouped_by_key.items():
                roles_by_transform[tkey].extend(role for _client, role, _req in grouped)
                if self._catchup_state.get(tkey) == "catching_up":
                    continue
                # Skip a chunk catch-up already replayed to avoid a duplicate at the seam.
                key_cache = self._role_chunk_cache.get(tkey)
                if key_cache and key_cache[-1].timestamp_us + key_cache[-1].duration_us > output_ts:
                    continue
                if tkey in encode_tasks:
                    continue
                transformer = grouped[0][2].transformer
                transformed_pcm = pcm_data
                if (
                    needs_s32_to_s24_conversion
                    and depth == 24
                    and isinstance(transformer, PcmPassthrough)
                ):
                    transformed_pcm = _convert_s32_to_s24(transformed_pcm)
                encode_tasks[tkey] = (transformer, transformed_pcm, output_ts, duration_us)

        # TransformKey -> list of (data, timestamp_us, duration_us)
        transformed: dict[TransformKey, list[tuple[bytes, int, int]]] = {}

        if encode_tasks:
            for processed, (tkey, (transformer, pcm_data, output_ts, duration_us)) in enumerate(
                encode_tasks.items(), start=1
            ):
                if not self._is_generation_active(commit_generation):
                    return {}
                transformed[tkey] = self._encode_transform_for_key(
                    tkey,
                    transformer,
                    pcm_data,
                    output_ts,
                    duration_us,
                )
                if processed % 2 == 0:
                    # Keep the loop responsive during large multi-role commits.
                    await asyncio.sleep(0)
                    if not self._is_generation_active(commit_generation):
                        return {}

        cache_results: defaultdict[TransformKey, list[CachedChunk]] = defaultdict(list)
        active_roles = {role for _client, role in self._get_audio_roles()}

        for tkey, frame_list in transformed.items():
            roles = roles_by_transform.get(tkey, [])
            if not any(r in active_roles for r in roles):
                # Every recipient was excluded mid-commit (e.g. a format change
                # deferred the sole role to the join path); caching these frames
                # would repopulate the transform key the boundary just evicted.
                continue
            cached_for_key: list[CachedChunk] = []
            for data, ts, dur in frame_list:
                cached = CachedChunk(
                    timestamp_us=ts, duration_us=dur, payload=data, byte_count=len(data)
                )
                cached_for_key.append(cached)
            if not cached_for_key:
                continue
            cache_results[tkey].extend(cached_for_key)

            # Deliver live chunks directly; connection layer enforces late-drop/backpressure.
            # Share one AudioChunk across roles so its packed frame is built once.
            audio_chunks = [
                AudioChunk(
                    data=cached_chunk.payload,
                    timestamp_us=cached_chunk.timestamp_us,
                    duration_us=cached_chunk.duration_us,
                    byte_count=cached_chunk.byte_count,
                )
                for cached_chunk in cached_for_key
            ]
            for role in roles:
                if role not in active_roles:
                    continue
                self._ensure_role_started(role)
                if role not in self._started_roles:
                    continue
                for audio_chunk in audio_chunks:
                    role.on_audio_chunk(audio_chunk)

        return cache_results

    async def _deliver_audio_to_roles(
        self,
        prepared: dict[UUID, tuple[bytes, AudioFormat]],
        channel_play_start: dict[UUID, int],
        *,
        commit_generation: int | None = None,
    ) -> dict[TransformKey, list[CachedChunk]]:
        """
        Deliver audio to roles using the hook-based flow.

        This method:
        1. Groups roles by unique PCM requirements
        2. Resamples source PCM to each unique target format
        3. Transforms and delivers via role.on_audio_chunk()

        Returns:
            Dict of TransformKey -> list of CachedChunk for late joiners.
        """
        roles_by_pcm = self._group_roles_by_pcm_requirements(prepared)
        if not roles_by_pcm:
            return {}

        if not self._is_generation_active(commit_generation):
            return {}
        resampled = await self._resample_for_roles(roles_by_pcm, prepared, channel_play_start)
        if not self._is_generation_active(commit_generation):
            return {}
        return await self._transform_and_deliver(
            roles_by_pcm,
            resampled,
            commit_generation=commit_generation,
        )

    def _prune_role_chunk_cache(self) -> None:
        """Remove old chunks from the role-based cache."""
        now_us = self._clock.now_us()

        for key in list(self._role_chunk_cache.keys()):
            chunks = self._role_chunk_cache[key]
            while chunks and chunks[0].timestamp_us + chunks[0].duration_us <= now_us:
                chunks.popleft()
            if not chunks:
                del self._role_chunk_cache[key]

        for channel_int in list(self._pcm_chunk_cache.keys()):
            pcm_chunks = self._pcm_chunk_cache[channel_int]
            while pcm_chunks and (pcm_chunks[0].timestamp_us + pcm_chunks[0].duration_us <= now_us):
                pcm_chunks.popleft()
            if not pcm_chunks:
                self._pcm_chunk_cache.pop(channel_int, None)

    def _prune_stale_channel_timing(self) -> None:
        """Remove _channel_timing entries for channels with no active roles or pending audio."""
        active = self._get_active_audio_channels()
        for ch in list(self._channel_timing):
            if ch in active:
                continue
            if ch in self._channel_buffers:
                continue
            if ch in self._historical_buffers:
                continue
            if ch in self._channels_with_committed_audio:
                # Keep timing for channels that have received real audio so the
                # timeline continues to advance. Without this, timing is reset
                # on every commit when no roles are active, preventing
                # sleep_to_limit_buffer from throttling the commit loop.
                continue
            del self._channel_timing[ch]
            self._channel_timing_residue.pop(ch, None)
            self._channel_timing_rate.pop(ch, None)
            self._channels_with_committed_audio.discard(ch)

    def _ensure_role_started(self, role: Role) -> None:
        if role in self._started_roles:
            return
        role.on_stream_start()
        self._started_roles.add(role)

    def _send_cached_chunks_to_role(
        self,
        role: Role,
        cached_chunks: list[CachedChunk],
        now_us: int,
    ) -> None:
        """Send cached chunks to a role, skipping chunks that are already late."""
        skipped_late = 0
        for cached_chunk in cached_chunks:
            if cached_chunk.timestamp_us + cached_chunk.duration_us <= now_us:
                skipped_late += 1
                continue

            self._ensure_role_started(role)

            chunk = AudioChunk(
                data=cached_chunk.payload,
                timestamp_us=cached_chunk.timestamp_us,
                duration_us=cached_chunk.duration_us,
                byte_count=cached_chunk.byte_count,
            )
            role.on_audio_chunk(chunk)

        if skipped_late > 0:
            _LOGGER.debug(
                "Skipped %s late cached chunk(s) for role %s (ts < now_us=%s)",
                skipped_late,
                role.role_family,
                now_us,
            )

    def on_role_leave(self, role: Role) -> None:
        """Remove role-specific state so re-joins get fresh stream/start."""
        self._started_roles.discard(role)
        self._pending_join_roles.discard(role)
        req = role.get_audio_requirements()
        if req is not None:
            channel_id = req.channel_id or MAIN_CHANNEL
            tkey = self._build_transform_key(req, channel_id, role)
            if not self._other_roles_use_transform_key(tkey, role):
                self._role_chunk_cache.pop(tkey, None)
                self._transform_last_input_end_us.pop(tkey, None)
                if req.transformer is not None:
                    req.transformer.reset()
            if not self._other_roles_share_resampler_shape(req, channel_id, role):
                # Drop orphan resampler so a rejoin sees no stale FIR state.
                for rkey in list(self._resamplers.keys()):
                    if (
                        rkey.channel_id == channel_id
                        and rkey.target_sample_rate == req.sample_rate
                        and rkey.target_bit_depth == req.bit_depth
                        and rkey.target_channels == req.channels
                    ):
                        self._resamplers.pop(rkey, None)
        for tkey in list(self._catchup_roles.keys()):
            roles = self._catchup_roles[tkey]
            roles.discard(role)
            if not roles:
                self._catchup_roles.pop(tkey, None)
                self._catchup_state.pop(tkey, None)
                task = self._catchup_tasks.pop(tkey, None)
                if task is not None:
                    task.cancel()
        # Drop cached transform keys for this role to avoid stale lookups.
        role_id = id(role)
        stale_transform_keys: set[TransformKey] = set()
        for cache_key in list(self._transform_key_cache.keys()):
            if cache_key[0] == role_id:
                stale_transform_keys.add(self._transform_key_cache[cache_key])
                self._transform_key_cache.pop(cache_key, None)
        for stale_tkey in stale_transform_keys:
            self._transform_last_input_end_us.pop(stale_tkey, None)

    def on_role_format_changed(self, role: Role) -> None:
        """Re-anchor a role whose audio format changed mid-stream.

        The role stays active (unlike on_role_leave()) but is treated as a
        late joiner: stale transform-key work is discarded and the join path
        re-anchors it near the playhead in the new format. In-flight commits
        need no abort: the synchronous join deferral moves the role into
        _pending_join_roles, which excludes it from live delivery.
        """
        # Invalidate transform key cache entries for this role
        role_id = id(role)
        stale_transform_keys: set[TransformKey] = set()
        for cache_key in list(self._transform_key_cache.keys()):
            if cache_key[0] == role_id:
                stale_transform_keys.add(self._transform_key_cache[cache_key])
                self._transform_key_cache.pop(cache_key, None)
        for stale_tkey in stale_transform_keys:
            self._transform_last_input_end_us.pop(stale_tkey, None)
            if not self._other_roles_use_transform_key(stale_tkey, role):
                # Sole user of the old format: drop its encoded cache and
                # reset the encoder so a change back starts clean.
                self._role_chunk_cache.pop(stale_tkey, None)
                transformer = self._group.transformer_pool.get(stale_tkey)
                if transformer is not None:
                    transformer.reset()

        # Clean up any catchup state referencing this role
        for tkey in list(self._catchup_roles.keys()):
            roles = self._catchup_roles[tkey]
            roles.discard(role)
            if not roles:
                self._catchup_roles.pop(tkey, None)
                self._catchup_state.pop(tkey, None)
                task = self._catchup_tasks.pop(tkey, None)
                if task is not None:
                    task.cancel()

        # The changing role's client discards its un-played buffer at the new
        # stream/start, so re-anchor it near the playhead via the late-join
        # path: the buffered content is re-encoded in the new format and
        # handed off to live at the tail, leaving the shared channel timeline
        # untouched for unchanged roles.
        self.on_role_join(role)

    def has_cached_chunks(self) -> bool:
        """Return True if there are cached chunks for late joiners."""
        return any(len(chunks) > 0 for chunks in self._role_chunk_cache.values())

    def on_role_join(self, role: Role) -> None:
        """
        Handle late joiner catch-up via hooks.

        Uses the role-based chunk cache to deliver cached audio to a role
        that just joined.

        Args:
            role: The role that joined.
        """
        # Join immediately so replay anchors to the current shared timeline.
        # Deferring by wall-clock time can desynchronize grouped players.
        if self._commit_in_flight > 0:
            # _role_chunk_cache not yet updated for the in-flight chunk, run once commit is done.
            self._pending_join_roles.add(role)
            return
        self._do_role_join(role)

    def _do_role_join(self, role: Role) -> None:
        """Execute role join with cached chunk replay."""
        self._pending_join_roles.discard(role)
        # A rejoining role (e.g. warm reconnect) must receive on_stream_start()
        # again so the new transport gets stream/start before any audio chunks.
        self._started_roles.discard(role)
        if self._is_stopped:
            return
        req = role.get_audio_requirements()
        if req is None:
            return

        channel_id = req.channel_id or MAIN_CHANNEL
        self._rebase_first_join_channel_timing(channel_id, role)

        # Get cached chunks for this transformer from the role-based cache
        cache_key = self._build_transform_key(req, channel_id, role)
        cached: deque[CachedChunk] = self._role_chunk_cache.get(cache_key, deque())

        if not cached:
            if cache_key in self._catchup_state:
                self._catchup_roles.setdefault(cache_key, set()).add(role)
                return

            if self._has_pcm_cache(channel_id) and not self._other_roles_use_transform_key(
                cache_key, role
            ):
                channel_pcm_cache = self._pcm_chunk_cache.get(channel_id.int)
                if channel_pcm_cache:
                    late_join_target_us = self.get_late_join_target_timestamp_us(
                        role=role,
                        channel_id=channel_id,
                        align_to_channel_tail=(channel_id != MAIN_CHANNEL),
                    )
                    latest_cached_end_us = (
                        channel_pcm_cache[-1].timestamp_us + channel_pcm_cache[-1].duration_us
                    )
                    if latest_cached_end_us <= late_join_target_us:
                        self._rebase_far_ahead_join_tail(channel_id, role)
                        if self._channel_timing:
                            self._ensure_role_started(role)
                        return

                if (
                    not role.replay_from_pcm_cache()
                    and self._has_established_resampler_for(req, channel_id)
                    and self._skip_replay_keeps_join_near_playhead(channel_id, role)
                ):
                    # A live role drives this resampler shape, so replaying through it
                    # would shift the live role's audio across the hand-off. Skip replay
                    # only when live audio lands near the playhead, else fall through to
                    # catch-up so the joiner is not stranded waiting at a far-ahead tail.
                    self._rebase_far_ahead_join_tail(channel_id, role)
                    if self._channel_timing:
                        self._ensure_role_started(role)
                    return

                self._catchup_state[cache_key] = "catching_up"
                self._catchup_roles[cache_key] = {role}
                task = create_task(self._start_catchup_encoding(role, req, channel_id, cache_key))
                # An eager task can finish inside create_task, after its own
                # cleanup already removed the map entry. Don't re-add it.
                if not task.done():
                    self._catchup_tasks[cache_key] = task
                return

            if self._channel_timing:
                self._rebase_far_ahead_join_tail(channel_id, role)
                self._ensure_role_started(role)
            return

        now_us = self._clock.now_us()
        min_timestamp_us = self.get_late_join_target_timestamp_us(
            role=role,
            channel_id=channel_id,
            align_to_channel_tail=False,
        )

        start_index = 0
        for chunk in cached:
            if chunk.timestamp_us + chunk.duration_us > min_timestamp_us:
                break
            start_index += 1

        if start_index >= len(cached):
            if self._channel_timing:
                self._ensure_role_started(role)
            return

        if _LOGGER.isEnabledFor(logging.DEBUG):
            first_ts = cached[start_index].timestamp_us
            last_ts = cached[-1].timestamp_us
            _LOGGER.debug(
                "Late join catch-up via role hook: chunks=%s ts_range=%s..%s",
                len(cached) - start_index,
                first_ts,
                last_ts,
            )

        self._send_cached_chunks_to_role(role, list(islice(cached, start_index, None)), now_us)

    def _other_roles_use_transform_key(self, cache_key: TransformKey, exclude_role: Role) -> bool:
        """Check if any other active pipeline role uses the same TransformKey."""
        for client in self._group.clients:
            for role in client.active_roles:
                if role is exclude_role:
                    continue
                if not self._role_in_audio_pipeline(client, role):
                    continue
                req = role.get_audio_requirements()
                if req is None:
                    continue
                channel_id = req.channel_id or MAIN_CHANNEL
                tkey = self._build_transform_key(req, channel_id, role)
                if tkey == cache_key:
                    return True
        return False

    def _other_roles_share_resampler_shape(
        self, req: AudioRequirements, channel_id: UUID, exclude_role: Role
    ) -> bool:
        """Check whether another role drives a resampler at the same target PCM shape."""
        for client in self._group.clients:
            for role in client.active_roles:
                if role is exclude_role:
                    continue
                if not self._role_in_audio_pipeline(client, role):
                    continue
                other_req = role.get_audio_requirements()
                if other_req is None:
                    continue
                other_channel = other_req.channel_id or MAIN_CHANNEL
                if other_channel != channel_id:
                    continue
                if (
                    other_req.sample_rate == req.sample_rate
                    and other_req.bit_depth == req.bit_depth
                    and other_req.channels == req.channels
                ):
                    return True
        return False

    def _has_established_resampler_for(self, req: AudioRequirements, channel_id: UUID) -> bool:
        """Check whether a live FIR resampler already exists at the same target PCM shape."""
        for rkey, rstate in self._resamplers.items():
            if rstate.is_passthrough:
                continue
            if (
                rkey.channel_id == channel_id
                and rkey.target_sample_rate == req.sample_rate
                and rkey.target_bit_depth == req.bit_depth
                and rkey.target_channels == req.channels
            ):
                return True
        return False

    def _channel_has_other_audio_roles(self, channel_id: UUID, exclude_role: Role) -> bool:
        """Check whether any other active pipeline role is subscribed to the channel."""
        for client in self._group.clients:
            for role in client.active_roles:
                if role is exclude_role:
                    continue
                if not self._role_in_audio_pipeline(client, role):
                    continue
                req = role.get_audio_requirements()
                if req is None:
                    continue
                role_channel_id = req.channel_id or MAIN_CHANNEL
                if role_channel_id == channel_id:
                    return True
        return False

    def _rebase_first_join_channel_timing(self, channel_id: UUID, joining_role: Role) -> None:
        """Rebase stale channel timing to the shared timeline for first joiners."""
        if channel_id not in self._channel_timing or not self._channel_timing:
            return
        if self._channel_has_other_audio_roles(channel_id, joining_role):
            return
        other_channel_timings = [
            timing_us for cid, timing_us in self._channel_timing.items() if cid != channel_id
        ]
        if not other_channel_timings:
            return
        reference_timing_us = min(other_channel_timings)
        self._channel_timing[channel_id] = max(
            self._channel_timing[channel_id], reference_timing_us
        )
        self._channel_timing_residue[channel_id] = 0

    def _skip_replay_keeps_join_near_playhead(self, channel_id: UUID, role: Role) -> bool:
        """Whether skipping PCM replay still lands the joiner near the playhead.

        True when the tail is within a normal buffer of now, or far ahead but
        clampable. A far-ahead tail that cannot be clamped means the joiner must
        replay instead of waiting for live audio at the tail.
        """
        tail_us = self._channel_timing.get(channel_id)
        if tail_us is None:
            return True
        now_us = self._clock.now_us()
        if tail_us <= now_us + self._role_send_ahead_us(role):
            return True
        return self._can_clamp_far_ahead_tail(channel_id, role)

    def _can_clamp_far_ahead_tail(self, channel_id: UUID, role: Role) -> bool:
        """Whether the tail can move, blocked by other roles or committed audio."""
        return not (
            self._channel_has_other_audio_roles(channel_id, role)
            or channel_id in self._channels_with_committed_audio
        )

    def _rebase_far_ahead_join_tail(self, channel_id: UUID, joining_role: Role) -> None:
        """Clamp far-ahead solo-channel timing so a rejoin can resume promptly."""
        if channel_id not in self._channel_timing:
            return
        if not self._can_clamp_far_ahead_tail(channel_id, joining_role):
            return
        now_us = self._clock.now_us()
        max_resume_start_us = now_us + self._role_send_ahead_us(joining_role)
        self._channel_timing[channel_id] = min(
            self._channel_timing[channel_id], max_resume_start_us
        )
        self._channel_timing_residue[channel_id] = 0

    async def _encode_pcm_sequence(
        self,
        pcm_chunks: list[CachedPCMChunk],
        encoder: AudioTransformer | None,
        req: AudioRequirements,
        channel_id: UUID,
        *,
        resamplers: dict[_ResamplerKey, _ResamplerState] | None = None,
        quantizers: dict[_ResamplerKey, _ResamplerState] | None = None,
    ) -> list[CachedChunk]:
        """Resample PCM chunks to the target format and encode them sequentially.

        Yields periodically so a deep cache does not stall the loop. Output timestamps
        come from the cached chunks, so they are unaffected by when the work runs.

        Pass `resamplers`/`quantizers` to share state across calls (single resampler
        instance per key reused between batches).
        """
        tkey = self._build_transform_key(req, channel_id)
        cached: list[CachedChunk] = []
        if resamplers is None:
            resamplers = {}
        if quantizers is None:
            quantizers = {}
        prev_resampler_key: _ResamplerKey | None = None
        since_yield_us = 0

        for chunk in pcm_chunks:
            since_yield_us += chunk.duration_us
            if since_yield_us >= _PCM_SEQUENCE_YIELD_INTERVAL_US:
                since_yield_us = 0
                await asyncio.sleep(0)

            source_format = AudioFormat(
                sample_rate=chunk.sample_rate,
                bit_depth=chunk.bit_depth,
                channels=chunk.channels,
                sample_type=chunk.sample_type,
            )
            target_format = _processing_format_for_roles(
                source_format,
                target_sample_rate=req.sample_rate,
                target_bit_depth=req.bit_depth,
                target_channels=req.channels,
            )
            current_resampler_key = _ResamplerKey(
                channel_id=channel_id,
                source_format=source_format,
                target_sample_rate=req.sample_rate,
                target_channels=req.channels,
                target_bit_depth=target_format.bit_depth,
                target_sample_type=target_format.sample_type,
            )
            # Format changed: flush prior resampler now to keep output timestamp-ordered.
            if prev_resampler_key is not None and prev_resampler_key != current_resampler_key:
                prev_state = resamplers.pop(prev_resampler_key, None)
                if prev_state is not None:
                    cached.extend(
                        self._flush_resampler_to_chunks(
                            prev_state, quantizers, encoder, req, channel_id
                        )
                    )

            resampler_state = resamplers.get(current_resampler_key)
            if resampler_state is None:
                resampler_state = _create_resampler_state(
                    current_resampler_key,
                    source_format,
                    target_format,
                )
                resamplers[current_resampler_key] = resampler_state
            prev_resampler_key = current_resampler_key

            resampled = _resample_pcm_standalone(
                resampler_state,
                chunk.pcm_data,
                source_format,
                chunk.timestamp_us,
            )
            if not resampled.pcm_data:
                continue

            resampled_pcm = resampled.pcm_data
            needs_s32_to_s24_conversion = resampled.needs_s32_to_s24_conversion
            output_start_ts = resampled.output_start_ts
            sample_count = resampled.sample_count

            if resampled.sample_type == "float":
                edge_quantized = _quantize_float_pcm(
                    channel_id=channel_id,
                    pcm_data=resampled_pcm,
                    output_ts=resampled.output_start_ts,
                    sample_rate=req.sample_rate,
                    channels=req.channels,
                    target_bit_depth=req.bit_depth,
                    resampler_cache=quantizers,
                )
                resampled_pcm = edge_quantized.pcm_data
                needs_s32_to_s24_conversion = edge_quantized.needs_s32_to_s24_conversion
                output_start_ts = edge_quantized.output_start_ts
                sample_count = edge_quantized.sample_count

            if (
                needs_s32_to_s24_conversion
                and req.bit_depth == 24
                and isinstance(encoder, PcmPassthrough)
            ):
                resampled_pcm = _convert_s32_to_s24(resampled_pcm)
            duration_us = int(sample_count * 1_000_000 / req.sample_rate) if sample_count > 0 else 0

            encoded_frames = self._encode_transform_for_key(
                tkey,
                encoder,
                resampled_pcm,
                output_start_ts,
                duration_us,
            )
            for data, ts, dur in encoded_frames:
                cached.append(
                    CachedChunk(
                        timestamp_us=ts,
                        duration_us=dur,
                        payload=data,
                        byte_count=len(data),
                    )
                )

        return cached

    async def _encode_catchup_sequence(
        self,
        pcm_chunks: list[CachedPCMChunk],
        encoder: AudioTransformer | None,
        req: AudioRequirements,
        channel_id: UUID,
        *,
        resamplers: dict[_ResamplerKey, _ResamplerState] | None = None,
        quantizers: dict[_ResamplerKey, _ResamplerState] | None = None,
    ) -> list[CachedChunk]:
        return await self._encode_pcm_sequence(
            pcm_chunks,
            encoder,
            req,
            channel_id,
            resamplers=resamplers,
            quantizers=quantizers,
        )

    def _flush_resampler_to_chunks(
        self,
        resampler_state: _ResamplerState,
        quantizers: dict[_ResamplerKey, _ResamplerState],
        encoder: AudioTransformer | None,
        req: AudioRequirements,
        channel_id: UUID,
    ) -> list[CachedChunk]:
        """Drain one resampler's FIR tail and run it through quantizer/encoder."""
        tkey = self._build_transform_key(req, channel_id)
        drained = _flush_resampler(resampler_state)
        if drained.sample_count == 0 or not drained.pcm_data:
            return []

        resampled_pcm = drained.pcm_data
        needs_s32_to_s24_conversion = drained.needs_s32_to_s24_conversion
        output_start_ts = drained.output_start_ts
        sample_count = drained.sample_count

        if drained.sample_type == "float":
            edge_quantized = _quantize_float_pcm(
                channel_id=channel_id,
                pcm_data=resampled_pcm,
                output_ts=output_start_ts,
                sample_rate=req.sample_rate,
                channels=req.channels,
                target_bit_depth=req.bit_depth,
                resampler_cache=quantizers,
            )
            resampled_pcm = edge_quantized.pcm_data
            needs_s32_to_s24_conversion = edge_quantized.needs_s32_to_s24_conversion
            output_start_ts = edge_quantized.output_start_ts
            sample_count = edge_quantized.sample_count
            if sample_count == 0 or not resampled_pcm:
                return []

        if (
            needs_s32_to_s24_conversion
            and req.bit_depth == 24
            and isinstance(encoder, PcmPassthrough)
        ):
            resampled_pcm = _convert_s32_to_s24(resampled_pcm)
        duration_us = int(sample_count * 1_000_000 / req.sample_rate) if sample_count > 0 else 0

        encoded_frames = self._encode_transform_for_key(
            tkey,
            encoder,
            resampled_pcm,
            output_start_ts,
            duration_us,
        )
        return [
            CachedChunk(
                timestamp_us=ts,
                duration_us=dur,
                payload=data,
                byte_count=len(data),
            )
            for data, ts, dur in encoded_frames
        ]

    async def _start_catchup_encoding(  # noqa: PLR0915
        self,
        role: Role,
        req: AudioRequirements,
        channel_id: UUID,
        cache_key: TransformKey,
    ) -> None:
        """Start catch-up encoding from PCM cache for a new TransformKey."""
        channel_int = channel_id.int
        encoder = req.transformer
        # Catchup-local caches: one resampler/quantizer state spans the full PCM
        # cache to avoid per-batch FIR transients. Drained at the end so encoder
        # pending reaches the live timeline; otherwise `_encode_for_transform_key`
        # back-shifts the first live chunk via candidate_base.
        catchup_resamplers: dict[_ResamplerKey, _ResamplerState] = {}
        catchup_quantizers: dict[_ResamplerKey, _ResamplerState] = {}
        # Pre-existing keys belong to other live roles; concurrent stubs for
        # this role's tkey will appear later and are safe to overwrite.
        established_resampler_keys = set(self._resamplers.keys())

        try:
            if encoder is not None:
                encoder.reset()
            self._transform_last_input_end_us.pop(cache_key, None)
            pcm_chunks = list(self._pcm_chunk_cache.get(channel_int, []))
            align_to_channel_tail = channel_id != MAIN_CHANNEL
            target_ts = self.get_late_join_target_timestamp_us(
                role=role,
                channel_id=channel_id,
                align_to_channel_tail=align_to_channel_tail,
            )
            encode_start_ts = target_ts
            if encoder is not None and align_to_channel_tail:
                encode_start_ts = max(0, target_ts - ENCODER_CATCHUP_WARMUP_US)
            eligible = [
                chunk
                for chunk in pcm_chunks
                if chunk.timestamp_us + chunk.duration_us > encode_start_ts
            ]

            if not eligible:
                if self._channel_timing:
                    for r in self._catchup_roles.get(cache_key, {role}):
                        self._ensure_role_started(r)
                return

            encoded = await self._encode_catchup_sequence(
                eligible,
                encoder,
                req,
                channel_id,
                resamplers=catchup_resamplers,
                quantizers=catchup_quantizers,
            )

            if encoded and self._catchup_state.get(cache_key) == "catching_up":
                self._role_chunk_cache[cache_key].extend(encoded)

            last_encoded_end_us = (
                encoded[-1].timestamp_us + encoded[-1].duration_us if encoded else 0
            )
            # Track source PCM progress separately from encoded progress. Some
            # codecs buffer input and may emit no packets for a given chunk.
            last_source_end_us = eligible[-1].timestamp_us + eligible[-1].duration_us
            if self._pcm_cache_signal is None:
                self._pcm_cache_signal = asyncio.Event()
            signal = self._pcm_cache_signal
            # Abandon catch-up if no new PCM arrives in time.
            idle_timeout_s = 0.5

            while last_encoded_end_us < target_ts:
                new_pcm = [
                    chunk
                    for chunk in self._pcm_chunk_cache.get(channel_int, [])
                    if chunk.timestamp_us >= last_source_end_us
                ]
                if not new_pcm:
                    # Clear-then-recheck guards against the commit_audio set()
                    # racing in between our cache scan and entering wait().
                    signal.clear()
                    new_pcm = [
                        chunk
                        for chunk in self._pcm_chunk_cache.get(channel_int, [])
                        if chunk.timestamp_us >= last_source_end_us
                    ]
                    if not new_pcm:
                        try:
                            await asyncio.wait_for(signal.wait(), timeout=idle_timeout_s)
                        except TimeoutError:
                            _LOGGER.debug(
                                "Catch-up idle timeout for %s (encoded_end=%s target=%s)",
                                cache_key,
                                last_encoded_end_us,
                                target_ts,
                            )
                            break
                        continue
                last_source_end_us = new_pcm[-1].timestamp_us + new_pcm[-1].duration_us

                new_encoded = await self._encode_catchup_sequence(
                    new_pcm,
                    encoder,
                    req,
                    channel_id,
                    resamplers=catchup_resamplers,
                    quantizers=catchup_quantizers,
                )

                if new_encoded and self._catchup_state.get(cache_key) == "catching_up":
                    self._role_chunk_cache[cache_key].extend(new_encoded)
                    last_encoded_end_us = new_encoded[-1].timestamp_us + new_encoded[-1].duration_us

            # Drain PCM committed after the snapshot (a commit racing the hand-off) so it
            # is replayed here, not dropped. The final empty check flows into the
            # synchronous send/flip below with no await between, so a racing commit cannot
            # slip a chunk past both this replay and the suppressed live delivery.
            while True:
                pending = [
                    chunk
                    for chunk in self._pcm_chunk_cache.get(channel_int, [])
                    if chunk.timestamp_us >= last_source_end_us
                ]
                if not pending:
                    break
                last_source_end_us = pending[-1].timestamp_us + pending[-1].duration_us
                drained = await self._encode_catchup_sequence(
                    pending,
                    encoder,
                    req,
                    channel_id,
                    resamplers=catchup_resamplers,
                    quantizers=catchup_quantizers,
                )
                if drained and self._catchup_state.get(cache_key) == "catching_up":
                    self._role_chunk_cache[cache_key].extend(drained)
                    last_encoded_end_us = drained[-1].timestamp_us + drained[-1].duration_us

            # Promote FIR-warmed catch-up state, skipping shared live keys.
            for rkey, rstate in catchup_resamplers.items():
                if rkey not in established_resampler_keys:
                    self._resamplers[rkey] = rstate
            for qkey, qstate in catchup_quantizers.items():
                if qkey not in established_resampler_keys:
                    self._resamplers[qkey] = qstate

            now_us = self._clock.now_us()
            encoded_cache: deque[CachedChunk] = self._role_chunk_cache.get(cache_key, deque())
            chunks_to_send = list(encoded_cache)
            for r in self._catchup_roles.get(cache_key, {role}):
                self._send_cached_chunks_to_role(r, chunks_to_send, now_us)

            self._catchup_state[cache_key] = "live"
        finally:
            if self._catchup_state.get(cache_key) != "live":
                # Catchup was cancelled (e.g. role left mid-catchup). Clear partial state
                # so a future re-join doesn't hit a stale cache or stale encoder pending
                # that would back-shift live chunks via candidate_base.
                self._catchup_state.pop(cache_key, None)
                self._catchup_roles.pop(cache_key, None)
                self._role_chunk_cache.pop(cache_key, None)
                self._transform_last_input_end_us.pop(cache_key, None)
                if encoder is not None:
                    encoder.reset()
            self._catchup_tasks.pop(cache_key, None)

    def _cancel_catchup_tasks(self) -> None:
        for task in self._catchup_tasks.values():
            task.cancel()
        self._catchup_tasks.clear()
        self._catchup_state.clear()
        self._catchup_roles.clear()

    def stop(self, *, keep_stream: bool = False) -> None:
        """
        Stop only this PushStream transport.

        After calling stop(), commit_audio() will raise StreamStoppedError.
        Resets transformers and sends stream/end via role hooks. Pass
        ``keep_stream=True`` to skip the stream/end emission when the
        caller will immediately create a new PushStream and wants clients
        to treat the transition as continuous. In that case, call
        ``clear()`` first if buffered client audio should be discarded
        before the successor PushStream takes over.

        Does not change the owning group's logical playback state. Call
        group.stop() to stop transport and also set playback state to STOPPED.
        """
        if self._is_stopped:
            return
        self._is_stopped = True
        self._stream_generation += 1

        # Reset transformers so any internal encoder state is discarded.
        transformers_by_key: dict[TransformKey, AudioTransformer] = {}
        for _client, role in self._get_audio_roles():
            req = role.get_audio_requirements()
            if req and req.transformer:
                channel_id = req.channel_id or MAIN_CHANNEL
                tkey = self._build_transform_key(req, channel_id, role)
                transformers_by_key.setdefault(tkey, req.transformer)
        for transformer in transformers_by_key.values():
            transformer.reset()

        if not keep_stream:
            # Send stream/end to all roles with audio requirements via hooks
            for _client, role in self._get_audio_roles():
                role.on_stream_end()

        # Clear role tracking state
        self._started_roles.clear()
        self._pending_join_roles.clear()
        self._cancel_catchup_tasks()
        self._pcm_chunk_cache.clear()
        self._historical_buffers.clear()
        self._historical_start_us.clear()
        self._transform_last_input_end_us.clear()
        self._channels_with_committed_audio.clear()

    def clear(self) -> None:
        """
        Clear all pending audio and reset timing.

        This is used for seek operations or track changes where buffered
        audio is discarded. Sends stream/clear to all roles via hooks.
        """
        # Bump the stream generation so any in-flight commit_audio()
        # coroutine bails out before delivering chunks under the new epoch.
        self._stream_generation += 1

        # Clear pending audio
        self._channel_buffers.clear()
        self._historical_buffers.clear()
        self._historical_start_us.clear()

        # Reset per-channel timing
        self._channel_timing.clear()
        self._channel_timing_residue.clear()
        self._channel_timing_rate.clear()
        self._channels_with_committed_audio.clear()

        # Clear chunk cache
        self._role_chunk_cache.clear()
        self._pending_join_roles.clear()
        self._pcm_chunk_cache.clear()
        self._transform_last_input_end_us.clear()
        self._cancel_catchup_tasks()

        # Reset inline resamplers
        self._resamplers.clear()

        # Reset transformers so they don't carry stale timestamp state
        reset_transformers: dict[TransformKey, AudioTransformer] = {}
        for _client, role in self._get_audio_roles():
            req = role.get_audio_requirements()
            if req and req.transformer:
                channel_id = req.channel_id or MAIN_CHANNEL
                tkey = self._build_transform_key(req, channel_id, role)
                reset_transformers.setdefault(tkey, req.transformer)

        for transformer in reset_transformers.values():
            transformer.reset()

        # Clear role tracking state
        self._started_roles.clear()

        # Send stream/clear to all roles with audio requirements via hooks
        for _client, role in self._get_audio_roles():
            role.on_stream_clear()
