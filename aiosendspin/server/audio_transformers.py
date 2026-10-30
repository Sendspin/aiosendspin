"""Audio transformers for role-specific encoding/processing.

Transformers convert resampled PCM audio into role-specific output formats.
They are managed by TransformerPool for deduplication across roles.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable

T = TypeVar("T", bound="AudioTransformer")


@runtime_checkable
class AudioTransformer(Protocol):
    """Protocol for audio transformers.

    Transformers process PCM audio into role-specific output.
    Examples: FlacEncoder, OpusEncoder, FFTComputer for visualizer.
    """

    @property
    def frame_duration_us(self) -> int:
        """Static frame duration, used for `TransformKey` identity and the send-ahead.

        Per-frame wire duration is emitted by `process` / `flush` and may
        vary by ±1µs at non-divisible rates.
        """
        ...

    def process(self, pcm: bytes, timestamp_us: int, duration_us: int) -> list[tuple[bytes, int]]:
        """Transform PCM chunk into output frames.

        Args:
            pcm: Raw PCM audio data (already resampled to target format).
            timestamp_us: Playback timestamp in microseconds, minus the timeline jumps
                this transformer has followed.
            duration_us: Duration of this chunk in microseconds.

        Returns:
            List of `(frame_bytes, frame_duration_us)` pairs. May be empty if
            buffering incomplete frame. May contain multiple frames if input
            spans multiple frame boundaries.
        """
        ...

    def flush(self) -> list[tuple[bytes, int]]:
        """Flush remaining buffered audio at stream end.

        Returns:
            Final `(frame_bytes, frame_duration_us)` pairs, possibly padded with silence.
        """
        ...

    @property
    def pending_timestamp_us(self) -> int | None:
        """Timestamp of the earliest audio sample not yet emitted, or None."""
        return None

    def reset(self) -> None:
        """Reset internal state.

        Called on stream/clear to discard buffered state.
        """
        ...


@dataclass(frozen=True, slots=True)
class TransformKey:
    """Stable identity for transformed output."""

    channel_id: int
    transformer_type: type
    sample_rate: int
    bit_depth: int
    channels: int
    frame_duration_us: int
    options: tuple[tuple[str, str], ...]
    kwargs_fingerprint: tuple[tuple[str, Hashable], ...] = ()


def normalize_options(options: Mapping[str, str] | None) -> tuple[tuple[str, str], ...]:
    """Normalize options mapping into a deterministic, hashable tuple."""
    if not options:
        return ()
    return tuple(sorted(((key, value) for key, value in options.items()), key=lambda kv: kv[0]))


def normalize_constructor_kwargs(
    kwargs: Mapping[str, object] | None,
) -> tuple[tuple[str, Hashable], ...]:
    """Normalize extra constructor kwargs into a deterministic key tuple."""
    if not kwargs:
        return ()
    normalized: list[tuple[str, Hashable]] = []
    for key, value in kwargs.items():
        if not isinstance(value, Hashable):
            raise TypeError(
                f"Transformer kwarg '{key}' has unhashable value type: {type(value).__name__}"
            )
        normalized.append((key, value))
    return tuple(sorted(normalized, key=lambda kv: kv[0]))


class TransformerPool:
    """Manages shared transformer instances.

    Transformers are keyed by:
    (channel_id, type, sample_rate, bit_depth, channels, frame, options, extra kwargs).
    Multiple roles with the same configuration share the same transformer, enabling encoding
    deduplication. Reuse is stream-safe because the key includes channel_id, and each channel
    has at most one active PushStream at a time.
    """

    def __init__(self) -> None:
        """Initialize an empty transformer pool."""
        self._transformers: dict[TransformKey, AudioTransformer] = {}

    def get_or_create(
        self,
        transformer_type: type[T],
        *,
        channel_id: int,
        sample_rate: int,
        bit_depth: int,
        channels: int,
        frame_duration_us: int,
        options: Mapping[str, str] | None = None,
        **constructor_kwargs: object,
    ) -> T:
        """Get existing transformer or create new one.

        Args:
            transformer_type: Transformer class to instantiate.
            channel_id: Output channel identity (separates active streams).
            sample_rate: Target sample rate in Hz.
            bit_depth: Target bit depth.
            channels: Target channel count.
            frame_duration_us: Output frame duration.
            options: Optional transformer options map (string key/value protocol options).
            **constructor_kwargs: Additional constructor kwargs that affect output behavior.
                These are included in dedupe keying using a deterministic frozen fingerprint.
        """
        transformer_kwargs: dict[str, object] = {
            "sample_rate": sample_rate,
            "bit_depth": bit_depth,
            "channels": channels,
            "chunk_duration_us": frame_duration_us,
        }
        transformer_kwargs.update(constructor_kwargs)
        if options is not None:
            transformer_kwargs["options"] = options

        key = TransformKey(
            channel_id=channel_id,
            transformer_type=transformer_type,
            sample_rate=sample_rate,
            bit_depth=bit_depth,
            channels=channels,
            frame_duration_us=frame_duration_us,
            options=normalize_options(options),
            kwargs_fingerprint=normalize_constructor_kwargs(constructor_kwargs),
        )
        if key not in self._transformers:
            self._transformers[key] = transformer_type(**transformer_kwargs)
        return self._transformers[key]  # type: ignore[return-value]

    def get(self, key: TransformKey) -> AudioTransformer | None:
        """Return the pooled transformer for key, if one exists."""
        return self._transformers.get(key)

    def reset_all(self) -> None:
        """Reset all transformers (called on stream/clear)."""
        for transformer in self._transformers.values():
            transformer.reset()


__all__ = [
    "AudioTransformer",
    "TransformKey",
    "TransformerPool",
    "normalize_constructor_kwargs",
    "normalize_options",
]
