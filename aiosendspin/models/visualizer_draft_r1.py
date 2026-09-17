"""Deprecated visualizer role models for the `visualizer@_draft_r1` wire."""

# DEPRECATED(spec-pr-86): remove in aiosendspin <version>

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from .base import SendspinConfig, SendspinModel

VisualizerType = Literal["loudness", "f_peak", "spectrum", "beat"]
SupportedVisualizerType = Literal["loudness", "f_peak", "spectrum"]
SpectrumScale = Literal["lin", "log", "mel"]

# TODO: Add "beat" once wire format + extraction support is implemented.
_SUPPORTED_TYPES: tuple[SupportedVisualizerType, ...] = ("loudness", "f_peak", "spectrum")


# Client -> Server: client/hello visualizer support object
@dataclass(frozen=True)
class ClientHelloVisualizerSpectrum(SendspinModel):
    """Spectrum configuration from client/hello visualizer support."""

    n_disp_bins: int
    scale: SpectrumScale
    f_min: int
    f_max: int
    rate_max: int

    def __post_init__(self) -> None:
        """Validate spectrum config bounds."""
        if self.n_disp_bins <= 0:
            raise ValueError(f"n_disp_bins must be > 0, got {self.n_disp_bins}")
        if self.f_min < 0:
            raise ValueError(f"f_min must be >= 0, got {self.f_min}")
        if self.f_max <= self.f_min:
            raise ValueError(f"f_max must be > f_min, got f_min={self.f_min}, f_max={self.f_max}")
        if self.rate_max <= 0:
            raise ValueError(f"rate_max must be > 0, got {self.rate_max}")

    def to_wire_dict(self) -> dict[str, Any]:
        """Serialize to stream/start visualizer.spectrum payload."""
        return self.to_dict()

    class Config(SendspinConfig):
        """Config for json serialization."""

        omit_none = True


@dataclass
class ClientHelloVisualizerSupport(SendspinModel):
    """Visualizer support payload for client/hello draft-r1 negotiation."""

    buffer_capacity: int
    types: list[str] | None = None
    batch_max: int | None = None
    spectrum: ClientHelloVisualizerSpectrum | None = None

    @classmethod
    def __pre_deserialize__(cls, payload: dict[str, Any]) -> dict[str, Any]:
        """Normalize incoming support payload before dataclass construction."""
        raw_types = payload.get("types")
        if isinstance(raw_types, list):
            deduped: list[str] = []
            for value in raw_types:
                if isinstance(value, str) and value in _SUPPORTED_TYPES and value not in deduped:
                    deduped.append(value)
            payload = dict(payload)
            payload["types"] = deduped
        return payload

    def __post_init__(self) -> None:
        """Validate support object constraints."""
        if self.types == []:
            raise ValueError(
                "visualizer support 'types' did not contain any supported type "
                f"(supported: {list(_SUPPORTED_TYPES)})"
            )
        if self.buffer_capacity <= 0:
            raise ValueError(f"buffer_capacity must be > 0, got {self.buffer_capacity}")
        if self.batch_max is not None and self.batch_max <= 0:
            raise ValueError(f"batch_max must be > 0, got {self.batch_max}")
        if self.types is not None and "spectrum" in self.types and self.spectrum is None:
            raise ValueError("visualizer support must include 'spectrum' object")

    class Config(SendspinConfig):
        """Config for json serialization."""

        omit_none = True


# Server -> Client: stream/start visualizer object
@dataclass(frozen=True)
class StreamStartVisualizer(SendspinModel):
    """Negotiated draft visualizer stream config returned in stream/start."""

    types: tuple[SupportedVisualizerType, ...]
    batch_max: int
    spectrum: ClientHelloVisualizerSpectrum | None = None

    @classmethod
    def from_support(cls, support: ClientHelloVisualizerSupport) -> StreamStartVisualizer:
        """Create server stream config from validated client support data."""
        if support.types is None:
            raise ValueError("visualizer support must include 'types'")
        if support.batch_max is None:
            raise ValueError("visualizer support must include 'batch_max'")
        stream_types = cast(
            "tuple[SupportedVisualizerType, ...]",
            tuple(typed for typed in support.types if typed in _SUPPORTED_TYPES),
        )
        if not stream_types:
            raise ValueError("visualizer stream must contain at least one supported type")
        return cls(
            types=stream_types,
            batch_max=support.batch_max,
            spectrum=support.spectrum,
        )

    def to_wire_dict(self) -> dict[str, Any]:
        """Serialize to stream/start payload format."""
        return self.to_dict()

    class Config(SendspinConfig):
        """Config for json serialization."""

        omit_none = True


# Client -> Server: stream/request-format visualizer object
@dataclass
class StreamRequestFormatVisualizer(SendspinModel):
    """Draft visualizer format request payload."""

    types: list[VisualizerType] | None = None
    batch_max: int | None = None
    spectrum: ClientHelloVisualizerSpectrum | None = None

    class Config(SendspinConfig):
        """Config for json serialization."""

        omit_none = True


@dataclass(slots=True)
class VisualizerFrame:
    """Single visualizer frame parsed by clients from binary payloads."""

    timestamp_us: int
    loudness: int | None = None
    f_peak: int | None = None
    spectrum: list[int] | None = None
