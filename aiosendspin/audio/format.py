"""PCM formats and sample conversion helpers."""

from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass
from typing import Literal


def _get_av() -> types.ModuleType:
    """Lazy import of av module to avoid slow startup and keep it optional."""
    return importlib.import_module("av")


_numpy_unavailable = False


def _get_numpy() -> types.ModuleType | None:
    """Lazy import numpy to optimize s32<->s24 conversion when available."""
    global _numpy_unavailable  # noqa: PLW0603
    if _numpy_unavailable:
        return None
    try:
        import numpy as np  # noqa: PLC0415
    except ImportError:
        _numpy_unavailable = True
        return None
    return np  # type: ignore[no-any-return,unused-ignore]


# PyAV channel layout names by channel count.
_CHANNEL_LAYOUTS = {
    1: "mono",
    2: "stereo",
    3: "2.1",
    4: "quad",
    5: "4.1",
    6: "5.1",
    7: "6.1",
    8: "7.1",
    10: "9.1",
}


@dataclass(frozen=True)
class AudioFormat:
    """Describe raw PCM audio."""

    sample_rate: int
    """Sample rate in Hz (e.g., 44100, 48000)."""
    bit_depth: int
    """Bit depth in bits per sample (16, 24, or 32)."""
    channels: int
    """Number of audio channels (1 for mono, 2 for stereo)."""
    sample_type: Literal["int", "float"] = "int"
    """PCM sample type. Use ``float`` to represent 32-bit floating-point PCM input."""

    def resolve_av_format(self) -> tuple[int, str, str, int]:
        """Return wire width, PyAV format, channel layout, and PyAV width."""
        if self.sample_type not in ("int", "float"):
            raise ValueError("sample_type must be 'int' or 'float'")

        if self.sample_type == "float":
            if self.bit_depth != 32:
                raise ValueError("Only 32-bit float PCM is supported")
            wire_bytes_per_sample = 4
            av_format = "flt"
            av_bytes_per_sample = 4
        elif self.bit_depth == 16:
            wire_bytes_per_sample = 2
            av_format = "s16"
            av_bytes_per_sample = 2
        elif self.bit_depth == 24:
            # Convert packed s24 through PyAV's s32 format.
            wire_bytes_per_sample = 3
            av_format = "s32"
            av_bytes_per_sample = 4
        elif self.bit_depth == 32:
            wire_bytes_per_sample = 4
            av_format = "s32"
            av_bytes_per_sample = 4
        else:
            raise ValueError("Only 16-bit, 24-bit, and 32-bit PCM are supported")

        if self.channels < 1:
            raise ValueError(f"Unsupported channel count: {self.channels}")
        # PyAV accepts an unnamed layout for any other channel count.
        layout = _CHANNEL_LAYOUTS.get(self.channels, f"{self.channels} channels")

        return wire_bytes_per_sample, av_format, layout, av_bytes_per_sample


def _convert_s24_to_s32(data: bytes) -> bytes:
    """Expand packed 24-bit PCM samples to PyAV's left-aligned s32 representation."""
    if len(data) % 3:
        raise ValueError("s24 PCM buffer length must be a multiple of 3 bytes")

    if np := _get_numpy():
        arr = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3)
        zero_column = np.zeros((arr.shape[0], 1), dtype=np.uint8)
        expanded = (
            np.concatenate((zero_column, arr), axis=1)
            if sys.byteorder == "little"
            else np.concatenate((arr, zero_column), axis=1)
        )
        return bytes(expanded.tobytes())

    if sys.byteorder == "little":
        return b"".join(b"\x00" + data[i : i + 3] for i in range(0, len(data), 3))
    return b"".join(data[i : i + 3] + b"\x00" for i in range(0, len(data), 3))


def _convert_s32_to_s24(data: bytes) -> bytes:
    """Convert 32-bit PCM samples to packed 24-bit samples."""
    if len(data) % 4:
        raise ValueError("s32 PCM buffer length must be a multiple of 4 bytes")

    if np := _get_numpy():
        if sys.byteorder == "little":
            arr = np.frombuffer(data, dtype="<i4")
            return bytes(arr.view(np.uint8).reshape(-1, 4)[:, 1:4].tobytes())
        arr = np.frombuffer(data, dtype=">i4")
        return bytes(arr.view(np.uint8).reshape(-1, 4)[:, 0:3].tobytes())

    if sys.byteorder == "little":
        return b"".join(data[i + 1 : i + 4] for i in range(0, len(data), 4))
    return b"".join(data[i : i + 3] for i in range(0, len(data), 4))


def _validate_pcm_buffer_length(data: bytes, *, expected: int, context: str) -> None:
    """Validate a PCM buffer's byte length."""
    if len(data) != expected:
        msg = f"{context} PCM buffer length {len(data)} does not match expected {expected} bytes"
        raise ValueError(msg)


__all__ = [
    "AudioFormat",
    "_convert_s24_to_s32",
    "_convert_s32_to_s24",
    "_get_av",
    "_get_numpy",
    "_validate_pcm_buffer_length",
]
