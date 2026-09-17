"""Tests for audio format helper utilities."""

import sys

import pytest

# The conversion helpers live in aiosendspin.audio.format and look up _get_numpy
# in that module's namespace, so monkeypatching must target the format module
# (server.audio only re-exports them for backward compatibility).
from aiosendspin.audio import format as format_module
from aiosendspin.server.audio import AudioFormat, _convert_s24_to_s32, _convert_s32_to_s24

S32_SAMPLES = bytes([0x01, 0x11, 0x21, 0x31, 0x02, 0x12, 0x22, 0x32])


def _expected_s24_samples() -> bytes:
    if sys.byteorder == "little":
        return bytes([0x11, 0x21, 0x31, 0x12, 0x22, 0x32])
    return bytes([0x01, 0x11, 0x21, 0x02, 0x12, 0x22])


def _expected_s32_from_s24_samples() -> bytes:
    if sys.byteorder == "little":
        return bytes([0x00, 0x11, 0x21, 0x31, 0x00, 0x12, 0x22, 0x32])
    return bytes([0x01, 0x11, 0x21, 0x00, 0x02, 0x12, 0x22, 0x00])


def test_resolve_audio_format_24_bit_uses_s32_in_pyav() -> None:
    """24-bit wire format should map to s32 for PyAV processing."""
    wire_bytes, av_format, layout, av_bytes = AudioFormat(
        sample_rate=48_000, bit_depth=24, channels=2
    ).resolve_av_format()
    assert wire_bytes == 3
    assert av_format == "s32"
    assert layout == "stereo"
    assert av_bytes == 4


def test_resolve_audio_format_32_bit_is_supported() -> None:
    """32-bit PCM should be supported by resolver."""
    wire_bytes, av_format, layout, av_bytes = AudioFormat(
        sample_rate=44_100, bit_depth=32, channels=1
    ).resolve_av_format()
    assert wire_bytes == 4
    assert av_format == "s32"
    assert layout == "mono"
    assert av_bytes == 4


def test_resolve_audio_format_32_bit_float_uses_flt_in_pyav() -> None:
    """32-bit float PCM should map to flt for PyAV processing."""
    wire_bytes, av_format, layout, av_bytes = AudioFormat(
        sample_rate=44_100, bit_depth=32, channels=2, sample_type="float"
    ).resolve_av_format()
    assert wire_bytes == 4
    assert av_format == "flt"
    assert layout == "stereo"
    assert av_bytes == 4


def test_resolve_audio_format_float_requires_32_bit() -> None:
    """Float PCM should reject non-32-bit configurations."""
    with pytest.raises(ValueError, match="32-bit float"):
        AudioFormat(
            sample_rate=48_000, bit_depth=16, channels=2, sample_type="float"
        ).resolve_av_format()


@pytest.mark.parametrize("channels", [9, 11, 64])
def test_resolve_audio_format_uses_unnamed_layout_without_a_named_one(channels: int) -> None:
    """Channel counts without a named layout map to PyAV's unnamed layout."""
    _, _, layout, _ = AudioFormat(
        sample_rate=48_000, bit_depth=16, channels=channels
    ).resolve_av_format()
    assert layout == f"{channels} channels"


def test_resolve_audio_format_rejects_no_channels() -> None:
    """A format needs at least one channel."""
    with pytest.raises(ValueError, match="channel count"):
        AudioFormat(sample_rate=48_000, bit_depth=16, channels=0).resolve_av_format()


def test_convert_s32_to_s24_drops_least_significant_byte_python_impl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """s32->s24 conversion should drop the LSB byte per sample in fallback path."""
    monkeypatch.setattr(format_module, "_get_numpy", lambda: None)
    converted = _convert_s32_to_s24(S32_SAMPLES)
    assert converted == _expected_s24_samples()


def test_convert_s24_to_s32_inserts_zero_lsb_python_impl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """s24->s32 conversion should expand samples into left-aligned PyAV representation."""
    monkeypatch.setattr(format_module, "_get_numpy", lambda: None)
    converted = _convert_s24_to_s32(_expected_s24_samples())
    assert converted == _expected_s32_from_s24_samples()


def test_convert_s32_to_s24_drops_least_significant_byte_numpy_impl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """s32->s24 conversion should drop the LSB byte per sample in NumPy path."""
    np = pytest.importorskip("numpy")
    monkeypatch.setattr(format_module, "_get_numpy", lambda: np)
    converted = _convert_s32_to_s24(S32_SAMPLES)
    assert converted == _expected_s24_samples()


def test_convert_s24_to_s32_inserts_zero_lsb_numpy_impl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """s24->s32 conversion should expand samples into left-aligned PyAV representation."""
    np = pytest.importorskip("numpy")
    monkeypatch.setattr(format_module, "_get_numpy", lambda: np)
    converted = _convert_s24_to_s32(_expected_s24_samples())
    assert converted == _expected_s32_from_s24_samples()


def test_convert_s32_to_s24_rejects_invalid_length() -> None:
    """Invalid byte lengths must be rejected."""
    with pytest.raises(ValueError, match="multiple of 4"):
        _convert_s32_to_s24(b"\x00\x01\x02")


def test_convert_s24_to_s32_rejects_invalid_length() -> None:
    """Invalid byte lengths must be rejected."""
    with pytest.raises(ValueError, match="multiple of 3"):
        _convert_s24_to_s32(b"\x00\x01")


def test_convert_s24_to_s32_round_trips_with_s32_to_s24() -> None:
    """Round-tripping packed s24 through PyAV's s32 layout should preserve wire bytes."""
    converted = _convert_s32_to_s24(_convert_s24_to_s32(_expected_s24_samples()))
    assert converted == _expected_s24_samples()
