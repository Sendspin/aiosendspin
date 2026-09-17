"""Encoder/decoder round-trip and header tests for the shared codecs."""

from __future__ import annotations

import struct
import types
from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest

from aiosendspin.audio.codecs import (
    create_decoder,
    create_encoder,
    decoded_bit_depth,
    flac_encoder_available,
    opus_available,
)
from aiosendspin.audio.format import _convert_s24_to_s32
from tests.conftest import sine_pcm_16bit


@pytest.fixture
def _uncached_opus_probe() -> Iterator[None]:
    """Run one opus_available() probe without reusing or leaving a cached result."""
    opus_available.cache_clear()
    yield
    opus_available.cache_clear()


@pytest.fixture
def _uncached_flac_probe() -> Iterator[None]:
    """Run one flac_encoder_available() probe without reusing or leaving a cached result."""
    flac_encoder_available.cache_clear()
    yield
    flac_encoder_available.cache_clear()


def _roundtrip(codec: str, pcm: bytes) -> bytes:
    enc = create_encoder(codec, sample_rate=48000, bit_depth=16, channels=2)
    frames = enc.process(pcm, 0, 0) + enc.flush()
    dec = create_decoder(
        codec, sample_rate=48000, bit_depth=16, channels=2, codec_header=enc.get_codec_header()
    )
    out = bytearray()
    for frame, _dur in frames:
        out += dec.decode(frame)
    out += dec.flush()
    return bytes(out)


def test_pcm_roundtrip_is_bit_exact() -> None:
    """Raw PCM survives encode+decode unchanged (the loopback confidence anchor)."""
    pcm = sine_pcm_16bit(48000)
    assert _roundtrip("pcm", pcm) == pcm


@pytest.mark.parametrize("codec", ["flac", "opus"])
def test_lossy_codec_roundtrip_recovers_audio(codec: str) -> None:
    """FLAC/Opus decode back to roughly the same amount of audio (structural check)."""
    pcm = sine_pcm_16bit(48000)
    out = _roundtrip(codec, pcm)
    # Within one decoder block of the input; never empty.
    assert len(out) >= len(pcm) - 4608 * 4
    assert len(out) <= len(pcm) + 4608 * 4


def test_codec_headers_match_their_wire_format() -> None:
    """Only flac carries a header, the `fLaC` marker plus STREAMINFO."""
    assert (
        create_encoder("pcm", sample_rate=48000, bit_depth=16, channels=2).get_codec_header()
        is None
    )
    flac = create_encoder("flac", sample_rate=48000, bit_depth=16, channels=2).get_codec_header()
    assert flac is not None
    assert flac[:4] == b"fLaC"
    # Opus is configured from the declared format, so it sends no header.
    assert (
        create_encoder("opus", sample_rate=48000, bit_depth=16, channels=2).get_codec_header()
        is None
    )


def test_unknown_codec_rejected() -> None:
    """Unsupported codec identifiers raise rather than silently no-op."""
    with pytest.raises(ValueError, match="Unsupported source codec"):
        create_encoder("mp3", sample_rate=48000, bit_depth=16, channels=2)
    with pytest.raises(ValueError, match="Unsupported source codec"):
        create_decoder("mp3", sample_rate=48000, bit_depth=16, channels=2, codec_header=None)


def test_flac_decoder_preserves_multichannel_frame_width() -> None:
    """FLAC decoding preserves every declared channel."""
    sample_rate = 48000
    channels = 6
    pcm = bytes(sample_rate * channels * 2)
    encoder = create_encoder("flac", sample_rate=sample_rate, bit_depth=16, channels=channels)
    frames = encoder.process(pcm, 0, 0) + encoder.flush()
    decoder = create_decoder(
        "flac",
        sample_rate=sample_rate,
        bit_depth=16,
        channels=channels,
        codec_header=encoder.get_codec_header(),
    )

    decoded = b"".join(decoder.decode(frame) for frame, _ in frames) + decoder.flush()

    assert len(pcm) <= len(decoded) <= len(pcm) + 4608 * channels * 2


@pytest.mark.parametrize(
    ("codec", "bit_depth", "expected"),
    [
        ("pcm", 8, 16),
        ("pcm", 16, 16),
        ("pcm", 24, 24),
        ("pcm", 32, 32),
        ("flac", 4, 16),
        ("flac", 12, 16),
        ("flac", 20, 24),
        ("flac", 28, 32),
        ("opus", 24, 16),
    ],
)
def test_decoded_bit_depth_rounds_up_to_a_packed_depth(
    codec: str, bit_depth: int, expected: int
) -> None:
    """Decoders emit the smallest of 16, 24 and 32 bits that holds the announced depth."""
    assert decoded_bit_depth(codec, bit_depth) == expected


def test_pcm_decoder_widens_8_bit_to_16_bit() -> None:
    """Signed 8-bit PCM becomes the same values scaled to little-endian 16-bit."""
    decoder = create_decoder("pcm", sample_rate=8000, bit_depth=8, channels=2, codec_header=None)

    assert decoder.decode(bytes([0x01, 0x7F, 0x80, 0xFF])) == struct.pack(
        "<4h", 0x0100, 0x7F00, -0x8000, -0x0100
    )


def test_flac_decoder_emits_packed_24_bit_for_a_20_bit_announcement() -> None:
    """A FLAC depth between 16 and 24 bits decodes to packed 24-bit PCM."""
    pcm = b"\x00\x10\x00\x00\xf0\xff" * 4800
    encoder = create_encoder("flac", sample_rate=48000, bit_depth=24, channels=2)
    frames = encoder.process(_convert_s24_to_s32(pcm), 0, 0) + encoder.flush()
    decoder = create_decoder(
        "flac",
        sample_rate=48000,
        bit_depth=20,
        channels=2,
        codec_header=encoder.get_codec_header(),
    )

    decoded = b"".join(decoder.decode(frame) for frame, _ in frames) + decoder.flush()

    assert decoded[: len(pcm)] == pcm


@pytest.mark.usefixtures("_uncached_opus_probe")
def test_opus_available_with_libopus(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opus is available when PyAV resolves the libopus encoder and decoder."""
    stub = types.SimpleNamespace(codec=types.SimpleNamespace(Codec=MagicMock()))
    monkeypatch.setattr("aiosendspin.audio.codecs._get_av", lambda: stub)

    assert opus_available() is True


@pytest.mark.usefixtures("_uncached_opus_probe")
def test_opus_unavailable_without_pyav(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opus is unavailable when PyAV itself cannot be imported."""

    def _no_av() -> types.ModuleType:
        raise ImportError("no av")

    monkeypatch.setattr("aiosendspin.audio.codecs._get_av", _no_av)

    assert opus_available() is False


@pytest.mark.usefixtures("_uncached_opus_probe")
def test_opus_unavailable_without_libopus(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opus is unavailable when PyAV's FFmpeg was built without libopus."""

    class _Codec:
        def __init__(self, name: str, mode: str) -> None:
            raise ValueError(f"unknown codec {name!r} for mode {mode!r}")

    stub = types.SimpleNamespace(codec=types.SimpleNamespace(Codec=_Codec))
    monkeypatch.setattr("aiosendspin.audio.codecs._get_av", lambda: stub)

    assert opus_available() is False


@pytest.mark.usefixtures("_uncached_flac_probe")
def test_flac_encoder_available_with_pyav() -> None:
    """FLAC encoding is available with the PyAV the test environment installs."""
    assert flac_encoder_available() is True


@pytest.mark.usefixtures("_uncached_flac_probe")
def test_flac_encoder_unavailable_without_pyav(monkeypatch: pytest.MonkeyPatch) -> None:
    """FLAC encoding is unavailable when PyAV itself cannot be imported."""

    def _no_av() -> types.ModuleType:
        raise ImportError("no av")

    monkeypatch.setattr("aiosendspin.audio.codecs._get_av", _no_av)

    assert flac_encoder_available() is False


@pytest.mark.usefixtures("_uncached_flac_probe")
def test_flac_encoder_unavailable_without_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    """FLAC encoding is unavailable when PyAV's FFmpeg lacks the FLAC encoder."""

    class _Codec:
        def __init__(self, name: str, mode: str) -> None:
            raise ValueError(f"unknown codec {name!r} for mode {mode!r}")

    stub = types.SimpleNamespace(codec=types.SimpleNamespace(Codec=_Codec))
    monkeypatch.setattr("aiosendspin.audio.codecs._get_av", lambda: stub)

    assert flac_encoder_available() is False
