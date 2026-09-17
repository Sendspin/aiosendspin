"""Tests for player-role format capability filtering."""

import pytest

from aiosendspin.models.player import SupportedAudioFormat
from aiosendspin.models.types import AudioCodec
from aiosendspin.server.roles.player.capabilities import (
    can_encode_format,
    filter_encodable_formats,
)


def test_can_encode_format_accepts_pcm_32_bit() -> None:
    """PCM 32-bit should be considered encodable."""
    fmt = SupportedAudioFormat(codec=AudioCodec.PCM, sample_rate=48_000, bit_depth=32, channels=2)
    assert can_encode_format(fmt)


def test_can_encode_format_accepts_flac_24_bit() -> None:
    """FLAC 24-bit should be considered encodable."""
    fmt = SupportedAudioFormat(codec=AudioCodec.FLAC, sample_rate=48_000, bit_depth=24, channels=2)
    assert can_encode_format(fmt)


def test_can_encode_format_rejects_flac_32_bit() -> None:
    """FLAC 32-bit should be rejected, since the server does not offer that depth."""
    fmt = SupportedAudioFormat(codec=AudioCodec.FLAC, sample_rate=48_000, bit_depth=32, channels=2)
    assert not can_encode_format(fmt)


def test_filter_encodable_formats_drops_flac_32_bit() -> None:
    """FLAC 32-bit should be filtered out while other formats keep client priority order."""
    flac_32 = SupportedAudioFormat(
        codec=AudioCodec.FLAC, sample_rate=48_000, bit_depth=32, channels=2
    )
    flac_24 = SupportedAudioFormat(
        codec=AudioCodec.FLAC, sample_rate=48_000, bit_depth=24, channels=2
    )
    pcm_16 = SupportedAudioFormat(
        codec=AudioCodec.PCM, sample_rate=48_000, bit_depth=16, channels=2
    )
    assert filter_encodable_formats([flac_32, flac_24, pcm_16]) == [flac_24, pcm_16]


def test_can_encode_format_rejects_opus_32_bit() -> None:
    """Opus with 32-bit input should be rejected."""
    fmt = SupportedAudioFormat(codec=AudioCodec.OPUS, sample_rate=48_000, bit_depth=32, channels=2)
    assert not can_encode_format(fmt)


def test_can_encode_format_rejects_invalid_opus_sample_rate() -> None:
    """Opus with unsupported sample rate should be rejected."""
    fmt = SupportedAudioFormat(codec=AudioCodec.OPUS, sample_rate=44_100, bit_depth=16, channels=2)
    assert not can_encode_format(fmt)


def test_can_encode_format_rejects_opus_without_libopus(monkeypatch: pytest.MonkeyPatch) -> None:
    """A PyAV build without libopus cannot encode Opus, however valid the format is."""
    monkeypatch.setattr(
        "aiosendspin.server.roles.player.capabilities.opus_available", lambda: False
    )
    fmt = SupportedAudioFormat(codec=AudioCodec.OPUS, sample_rate=48_000, bit_depth=16, channels=2)

    assert not can_encode_format(fmt)


def test_filter_encodable_formats_drops_opus_without_libopus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opus is filtered out without libopus, leaving the always-available codecs."""
    monkeypatch.setattr(
        "aiosendspin.server.roles.player.capabilities.opus_available", lambda: False
    )
    opus = SupportedAudioFormat(codec=AudioCodec.OPUS, sample_rate=48_000, bit_depth=16, channels=2)
    flac = SupportedAudioFormat(codec=AudioCodec.FLAC, sample_rate=48_000, bit_depth=16, channels=2)
    pcm = SupportedAudioFormat(codec=AudioCodec.PCM, sample_rate=48_000, bit_depth=16, channels=2)

    assert filter_encodable_formats([opus, flac, pcm]) == [flac, pcm]


def test_can_encode_format_rejects_flac_without_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a FLAC encoder, FLAC is not producible however valid the format is."""
    monkeypatch.setattr(
        "aiosendspin.server.roles.player.capabilities.flac_encoder_available", lambda: False
    )
    fmt = SupportedAudioFormat(codec=AudioCodec.FLAC, sample_rate=48_000, bit_depth=16, channels=2)

    assert not can_encode_format(fmt)


def test_filter_encodable_formats_drops_flac_without_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FLAC is filtered out without its encoder, leaving PCM."""
    monkeypatch.setattr(
        "aiosendspin.server.roles.player.capabilities.flac_encoder_available", lambda: False
    )
    flac = SupportedAudioFormat(codec=AudioCodec.FLAC, sample_rate=48_000, bit_depth=16, channels=2)
    pcm = SupportedAudioFormat(codec=AudioCodec.PCM, sample_rate=48_000, bit_depth=16, channels=2)

    assert filter_encodable_formats([flac, pcm]) == [pcm]
