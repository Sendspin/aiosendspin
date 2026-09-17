"""Server-side format validation for player roles.

Provides utilities to check if the server can encode a client's requested format
based on codec-specific constraints (sample rates, bit depths, channels).
"""

from __future__ import annotations

from aiosendspin.audio.codecs import flac_encoder_available, opus_available
from aiosendspin.models import AudioCodec
from aiosendspin.models.player import SupportedAudioFormat
from aiosendspin.server.roles.player.audio_transformers import FlacEncoder, OpusEncoder

PCM_BIT_DEPTHS: frozenset[int] = frozenset({16, 24, 32})
FLAC_BIT_DEPTHS: frozenset[int] = FlacEncoder.VALID_BIT_DEPTHS
OPUS_BIT_DEPTHS: frozenset[int] = frozenset({16})

# Supported channel counts — matches the layout map in server/audio.py.
VALID_CHANNELS: frozenset[int] = frozenset({1, 2, 3, 4, 5, 6, 7, 8, 10})


def can_encode_format(fmt: SupportedAudioFormat) -> bool:
    """Check if the server can encode this format.

    Validates against server encoding constraints:
    - PCM bit depth: 16, 24, or 32; channels: 1-8 or 10 (up to 9.1)
    - FLAC bit depth: 16 or 24; channels: 1-8 or 10 (up to 9.1); also requires PyAV with
      FFmpeg's FLAC encoder
    - Opus bit depth: 16 only; channels: 1 or 2 only, as multichannel not yet implemented;
      also requires PyAV built with libopus
    - Opus: sample rate must be one of 8k, 12k, 16k, 24k, 48k
    - FLAC/PCM: any sample rate

    Args:
        fmt: The format to validate.

    Returns:
        True if the server can encode this format.
    """
    if fmt.sample_rate <= 0:
        return False
    if fmt.channels not in VALID_CHANNELS:
        return False
    codec = fmt.codec.value
    if codec == AudioCodec.OPUS.value:
        if not opus_available():
            return False
        if fmt.channels not in {1, 2}:
            return False
        if fmt.bit_depth not in OPUS_BIT_DEPTHS:
            return False
        return fmt.sample_rate in OpusEncoder.VALID_SAMPLE_RATES
    if codec == AudioCodec.FLAC.value:
        if not flac_encoder_available():
            return False
        return fmt.bit_depth in FLAC_BIT_DEPTHS
    if codec == AudioCodec.PCM.value:
        return fmt.bit_depth in PCM_BIT_DEPTHS
    return False


def filter_encodable_formats(
    formats: list[SupportedAudioFormat],
) -> list[SupportedAudioFormat]:
    """Filter to server-encodable formats, preserving client priority order.

    Args:
        formats: Client's supported formats in priority order.

    Returns:
        Formats the server can encode, maintaining the client's priority order.
    """
    return [fmt for fmt in formats if can_encode_format(fmt)]
