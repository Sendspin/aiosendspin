"""Tests for client-side codec-aware audio format handling."""

from __future__ import annotations

import base64

import pytest

from aiosendspin.client.client import AudioFormat
from aiosendspin.client.connection import SendspinConnection
from aiosendspin.models.core import StreamStartMessage, StreamStartPayload
from aiosendspin.models.player import (
    ClientHelloPlayerSupport,
    StreamStartPlayer,
    SupportedAudioFormat,
)
from aiosendspin.models.types import AudioCodec, Roles

from .conftest import make_sdk_client


def _player_support() -> ClientHelloPlayerSupport:
    return ClientHelloPlayerSupport(
        supported_formats=[
            SupportedAudioFormat(
                codec=AudioCodec.PCM, sample_rate=48_000, bit_depth=16, channels=2
            ),
            SupportedAudioFormat(
                codec=AudioCodec.FLAC, sample_rate=48_000, bit_depth=16, channels=2
            ),
        ],
        buffer_capacity=100_000,
        supported_commands=[],
    )


def test_client_rejects_undecodable_advertised_codec() -> None:
    """Advertising a codec the SDK cannot decode fails fast instead of silent no-audio."""
    support = ClientHelloPlayerSupport(
        supported_formats=[
            SupportedAudioFormat(
                codec=AudioCodec.OPUS, sample_rate=48_000, bit_depth=16, channels=2
            ),
        ],
        buffer_capacity=100_000,
        supported_commands=[],
    )
    with pytest.raises(ValueError, match="cannot decode"):
        make_sdk_client(
            client_name="Test Client",
            roles=[Roles.PLAYER],
            player_support=support,
        )


@pytest.mark.asyncio
async def test_stream_start_flac_decodes_codec_header_and_notifies_audio_callbacks() -> None:
    """Client should expose codec-aware format and decoded FLAC header in callbacks."""
    client = make_sdk_client(
        client_name="Test Client",
        roles=[Roles.PLAYER],
        player_support=_player_support(),
    )

    header = b"flac-header"
    captured: list[tuple[int, bytes, AudioFormat, int]] = []
    client.add_audio_chunk_listener(
        lambda ts, payload, fmt, send_ahead: captured.append((ts, payload, fmt, send_ahead))
    )

    connection = SendspinConnection(client)
    await connection._handle_stream_start(  # noqa: SLF001
        StreamStartMessage(
            payload=StreamStartPayload(
                player=StreamStartPlayer(
                    codec=AudioCodec.FLAC,
                    sample_rate=48_000,
                    channels=2,
                    bit_depth=24,
                    codec_header=base64.b64encode(header).decode(),
                )
            )
        )
    )
    connection._handle_audio_chunk(123_456, b"abc", 40_000)  # noqa: SLF001

    assert len(captured) == 1
    ts, payload, fmt, send_ahead = captured[0]
    assert ts == 123_456
    assert send_ahead == 40_000
    assert payload == b"abc"
    assert fmt.codec == AudioCodec.FLAC
    assert fmt.pcm_format.sample_rate == 48_000
    assert fmt.pcm_format.channels == 2
    assert fmt.pcm_format.bit_depth == 24
    assert fmt.codec_header == header
