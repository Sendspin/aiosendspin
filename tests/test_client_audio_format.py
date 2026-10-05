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
    pack_player_audio_header,
)
from aiosendspin.models.types import AudioCodec, Roles

from .conftest import make_sdk_client


def _player_support() -> ClientHelloPlayerSupport:
    return ClientHelloPlayerSupport(
        supported_formats=[
            SupportedAudioFormat(
                codec=AudioCodec.PCM, sample_rate=48_000, bit_depth=24, channels=2
            ),
            SupportedAudioFormat(
                codec=AudioCodec.FLAC, sample_rate=48_000, bit_depth=24, channels=2
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
@pytest.mark.parametrize(
    ("codec", "header", "audio_data"),
    [
        (AudioCodec.PCM, None, b"\x01\x00\x00\x02\x00\x00\x03\x00\x00\x04\x00\x00"),
        (AudioCodec.FLAC, b"flac-header", b"synthetic-flac-payload"),
    ],
)
async def test_audio_callback_receives_negotiated_codec_payload_and_format(
    codec: AudioCodec, header: bytes | None, audio_data: bytes
) -> None:
    """Binary dispatch preserves codec bytes and exposes their format and decoded header."""
    client = make_sdk_client(
        client_name="Test Client",
        roles=[Roles.PLAYER],
        player_support=_player_support(),
    )

    captured: list[tuple[int, bytes, AudioFormat, int]] = []
    client.add_audio_chunk_listener(
        lambda ts, payload, fmt, send_ahead: captured.append((ts, payload, fmt, send_ahead))
    )

    connection = SendspinConnection(client)
    await connection._handle_stream_start(  # noqa: SLF001
        StreamStartMessage(
            payload=StreamStartPayload(
                player=StreamStartPlayer(
                    codec=codec,
                    sample_rate=48_000,
                    channels=2,
                    bit_depth=24,
                    codec_header=base64.b64encode(header).decode() if header is not None else None,
                )
            )
        )
    )
    connection._handle_binary_message(  # noqa: SLF001
        pack_player_audio_header(123_456, 40_000) + audio_data
    )

    assert len(captured) == 1
    ts, payload, fmt, send_ahead = captured[0]
    assert ts == 123_456
    assert send_ahead == 40_000
    assert payload == audio_data
    assert fmt.codec == codec
    assert fmt.pcm_format.sample_rate == 48_000
    assert fmt.pcm_format.channels == 2
    assert fmt.pcm_format.bit_depth == 24
    assert fmt.codec_header == header
