"""Tests for source-specific client API wiring."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from aiosendspin.client.client import SendspinClient
from aiosendspin.client.models import ServerInfo
from aiosendspin.models.player import SupportedAudioFormat
from aiosendspin.models.types import AudioCodec, Roles
from tests.conftest import make_sdk_client


class _FakeConnection:
    """Connection double exposing only what a source capture reads."""

    def __init__(self, source_codecs: frozenset[AudioCodec] | None) -> None:
        self.server_info = ServerInfo(
            server_id="server-1", name="Server", source_codecs=source_codecs
        )


def _source_client(source_codecs: frozenset[AudioCodec] | None) -> SendspinClient:
    client = SendspinClient.__new__(SendspinClient)
    client._roles = [Roles.SOURCE]  # noqa: SLF001
    client._admitted_connection = _FakeConnection(source_codecs)  # type: ignore[assignment]  # noqa: SLF001
    return client


def _opus_format() -> SupportedAudioFormat:
    return SupportedAudioFormat(codec=AudioCodec.OPUS, sample_rate=48_000, bit_depth=16, channels=2)


async def test_source_role_requires_source_support() -> None:
    """A source client must provide its versioned support object."""
    with pytest.raises(ValueError, match="source_support"):
        make_sdk_client(client_name="source", roles=[Roles.SOURCE])


async def test_send_available_uses_admitted_connection() -> None:
    """The public availability API delegates to the active connection."""
    client = SendspinClient.__new__(SendspinClient)
    connection = AsyncMock()
    client._admitted_connection = connection  # type: ignore[assignment]  # noqa: SLF001

    await client.send_available(available=False)

    connection.send_available.assert_awaited_once_with(available=False)


def test_source_capture_keeps_codec_the_server_accepts(monkeypatch: pytest.MonkeyPatch) -> None:
    """A listed codec this client can encode is streamed as requested."""
    monkeypatch.setattr("aiosendspin.client.client.opus_available", lambda: True)
    client = _source_client(frozenset({AudioCodec.FLAC, AudioCodec.PCM, AudioCodec.OPUS}))

    capture = client.create_source_capture(_opus_format())

    assert capture.codec is AudioCodec.OPUS


def test_source_capture_falls_back_when_codec_not_accepted() -> None:
    """An unlisted codec falls back to FLAC, keeping the requested PCM shape."""
    client = _source_client(frozenset({AudioCodec.FLAC, AudioCodec.PCM}))

    capture = client.create_source_capture(_opus_format())

    assert capture.codec is AudioCodec.FLAC
    assert capture.audio_format.sample_rate == 48_000
    assert capture.audio_format.bit_depth == 16
    assert capture.audio_format.channels == 2


def test_source_capture_falls_back_when_client_lacks_libopus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opus the server accepts is still downgraded when this client cannot encode it."""
    monkeypatch.setattr("aiosendspin.client.client.opus_available", lambda: False)
    client = _source_client(frozenset({AudioCodec.FLAC, AudioCodec.PCM, AudioCodec.OPUS}))

    capture = client.create_source_capture(_opus_format())

    assert capture.codec is AudioCodec.FLAC


def test_source_capture_falls_back_when_server_listed_no_codecs() -> None:
    """A server that sent no codec list is assumed to accept only flac and pcm."""
    client = _source_client(None)

    capture = client.create_source_capture(_opus_format())

    assert capture.codec is AudioCodec.FLAC
