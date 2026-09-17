"""End-to-end source loopback: client SourceCapture into the server SourceV1Role.

The strongest confidence anchor for the source role: PCM captured and streamed by
the client SDK is reconstructed bit-exact out of the server role's decoded handle.
"""

from __future__ import annotations

from typing import Any

import pytest

from aiosendspin.client.source import SourceCapture
from aiosendspin.models.core import ClientStatePayload
from aiosendspin.models.player import SupportedAudioFormat
from aiosendspin.models.source import (
    ClientHelloSourceFeatures,
    ClientHelloSourceSupport,
    ClientStreamStartPayload,
    ClientStreamStartSource,
    SourceStatePayload,
)
from aiosendspin.models.types import AudioCodec, BinaryMessageType
from aiosendspin.server.roles.source import SourceStreamStartedEvent
from aiosendspin.server.roles.source.v1 import SourceV1Role
from tests.conftest import sine_pcm_16bit


class _ServerSideConnection:
    def record_source_start(self) -> None:
        pass


class _ServerSideClient:
    """Minimal stand-in for the server's SendspinClient used by the role under test."""

    def __init__(self) -> None:
        self.events: list[Any] = []

        class _Info:
            source_support = ClientHelloSourceSupport(
                features=ClientHelloSourceFeatures(line_sense=True)
            )

        self.info = _Info()
        self.connection = _ServerSideConnection()
        self.available = True
        self.sent: list[Any] = []

    def _signal_event(self, event: Any) -> None:
        self.events.append(event)

    def awaits_role_state(self, _role_family: str) -> bool:
        return False

    def send_role_message(self, _family: str, message: Any) -> None:
        self.sent.append(message)


class _LoopbackConnection:
    """Client connection that forwards the source wire straight into a server role."""

    def __init__(self, role: SourceV1Role) -> None:
        self._role = role

    async def send_client_stream_start(
        self,
        *,
        codec: AudioCodec,
        sample_rate: int,
        channels: int,
        bit_depth: int,
        codec_header: str | None,
    ) -> None:
        self._role.on_client_stream_start(
            ClientStreamStartPayload(
                source=ClientStreamStartSource(
                    codec=codec,
                    channels=channels,
                    sample_rate=sample_rate,
                    bit_depth=bit_depth,
                    codec_header=codec_header,
                )
            )
        )

    async def send_source_chunk(self, frame: bytes, *, timestamp_us: int) -> None:
        self._role.on_binary_chunk(BinaryMessageType.SOURCE_AUDIO_CHUNK.value, timestamp_us, frame)

    async def send_client_stream_end(self) -> None:
        self._role.on_client_stream_end()

    def compute_source_timestamp(self, capture_timestamp_us: int) -> int:
        return capture_timestamp_us

    def is_time_synchronized(self) -> bool:
        return True

    def is_source_stream_active(self) -> bool:
        return self._role.stream_active


class _ClientSideClient:
    def now_us(self) -> int:
        return 1_000_000


@pytest.mark.parametrize("codec", [AudioCodec.PCM, AudioCodec.OPUS, AudioCodec.FLAC])
async def test_source_loopback(codec: AudioCodec) -> None:
    """Audio captured by the client is decoded back out of the server handle.

    PCM is asserted bit-exact; lossy codecs are checked structurally (the handle
    yields a comparable amount of audio).
    """
    server_client = _ServerSideClient()
    role = SourceV1Role(client=server_client)  # type: ignore[arg-type]
    role.on_connect()
    role.on_client_state(ClientStatePayload(available=True, source=SourceStatePayload()))
    role.request_start()
    conn = _LoopbackConnection(role)

    fmt = SupportedAudioFormat(codec=codec, channels=2, sample_rate=48000, bit_depth=16)
    capture = SourceCapture(_ClientSideClient(), conn, fmt)  # type: ignore[arg-type]

    pcm = sine_pcm_16bit(48000)
    await capture.start()
    await capture.feed(pcm, capture_timestamp_us=1_000_000)
    await capture.stop()

    handle = next(e for e in server_client.events if isinstance(e, SourceStreamStartedEvent)).handle
    received = bytearray()
    timestamps: list[int] = []
    async for chunk, ts in handle:
        received += chunk
        timestamps.append(ts)

    # No decoded chunk (including the flushed tail) is stamped at epoch 0; capture
    # was anchored at 1_000_000us.
    assert timestamps
    assert all(ts > 0 for ts in timestamps)

    if codec is AudioCodec.PCM:
        assert bytes(received) == pcm
    else:
        assert abs(len(received) - len(pcm)) <= 4608 * 4


async def test_flac_24_bit_source_loopback_preserves_packed_pcm() -> None:
    """Packed 24-bit FLAC capture survives the client and server codec path."""
    pytest.importorskip("av")
    server_client = _ServerSideClient()
    role = SourceV1Role(client=server_client)  # type: ignore[arg-type]
    role.on_connect()
    role.on_client_state(ClientStatePayload(available=True, source=SourceStatePayload()))
    role.request_start()
    conn = _LoopbackConnection(role)
    capture = SourceCapture(
        _ClientSideClient(),
        conn,
        SupportedAudioFormat(
            codec=AudioCodec.FLAC,
            channels=2,
            sample_rate=48000,
            bit_depth=24,
        ),
    )  # type: ignore[arg-type]
    pcm = b"\x00\x00\x10\x00\x00\xf0" * 48000

    await capture.start()
    await capture.feed(pcm, capture_timestamp_us=1_000_000)
    await capture.stop()

    handle = next(e for e in server_client.events if isinstance(e, SourceStreamStartedEvent)).handle
    received = b"".join([chunk async for chunk, _ in handle])
    assert received[: len(pcm)] == pcm
