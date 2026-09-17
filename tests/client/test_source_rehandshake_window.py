"""A re-handshake's quiet period runs to the new server/activate, not to the exchange's end."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from aiosendspin.client.connection import SendspinConnection
from aiosendspin.client.source import SourceCapture
from aiosendspin.models.core import ServerActivatePayload
from aiosendspin.models.player import SupportedAudioFormat
from aiosendspin.models.source import ClientHelloSourceFeatures, ClientHelloSourceSupport
from aiosendspin.models.types import Activity, AudioCodec, Roles, SignalState
from aiosendspin.noise.trust_store import PskCategory, ResolvedPsk
from tests.conftest import make_sdk_client, sine_pcm_16bit


class _Ws:
    closed = False
    session = MagicMock()

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.sent_bytes: list[bytes] = []

    async def send_str(self, data: str) -> None:
        self.sent.append(data)

    async def send_bytes(self, data: bytes) -> None:
        self.sent_bytes.append(data)

    async def close(self, **_: object) -> None:
        self.closed = True


def _streaming_source_connection(ws: _Ws) -> SendspinConnection:
    """Build a paired connection with source@v1 active and its stream open."""
    client = make_sdk_client(
        client_name="src",
        roles=[Roles.SOURCE],
        source_support=ClientHelloSourceSupport(
            features=ClientHelloSourceFeatures(line_sense=True)
        ),
    )
    conn = SendspinConnection(client)
    conn._ws = ws  # type: ignore[assignment]  # noqa: SLF001
    conn._connected = True  # noqa: SLF001
    conn._server_id = "server-0"  # noqa: SLF001
    conn._handshake_hash = b"hash"  # noqa: SLF001
    conn._noise_psk = ResolvedPsk("paired", b"\x00" * 32, PskCategory.LONG_TERM)  # noqa: SLF001
    conn._active_roles = [Roles.SOURCE.value]  # noqa: SLF001
    conn._source_stream_active = True  # noqa: SLF001
    conn._source_start_authorized = True  # noqa: SLF001
    conn.is_time_synchronized = lambda: True  # type: ignore[method-assign]
    return conn


def _downgrade(
    conn: SendspinConnection,
    monkeypatch: pytest.MonkeyPatch,
    *,
    category: PskCategory,
    active_roles: list[str],
    unpaired_access: bool = True,
) -> ResolvedPsk:
    """Arm a re-handshake to ``category`` whose server/activate carries ``active_roles``."""
    downgraded = ResolvedPsk("unpaired", b"\x01" * 32, category)

    async def _rehandshake(*_: object, **__: object) -> MagicMock:
        return MagicMock(psk=downgraded, handshake_hash=b"next")

    monkeypatch.setattr("aiosendspin.client.connection.run_rehandshake_client", _rehandshake)

    async def _receive_activate() -> ServerActivatePayload:
        return ServerActivatePayload(activities=[Activity.PLAYBACK], active_roles=active_roles)

    async def _unpaired_access_enabled() -> bool:
        return unpaired_access

    conn._receive_server_activate = _receive_activate  # type: ignore[method-assign]  # noqa: SLF001
    conn._unpaired_access_enabled = _unpaired_access_enabled  # type: ignore[method-assign]  # noqa: SLF001
    return downgraded


def _probe_quiet_period(conn: SendspinConnection, probe: Any) -> None:
    """Run ``probe`` in the window between the key swap and the activation being applied."""
    original = conn._handle_server_activate  # noqa: SLF001

    async def _wrapped(payload: ServerActivatePayload, **kwargs: Any) -> None:
        await probe()
        await original(payload, **kwargs)

    conn._handle_server_activate = _wrapped  # type: ignore[method-assign]  # noqa: SLF001


def _source_capture(conn: SendspinConnection) -> SourceCapture:
    """Build a started PCM capture bound to ``conn``'s open source stream."""
    capture = SourceCapture(
        conn._client,  # noqa: SLF001
        conn,
        SupportedAudioFormat(codec=AudioCodec.PCM, channels=2, sample_rate=48000, bit_depth=16),
    )
    capture._started = True  # noqa: SLF001
    return capture


@pytest.mark.parametrize("category", [PskCategory.SENTINEL, PskCategory.PAIRING])
async def test_no_source_message_reaches_the_wire_in_the_quiet_period(
    monkeypatch: pytest.MonkeyPatch, category: PskCategory
) -> None:
    """Every source send path is suppressed until the new server/activate is applied."""
    ws = _Ws()
    conn = _streaming_source_connection(ws)
    # The server drops source from the unpaired session.
    downgraded = _downgrade(conn, monkeypatch, category=category, active_roles=[])

    async def _probe() -> None:
        assert conn._noise_psk is downgraded  # noqa: SLF001
        await conn.send_source_chunk(b"audio", timestamp_us=1)
        for refused in (
            conn.send_client_stream_start(
                codec=AudioCodec.PCM,
                sample_rate=48000,
                channels=2,
                bit_depth=16,
                codec_header=None,
            ),
            conn.send_client_stream_end(),
        ):
            with pytest.raises(RuntimeError, match="in-band exchange"):
                await refused
        await conn.send_source_signal(SignalState.PRESENT)
        assert ws.sent == []
        assert ws.sent_bytes == []

    _probe_quiet_period(conn, _probe)

    await conn._handle_handshake("hs1")  # noqa: SLF001

    # The activation itself is sent normally: dropping source ends the stream.
    assert any("client-stream/end" in m for m in ws.sent)
    assert not any("client-stream/start" in m for m in ws.sent)
    assert ws.sent_bytes == []


async def test_source_resumes_once_the_activation_re_admits_the_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-admitting source@v1 on the new session releases the send paths again."""
    ws = _Ws()
    conn = _streaming_source_connection(ws)
    _downgrade(conn, monkeypatch, category=PskCategory.SENTINEL, active_roles=[Roles.SOURCE.value])

    await conn._handle_handshake("hs1")  # noqa: SLF001

    assert conn._active_roles == [Roles.SOURCE.value]  # noqa: SLF001
    await conn.send_source_chunk(b"audio", timestamp_us=1)
    assert ws.sent_bytes != []


async def test_capture_refused_in_the_quiet_period_still_ends_its_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stop() the quiet period refuses leaves the stream endable once source is retained."""
    ws = _Ws()
    conn = _streaming_source_connection(ws)
    _downgrade(conn, monkeypatch, category=PskCategory.SENTINEL, active_roles=[Roles.SOURCE.value])
    capture = _source_capture(conn)

    async def _probe() -> None:
        await capture.feed(sine_pcm_16bit(480))
        with pytest.raises(RuntimeError, match="in-band exchange"):
            await capture.stop()
        assert ws.sent == []
        assert ws.sent_bytes == []

    _probe_quiet_period(conn, _probe)

    await conn._handle_handshake("hs1")  # noqa: SLF001

    # The stream persists across the re-handshake, so the capture can still close it.
    assert conn.is_source_stream_active()
    await capture.stop()
    assert any("client-stream/end" in m for m in ws.sent)
    assert not conn.is_source_stream_active()
