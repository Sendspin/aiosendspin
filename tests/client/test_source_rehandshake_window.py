"""A PSK downgrade suspends the source stream until the next activation re-admits it."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from aiosendspin.client.connection import SendspinConnection
from aiosendspin.models.core import ServerActivatePayload
from aiosendspin.models.source import ClientHelloSourceFeatures, ClientHelloSourceSupport
from aiosendspin.models.types import Activity, AudioCodec, Roles
from aiosendspin.noise.trust_store import PskCategory, ResolvedPsk
from tests.conftest import make_sdk_client


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


async def _attempt_source_sends(conn: SendspinConnection) -> list[str]:
    """Try both source send paths, returning a label per attempt that reached the wire."""
    reached: list[str] = []
    try:
        await conn.send_source_chunk(b"audio", timestamp_us=1)
        reached.append("chunk")
    except RuntimeError:
        pass
    try:
        await conn.send_client_stream_start(
            codec=AudioCodec.PCM,
            sample_rate=48000,
            channels=2,
            bit_depth=16,
            codec_header=None,
        )
        reached.append("client-stream/start")
    except RuntimeError:
        pass
    return reached


@pytest.mark.parametrize("category", [PskCategory.SENTINEL, PskCategory.PAIRING])
async def test_source_is_refused_between_a_psk_downgrade_and_its_activation(
    monkeypatch: pytest.MonkeyPatch, category: PskCategory
) -> None:
    """Source cannot reach the wire under a downgraded PSK before the activation applies."""
    ws = _Ws()
    conn = _streaming_source_connection(ws)
    # The server drops source from the unpaired session.
    downgraded = _downgrade(conn, monkeypatch, category=category, active_roles=[])

    # Probe the window: the exchange has closed and the new PSK is in place, but the
    # activation that settles active_roles has not been applied yet.
    in_window: list[str] = []
    original = conn._handle_server_activate  # noqa: SLF001

    async def _probe(payload: ServerActivatePayload, **kwargs: Any) -> None:
        assert conn._noise_psk is downgraded  # noqa: SLF001
        assert not conn._exchange_in_progress  # noqa: SLF001
        in_window.extend(await _attempt_source_sends(conn))
        await original(payload, **kwargs)

    conn._handle_server_activate = _probe  # type: ignore[method-assign]  # noqa: SLF001

    await conn._handle_handshake("hs1")  # noqa: SLF001

    assert in_window == []
    assert ws.sent_bytes == []
    assert not any("client-stream/start" in m for m in ws.sent)


async def test_source_resumes_once_the_activation_re_admits_the_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-admitting source@v1 on the new session releases the send paths again."""
    ws = _Ws()
    conn = _streaming_source_connection(ws)
    _downgrade(
        conn,
        monkeypatch,
        category=PskCategory.SENTINEL,
        active_roles=[Roles.SOURCE.value],
    )

    await conn._handle_handshake("hs1")  # noqa: SLF001

    assert conn._active_roles == [Roles.SOURCE.value]  # noqa: SLF001
    await conn.send_source_chunk(b"audio", timestamp_us=1)
    assert ws.sent_bytes != []
