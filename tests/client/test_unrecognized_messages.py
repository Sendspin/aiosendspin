"""The client ignores server messages it does not recognise once activated."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import WSMessage, WSMsgType

from aiosendspin.client.connection import SendspinConnection
from aiosendspin.models.core import ServerActivatePayload
from aiosendspin.models.player import (
    ClientHelloPlayerSupport,
    SupportedAudioFormat,
    pack_player_audio_header,
)
from aiosendspin.models.types import Activity, AudioCodec, Roles
from aiosendspin.noise.trust_store import PskCategory, ResolvedPsk
from aiosendspin.noise.wire import EncryptedWebSocket
from tests.conftest import make_sdk_client
from tests.noise.conftest import FakeWebSocket, make_paired_sessions


async def _activated_connection() -> tuple[SendspinConnection, list[bytes]]:
    """Return an activated player connection with no stream, and the audio it delivers."""
    client = make_sdk_client(
        client_name="Test Client",
        roles=[Roles.PLAYER],
        player_support=ClientHelloPlayerSupport(
            supported_formats=[
                SupportedAudioFormat(
                    codec=AudioCodec.PCM, sample_rate=48_000, bit_depth=16, channels=2
                )
            ],
            buffer_capacity=100_000,
        ),
    )
    audio: list[bytes] = []
    client.add_audio_chunk_listener(lambda _ts, data, _fmt, _ahead: audio.append(data))
    conn = SendspinConnection(client)
    conn._noise_psk = ResolvedPsk("id", b"\x00" * 32, PskCategory.LONG_TERM)  # noqa: SLF001
    conn._ws = MagicMock(closed=False, send_str=AsyncMock(), close=AsyncMock())  # noqa: SLF001
    conn._connected = True  # noqa: SLF001
    assert (
        await conn._apply_activation(  # noqa: SLF001
            ServerActivatePayload(activities=[Activity.PLAYBACK], active_roles=[Roles.PLAYER.value])
        )
        is None
    )
    conn.disconnect = AsyncMock()  # type: ignore[method-assign]
    return conn, audio


async def test_unknown_json_type_is_ignored() -> None:
    """A well-formed message of an unknown type is ignored and the connection stays open."""
    conn, _ = await _activated_connection()

    await conn._handle_ws_message(  # noqa: SLF001
        WSMessage(WSMsgType.TEXT, '{"type":"server/from-the-future","payload":{"x":1}}', "")
    )

    conn.disconnect.assert_not_awaited()  # type: ignore[attr-defined]
    assert conn._protocol_error_task is None  # noqa: SLF001
    assert conn.connected


@pytest.mark.parametrize("message_id", [24, 100, 191])
async def test_unimplemented_binary_id_is_ignored(message_id: int) -> None:
    """A binary ID the client does not implement is ignored and the connection stays open."""
    conn, _ = await _activated_connection()

    await conn._handle_ws_message(  # noqa: SLF001
        WSMessage(WSMsgType.BINARY, bytes([message_id]) + b"data", "")
    )

    conn.disconnect.assert_not_awaited()  # type: ignore[attr-defined]
    assert conn._protocol_error_task is None  # noqa: SLF001


async def test_implemented_binary_id_with_inactive_stream_is_dropped() -> None:
    """Audio for an active player role without a stream is dropped, not treated as unknown."""
    conn, audio = await _activated_connection()

    await conn._handle_ws_message(  # noqa: SLF001
        WSMessage(WSMsgType.BINARY, pack_player_audio_header(1, 0) + b"\x00\x00\x00\x00", "")
    )

    assert audio == []
    conn.disconnect.assert_not_awaited()  # type: ignore[attr-defined]
    assert conn._protocol_error_task is None  # noqa: SLF001


@pytest.mark.parametrize("type_byte", [2, 3])
async def test_reserved_binary_id_is_ignored(type_byte: int) -> None:
    """A binary ID 2 or 3 is delivered by the transport, ignored, and the connection stays open."""
    conn, _ = await _activated_connection()
    initiator, responder = make_paired_sessions()
    raw = FakeWebSocket()
    ws = EncryptedWebSocket(raw, responder)
    await raw.push(WSMessage(WSMsgType.BINARY, initiator.encrypt(bytes([type_byte, 4, 5])), ""))

    await conn._handle_ws_message(await ws.receive())  # noqa: SLF001

    conn.disconnect.assert_not_awaited()  # type: ignore[attr-defined]
    assert conn._protocol_error_task is None  # noqa: SLF001


class _RehandshakeWs:
    """Yields one Noise handshake message, then answers ``receive`` with ``reply``."""

    closed = False

    def __init__(self, reply: WSMessage) -> None:
        self._pending = [
            WSMessage(WSMsgType.TEXT, '{"type":"noise/handshake","payload":{"data":"AA"}}', "")
        ]
        self._reply = reply

    def __aiter__(self) -> _RehandshakeWs:
        return self

    async def __anext__(self) -> WSMessage:
        if not self._pending:
            raise StopAsyncIteration
        return self._pending.pop()

    async def receive(self) -> WSMessage:
        return self._reply


@pytest.mark.parametrize(
    "reply",
    [
        WSMessage(WSMsgType.TEXT, '{"type":"server/from-the-future","payload":{}}', ""),
        WSMessage(WSMsgType.TEXT, '{"type":"server/time","payload":{}}', ""),
        WSMessage(WSMsgType.BINARY, b"\x04audio", ""),
    ],
    ids=["unknown-type", "other-known-type", "binary"],
)
async def test_rehandshake_closes_on_anything_but_server_activate(reply: WSMessage) -> None:
    """After a re-handshake, a message other than server/activate closes the connection."""
    conn, _ = await _activated_connection()
    conn._ws = _RehandshakeWs(reply)  # type: ignore[assignment]  # noqa: SLF001
    conn._rehandshake = AsyncMock()  # type: ignore[method-assign]  # noqa: SLF001
    conn._handle_server_activate = AsyncMock()  # type: ignore[method-assign]  # noqa: SLF001

    await conn._reader_loop()  # noqa: SLF001

    conn._rehandshake.assert_awaited_once()  # noqa: SLF001
    conn._handle_server_activate.assert_not_awaited()  # noqa: SLF001
    conn.disconnect.assert_awaited()  # type: ignore[attr-defined]
