"""A server source ``start`` authorizes exactly one client-stream/start."""

from __future__ import annotations

import asyncio
from typing import Literal
from unittest.mock import AsyncMock

import orjson
import pytest

from aiosendspin.client.connection import SendspinConnection
from aiosendspin.models.core import ServerActivatePayload, ServerCommandPayload
from aiosendspin.models.player import PlayerCommandPayload
from aiosendspin.models.source import (
    ClientHelloSourceFeatures,
    ClientHelloSourceSupport,
    SourceCommandServerPayload,
)
from aiosendspin.models.types import Activity, AudioCodec, PlayerCommand, Roles
from aiosendspin.noise.trust_store import PskCategory, ResolvedPsk
from tests.client.test_source_deactivation import _connection, _FakeClient, _FakeWs
from tests.conftest import make_sdk_client


class _CommandClient(_FakeClient):
    def __init__(self, ws: _FakeWs) -> None:
        self._ws = ws
        self.commands: list[tuple[ServerCommandPayload, list[str]]] = []

    def notify_server_command_callback(self, payload: ServerCommandPayload) -> None:
        # Record what was on the wire when the embedder was notified.
        self.commands.append((payload, [orjson.loads(m)["type"] for m in self._ws.sent]))


def _source_connection(
    *, stream_active: bool = False
) -> tuple[SendspinConnection, _FakeWs, _CommandClient]:
    ws = _FakeWs()
    conn = _connection(ws, active_roles=[Roles.SOURCE.value], stream_active=stream_active)
    client = _CommandClient(ws)
    conn._client = client  # type: ignore[assignment]  # noqa: SLF001
    return conn, ws, client


def _command(command: Literal["start", "stop"]) -> ServerCommandPayload:
    return ServerCommandPayload(source=SourceCommandServerPayload(command=command))


async def _open(conn: SendspinConnection) -> None:
    await conn.send_client_stream_start(
        codec=AudioCodec.PCM, sample_rate=48000, channels=2, bit_depth=16, codec_header=None
    )


def _sent_types(ws: _FakeWs) -> list[str]:
    return [orjson.loads(message)["type"] for message in ws.sent]


async def test_stream_start_requires_server_start() -> None:
    """No input stream opens before the server sends start."""
    conn, ws, _ = _source_connection()

    with pytest.raises(RuntimeError, match="server start"):
        await _open(conn)

    assert ws.sent == []
    assert not conn.is_source_stream_active()


async def test_start_authorizes_one_opening() -> None:
    """A start is consumed by the opening it authorizes."""
    conn, ws, client = _source_connection()

    await conn._handle_server_command(_command("start"))  # noqa: SLF001
    assert conn.is_source_start_authorized()
    await _open(conn)
    assert not conn.is_source_start_authorized()
    await conn.send_client_stream_end()

    with pytest.raises(RuntimeError, match="server start"):
        await _open(conn)
    assert _sent_types(ws) == ["client-stream/start", "client-stream/end"]
    assert [payload for payload, _ in client.commands] == [_command("start")]


async def test_repeated_start_does_not_accumulate() -> None:
    """A start while pending or open has no effect and authorizes no later reopening."""
    conn, _, client = _source_connection()

    await conn._handle_server_command(_command("start"))  # noqa: SLF001
    await conn._handle_server_command(_command("start"))  # noqa: SLF001
    await _open(conn)
    await conn._handle_server_command(_command("start"))  # noqa: SLF001
    await conn.send_client_stream_end()

    assert not conn.is_source_start_authorized()
    with pytest.raises(RuntimeError, match="server start"):
        await _open(conn)
    assert len(client.commands) == 1


async def test_stop_while_pending_prevents_opening() -> None:
    """A stop clears a pending start without sending client-stream/end."""
    conn, ws, client = _source_connection()
    await conn._handle_server_command(_command("start"))  # noqa: SLF001

    await conn._handle_server_command(_command("stop"))  # noqa: SLF001

    assert not conn.is_source_start_authorized()
    with pytest.raises(RuntimeError, match="server start"):
        await _open(conn)
    assert ws.sent == []
    assert client.commands[-1] == (_command("stop"), [])


async def test_stop_while_open_ends_stream_before_notifying() -> None:
    """A stop ends the open stream before the embedder sees it."""
    conn, _, client = _source_connection()
    await conn._handle_server_command(_command("start"))  # noqa: SLF001
    await _open(conn)

    await conn._handle_server_command(_command("stop"))  # noqa: SLF001

    assert not conn.is_source_stream_active()
    assert not conn.is_source_start_authorized()
    assert client.commands[-1] == (_command("stop"), ["client-stream/start", "client-stream/end"])


async def test_stop_ends_stream_whose_start_is_in_flight() -> None:
    """A stop arriving while the authorized client-stream/start is being sent still ends it."""
    conn, ws, _ = _source_connection()
    await conn._handle_server_command(_command("start"))  # noqa: SLF001
    release = asyncio.Event()
    send_str = ws.send_str

    async def _blocking_send(data: str) -> None:
        await release.wait()
        await send_str(data)

    ws.send_str = _blocking_send  # type: ignore[method-assign]
    open_task = asyncio.create_task(_open(conn))
    await asyncio.sleep(0)
    stop_task = asyncio.create_task(
        conn._handle_server_command(_command("stop"))  # noqa: SLF001
    )
    await asyncio.sleep(0)
    release.set()

    await open_task
    await stop_task

    assert _sent_types(ws) == ["client-stream/start", "client-stream/end"]
    assert not conn.is_source_stream_active()
    assert not conn.is_source_start_authorized()


async def test_stop_without_pending_or_open_is_ignored() -> None:
    """A stop with nothing to stop sends nothing and is not forwarded."""
    conn, ws, client = _source_connection()

    await conn._handle_server_command(_command("stop"))  # noqa: SLF001

    assert ws.sent == []
    assert client.commands == []


async def test_ignored_source_command_keeps_player_command() -> None:
    """Dropping an ignored source command still forwards the player command beside it."""
    conn, _, client = _source_connection()
    conn._reported_supported_commands = [PlayerCommand.MUTE]  # noqa: SLF001
    player = PlayerCommandPayload(command=PlayerCommand.MUTE, mute=True)

    await conn._handle_server_command(  # noqa: SLF001
        ServerCommandPayload(player=player, source=SourceCommandServerPayload(command="stop"))
    )

    assert [payload for payload, _ in client.commands] == [ServerCommandPayload(player=player)]


async def test_start_while_unavailable_is_ignored() -> None:
    """A start received while unavailable does not survive becoming available."""
    conn, ws, client = _source_connection()
    await conn.send_available(available=False)

    await conn._handle_server_command(_command("start"))  # noqa: SLF001
    await conn.send_available(available=True)

    assert not conn.is_source_start_authorized()
    assert client.commands == []
    with pytest.raises(RuntimeError, match="server start"):
        await _open(conn)
    assert _sent_types(ws) == ["client/state", "client/state"]


async def test_start_while_role_inactive_is_ignored() -> None:
    """A start for an inactive source role is not recorded."""
    conn, _, client = _source_connection()
    conn._active_roles = []  # noqa: SLF001

    await conn._handle_server_command(_command("start"))  # noqa: SLF001

    assert not conn.is_source_start_authorized()
    assert client.commands == []


async def test_unavailable_clears_pending_start() -> None:
    """Becoming unavailable clears a pending start."""
    conn, _, _ = _source_connection()
    await conn._handle_server_command(_command("start"))  # noqa: SLF001

    await conn.send_available(available=False)
    await conn.send_available(available=True)

    assert not conn.is_source_start_authorized()
    with pytest.raises(RuntimeError, match="server start"):
        await _open(conn)


async def test_role_removal_clears_pending_start() -> None:
    """After the source role is removed and reactivated, a new start is required."""
    conn, ws, _ = _source_connection()
    await conn._handle_server_command(_command("start"))  # noqa: SLF001

    await conn._apply_activation(  # noqa: SLF001
        ServerActivatePayload(activities=[Activity.PLAYBACK], active_roles=[])
    )
    await conn._apply_activation(  # noqa: SLF001
        ServerActivatePayload(activities=[Activity.PLAYBACK], active_roles=[Roles.SOURCE.value])
    )

    assert not conn.is_source_start_authorized()
    with pytest.raises(RuntimeError, match="server start"):
        await _open(conn)
    assert ws.sent == []
    await conn._handle_server_command(_command("start"))  # noqa: SLF001
    await _open(conn)
    assert _sent_types(ws) == ["client-stream/start"]


async def test_disconnect_clears_pending_start() -> None:
    """A pending start does not survive the connection."""
    client = make_sdk_client(
        client_name="source",
        roles=[Roles.SOURCE],
        source_support=ClientHelloSourceSupport(features=ClientHelloSourceFeatures()),
    )
    conn = SendspinConnection(client)
    conn._ws = AsyncMock(closed=False)  # noqa: SLF001
    conn._connected = True  # noqa: SLF001
    conn._noise_psk = ResolvedPsk("id", b"\x00" * 32, PskCategory.LONG_TERM)  # noqa: SLF001
    conn._active_roles = [Roles.SOURCE.value]  # noqa: SLF001
    await conn._handle_server_command(_command("start"))  # noqa: SLF001
    assert conn.is_source_start_authorized()

    await conn.disconnect()

    assert not conn.is_source_start_authorized()
