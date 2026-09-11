"""Regression tests for buffer registration timing (queue-time vs send-time)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.models.types import AudioCodec, PlayerCommand, Roles
from aiosendspin.server.audio_transformers import TransformerPool
from aiosendspin.server.channels import MAIN_CHANNEL
from aiosendspin.server.client import SendspinClient
from aiosendspin.server.clock import LoopClock
from aiosendspin.server.connection import SendspinConnection
from aiosendspin.server.roles.base import AudioChunk


@dataclass(slots=True)
class _DummyServer:
    loop: Any
    clock: Any
    id: str = "srv"
    name: str = "server"

    def get_or_create_client(self, client_id: str) -> Any:  # noqa: ARG002
        raise AssertionError("unexpected get_or_create_client() in this test")

    def is_external_player(self, client_id: str) -> bool:  # noqa: ARG002
        return False

    def _signal_client_connected(self, client_id: str) -> None:
        pass

    def _signal_client_disconnected(self, client_id: str, goodbye_reason: object = None) -> None:
        pass


class _DummyGroup:
    def __init__(self, clients: list[Any], group_id: str = "g1") -> None:
        self.clients = clients
        self.group_id = group_id
        self.transformer_pool = TransformerPool()

    group_name = "dummy group"

    def _publish_if_name_changed(self, previous: str) -> None:  # noqa: ARG002
        return

    def on_client_connected(self, client: Any) -> None:  # noqa: ARG002
        return

    def _register_client_events(self, client: Any) -> None:  # noqa: ARG002
        return

    def group_role(self, family: str) -> None:  # noqa: ARG002
        return None

    def get_channel_for_player(self, player_id: str) -> UUID:  # noqa: ARG002
        return MAIN_CHANNEL


@pytest.mark.asyncio
async def test_buffer_tracker_does_not_backpressure_until_send() -> None:  # noqa: PLR0915
    """
    Backpressure must reflect bytes that have actually left the server.

    This test blocks websocket send to create a gap between "queued" and "sent".
    Before the send completes, BufferTracker must remain at 0 bytes (no backpressure).
    """
    loop = asyncio.get_running_loop()
    clock = LoopClock(loop)
    server = _DummyServer(loop=loop, clock=clock)

    send_event = asyncio.Event()
    wsock = MagicMock()
    wsock.closed = False
    wsock.send_str = AsyncMock()

    async def slow_send_bytes(_: bytes) -> None:
        await send_event.wait()

    wsock.send_bytes = AsyncMock(side_effect=slow_send_bytes)

    conn = SendspinConnection(server, wsock_client=wsock)
    conn._transport = wsock  # noqa: SLF001
    await conn._setup_connection()  # noqa: SLF001
    # Streaming player: initial state already received.
    conn._initial_state_received = True  # noqa: SLF001
    conn._writer_task = asyncio.create_task(conn._writer())  # noqa: SLF001

    group = _DummyGroup(clients=[])
    client = SendspinClient(server, client_id="p1")
    client._group = group  # noqa: SLF001
    group.clients.append(client)

    hello = type("Hello", (), {})()
    hello.client_id = "p1"
    hello.name = "p1"
    hello.player_support = ClientHelloPlayerSupport(
        supported_formats=[
            SupportedAudioFormat(
                codec=AudioCodec.PCM,
                channels=2,
                sample_rate=48000,
                bit_depth=16,
            )
        ],
        buffer_capacity=100,
        supported_commands=[PlayerCommand.VOLUME],
    )
    hello.artwork_support = None
    hello.visualizer_support = None

    client.attach_connection(
        conn,
        client_info=hello,
        negotiated_roles=[Roles.PLAYER.value],
        active_roles=[Roles.PLAYER.value],
    )
    client.mark_connected()
    conn._client = client  # noqa: SLF001

    role = client.role("player@v1")
    assert role is not None
    role._stream_started = True  # noqa: SLF001
    buffer_tracker = role.get_buffer_tracker()
    assert buffer_tracker is not None

    now_us = clock.now_us()
    chunk = AudioChunk(
        timestamp_us=now_us + 100_000,
        data=b"x" * 100,
        byte_count=100,
        duration_us=100_000,
    )

    try:
        # Use the hook-based on_audio_chunk method
        sent = role.on_audio_chunk(chunk)
        assert sent is None

        # Regression assertion: queued-but-not-sent data must not cause backpressure.
        assert buffer_tracker.buffered_bytes == 0
        assert buffer_tracker.time_until_capacity(1) == 0

        send_event.set()
        for _ in range(50):
            if wsock.send_bytes.called:
                break
            await asyncio.sleep(0)

        assert wsock.send_bytes.call_count == 1
        assert buffer_tracker.buffered_bytes == 100
    finally:
        send_event.set()
        await conn.disconnect(retry_connection=False)
