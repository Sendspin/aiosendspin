"""Tests for droppable binary queue behavior on stream lifecycle messages."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Never
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiosendspin.models.core import (
    StreamClearMessage,
    StreamClearPayload,
    StreamEndMessage,
    StreamEndPayload,
)
from aiosendspin.server.clock import LoopClock
from aiosendspin.server.connection import SendspinConnection


class _DummyServer:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.clock = LoopClock(loop)
        self.id = "srv"
        self.name = "server"

    def get_or_create_client(self, client_id: str) -> Never:
        _ = client_id
        raise AssertionError("client/hello not used in this test")

    def is_external_player(self, client_id: str) -> bool:  # noqa: ARG002
        return False


@pytest.mark.asyncio
async def test_stream_end_drops_queued_binary_before_sending() -> None:
    """stream/end should drop queued binary payloads so it can take effect promptly."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop)

    wsock = MagicMock()
    wsock.closed = False
    wsock.send_bytes = AsyncMock()
    wsock.send_str = AsyncMock()
    conn = SendspinConnection(server, wsock_client=wsock)
    conn._transport = wsock  # noqa: SLF001
    conn._streaming_roles.add("player")  # noqa: SLF001

    writer = asyncio.create_task(conn._writer())  # noqa: SLF001
    try:
        # Queue a binary payload, then a stream/end message.
        conn.send_binary(
            b"\x04" + b"\x00" * 8 + b"audio",
            role="player",
            timestamp_us=0,
            message_type=4,
        )
        conn.send_role_message("player", StreamEndMessage(payload=StreamEndPayload(roles=None)))

        # Wait until stream/end is sent.
        for _ in range(50):
            if wsock.send_str.called:
                break
            await asyncio.sleep(0)

        assert wsock.send_str.call_count == 1
        # The queued binary should have been dropped, not sent.
        assert wsock.send_bytes.call_count == 0
    finally:
        writer.cancel()
        with suppress(asyncio.CancelledError):
            await writer


@pytest.mark.asyncio
async def test_stream_clear_drops_queued_binary_before_sending() -> None:
    """stream/clear should drop queued binary payloads to discard old buffered audio promptly."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop)

    wsock = MagicMock()
    wsock.closed = False
    wsock.send_bytes = AsyncMock()
    wsock.send_str = AsyncMock()
    conn = SendspinConnection(server, wsock_client=wsock)
    conn._transport = wsock  # noqa: SLF001

    writer = asyncio.create_task(conn._writer())  # noqa: SLF001
    try:
        conn.send_binary(
            b"\x04" + b"\x00" * 8 + b"audio",
            role="player",
            timestamp_us=0,
            message_type=4,
        )
        conn.send_role_message(
            "player",
            StreamClearMessage(payload=StreamClearPayload(roles=["player"])),
        )

        for _ in range(50):
            if wsock.send_str.called:
                break
            await asyncio.sleep(0)

        assert wsock.send_str.call_count == 1
        assert wsock.send_bytes.call_count == 0
    finally:
        writer.cancel()
        with suppress(asyncio.CancelledError):
            await writer
