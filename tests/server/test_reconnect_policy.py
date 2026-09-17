"""Server-initiated reconnect policy: per-goodbye-reason and activity-aware."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from aiosendspin.models.types import Activity, GoodbyeReason
from aiosendspin.server.clock import LoopClock
from aiosendspin.server.connection import SendspinConnection


@dataclass(slots=True)
class _DummyServer:
    loop: asyncio.AbstractEventLoop
    clock: Any
    id: str = "srv"
    name: str = "server"


def _conn() -> SendspinConnection:
    loop = asyncio.get_running_loop()
    return SendspinConnection(
        _DummyServer(loop=loop, clock=LoopClock(loop)), wsock_client=MagicMock()
    )


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        (GoodbyeReason.RESTART, True),
        (GoodbyeReason.CONCURRENT_ATTEMPT, False),
        (GoodbyeReason.ANOTHER_SERVER, False),
        (GoodbyeReason.SHUTDOWN, False),
        (GoodbyeReason.USER_REQUEST, False),
        (GoodbyeReason.UNAUTHORIZED, False),
        (GoodbyeReason.PAIRING_REQUIRED, False),
        (GoodbyeReason.UNPAIRED, False),
    ],
)
@pytest.mark.asyncio
async def test_retry_decision_per_goodbye_reason(
    reason: GoodbyeReason,
    expected: bool,  # noqa: FBT001
) -> None:
    """Only restart warrants an auto-reconnect."""
    conn = _conn()
    conn._last_goodbye_reason = reason  # noqa: SLF001
    assert conn.should_retry_server_initiated_connection is expected


@pytest.mark.parametrize(
    ("activities", "expected"),
    [
        (None, True),
        ([], True),
        ([Activity.PLAYBACK], True),
        ([Activity.MANAGEMENT], False),
    ],
)
@pytest.mark.asyncio
async def test_no_goodbye_retry_depends_on_activities(
    activities: list[Activity] | None,
    expected: bool,  # noqa: FBT001
) -> None:
    """Without a goodbye, retry only when the connection was idle or carried playback."""
    conn = _conn()
    conn._declared_activities = activities  # noqa: SLF001
    assert conn.should_retry_server_initiated_connection is expected


@pytest.mark.asyncio
async def test_closing_never_retries() -> None:
    """A deliberately closing connection never reconnects."""
    conn = _conn()
    conn._closing = True  # noqa: SLF001
    conn._last_goodbye_reason = GoodbyeReason.RESTART  # noqa: SLF001
    assert conn.should_retry_server_initiated_connection is False
