"""The 'playback' activity a source start command needs."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aiosendspin.models.core import ClientStatePayload
from aiosendspin.models.source import SourceStatePayload
from aiosendspin.models.types import Activity, PlaybackStateType, Roles
from aiosendspin.noise.trust_store import PskCategory
from aiosendspin.server.connection import SendspinConnection
from aiosendspin.server.roles.source.v1 import SourceV1Role
from tests.server.test_role_activation import (
    _PLAYER_STATE,
    _client,
    _connect,
    _drain_priority,
    _FakeTransport,
    _hello,
)


async def _source(
    roles: list[str], *, category: PskCategory = PskCategory.LONG_TERM
) -> tuple[SendspinConnection, _FakeTransport, SourceV1Role]:
    conn, fake = await _connect(
        _hello(roles),
        send_state=False,
        category=category,
        approved_roles=frozenset({"source"}),
    )
    player_state = _PLAYER_STATE if Roles.PLAYER.value in roles else None
    await conn._handle_client_state(  # noqa: SLF001
        ClientStatePayload(available=True, player=player_state, source=SourceStatePayload())
    )
    fake.sent.clear()
    role = _client(conn).role("source@v1")
    assert isinstance(role, SourceV1Role)
    return conn, fake, role


async def _send_all(conn: SendspinConnection, fake: _FakeTransport) -> list[dict[str, object]]:
    """Put queued messages on the wire in the writer's order: priority first, then role queues."""
    await _drain_priority(conn, fake)
    while (entry := conn._peek_ready_entry()) is not None:  # noqa: SLF001
        await conn._send_role_entry(fake, entry, 0)  # type: ignore[arg-type]  # noqa: SLF001
    return [message["payload"] for message in fake.sent_payloads()]


@pytest.mark.asyncio
async def test_source_only_client_gets_playback_ahead_of_start_until_stop() -> None:
    """An idle source-only client is activated for playback before the start, and back after."""
    conn, fake, role = await _source(["source@v1"])
    assert conn._declared_activities == []  # noqa: SLF001

    role.request_start()
    assert await _send_all(conn, fake) == [
        {"activities": [Activity.PLAYBACK.value]},
        {"source": {"command": "start"}},
    ]

    fake.sent.clear()
    role.request_stop()
    assert await _send_all(conn, fake) == [{"activities": []}, {"source": {"command": "stop"}}]


@pytest.mark.asyncio
async def test_group_stopping_keeps_playback_for_a_wanted_source() -> None:
    """A source still asked to stream keeps 'playback' when its group stops."""
    conn, fake, role = await _source([Roles.PLAYER.value, "source@v1"])
    conn._subscribe_activity_events()  # noqa: SLF001
    group = _client(conn).group
    group._set_playback_state(PlaybackStateType.PLAYING)  # noqa: SLF001
    role.request_start()

    group._set_playback_state(PlaybackStateType.STOPPED)  # noqa: SLF001

    assert conn._declared_activities == [Activity.PLAYBACK]  # noqa: SLF001
    sent = await _send_all(conn, fake)
    assert sent[-1] == {"source": {"command": "start"}}


@pytest.mark.asyncio
async def test_start_requested_during_pairing_waits_for_playback() -> None:
    """A start never goes out under a pairing activation without 'playback', only after it."""
    conn, fake, role = await _source(["source@v1"], category=PskCategory.SENTINEL)
    conn._in_pairing = True  # noqa: SLF001
    conn._pairing_activities = [Activity.PAIRING]  # noqa: SLF001

    role.request_start()
    assert await _send_all(conn, fake) == []

    with patch.object(conn, "_resume_writer"):
        await conn._leave_pairing()  # noqa: SLF001
    sent = await _send_all(conn, fake)
    assert sent[0]["activities"] == [Activity.PLAYBACK.value]
    assert sent[-1] == {"source": {"command": "start"}}
