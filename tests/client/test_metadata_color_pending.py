"""Tests for scheduled metadata and color updates in client server/state handling."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from aiosendspin.client.connection import SendspinConnection
from aiosendspin.client.time_sync import SendspinTimeFilter
from aiosendspin.clock import ManualClock
from aiosendspin.models.color import SessionUpdateColor
from aiosendspin.models.controller import ControllerStatePayload
from aiosendspin.models.core import ServerActivatePayload, ServerStatePayload, ServerTimePayload
from aiosendspin.models.metadata import SessionUpdateMetadata
from aiosendspin.models.types import Activity, RepeatMode

_NOW_US = 1_000_000
# Far enough ahead that the update stays pending for the whole test.
_LATER_US = _NOW_US + 60_000_000
_SOON_US = _NOW_US + 20_000


def _make_connection(*, synced: bool = True) -> tuple[SendspinConnection, MagicMock]:
    """Build a connection whose time filter maps server time 1:1 onto client time."""
    conn = SendspinConnection.__new__(SendspinConnection)
    client = MagicMock()
    client.clock = ManualClock(now_us_value=_NOW_US)
    client.loop = asyncio.get_running_loop()
    conn._client = client  # noqa: SLF001
    conn._time_filter = SendspinTimeFilter()  # noqa: SLF001
    if synced:
        conn._time_filter.update(0, 0, _NOW_US)  # noqa: SLF001
    conn._pending_state = {}  # noqa: SLF001
    conn._active_roles = ["metadata@v1", "color@v1"]  # noqa: SLF001
    return conn, client


async def _activate(conn: SendspinConnection, active_roles: list[str]) -> None:
    """Apply a server/activate that leaves `active_roles` active."""

    async def _apply(_payload: ServerActivatePayload) -> None:
        conn._active_roles = active_roles  # noqa: SLF001

    conn._apply_activation = _apply  # type: ignore[method-assign]  # noqa: SLF001
    conn._cancel_pairing_attempt = AsyncMock()  # type: ignore[method-assign]  # noqa: SLF001
    conn._resume_time_sync = MagicMock()  # type: ignore[method-assign]  # noqa: SLF001
    conn._activities = []  # noqa: SLF001
    conn._initial_state_sent = True  # noqa: SLF001
    await conn._handle_server_activate(  # noqa: SLF001
        ServerActivatePayload(activities=[Activity.PLAYBACK], active_roles=active_roles)
    )


async def test_future_metadata_is_pending_until_its_timestamp() -> None:
    """Future metadata is reported as scheduled, then as current once its time is reached."""
    conn, client = _make_connection()
    current = ServerStatePayload(metadata=SessionUpdateMetadata(timestamp=_NOW_US, title="Now"))
    scheduled = ServerStatePayload(metadata=SessionUpdateMetadata(timestamp=_SOON_US, title="Next"))
    conn._handle_server_state(current)  # noqa: SLF001

    conn._handle_server_state(scheduled)  # noqa: SLF001

    client.notify_scheduled_metadata.assert_called_once_with(scheduled)
    client.notify_metadata_callback.assert_called_once_with(current)
    assert conn._server_state == current  # noqa: SLF001

    await asyncio.sleep(0.05)

    client.notify_metadata_callback.assert_called_with(scheduled)
    assert conn._server_state == scheduled  # noqa: SLF001
    assert conn._pending_state == {}  # noqa: SLF001


async def test_past_color_applies_at_once() -> None:
    """A past or present color is current at once and discards a scheduled one."""
    conn, client = _make_connection()
    conn._handle_server_state(  # noqa: SLF001
        ServerStatePayload(color=SessionUpdateColor(timestamp=_LATER_US))
    )
    current = ServerStatePayload(color=SessionUpdateColor(timestamp=_NOW_US, primary=(1, 2, 3)))

    conn._handle_server_state(current)  # noqa: SLF001

    client.notify_color_callback.assert_called_once_with(current)
    assert conn._pending_state == {}  # noqa: SLF001


async def test_later_arrival_replaces_scheduled_update() -> None:
    """A newer scheduled update replaces the held one, even with an earlier timestamp."""
    conn, client = _make_connection()
    first = ServerStatePayload(color=SessionUpdateColor(timestamp=_LATER_US, primary=(1, 1, 1)))
    second = ServerStatePayload(color=SessionUpdateColor(timestamp=_SOON_US, primary=(2, 2, 2)))

    conn._handle_server_state(first)  # noqa: SLF001
    conn._handle_server_state(second)  # noqa: SLF001
    await asyncio.sleep(0.05)

    client.notify_color_callback.assert_called_once_with(second)
    assert conn._pending_state == {}  # noqa: SLF001


async def test_null_role_object_discards_scheduled_update() -> None:
    """A null role object clears the current state and the scheduled update at once."""
    conn, client = _make_connection()
    conn._handle_server_state(  # noqa: SLF001
        ServerStatePayload(metadata=SessionUpdateMetadata(timestamp=_SOON_US))
    )
    cleared = ServerStatePayload(metadata=None)

    conn._handle_server_state(cleared)  # noqa: SLF001
    await asyncio.sleep(0.05)

    client.notify_metadata_callback.assert_called_once_with(cleared)


async def test_other_role_objects_apply_while_one_is_scheduled() -> None:
    """Role objects without a future timestamp in the same message apply at once."""
    conn, client = _make_connection()
    controller = ControllerStatePayload(
        supported_commands=[], volume=50, muted=False, repeat=RepeatMode.OFF, shuffle=False
    )
    payload = ServerStatePayload(
        controller=controller,
        metadata=SessionUpdateMetadata(timestamp=_LATER_US),
        color=SessionUpdateColor(timestamp=_NOW_US),
    )

    conn._handle_server_state(payload)  # noqa: SLF001

    client.notify_controller_callback.assert_called_once_with(payload)
    client.notify_color_callback.assert_called_once_with(payload)
    client.notify_metadata_callback.assert_not_called()
    assert conn._server_state == ServerStatePayload(  # noqa: SLF001
        controller=controller, color=SessionUpdateColor(timestamp=_NOW_US)
    )


async def test_unsynchronized_client_applies_at_once() -> None:
    """Without a clock estimate, a timestamped update applies at once."""
    conn, client = _make_connection(synced=False)
    payload = ServerStatePayload(metadata=SessionUpdateMetadata(timestamp=_LATER_US))

    conn._handle_server_state(payload)  # noqa: SLF001

    client.notify_metadata_callback.assert_called_once_with(payload)
    client.notify_scheduled_metadata.assert_not_called()


async def test_clock_update_reschedules_scheduled_update() -> None:
    """A new clock estimate moves a scheduled update to its newly mapped local time."""
    conn, client = _make_connection()
    conn._active_roles = []  # noqa: SLF001
    payload = ServerStatePayload(metadata=SessionUpdateMetadata(timestamp=_LATER_US))
    conn._handle_server_state(payload)  # noqa: SLF001

    # The new estimate maps the update's timestamp to the local present.
    time_filter = MagicMock(count=1)
    time_filter.compute_client_time.side_effect = lambda server_us: server_us - _LATER_US + _NOW_US
    conn._time_filter = time_filter  # noqa: SLF001
    await conn._handle_server_time(  # noqa: SLF001
        ServerTimePayload(client_transmitted=0, server_received=0, server_transmitted=0)
    )
    await asyncio.sleep(0.05)

    client.notify_metadata_callback.assert_called_once_with(payload)


async def test_removed_role_discards_current_and_scheduled_state() -> None:
    """Removing a role discards its state and tells listeners; kept roles are unchanged."""
    conn, client = _make_connection()
    conn._handle_server_state(  # noqa: SLF001
        ServerStatePayload(
            metadata=SessionUpdateMetadata(timestamp=_SOON_US),
            color=SessionUpdateColor(timestamp=_NOW_US),
        )
    )
    client.notify_color_callback.reset_mock()

    await _activate(conn, ["color@v1"])
    await asyncio.sleep(0.05)

    client.notify_metadata_callback.assert_called_once_with(ServerStatePayload(metadata=None))
    client.notify_color_callback.assert_not_called()
    assert conn._pending_state == {}  # noqa: SLF001


async def test_removed_role_without_state_notifies_nothing() -> None:
    """Removing a role that never received state leaves listeners alone."""
    conn, client = _make_connection()

    await _activate(conn, [])

    client.notify_metadata_callback.assert_not_called()
    client.notify_color_callback.assert_not_called()
