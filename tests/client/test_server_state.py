"""Tests for client handling of server/state role objects."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiosendspin.client.connection import SendspinConnection
from aiosendspin.client.time_sync import SendspinTimeFilter
from aiosendspin.clock import ManualClock
from aiosendspin.models.color import SessionUpdateColor
from aiosendspin.models.controller import ControllerStatePayload
from aiosendspin.models.core import ServerActivatePayload, ServerStatePayload, ServerTimePayload
from aiosendspin.models.metadata import SessionUpdateMetadata
from aiosendspin.models.types import Activity, RepeatMode, Roles
from aiosendspin.noise.trust_store import PskCategory, ResolvedPsk

_METADATA = SessionUpdateMetadata(timestamp=1, title="A")
_COLOR = SessionUpdateColor(timestamp=2, primary=(1, 2, 3))
_CONTROLLER = ControllerStatePayload(
    supported_commands=[], volume=10, muted=False, repeat=RepeatMode.OFF, shuffle=False
)
_STATE_ROLES = [Roles.METADATA.value, Roles.COLOR.value, Roles.CONTROLLER.value, "_acme@v1"]


def _make_connection() -> tuple[SendspinConnection, MagicMock]:
    conn = SendspinConnection.__new__(SendspinConnection)
    client = MagicMock()
    conn._client = client  # noqa: SLF001
    conn._time_filter = SendspinTimeFilter()  # noqa: SLF001
    conn._pending_state = {}  # noqa: SLF001
    return conn, client


def _activated_connection() -> tuple[SendspinConnection, MagicMock]:
    """Return a long-term connection holding state for every active state role."""
    conn, client = _make_connection()
    client.note_playback_activity = AsyncMock()
    conn._noise_psk = ResolvedPsk("id", b"\x00" * 32, PskCategory.LONG_TERM)  # noqa: SLF001
    conn._active_roles = list(_STATE_ROLES)  # noqa: SLF001
    conn._source_stream_active = False  # noqa: SLF001
    conn._unpaired_access_enabled = AsyncMock(return_value=False)  # type: ignore[method-assign]  # noqa: SLF001
    conn._handle_server_state(  # noqa: SLF001
        ServerStatePayload(
            metadata=_METADATA,
            color=_COLOR,
            controller=_CONTROLLER,
            application_objects={"_acme": {"on": True}},
        )
    )
    client.reset_mock()
    return conn, client


def test_absent_role_does_not_fire_callback() -> None:
    """A role omitted from server/state (UndefinedField) fires no callback."""
    conn, client = _make_connection()
    conn._handle_server_state(  # noqa: SLF001
        ServerStatePayload(metadata=SessionUpdateMetadata(timestamp=1))
    )
    client.notify_metadata_callback.assert_called_once()
    client.notify_color_callback.assert_not_called()
    client.notify_controller_callback.assert_not_called()


def test_server_state_keeps_omitted_roles_and_replaces_present_ones() -> None:
    """Held state keeps omitted role objects; present objects replace them."""
    conn, _ = _make_connection()
    conn._handle_server_state(  # noqa: SLF001
        ServerStatePayload(
            metadata=SessionUpdateMetadata(timestamp=1, title="A", album="B"),
            controller=_CONTROLLER,
        )
    )

    conn._handle_server_state(ServerStatePayload(color=_COLOR))  # noqa: SLF001
    conn._handle_server_state(  # noqa: SLF001
        ServerStatePayload(metadata=SessionUpdateMetadata(timestamp=3, title="C"))
    )

    assert conn._server_state == ServerStatePayload(  # noqa: SLF001
        metadata=SessionUpdateMetadata(timestamp=3, title="C"),
        controller=_CONTROLLER,
        color=_COLOR,
    )


@pytest.mark.parametrize(
    "payload",
    [
        ServerActivatePayload(activities=[Activity.PLAYBACK], active_roles=[]),
        ServerActivatePayload(activities=[Activity.PAIRING]),
        ServerActivatePayload(
            activities=[Activity.PLAYBACK],
            active_roles=["metadata@v2", "color@v2", "controller@v2", "_acme@v2"],
        ),
    ],
    ids=["explicit", "not-playback-capable", "version-replacement"],
)
async def test_activation_discards_removed_role_state(payload: ServerActivatePayload) -> None:
    """Applying a server/activate discards every removed state role and signals None."""
    conn, client = _activated_connection()

    assert await conn._apply_activation(payload) is None  # noqa: SLF001

    assert conn._server_state == ServerStatePayload()  # noqa: SLF001
    client.notify_metadata_callback.assert_called_once_with(None)
    client.notify_color_callback.assert_called_once_with(None)
    client.notify_controller_callback.assert_called_once_with(None)


async def test_activation_keeps_state_of_retained_roles() -> None:
    """Roles that stay active at the same version keep their state and fire no callback."""
    conn, client = _activated_connection()

    await conn._apply_activation(  # noqa: SLF001
        ServerActivatePayload(
            activities=[Activity.PLAYBACK], active_roles=[Roles.METADATA.value, "_acme@v1"]
        )
    )

    assert conn._server_state == ServerStatePayload(  # noqa: SLF001
        metadata=_METADATA, application_objects={"_acme": {"on": True}}
    )
    client.notify_metadata_callback.assert_not_called()
    client.notify_color_callback.assert_called_once_with(None)
    client.notify_controller_callback.assert_called_once_with(None)


async def test_activation_signals_no_discard_for_role_without_state() -> None:
    """A removed role that never received state fires no callback."""
    conn, client = _activated_connection()
    conn._active_roles.append(Roles.PLAYER.value)  # noqa: SLF001
    conn._server_state = ServerStatePayload(metadata=_METADATA)  # noqa: SLF001

    await conn._apply_activation(  # noqa: SLF001
        ServerActivatePayload(activities=[Activity.PLAYBACK], active_roles=[Roles.METADATA.value])
    )

    assert conn._server_state == ServerStatePayload(metadata=_METADATA)  # noqa: SLF001
    client.notify_metadata_callback.assert_not_called()
    client.notify_color_callback.assert_not_called()
    client.notify_controller_callback.assert_not_called()


_NOW_US = 1_000_000
# Far enough ahead that the update stays pending for the whole test.
_LATER_US = _NOW_US + 60_000_000
_SOON_US = _NOW_US + 20_000


def _make_scheduling_connection(*, synced: bool = True) -> tuple[SendspinConnection, MagicMock]:
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


async def test_future_metadata_is_pending_until_its_timestamp() -> None:
    """Future metadata is reported as scheduled, then as current once its time is reached."""
    conn, client = _make_scheduling_connection()
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
    conn, client = _make_scheduling_connection()
    conn._handle_server_state(  # noqa: SLF001
        ServerStatePayload(color=SessionUpdateColor(timestamp=_LATER_US))
    )
    current = ServerStatePayload(color=SessionUpdateColor(timestamp=_NOW_US, primary=(1, 2, 3)))

    conn._handle_server_state(current)  # noqa: SLF001

    client.notify_color_callback.assert_called_once_with(current)
    assert conn._pending_state == {}  # noqa: SLF001


async def test_later_arrival_replaces_scheduled_update() -> None:
    """A newer scheduled update replaces the held one, even with an earlier timestamp."""
    conn, client = _make_scheduling_connection()
    first = ServerStatePayload(color=SessionUpdateColor(timestamp=_LATER_US, primary=(1, 1, 1)))
    second = ServerStatePayload(color=SessionUpdateColor(timestamp=_SOON_US, primary=(2, 2, 2)))

    conn._handle_server_state(first)  # noqa: SLF001
    conn._handle_server_state(second)  # noqa: SLF001
    await asyncio.sleep(0.05)

    client.notify_color_callback.assert_called_once_with(second)
    assert conn._pending_state == {}  # noqa: SLF001


async def test_present_clear_discards_scheduled_update() -> None:
    """A timestamp-only object taking effect now clears the state and the scheduled update."""
    conn, client = _make_scheduling_connection()
    conn._handle_server_state(  # noqa: SLF001
        ServerStatePayload(metadata=SessionUpdateMetadata(timestamp=_SOON_US, title="Next"))
    )
    cleared = ServerStatePayload(metadata=SessionUpdateMetadata(timestamp=_NOW_US))

    conn._handle_server_state(cleared)  # noqa: SLF001
    await asyncio.sleep(0.05)

    client.notify_metadata_callback.assert_called_once_with(cleared)
    assert conn._server_state == cleared  # noqa: SLF001


async def test_other_role_objects_apply_while_one_is_scheduled() -> None:
    """Role objects without a future timestamp in the same message apply at once."""
    conn, client = _make_scheduling_connection()
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
    conn, client = _make_scheduling_connection(synced=False)
    payload = ServerStatePayload(metadata=SessionUpdateMetadata(timestamp=_LATER_US))

    conn._handle_server_state(payload)  # noqa: SLF001

    client.notify_metadata_callback.assert_called_once_with(payload)
    client.notify_scheduled_metadata.assert_not_called()


async def test_clock_update_reschedules_scheduled_update() -> None:
    """A new clock estimate moves a scheduled update to its newly mapped local time."""
    conn, client = _make_scheduling_connection()
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


def _activatable(conn: SendspinConnection) -> None:
    """Give a scheduling connection what applying a server/activate needs."""
    conn._client.note_playback_activity = AsyncMock()  # noqa: SLF001
    conn._noise_psk = ResolvedPsk("id", b"\x00" * 32, PskCategory.LONG_TERM)  # noqa: SLF001
    conn._source_stream_active = False  # noqa: SLF001
    conn._unpaired_access_enabled = AsyncMock(return_value=False)  # type: ignore[method-assign]  # noqa: SLF001


async def test_activation_discards_removed_role_scheduled_state() -> None:
    """Removing a role discards its current and scheduled state; kept roles are unchanged."""
    conn, client = _make_scheduling_connection()
    _activatable(conn)
    color = ServerStatePayload(color=SessionUpdateColor(timestamp=_NOW_US))
    conn._handle_server_state(  # noqa: SLF001
        ServerStatePayload(metadata=SessionUpdateMetadata(timestamp=_NOW_US, title="Now"))
    )
    conn._handle_server_state(  # noqa: SLF001
        ServerStatePayload(metadata=SessionUpdateMetadata(timestamp=_SOON_US, title="Next"))
    )
    conn._handle_server_state(color)  # noqa: SLF001
    client.reset_mock()

    await conn._apply_activation(  # noqa: SLF001
        ServerActivatePayload(activities=[Activity.PLAYBACK], active_roles=[Roles.COLOR.value])
    )
    await asyncio.sleep(0.05)

    client.notify_metadata_callback.assert_called_once_with(None)
    client.notify_color_callback.assert_not_called()
    assert conn._pending_state == {}  # noqa: SLF001
    assert conn._server_state == color  # noqa: SLF001


async def test_activation_discards_scheduled_state_of_role_without_current_state() -> None:
    """A removed role holding only a scheduled update still signals the discard."""
    conn, client = _make_scheduling_connection()
    _activatable(conn)
    conn._handle_server_state(  # noqa: SLF001
        ServerStatePayload(color=SessionUpdateColor(timestamp=_SOON_US, primary=(1, 2, 3)))
    )
    client.reset_mock()

    await conn._apply_activation(  # noqa: SLF001
        ServerActivatePayload(activities=[Activity.PLAYBACK], active_roles=[])
    )
    await asyncio.sleep(0.05)

    client.notify_color_callback.assert_called_once_with(None)
    assert conn._pending_state == {}  # noqa: SLF001
