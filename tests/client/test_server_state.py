"""Tests for client handling of server/state role objects."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiosendspin.client.connection import SendspinConnection
from aiosendspin.models.color import SessionUpdateColor
from aiosendspin.models.controller import ControllerStatePayload
from aiosendspin.models.core import ServerActivatePayload, ServerStatePayload
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
    client.clock = ManualClock(now_us_value=1_000_000)
    conn._client = client  # noqa: SLF001
    conn._time_filter = SendspinTimeFilter()  # noqa: SLF001
    conn._init_state_trackers()  # noqa: SLF001
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
    client.notify_effective_metadata.assert_called_once()
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
