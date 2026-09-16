"""Tests for client handling of server/state role objects."""

from __future__ import annotations

from unittest.mock import MagicMock

from aiosendspin.client.connection import SendspinConnection
from aiosendspin.models.color import SessionUpdateColor
from aiosendspin.models.controller import ControllerStatePayload
from aiosendspin.models.core import ServerStatePayload
from aiosendspin.models.metadata import SessionUpdateMetadata
from aiosendspin.models.types import RepeatMode


def _make_connection() -> tuple[SendspinConnection, MagicMock]:
    conn = SendspinConnection.__new__(SendspinConnection)
    client = MagicMock()
    conn._client = client  # noqa: SLF001
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


def test_whole_role_null_fires_callback() -> None:
    """A whole-role null clears the role and fires its callback."""
    conn, client = _make_connection()
    conn._handle_server_state(ServerStatePayload(color=None))  # noqa: SLF001
    client.notify_color_callback.assert_called_once()
    client.notify_metadata_callback.assert_not_called()
    client.notify_controller_callback.assert_not_called()


def test_server_state_keeps_omitted_roles_and_replaces_present_ones() -> None:
    """Held state keeps omitted role objects; present objects, including null, replace them."""
    conn, _ = _make_connection()
    controller = ControllerStatePayload(
        supported_commands=[], volume=10, muted=False, repeat=RepeatMode.OFF, shuffle=False
    )
    conn._handle_server_state(  # noqa: SLF001
        ServerStatePayload(
            metadata=SessionUpdateMetadata(timestamp=1, title="A", album="B"),
            controller=controller,
        )
    )

    conn._handle_server_state(  # noqa: SLF001
        ServerStatePayload(color=SessionUpdateColor(timestamp=2, primary=(1, 2, 3)))
    )
    conn._handle_server_state(  # noqa: SLF001
        ServerStatePayload(metadata=SessionUpdateMetadata(timestamp=3, title="C"))
    )

    assert conn._server_state == ServerStatePayload(  # noqa: SLF001
        metadata=SessionUpdateMetadata(timestamp=3, title="C"),
        controller=controller,
        color=SessionUpdateColor(timestamp=2, primary=(1, 2, 3)),
    )

    conn._handle_server_state(ServerStatePayload(color=None))  # noqa: SLF001

    assert conn._server_state is not None  # noqa: SLF001
    assert conn._server_state.color is None  # noqa: SLF001
    assert conn._server_state.controller == controller  # noqa: SLF001
