"""Tests for ControllerV1Role (v1) implementation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aiosendspin.models.controller import ControllerCommandPayload
from aiosendspin.models.types import MediaCommand
from aiosendspin.server.roles.controller.v1 import ControllerV1Role


def _make_client_stub() -> MagicMock:
    """Create a mock client for testing."""
    client = MagicMock()
    client.group = MagicMock()
    client.group.group_role.return_value = None
    return client


def test_controller_role_on_deactivate_sends_no_state() -> None:
    """on_deactivate() sends no server/state; the server/activate carries the discard."""
    client = _make_client_stub()
    client.connection.clears_role_state_with_null = False
    role = ControllerV1Role(client=client)
    role.on_connect()
    client.send_role_message.reset_mock()

    role.on_deactivate()

    client.send_role_message.assert_not_called()


# DEPRECATED(spec-pr-275): remove in aiosendspin <version>
def test_controller_role_on_deactivate_clears_legacy_state_with_null() -> None:
    """on_deactivate() sends a null controller object to a legacy-generation client."""
    client = _make_client_stub()
    client.connection.clears_role_state_with_null = True
    role = ControllerV1Role(client=client)
    role.on_connect()
    client.send_role_message.reset_mock()

    role.on_deactivate()

    client.send_role_message.assert_called_once()
    assert client.send_role_message.call_args.args[1].to_dict() == {
        "type": "server/state",
        "payload": {"controller": None},
    }


def test_controller_role_has_role_id() -> None:
    """ControllerV1Role has role_id of 'controller@v1'."""
    client = _make_client_stub()
    role = ControllerV1Role(client=client)
    assert role.role_id == "controller@v1"


def test_controller_role_has_role_family() -> None:
    """ControllerV1Role has role_family of 'controller'."""
    client = _make_client_stub()
    role = ControllerV1Role(client=client)
    assert role.role_family == "controller"


def test_controller_role_requires_client() -> None:
    """ControllerV1Role raises ValueError if no client provided."""
    with pytest.raises(ValueError, match="requires a client"):
        ControllerV1Role(client=None)


def test_controller_role_on_connect_subscribes_to_group_role() -> None:
    """on_connect() subscribes to ControllerGroupRole."""
    client = _make_client_stub()
    group_role = MagicMock()
    client.group.group_role.return_value = group_role

    role = ControllerV1Role(client=client)
    role.on_connect()

    client.group.group_role.assert_called_with("controller")
    group_role.subscribe.assert_called_once_with(role)


def test_controller_role_on_disconnect_unsubscribes_from_group_role() -> None:
    """on_disconnect() unsubscribes from ControllerGroupRole."""
    client = _make_client_stub()
    group_role = MagicMock()
    client.group.group_role.return_value = group_role

    role = ControllerV1Role(client=client)
    role.on_connect()
    role.on_disconnect()

    group_role.unsubscribe.assert_called_once_with(role)


def test_controller_role_handle_command_forwards_to_group_role() -> None:
    """handle_command() forwards command to group role."""
    client = _make_client_stub()
    group_role = MagicMock()
    client.group.group_role.return_value = group_role

    role = ControllerV1Role(client=client)
    role.on_connect()

    cmd = ControllerCommandPayload(command=MediaCommand.PLAY)
    role.handle_command(cmd)

    group_role.handle_command.assert_called_once_with(cmd)


def test_controller_role_handle_command_noop_without_group_role() -> None:
    """handle_command() is a no-op if not subscribed to group role."""
    client = _make_client_stub()
    client.group.group_role.return_value = None

    role = ControllerV1Role(client=client)
    role.on_connect()

    cmd = ControllerCommandPayload(command=MediaCommand.PLAY)
    role.handle_command(cmd)  # Should not raise


def test_controller_role_has_no_stream_requirements() -> None:
    """ControllerV1Role does not send binary streams."""
    client = _make_client_stub()
    role = ControllerV1Role(client=client)
    assert role.get_stream_requirements() is None


def test_controller_role_has_no_audio_requirements() -> None:
    """ControllerV1Role does not receive audio."""
    client = _make_client_stub()
    role = ControllerV1Role(client=client)
    assert role.get_audio_requirements() is None
