"""Tests for MetadataV1Role (v1) implementation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aiosendspin.models.core import ServerStateMessage
from aiosendspin.server.roles.metadata.group import MetadataGroupRole
from aiosendspin.server.roles.metadata.state import Metadata
from aiosendspin.server.roles.metadata.v1 import MetadataV1Role


def _make_client_stub() -> MagicMock:
    """Create a mock client for testing."""
    client = MagicMock()
    client.group = MagicMock()
    client.group.group_role.return_value = None
    return client


def test_metadata_role_has_role_id() -> None:
    """MetadataV1Role has role_id of 'metadata@v1'."""
    client = _make_client_stub()
    role = MetadataV1Role(client=client)
    assert role.role_id == "metadata@v1"


def test_metadata_role_has_role_family() -> None:
    """MetadataV1Role has role_family of 'metadata'."""
    client = _make_client_stub()
    role = MetadataV1Role(client=client)
    assert role.role_family == "metadata"


def test_metadata_role_requires_client() -> None:
    """MetadataV1Role raises ValueError if no client provided."""
    with pytest.raises(ValueError, match="requires a client"):
        MetadataV1Role(client=None)


def test_metadata_role_on_connect_subscribes_to_group_role() -> None:
    """on_connect() subscribes to MetadataGroupRole."""
    client = _make_client_stub()
    group_role = MagicMock()
    client.group.group_role.return_value = group_role

    role = MetadataV1Role(client=client)
    role.on_connect()

    client.group.group_role.assert_called_with("metadata")
    group_role.subscribe.assert_called_once_with(role)


def test_metadata_role_on_disconnect_unsubscribes_from_group_role() -> None:
    """on_disconnect() unsubscribes from MetadataGroupRole."""
    client = _make_client_stub()
    group_role = MagicMock()
    client.group.group_role.return_value = group_role

    role = MetadataV1Role(client=client)
    role.on_connect()
    role.on_disconnect()

    group_role.unsubscribe.assert_called_once_with(role)


def test_metadata_role_on_deactivate_sends_no_state() -> None:
    """on_deactivate() sends no server/state; the server/activate carries the discard."""
    client = _make_client_stub()
    client.connection.clears_role_state_with_null = False
    role = MetadataV1Role(client=client)
    role.on_connect()
    client.send_role_message.reset_mock()

    role.on_deactivate()

    client.send_role_message.assert_not_called()


# DEPRECATED(spec-pr-275): remove in aiosendspin <version>
def test_metadata_role_on_deactivate_clears_legacy_state_with_null() -> None:
    """on_deactivate() sends a null metadata object to a legacy-generation client."""
    client = _make_client_stub()
    client.connection.clears_role_state_with_null = True
    role = MetadataV1Role(client=client)
    role.on_connect()
    client.send_role_message.reset_mock()

    role.on_deactivate()

    client.send_role_message.assert_called_once()
    assert client.send_role_message.call_args.args[1].to_dict() == {
        "type": "server/state",
        "payload": {"metadata": None},
    }


def test_metadata_role_has_no_stream_requirements() -> None:
    """MetadataV1Role does not send binary streams."""
    client = _make_client_stub()
    role = MetadataV1Role(client=client)
    assert role.get_stream_requirements() is None


def test_metadata_role_has_no_audio_requirements() -> None:
    """MetadataV1Role does not receive audio."""
    client = _make_client_stub()
    role = MetadataV1Role(client=client)
    assert role.get_audio_requirements() is None


def test_metadata_role_on_connect_sends_state_exactly_once() -> None:
    """on_connect() emits a single ServerStateMessage via on_member_join."""
    group = MagicMock()
    group._server = MagicMock()  # noqa: SLF001
    group._server.clock.now_us.return_value = 1_000_000  # noqa: SLF001
    group.has_active_stream = False
    group_role = MetadataGroupRole(group)
    group_role.set_metadata(Metadata(title="Test"))

    client = MagicMock()
    client.connection = MagicMock()
    client.group.group_role.return_value = group_role
    client.send_role_message = MagicMock()

    role = MetadataV1Role(client=client)
    role.on_connect()

    state_calls = [
        call
        for call in client.send_role_message.call_args_list
        if isinstance(call.args[1], ServerStateMessage)
    ]
    assert len(state_calls) == 1
