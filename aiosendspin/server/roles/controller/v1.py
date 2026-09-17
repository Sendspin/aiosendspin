"""ControllerV1Role implementation (v1).

This role handles bidirectional communication:
- Inbound: client/command controller messages
- Outbound: server/state controller messages
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aiosendspin.models.controller import ControllerCommandPayload
from aiosendspin.models.core import LegacyServerStateClearMessage
from aiosendspin.models.types import MediaCommand
from aiosendspin.server.roles.base import Role
from aiosendspin.util import create_task

if TYPE_CHECKING:
    from aiosendspin.models.core import ClientCommandPayload
    from aiosendspin.server.client import SendspinClient
    from aiosendspin.server.roles.controller.group import ControllerGroupRole


logger = logging.getLogger(__name__)


class ControllerV1Role(Role):
    """Role implementation for controller clients.

    Receives controller state from ControllerGroupRole and forwards commands
    from the client to the group. State-machine concerns that span all roles
    (external_source, switch cycling, previous-group rejoin) live on
    :class:`SendspinClient` because they apply even to clients without the
    controller role.
    """

    def __init__(self, client: SendspinClient | None = None) -> None:
        """Initialize ControllerV1Role.

        Args:
            client: The owning SendspinClient.
        """
        if client is None:
            msg = "ControllerV1Role requires a client"
            raise ValueError(msg)
        self._client = client
        self._stream_started = False
        self._buffer_tracker = None
        self._group_role: ControllerGroupRole | None = None
        self._logger = logger.getChild(str(client.client_id))

    @property
    def role_id(self) -> str:
        """Versioned role identifier."""
        return "controller@v1"

    @property
    def role_family(self) -> str:
        """Role family name for protocol messages."""
        return "controller"

    def on_connect(self) -> None:
        """Subscribe to ControllerGroupRole for state updates."""
        self._subscribe_to_group_role()

    def on_deactivate(self) -> None:
        """Stop controller updates when the role is deactivated while still connected."""
        # DEPRECATED(spec-pr-275): remove in aiosendspin <version>
        if self.clears_state_with_null():
            self.send_message(LegacyServerStateClearMessage(self.role_family))
        super().on_deactivate()

    def on_disconnect(self) -> None:
        """Unsubscribe from ControllerGroupRole."""
        self._unsubscribe_from_group_role()

    def on_command(self, payload: ClientCommandPayload) -> None:
        """Handle client/command payload."""
        controller_cmd = payload.controller
        if controller_cmd is None:
            return

        if controller_cmd.command == MediaCommand.SWITCH:
            create_task(self._client.handle_switch_command())
            return

        # Forward other commands to group role
        if self._group_role is not None:
            self._group_role.handle_command(controller_cmd)

    def handle_command(self, cmd: ControllerCommandPayload) -> None:
        """Forward a controller command to the group role.

        Args:
            cmd: The controller command from the client.
        """
        if self._group_role is not None:
            self._group_role.handle_command(cmd)
