"""Bring-up failure handling for incoming (server-initiated) connections."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiosendspin.client.connection import SendspinConnection
from aiosendspin.models.core import ServerActivatePayload
from aiosendspin.models.types import Activity, Roles
from aiosendspin.noise.trust_store import PskCategory, ResolvedPsk
from tests.conftest import make_sdk_client

if TYPE_CHECKING:
    from aiohttp import web


async def test_attach_websocket_unexpected_failure_disconnects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bring-up failure outside the expected set tears the connection down and propagates."""
    sdk = make_sdk_client(client_name="c", roles=[Roles.CONTROLLER])
    disconnects: list[SendspinConnection] = []

    async def boom(
        self: SendspinConnection,  # noqa: ARG001
        ws: web.WebSocketResponse,  # noqa: ARG001
        *,
        expected_server_id: str | None = None,  # noqa: ARG001
    ) -> None:
        raise UnicodeDecodeError("utf-8", b"x", 0, 1, "bad")

    async def record_disconnect(self: SendspinConnection) -> None:
        disconnects.append(self)

    monkeypatch.setattr(SendspinConnection, "attach_websocket", boom)
    monkeypatch.setattr(SendspinConnection, "disconnect", record_disconnect)

    with pytest.raises(UnicodeDecodeError):
        await sdk.attach_websocket(MagicMock())

    assert len(disconnects) == 1
    assert not sdk._provisional_connections  # noqa: SLF001


async def test_admission_failure_disconnects_incoming_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An admission failure disconnects the incoming connection instead of orphaning it."""
    sdk = make_sdk_client(client_name="c", roles=[Roles.CONTROLLER])
    disconnects: list[SendspinConnection] = []

    async def bring_up(
        self: SendspinConnection,  # noqa: ARG001
        ws: web.WebSocketResponse,  # noqa: ARG001
        *,
        expected_server_id: str | None = None,  # noqa: ARG001
    ) -> None:
        return

    async def boom() -> None:
        raise OSError("store unavailable")

    async def record_disconnect(self: SendspinConnection) -> None:
        disconnects.append(self)

    monkeypatch.setattr(SendspinConnection, "attach_websocket", bring_up)
    monkeypatch.setattr(SendspinConnection, "disconnect", record_disconnect)
    monkeypatch.setattr(sdk, "_ensure_last_playback_loaded", boom)

    with pytest.raises(OSError, match="store unavailable"):
        await sdk.attach_websocket(MagicMock())

    assert len(disconnects) == 1
    assert not sdk._provisional_connections  # noqa: SLF001


async def test_admission_refuses_unpaired_access_disabled_during_bring_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A connection relying on unpaired access disabled before its admission is refused."""
    sdk = make_sdk_client(client_name="c", roles=[Roles.CONTROLLER])
    store = sdk.pairing_store
    await store.store_pairing_config(
        replace(await store.get_pairing_config(), unpaired_access_enabled=True)
    )
    sent: list[dict[str, object]] = []
    connections: list[SendspinConnection] = []

    async def capture(payload: str) -> None:
        sent.append(json.loads(payload))

    async def bring_up(
        self: SendspinConnection,
        ws: web.WebSocketResponse,  # noqa: ARG001
        *,
        expected_server_id: str | None = None,  # noqa: ARG001
    ) -> None:
        connections.append(self)
        self._ws = MagicMock(closed=False, send_str=capture, close=AsyncMock())
        self._connected = True
        self._server_id = "server-1"
        self._noise_psk = ResolvedPsk("psk-id", b"\x00" * 32, PskCategory.SENTINEL)
        activation = ServerActivatePayload(activities=[Activity.PLAYBACK])
        assert await self._apply_activation(activation) is None
        # The operator disables unpaired access while the connection is still provisional.
        await sdk.set_unpaired_access(enabled=False)

    monkeypatch.setattr(SendspinConnection, "attach_websocket", bring_up)

    await sdk.attach_websocket(MagicMock())

    assert sent == [{"type": "client/goodbye", "payload": {"reason": "pairing_required"}}]
    assert not connections[0].connected
    assert sdk._admitted_connection is None  # noqa: SLF001
    assert not sdk.connected
