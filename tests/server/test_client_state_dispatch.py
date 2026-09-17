"""The connection dispatch forwards client availability to the client state machine."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest

from aiosendspin.models.artwork import (
    ArtworkChannel,
    ClientStateArtwork,
    StreamRequestFormatArtwork,
)
from aiosendspin.models.core import (
    ClientHelloMessage,
    ClientHelloPayload,
    ClientStateMessage,
    ClientStatePayload,
    StreamRequestFormatMessage,
    StreamRequestFormatPayload,
)
from aiosendspin.models.management import ManagementResultMessage, ManagementResultPayload
from aiosendspin.models.player import PlayerStatePayload, StreamRequestFormatPlayer
from aiosendspin.models.types import ArtworkSource, ManagementResult, PlayerCommand
from aiosendspin.models.visualizer import (
    ClientHelloVisualizerSupport,
    StreamRequestFormatVisualizer,
    VisualizerStatePayload,
)
from aiosendspin.server.clock import LoopClock
from aiosendspin.server.compliance import ClientComplianceError
from aiosendspin.server.connection import SendspinConnection
from aiosendspin.server.roles import PlayerV1Role
from aiosendspin.server.roles.visualizer.v1 import VisualizerV1Role


@dataclass(slots=True)
class _DummyServer:
    loop: asyncio.AbstractEventLoop
    clock: Any
    id: str = "srv"
    name: str = "server"

    def on_client_first_connect(self, client_id: str) -> None:
        """No-op: the dispatch tests don't exercise first-connect side effects."""


@pytest.mark.asyncio
async def test_available_false_drives_external_source_transition() -> None:
    """A new client's `available: false` must trigger the external-source transition."""
    loop = asyncio.get_running_loop()
    conn = SendspinConnection(
        _DummyServer(loop=loop, clock=LoopClock(loop)), wsock_client=MagicMock()
    )

    client = MagicMock()
    client.available = True
    client.handle_availability_change = AsyncMock()
    client.active_roles = []
    conn._client = client  # noqa: SLF001
    conn._initial_state_received = True  # noqa: SLF001

    await conn._handle_message(  # noqa: SLF001
        ClientStateMessage(payload=ClientStatePayload(available=False)), timestamp_us=0
    )

    client.handle_availability_change.assert_awaited_once_with(available=False)


def _conn_with_client() -> tuple[SendspinConnection, MagicMock]:
    loop = asyncio.get_running_loop()
    conn = SendspinConnection(
        _DummyServer(loop=loop, clock=LoopClock(loop)), wsock_client=MagicMock()
    )
    client = MagicMock()
    client.awaits_role_state.return_value = False
    conn._client = client  # noqa: SLF001
    return conn, client


@pytest.mark.asyncio
async def test_second_client_hello_is_flagged() -> None:
    """A client/hello after the hello exchange is flagged as non-compliant."""
    conn, client = _conn_with_client()
    await conn._handle_message(  # noqa: SLF001
        ClientHelloMessage(payload=ClientHelloPayload(name="c", supported_roles=[])),
        timestamp_us=0,
    )
    client.flag_noncompliance.assert_called_once()


@pytest.mark.asyncio
async def test_unsolicited_management_result_is_flagged() -> None:
    """A management/result with no request in flight is flagged as non-compliant."""
    conn, client = _conn_with_client()
    conn._management_waiter = None  # noqa: SLF001
    await conn._handle_message(  # noqa: SLF001
        ManagementResultMessage(payload=ManagementResultPayload(result=ManagementResult.OK)),
        timestamp_us=0,
    )
    client.flag_noncompliance.assert_called_once()


@pytest.mark.asyncio
async def test_state_before_the_initial_gate_opens_is_not_initial() -> None:
    """A client/state read before any activation needs one (connect-time pairing) is ordinary."""
    conn, client = _conn_with_client()
    client.active_roles = []
    client.available = True

    await conn._handle_client_state(ClientStatePayload(available=True))  # noqa: SLF001

    client.mark_connected.assert_not_called()
    assert conn._initial_state_received is False  # noqa: SLF001


def _role_mock(deviations: list[str]) -> MagicMock:
    role = MagicMock()
    role.initial_state_deviations.return_value = deviations
    return role


@pytest.mark.asyncio
async def test_initial_state_flags_missing_available_and_role_reasons() -> None:
    """Missing `available` plus each active role's own deviations are each flagged."""
    conn, client = _conn_with_client()
    client.active_roles = [_role_mock(["role-specific problem"])]
    conn._flag_initial_state_deviations(ClientStatePayload())  # noqa: SLF001
    flagged = [call.args[0] for call in client.flag_noncompliance.call_args_list]
    assert any("available" in r for r in flagged)
    assert any("role-specific problem" in r for r in flagged)


@pytest.mark.asyncio
async def test_initial_state_complete_state_is_not_flagged() -> None:
    """A complete initial client/state with compliant roles is not flagged."""
    conn, client = _conn_with_client()
    client.active_roles = [_role_mock([])]
    conn._flag_initial_state_deviations(ClientStatePayload(available=True))  # noqa: SLF001
    client.flag_noncompliance.assert_not_called()


@pytest.mark.asyncio
async def test_missing_initial_state_rejects_when_flagged() -> None:
    """A never-sent initial state hard-disconnects when the flag raises."""
    conn, client = _conn_with_client()
    client.flag_noncompliance.side_effect = ClientComplianceError("nope")
    conn.disconnect = AsyncMock()  # type: ignore[method-assign]
    conn._initial_state_timeout_callback()  # noqa: SLF001
    await asyncio.sleep(0)
    conn.disconnect.assert_awaited_once_with(retry_connection=False)
    client.mark_connected.assert_not_called()


@pytest.mark.asyncio
async def test_missing_initial_state_marks_connected_when_lenient() -> None:
    """A never-sent initial state is tolerated: the client is marked connected."""
    conn, client = _conn_with_client()
    conn._initial_state_timeout_callback()  # noqa: SLF001
    assert conn._initial_state_received is True  # noqa: SLF001
    client.mark_connected.assert_called_once()


def _role(family: str) -> MagicMock:
    role = MagicMock()
    role.role_family = family
    return role


@pytest.mark.asyncio
async def test_binary_is_held_until_initial_state() -> None:
    """Binary enqueued before the initial client/state is buffered, then flushed on arrival."""
    conn, client = _conn_with_client()
    role = _role("artwork")
    role.requires_initial_state.return_value = True
    client.active_roles = [role]

    conn.send_binary(b"snapshot", role="artwork", timestamp_us=0, message_type=30)
    assert not conn._role_queues.get("artwork")  # nothing on the wire yet  # noqa: SLF001
    assert len(conn._pending_binary) == 1  # noqa: SLF001

    conn._initial_state_received = True  # noqa: SLF001
    conn._flush_pending_binary()  # noqa: SLF001
    assert conn._pending_binary == []  # noqa: SLF001
    assert conn._role_queues.get("artwork")  # now enqueued  # noqa: SLF001


@pytest.mark.asyncio
async def test_pending_binary_dropped_when_stream_boundary_intervenes() -> None:
    """A stream boundary during the wait bumps the epoch and discards stale buffered binary."""
    conn, client = _conn_with_client()
    role = _role("artwork")
    role.requires_initial_state.return_value = True
    client.active_roles = [role]

    conn.send_binary(b"snapshot", role="artwork", timestamp_us=0, message_type=30)
    assert len(conn._pending_binary) == 1  # noqa: SLF001

    conn.drop_pending_binary(["artwork"])  # stream/clear or stream/end bumps the epoch
    conn._initial_state_received = True  # noqa: SLF001
    conn._flush_pending_binary()  # noqa: SLF001
    assert not conn._role_queues.get("artwork")  # stale binary not replayed  # noqa: SLF001


@pytest.mark.asyncio
async def test_pending_epoch_exempt_binary_survives_stream_boundary() -> None:
    """Epoch-exempt binary buffered before the initial state is still flushed after a boundary."""
    conn, client = _conn_with_client()
    role = _role("artwork")
    role.requires_initial_state.return_value = True
    client.active_roles = [role]

    conn.send_binary(b"snapshot", role="artwork", timestamp_us=0, message_type=30)
    conn.send_binary(b"cancel", role="artwork", timestamp_us=0, message_type=30, epoch_exempt=True)
    conn.drop_pending_binary(["artwork"])
    conn._initial_state_received = True  # noqa: SLF001
    conn._flush_pending_binary()  # noqa: SLF001

    queued = conn._role_queues.get("artwork")  # noqa: SLF001
    assert queued is not None
    assert [entry.binary.data for _, _, entry in queued if entry.binary] == [b"cancel"]


@pytest.mark.asyncio
async def test_client_state_player_object_for_inactive_role_is_flagged() -> None:
    """A player state object with no active player role is flagged."""
    conn, client = _conn_with_client()
    client.active_roles = [_role("controller")]
    conn._initial_state_received = True  # noqa: SLF001
    client.available = None
    await conn._handle_message(  # noqa: SLF001
        ClientStateMessage(payload=ClientStatePayload(player=PlayerStatePayload())), timestamp_us=0
    )
    flagged = [call.args[0] for call in client.flag_noncompliance.call_args_list]
    assert any("player" in r and "inactive role" in r for r in flagged)


@pytest.mark.asyncio
async def test_client_state_player_object_for_active_role_is_not_flagged() -> None:
    """A player state object with an active player role is not flagged."""
    conn, client = _conn_with_client()
    client.active_roles = [_role("player")]
    conn._initial_state_received = True  # noqa: SLF001
    client.available = None
    await conn._handle_message(  # noqa: SLF001
        ClientStateMessage(payload=ClientStatePayload(player=PlayerStatePayload())), timestamp_us=0
    )
    client.flag_noncompliance.assert_not_called()


@pytest.mark.asyncio
async def test_strict_rejection_applies_no_side_effects() -> None:
    """A rejected client/state does not change availability before the rejection."""
    conn, client = _conn_with_client()
    conn._initial_state_received = True  # noqa: SLF001
    client.flag_noncompliance.side_effect = ClientComplianceError("nope")
    client.handle_availability_change = AsyncMock()
    client.active_roles = [_role("controller")]  # player object below is for an inactive role
    with pytest.raises(ClientComplianceError):
        await conn._handle_message(  # noqa: SLF001
            ClientStateMessage(
                payload=ClientStatePayload(available=False, player=PlayerStatePayload())
            ),
            timestamp_us=0,
        )
    client.mark_connected.assert_not_called()
    client.handle_availability_change.assert_not_called()


@pytest.mark.asyncio
async def test_role_client_state_deviation_flagged_before_side_effects() -> None:
    """A role's client/state deviation is flagged before availability is applied."""
    conn, client = _conn_with_client()
    conn._initial_state_received = True  # noqa: SLF001
    client.flag_noncompliance.side_effect = ClientComplianceError("nope")
    client.handle_availability_change = AsyncMock()
    role = _role("player")
    role.client_state_deviations.return_value = ["used legacy player.state"]
    client.active_roles = [role]
    with pytest.raises(ClientComplianceError):
        await conn._handle_message(  # noqa: SLF001
            ClientStateMessage(payload=ClientStatePayload(available=False)), timestamp_us=0
        )
    client.handle_availability_change.assert_not_called()


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
@pytest.mark.asyncio
async def test_player_request_format_is_flagged_and_still_routed() -> None:
    """A pre-#195 player format request is flagged, then handed to the roles."""
    conn, client = _conn_with_client()
    role = _role("player")
    client.active_roles = [role]
    payload = StreamRequestFormatPayload(player=StreamRequestFormatPlayer(sample_rate=44100))

    await conn._handle_message(  # noqa: SLF001
        StreamRequestFormatMessage(payload=payload), timestamp_us=0
    )

    flagged = [call.args[0] for call in client.flag_noncompliance.call_args_list]
    assert flagged == [
        "sent a stream/request-format player object, superseded by the client/state player format"
    ]
    role.on_stream_request_format.assert_called_once_with(payload)


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
@pytest.mark.asyncio
async def test_strict_rejection_of_player_request_format_skips_roles() -> None:
    """A strict server rejects a player format request before any role applies it."""
    conn, client = _conn_with_client()
    client.flag_noncompliance.side_effect = ClientComplianceError("nope")
    role = _role("player")
    client.active_roles = [role]

    with pytest.raises(ClientComplianceError):
        await conn._handle_message(  # noqa: SLF001
            StreamRequestFormatMessage(
                payload=StreamRequestFormatPayload(player=StreamRequestFormatPlayer())
            ),
            timestamp_us=0,
        )
    role.on_stream_request_format.assert_not_called()


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
@pytest.mark.asyncio
async def test_visualizer_request_format_is_flagged_and_still_routed() -> None:
    """A pre-#195 visualizer format request is flagged, then handed to the roles."""
    conn, client = _conn_with_client()
    role = _role("visualizer")
    client.active_roles = [role]
    payload = StreamRequestFormatPayload(visualizer=StreamRequestFormatVisualizer(rate_max=15))

    await conn._handle_message(  # noqa: SLF001
        StreamRequestFormatMessage(payload=payload), timestamp_us=0
    )

    flagged = [call.args[0] for call in client.flag_noncompliance.call_args_list]
    assert flagged == [
        (
            "sent a stream/request-format visualizer object, "
            "superseded by the client/state visualizer object"
        )
    ]
    role.on_stream_request_format.assert_called_once_with(payload)


_VISUALIZER_STATE = VisualizerStatePayload(types=["loudness", "spectrum"], rate_max=30)


@pytest.mark.asyncio
async def test_client_state_visualizer_object_for_inactive_role_is_flagged() -> None:
    """A visualizer state object with no active visualizer role is flagged."""
    conn, client = _conn_with_client()
    client.active_roles = [_role("controller")]
    conn._initial_state_received = True  # noqa: SLF001
    client.available = None
    await conn._handle_message(  # noqa: SLF001
        ClientStateMessage(payload=ClientStatePayload(visualizer=_VISUALIZER_STATE)),
        timestamp_us=0,
    )
    flagged = [call.args[0] for call in client.flag_noncompliance.call_args_list]
    assert flagged == ["client/state carried a visualizer object for an inactive role"]


def _visualizer_client(conn: SendspinConnection, client: MagicMock) -> VisualizerV1Role:
    client.info.visualizer_support = ClientHelloVisualizerSupport(buffer_capacity=65_536)
    client.available = None
    role = VisualizerV1Role(client=client)
    role.on_connect()
    client.active_roles = [role]
    conn._initial_state_received = True  # noqa: SLF001
    return role


@pytest.mark.asyncio
async def test_strict_rejects_visualizer_spectrum_without_config() -> None:
    """Requesting `spectrum` without its configuration rejects a strict client."""
    conn, client = _conn_with_client()
    client.flag_noncompliance.side_effect = ClientComplianceError("nope")
    role = _visualizer_client(conn, client)

    with pytest.raises(ClientComplianceError):
        await conn._handle_message(  # noqa: SLF001
            ClientStateMessage(payload=ClientStatePayload(visualizer=_VISUALIZER_STATE)),
            timestamp_us=0,
        )
    client.join_active_stream.assert_not_called()
    role.on_stream_start()
    client.send_role_message.assert_not_called()


@pytest.mark.asyncio
async def test_lenient_keeps_client_but_omits_spectrum_without_config() -> None:
    """A lenient server flags the missing configuration and streams the other types."""
    conn, client = _conn_with_client()
    role = _visualizer_client(conn, client)

    await conn._handle_message(  # noqa: SLF001
        ClientStateMessage(payload=ClientStatePayload(visualizer=_VISUALIZER_STATE)),
        timestamp_us=0,
    )

    flagged = [call.args[0] for call in client.flag_noncompliance.call_args_list]
    assert flagged == [
        "client/state requested visualizer 'spectrum' without a spectrum configuration"
    ]
    role.on_stream_start()
    start = client.send_role_message.call_args.args[1]
    assert start.payload.visualizer.types == ("loudness",)


@pytest.mark.asyncio
async def test_strict_rejects_nonpositive_visualizer_rate_max() -> None:
    """A non-positive visualizer rate_max rejects a strict client."""
    conn, client = _conn_with_client()
    client.flag_noncompliance.side_effect = ClientComplianceError("nope")
    _visualizer_client(conn, client)

    with pytest.raises(ClientComplianceError):
        await conn._handle_message(  # noqa: SLF001
            ClientStateMessage(
                payload=ClientStatePayload(
                    visualizer=VisualizerStatePayload(types=["loudness"], rate_max=0)
                )
            ),
            timestamp_us=0,
        )


@pytest.mark.asyncio
async def test_client_state_artwork_object_for_inactive_role_is_flagged() -> None:
    """An artwork state object with no active artwork role is flagged."""
    conn, client = _conn_with_client()
    client.active_roles = [_role("player")]
    conn._initial_state_received = True  # noqa: SLF001
    client.available = None
    artwork = ClientStateArtwork(channels=[ArtworkChannel(source=ArtworkSource.NONE)])
    await conn._handle_message(  # noqa: SLF001
        ClientStateMessage(payload=ClientStatePayload(artwork=artwork)), timestamp_us=0
    )
    flagged = [call.args[0] for call in client.flag_noncompliance.call_args_list]
    assert any("artwork" in r and "inactive role" in r for r in flagged)


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
@pytest.mark.asyncio
async def test_artwork_request_format_is_flagged_and_still_routed() -> None:
    """A pre-#195 artwork format request is flagged, then handed to the roles."""
    conn, client = _conn_with_client()
    role = _role("artwork")
    client.active_roles = [role]
    payload = StreamRequestFormatPayload(artwork=StreamRequestFormatArtwork(channel=0))

    await conn._handle_message(  # noqa: SLF001
        StreamRequestFormatMessage(payload=payload), timestamp_us=0
    )

    flagged = [call.args[0] for call in client.flag_noncompliance.call_args_list]
    assert flagged == [
        "sent a stream/request-format artwork object, superseded by the client/state artwork object"
    ]
    role.on_stream_request_format.assert_called_once_with(payload)


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
@pytest.mark.asyncio
async def test_strict_rejection_of_artwork_request_format_skips_roles() -> None:
    """A strict server rejects an artwork format request before any role applies it."""
    conn, client = _conn_with_client()
    client.flag_noncompliance.side_effect = ClientComplianceError("nope")
    role = _role("artwork")
    client.active_roles = [role]

    with pytest.raises(ClientComplianceError):
        await conn._handle_message(  # noqa: SLF001
            StreamRequestFormatMessage(
                payload=StreamRequestFormatPayload(artwork=StreamRequestFormatArtwork(channel=0))
            ),
            timestamp_us=0,
        )
    role.on_stream_request_format.assert_not_called()


@pytest.mark.asyncio
async def test_unrecognized_player_command_is_flagged_and_state_still_applies() -> None:
    """An unrecognized player command is dropped and flagged; the rest of the state applies."""
    conn, client = _conn_with_client()
    conn._initial_state_received = True  # noqa: SLF001
    client.available = True
    client.handle_availability_change = AsyncMock()
    role = _role("player")
    role.client_state_deviations.side_effect = PlayerV1Role(
        client=MagicMock()
    ).client_state_deviations
    client.active_roles = [role]
    message = SendspinConnection._deserialize_client_message(  # noqa: SLF001
        orjson.dumps(
            {
                "type": "client/state",
                "payload": {
                    "available": False,
                    "player": {"volume": 40, "supported_commands": ["volume", "teleport"]},
                },
            }
        ).decode()
    )

    await conn._handle_message(message, timestamp_us=0)  # noqa: SLF001

    flagged = [call.args[0] for call in client.flag_noncompliance.call_args_list]
    assert flagged == ["client/state declared unrecognized supported_commands: teleport"]
    client.handle_availability_change.assert_awaited_once_with(available=False)
    (applied,) = role.on_client_state.call_args.args
    assert applied.player.volume == 40
    assert applied.player.supported_commands == [PlayerCommand.VOLUME]


@pytest.mark.asyncio
async def test_unrecognized_player_command_rejected_when_strict() -> None:
    """A strict server rejects a client/state declaring an unrecognized player command."""
    conn, client = _conn_with_client()
    conn._initial_state_received = True  # noqa: SLF001
    client.flag_noncompliance.side_effect = ClientComplianceError("nope")
    role = _role("player")
    role.client_state_deviations.side_effect = PlayerV1Role(
        client=MagicMock()
    ).client_state_deviations
    client.active_roles = [role]
    payload = ClientStatePayload.from_dict({"player": {"supported_commands": ["teleport"]}})

    with pytest.raises(ClientComplianceError):
        await conn._handle_message(  # noqa: SLF001
            ClientStateMessage(payload=payload), timestamp_us=0
        )
    role.on_client_state.assert_not_called()
