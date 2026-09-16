"""Role (de)activation around server/activate: teardown order and the client/state hold."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import WSMessage, WSMsgType

from aiosendspin.models.artwork import ArtworkChannel, ClientHelloArtworkSupport, ClientStateArtwork
from aiosendspin.models.core import (
    ClientHelloMessage,
    ClientHelloPayload,
    ClientStatePayload,
    ServerCommandMessage,
    ServerCommandPayload,
    UnpairedAccess,
)
from aiosendspin.models.player import (
    ClientHelloPlayerSupport,
    PlayerCommandPayload,
    PlayerStatePayload,
    SupportedAudioFormat,
)
from aiosendspin.models.types import (
    ArtworkSource,
    AudioCodec,
    BinaryMessageType,
    PictureFormat,
    PlayerCommand,
    Roles,
)
from aiosendspin.models.visualizer import ClientHelloVisualizerSupport, VisualizerStatePayload
from aiosendspin.noise.keys import generate_psk, psk_id_for
from aiosendspin.noise.trust_store import PskCategory, ResolvedPsk, TrustedUnpairedClient
from aiosendspin.server.clock import LoopClock
from aiosendspin.server.connection import SendspinConnection
from aiosendspin.server.push_stream import PushStream
from aiosendspin.server.roles.player.v1 import PlayerV1Role
from tests.server.test_multi_server import _FakeTransport, _MockServer

if TYPE_CHECKING:
    from aiosendspin.server.client import SendspinClient

CLIENT_ID = "client-1"
_ARTWORK_CHANNEL = ArtworkChannel(
    source=ArtworkSource.ALBUM, format=PictureFormat.JPEG, width=300, height=300
)
_PLAYER_STATE = PlayerStatePayload(
    volume=50,
    muted=False,
    supported_commands=[PlayerCommand.VOLUME, PlayerCommand.MUTE],
    output_delay_ms=0,
    required_lead_time_ms=0,
    min_buffer_ms=0,
)


def _hello(roles: list[str], *, legacy: bool = False) -> ClientHelloPayload:
    return ClientHelloPayload(
        client_id=CLIENT_ID,
        name=CLIENT_ID,
        version=1,
        supported_roles=roles,
        unpaired_access=UnpairedAccess(enabled=True),
        player_support=ClientHelloPlayerSupport(
            supported_formats=[
                SupportedAudioFormat(
                    codec=AudioCodec.PCM, channels=2, sample_rate=48000, bit_depth=16
                )
            ],
            buffer_capacity=100_000,
            supported_commands=[],
        ),
        artwork_support=ClientHelloArtworkSupport(channels=[_ARTWORK_CHANNEL]) if legacy else None,
        visualizer_support=ClientHelloVisualizerSupport(
            buffer_capacity=10_000,
            rate_max=30 if legacy else None,
            types=["loudness"] if legacy else None,
        ),
    )


def _volume_command() -> ServerCommandMessage:
    return ServerCommandMessage(
        payload=ServerCommandPayload(
            player=PlayerCommandPayload(command=PlayerCommand.VOLUME, volume=10)
        )
    )


def _full_state() -> ClientStatePayload:
    return ClientStatePayload(
        available=True,
        player=_PLAYER_STATE,
        artwork=ClientStateArtwork(channels=[_ARTWORK_CHANNEL]),
        visualizer=VisualizerStatePayload(types=["loudness"], rate_max=30),
    )


async def _connect(
    hello: ClientHelloPayload, *, trusted: bool = True
) -> tuple[SendspinConnection, _FakeTransport]:
    """Connect an unpaired client; a trusted one also delivers its initial client/state."""
    loop = asyncio.get_running_loop()
    server = _MockServer(loop=loop, clock=LoopClock(loop))
    if trusted:
        await server.pairing_store.add_trusted_unpaired(TrustedUnpairedClient(client_id=CLIENT_ID))
    conn = SendspinConnection(server, wsock_client=AsyncMock())
    psk = generate_psk()
    conn._client_id = CLIENT_ID  # noqa: SLF001
    conn._noise_psk = ResolvedPsk(  # noqa: SLF001
        psk_id=psk_id_for(psk), psk=psk, category=PskCategory.SENTINEL, counterparty_id=CLIENT_ID
    )
    fake = _FakeTransport([WSMessage(WSMsgType.TEXT, ClientHelloMessage(hello).to_json(), "")])
    conn._transport = fake  # type: ignore[assignment]  # noqa: SLF001
    assert await conn._exchange_hellos()  # noqa: SLF001
    if trusted:
        await conn._handle_client_state(_full_state())  # noqa: SLF001
    fake.sent.clear()
    return conn, fake


async def _set_trusted(conn: SendspinConnection, *, trusted: bool) -> None:
    store = conn._server.pairing_store  # noqa: SLF001
    if trusted:
        await store.add_trusted_unpaired(TrustedUnpairedClient(client_id=CLIENT_ID))
    else:
        await store.remove_trusted_unpaired(CLIENT_ID)
    await conn.refresh_trusted_unpaired()


async def _drain_priority(conn: SendspinConnection, fake: _FakeTransport) -> list[str]:
    while await conn._process_priority_messages(fake):  # type: ignore[arg-type]  # noqa: SLF001
        pass
    return [payload["type"] for payload in fake.sent_payloads()]


def _client(conn: SendspinConnection) -> SendspinClient:
    client = conn._client  # noqa: SLF001
    assert client is not None
    return client


@pytest.mark.asyncio
async def test_removed_roles_are_torn_down_before_server_activate() -> None:
    """refresh_trusted_unpaired writes stream/end and the null state ahead of server/activate."""
    conn, fake = await _connect(_hello([Roles.PLAYER.value, Roles.METADATA.value]))
    player = _client(conn).role(Roles.PLAYER.value)
    assert isinstance(player, PlayerV1Role)
    player._stream_started = True  # noqa: SLF001
    # Anything still queued for a removed role must not follow the activation.
    conn.send_role_message("player", _volume_command())

    await _set_trusted(conn, trusted=False)

    # Reverse attach order: metadata unwinds before the player.
    assert await _drain_priority(conn, fake) == ["server/state", "stream/end", "server/activate"]
    assert fake.sent_payloads()[0]["payload"] == {"metadata": None}
    assert fake.sent_payloads()[2]["payload"]["active_roles"] == []
    assert not conn._role_queues.get("player")  # noqa: SLF001


@pytest.mark.asyncio
async def test_activate_tears_down_removed_roles_first() -> None:
    """_activate puts a removed role's teardown on the wire before server/activate."""
    conn, fake = await _connect(_hello([Roles.PLAYER.value]))
    player = _client(conn).role(Roles.PLAYER.value)
    assert isinstance(player, PlayerV1Role)
    player._stream_started = True  # noqa: SLF001
    conn._trusted_unpaired = False  # noqa: SLF001

    await conn._activate()  # noqa: SLF001

    assert [payload["type"] for payload in fake.sent_payloads()] == [
        "stream/end",
        "server/activate",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("family", ["player", "artwork", "visualizer"])
async def test_reactivated_role_waits_for_its_client_state(family: str) -> None:
    """A re-added role gets no stream and no binary until client/state carries its object."""
    conn, _fake = await _connect(
        _hello([Roles.PLAYER.value, Roles.ARTWORK.value, Roles.VISUALIZER.value])
    )
    client = _client(conn)
    await _set_trusted(conn, trusted=False)

    with patch.object(client.group, "on_role_activated") as activated:
        await _set_trusted(conn, trusted=True)
        role = client.roles_by_family(family)[0]
        assert client.awaits_role_state(family)
        assert not PushStream._role_in_audio_pipeline(client, role)  # noqa: SLF001
        activated.assert_not_called()

        conn.send_binary(b"stale", role=family, timestamp_us=0, message_type=0)
        conn.drop_pending_binary([family])  # a stream boundary while held
        conn.send_binary(b"fresh", role=family, timestamp_us=0, message_type=0)
        assert not conn._role_queues.get(family)  # noqa: SLF001

        def joins() -> list[object]:
            return [call.args[0] for call in activated.call_args_list if call.args[0] is role]

        # A client/state carrying only the other roles' objects keeps this one held.
        full = _full_state()
        without = ClientStatePayload(
            player=None if family == "player" else full.player,
            artwork=None if family == "artwork" else full.artwork,
            visualizer=None if family == "visualizer" else full.visualizer,
        )
        await conn._handle_client_state(without)  # noqa: SLF001
        assert client.awaits_role_state(family)
        assert joins() == []

        await conn._handle_client_state(full)  # noqa: SLF001
        assert not client.awaits_role_state(family)
        assert PushStream._role_in_audio_pipeline(client, role)  # noqa: SLF001
        assert joins() == [role]
        assert conn._activation_state_timeout_handle is None  # noqa: SLF001

    queued = [entry.binary for _, _, entry in conn._role_queues[family] if entry.binary]  # noqa: SLF001
    assert [binary.data for binary in queued] == [b"fresh"]


@pytest.mark.asyncio
@pytest.mark.parametrize("family", ["artwork", "visualizer"])
async def test_hello_configured_role_is_not_held(family: str) -> None:
    """A role configured by a pre-#195 hello streams on re-activation without a state object."""
    conn, _fake = await _connect(_hello([Roles.ARTWORK.value, Roles.VISUALIZER.value], legacy=True))
    client = _client(conn)
    await _set_trusted(conn, trusted=False)
    await _set_trusted(conn, trusted=True)

    assert not client.awaits_role_state(family)
    assert conn._activation_state_timeout_handle is None  # noqa: SLF001
    message_type = BinaryMessageType.ARTWORK_CHANNEL_0.value
    conn.send_binary(b"data", role=family, timestamp_us=0, message_type=message_type)
    assert conn._role_queues.get(family)  # noqa: SLF001


async def _reactivated_player() -> tuple[SendspinConnection, SendspinClient]:
    conn, _fake = await _connect(_hello([Roles.PLAYER.value]))
    await _set_trusted(conn, trusted=False)
    await _set_trusted(conn, trusted=True)
    client = _client(conn)
    assert client.awaits_role_state("player")
    return conn, client


@pytest.mark.asyncio
async def test_activation_state_deviations_are_flagged() -> None:
    """A re-added player's client/state is checked like an initial one."""
    conn, client = await _reactivated_player()
    partial = PlayerStatePayload(
        volume=50, muted=False, supported_commands=[PlayerCommand.VOLUME, PlayerCommand.MUTE]
    )
    with patch.object(conn, "_flag_noncompliance") as flag:
        await conn._handle_client_state(ClientStatePayload(player=partial))  # noqa: SLF001

    flag.assert_any_call("client/state after server/activate omitted required player timing fields")
    assert not client.awaits_role_state("player")


@pytest.mark.asyncio
async def test_held_role_is_flagged_and_started_when_its_state_never_arrives() -> None:
    """The activation's client/state timeout flags the client, then starts the role anyway."""
    conn, client = await _reactivated_player()
    handle = conn._activation_state_timeout_handle  # noqa: SLF001
    assert handle is not None
    handle.cancel()
    role = client.role(Roles.PLAYER.value)

    with (
        patch.object(conn, "_flag_noncompliance") as flag,
        patch.object(client.group, "on_role_activated") as activated,
    ):
        conn._activation_state_timeout_callback()  # noqa: SLF001

    flag.assert_called_once_with(
        "did not send the player client/state object after server/activate in time"
    )
    assert not client.awaits_role_state("player")
    activated.assert_called_once_with(role)


@pytest.mark.asyncio
async def test_leaving_pairing_holds_roles_until_client_state() -> None:
    """Roles restored after pairing wait for the client/state that follows server/activate."""
    conn, fake = await _connect(_hello([Roles.PLAYER.value, Roles.VISUALIZER.value]))
    client = _client(conn)
    await conn._quiesce_for_pairing()  # noqa: SLF001
    conn._in_pairing = True  # noqa: SLF001
    fake.sent.clear()

    with patch.object(conn, "_resume_writer"):
        await conn._leave_pairing()  # noqa: SLF001

    assert [payload["type"] for payload in fake.sent_payloads()] == ["server/activate"]
    assert client.awaits_role_state("player")
    assert client.awaits_role_state("visualizer")

    await conn._handle_client_state(_full_state())  # noqa: SLF001
    assert not client.awaits_role_state("player")
    assert not client.awaits_role_state("visualizer")


@pytest.mark.asyncio
async def test_reconnect_drops_a_leftover_hold() -> None:
    """A new connection starts without holds; its initial client/state gates the roles."""
    conn, client = await _reactivated_player()

    client.attach_connection(
        conn,
        client_info=client.info,
        negotiated_roles=client.negotiated_role_ids,
        active_roles=client.active_role_ids,
    )

    assert not client.awaits_role_state("player")


@pytest.mark.asyncio
async def test_reactivated_player_gets_no_command_before_its_state() -> None:
    """A re-added player receives no server/command until its client/state declares commands."""
    conn, _fake = await _connect(_hello([Roles.PLAYER.value]))
    client = _client(conn)
    await _set_trusted(conn, trusted=False)
    await _set_trusted(conn, trusted=True)
    player = client.role(Roles.PLAYER.value)
    assert isinstance(player, PlayerV1Role)

    def queued_commands() -> list[object]:
        return [
            message
            for message in conn._normal_messages  # noqa: SLF001
            if isinstance(message, ServerCommandMessage)
        ]

    player.set_volume(20)
    player.set_mute(True)
    assert queued_commands() == []

    await conn._handle_client_state(ClientStatePayload(player=_PLAYER_STATE))  # noqa: SLF001
    player.set_volume(20)
    assert len(queued_commands()) == 1


@pytest.mark.asyncio
async def test_role_added_after_a_stateless_connect_is_held_with_a_timeout() -> None:
    """A connection that needed no initial state holds roles activated later like any other."""
    conn, _fake = await _connect(_hello([Roles.PLAYER.value]), trusted=False)
    client = _client(conn)
    assert client.active_roles == ()
    assert client.is_connected

    await _set_trusted(conn, trusted=True)
    assert client.awaits_role_state("player")
    assert conn._activation_state_timeout_handle is not None  # noqa: SLF001

    with patch.object(conn._server, "on_client_first_connect") as first_connect:  # noqa: SLF001
        await conn._handle_client_state(ClientStatePayload(player=_PLAYER_STATE))  # noqa: SLF001

    first_connect.assert_not_called()
    assert not client.awaits_role_state("player")
    assert conn._activation_state_timeout_handle is None  # noqa: SLF001
