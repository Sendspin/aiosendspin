"""Role (de)activation around server/activate: teardown order and the client/state hold."""

from __future__ import annotations

import asyncio
import dataclasses
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, PropertyMock, patch

import orjson
import pytest
from aiohttp import WSMessage, WSMsgType

from aiosendspin.models.artwork import (
    ArtworkChannel,
    ClientHelloArtworkSupport,
    ClientStateArtwork,
    pack_artwork_cancel,
)
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
from aiosendspin.models.source import ClientHelloSourceSupport, SourceStatePayload
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
from aiosendspin.server.compliance import ClientComplianceError
from aiosendspin.server.connection import SendspinConnection
from aiosendspin.server.group import SendspinGroup
from aiosendspin.server.push_stream import PushStream
from aiosendspin.server.roles.artwork.v1 import ArtworkV1Role
from aiosendspin.server.roles.player.v1 import PlayerV1Role
from aiosendspin.server.roles.source.v1 import SourceV1Role
from tests.server.test_multi_server import _FakeTransport, _MockServer

if TYPE_CHECKING:
    from aiosendspin.server.client import SendspinClient

CLIENT_ID = "client-1"
_ARTWORK_CHANNEL = ArtworkChannel(
    source=ArtworkSource.ALBUM, format=PictureFormat.JPEG, width=300, height=300
)
_ALTERNATE_FORMAT = SupportedAudioFormat(
    codec=AudioCodec.PCM, channels=2, sample_rate=44100, bit_depth=16
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
                ),
                _ALTERNATE_FORMAT,
            ],
            buffer_capacity=100_000,
            supported_commands=[] if legacy else None,
        ),
        artwork_support=ClientHelloArtworkSupport(channels=[_ARTWORK_CHANNEL]) if legacy else None,
        source_support=ClientHelloSourceSupport(),
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
    hello: ClientHelloPayload,
    *,
    trusted: bool = True,
    send_state: bool = True,
    category: PskCategory = PskCategory.SENTINEL,
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
        psk_id=psk_id_for(psk), psk=psk, category=category, counterparty_id=CLIENT_ID
    )
    fake = _FakeTransport([WSMessage(WSMsgType.TEXT, ClientHelloMessage(hello).to_json(), "")])
    conn._transport = fake  # type: ignore[assignment]  # noqa: SLF001
    assert await conn._exchange_hellos()  # noqa: SLF001
    if trusted and send_state:
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
    """refresh_trusted_unpaired writes stream/end, and no state role null, ahead of activate."""
    state_roles = [Roles.METADATA.value, Roles.COLOR.value, Roles.CONTROLLER.value]
    conn, fake = await _connect(_hello([Roles.PLAYER.value, *state_roles]))
    assert all(_client(conn).role(role_id) is not None for role_id in state_roles)
    player = _client(conn).role(Roles.PLAYER.value)
    assert isinstance(player, PlayerV1Role)
    player._stream_started = True  # noqa: SLF001
    # Anything still queued for a removed role must not follow the activation.
    conn.send_role_message("player", _volume_command())

    await _set_trusted(conn, trusted=False)

    assert await _drain_priority(conn, fake) == ["stream/end", "server/activate"]
    assert fake.sent_payloads()[1]["payload"]["active_roles"] == []
    assert not conn._role_queues.get("player")  # noqa: SLF001


# DEPRECATED(spec-pr-275): remove in aiosendspin <version>
@pytest.mark.asyncio
async def test_removed_roles_send_legacy_null_state_before_server_activate() -> None:
    """A legacy-generation client gets the null metadata object ahead of server/activate."""
    conn, fake = await _connect(_hello([Roles.PLAYER.value, Roles.METADATA.value], legacy=True))
    player = _client(conn).role(Roles.PLAYER.value)
    assert isinstance(player, PlayerV1Role)
    player._stream_started = True  # noqa: SLF001

    await _set_trusted(conn, trusted=False)

    # Reverse attach order: metadata unwinds before the player.
    assert await _drain_priority(conn, fake) == ["server/state", "stream/end", "server/activate"]
    assert fake.sent_payloads()[0]["payload"] == {"metadata": None}


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


class _RecordingTransport(_FakeTransport):
    """Transport double that also records binary frames, in send order with text frames."""

    def __init__(self) -> None:
        super().__init__()
        self.frames: list[str | bytes] = []

    async def send_str(self, data: str) -> None:
        await super().send_str(data)
        self.frames.append(data)

    async def send_bytes(self, data: bytes) -> None:
        self.frames.append(data)


@pytest.mark.asyncio
async def test_removed_artwork_role_cancels_its_transfer_before_stream_end() -> None:
    """A transfer in flight on a removed artwork role is cancelled ahead of its stream/end."""
    conn, _fake = await _connect(_hello([Roles.ARTWORK.value]))
    artwork = _client(conn).roles_by_family("artwork")[0]
    assert isinstance(artwork, ArtworkV1Role)
    artwork._in_flight = 0  # noqa: SLF001
    recorder = _RecordingTransport()
    conn._transport = recorder  # type: ignore[assignment]  # noqa: SLF001

    await _set_trusted(conn, trusted=False)
    await _drain_priority(conn, recorder)

    assert recorder.frames[0] == pack_artwork_cancel(0)
    assert [orjson.loads(frame)["type"] for frame in recorder.frames[1:]] == [
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
async def test_removed_player_drops_its_queued_commands() -> None:
    """A server/command queued for a player being removed is not sent after server/activate."""
    conn, fake = await _connect(_hello([Roles.PLAYER.value]))
    player = _client(conn).role(Roles.PLAYER.value)
    assert isinstance(player, PlayerV1Role)
    player.set_volume(20)
    assert any(isinstance(m, ServerCommandMessage) for m in conn._normal_messages)  # noqa: SLF001

    await _set_trusted(conn, trusted=False)

    assert not any(isinstance(m, ServerCommandMessage) for m in conn._normal_messages)  # noqa: SLF001
    assert await _drain_priority(conn, fake) == ["server/activate"]


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


@pytest.mark.asyncio
async def test_held_player_format_change_joins_the_stream_once() -> None:
    """A re-added player whose state picks a new format joins only on release."""
    conn, client = await _reactivated_player()
    role = client.role(Roles.PLAYER.value)
    state = dataclasses.replace(_PLAYER_STATE, format=_ALTERNATE_FORMAT)

    with (
        patch.object(
            SendspinGroup, "has_active_stream", new_callable=PropertyMock, return_value=True
        ),
        patch.object(client.group, "on_role_format_changed") as format_changed,
        patch.object(client.group, "on_role_activated") as activated,
    ):
        await conn._handle_client_state(ClientStatePayload(player=state))  # noqa: SLF001

    format_changed.assert_not_called()
    activated.assert_called_once_with(role)


@pytest.mark.asyncio
async def test_initial_state_completes_after_its_roles_were_removed() -> None:
    """The first client/state is the initial one even when the roles that awaited it are gone."""
    conn, _fake = await _connect(_hello([Roles.PLAYER.value]), send_state=False)
    client = _client(conn)
    assert conn._initial_state_timeout_handle is not None  # noqa: SLF001
    await _set_trusted(conn, trusted=False)
    assert not client.is_connected

    await conn._handle_client_state(ClientStatePayload(available=True))  # noqa: SLF001

    assert client.is_connected
    assert conn._initial_state_timeout_handle is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_held_player_joins_with_the_timing_of_its_state() -> None:
    """The release join sees the timing values the releasing client/state reports."""
    conn, client = await _reactivated_player()
    player = client.role(Roles.PLAYER.value)
    assert isinstance(player, PlayerV1Role)
    state = dataclasses.replace(
        _PLAYER_STATE, output_delay_ms=40, required_lead_time_ms=500, min_buffer_ms=2000
    )
    timing_at_join: list[tuple[int, int, int]] = []

    def record_join(role: PlayerV1Role) -> None:
        timing_at_join.append(
            (role.output_delay_ms, role.required_lead_time_ms, role.min_buffer_ms)
        )

    with patch.object(client.group, "on_role_activated", side_effect=record_join):
        await conn._handle_client_state(ClientStatePayload(player=state))  # noqa: SLF001

    assert timing_at_join == [(40, 500, 2000)]


async def _reactivated_source() -> tuple[SendspinConnection, SendspinClient, SourceV1Role]:
    conn, _fake = await _connect(
        _hello([Roles.PLAYER.value, "source@v1"]), category=PskCategory.LONG_TERM
    )
    conn._send_activation([Roles.PLAYER.value])  # noqa: SLF001
    conn._send_activation([Roles.PLAYER.value, "source@v1"])  # noqa: SLF001
    client = _client(conn)
    source = client.role("source@v1")
    assert isinstance(source, SourceV1Role)
    assert client.awaits_role_state("source")
    return conn, client, source


@pytest.mark.asyncio
async def test_reactivated_source_start_waits_for_its_client_state() -> None:
    """A start requested for a held source is sent when client/state releases the hold."""
    conn, client, source = await _reactivated_source()

    with patch.object(source, "send_message") as send:
        source.request_start()
        await conn._handle_client_state(  # noqa: SLF001
            ClientStatePayload(available=True, player=_PLAYER_STATE)
        )
        assert client.awaits_role_state("source")
        send.assert_not_called()

        with patch.object(conn, "_flag_noncompliance") as flag:
            await conn._handle_client_state(  # noqa: SLF001
                ClientStatePayload(
                    available=True, player=_PLAYER_STATE, source=SourceStatePayload()
                )
            )
        flag.assert_not_called()
        assert not client.awaits_role_state("source")
        assert conn._activation_state_timeout_handle is None  # noqa: SLF001
        send.assert_called_once()


@pytest.mark.asyncio
async def test_source_released_by_timeout_keeps_its_start_queued() -> None:
    """The lenient timeout release does not stand in for the missing source object."""
    conn, client, source = await _reactivated_source()
    handle = conn._activation_state_timeout_handle  # noqa: SLF001
    assert handle is not None
    handle.cancel()

    with patch.object(source, "send_message") as send:
        source.request_start()
        with patch.object(conn, "_flag_noncompliance") as flag:
            conn._activation_state_timeout_callback()  # noqa: SLF001

        flag.assert_called_once_with(
            "did not send the source client/state object after server/activate in time"
        )
        assert not client.awaits_role_state("source")
        send.assert_not_called()

        await conn._handle_client_state(  # noqa: SLF001
            ClientStatePayload(available=True, source=SourceStatePayload())
        )
    send.assert_called_once()


async def _source_awaiting_initial_state(
    *, strict: bool
) -> tuple[SendspinConnection, SourceV1Role]:
    conn, _fake = await _connect(
        _hello([Roles.PLAYER.value, "source@v1"]),
        send_state=False,
        category=PskCategory.LONG_TERM,
    )
    conn._server.allow_noncompliant_clients = not strict  # type: ignore[misc]  # noqa: SLF001
    source = _client(conn).role("source@v1")
    assert isinstance(source, SourceV1Role)
    return conn, source


@pytest.mark.asyncio
async def test_start_requested_before_the_initial_state_is_sent_by_it() -> None:
    """A start requested right after connecting goes out once the initial client/state lands."""
    conn, source = await _source_awaiting_initial_state(strict=False)

    with patch.object(source, "send_message") as send:
        source.request_start()
        send.assert_not_called()
        await conn._handle_client_state(  # noqa: SLF001
            ClientStatePayload(available=True, player=_PLAYER_STATE, source=SourceStatePayload())
        )
        await conn._handle_client_state(  # noqa: SLF001
            ClientStatePayload(available=True, player=_PLAYER_STATE, source=SourceStatePayload())
        )

    send.assert_called_once()


@pytest.mark.asyncio
async def test_start_requested_on_connect_waits_for_initial_availability() -> None:
    """A start requested as the client connects waits while its initial state is unavailable."""
    conn, source = await _source_awaiting_initial_state(strict=False)

    with (
        patch.object(source, "send_message") as send,
        patch.object(
            conn._server,  # noqa: SLF001
            "on_client_first_connect",
            side_effect=lambda _client_id: source.request_start(),
        ),
    ):
        await conn._handle_client_state(  # noqa: SLF001
            ClientStatePayload(available=False, player=_PLAYER_STATE, source=SourceStatePayload())
        )
        send.assert_not_called()

        await conn._handle_client_state(  # noqa: SLF001
            ClientStatePayload(available=True, player=_PLAYER_STATE, source=SourceStatePayload())
        )
    send.assert_called_once()


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
@pytest.mark.asyncio
async def test_lenient_server_starts_source_without_source_object() -> None:
    """A tolerated client whose initial client/state lacks the source object is startable."""
    conn, source = await _source_awaiting_initial_state(strict=False)

    with (
        patch.object(conn, "_flag_noncompliance") as flag,
        patch.object(source, "send_message") as send,
    ):
        source.request_start()
        await conn._handle_client_state(  # noqa: SLF001
            ClientStatePayload(available=True, player=_PLAYER_STATE)
        )

    flag.assert_any_call("initial client/state has an active source role but no source state")
    send.assert_called_once()


@pytest.mark.asyncio
async def test_strict_server_rejects_initial_state_without_source_object() -> None:
    """A strict server rejects the client instead of starting it without its source object."""
    conn, source = await _source_awaiting_initial_state(strict=True)

    with patch.object(source, "send_message") as send:
        source.request_start()
        with pytest.raises(ClientComplianceError, match="no source state"):
            await conn._handle_client_state(  # noqa: SLF001
                ClientStatePayload(available=True, player=_PLAYER_STATE)
            )

    assert not source.can_start
    send.assert_not_called()


@pytest.mark.asyncio
async def test_unsent_stream_end_still_precedes_server_activate() -> None:
    """A stream/end still queued for a removed role is sent, not discarded with its queue."""
    conn, fake = await _connect(_hello([Roles.PLAYER.value]))
    player = _client(conn).role(Roles.PLAYER.value)
    assert isinstance(player, PlayerV1Role)
    player._stream_started = True  # noqa: SLF001
    player.on_stream_end()

    await _set_trusted(conn, trusted=False)

    assert await _drain_priority(conn, fake) == ["stream/end", "server/activate"]


@pytest.mark.asyncio
async def test_activation_timeout_does_not_run_while_the_releasing_state_is_applied() -> None:
    """The timeout is off while a releasing client/state is dispatched, so the join runs once."""
    conn, client = await _reactivated_player()
    role = client.role(Roles.PLAYER.value)
    handles_during_dispatch: list[object] = []

    async def record_handle(*, available: bool) -> None:  # noqa: ARG001
        handles_during_dispatch.append(conn._activation_state_timeout_handle)  # noqa: SLF001

    with (
        patch.object(client, "handle_availability_change", side_effect=record_handle),
        patch.object(client.group, "on_role_activated") as activated,
    ):
        await conn._handle_client_state(  # noqa: SLF001
            ClientStatePayload(available=False, player=_PLAYER_STATE)
        )
        conn._release_roles([role])  # noqa: SLF001

    assert handles_during_dispatch == [None]
    activated.assert_called_once_with(role)
    assert conn._activation_state_timeout_handle is None  # noqa: SLF001
