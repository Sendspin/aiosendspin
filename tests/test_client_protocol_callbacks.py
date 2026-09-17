"""Tests for public protocol callback hooks on the Sendspin client."""

from __future__ import annotations

import asyncio
import json
import struct
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiosendspin.client import SendspinClient
from aiosendspin.client.connection import SendspinConnection
from aiosendspin.client.models import AudioFormat
from aiosendspin.clock import ManualClock
from aiosendspin.models.artwork import (
    ArtworkChannel,
    StreamArtworkChannelConfig,
    StreamStartArtwork,
    pack_artwork_announce,
    pack_artwork_parts,
)
from aiosendspin.models.controller import ControllerStatePayload
from aiosendspin.models.core import (
    ActivatePairing,
    ServerActivatePayload,
    ServerCommandPayload,
    ServerHelloPayload,
    ServerStatePayload,
    ServerTimePayload,
    StreamClearMessage,
    StreamClearPayload,
    StreamEndMessage,
    StreamEndPayload,
    StreamStartMessage,
    StreamStartPayload,
)
from aiosendspin.models.metadata import Progress, SessionUpdateMetadata
from aiosendspin.models.player import (
    SEND_AHEAD_MAX,
    ClientHelloPlayerSupport,
    PlayerCommandPayload,
    StreamStartPlayer,
    SupportedAudioFormat,
    pack_player_audio_header,
)
from aiosendspin.models.source import (
    ClientHelloSourceFeatures,
    ClientHelloSourceSupport,
    ServerHelloSourceSupport,
)
from aiosendspin.models.types import (
    Activity,
    ArtworkSource,
    AudioCodec,
    BinaryMessageType,
    GoodbyeReason,
    MediaCommand,
    PairMethod,
    PictureFormat,
    PlayerCommand,
    RepeatMode,
    Roles,
    SignalState,
)
from aiosendspin.models.visualizer import (
    ClientHelloVisualizerSpectrum,
    ClientHelloVisualizerSupport,
    VisualizerFrame,
    VisualizerStatePayload,
)
from aiosendspin.noise.keys import generate_psk, psk_id_for
from aiosendspin.noise.trust_store import (
    InMemoryClientPairingStore,
    PskCategory,
    ResolvedPsk,
)

from .conftest import make_sdk_client


def _player_support() -> ClientHelloPlayerSupport:
    return ClientHelloPlayerSupport(
        supported_formats=[
            SupportedAudioFormat(
                codec=AudioCodec.PCM,
                sample_rate=48_000,
                bit_depth=16,
                channels=2,
            )
        ],
        buffer_capacity=100_000,
    )


async def _exchange_hellos_with(
    monkeypatch: pytest.MonkeyPatch, hello: ServerHelloPayload
) -> SendspinClient:
    """Drive the client through a hello exchange against ``hello``."""
    client = make_sdk_client(
        client_name="Test Client",
        roles=[Roles.PLAYER],
        player_support=_player_support(),
    )

    connection = SendspinConnection(client)
    client._admitted_connection = connection  # noqa: SLF001
    # server_id comes from the Noise handshake, not the hello payload.
    connection._server_id = "server-1"  # noqa: SLF001

    async def receive_hello() -> ServerHelloPayload:
        return hello

    async def send_client_hello() -> None: ...

    async def receive_activate() -> ServerActivatePayload:
        return ServerActivatePayload(activities=[])

    monkeypatch.setattr(connection, "_receive_server_hello", receive_hello)
    monkeypatch.setattr(connection, "_send_client_hello", send_client_hello)
    monkeypatch.setattr(connection, "_receive_server_activate", receive_activate)

    await connection._exchange_hellos()  # noqa: SLF001
    return client


@pytest.mark.asyncio
async def test_server_hello_populates_server_info(monkeypatch: pytest.MonkeyPatch) -> None:
    """Receiving server/hello records the server's name."""
    client = await _exchange_hellos_with(monkeypatch, ServerHelloPayload(name="Test Server"))

    assert client.server_info is not None
    assert client.server_info.server_id == "server-1"
    # A server without source support leaves the accepted codecs unknown.
    assert client.server_info.source_codecs is None


@pytest.mark.asyncio
async def test_server_hello_records_accepted_source_codecs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The codecs a server accepts from sources are recorded for source captures."""
    hello = ServerHelloPayload(
        name="Test Server",
        source_support=ServerHelloSourceSupport(supported_codecs=[AudioCodec.FLAC, AudioCodec.PCM]),
    )

    client = await _exchange_hellos_with(monkeypatch, hello)

    assert client.server_info is not None
    assert client.server_info.source_codecs == frozenset({AudioCodec.FLAC, AudioCodec.PCM})


async def _connection(
    category: PskCategory, *, unpaired_access: bool = False, **client_kwargs: Any
) -> SendspinConnection:
    store = InMemoryClientPairingStore()
    await store.store_pairing_config(
        replace(await store.get_pairing_config(), unpaired_access_enabled=unpaired_access)
    )
    client_kwargs.setdefault("roles", [Roles.PLAYER, Roles.ARTWORK])
    client = make_sdk_client(
        client_name="Test Client",
        player_support=_player_support(),
        artwork_channels=_artwork_channels(),
        pairing_store=store,
        **client_kwargs,
    )
    connection = SendspinConnection(client)
    psk = generate_psk()
    connection._noise_psk = ResolvedPsk(  # noqa: SLF001
        psk_id=psk_id_for(psk),
        psk=psk,
        category=category,
        counterparty_id="server-1",
    )
    return connection


async def _sentinel_connection(*, unpaired_access: bool) -> SendspinConnection:
    return await _connection(PskCategory.SENTINEL, unpaired_access=unpaired_access)


@pytest.mark.asyncio
async def test_sentinel_role_activation_rejected_without_unpaired_access() -> None:
    """On Sentinel, a server activating roles without unpaired access is refused."""
    connection = await _sentinel_connection(unpaired_access=False)
    reason = await connection._apply_activation(  # noqa: SLF001
        ServerActivatePayload(activities=[], active_roles=[Roles.PLAYER.value])
    )
    assert reason is GoodbyeReason.PAIRING_REQUIRED


@pytest.mark.asyncio
async def test_sentinel_role_activation_admitted_with_unpaired_access() -> None:
    """On Sentinel, role activation is admitted when the client allows unpaired access."""
    connection = await _sentinel_connection(unpaired_access=True)
    reason = await connection._apply_activation(  # noqa: SLF001
        ServerActivatePayload(activities=[], active_roles=[Roles.PLAYER.value])
    )
    assert reason is None


@pytest.mark.asyncio
async def test_sentinel_idle_activation_admitted_without_unpaired_access() -> None:
    """On Sentinel, an idle activation (no roles, no playback) is admitted regardless."""
    connection = await _sentinel_connection(unpaired_access=False)
    reason = await connection._apply_activation(  # noqa: SLF001
        ServerActivatePayload(activities=[], active_roles=[])
    )
    assert reason is None


@pytest.mark.asyncio
async def test_sentinel_playback_activity_rejected_without_unpaired_access() -> None:
    """On Sentinel, declaring the playback activity is refused without unpaired access."""
    connection = await _sentinel_connection(unpaired_access=False)
    reason = await connection._apply_activation(  # noqa: SLF001
        ServerActivatePayload(activities=[Activity.PLAYBACK], active_roles=[])
    )
    assert reason is GoodbyeReason.PAIRING_REQUIRED


async def _admitted_connection(
    category: PskCategory, activation: ServerActivatePayload
) -> tuple[SendspinConnection, list[dict[str, Any]]]:
    """Admit a live connection under ``activation`` with unpaired access enabled.

    Returns the connection and the JSON messages it sends.
    """
    connection = await _connection(category, unpaired_access=True)
    sent: list[dict[str, Any]] = []

    async def _capture(payload: str) -> None:
        sent.append(json.loads(payload))

    connection._ws = MagicMock(closed=False, send_str=_capture, close=AsyncMock())  # noqa: SLF001
    connection._connected = True  # noqa: SLF001
    assert await connection._apply_activation(activation) is None  # noqa: SLF001
    connection._client._admitted_connection = connection  # noqa: SLF001
    return connection, sent


@pytest.mark.parametrize(
    "activation",
    [
        pytest.param(ServerActivatePayload(activities=[Activity.PLAYBACK]), id="playback"),
        pytest.param(
            ServerActivatePayload(activities=[], active_roles=[Roles.PLAYER.value]), id="roles"
        ),
        pytest.param(
            ServerActivatePayload(
                activities=[Activity.PLAYBACK, Activity.PAIRING],
                active_roles=[Roles.PLAYER.value],
                pairing=ActivatePairing(method=PairMethod.DYNAMIC_PAIRING_CODE, format="digits"),
            ),
            id="playback_and_pairing",
        ),
    ],
)
async def test_disabling_unpaired_access_closes_a_connection_relying_on_it(
    activation: ServerActivatePayload,
) -> None:
    """An unpaired connection with playback or roles is closed with pairing_required."""
    connection, sent = await _admitted_connection(PskCategory.SENTINEL, activation)
    client = connection._client  # noqa: SLF001

    await client.set_unpaired_access(enabled=False)

    assert not (await client.pairing_store.get_pairing_config()).unpaired_access_enabled
    assert sent == [{"type": "client/goodbye", "payload": {"reason": "pairing_required"}}]
    assert not connection.connected
    assert not client.connected


@pytest.mark.parametrize(
    ("category", "activation"),
    [
        pytest.param(
            PskCategory.LONG_TERM,
            ServerActivatePayload(
                activities=[Activity.PLAYBACK], active_roles=[Roles.PLAYER.value]
            ),
            id="paired",
        ),
        pytest.param(
            PskCategory.SENTINEL,
            ServerActivatePayload(
                activities=[Activity.PAIRING],
                pairing=ActivatePairing(method=PairMethod.DYNAMIC_PAIRING_CODE, format="digits"),
            ),
            id="unpaired_pairing_only",
        ),
        pytest.param(
            PskCategory.SENTINEL, ServerActivatePayload(activities=[]), id="unpaired_idle"
        ),
    ],
)
async def test_disabling_unpaired_access_keeps_a_connection_not_relying_on_it(
    category: PskCategory, activation: ServerActivatePayload
) -> None:
    """A paired connection, or an unpaired one without playback or roles, stays open."""
    connection, sent = await _admitted_connection(category, activation)
    client = connection._client  # noqa: SLF001

    await client.set_unpaired_access(enabled=False)

    assert not (await client.pairing_store.get_pairing_config()).unpaired_access_enabled
    assert sent == []
    assert client.connected


async def test_enabling_unpaired_access_closes_nothing() -> None:
    """Enabling persists the setting and leaves the admitted connection alone."""
    connection, sent = await _admitted_connection(
        PskCategory.SENTINEL, ServerActivatePayload(activities=[Activity.PLAYBACK])
    )
    client = connection._client  # noqa: SLF001
    await client.pairing_store.store_pairing_config(
        replace(await client.pairing_store.get_pairing_config(), unpaired_access_enabled=False)
    )
    await client.set_unpaired_access(enabled=True)

    assert (await client.pairing_store.get_pairing_config()).unpaired_access_enabled
    assert sent == []
    assert client.connected


async def test_setting_unpaired_access_without_a_connection_persists_it() -> None:
    """With no admitted connection the setting is stored and later hellos advertise it."""
    connection = await _connection(PskCategory.SENTINEL, unpaired_access=True)
    client = connection._client  # noqa: SLF001

    await client.set_unpaired_access(enabled=False)

    hello = await connection._build_client_hello()  # noqa: SLF001
    assert hello.payload.unpaired_access is not None
    assert hello.payload.unpaired_access.enabled is False


async def test_unpaired_access_store_failure_closes_nothing() -> None:
    """A failed write propagates and the connection relying on the old setting stays open."""
    connection, sent = await _admitted_connection(
        PskCategory.SENTINEL, ServerActivatePayload(activities=[Activity.PLAYBACK])
    )
    client = connection._client  # noqa: SLF001
    client.pairing_store.store_pairing_config = AsyncMock(  # type: ignore[method-assign]
        side_effect=OSError("disk full")
    )

    with pytest.raises(OSError, match="disk full"):
        await client.set_unpaired_access(enabled=False)

    assert sent == []
    assert client.connected


@pytest.mark.asyncio
async def test_start_runs_pairing_alongside_reader_and_time_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pairing activation on connect starts the attempt next to the steady-state tasks."""
    connection = await _connection(PskCategory.SENTINEL)
    connection._activities = [Activity.PAIRING]  # noqa: SLF001 — drive the is_pairing branch
    connection._ws = MagicMock()  # noqa: SLF001
    connection._connected = True  # noqa: SLF001
    started: list[int] = []

    async def _fake_pair(_ws: object, pairing_index: int) -> None:
        started.append(pairing_index)

    async def _idle() -> None:
        return

    monkeypatch.setattr(connection, "_pair", _fake_pair)
    monkeypatch.setattr(connection, "_reader_loop", _idle)
    monkeypatch.setattr(connection, "_time_sync_loop", _idle)
    monkeypatch.setattr("aiosendspin.client.connection.QueuedEncryptedWebSocket", MagicMock())
    await connection.start()
    await asyncio.sleep(0)
    assert started == [1]
    assert connection._reader_task is not None  # noqa: SLF001
    assert connection._time_task is not None  # noqa: SLF001


@pytest.mark.parametrize(
    ("category", "activities", "roles", "unpaired", "expected"),
    [
        # Long-term PSK: [] or ['playback'], plus management.
        (PskCategory.LONG_TERM, [], [], False, None),
        (PskCategory.LONG_TERM, [Activity.PLAYBACK], [Roles.PLAYER.value], False, None),
        (PskCategory.LONG_TERM, [Activity.MANAGEMENT], [], False, None),
        (PskCategory.LONG_TERM, [Activity.PLAYBACK, Activity.MANAGEMENT], [], False, None),
        # Roles allowed without 'playback' in activities: the set is playback-capable.
        (PskCategory.LONG_TERM, [Activity.MANAGEMENT], [Roles.PLAYER.value], False, None),
        # ['pairing'] alone re-verifies the pairing; it is not playback-capable.
        (PskCategory.LONG_TERM, [Activity.PAIRING], [], False, None),
        (
            PskCategory.LONG_TERM,
            [Activity.PAIRING],
            [Roles.PLAYER.value],
            False,
            GoodbyeReason.UNAUTHORIZED,
        ),
        # Pairing never runs alongside playback on a long-term PSK.
        (
            PskCategory.LONG_TERM,
            [Activity.PAIRING, Activity.PLAYBACK],
            [],
            False,
            GoodbyeReason.UNAUTHORIZED,
        ),
        (
            PskCategory.LONG_TERM,
            [Activity.PAIRING, Activity.MANAGEMENT],
            [],
            False,
            GoodbyeReason.UNAUTHORIZED,
        ),
        # Pairing PSK: [], ['pairing'], and with unpaired access also with 'playback'.
        (PskCategory.PAIRING, [], [], False, None),
        (PskCategory.PAIRING, [Activity.PAIRING], [], False, None),
        (PskCategory.PAIRING, [Activity.PLAYBACK], [], True, None),
        (
            PskCategory.PAIRING,
            [Activity.PLAYBACK, Activity.PAIRING],
            [Roles.PLAYER.value],
            True,
            None,
        ),
        (PskCategory.PAIRING, [Activity.PAIRING], [Roles.PLAYER.value], True, None),
        (PskCategory.PAIRING, [Activity.PLAYBACK], [], False, GoodbyeReason.PAIRING_REQUIRED),
        (
            PskCategory.PAIRING,
            [Activity.PLAYBACK, Activity.PAIRING],
            [],
            False,
            GoodbyeReason.PAIRING_REQUIRED,
        ),
        (
            PskCategory.PAIRING,
            [Activity.PAIRING],
            [Roles.PLAYER.value],
            False,
            GoodbyeReason.PAIRING_REQUIRED,
        ),
        (PskCategory.PAIRING, [Activity.MANAGEMENT], [], True, GoodbyeReason.UNAUTHORIZED),
        # Sentinel: the same sets as the pairing PSK.
        (PskCategory.SENTINEL, [], [], False, None),
        (PskCategory.SENTINEL, [Activity.PAIRING], [], False, None),
        (PskCategory.SENTINEL, [Activity.PLAYBACK], [Roles.PLAYER.value], True, None),
        (PskCategory.SENTINEL, [Activity.PLAYBACK, Activity.PAIRING], [], True, None),
        (
            PskCategory.SENTINEL,
            [Activity.PAIRING, Activity.PLAYBACK],
            [],
            False,
            GoodbyeReason.PAIRING_REQUIRED,
        ),
        # Management is the real problem here, so unauthorized wins over pairing_required.
        (
            PskCategory.SENTINEL,
            [Activity.PLAYBACK, Activity.MANAGEMENT],
            [],
            False,
            GoodbyeReason.UNAUTHORIZED,
        ),
        (PskCategory.SENTINEL, [Activity.MANAGEMENT], [], False, GoodbyeReason.UNAUTHORIZED),
    ],
)
@pytest.mark.asyncio
async def test_activation_admissibility(
    category: PskCategory,
    activities: list[Activity],
    roles: list[str],
    unpaired: bool,  # noqa: FBT001
    expected: GoodbyeReason | None,
) -> None:
    """server/activate enforcement reproduces the per-PSK activity-set table."""
    connection = await _connection(category, unpaired_access=unpaired)
    reason = await connection._apply_activation(  # noqa: SLF001
        ServerActivatePayload(activities=activities, active_roles=roles)
    )
    assert reason is expected


@pytest.mark.asyncio
async def test_unrecognized_activity_is_ignored_and_activation_applies() -> None:
    """A server/activate naming an unknown activity still applies its known fields."""
    connection = await _connection(PskCategory.LONG_TERM)
    handled: list[ServerActivatePayload] = []

    async def _record(payload: ServerActivatePayload) -> None:
        handled.append(payload)
        assert await connection._apply_activation(payload) is None  # noqa: SLF001

    connection._handle_server_activate = _record  # type: ignore[method-assign]  # noqa: SLF001

    await connection._handle_json_message(  # noqa: SLF001
        json.dumps(
            {
                "type": "server/activate",
                "payload": {
                    "activities": ["playback", "teleport"],
                    "active_roles": [Roles.PLAYER.value],
                },
            }
        )
    )

    (payload,) = handled
    assert payload.activities == [Activity.PLAYBACK]
    assert payload.ignored_activities == ["teleport"]
    assert connection._activities == [Activity.PLAYBACK]  # noqa: SLF001
    assert connection._active_roles == [Roles.PLAYER.value]  # noqa: SLF001


@pytest.mark.parametrize(
    ("category", "unpaired", "activities"),
    [
        # Pairing on a long-term PSK is never playback-capable.
        (PskCategory.LONG_TERM, False, [Activity.PAIRING]),
        # Without unpaired access an unpaired session is never playback-capable.
        (PskCategory.SENTINEL, False, [Activity.PAIRING]),
        (PskCategory.PAIRING, False, []),
    ],
)
@pytest.mark.asyncio
async def test_persisted_roles_lapse_when_no_longer_playback_capable(
    category: PskCategory,
    unpaired: bool,  # noqa: FBT001
    activities: list[Activity],
) -> None:
    """An activation that omits active_roles on a non-playback-capable set empties them."""
    connection = await _connection(category, unpaired_access=unpaired)
    connection._active_roles = [Roles.PLAYER.value]  # noqa: SLF001

    reason = await connection._apply_activation(  # noqa: SLF001
        ServerActivatePayload(activities=activities)
    )

    assert reason is None
    assert connection._active_roles == []  # noqa: SLF001


@pytest.mark.asyncio
async def test_persisted_roles_stay_while_playback_capable() -> None:
    """Adding 'pairing' next to playback keeps the persisted roles."""
    connection = await _connection(PskCategory.SENTINEL, unpaired_access=True)
    connection._active_roles = [Roles.PLAYER.value]  # noqa: SLF001

    reason = await connection._apply_activation(  # noqa: SLF001
        ServerActivatePayload(
            activities=[Activity.PLAYBACK, Activity.PAIRING],
            pairing=ActivatePairing(method=PairMethod.DYNAMIC_PAIRING_CODE, format="digits"),
        )
    )

    assert reason is None
    assert connection._active_roles == [Roles.PLAYER.value]  # noqa: SLF001


@pytest.mark.asyncio
async def test_lapsed_source_role_ends_its_stream() -> None:
    """A source role emptied by a pairing activation on a long-term PSK ends its stream."""
    connection = await _connection(
        PskCategory.LONG_TERM,
        roles=[Roles.SOURCE],
        source_support=ClientHelloSourceSupport(
            features=ClientHelloSourceFeatures(line_sense=True)
        ),
    )
    connection._active_roles = [Roles.SOURCE.value]  # noqa: SLF001
    connection._source_stream_active = True  # noqa: SLF001
    ended: list[bool] = []

    async def send_client_stream_end() -> None:
        ended.append(True)

    connection.send_client_stream_end = send_client_stream_end  # type: ignore[method-assign]
    connection._ws = MagicMock(closed=False)  # noqa: SLF001
    connection._connected = True  # noqa: SLF001

    reason = await connection._apply_activation(  # noqa: SLF001
        ServerActivatePayload(activities=[Activity.PAIRING])
    )

    assert reason is None
    assert connection._active_roles == []  # noqa: SLF001
    assert ended == [True]


@pytest.mark.asyncio
async def test_artwork_listener_receives_binary_frames_after_artwork_stream_start() -> None:
    """Client should expose artwork binary frames without private overrides."""
    client = make_sdk_client(
        client_name="Test Client",
        roles=[Roles.ARTWORK],
        artwork_channels=_artwork_channels(),
    )
    captured: list[tuple[int, bytes]] = []
    client.add_artwork_listener(lambda channel, data: captured.append((channel, data)))

    connection = SendspinConnection(client)
    await connection._handle_stream_start(  # noqa: SLF001
        StreamStartMessage(
            payload=StreamStartPayload(
                artwork=StreamStartArtwork(
                    channels=[
                        StreamArtworkChannelConfig(
                            source=ArtworkSource.ALBUM,
                            format=PictureFormat.JPEG,
                            width=512,
                            height=512,
                        )
                    ]
                )
            )
        )
    )

    payload = b"artwork-bytes"
    connection._handle_binary_message(pack_artwork_announce(0, 123_456, len(payload)))  # noqa: SLF001
    connection._handle_binary_message(next(pack_artwork_parts(0, payload)))  # noqa: SLF001

    assert captured == [(0, payload)]


def _artwork_channels() -> list[ArtworkChannel]:
    return [
        ArtworkChannel(
            source=ArtworkSource.ALBUM,
            format=PictureFormat.JPEG,
            width=256,
            height=256,
        )
    ]


def _visualizer_support() -> ClientHelloVisualizerSupport:
    return ClientHelloVisualizerSupport(buffer_capacity=4096)


_VISUALIZER_STATE = VisualizerStatePayload(types=["loudness"], rate_max=30)


def _stream_start_player() -> StreamStartPlayer:
    return StreamStartPlayer(
        codec=AudioCodec.PCM,
        sample_rate=48_000,
        channels=2,
        bit_depth=16,
    )


def _artwork_stream_start() -> StreamStartMessage:
    return StreamStartMessage(
        payload=StreamStartPayload(
            artwork=StreamStartArtwork(
                channels=[
                    StreamArtworkChannelConfig(
                        source=ArtworkSource.ALBUM,
                        format=PictureFormat.JPEG,
                        width=512,
                        height=512,
                    )
                ]
            )
        )
    )


@pytest.mark.asyncio
async def test_stream_start_with_only_application_objects_reaches_listener() -> None:
    """A stream/start for application-specific roles alone is delivered to the embedder."""
    client = make_sdk_client(
        client_name="Test Client",
        roles=[Roles.PLAYER],
        player_support=_player_support(),
    )
    captured: list[StreamStartMessage] = []
    client.add_stream_start_listener(captured.append)
    connection = SendspinConnection(client)
    message = StreamStartMessage(
        payload=StreamStartPayload(application_objects={"_acme": {"session": 1}})
    )

    await connection._handle_stream_start(message)  # noqa: SLF001

    assert captured == [message]


@pytest.mark.asyncio
async def test_application_binary_ids_reach_application_listener() -> None:
    """IDs 192-255 go to the application binary listener; other unknown IDs are dropped."""
    client = make_sdk_client(
        client_name="Test Client",
        roles=[Roles.PLAYER],
        player_support=_player_support(),
    )
    captured: list[tuple[int, bytes]] = []
    remove = client.add_application_binary_listener(
        lambda message_id, data: captured.append((message_id, data))
    )
    connection = SendspinConnection(client)

    connection._handle_binary_message(bytes([192]) + b"first")  # noqa: SLF001
    connection._handle_binary_message(bytes([255]))  # noqa: SLF001
    connection._handle_binary_message(bytes([191]) + b"reserved")  # noqa: SLF001
    connection._handle_binary_message(bytes([2]) + b"reserved")  # noqa: SLF001
    remove()
    connection._handle_binary_message(bytes([200]) + b"after-remove")  # noqa: SLF001

    assert captured == [(192, b"first"), (255, b"")]


@pytest.mark.asyncio
async def test_artwork_binary_dropped_when_only_player_stream_active() -> None:
    """Artwork binaries must be rejected when only the player stream is active."""
    client = make_sdk_client(
        client_name="Test Client",
        roles=[Roles.PLAYER, Roles.ARTWORK],
        player_support=_player_support(),
        artwork_channels=_artwork_channels(),
    )
    captured: list[tuple[int, bytes]] = []
    client.add_artwork_listener(lambda channel, data: captured.append((channel, data)))

    connection = SendspinConnection(client)
    await connection._handle_stream_start(  # noqa: SLF001
        StreamStartMessage(payload=StreamStartPayload(player=_stream_start_player()))
    )

    connection._handle_binary_message(pack_artwork_announce(0, 123_456, 3))  # noqa: SLF001
    connection._handle_binary_message(next(pack_artwork_parts(0, b"art")))  # noqa: SLF001

    assert captured == []


@pytest.mark.asyncio
async def test_audio_binary_dropped_when_only_artwork_stream_active() -> None:
    """Audio binaries must be rejected when only the artwork stream is active."""
    client = make_sdk_client(
        client_name="Test Client",
        roles=[Roles.PLAYER, Roles.ARTWORK],
        player_support=_player_support(),
        artwork_channels=_artwork_channels(),
    )
    captured: list[tuple[int, bytes, AudioFormat, int]] = []
    client.add_audio_chunk_listener(
        lambda ts, data, fmt, send_ahead: captured.append((ts, data, fmt, send_ahead)),
    )

    connection = SendspinConnection(client)
    await connection._handle_stream_start(_artwork_stream_start())  # noqa: SLF001

    connection._handle_binary_message(  # noqa: SLF001
        pack_player_audio_header(123_456, 50_000) + b"\x00\x00\x00\x00"
    )

    assert captured == []


@pytest.mark.parametrize("send_ahead", [0, 50_000, SEND_AHEAD_MAX])
@pytest.mark.asyncio
async def test_audio_binary_passes_raw_send_ahead(send_ahead: int) -> None:
    """The 13-byte audio header's send_ahead, saturated or not, reaches the listener as sent."""
    client = make_sdk_client(
        client_name="Test Client",
        roles=[Roles.PLAYER],
        player_support=_player_support(),
    )
    captured: list[tuple[int, bytes, int]] = []
    client.add_audio_chunk_listener(
        lambda ts, data, _fmt, send_ahead: captured.append((ts, data, send_ahead)),
    )

    connection = SendspinConnection(client)
    await connection._handle_stream_start(  # noqa: SLF001
        StreamStartMessage(payload=StreamStartPayload(player=_stream_start_player()))
    )
    connection._handle_binary_message(  # noqa: SLF001
        pack_player_audio_header(123_456, send_ahead) + b"\x01\x02\x03\x04"
    )

    assert captured == [(123_456, b"\x01\x02\x03\x04", send_ahead)]


@pytest.mark.asyncio
async def test_truncated_audio_binary_is_dropped() -> None:
    """An audio frame shorter than the 13-byte header never reaches the listener."""
    client = make_sdk_client(
        client_name="Test Client",
        roles=[Roles.PLAYER],
        player_support=_player_support(),
    )
    captured: list[bytes] = []
    client.add_audio_chunk_listener(lambda _ts, data, _fmt, _send_ahead: captured.append(data))

    connection = SendspinConnection(client)
    await connection._handle_stream_start(  # noqa: SLF001
        StreamStartMessage(payload=StreamStartPayload(player=_stream_start_player()))
    )
    connection._handle_binary_message(  # noqa: SLF001
        pack_player_audio_header(123_456, 0)[:-1]
    )

    assert captured == []


@pytest.mark.asyncio
async def test_visualizer_binary_dropped_when_only_player_stream_active() -> None:
    """Visualizer binaries must be rejected when only the player stream is active."""
    client = make_sdk_client(
        client_name="Test Client",
        roles=[Roles.PLAYER, Roles.VISUALIZER],
        player_support=_player_support(),
        visualizer_support=_visualizer_support(),
        visualizer_state=_VISUALIZER_STATE,
    )
    captured: list[list[VisualizerFrame]] = []
    client.add_visualizer_listener(captured.append)

    connection = SendspinConnection(client)
    await connection._handle_stream_start(  # noqa: SLF001
        StreamStartMessage(payload=StreamStartPayload(player=_stream_start_player()))
    )

    # Loudness frame: type byte + 8-byte timestamp + 2-byte value.
    loudness_payload = (
        bytes([BinaryMessageType.VISUALIZATION_LOUDNESS.value])
        + struct.pack(">q", 1_000)
        + struct.pack(">H", 42)
    )
    connection._handle_binary_message(loudness_payload)  # noqa: SLF001

    assert captured == []


@pytest.mark.asyncio
async def test_artwork_binary_dispatched_when_artwork_stream_active() -> None:
    """Artwork binaries must reach listeners once artwork stream is active."""
    client = make_sdk_client(
        client_name="Test Client",
        roles=[Roles.ARTWORK],
        artwork_channels=_artwork_channels(),
    )
    captured: list[tuple[int, bytes]] = []
    client.add_artwork_listener(lambda channel, data: captured.append((channel, data)))

    connection = SendspinConnection(client)
    await connection._handle_stream_start(_artwork_stream_start())  # noqa: SLF001

    payload = b"artwork-bytes-2"
    connection._handle_binary_message(pack_artwork_announce(1, 234_567, len(payload)))  # noqa: SLF001
    connection._handle_binary_message(next(pack_artwork_parts(1, payload)))  # noqa: SLF001

    assert captured == [(1, payload)]


@pytest.mark.parametrize("available", [True, False])
@pytest.mark.asyncio
async def test_send_player_state_reports_client_level_available(
    available: bool,  # noqa: FBT001
) -> None:
    """The SDK reports availability at the client/state level, not the deprecated player field."""
    client = make_sdk_client(
        client_name="Test Client",
        roles=[Roles.PLAYER],
        player_support=_player_support(),
    )
    connection = SendspinConnection(client)

    sent: list[str] = []

    async def _capture(payload: str) -> None:
        sent.append(payload)

    mock_ws = MagicMock()
    mock_ws.closed = False
    connection._ws = mock_ws  # noqa: SLF001
    connection._connected = True  # noqa: SLF001
    connection._send_message = _capture  # noqa: SLF001

    await connection.send_player_state(available=available, volume=50, muted=False)

    assert len(sent) == 1
    msg = json.loads(sent[0])
    assert msg["payload"]["available"] is available
    assert "state" not in msg["payload"].get("player", {})


async def _controller_connection(
    controller: ControllerStatePayload | None,
) -> tuple[SendspinConnection, list[dict[str, Any]]]:
    """Return a connected connection that has received `controller` state, and its sent messages."""
    client = make_sdk_client(
        client_name="Test Client",
        roles=[Roles.CONTROLLER],
    )
    connection = SendspinConnection(client)
    sent: list[dict[str, Any]] = []

    async def _capture(payload: str) -> None:
        sent.append(json.loads(payload))

    connection._ws = MagicMock(closed=False)  # noqa: SLF001
    connection._connected = True  # noqa: SLF001
    connection._send_message = _capture  # noqa: SLF001
    if controller is not None:
        connection._handle_server_state(ServerStatePayload(controller=controller))  # noqa: SLF001
    return connection, sent


def _controller_state(
    commands: list[MediaCommand], seek_max_ms: int | None = None
) -> ControllerStatePayload:
    return ControllerStatePayload(
        supported_commands=commands,
        volume=100,
        muted=False,
        repeat=RepeatMode.OFF,
        shuffle=False,
        seek_max_ms=seek_max_ms,
    )


async def test_send_group_command_seek_forwards_position_ms() -> None:
    """send_group_command must include position_ms in the outgoing JSON for seek."""
    connection, sent = await _controller_connection(
        _controller_state([MediaCommand.SEEK], seek_max_ms=12_000)
    )

    await connection.send_group_command(MediaCommand.SEEK, position_ms=12_000)

    assert [msg["payload"]["controller"] for msg in sent] == [
        {"command": "seek", "position_ms": 12_000}
    ]


async def test_send_group_command_sends_listed_command() -> None:
    """A command listed in the latest controller state is sent."""
    connection, sent = await _controller_connection(_controller_state([MediaCommand.PLAY]))

    await connection.send_group_command(MediaCommand.PLAY)

    assert [msg["payload"]["controller"] for msg in sent] == [{"command": "play"}]


async def test_send_group_command_rejects_unlisted_command() -> None:
    """A command missing from the latest controller state is rejected without sending."""
    connection, sent = await _controller_connection(_controller_state([MediaCommand.PLAY]))
    connection._handle_server_state(  # noqa: SLF001
        ServerStatePayload(controller=_controller_state([MediaCommand.PAUSE]))
    )

    with pytest.raises(ValueError, match="'play' is not supported"):
        await connection.send_group_command(MediaCommand.PLAY)

    assert sent == []


@pytest.mark.parametrize("controller", [None, "cleared"])
async def test_send_group_command_rejects_without_controller_state(
    controller: str | None,
) -> None:
    """Commands are rejected without sending until a controller state is received."""
    connection, sent = await _controller_connection(None)
    if controller == "cleared":
        connection._handle_server_state(ServerStatePayload(controller=None))  # noqa: SLF001

    with pytest.raises(ValueError, match="No controller state"):
        await connection.send_group_command(MediaCommand.PLAY)

    assert sent == []


@pytest.mark.parametrize(
    ("position_ms", "match"),
    [
        (None, "position_ms must be provided"),
        (-1, "position_ms must be non-negative"),
        (12_001, "position_ms must be at most seek_max_ms"),
    ],
)
async def test_send_group_command_rejects_invalid_seek_position(
    position_ms: int | None, match: str
) -> None:
    """A seek without a position or outside 0 to seek_max_ms is rejected without sending."""
    connection, sent = await _controller_connection(
        _controller_state([MediaCommand.SEEK], seek_max_ms=12_000)
    )

    with pytest.raises(ValueError, match=match):
        await connection.send_group_command(MediaCommand.SEEK, position_ms=position_ms)

    assert sent == []


async def test_client_send_group_command_rejects_unlisted_command() -> None:
    """The client-level send_group_command applies the same check."""
    connection, sent = await _controller_connection(_controller_state([MediaCommand.PAUSE]))
    client = connection._client  # noqa: SLF001
    client._admitted_connection = connection  # noqa: SLF001

    with pytest.raises(ValueError, match="'play' is not supported"):
        await client.send_group_command(MediaCommand.PLAY)

    assert sent == []


def _position_connection(
    metadata: SessionUpdateMetadata | None, *, synced: bool = True
) -> tuple[SendspinConnection, ManualClock]:
    """Return a connection whose server clock runs 5 s ahead, with `metadata` received."""
    clock = ManualClock(now_us_value=1_000_000)
    client = make_sdk_client(client_name="Test Client", roles=[Roles.METADATA], clock=clock)
    connection = SendspinConnection(client)
    if synced:
        connection._time_filter.update(5_000_000, 1_000, 500_000)  # noqa: SLF001
        connection._time_filter.update(5_000_000, 1_000, 1_000_000)  # noqa: SLF001
    if metadata is not None:
        connection._handle_server_state(ServerStatePayload(metadata=metadata))  # noqa: SLF001
    client._admitted_connection = connection  # noqa: SLF001
    return connection, clock


def _progress_metadata(
    track_progress: int,
    *,
    track_duration: int = 180_000,
    playback_speed: int = 1000,
    timestamp: int = 4_000_000,
) -> SessionUpdateMetadata:
    return SessionUpdateMetadata(
        timestamp=timestamp,
        progress=Progress(
            track_progress=track_progress,
            track_duration=track_duration,
            playback_speed=playback_speed,
        ),
    )


async def test_current_track_position_advances_with_server_clock() -> None:
    """The position advances from the metadata timestamp at the playback speed."""
    connection, clock = _position_connection(_progress_metadata(30_000))
    client = connection._client  # noqa: SLF001

    assert client.current_track_position() == 32_000
    clock.advance_us(1_500_000)
    assert client.current_track_position() == 33_500


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        (_progress_metadata(30_000, playback_speed=1500), 33_000),
        (_progress_metadata(30_000, playback_speed=0), 30_000),
        (_progress_metadata(179_000), 180_000),
        (_progress_metadata(179_000, track_duration=0), 181_000),
        (_progress_metadata(1_000, timestamp=10_000_000), 0),
        (_progress_metadata(1_000, track_duration=0, timestamp=10_000_000), 0),
    ],
    ids=["speed", "paused", "clamped-to-duration", "live", "clamped-to-start", "live-to-start"],
)
async def test_current_track_position_formula(
    metadata: SessionUpdateMetadata, expected: int
) -> None:
    """The position follows the spec formula, clamped to 0 and a non-zero duration."""
    connection, _ = _position_connection(metadata)

    assert connection.current_track_position() == expected


@pytest.mark.parametrize(
    ("metadata", "synced"),
    [
        (None, True),
        (SessionUpdateMetadata(timestamp=4_000_000, title="Song"), True),
        (_progress_metadata(30_000), False),
    ],
    ids=["no-metadata", "no-progress", "unsynchronized"],
)
async def test_current_track_position_unknown(
    metadata: SessionUpdateMetadata | None, *, synced: bool
) -> None:
    """The position is None without progress or before time sync converges."""
    connection, _ = _position_connection(metadata, synced=synced)

    assert connection._client.current_track_position() is None  # noqa: SLF001


async def test_current_track_position_after_metadata_cleared() -> None:
    """Clearing the metadata clears the position."""
    connection, _ = _position_connection(_progress_metadata(30_000))
    connection._handle_server_state(ServerStatePayload(metadata=None))  # noqa: SLF001

    assert connection.current_track_position() is None


async def test_current_track_position_without_connection() -> None:
    """A client without a connection has no position."""
    client = make_sdk_client(client_name="Test Client", roles=[Roles.METADATA])

    assert client.current_track_position() is None


async def _reporting_connection(
    client: SendspinClient,
) -> tuple[SendspinConnection, list[dict[str, Any]]]:
    """Return a connected connection that has sent its player state, and the sent messages."""
    connection = SendspinConnection(client)
    sent: list[dict[str, Any]] = []

    async def _capture(payload: str) -> None:
        sent.append(json.loads(payload))

    connection._ws = MagicMock(closed=False)  # noqa: SLF001
    connection._connected = True  # noqa: SLF001
    connection._send_message = _capture  # noqa: SLF001
    await connection.send_player_state(available=True, volume=50, muted=False)
    return connection, sent


async def test_build_client_hello_omits_player_supported_commands() -> None:
    """The hello never carries player supported_commands, even when the embedder set them."""
    client = make_sdk_client(
        client_name="Test Client",
        roles=[Roles.PLAYER],
        player_support=replace(_player_support(), supported_commands=[PlayerCommand.VOLUME]),
    )
    connection = SendspinConnection(client)

    hello = (await connection._build_client_hello()).to_dict()  # noqa: SLF001

    assert "supported_commands" not in hello["payload"]["player@v1_support"]


@pytest.mark.parametrize(
    ("state_commands", "expected"),
    [(None, []), ([PlayerCommand.SET_OUTPUT_DELAY], ["set_output_delay"])],
)
async def test_send_player_state_always_carries_supported_commands(
    state_commands: list[PlayerCommand] | None, expected: list[str]
) -> None:
    """The player state always carries supported_commands, as an explicit list when empty."""
    client = make_sdk_client(
        client_name="Test Client",
        roles=[Roles.PLAYER],
        player_support=_player_support(),
        state_supported_commands=state_commands,
    )

    _, sent = await _reporting_connection(client)

    assert sent[0]["payload"]["player"]["supported_commands"] == expected


async def test_player_support_commands_fold_into_state_list() -> None:
    """Commands an embedder declared on player_support are reported in client/state."""
    client = make_sdk_client(
        client_name="Test Client",
        roles=[Roles.PLAYER],
        player_support=replace(
            _player_support(), supported_commands=[PlayerCommand.VOLUME, PlayerCommand.MUTE]
        ),
        state_supported_commands=[PlayerCommand.SET_OUTPUT_DELAY, PlayerCommand.VOLUME],
    )

    _, sent = await _reporting_connection(client)

    assert sent[0]["payload"]["player"]["supported_commands"] == [
        "volume",
        "mute",
        "set_output_delay",
    ]


_FLAC_48K = SupportedAudioFormat(
    codec=AudioCodec.FLAC, sample_rate=48_000, bit_depth=16, channels=2
)
_PCM_44K = SupportedAudioFormat(codec=AudioCodec.PCM, sample_rate=44_100, bit_depth=16, channels=2)


def _multi_format_client() -> SendspinClient:
    return make_sdk_client(
        client_name="Test Client",
        roles=[Roles.PLAYER],
        player_support=replace(
            _player_support(),
            supported_formats=[*_player_support().supported_formats, _FLAC_48K, _PCM_44K],
        ),
    )


def _activate_player(client: SendspinClient, connection: SendspinConnection) -> None:
    client._admitted_connection = connection  # noqa: SLF001
    connection._active_roles = ["player@v1"]  # noqa: SLF001


async def test_send_player_state_omits_format_without_preference() -> None:
    """Without a preference the player state carries no format."""
    _, sent = await _reporting_connection(_multi_format_client())

    assert "format" not in sent[0]["payload"]["player"]


async def test_preference_set_before_connecting_is_in_initial_state() -> None:
    """A preference set while disconnected is carried by the first player state."""
    client = _multi_format_client()
    await client.set_preferred_format(_FLAC_48K)

    _, sent = await _reporting_connection(client)

    assert sent[0]["payload"]["player"]["format"] == _FLAC_48K.to_dict()


async def test_set_preferred_format_sends_full_player_state() -> None:
    """Setting a preference sends a full player state carrying it; None clears it."""
    client = _multi_format_client()
    connection, sent = await _reporting_connection(client)
    _activate_player(client, connection)
    sent.clear()

    await client.set_preferred_format(_FLAC_48K)
    await client.set_preferred_format(None)

    assert [msg["type"] for msg in sent] == ["client/state", "client/state"]
    player = sent[0]["payload"]["player"]
    assert player["format"] == _FLAC_48K.to_dict()
    assert player["volume"] == 50
    assert player["supported_commands"] == []
    assert "output_delay_ms" in player
    assert "format" not in sent[1]["payload"]["player"]


async def test_set_preferred_format_skips_send_without_active_player_role() -> None:
    """Without an active player role the preference is stored but not sent."""
    client = _multi_format_client()
    connection, sent = await _reporting_connection(client)
    client._admitted_connection = connection  # noqa: SLF001
    sent.clear()

    await client.set_preferred_format(_FLAC_48K)

    assert sent == []
    assert client.preferred_format == _FLAC_48K


async def test_player_state_updates_keep_preferred_format() -> None:
    """Later player state updates repeat the preference instead of clearing it."""
    client = _multi_format_client()
    await client.set_preferred_format(_FLAC_48K)
    connection, sent = await _reporting_connection(client)

    await connection.send_player_state(available=True, volume=20, muted=True)

    assert len(sent) == 2
    assert all(msg["payload"]["player"]["format"] == _FLAC_48K.to_dict() for msg in sent)


async def test_set_preferred_format_rejects_unsupported_format() -> None:
    """A preference outside the client's own supported_formats raises and is not stored."""
    client = _multi_format_client()
    await client.set_preferred_format(_FLAC_48K)

    with pytest.raises(ValueError, match="supported_formats"):
        await client.set_preferred_format(replace(_FLAC_48K, bit_depth=24))

    assert client.preferred_format == _FLAC_48K


async def test_sdk_never_sends_stream_request_format() -> None:
    """Changing the preference is reported via client/state only."""
    client = _multi_format_client()
    connection, sent = await _reporting_connection(client)
    _activate_player(client, connection)

    await client.set_preferred_format(_PCM_44K)

    assert {msg["type"] for msg in sent} == {"client/state"}


async def test_server_command_not_reported_is_ignored() -> None:
    """A server/command absent from the last reported supported_commands is not delivered."""
    client = make_sdk_client(
        client_name="Test Client",
        roles=[Roles.PLAYER],
        player_support=_player_support(),
        state_supported_commands=[PlayerCommand.MUTE],
    )
    connection, _ = await _reporting_connection(client)

    received: list[ServerCommandPayload] = []
    client.add_server_command_listener(received.append)

    await connection._handle_server_command(  # noqa: SLF001
        ServerCommandPayload(
            player=PlayerCommandPayload(command=PlayerCommand.SET_OUTPUT_DELAY, output_delay_ms=250)
        )
    )
    await connection._handle_server_command(  # noqa: SLF001
        ServerCommandPayload(player=PlayerCommandPayload(command=PlayerCommand.VOLUME, volume=10))
    )
    mute = ServerCommandPayload(player=PlayerCommandPayload(command=PlayerCommand.MUTE, mute=True))
    await connection._handle_server_command(mute)  # noqa: SLF001

    assert connection.output_delay_ms == 0.0
    assert received == [mute]


async def test_server_command_set_output_delay_applies_and_notifies() -> None:
    """A server/command SET_OUTPUT_DELAY updates the offset and fires the callback."""
    client = make_sdk_client(
        client_name="Test Client",
        roles=[Roles.PLAYER],
        player_support=_player_support(),
        state_supported_commands=[PlayerCommand.SET_OUTPUT_DELAY],
    )
    connection, _ = await _reporting_connection(client)
    client._admitted_connection = connection  # noqa: SLF001

    received: list[ServerCommandPayload] = []
    client.add_server_command_listener(received.append)
    delays: list[float] = []
    client.add_output_delay_listener(delays.append)

    payload = ServerCommandPayload(
        player=PlayerCommandPayload(command=PlayerCommand.SET_OUTPUT_DELAY, output_delay_ms=250)
    )
    await connection._handle_server_command(payload)  # noqa: SLF001

    assert connection.output_delay_ms == 250.0
    assert client.output_delay_us == 250_000
    assert delays == [250.0]
    assert received == [payload]


async def test_server_command_pre_rename_delay_applies_and_notifies() -> None:
    """A pre-rename server/command set_static_delay updates the offset and fires the callback."""
    client = make_sdk_client(
        client_name="Test Client",
        roles=[Roles.PLAYER],
        player_support=_player_support(),
        state_supported_commands=[PlayerCommand.SET_STATIC_DELAY],
    )
    connection, _ = await _reporting_connection(client)
    client._admitted_connection = connection  # noqa: SLF001

    received: list[ServerCommandPayload] = []
    client.add_server_command_listener(received.append)

    payload = ServerCommandPayload.from_dict(
        {"player": {"command": "set_static_delay", "static_delay_ms": 250}}
    )
    await connection._handle_server_command(payload)  # noqa: SLF001

    assert connection.output_delay_ms == 250.0
    assert received == [payload]


async def test_output_delay_listener_fires_on_changes_only() -> None:
    """The output delay listener receives each clamped change, not no-op sets."""
    client = make_sdk_client(
        client_name="Test Client",
        roles=[Roles.PLAYER],
        player_support=_player_support(),
        output_delay_ms=100.0,
    )
    delays: list[float] = []
    remove = client.add_output_delay_listener(delays.append)

    client.set_output_delay_ms(100.0)
    client.set_output_delay_ms(120.5)
    client.set_output_delay_ms(9_000.0)
    client.set_output_delay_ms(5_000.0)
    remove()
    client.set_output_delay_ms(0.0)

    assert delays == [120.5, 5_000.0]
    assert client.output_delay_us == 0


async def test_server_command_without_player_only_notifies() -> None:
    """A server/command with no player sub-command leaves the delay unchanged but still notifies."""
    client = make_sdk_client(
        client_name="Test Client", roles=[Roles.PLAYER], player_support=_player_support()
    )
    connection = SendspinConnection(client)

    received: list[ServerCommandPayload] = []
    client.add_server_command_listener(received.append)

    payload = ServerCommandPayload()
    await connection._handle_server_command(payload)  # noqa: SLF001

    assert connection.output_delay_ms == 0.0
    assert received == [payload]


_SERVER_TIME = ServerTimePayload(client_transmitted=0, server_received=0, server_transmitted=0)


class _FakeTimeFilter:
    def __init__(self) -> None:
        self.is_synchronized = False

    def update(self, _offset: int, _delay: int, _now_us: int) -> None:
        self.is_synchronized = True


async def _state_connection(
    active_roles: list[str], **client_kwargs: Any
) -> tuple[SendspinConnection, list[dict[str, Any]]]:
    """Return a connected, unsynchronized connection and the client/state payloads it sends."""
    connection = await _connection(PskCategory.LONG_TERM, **client_kwargs)
    sent: list[dict[str, Any]] = []

    async def _capture(payload: str) -> None:
        message = json.loads(payload)
        if message["type"] == "client/state":
            sent.append(message["payload"])

    async def _idle() -> None: ...

    connection._ws = MagicMock(closed=False)  # noqa: SLF001
    connection._connected = True  # noqa: SLF001
    connection._send_message = _capture  # type: ignore[method-assign]  # noqa: SLF001
    connection._time_filter = _FakeTimeFilter()  # type: ignore[assignment]  # noqa: SLF001
    connection._active_roles = active_roles  # noqa: SLF001
    connection._reader_loop = _idle  # type: ignore[method-assign]  # noqa: SLF001
    connection._time_sync_loop = _idle  # type: ignore[method-assign]  # noqa: SLF001
    return connection, sent


async def test_player_initial_state_unavailable_until_clock_synchronizes() -> None:
    """A player's initial state is unavailable; convergence sends it again as available."""
    connection, sent = await _state_connection([Roles.PLAYER.value])

    await connection.start()
    await connection._handle_server_time(_SERVER_TIME)  # noqa: SLF001

    assert [(state["available"], "player" in state) for state in sent] == [
        (False, True),
        (True, True),
    ]


async def test_player_reported_unavailable_stays_unavailable_after_sync() -> None:
    """Clock convergence keeps the availability the application reported."""
    connection, sent = await _state_connection([Roles.PLAYER.value])

    await connection.send_player_state(available=False, volume=50, muted=False)
    await connection._handle_server_time(_SERVER_TIME)  # noqa: SLF001

    assert [(state["available"], "player" in state) for state in sent] == [
        (False, True),
        (False, True),
    ]


async def test_unavailable_player_discards_audio_and_keeps_stream_control() -> None:
    """While unavailable, audio is discarded without closing; stream control still applies."""
    connection, _ = await _state_connection([Roles.PLAYER.value])
    connection.disconnect = AsyncMock()  # type: ignore[method-assign]
    client = connection._client  # noqa: SLF001
    chunks: list[int] = []
    control: list[str] = []
    client.add_audio_chunk_listener(lambda ts, _data, _fmt, _send_ahead: chunks.append(ts))
    client.add_stream_start_listener(lambda _message: control.append("start"))
    client.add_stream_clear_listener(lambda _roles: control.append("clear"))
    client.add_stream_end_listener(lambda _roles: control.append("end"))
    start = StreamStartMessage(payload=StreamStartPayload(player=_stream_start_player()))

    await connection.send_player_state(available=False, volume=50, muted=False)
    await connection._handle_stream_start(start)  # noqa: SLF001
    connection._handle_binary_message(pack_player_audio_header(1, 0) + b"\x00")  # noqa: SLF001
    connection._handle_stream_clear(  # noqa: SLF001
        StreamClearMessage(payload=StreamClearPayload(roles=["player"]))
    )
    connection._handle_stream_end(  # noqa: SLF001
        StreamEndMessage(payload=StreamEndPayload(roles=["player"]))
    )
    await connection._handle_stream_start(start)  # noqa: SLF001
    connection._handle_binary_message(pack_player_audio_header(2, 0) + b"\x00")  # noqa: SLF001
    assert chunks == []
    assert control == ["start", "clear", "end", "start"]

    await connection.send_player_state(available=True, volume=50, muted=False)
    connection._handle_binary_message(pack_player_audio_header(3, 0) + b"\x00")  # noqa: SLF001

    assert chunks == [3]
    await asyncio.sleep(0)
    connection.disconnect.assert_not_awaited()  # type: ignore[attr-defined]


async def _report_player_state(connection: SendspinConnection) -> None:
    await connection.send_player_state(available=True, volume=50, muted=False)


async def _report_available(connection: SendspinConnection) -> None:
    await connection.send_available(available=True)


@pytest.mark.parametrize("report", [_report_player_state, _report_available])
async def test_player_available_withheld_until_clock_synchronizes(
    report: Callable[[SendspinConnection], Awaitable[None]],
) -> None:
    """An active player reports available only once its clock has converged."""
    connection, sent = await _state_connection([Roles.PLAYER.value])

    await report(connection)
    assert sent[-1]["available"] is False

    await connection._handle_server_time(_SERVER_TIME)  # noqa: SLF001
    await report(connection)
    assert sent[-1]["available"] is True


async def test_player_and_source_states_carry_source_object() -> None:
    """With player and source active, every client/state carries the source object."""
    connection, sent = await _state_connection(
        [Roles.PLAYER.value, Roles.SOURCE.value],
        roles=[Roles.PLAYER, Roles.SOURCE],
        source_support=ClientHelloSourceSupport(),
    )

    await connection.start()
    await connection._handle_server_time(_SERVER_TIME)  # noqa: SLF001

    assert [(state["available"], "player" in state, state.get("source")) for state in sent] == [
        (False, True, {}),
        (True, True, {}),
    ]


@pytest.mark.parametrize(
    "features",
    [None, ClientHelloSourceFeatures(), ClientHelloSourceFeatures(line_sense=False)],
)
async def test_source_signal_requires_line_sense(
    features: ClientHelloSourceFeatures | None,
) -> None:
    """A source that did not advertise line_sense cannot report a signal."""
    connection, sent = await _state_connection(
        [Roles.SOURCE.value],
        roles=[Roles.SOURCE],
        source_support=ClientHelloSourceSupport(features=features),
    )
    await connection._handle_server_time(_SERVER_TIME)  # noqa: SLF001

    with pytest.raises(RuntimeError, match="line_sense"):
        await connection.send_source_signal(SignalState.PRESENT)
    await connection.send_available(available=True)

    assert [state["source"] for state in sent] == [{}, {}]


async def test_source_signal_is_reported_with_line_sense() -> None:
    """A source that advertised line_sense reports its signal in client/state."""
    connection, sent = await _state_connection(
        [Roles.SOURCE.value],
        roles=[Roles.SOURCE],
        source_support=ClientHelloSourceSupport(
            features=ClientHelloSourceFeatures(line_sense=True)
        ),
    )
    await connection._handle_server_time(_SERVER_TIME)  # noqa: SLF001

    await connection.send_source_signal(SignalState.PRESENT)

    assert sent[-1]["source"] == {"signal": "present"}


@pytest.mark.parametrize("role", [Roles.CONTROLLER, Roles.METADATA])
async def test_stateless_roles_send_initial_state(role: Roles) -> None:
    """A client with only stateless roles active still sends its initial client/state."""
    connection, sent = await _state_connection([role.value])

    await connection.start()

    assert sent == [{"available": True}]


async def test_initial_state_sent_once_when_roles_first_activate() -> None:
    """Only the first activation with roles sends client/state for stateless roles."""
    connection, sent = await _state_connection([])
    counts = []

    for roles in (
        [],
        [Roles.CONTROLLER.value],
        [Roles.CONTROLLER.value, Roles.METADATA.value],
        [],
        [Roles.CONTROLLER.value],
    ):
        await connection._handle_server_activate(  # noqa: SLF001
            ServerActivatePayload(activities=[], active_roles=roles)
        )
        counts.append(len(sent))

    assert counts == [0, 1, 1, 1, 1]
    assert sent == [{"available": True}]


_ARTWORK_STATE = {"channels": [{"source": "album", "format": "jpeg", "width": 256, "height": 256}]}


async def test_artwork_initial_state_carries_artwork_object() -> None:
    """An active artwork role declares its channels in the initial client/state."""
    connection, sent = await _state_connection([Roles.ARTWORK.value])

    await connection.start()

    assert sent == [{"available": True, "artwork": _ARTWORK_STATE}]


async def test_player_and_artwork_initial_state_is_one_message() -> None:
    """The initial client/state carries the player and artwork objects together."""
    connection, sent = await _state_connection([Roles.PLAYER.value, Roles.ARTWORK.value])

    await connection.start()

    assert len(sent) == 1
    assert "player" in sent[0]
    assert sent[0]["artwork"] == _ARTWORK_STATE


async def test_artwork_activation_sends_artwork_object() -> None:
    """Activating the artwork role after the initial state sends the artwork object."""
    connection, sent = await _state_connection([Roles.PLAYER.value])
    await connection.start()

    await connection._handle_server_activate(  # noqa: SLF001
        ServerActivatePayload(activities=[], active_roles=[Roles.PLAYER.value, Roles.ARTWORK.value])
    )

    assert "artwork" not in sent[0]
    assert sent[-1]["artwork"] == _ARTWORK_STATE


async def test_set_artwork_channels_reports_new_channels() -> None:
    """set_artwork_channels sends the full artwork object in a client/state."""
    connection, sent = await _state_connection([Roles.ARTWORK.value])
    client = connection._client  # noqa: SLF001
    client._admitted_connection = connection  # noqa: SLF001

    await client.set_artwork_channels(
        [
            ArtworkChannel(source=ArtworkSource.NONE),
            ArtworkChannel(
                source=ArtworkSource.ARTIST, format=PictureFormat.PNG, width=64, height=32
            ),
        ]
    )

    assert sent == [
        {
            "available": True,
            "artwork": {
                "channels": [
                    {"source": "none"},
                    {"source": "artist", "format": "png", "width": 64, "height": 32},
                ]
            },
        }
    ]


async def test_set_artwork_channels_skipped_while_role_inactive() -> None:
    """set_artwork_channels stores the channels but sends nothing while artwork is inactive."""
    connection, sent = await _state_connection([Roles.PLAYER.value])
    client = connection._client  # noqa: SLF001
    client._admitted_connection = connection  # noqa: SLF001
    channels = [ArtworkChannel(source=ArtworkSource.NONE)]

    await client.set_artwork_channels(channels)

    assert sent == []
    assert client.artwork_state is not None
    assert client.artwork_state.channels == channels


@pytest.mark.parametrize("count", [0, 5])
async def test_set_artwork_channels_rejects_invalid_length(count: int) -> None:
    """set_artwork_channels accepts only 1-4 channels."""
    client = make_sdk_client(
        client_name="Test Client",
        roles=[Roles.ARTWORK],
        artwork_channels=_artwork_channels(),
    )

    with pytest.raises(ValueError, match="1-4"):
        await client.set_artwork_channels(_artwork_channels() * count)


async def test_artwork_role_requires_artwork_channels() -> None:
    """The ARTWORK role cannot be declared without its channels."""
    with pytest.raises(ValueError, match="artwork_channels"):
        make_sdk_client(client_name="Test Client", roles=[Roles.ARTWORK])


async def test_build_client_hello_omits_artwork_support() -> None:
    """The hello lists artwork@v1 without a support object."""
    client = make_sdk_client(
        client_name="Test Client",
        roles=[Roles.ARTWORK],
        artwork_channels=_artwork_channels(),
    )
    connection = SendspinConnection(client)

    hello = (await connection._build_client_hello()).to_dict()  # noqa: SLF001

    assert hello["payload"]["supported_roles"] == [Roles.ARTWORK.value]
    assert "artwork@v1_support" not in hello["payload"]


# ---------------------------------------------------------------------------
# Visualizer stream configuration in client/state
# ---------------------------------------------------------------------------

_VISUALIZER_CLIENT = {
    "roles": [Roles.PLAYER, Roles.VISUALIZER],
    "visualizer_support": _visualizer_support(),
    "visualizer_state": _VISUALIZER_STATE,
}


async def test_build_client_hello_visualizer_support_carries_only_buffer_capacity() -> None:
    """The visualizer hello support object carries only buffer_capacity."""
    connection = SendspinConnection(
        make_sdk_client(client_name="c", player_support=_player_support(), **_VISUALIZER_CLIENT)
    )

    hello = (await connection._build_client_hello()).to_dict()  # noqa: SLF001

    assert hello["payload"]["visualizer@v1_support"] == {"buffer_capacity": 4096}


def test_visualizer_client_requires_state_without_hello_stream_config() -> None:
    """The constructor requires visualizer_state and rejects stream config on the support."""
    with pytest.raises(ValueError, match="visualizer_state is required"):
        make_sdk_client(
            client_name="c", roles=[Roles.VISUALIZER], visualizer_support=_visualizer_support()
        )
    with pytest.raises(ValueError, match="belong in visualizer_state"):
        make_sdk_client(
            client_name="c",
            roles=[Roles.VISUALIZER],
            visualizer_support=ClientHelloVisualizerSupport(buffer_capacity=4096, rate_max=30),
            visualizer_state=_VISUALIZER_STATE,
        )


async def test_visualizer_only_initial_state_carries_visualizer_object() -> None:
    """A visualizer-only activation sends one initial client/state with the visualizer object."""
    connection, sent = await _state_connection([Roles.VISUALIZER.value], **_VISUALIZER_CLIENT)

    await connection.start()

    assert sent == [{"available": True, "visualizer": _VISUALIZER_STATE.to_dict()}]


async def test_player_initial_state_carries_visualizer_object() -> None:
    """With player and visualizer active, the player state also carries the visualizer object."""
    connection, sent = await _state_connection(
        [Roles.PLAYER.value, Roles.VISUALIZER.value], **_VISUALIZER_CLIENT
    )

    await connection.start()

    assert len(sent) == 1
    assert "player" in sent[0]
    assert sent[0]["visualizer"] == _VISUALIZER_STATE.to_dict()


async def test_visualizer_object_omitted_while_role_inactive() -> None:
    """No visualizer object is sent while the server has not activated the role."""
    connection, sent = await _state_connection([Roles.PLAYER.value], **_VISUALIZER_CLIENT)

    await connection.start()

    assert "visualizer" not in sent[0]


async def test_visualizer_activation_resends_state_with_visualizer_object() -> None:
    """Activating the visualizer role sends a client/state carrying its object."""
    connection, sent = await _state_connection([Roles.PLAYER.value], **_VISUALIZER_CLIENT)
    await connection.start()
    sent.clear()

    await connection._handle_server_activate(  # noqa: SLF001
        ServerActivatePayload(
            activities=[], active_roles=[Roles.PLAYER.value, Roles.VISUALIZER.value]
        )
    )

    assert len(sent) == 1
    assert sent[0]["visualizer"] == _VISUALIZER_STATE.to_dict()


async def test_set_visualizer_state_sends_client_state() -> None:
    """Changing the requested visualizer configuration reports it via client/state."""
    connection, sent = await _state_connection([Roles.VISUALIZER.value], **_VISUALIZER_CLIENT)
    client = connection._client  # noqa: SLF001
    client._admitted_connection = connection  # noqa: SLF001
    spectrum = ClientHelloVisualizerSpectrum(n_disp_bins=16, scale="mel", f_min=20, f_max=16_000)
    state = VisualizerStatePayload(types=["spectrum", "beat"], rate_max=15, spectrum=spectrum)

    await client.set_visualizer_state(state)

    assert client.visualizer_state == state
    assert sent == [{"available": True, "visualizer": state.to_dict()}]


async def test_set_visualizer_state_skips_send_without_active_role() -> None:
    """Without an active visualizer role the configuration is stored but not sent."""
    connection, sent = await _state_connection([Roles.PLAYER.value], **_VISUALIZER_CLIENT)
    client = connection._client  # noqa: SLF001
    client._admitted_connection = connection  # noqa: SLF001
    state = VisualizerStatePayload(types=[], rate_max=10)

    await client.set_visualizer_state(state)

    assert sent == []
    assert client.visualizer_state == state


async def test_set_visualizer_state_requires_visualizer_role() -> None:
    """A client without the visualizer role cannot set a visualizer configuration."""
    client = make_sdk_client(
        client_name="c", roles=[Roles.PLAYER], player_support=_player_support()
    )

    with pytest.raises(ValueError, match="VISUALIZER role"):
        await client.set_visualizer_state(_VISUALIZER_STATE)


async def test_initial_state_carries_artwork_and_visualizer_objects() -> None:
    """With artwork and visualizer active, one initial client/state carries both objects."""
    connection, sent = await _state_connection(
        [Roles.PLAYER.value, Roles.ARTWORK.value, Roles.VISUALIZER.value],
        roles=[Roles.PLAYER, Roles.ARTWORK, Roles.VISUALIZER],
        visualizer_support=_visualizer_support(),
        visualizer_state=_VISUALIZER_STATE,
    )

    await connection.start()

    assert len(sent) == 1
    assert "player" in sent[0]
    assert sent[0]["artwork"] == _ARTWORK_STATE
    assert sent[0]["visualizer"] == _VISUALIZER_STATE.to_dict()
