"""Tests for public protocol callback hooks on the Sendspin client."""

from __future__ import annotations

import json
import struct
from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from aiosendspin.client.connection import SendspinConnection
from aiosendspin.client.models import AudioFormat
from aiosendspin.models import pack_binary_header_raw
from aiosendspin.models.artwork import (
    ArtworkChannel,
    ClientHelloArtworkSupport,
    StreamArtworkChannelConfig,
    StreamStartArtwork,
)
from aiosendspin.models.core import (
    ServerActivatePayload,
    ServerCommandPayload,
    ServerHelloPayload,
    StreamStartMessage,
    StreamStartPayload,
)
from aiosendspin.models.player import (
    ClientHelloPlayerSupport,
    PlayerCommandPayload,
    StreamStartPlayer,
    SupportedAudioFormat,
)
from aiosendspin.models.types import (
    Activity,
    ArtworkSource,
    AudioCodec,
    BinaryMessageType,
    GoodbyeReason,
    MediaCommand,
    PictureFormat,
    PlayerCommand,
    Roles,
)
from aiosendspin.models.visualizer import ClientHelloVisualizerSupport, VisualizerFrame
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
        supported_commands=[],
    )


@pytest.mark.asyncio
async def test_server_hello_populates_server_info(monkeypatch: pytest.MonkeyPatch) -> None:
    """Receiving server/hello records the server's name."""
    client = make_sdk_client(
        client_name="Test Client",
        roles=[Roles.PLAYER],
        player_support=_player_support(),
    )

    connection = SendspinConnection(client)
    client._admitted_connection = connection  # noqa: SLF001
    # server_id comes from the Noise handshake, not the hello payload.
    connection._server_id = "server-1"  # noqa: SLF001

    hello = ServerHelloPayload(name="Test Server")

    async def receive_hello() -> ServerHelloPayload:
        return hello

    async def send_client_hello() -> None: ...

    async def receive_activate() -> ServerActivatePayload:
        return ServerActivatePayload(activities=[])

    monkeypatch.setattr(connection, "_receive_server_hello", receive_hello)
    monkeypatch.setattr(connection, "_send_client_hello", send_client_hello)
    monkeypatch.setattr(connection, "_receive_server_activate", receive_activate)

    await connection._exchange_hellos()  # noqa: SLF001

    assert client.server_info is not None
    assert client.server_info.server_id == "server-1"


async def _connection(
    category: PskCategory, *, unpaired_access: bool = False
) -> SendspinConnection:
    store = InMemoryClientPairingStore()
    await store.store_pairing_config(
        replace(await store.get_pairing_config(), unpaired_access_enabled=unpaired_access)
    )
    client = make_sdk_client(
        client_name="Test Client",
        roles=[Roles.PLAYER],
        player_support=_player_support(),
        pairing_store=store,
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


@pytest.mark.asyncio
async def test_start_skips_reader_when_pairing_closes_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If pairing rejects the post-pairing activation and closes, start() starts no reader."""
    connection = await _connection(PskCategory.SENTINEL)
    connection._activities = [Activity.PAIRING]  # noqa: SLF001 — drive the is_pairing branch

    async def _fake_pair() -> None:
        # Mimic _pair rejecting the post-pairing server/activate and disconnecting.
        connection._connected = False  # noqa: SLF001

    monkeypatch.setattr(connection, "_pair", _fake_pair)
    await connection.start()
    assert connection._reader_task is None  # noqa: SLF001


@pytest.mark.parametrize(
    ("category", "activities", "roles", "unpaired", "expected"),
    [
        # Sendspin PSK: ['pairing'] or any subset of {playback, management}.
        (PskCategory.LONG_TERM, [Activity.MANAGEMENT], [], False, None),
        (PskCategory.LONG_TERM, [Activity.PLAYBACK, Activity.MANAGEMENT], [], False, None),
        # Roles allowed without 'playback' in activities: the set is playback-capable.
        (PskCategory.LONG_TERM, [Activity.MANAGEMENT], [Roles.PLAYER.value], False, None),
        # 'pairing' is exclusive.
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
        # Pairing PSK: only ['pairing'].
        (PskCategory.PAIRING, [Activity.PAIRING], [], False, None),
        (PskCategory.PAIRING, [], [], False, GoodbyeReason.UNAUTHORIZED),
        (PskCategory.PAIRING, [Activity.PLAYBACK], [], False, GoodbyeReason.UNAUTHORIZED),
        # A pairing connection is never playback-capable, so it may not carry roles.
        (
            PskCategory.PAIRING,
            [Activity.PAIRING],
            [Roles.PLAYER.value],
            False,
            GoodbyeReason.UNAUTHORIZED,
        ),
        # Sentinel: 'pairing' combined with playback is malformed, not a pair-first case.
        (
            PskCategory.SENTINEL,
            [Activity.PAIRING, Activity.PLAYBACK],
            [],
            False,
            GoodbyeReason.UNAUTHORIZED,
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
async def test_artwork_listener_receives_binary_frames_after_artwork_stream_start() -> None:
    """Client should expose artwork binary frames without private overrides."""
    client = make_sdk_client(
        client_name="Test Client",
        roles=[Roles.ARTWORK],
        artwork_support=ClientHelloArtworkSupport(
            channels=[
                ArtworkChannel(
                    source=ArtworkSource.ALBUM,
                    format=PictureFormat.JPEG,
                    width=256,
                    height=256,
                )
            ]
        ),
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
    connection._handle_binary_message(  # noqa: SLF001
        pack_binary_header_raw(BinaryMessageType.ARTWORK_CHANNEL_0.value, 123_456) + payload
    )

    assert captured == [(0, payload)]


def _artwork_support() -> ClientHelloArtworkSupport:
    return ClientHelloArtworkSupport(
        channels=[
            ArtworkChannel(
                source=ArtworkSource.ALBUM,
                format=PictureFormat.JPEG,
                width=256,
                height=256,
            )
        ]
    )


def _visualizer_support() -> ClientHelloVisualizerSupport:
    return ClientHelloVisualizerSupport(
        buffer_capacity=4096,
        rate_max=30,
        types=["loudness"],
    )


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
async def test_artwork_binary_dropped_when_only_player_stream_active() -> None:
    """Artwork binaries must be rejected when only the player stream is active."""
    client = make_sdk_client(
        client_name="Test Client",
        roles=[Roles.PLAYER, Roles.ARTWORK],
        player_support=_player_support(),
        artwork_support=_artwork_support(),
    )
    captured: list[tuple[int, bytes]] = []
    client.add_artwork_listener(lambda channel, data: captured.append((channel, data)))

    connection = SendspinConnection(client)
    await connection._handle_stream_start(  # noqa: SLF001
        StreamStartMessage(payload=StreamStartPayload(player=_stream_start_player()))
    )

    connection._handle_binary_message(  # noqa: SLF001
        pack_binary_header_raw(BinaryMessageType.ARTWORK_CHANNEL_0.value, 123_456) + b"art"
    )

    assert captured == []


@pytest.mark.asyncio
async def test_audio_binary_dropped_when_only_artwork_stream_active() -> None:
    """Audio binaries must be rejected when only the artwork stream is active."""
    client = make_sdk_client(
        client_name="Test Client",
        roles=[Roles.PLAYER, Roles.ARTWORK],
        player_support=_player_support(),
        artwork_support=_artwork_support(),
    )
    captured: list[tuple[int, bytes, AudioFormat]] = []
    client.add_audio_chunk_listener(
        lambda ts, data, fmt: captured.append((ts, data, fmt)),
    )

    connection = SendspinConnection(client)
    await connection._handle_stream_start(_artwork_stream_start())  # noqa: SLF001

    connection._handle_binary_message(  # noqa: SLF001
        pack_binary_header_raw(BinaryMessageType.AUDIO_CHUNK.value, 123_456) + b"\x00\x00\x00\x00"
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
        artwork_support=_artwork_support(),
    )
    captured: list[tuple[int, bytes]] = []
    client.add_artwork_listener(lambda channel, data: captured.append((channel, data)))

    connection = SendspinConnection(client)
    await connection._handle_stream_start(_artwork_stream_start())  # noqa: SLF001

    payload = b"artwork-bytes-2"
    connection._handle_binary_message(  # noqa: SLF001
        pack_binary_header_raw(BinaryMessageType.ARTWORK_CHANNEL_1.value, 234_567) + payload
    )

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


@pytest.mark.asyncio
async def test_send_group_command_seek_forwards_position_ms() -> None:
    """send_group_command must include position_ms in the outgoing JSON for seek."""
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

    await connection.send_group_command(MediaCommand.SEEK, position_ms=12_000)

    assert len(sent) == 1
    msg = json.loads(sent[0])
    assert msg["payload"]["controller"]["command"] == "seek"
    assert msg["payload"]["controller"]["position_ms"] == 12_000


async def test_server_command_set_static_delay_applies_and_notifies() -> None:
    """A server/command SET_STATIC_DELAY updates the offset and fires the callback."""
    client = make_sdk_client(
        client_name="Test Client", roles=[Roles.PLAYER], player_support=_player_support()
    )
    connection = SendspinConnection(client)

    received: list[ServerCommandPayload] = []
    client.add_server_command_listener(received.append)

    payload = ServerCommandPayload(
        player=PlayerCommandPayload(command=PlayerCommand.SET_STATIC_DELAY, static_delay_ms=250)
    )
    connection._handle_server_command(payload)  # noqa: SLF001

    assert connection.static_delay_ms == 250.0
    assert received == [payload]


async def test_server_command_without_player_only_notifies() -> None:
    """A server/command with no player sub-command leaves the delay unchanged but still notifies."""
    client = make_sdk_client(
        client_name="Test Client", roles=[Roles.PLAYER], player_support=_player_support()
    )
    connection = SendspinConnection(client)

    received: list[ServerCommandPayload] = []
    client.add_server_command_listener(received.append)

    payload = ServerCommandPayload()
    connection._handle_server_command(payload)  # noqa: SLF001

    assert connection.static_delay_ms == 0.0
    assert received == [payload]
