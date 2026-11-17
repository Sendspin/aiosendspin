"""The client ends the streams of roles a server/activate removes."""

from __future__ import annotations

import struct

import pytest

from aiosendspin.client.connection import SendspinConnection
from aiosendspin.models.artwork import (
    ArtworkChannel,
    StreamArtworkChannelConfig,
    StreamStartArtwork,
    pack_artwork_announce,
    pack_artwork_parts,
)
from aiosendspin.models.core import (
    ServerActivatePayload,
    StreamEndMessage,
    StreamEndPayload,
    StreamStartMessage,
    StreamStartPayload,
)
from aiosendspin.models.player import (
    ClientHelloPlayerSupport,
    StreamStartPlayer,
    SupportedAudioFormat,
    pack_player_audio_header,
)
from aiosendspin.models.types import (
    Activity,
    ArtworkSource,
    AudioCodec,
    BinaryMessageType,
    PictureFormat,
    Roles,
)
from aiosendspin.models.visualizer import (
    ClientHelloVisualizerSupport,
    StreamStartVisualizer,
    VisualizerFrame,
    VisualizerStatePayload,
)
from aiosendspin.noise.trust_store import PskCategory, ResolvedPsk
from tests.conftest import make_sdk_client

_STREAM_ROLES = [Roles.PLAYER.value, Roles.ARTWORK.value, Roles.VISUALIZER.value]
_ARTWORK = b"art"


class _Recorder:
    def __init__(self) -> None:
        self.ended: list[list[str] | None] = []
        self.artwork: list[tuple[int, bytes]] = []
        self.audio: list[bytes] = []
        self.started: list[StreamStartMessage] = []
        self.visualizer: list[list[VisualizerFrame]] = []


async def _streaming_connection(
    active_roles: list[str] | None = None,
    activities: list[Activity] | None = None,
) -> tuple[SendspinConnection, _Recorder]:
    """Return a long-term connection that received a player, artwork and visualizer stream/start."""
    client = make_sdk_client(
        client_name="Test Client",
        roles=[Roles.PLAYER, Roles.ARTWORK, Roles.VISUALIZER],
        player_support=ClientHelloPlayerSupport(
            supported_formats=[
                SupportedAudioFormat(
                    codec=AudioCodec.PCM, sample_rate=48_000, bit_depth=16, channels=2
                )
            ],
            buffer_capacity=100_000,
        ),
        artwork_channels=[
            ArtworkChannel(
                source=ArtworkSource.ALBUM, format=PictureFormat.JPEG, width=256, height=256
            )
        ],
        visualizer_support=ClientHelloVisualizerSupport(buffer_capacity=4096),
        visualizer_state=VisualizerStatePayload(types=["loudness"], rate_max=30),
    )
    recorder = _Recorder()
    client.add_stream_end_listener(recorder.ended.append)
    client.add_artwork_listener(lambda channel, data: recorder.artwork.append((channel, data)))
    client.add_audio_chunk_listener(lambda _ts, data, _fmt, _ahead: recorder.audio.append(data))
    client.add_stream_start_listener(recorder.started.append)
    client.add_visualizer_listener(recorder.visualizer.append)

    conn = SendspinConnection(client)
    conn._noise_psk = ResolvedPsk("id", b"\x00" * 32, PskCategory.LONG_TERM)  # noqa: SLF001
    conn._active_roles = list(_STREAM_ROLES if active_roles is None else active_roles)  # noqa: SLF001
    conn._activities = [Activity.PLAYBACK] if activities is None else activities  # noqa: SLF001
    await conn._handle_stream_start(  # noqa: SLF001
        StreamStartMessage(
            payload=StreamStartPayload(
                player=StreamStartPlayer(
                    codec=AudioCodec.PCM, sample_rate=48_000, channels=2, bit_depth=16
                ),
                artwork=StreamStartArtwork(
                    channels=[
                        StreamArtworkChannelConfig(
                            source=ArtworkSource.ALBUM,
                            format=PictureFormat.JPEG,
                            width=256,
                            height=256,
                        )
                    ]
                ),
                visualizer=StreamStartVisualizer(types=("loudness",), rate_max=30),
            )
        )
    )
    conn._handle_binary_message(pack_artwork_announce(0, 1, len(_ARTWORK)))  # noqa: SLF001
    conn._handle_binary_message(next(pack_artwork_parts(0, _ARTWORK)))  # noqa: SLF001
    assert recorder.artwork == [(0, _ARTWORK)]
    recorder.artwork.clear()
    return conn, recorder


def _send_audio(conn: SendspinConnection) -> None:
    conn._handle_binary_message(pack_player_audio_header(1, 0) + b"\x00\x00\x00\x00")  # noqa: SLF001


@pytest.mark.parametrize(
    ("category", "payload"),
    [
        (
            PskCategory.LONG_TERM,
            ServerActivatePayload(activities=[Activity.PLAYBACK], active_roles=[]),
        ),
        (PskCategory.SENTINEL, ServerActivatePayload(activities=[Activity.PAIRING])),
        (
            PskCategory.LONG_TERM,
            ServerActivatePayload(
                activities=[Activity.PLAYBACK],
                active_roles=["player@v2", "artwork@v2", "visualizer@v2"],
            ),
        ),
    ],
    ids=["explicit", "not-playback-capable", "version-replacement"],
)
async def test_activation_ends_removed_role_streams(
    category: PskCategory, payload: ServerActivatePayload
) -> None:
    """A removed stream role ends its stream, blanks shown artwork and signals stream end once."""
    conn, recorder = await _streaming_connection()
    conn._noise_psk = ResolvedPsk("id", b"\x00" * 32, category)  # noqa: SLF001

    assert await conn._apply_activation(payload) is None  # noqa: SLF001

    assert recorder.ended == [["artwork", "player", "visualizer"]]
    assert recorder.artwork == [(0, b"")]
    assert not conn._stream_active  # noqa: SLF001
    assert conn._current_audio_format is None  # noqa: SLF001
    assert not conn._artwork_stream_active  # noqa: SLF001
    assert not conn._visualizer_stream_active  # noqa: SLF001
    assert conn._current_visualizer_config is None  # noqa: SLF001
    _send_audio(conn)
    assert recorder.audio == []


async def test_activation_ends_removed_role_after_earlier_stream_end() -> None:
    """Removal still signals stream end when a stream/end already ended the stream."""
    conn, recorder = await _streaming_connection()
    conn._handle_stream_end(  # noqa: SLF001
        StreamEndMessage(payload=StreamEndPayload(roles=["player"]))
    )

    await conn._apply_activation(  # noqa: SLF001
        ServerActivatePayload(
            activities=[Activity.PLAYBACK],
            active_roles=[Roles.ARTWORK.value, Roles.VISUALIZER.value],
        )
    )

    assert recorder.ended == [["player"], ["player"]]
    assert conn._artwork_stream_active  # noqa: SLF001
    assert conn._visualizer_stream_active  # noqa: SLF001


async def test_activation_keeps_streams_of_retained_roles() -> None:
    """Roles kept at the same version keep their streams and signal nothing."""
    conn, recorder = await _streaming_connection([*_STREAM_ROLES, Roles.METADATA.value])

    await conn._apply_activation(  # noqa: SLF001
        ServerActivatePayload(activities=[Activity.PLAYBACK], active_roles=list(_STREAM_ROLES))
    )

    assert recorder.ended == []
    assert recorder.artwork == []
    _send_audio(conn)
    assert recorder.audio == [b"\x00\x00\x00\x00"]


async def test_activation_ends_removed_application_role() -> None:
    """A removed application-specific role signals stream end for its family only."""
    conn, recorder = await _streaming_connection([*_STREAM_ROLES, "_acme@v1"])

    await conn._apply_activation(  # noqa: SLF001
        ServerActivatePayload(activities=[Activity.PLAYBACK], active_roles=list(_STREAM_ROLES))
    )

    assert recorder.ended == [["_acme"]]
    assert conn._stream_active  # noqa: SLF001
    assert conn._artwork_stream_active  # noqa: SLF001


def _send_loudness(conn: SendspinConnection) -> None:
    conn._handle_binary_message(  # noqa: SLF001
        bytes([BinaryMessageType.VISUALIZATION_LOUDNESS.value])
        + struct.pack(">q", 1)
        + struct.pack(">H", 42)
    )


async def test_stream_start_without_playback_applies_only_artwork() -> None:
    """Without 'playback', stream/start starts artwork and ignores the player and visualizer."""
    conn, recorder = await _streaming_connection(activities=[])

    [started] = recorder.started
    assert started.payload.player is None
    assert started.payload.visualizer is None
    assert started.payload.artwork is not None
    assert not conn._stream_active  # noqa: SLF001
    assert conn._current_audio_format is None  # noqa: SLF001
    assert not conn._visualizer_stream_active  # noqa: SLF001


async def test_dropping_playback_discards_audio_and_visualization() -> None:
    """After 'playback' is dropped, player audio and visualization data are discarded."""
    conn, recorder = await _streaming_connection()
    _send_audio(conn)
    _send_loudness(conn)
    assert recorder.audio != []
    assert recorder.visualizer != []
    recorder.audio.clear()
    recorder.visualizer.clear()

    await conn._apply_activation(ServerActivatePayload(activities=[]))  # noqa: SLF001
    _send_audio(conn)
    _send_loudness(conn)

    assert recorder.audio == []
    assert recorder.visualizer == []
