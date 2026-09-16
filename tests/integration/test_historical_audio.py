"""Integration tests for historical audio injection."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from uuid import UUID

import pytest

from aiosendspin.models import unpack_binary_header
from aiosendspin.models.core import StreamStartMessage
from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.models.types import AudioCodec, PlayerCommand, Roles
from aiosendspin.server.audio import AudioFormat
from aiosendspin.server.audio_transformers import TransformerPool
from aiosendspin.server.channels import MAIN_CHANNEL
from aiosendspin.server.client import SendspinClient
from aiosendspin.server.clock import ManualClock
from aiosendspin.server.push_stream import PushStream
from aiosendspin.server.roles import AudioRequirements
from aiosendspin.server.roles.player.audio_transformers import PcmPassthrough


@dataclass(slots=True)
class _DummyServer:
    loop: asyncio.AbstractEventLoop
    clock: ManualClock
    id: str = "srv"
    name: str = "server"

    def _signal_client_connected(self, client_id: str) -> None:
        pass

    def _signal_client_disconnected(self, client_id: str, goodbye_reason: object = None) -> None:
        pass


class _DummyGroup:
    def __init__(self, clients: list[SendspinClient]) -> None:
        self.clients = clients
        self.transformer_pool = TransformerPool()

    group_name = "dummy group"

    def _publish_if_name_changed(self, previous: str) -> None:  # noqa: ARG002
        return

    def on_client_connected(self, client: SendspinClient) -> None:  # noqa: ARG002
        return

    def _register_client_events(self, client: SendspinClient) -> None:  # noqa: ARG002
        return

    def group_role(self, family: str) -> None:  # noqa: ARG002
        return None

    def get_channel_for_player(self, player_id: str) -> UUID:  # noqa: ARG002
        return MAIN_CHANNEL


class _CaptureConnection:
    def __init__(self) -> None:
        self.sent_json: list[object] = []
        self.sent_binary: list[bytes] = []
        self.buffer_tracker = None

    async def disconnect(self, *, retry_connection: bool = True) -> None:  # noqa: ARG002
        return

    def send_message(self, message: object) -> None:
        self.sent_json.append(message)

    def send_role_message(self, role: str, message: object) -> None:  # noqa: ARG002
        self.sent_json.append(message)

    def send_binary(
        self,
        data: bytes,
        *,
        role: str,  # noqa: ARG002
        timestamp_us: int,  # noqa: ARG002
        message_type: int,  # noqa: ARG002
        buffer_end_time_us: int | None = None,
        buffer_byte_count: int | None = None,
        duration_us: int | None = None,
    ) -> bool:
        self.sent_binary.append(data)
        if (
            self.buffer_tracker is not None
            and buffer_end_time_us is not None
            and buffer_byte_count is not None
        ):
            self.buffer_tracker.register(buffer_end_time_us, buffer_byte_count, duration_us or 0)
        return True


def _make_connected_player(
    server: _DummyServer,
    group: _DummyGroup,
    client_id: str,
    *,
    channel_id: UUID = MAIN_CHANNEL,
) -> tuple[SendspinClient, _CaptureConnection]:
    """Create a connected PCM player optionally bound to a specific channel."""
    client = SendspinClient(server, client_id=client_id)
    client._group = group  # noqa: SLF001
    group.clients.append(client)

    conn = _CaptureConnection()
    hello = type("Hello", (), {})()
    hello.client_id = client_id
    hello.name = client_id
    hello.player_support = ClientHelloPlayerSupport(
        supported_formats=[
            SupportedAudioFormat(
                codec=AudioCodec.PCM,
                channels=2,
                sample_rate=48000,
                bit_depth=16,
            )
        ],
        buffer_capacity=200_000,
        supported_commands=[PlayerCommand.VOLUME, PlayerCommand.MUTE],
    )
    hello.artwork_support = None
    hello.visualizer_support = None

    client.attach_connection(
        conn,
        client_info=hello,
        negotiated_roles=[Roles.PLAYER.value],
        active_roles=[Roles.PLAYER.value],
    )
    client.mark_connected()
    role = client.role("player@v1")
    if role is not None:
        conn.buffer_tracker = role.get_buffer_tracker()

    if role is not None:
        transformer = group.transformer_pool.get_or_create(
            PcmPassthrough,
            channel_id=channel_id.int,
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            frame_duration_us=25_000,
        )
        role._audio_requirements = AudioRequirements(  # noqa: SLF001
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=transformer,
            channel_id=channel_id,
            frame_duration_us=25_000,
        )

    return client, conn


@pytest.mark.asyncio
async def test_historical_injection_enables_seamless_late_join() -> None:
    """Inject historical audio into a new channel and verify a client receives it.

    Simulates the full flow:
    1. Stream audio on MAIN_CHANNEL
    2. Create a new channel with historical audio from MAIN_CHANNEL PCM cache
    3. Connect a client to the new channel
    4. Verify the client receives historical + live audio seamlessly
    """
    loop = asyncio.get_running_loop()
    clock = ManualClock()
    server = _DummyServer(loop=loop, clock=clock)
    group = _DummyGroup(clients=[])
    new_channel = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)

    # Step 1: Start stream with a MAIN_CHANNEL client
    _, conn_main = _make_connected_player(server, group, "main-player")
    stream = PushStream(loop=loop, clock=clock, group=group)

    # Commit several chunks to build PCM cache on MAIN_CHANNEL
    for _ in range(5):
        stream.prepare_audio(bytes(4800), fmt)
        await stream.commit_audio()
        clock.advance_us(25_000)

    assert conn_main.sent_binary, "MAIN_CHANNEL client should have received audio"

    # Step 2: Retrieve MAIN_CHANNEL PCM cache and inject as historical audio
    cached_pcm = stream.get_cached_pcm_chunks(MAIN_CHANNEL)
    assert len(cached_pcm) >= 3, "Should have cached PCM chunks"

    stream.enable_pcm_cache_for_channel(new_channel)
    for chunk in cached_pcm:
        stream.prepare_historical_audio(
            chunk.pcm_data,
            AudioFormat(
                sample_rate=chunk.sample_rate,
                bit_depth=chunk.bit_depth,
                channels=chunk.channels,
            ),
            channel_id=new_channel,
        )

    # Step 3: Connect a client on the new channel
    _, conn_new = _make_connected_player(server, group, "new-player", channel_id=new_channel)
    join_now_us = clock.now_us()

    # Step 4: Commit with both historical (new channel) and live (main channel)
    stream.prepare_audio(bytes(4800), fmt)
    await stream.commit_audio()

    # New channel client should have received stream/start and audio
    assert any(isinstance(m, StreamStartMessage) for m in conn_new.sent_json)
    assert conn_new.sent_binary, "New channel client should have received historical + live audio"
    first_chunk_ts = unpack_binary_header(conn_new.sent_binary[0]).timestamp_us
    assert first_chunk_ts >= join_now_us
    assert first_chunk_ts - join_now_us <= 1_000_000


@pytest.mark.asyncio
async def test_historical_injection_concurrent_channels() -> None:
    """Multiple channels can receive historical audio simultaneously."""
    loop = asyncio.get_running_loop()
    clock = ManualClock()
    server = _DummyServer(loop=loop, clock=clock)
    group = _DummyGroup(clients=[])
    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)

    channel_a = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    channel_b = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")

    # Start with MAIN_CHANNEL client to build cache
    _make_connected_player(server, group, "main-player")
    stream = PushStream(loop=loop, clock=clock, group=group)

    for _ in range(3):
        stream.prepare_audio(bytes(4800), fmt)
        await stream.commit_audio()
        clock.advance_us(25_000)

    cached_pcm = stream.get_cached_pcm_chunks(MAIN_CHANNEL)
    assert cached_pcm

    # Connect clients to both new channels
    _, conn_a = _make_connected_player(server, group, "player-a", channel_id=channel_a)
    _, conn_b = _make_connected_player(server, group, "player-b", channel_id=channel_b)

    # Inject historical into both channels
    stream.enable_pcm_cache_for_channel(channel_a)
    stream.enable_pcm_cache_for_channel(channel_b)
    for chunk in cached_pcm:
        chunk_fmt = AudioFormat(
            sample_rate=chunk.sample_rate,
            bit_depth=chunk.bit_depth,
            channels=chunk.channels,
        )
        stream.prepare_historical_audio(chunk.pcm_data, chunk_fmt, channel_id=channel_a)
        stream.prepare_historical_audio(chunk.pcm_data, chunk_fmt, channel_id=channel_b)

    # Commit
    stream.prepare_audio(bytes(4800), fmt)
    await stream.commit_audio()

    # Both new clients should have received audio
    assert conn_a.sent_binary, "Channel A client should have received audio"
    assert conn_b.sent_binary, "Channel B client should have received audio"

    # Both should have roughly similar amounts of audio
    assert abs(len(conn_a.sent_binary) - len(conn_b.sent_binary)) <= 1
