"""Integration test for late joiners requesting a different codec."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from uuid import UUID

import pytest

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
from aiosendspin.server.roles.player.audio_transformers import FlacEncoder, PcmPassthrough


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
        self.sent_binary: list[tuple[int, bytes]] = []
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
        timestamp_us: int,
        message_type: int,  # noqa: ARG002
        buffer_end_time_us: int | None = None,
        buffer_byte_count: int | None = None,
        duration_us: int | None = None,
    ) -> bool:
        self.sent_binary.append((timestamp_us, data))
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
    codec: AudioCodec,
) -> tuple[SendspinClient, _CaptureConnection]:
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
                codec=codec,
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
        if codec == AudioCodec.FLAC:
            transformer = group.transformer_pool.get_or_create(
                FlacEncoder,
                channel_id=MAIN_CHANNEL.int,
                sample_rate=48000,
                bit_depth=16,
                channels=2,
                frame_duration_us=25_000,
            )
        else:
            transformer = group.transformer_pool.get_or_create(
                PcmPassthrough,
                channel_id=MAIN_CHANNEL.int,
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
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )

    return client, conn


@pytest.mark.asyncio
async def test_late_joiner_receives_catchup_for_uncached_codec() -> None:
    """Late joiner with an uncached codec receives catch-up audio."""
    loop = asyncio.get_running_loop()
    clock = ManualClock()
    server = _DummyServer(loop=loop, clock=clock)
    group = _DummyGroup(clients=[])

    _, conn1 = _make_connected_player(server, group, "flac-client", codec=AudioCodec.FLAC)
    stream = PushStream(loop=loop, clock=clock, group=group)

    stream.prepare_audio(
        bytes(19_200),
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
    )
    await stream.commit_audio()
    assert conn1.sent_binary

    _, conn2 = _make_connected_player(server, group, "pcm-client", codec=AudioCodec.PCM)
    role2 = group.clients[-1].role("player@v1")
    assert role2 is not None
    role2.get_join_delay_s = lambda: 0.0  # type: ignore[method-assign]
    stream.on_role_join(role2)
    stream.prepare_audio(
        bytes(19_200),
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
    )
    await stream.commit_audio()

    await asyncio.sleep(0)

    assert any(isinstance(m, StreamStartMessage) for m in conn2.sent_json)
    assert conn2.sent_binary, "expected PCM catch-up audio for late joiner"
    first_ts, _ = conn2.sent_binary[0]
    assert first_ts >= clock.now_us(), (
        f"late joiner first chunk ts {first_ts} < now {clock.now_us()} "
        "(spec README:177 mandates future-only timestamps)"
    )
