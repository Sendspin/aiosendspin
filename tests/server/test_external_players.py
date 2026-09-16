"""Tests for externally managed player registration and stream-start callbacks."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiosendspin.models.core import ClientHelloPayload
from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.models.types import (
    AudioCodec,
    BinaryMessageType,
    ConnectionReason,
    PlayerCommand,
    Roles,
)
from aiosendspin.noise.keys import Identity
from aiosendspin.noise.trust_store import InMemoryServerPairingStore
from aiosendspin.server import ClientAddedEvent, ExternalStreamStartRequest, SendspinServer
from aiosendspin.server.audio import AudioFormat
from aiosendspin.server.client import SendspinClient
from aiosendspin.server.group import SendspinGroup
from aiosendspin.server.roles.base import AudioChunk, AudioRequirements, Role
from aiosendspin.server.roles.player.v1 import PlayerPersistentState
from aiosendspin.server.roles.registry import ROLE_FACTORIES


class _DummyConnection:
    async def disconnect(self, *, retry_connection: bool = True) -> None:  # noqa: ARG002
        return

    def send_message(self, message: object) -> None:  # noqa: ARG002
        return

    def send_role_message(self, role: str, message: object) -> None:  # noqa: ARG002
        return

    def send_binary(
        self,
        data: bytes,  # noqa: ARG002
        *,
        role: str,  # noqa: ARG002
        timestamp_us: int,  # noqa: ARG002
        message_type: int,  # noqa: ARG002
        buffer_end_time_us: int | None = None,  # noqa: ARG002
        buffer_byte_count: int | None = None,  # noqa: ARG002
        duration_us: int | None = None,  # noqa: ARG002
        player_audio_header: bool = False,  # noqa: ARG002
    ) -> bool:
        return True


def _player_hello(client_id: str) -> ClientHelloPayload:
    return ClientHelloPayload(
        client_id=client_id,
        name=client_id,
        version=1,
        supported_roles=[Roles.PLAYER.value],
        player_support=ClientHelloPlayerSupport(
            supported_formats=[
                SupportedAudioFormat(
                    codec=AudioCodec.PCM,
                    channels=2,
                    sample_rate=48000,
                    bit_depth=16,
                )
            ],
            buffer_capacity=100_000,
            supported_commands=[PlayerCommand.VOLUME, PlayerCommand.MUTE],
        ),
    )


def _player_and_metadata_hello(client_id: str) -> ClientHelloPayload:
    return ClientHelloPayload(
        client_id=client_id,
        name=client_id,
        version=1,
        supported_roles=[Roles.PLAYER.value, Roles.METADATA.value],
        player_support=ClientHelloPlayerSupport(
            supported_formats=[
                SupportedAudioFormat(
                    codec=AudioCodec.PCM,
                    channels=2,
                    sample_rate=48000,
                    bit_depth=16,
                )
            ],
            buffer_capacity=100_000,
            supported_commands=[PlayerCommand.VOLUME, PlayerCommand.MUTE],
        ),
    )


def _custom_audio_hello(client_id: str) -> ClientHelloPayload:
    return ClientHelloPayload(
        client_id=client_id,
        name=client_id,
        version=1,
        supported_roles=["customaudio@v1"],
    )


def _custom_role_hello(client_id: str, role_id: str) -> ClientHelloPayload:
    return ClientHelloPayload(
        client_id=client_id,
        name=client_id,
        version=1,
        supported_roles=[role_id],
    )


def _make_server() -> SendspinServer:
    loop = asyncio.get_running_loop()
    client_session = MagicMock()
    client_session.closed = True
    client_session.close = AsyncMock()
    return SendspinServer(
        loop=loop,
        identity=Identity.generate(),
        server_name="server",
        client_session=client_session,
        pairing_store=InMemoryServerPairingStore(),
    )


async def _flush_asyncio_callbacks() -> None:
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_register_external_player_preloads_identity_and_fires_on_start_stream() -> None:
    """External player should be visible and trigger callback on stream start."""
    server = _make_server()
    callback_calls: list[ExternalStreamStartRequest] = []
    added_client_ids: list[str] = []

    def _on_server_event(_server: SendspinServer, event: object) -> None:
        if isinstance(event, ClientAddedEvent):
            added_client_ids.append(event.client_id)

    server.add_event_listener(_on_server_event)

    player = server.register_external_player(
        _player_hello("external-1"),
        on_stream_start=callback_calls.append,
    )

    assert player.client_id == "external-1"
    assert player.name == "external-1"
    assert player.info.client_id == "external-1"
    assert not player.is_connected
    assert server.is_external_player("external-1")
    assert added_client_ids == ["external-1"]

    # External clients should not emit ClientAddedEvent again on later handshake.
    server.on_client_first_connect("external-1")
    assert added_client_ids == ["external-1"]

    player.group.start_stream()

    assert len(callback_calls) == 1
    assert callback_calls[0].client_id == "external-1"


@pytest.mark.asyncio
async def test_register_external_player_cold_preinit_builds_roles_without_group_membership() -> (
    None
):
    """External registration should build cold role/caches without lifecycle side effects."""
    server = _make_server()

    player = server.register_external_player(
        _player_hello("external-preinit"),
        on_stream_start=lambda _req: None,
    )

    assert player.role(Roles.PLAYER.value) is not None
    assert player.get_binary_handling_cached(BinaryMessageType.AUDIO_CHUNK.value) is not None
    assert player.get_role_state("player", PlayerPersistentState) is None

    player_group_role = player.group.group_role("player")
    assert player_group_role is not None
    assert player_group_role.get_player_clients() == []


@pytest.mark.asyncio
async def test_attach_connection_reuses_external_cold_preinitialized_roles() -> None:
    """Attach should reuse cold-preinitialized roles and then run on_connect."""
    server = _make_server()

    player = server.register_external_player(
        _player_hello("external-reuse"),
        on_stream_start=lambda _req: None,
    )
    precreated_role = player.role(Roles.PLAYER.value)
    assert precreated_role is not None
    assert player.get_role_state("player", PlayerPersistentState) is None

    player.attach_connection(
        _DummyConnection(),
        client_info=_player_hello("external-reuse"),
        negotiated_roles=[Roles.PLAYER.value],
        active_roles=[Roles.PLAYER.value],
    )

    assert player.role(Roles.PLAYER.value) is precreated_role
    assert player.get_role_state("player", PlayerPersistentState) is not None


@pytest.mark.asyncio
async def test_attach_connection_reconciles_roles_when_cold_preinit_mismatches() -> None:
    """Attach reconciles a mismatched cold-preinit set: reuse survivors, add new roles."""
    server = _make_server()

    player = server.register_external_player(
        _player_hello("external-mismatch"),
        on_stream_start=lambda _req: None,
    )
    precreated_player_role = player.role(Roles.PLAYER.value)
    assert precreated_player_role is not None
    assert player.role(Roles.METADATA.value) is None

    player.attach_connection(
        _DummyConnection(),
        client_info=_player_and_metadata_hello("external-mismatch"),
        negotiated_roles=[Roles.PLAYER.value, Roles.METADATA.value],
        active_roles=[Roles.PLAYER.value, Roles.METADATA.value],
    )

    assert player.role(Roles.PLAYER.value) is precreated_player_role
    assert player.role(Roles.METADATA.value) is not None


@pytest.mark.asyncio
async def test_reregister_external_player_refreshes_cold_preinit_state() -> None:
    """Repeated disconnected registration should refresh cold-preinitialized role state."""
    server = _make_server()

    player = server.register_external_player(
        _player_hello("external-refresh"),
        on_stream_start=lambda _req: None,
    )
    first_player_role = player.role(Roles.PLAYER.value)
    assert first_player_role is not None
    assert player.role(Roles.METADATA.value) is None

    refreshed = server.register_external_player(
        _player_and_metadata_hello("external-refresh"),
        on_stream_start=lambda _req: None,
    )

    assert refreshed is player
    assert player.role(Roles.PLAYER.value) is not first_player_role
    assert player.role(Roles.METADATA.value) is not None
    assert not player.is_connected


@pytest.mark.asyncio
async def test_cold_preinitialized_custom_role_can_receive_audio_without_warm_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Custom roles can opt into preconnect audio without touching warm-disconnected state."""

    class _CustomAudioRole(Role):
        def __init__(self, client: SendspinClient) -> None:
            self._client = client
            self._chunks: list[AudioChunk] = []

        @property
        def role_id(self) -> str:
            return "customaudio@v1"

        @property
        def role_family(self) -> str:
            return "customaudio"

        @property
        def chunk_count(self) -> int:
            return len(self._chunks)

        def get_audio_requirements(self) -> AudioRequirements | None:
            return AudioRequirements(sample_rate=48000, bit_depth=16, channels=2)

        def supports_preconnect_audio(self) -> bool:
            return True

        def on_connect(self) -> None:
            return

        def on_disconnect(self) -> None:
            return

        def on_audio_chunk(self, chunk: AudioChunk) -> None:
            self._chunks.append(chunk)

    monkeypatch.setitem(ROLE_FACTORIES, "customaudio@v1", _CustomAudioRole)

    server = _make_server()
    player = server.register_external_player(
        _custom_audio_hello("external-custom"),
        on_stream_start=lambda _req: None,
    )
    assert not player.has_warm_disconnected_roles
    assert player.has_cold_preinitialized_roles

    role = player.role("customaudio@v1")
    assert isinstance(role, _CustomAudioRole)
    assert role.chunk_count == 0

    stream = player.group.start_stream()
    stream.prepare_audio(
        b"\x00\x00" * 480,
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
    )
    await stream.commit_audio()

    assert role.chunk_count > 0


@pytest.mark.asyncio
async def test_add_external_player_to_active_group_requests_external_connect() -> None:
    """Adding disconnected external player to active stream should call callback."""
    server = _make_server()
    callback_calls: list[ExternalStreamStartRequest] = []

    owner = SendspinClient(server, client_id="owner")
    SendspinGroup(server, owner)
    owner.attach_connection(
        _DummyConnection(),
        client_info=_player_hello("owner"),
        negotiated_roles=[Roles.PLAYER.value],
        active_roles=[Roles.PLAYER.value],
    )
    owner.mark_connected()
    owner.group.start_stream()

    external = server.register_external_player(
        _player_hello("external-2"),
        on_stream_start=callback_calls.append,
    )

    await owner.group.add_client(external)

    assert len(callback_calls) == 1
    assert callback_calls[0].client_id == "external-2"


@pytest.mark.asyncio
async def test_add_external_preconnect_player_to_active_group_replays_cached_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disconnected preconnect roles should get late-join replay on active stream add."""

    class _PreconnectAudioRole(Role):
        def __init__(self, client: SendspinClient) -> None:
            self._client = client
            self._chunks: list[AudioChunk] = []
            self.got_chunk = asyncio.Event()

        @property
        def role_id(self) -> str:
            return "preconnectaudio@v1"

        @property
        def role_family(self) -> str:
            return "preconnectaudio"

        @property
        def chunk_count(self) -> int:
            return len(self._chunks)

        def get_audio_requirements(self) -> AudioRequirements | None:
            return AudioRequirements(sample_rate=48000, bit_depth=16, channels=2)

        def supports_preconnect_audio(self) -> bool:
            return True

        def on_connect(self) -> None:
            return

        def on_disconnect(self) -> None:
            return

        def on_audio_chunk(self, chunk: AudioChunk) -> None:
            self._chunks.append(chunk)
            self.got_chunk.set()

    monkeypatch.setitem(ROLE_FACTORIES, "preconnectaudio@v1", _PreconnectAudioRole)

    server = _make_server()
    callback_calls: list[ExternalStreamStartRequest] = []

    owner = SendspinClient(server, client_id="owner-preconnect")
    SendspinGroup(server, owner)
    owner.attach_connection(
        _DummyConnection(),
        client_info=_player_hello("owner-preconnect"),
        negotiated_roles=[Roles.PLAYER.value],
        active_roles=[Roles.PLAYER.value],
    )
    owner.mark_connected()
    stream = owner.group.start_stream()
    stream.prepare_audio(
        bytes(48000 * 2 * 2),
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
    )
    await stream.commit_audio()

    external = server.register_external_player(
        _custom_role_hello("external-preconnect", "preconnectaudio@v1"),
        on_stream_start=callback_calls.append,
    )
    role = external.role("preconnectaudio@v1")
    assert isinstance(role, _PreconnectAudioRole)
    assert role.chunk_count == 0

    await owner.group.add_client(external)

    assert len(callback_calls) == 1
    assert callback_calls[0].client_id == "external-preconnect"

    await asyncio.wait_for(role.got_chunk.wait(), timeout=2.0)
    assert role.chunk_count > 0


@pytest.mark.asyncio
async def test_add_external_non_preconnect_player_to_active_group_skips_role_join(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disconnected external roles without preconnect support should not run role join."""

    class _ColdAudioRole(Role):
        def __init__(self, client: SendspinClient) -> None:
            self._client = client

        @property
        def role_id(self) -> str:
            return "coldaudio@v1"

        @property
        def role_family(self) -> str:
            return "coldaudio"

        def get_audio_requirements(self) -> AudioRequirements | None:
            return AudioRequirements(sample_rate=48000, bit_depth=16, channels=2)

        def supports_preconnect_audio(self) -> bool:
            return False

        def on_connect(self) -> None:
            return

        def on_disconnect(self) -> None:
            return

    monkeypatch.setitem(ROLE_FACTORIES, "coldaudio@v1", _ColdAudioRole)

    server = _make_server()
    callback_calls: list[ExternalStreamStartRequest] = []

    owner = SendspinClient(server, client_id="owner-cold")
    SendspinGroup(server, owner)
    owner.attach_connection(
        _DummyConnection(),
        client_info=_player_hello("owner-cold"),
        negotiated_roles=[Roles.PLAYER.value],
        active_roles=[Roles.PLAYER.value],
    )
    owner.mark_connected()
    stream = owner.group.start_stream()

    external = server.register_external_player(
        _custom_role_hello("external-cold", "coldaudio@v1"),
        on_stream_start=callback_calls.append,
    )

    with patch.object(stream, "on_role_join") as mock_on_role_join:
        await owner.group.add_client(external)

    assert len(callback_calls) == 1
    assert callback_calls[0].client_id == "external-cold"
    mock_on_role_join.assert_not_called()


@pytest.mark.asyncio
async def test_register_external_player_timeout_full_unregisters_disconnected_client() -> None:
    """External registration timeout removes callback and client while disconnected."""
    server = _make_server()
    callback_calls: list[ExternalStreamStartRequest] = []

    server.register_external_player(
        _player_hello("external-timeout"),
        on_stream_start=callback_calls.append,
        timeout_s=0.05,
    )
    assert server.get_client("external-timeout") is not None
    assert server.is_external_player("external-timeout")

    await asyncio.sleep(0.08)
    await _flush_asyncio_callbacks()

    assert server.get_client("external-timeout") is None
    assert not server.is_external_player("external-timeout")


@pytest.mark.asyncio
async def test_register_external_player_timeout_cancelled_on_transport_attach() -> None:
    """External registration timeout is cancelled once transport attaches."""
    server = _make_server()

    client = server.register_external_player(
        _player_hello("external-connected"),
        on_stream_start=lambda _req: None,
        timeout_s=0.05,
    )
    client.attach_connection(
        _DummyConnection(),
        client_info=_player_hello("external-connected"),
        negotiated_roles=[Roles.PLAYER.value],
        active_roles=[Roles.PLAYER.value],
    )

    await asyncio.sleep(0.08)
    await _flush_asyncio_callbacks()

    assert server.get_client("external-connected") is client
    assert server.is_external_player("external-connected")


@pytest.mark.asyncio
async def test_reclaim_timeout_full_unregisters_disconnected_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reclaim timeout removes client when reconnect never succeeds."""
    server = _make_server()
    added_client_ids: list[str] = []
    server.add_event_listener(
        lambda _server, event: (
            added_client_ids.append(event.client_id)
            if isinstance(event, ClientAddedEvent)
            else None
        )
    )

    server.get_or_create_client("speaker-timeout")
    assert added_client_ids == []
    server.on_client_first_connect("speaker-timeout")
    assert added_client_ids == ["speaker-timeout"]

    server.register_client_url("speaker-timeout", "ws://127.0.0.1:9000/sendspin")

    connect_calls: list[tuple[str, ConnectionReason]] = []

    def _connect_to_client(url: str, *, connection_reason: ConnectionReason) -> None:
        connect_calls.append((url, connection_reason))

    monkeypatch.setattr(server, "connect_to_client", _connect_to_client)

    assert server.reclaim_client_for_playback("speaker-timeout", timeout_s=0.05)
    assert connect_calls == [("ws://127.0.0.1:9000/sendspin", ConnectionReason.PLAYBACK)]

    await asyncio.sleep(0.08)
    await _flush_asyncio_callbacks()

    assert server.get_client("speaker-timeout") is None
    assert server.get_client_url("speaker-timeout") is None


@pytest.mark.asyncio
async def test_full_unregister_skips_client_with_reconnect_in_flight() -> None:
    """A reconnect handshake in flight keeps the client until it attaches or fails."""
    server = _make_server()
    server.get_or_create_client("reconnecting")
    url = "ws://127.0.0.1:9003/sendspin"
    server.register_client_url("reconnecting", url)

    async def _never() -> None:
        await asyncio.Event().wait()

    task = asyncio.get_running_loop().create_task(_never())
    server._connection_tasks[url] = task  # noqa: SLF001
    try:
        await server._full_unregister_disconnected_client("reconnecting")  # noqa: SLF001
        assert server.get_client("reconnecting") is not None
    finally:
        task.cancel()


async def _defer_unregister_with_task(server: SendspinServer, client_id: str, url: str) -> None:
    """Register a client and defer its unregister behind an in-flight reconnect task."""
    server.get_or_create_client(client_id)
    server.register_client_url(client_id, url)
    task = asyncio.get_running_loop().create_task(asyncio.Event().wait())
    server._connection_tasks[url] = task  # noqa: SLF001
    try:
        await server._full_unregister_disconnected_client(client_id)  # noqa: SLF001
        assert server.get_client(client_id) is not None
    finally:
        task.cancel()
    server._connection_tasks.pop(url, None)  # noqa: SLF001


@pytest.mark.asyncio
async def test_deferred_unregister_completes_when_reconnect_gives_up() -> None:
    """A deferred unregister fires once the reconnect task ends without connecting."""
    server = _make_server()
    url = "ws://127.0.0.1:9004/sendspin"
    await _defer_unregister_with_task(server, "reconnecting", url)

    await server._complete_deferred_unregister(  # noqa: SLF001
        url, connection_succeeded=False, cancelled=False
    )

    assert server.get_client("reconnecting") is None
    assert server.get_client_url("reconnecting") is None


@pytest.mark.asyncio
async def test_deferred_unregister_suppressed_when_reconnect_attaches() -> None:
    """A deferred unregister is dropped, not fired, when the reconnect established a session."""
    server = _make_server()
    url = "ws://127.0.0.1:9005/sendspin"
    await _defer_unregister_with_task(server, "reconnecting", url)

    await server._complete_deferred_unregister(  # noqa: SLF001
        url, connection_succeeded=True, cancelled=False
    )

    assert server.get_client("reconnecting") is not None


@pytest.mark.asyncio
async def test_reclaim_timeout_refreshes_on_repeated_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated reclaim requests refresh the timeout window."""
    server = _make_server()
    server.get_or_create_client("speaker-refresh")
    server.register_client_url("speaker-refresh", "ws://127.0.0.1:9001/sendspin")

    monkeypatch.setattr(
        server,
        "connect_to_client",
        lambda _url, *, connection_reason: None,  # noqa: ARG005
    )

    # Inflated to ~5x the timeout to absorb CI scheduler jitter (#28 in REVIEW.md)
    assert server.reclaim_client_for_playback("speaker-refresh", timeout_s=0.5)
    await asyncio.sleep(0.3)
    assert server.reclaim_client_for_playback("speaker-refresh", timeout_s=0.5)

    await asyncio.sleep(0.3)
    await _flush_asyncio_callbacks()
    assert server.get_client("speaker-refresh") is not None

    await asyncio.sleep(0.7)
    await _flush_asyncio_callbacks()
    assert server.get_client("speaker-refresh") is None


@pytest.mark.asyncio
async def test_reclaim_timeout_cancelled_on_transport_attach(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reclaim timeout does not remove client once transport has attached."""
    server = _make_server()
    client = server.get_or_create_client("speaker-connected")
    server.register_client_url("speaker-connected", "ws://127.0.0.1:9002/sendspin")

    monkeypatch.setattr(
        server,
        "connect_to_client",
        lambda _url, *, connection_reason: None,  # noqa: ARG005
    )

    assert server.reclaim_client_for_playback("speaker-connected", timeout_s=0.05)
    client.attach_connection(
        _DummyConnection(),
        client_info=_player_hello("speaker-connected"),
        negotiated_roles=[Roles.PLAYER.value],
        active_roles=[Roles.PLAYER.value],
    )

    await asyncio.sleep(0.08)
    await _flush_asyncio_callbacks()

    assert server.get_client("speaker-connected") is client


_TRACKING_ROLE_ID = "player@_tracking"


class _TrackingRole(Role):
    """Player-family role that records lifecycle hook invocations."""

    def __init__(self, client: SendspinClient) -> None:
        self._client = client
        self.connects = 0
        self.disconnects = 0

    @property
    def role_id(self) -> str:
        return _TRACKING_ROLE_ID

    @property
    def role_family(self) -> str:
        return "player"

    def on_connect(self) -> None:
        self.connects += 1
        self._subscribe_to_group_role()

    def on_disconnect(self) -> None:
        self.disconnects += 1
        self._unsubscribe_from_group_role()


@pytest.mark.asyncio
async def test_attach_preinitialized_roles_subscribes_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """attach_preinitialized_roles runs connect hooks exactly once and joins group roles."""
    monkeypatch.setitem(ROLE_FACTORIES, _TRACKING_ROLE_ID, _TrackingRole)
    server = _make_server()

    player = server.register_external_player(
        _custom_role_hello("external-attach", _TRACKING_ROLE_ID),
        on_stream_start=lambda _req: None,
    )
    role = player.role(_TRACKING_ROLE_ID)
    assert isinstance(role, _TrackingRole)
    group_role = player.group.group_role("player")
    assert group_role is not None
    assert role not in group_role._members  # noqa: SLF001

    player.attach_preinitialized_roles()
    player.attach_preinitialized_roles()

    assert role.connects == 1
    assert role in group_role._members  # noqa: SLF001
    assert player.has_cold_preinitialized_roles


@pytest.mark.asyncio
async def test_transport_attach_after_attach_preinitialized_roles_connects_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later transport attach reuses externally attached roles without a second on_connect."""
    monkeypatch.setitem(ROLE_FACTORIES, _TRACKING_ROLE_ID, _TrackingRole)
    server = _make_server()

    player = server.register_external_player(
        _custom_role_hello("external-attach-reuse", _TRACKING_ROLE_ID),
        on_stream_start=lambda _req: None,
    )
    player.attach_preinitialized_roles()
    role = player.role(_TRACKING_ROLE_ID)
    assert isinstance(role, _TrackingRole)

    player.attach_connection(
        _DummyConnection(),
        client_info=_custom_role_hello("external-attach-reuse", _TRACKING_ROLE_ID),
        negotiated_roles=[_TRACKING_ROLE_ID],
        active_roles=[_TRACKING_ROLE_ID],
    )

    assert player.role(_TRACKING_ROLE_ID) is role
    assert role.connects == 1
    assert role.disconnects == 0
    group_role = player.group.group_role("player")
    assert group_role is not None
    assert role in group_role._members  # noqa: SLF001


@pytest.mark.asyncio
async def test_transport_attach_disconnects_dropped_externally_attached_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A role set change on transport attach runs disconnect hooks on attached externals."""
    monkeypatch.setitem(ROLE_FACTORIES, _TRACKING_ROLE_ID, _TrackingRole)
    server = _make_server()

    player = server.register_external_player(
        _custom_role_hello("external-attach-drop", _TRACKING_ROLE_ID),
        on_stream_start=lambda _req: None,
    )
    player.attach_preinitialized_roles()
    role = player.role(_TRACKING_ROLE_ID)
    assert isinstance(role, _TrackingRole)

    player.attach_connection(
        _DummyConnection(),
        client_info=_player_hello("external-attach-drop"),
        negotiated_roles=[Roles.PLAYER.value],
        active_roles=[Roles.PLAYER.value],
    )

    assert player.role(_TRACKING_ROLE_ID) is None
    assert role.disconnects == 1
    group_role = player.group.group_role("player")
    assert group_role is not None
    assert role not in group_role._members  # noqa: SLF001


@pytest.mark.asyncio
async def test_reregister_disconnects_externally_attached_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-registration after attach_preinitialized_roles unsubscribes the old roles."""
    monkeypatch.setitem(ROLE_FACTORIES, _TRACKING_ROLE_ID, _TrackingRole)
    server = _make_server()

    player = server.register_external_player(
        _custom_role_hello("external-rereg", _TRACKING_ROLE_ID),
        on_stream_start=lambda _req: None,
    )
    player.attach_preinitialized_roles()
    role = player.role(_TRACKING_ROLE_ID)
    assert isinstance(role, _TrackingRole)

    server.register_external_player(
        _player_hello("external-rereg"),
        on_stream_start=lambda _req: None,
    )

    assert role.disconnects == 1
    group_role = player.group.group_role("player")
    assert group_role is not None
    assert role not in group_role._members  # noqa: SLF001


@pytest.mark.asyncio
async def test_info_accessors_before_and_after_hello() -> None:
    """Info raises before the first hello; info_or_none returns None, then the payload."""
    server = _make_server()
    client = server.get_or_create_client("no-hello-yet")

    assert client.info_or_none is None
    with pytest.raises(RuntimeError, match="client/hello has not been processed"):
        _ = client.info

    hello = _player_hello("no-hello-yet")
    server.register_external_player(hello, on_stream_start=lambda _req: None)

    assert client.info_or_none is hello
    assert client.info is hello


@pytest.mark.asyncio
async def test_register_external_player_cancels_pending_cleanup() -> None:
    """Registering just after a warm disconnect disarms the pending cleanup timer."""
    server = _make_server()
    client = server.get_or_create_client("late-register")
    client.attach_connection(
        _DummyConnection(),
        client_info=_player_hello("late-register"),
        negotiated_roles=[Roles.PLAYER.value],
        active_roles=[Roles.PLAYER.value],
    )
    client.mark_connected()

    # Warm disconnect arms the delayed registry-cleanup timer.
    client.detach_connection(goodbye_reason=None)
    assert client._cleanup_handle is not None  # noqa: SLF001

    server.register_external_player(
        _player_hello("late-register"),
        on_stream_start=lambda _req: None,
    )

    assert client._cleanup_handle is None  # noqa: SLF001
    assert server.get_client("late-register") is client
