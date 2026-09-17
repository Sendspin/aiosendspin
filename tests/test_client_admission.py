"""Tests for the client-side connection arbiter (multi-server admission)."""
# ruff: noqa: SLF001 - exercises the client's internal admission methods directly.

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiosendspin.client.client import SendspinClient
from aiosendspin.client.connection import SendspinConnection
from aiosendspin.models.types import Activity, GoodbyeReason, PairAbortReason, Roles
from aiosendspin.noise.keys import generate_psk, psk_id_for
from aiosendspin.noise.trust_store import (
    ClientPairingRecord,
    InMemoryClientPairingStore,
    PskCategory,
    ResolvedPsk,
)

from .conftest import make_sdk_client
from .pairing_stores import seed_used_client_records


class _FakeConnection:
    """Stand-in exposing only the surface the arbiter touches on a connection."""

    def __init__(
        self,
        *,
        server_id: str | None = "server",
        activities: list[Activity] | None = None,
        connected: bool = True,
        pairing_attempt_in_progress: bool = False,
        client: SendspinClient | None = None,
    ) -> None:
        """Record the connection's arbitration-relevant state and call sinks."""
        self.server_id = server_id
        self.activities = activities or []
        self.connected = connected
        self.pairing_attempt_in_progress = pairing_attempt_in_progress
        self._client = client
        self.goodbye_reason: GoodbyeReason | None = None
        self.pair_abort_reason: PairAbortReason | None = None
        self.disconnected = False

    @property
    def is_pairing(self) -> bool:
        """Whether this connection is currently a pairing connection."""
        return Activity.PAIRING in self.activities

    async def send_goodbye(self, reason: GoodbyeReason) -> None:
        """Record the goodbye reason the arbiter sent."""
        self.goodbye_reason = reason

    async def send_pair_abort(self, reason: PairAbortReason) -> None:
        """Record the pair/abort reason the arbiter sent."""
        self.pair_abort_reason = reason

    async def disconnect(self) -> None:
        """Mark disconnected and notify the owning client."""
        self.disconnected = True
        if self._client is not None:
            self._client.on_connection_closed(self)  # type: ignore[arg-type]


def _client() -> SendspinClient:
    return make_sdk_client(client_name="Test Client", roles=[Roles.CONTROLLER])


# --- _activity_rank ---


def test_activity_rank_orders_playback_over_pairing() -> None:
    """Playback > pairing > none."""
    rank = SendspinClient._activity_rank
    assert rank([Activity.PLAYBACK]) == 2
    assert rank([Activity.PAIRING]) == 1
    assert rank([]) == 0


# DEPRECATED(spec-pr-183): remove in aiosendspin <version>
def test_activity_rank_ignores_management() -> None:
    """Management adds nothing to a connection's rank."""
    rank = SendspinClient._activity_rank
    assert rank([Activity.MANAGEMENT, Activity.PLAYBACK]) == 2
    assert rank([Activity.MANAGEMENT]) == 0


# --- _should_admit_connection ---


async def test_admits_when_no_current_connection() -> None:
    """The first connection is always admitted."""
    client = _client()
    incoming = _FakeConnection(activities=[Activity.PAIRING])
    assert client._should_admit_connection(incoming) is True


async def test_admits_when_current_connection_disconnected() -> None:
    """A dead holder never blocks an incoming connection."""
    client = _client()
    client._admitted_connection = _FakeConnection(connected=False)  # type: ignore[assignment]
    incoming = _FakeConnection(activities=[Activity.PAIRING])
    assert client._should_admit_connection(incoming) is True


async def test_higher_rank_displaces_lower() -> None:
    """A playback connection displaces an admitted pairing connection."""
    client = _client()
    client._admitted_connection = _FakeConnection(  # type: ignore[assignment]
        activities=[Activity.PAIRING]
    )
    incoming = _FakeConnection(activities=[Activity.PLAYBACK])
    assert client._should_admit_connection(incoming) is True


async def test_lower_rank_rejected() -> None:
    """A pairing connection does not displace an admitted playback connection."""
    client = _client()
    client._admitted_connection = _FakeConnection(  # type: ignore[assignment]
        activities=[Activity.PLAYBACK]
    )
    incoming = _FakeConnection(activities=[Activity.PAIRING])
    assert client._should_admit_connection(incoming) is False


# DEPRECATED(spec-pr-183): remove in aiosendspin <version>
async def test_management_does_not_outrank_playback() -> None:
    """A management-only connection does not displace an admitted playback connection."""
    client = _client()
    client._admitted_connection = _FakeConnection(  # type: ignore[assignment]
        activities=[Activity.PLAYBACK]
    )
    incoming = _FakeConnection(activities=[Activity.MANAGEMENT])
    assert client._should_admit_connection(incoming) is False


# DEPRECATED(spec-pr-183): remove in aiosendspin <version>
async def test_management_ranks_with_its_other_activities() -> None:
    """Management alongside playback ranks as playback against an admitted playback holder."""
    client = _client()
    client._admitted_connection = _FakeConnection(  # type: ignore[assignment]
        activities=[Activity.PLAYBACK]
    )
    incoming = _FakeConnection(activities=[Activity.PLAYBACK, Activity.MANAGEMENT])
    assert client._should_admit_connection(incoming) is True


async def test_equal_nonempty_rank_admitted() -> None:
    """An incoming playback connection displaces an admitted playback connection."""
    client = _client()
    client._admitted_connection = _FakeConnection(  # type: ignore[assignment]
        activities=[Activity.PLAYBACK]
    )
    incoming = _FakeConnection(activities=[Activity.PLAYBACK])
    assert client._should_admit_connection(incoming) is True


async def test_inflight_pairing_not_displaced_by_pairing() -> None:
    """An in-flight pairing attempt is protected from a competing incoming pairing."""
    client = _client()
    client._admitted_connection = _FakeConnection(  # type: ignore[assignment]
        activities=[Activity.PAIRING], pairing_attempt_in_progress=True
    )
    incoming = _FakeConnection(activities=[Activity.PAIRING])
    assert client._should_admit_connection(incoming) is False


async def test_inflight_pairing_not_displaced_by_playback() -> None:
    """An in-flight pairing attempt is protected from an incoming playback connection."""
    client = _client()
    client._admitted_connection = _FakeConnection(  # type: ignore[assignment]
        activities=[Activity.PAIRING], pairing_attempt_in_progress=True
    )
    incoming = _FakeConnection(activities=[Activity.PLAYBACK])
    assert client._should_admit_connection(incoming) is False


# DEPRECATED(spec-pr-183): remove in aiosendspin <version>
@pytest.mark.parametrize(
    "activities",
    [[Activity.MANAGEMENT], [Activity.PLAYBACK, Activity.MANAGEMENT]],
)
async def test_inflight_pairing_not_displaced_by_management(activities: list[Activity]) -> None:
    """Management never displaces an in-flight pairing attempt."""
    client = _client()
    client._admitted_connection = _FakeConnection(  # type: ignore[assignment]
        activities=[Activity.PAIRING], pairing_attempt_in_progress=True
    )
    incoming = _FakeConnection(activities=activities)
    assert client._should_admit_connection(incoming) is False


async def test_idle_pairing_displaced_by_playback() -> None:
    """A pairing connection with no attempt in progress is not protected."""
    client = _client()
    client._admitted_connection = _FakeConnection(  # type: ignore[assignment]
        activities=[Activity.PAIRING], pairing_attempt_in_progress=False
    )
    incoming = _FakeConnection(activities=[Activity.PLAYBACK])
    assert client._should_admit_connection(incoming) is True


async def test_pairing_displaces_idle() -> None:
    """A pairing connection displaces an admitted idle (no-activity) connection."""
    client = _client()
    client._admitted_connection = _FakeConnection(activities=[])  # type: ignore[assignment]
    incoming = _FakeConnection(activities=[Activity.PAIRING])
    assert client._should_admit_connection(incoming) is True


async def test_both_idle_incoming_wins_when_last_playback() -> None:
    """Two idle connections resolve in favour of the last-playback server."""
    client = _client()
    client.last_playback_server_id = "server-A"
    client._admitted_connection = _FakeConnection(  # type: ignore[assignment]
        server_id="server-B", activities=[]
    )
    incoming = _FakeConnection(server_id="server-A", activities=[])
    assert client._should_admit_connection(incoming) is True


async def test_both_idle_incoming_loses_when_not_last_playback() -> None:
    """An idle incoming connection does not displace an idle holder it can't beat."""
    client = _client()
    client.last_playback_server_id = "server-A"
    client._admitted_connection = _FakeConnection(  # type: ignore[assignment]
        server_id="server-A", activities=[]
    )
    incoming = _FakeConnection(server_id="server-B", activities=[])
    assert client._should_admit_connection(incoming) is False


# --- _admit_connection / _reject_connection ---


async def test_admit_displaces_previous_with_another_server() -> None:
    """Admitting a new connection sends the prior holder another_server and drops it."""
    client = _client()
    previous = _FakeConnection(activities=[Activity.PLAYBACK], client=client)
    client._admitted_connection = previous  # type: ignore[assignment]
    incoming = _FakeConnection(activities=[Activity.PLAYBACK], client=client)

    await client._admit_connection(incoming)  # type: ignore[arg-type]

    assert client._admitted_connection is incoming  # type: ignore[comparison-overlap]
    assert previous.goodbye_reason is GoodbyeReason.ANOTHER_SERVER
    assert previous.disconnected is True


async def test_admit_same_connection_is_noop() -> None:
    """Re-admitting the current connection neither dismisses nor disconnects it."""
    client = _client()
    current = _FakeConnection(activities=[Activity.PLAYBACK], client=client)
    client._admitted_connection = current  # type: ignore[assignment]

    await client._admit_connection(current)  # type: ignore[arg-type]

    assert client._admitted_connection is current  # type: ignore[comparison-overlap]
    assert current.disconnected is False
    assert current.goodbye_reason is None


async def test_admit_displaces_previous_pairing_with_pair_abort() -> None:
    """Displacing an in-flight pairing aborts it rather than sending a goodbye."""
    client = _client()
    previous = _FakeConnection(activities=[Activity.PAIRING], client=client)
    client._admitted_connection = previous  # type: ignore[assignment]
    incoming = _FakeConnection(activities=[Activity.PLAYBACK], client=client)

    await client._admit_connection(incoming)  # type: ignore[arg-type]

    assert previous.pair_abort_reason is PairAbortReason.CONCURRENT_ATTEMPT
    assert previous.goodbye_reason is None
    assert previous.disconnected is True


async def test_reject_sends_concurrent_attempt_and_disconnects() -> None:
    """A rejected incoming connection gets concurrent_attempt and is dropped."""
    client = _client()
    incoming = _FakeConnection(activities=[Activity.PLAYBACK], client=client)

    await client._reject_connection(incoming)  # type: ignore[arg-type]

    assert incoming.goodbye_reason is GoodbyeReason.CONCURRENT_ATTEMPT
    assert incoming.disconnected is True
    assert client._admitted_connection is None


async def test_reject_pairing_sends_pair_abort() -> None:
    """A rejected incoming pairing connection gets a pair/abort, not a goodbye."""
    client = _client()
    incoming = _FakeConnection(activities=[Activity.PAIRING], client=client)

    await client._reject_connection(incoming)  # type: ignore[arg-type]

    assert incoming.pair_abort_reason is PairAbortReason.CONCURRENT_ATTEMPT
    assert incoming.goodbye_reason is None
    assert incoming.disconnected is True


# --- note_playback_activity ---


async def test_admit_playback_connection_records_last_playback() -> None:
    """Admitting a connection that carries the playback activity records its server."""
    client = _client()
    incoming = _FakeConnection(server_id="server-A", activities=[Activity.PLAYBACK], client=client)

    await client._admit_connection(incoming)  # type: ignore[arg-type]

    assert client.last_playback_server_id == "server-A"


async def test_admit_idle_connection_does_not_record() -> None:
    """Admitting an idle connection leaves the last-playback pointer untouched."""
    client = _client()
    client.last_playback_server_id = "server-A"
    incoming = _FakeConnection(server_id="server-B", activities=[], client=client)

    await client._admit_connection(incoming)  # type: ignore[arg-type]

    assert client.last_playback_server_id == "server-A"


async def test_admit_pairing_connection_does_not_record() -> None:
    """A pairing connection is not playback and never becomes the last-playback server."""
    client = _client()
    incoming = _FakeConnection(server_id="server-B", activities=[Activity.PAIRING], client=client)

    await client._admit_connection(incoming)  # type: ignore[arg-type]

    assert client.last_playback_server_id is None


async def test_note_playback_activity_ignores_non_admitted() -> None:
    """A connection that is not the admitted holder cannot move the pointer."""
    client = _client()
    admitted = _FakeConnection(server_id="server-A", activities=[], client=client)
    client._admitted_connection = admitted  # type: ignore[assignment]
    other = _FakeConnection(server_id="server-B", activities=[Activity.PLAYBACK], client=client)

    await client.note_playback_activity(other)  # type: ignore[arg-type]

    assert client.last_playback_server_id is None


async def test_note_playback_activity_persists_later_activation() -> None:
    """A server/activate that adds playback after admission is persisted, not just cached."""
    store = InMemoryClientPairingStore()
    client = make_sdk_client(client_name="c", roles=[Roles.CONTROLLER], pairing_store=store)
    admitted = _FakeConnection(server_id="server-A", activities=[], client=client)
    client._admitted_connection = admitted  # type: ignore[assignment]
    admitted.activities = [Activity.PLAYBACK]  # a later activate declares playback

    await client.note_playback_activity(admitted)  # type: ignore[arg-type]

    assert await store.get_last_playback_server_id() == "server-A"


async def test_failed_last_playback_write_retries_on_next_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed store write leaves the cached pointer stale so a later activation retries."""
    store = InMemoryClientPairingStore()
    client = make_sdk_client(client_name="c", roles=[Roles.CONTROLLER], pairing_store=store)
    admitted = _FakeConnection(server_id="server-A", activities=[Activity.PLAYBACK], client=client)
    client._admitted_connection = admitted  # type: ignore[assignment]
    written: list[str | None] = []

    async def flaky(server_id: str | None) -> None:
        written.append(server_id)
        if len(written) == 1:
            raise OSError("store unavailable")

    monkeypatch.setattr(store, "set_last_playback_server_id", flaky)

    with pytest.raises(OSError, match="store unavailable"):
        await client.note_playback_activity(admitted)  # type: ignore[arg-type]
    assert client.last_playback_server_id is None

    await client.note_playback_activity(admitted)  # type: ignore[arg-type]

    assert written == ["server-A", "server-A"]


async def test_failed_last_playback_write_leaves_admission_unpublished(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed store write during admission keeps the previous connection admitted."""
    store = InMemoryClientPairingStore()
    client = make_sdk_client(client_name="c", roles=[Roles.CONTROLLER], pairing_store=store)
    previous = _FakeConnection(server_id="server-A", activities=[], client=client)
    client._admitted_connection = previous  # type: ignore[assignment]
    incoming = _FakeConnection(server_id="server-B", activities=[Activity.PLAYBACK], client=client)

    async def boom(_server_id: str | None) -> None:
        raise OSError("store unavailable")

    monkeypatch.setattr(store, "set_last_playback_server_id", boom)

    with pytest.raises(OSError, match="store unavailable"):
        await client._admit_connection(incoming)  # type: ignore[arg-type]

    assert client._admitted_connection is previous
    assert not previous.disconnected


async def test_failed_last_playback_read_retries_on_next_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed store read leaves the loader armed so a later admission retries it."""
    store = InMemoryClientPairingStore()
    await store.set_last_playback_server_id("server-Z")
    client = make_sdk_client(client_name="c", roles=[Roles.CONTROLLER], pairing_store=store)
    original = store.get_last_playback_server_id
    reads: list[None] = []

    async def flaky() -> str | None:
        reads.append(None)
        if len(reads) == 1:
            raise OSError("store unavailable")
        return await original()

    monkeypatch.setattr(store, "get_last_playback_server_id", flaky)

    with pytest.raises(OSError, match="store unavailable"):
        await client._ensure_last_playback_loaded()
    assert client.last_playback_server_id is None

    await client._ensure_last_playback_loaded()

    assert client.last_playback_server_id == "server-Z"


# --- on_connection_closed ---


async def test_admitted_close_clears_slot_and_notifies() -> None:
    """Losing the admitted connection clears the slot and fires the disconnect callback."""
    client = _client()
    fired: list[bool] = []
    client.add_disconnect_listener(lambda: fired.append(True))
    admitted = _FakeConnection(client=client)
    client._admitted_connection = admitted  # type: ignore[assignment]

    client.on_connection_closed(admitted)  # type: ignore[arg-type]

    assert client._admitted_connection is None
    assert fired == [True]


async def test_provisional_close_is_silent() -> None:
    """A provisional connection closing is not reported as a client disconnect."""
    client = _client()
    fired: list[bool] = []
    client.add_disconnect_listener(lambda: fired.append(True))
    provisional = _FakeConnection(client=client)
    client._provisional_connections.add(provisional)  # type: ignore[arg-type]

    client.on_connection_closed(provisional)  # type: ignore[arg-type]

    assert provisional not in client._provisional_connections
    assert fired == []


# --- provisional bring-up timeout ---


class _BlockingWebSocket:
    """A websocket whose ``receive()`` never returns, to probe bring-up deadlines."""

    closed = False

    def __init__(self) -> None:
        """Set up the never-set event backing ``receive``."""
        self._never = asyncio.Event()

    async def receive(self) -> object:
        """Block forever (until the awaiting task is cancelled)."""
        await self._never.wait()
        raise AssertionError  # pragma: no cover - the event is never set

    async def close(self) -> None:
        """Mark the socket closed."""
        self.closed = True


async def test_provisional_connection_times_out_during_bringup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An incoming connection drops itself if bring-up never reaches server/activate."""
    monkeypatch.setattr("aiosendspin.client.connection.PROVISIONAL_CONNECTION_TIMEOUT_S", 0.02)
    client = _client()
    connection = SendspinConnection(client)
    assert client._claim_connection_slot(connection)

    async def fake_handshake(ws: object, **_: object) -> None:
        connection._connected = True
        connection._ws = ws  # type: ignore[assignment]

    async def fake_inner() -> None:
        await asyncio.Event().wait()  # never reaches server/activate

    monkeypatch.setattr(connection, "_run_noise_handshake", fake_handshake)
    monkeypatch.setattr(connection, "_run_inner_handshake", fake_inner)

    with pytest.raises(TimeoutError):
        await connection.attach_websocket(_BlockingWebSocket(), expected_server_id=None)  # type: ignore[arg-type]
    assert connection._closed.is_set()


async def test_client_initiated_connection_times_out_during_bringup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A client-initiated dial drops itself if bring-up never reaches server/activate."""
    monkeypatch.setattr("aiosendspin.client.connection.PROVISIONAL_CONNECTION_TIMEOUT_S", 0.02)
    client = _client()
    connection = SendspinConnection(client)
    assert client._claim_connection_slot(connection)

    async def fake_handshake(raw_ws: object, **_: object) -> None:
        connection._connected = True
        connection._ws = raw_ws  # type: ignore[assignment]

    async def fake_inner() -> None:
        await asyncio.Event().wait()  # never reaches server/activate

    monkeypatch.setattr(connection, "_run_noise_handshake", fake_handshake)
    monkeypatch.setattr(connection, "_run_inner_handshake", fake_inner)

    with pytest.raises(TimeoutError):
        await connection.connect(_BlockingWebSocket(), expected_server_id=None)  # type: ignore[arg-type]
    assert connection._closed.is_set()


async def test_client_seeds_last_playback_server_from_store() -> None:
    """The client seeds its last-playback tiebreak from the persisted store value."""
    store = InMemoryClientPairingStore()
    await store.set_last_playback_server_id("server-Z")
    sdk = make_sdk_client(client_name="c", roles=[Roles.CONTROLLER], pairing_store=store)

    await sdk._ensure_last_playback_loaded()

    assert sdk.last_playback_server_id == "server-Z"


# --- open-connection limit ---


class _OpenConnection:
    """Stand-in for an open connection backed by the given pairing records."""

    def __init__(self, *record_psk_ids: str) -> None:
        self.record_psk_ids = set(record_psk_ids)
        self.connected = True


def _client_with_capacity(record_capacity: int) -> SendspinClient:
    return make_sdk_client(
        client_name="Test Client",
        roles=[Roles.CONTROLLER],
        pairing_store=InMemoryClientPairingStore(record_capacity=record_capacity),
    )


async def test_open_connections_stay_below_record_capacity() -> None:
    """Slots run out one below the record capacity and free up when a connection closes."""
    client = _client_with_capacity(5)
    connections = [_OpenConnection() for _ in range(5)]

    claimed = [client._claim_connection_slot(c) for c in connections]  # type: ignore[arg-type]

    assert claimed == [True, True, True, True, False]
    assert not client.has_connection_slot(connections[4])  # type: ignore[arg-type]
    client.on_connection_closed(connections[0])  # type: ignore[arg-type]
    assert client._claim_connection_slot(connections[4])  # type: ignore[arg-type]


async def test_protected_psk_ids_cover_every_open_connection() -> None:
    """Records backing any open connection, provisional or admitted, are protected."""
    client = _client_with_capacity(5)
    provisional = _OpenConnection("psk-a")
    admitted = _OpenConnection("psk-b", "psk-c")
    unpaired = _OpenConnection()
    for connection in (provisional, admitted, unpaired):
        assert client._claim_connection_slot(connection)  # type: ignore[arg-type]
    client._admitted_connection = admitted  # type: ignore[assignment]

    assert client.protected_psk_ids() == {"psk-a", "psk-b", "psk-c"}


async def test_connect_over_the_open_connection_limit_raises() -> None:
    """An explicit connect fails before dialing when no connection slot is free."""
    client = _client_with_capacity(5)
    for _ in range(4):
        assert client._claim_connection_slot(_OpenConnection())  # type: ignore[arg-type]
    try:
        with pytest.raises(RuntimeError, match="open connection limit reached"):
            await client.connect("ws://127.0.0.1:9/sendspin")
        assert not client._provisional_connections
        assert client._session is None
    finally:
        await client.disconnect()


async def test_failed_bring_up_releases_its_connection_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An incoming connection whose bring-up fails gives its slot back."""
    client = _client_with_capacity(5)

    async def fail(
        self: SendspinConnection,  # noqa: ARG001
        ws: object,  # noqa: ARG001
        *,
        expected_server_id: str | None = None,  # noqa: ARG001
    ) -> None:
        raise OSError("handshake failed")

    monkeypatch.setattr(SendspinConnection, "attach_websocket", fail)

    await client.attach_websocket(None)  # type: ignore[arg-type]

    assert not client._open_connections


async def test_incoming_connection_without_a_slot_is_rejected_after_the_handshake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a slot, bring-up sends client/goodbye concurrent_attempt and stops."""
    client = _client()
    connection = SendspinConnection(client)
    goodbyes: list[GoodbyeReason] = []

    async def fake_handshake(ws: object, **_: object) -> None:
        connection._connected = True
        connection._ws = ws  # type: ignore[assignment]

    async def record_goodbye(reason: GoodbyeReason) -> None:
        goodbyes.append(reason)
        await connection.disconnect()

    async def fail_inner() -> None:
        raise AssertionError  # pragma: no cover - the hello exchange must not start

    monkeypatch.setattr(connection, "_run_noise_handshake", fake_handshake)
    monkeypatch.setattr(connection, "goodbye_and_disconnect", record_goodbye)
    monkeypatch.setattr(connection, "_run_inner_handshake", fail_inner)

    with pytest.raises(RuntimeError, match="open connection limit reached"):
        await connection.attach_websocket(_BlockingWebSocket(), expected_server_id=None)  # type: ignore[arg-type]
    assert goodbyes == [GoodbyeReason.CONCURRENT_ATTEMPT]
    assert connection._closed.is_set()


async def test_connection_protects_the_record_its_handshake_resolved() -> None:
    """A connection's long-term record is protected from resolution until it closes."""
    store = InMemoryClientPairingStore(record_capacity=5)
    (record,) = await seed_used_client_records(store, 1)
    client = make_sdk_client(client_name="c", roles=[Roles.CONTROLLER], pairing_store=store)
    connection = SendspinConnection(client)
    assert client._claim_connection_slot(connection)
    assert client.protected_psk_ids() == set()

    await connection._resolve_psk(record.psk_id, PskCategory.LONG_TERM)
    assert client.protected_psk_ids() == {record.psk_id}

    connection._noise_psk = record.as_resolved()
    connection._resolving_psk_id = None
    assert client.protected_psk_ids() == {record.psk_id}

    connection._noise_psk = ResolvedPsk("sentinel", bytes(32), PskCategory.SENTINEL)
    assert client.protected_psk_ids() == set()
    client.on_connection_closed(connection)
    assert not client._open_connections


def _fail_after_noise(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every bring-up complete the Noise handshake and then fail."""

    async def handshake(self: SendspinConnection, ws: object, **_: object) -> None:
        self._connected = True
        self._ws = ws  # type: ignore[assignment]
        raise OSError("record store unavailable")

    monkeypatch.setattr(SendspinConnection, "_run_noise_handshake", handshake)


async def test_incoming_failure_after_the_handshake_releases_its_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bring-up that fails past the Noise handshake closes the socket and frees the slot."""
    client = _client_with_capacity(5)
    _fail_after_noise(monkeypatch)

    for _ in range(5):
        ws = _BlockingWebSocket()
        await client.attach_websocket(ws)  # type: ignore[arg-type]
        assert ws.closed

    assert not client._open_connections
    assert not client._provisional_connections


async def test_connect_failure_after_the_handshake_releases_its_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dial that fails past the Noise handshake closes the socket and frees the slot."""
    sockets: list[_BlockingWebSocket] = []

    async def ws_connect(*_: object, **__: object) -> _BlockingWebSocket:
        sockets.append(_BlockingWebSocket())
        return sockets[-1]

    session = MagicMock()
    session.ws_connect = ws_connect
    client = make_sdk_client(
        client_name="c",
        roles=[Roles.CONTROLLER],
        pairing_store=InMemoryClientPairingStore(record_capacity=5),
        session=session,
    )
    _fail_after_noise(monkeypatch)

    for _ in range(5):
        with pytest.raises(OSError, match="record store unavailable"):
            await client.connect("ws://127.0.0.1:9/sendspin")

    assert all(ws.closed for ws in sockets)
    assert not client._open_connections


async def test_rehandshake_refreshes_the_record_last_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A re-handshake onto a long-term record counts as a use for eviction order."""
    store = InMemoryClientPairingStore(record_capacity=5)
    (record,) = await seed_used_client_records(store, 1)
    client = make_sdk_client(client_name="c", roles=[Roles.CONTROLLER], pairing_store=store)
    connection = SendspinConnection(client)
    connection._ws = MagicMock()
    connection._server_id = "server-0"
    connection._handshake_hash = b"hash"

    async def rehandshake(*_: object, **__: object) -> MagicMock:
        return MagicMock(psk=record.as_resolved(), handshake_hash=b"next")

    monkeypatch.setattr("aiosendspin.client.connection.run_rehandshake_client", rehandshake)

    await connection._rehandshake("hs1")

    refreshed = await store.record_by_psk_id(record.psk_id)
    assert refreshed is not None
    assert refreshed.last_used_at > record.last_used_at


def _bring_up_paired(
    monkeypatch: pytest.MonkeyPatch, record: ClientPairingRecord, sockets: list[_BlockingWebSocket]
) -> None:
    """Make every bring-up succeed as a paired connection on ``record``."""

    async def bring_up(self: SendspinConnection, *_: object, **__: object) -> None:
        sockets.append(_BlockingWebSocket())
        self._ws = sockets[-1]  # type: ignore[assignment]
        self._connected = True
        self._noise_psk = record.as_resolved()

    monkeypatch.setattr(SendspinConnection, "attach_websocket", bring_up)
    monkeypatch.setattr(SendspinConnection, "connect", bring_up)


async def _paired_client(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[SendspinClient, list[_BlockingWebSocket]]:
    store = InMemoryClientPairingStore(record_capacity=5)
    (record,) = await seed_used_client_records(store, 1)
    client = make_sdk_client(
        client_name="c",
        roles=[Roles.CONTROLLER],
        pairing_store=store,
        session=MagicMock(ws_connect=AsyncMock()),
    )
    sockets: list[_BlockingWebSocket] = []
    _bring_up_paired(monkeypatch, record, sockets)
    return client, sockets


async def _await_provisional(client: SendspinClient) -> None:
    async with asyncio.timeout(1):
        while not client._provisional_connections:  # noqa: ASYNC110
            await asyncio.sleep(0)


@pytest.mark.parametrize("entry", ["attach", "connect"])
async def test_shutdown_during_admission_admits_nothing(
    monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    """A connection waiting for admission is closed by shutdown and never admitted."""
    client, sockets = await _paired_client(monkeypatch)
    with pytest.MonkeyPatch.context() as m:
        m.setattr(client, "_owns_session", False)
        async with client._admission_lock:
            if entry == "attach":
                task = asyncio.create_task(client.attach_websocket(MagicMock()))
            else:
                task = asyncio.create_task(client.connect("ws://127.0.0.1:9/sendspin"))
            await _await_provisional(client)
            await client.disconnect()
        if entry == "attach":
            await task
        else:
            with pytest.raises(RuntimeError, match="closed before admission"):
                await task

    assert client._admitted_connection is None
    assert not client._provisional_connections
    assert not client._open_connections
    assert all(ws.closed for ws in sockets)


@pytest.mark.parametrize("entry", ["attach", "connect"])
async def test_start_failure_after_admission_releases_the_slot(
    monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    """An admitted connection that fails to start is closed and gives its slot back."""
    client, sockets = await _paired_client(monkeypatch)

    async def fail_start(self: SendspinConnection) -> None:  # noqa: ARG001
        raise OSError("state send failed")

    monkeypatch.setattr(SendspinConnection, "start", fail_start)

    if entry == "attach":
        await client.attach_websocket(MagicMock())
    else:
        with pytest.raises(OSError, match="state send failed"):
            await client.connect("ws://127.0.0.1:9/sendspin")

    assert client._admitted_connection is None
    assert not client._open_connections
    assert sockets[0].closed


async def test_over_limit_handshake_protects_the_record_it_resolved() -> None:
    """A connection over the limit still protects the record its handshake resolved."""
    store = InMemoryClientPairingStore(record_capacity=5)
    (record,) = await seed_used_client_records(store, 1)
    client = make_sdk_client(client_name="c", roles=[Roles.CONTROLLER], pairing_store=store)
    connection = SendspinConnection(client)
    client._provisional_connections.add(connection)

    await connection._resolve_psk(record.psk_id, PskCategory.LONG_TERM)

    assert not client.has_connection_slot(connection)
    assert client.protected_psk_ids() == {record.psk_id}


async def test_handshake_failure_closes_the_raw_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any handshake failure before the session is up closes the raw socket."""
    client = _client_with_capacity(5)

    async def handshake(self: SendspinConnection, *_: object, **__: object) -> None:  # noqa: ARG001
        raise OSError("connection reset")

    monkeypatch.setattr(SendspinConnection, "_run_noise_handshake", handshake)
    ws = _BlockingWebSocket()

    await client.attach_websocket(ws)  # type: ignore[arg-type]

    assert ws.closed
    assert not client._open_connections


async def test_closing_a_connection_drops_the_record_a_repairing_replaced() -> None:
    """A server's prior record kept for an open connection goes once that connection closes."""
    store = InMemoryClientPairingStore(record_capacity=5)
    (old,) = await seed_used_client_records(store, 1)
    client = make_sdk_client(client_name="c", roles=[Roles.CONTROLLER], pairing_store=store)
    connection = SendspinConnection(client)
    assert client._claim_connection_slot(connection)
    connection._resolving_psk_id = old.psk_id
    psk = generate_psk()
    new = ClientPairingRecord(psk_id=psk_id_for(psk), psk=psk, server_id=old.server_id)

    await store.replace_record_for_server_id(new, protected=client.protected_psk_ids())
    assert await store.record_by_psk_id(old.psk_id) is not None
    assert await store.record_by_server_id("server-0") == new

    connection._resolving_psk_id = None
    client.on_connection_closed(connection)
    await asyncio.sleep(0)

    assert await store.record_by_psk_id(old.psk_id) is None
    assert await store.record_by_psk_id(new.psk_id) == new
