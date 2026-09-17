"""Tests for initial server-initiated connection behavior."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientConnectionError, ClientWebSocketResponse

from aiosendspin.models.types import GoodbyeReason, PairMethod
from aiosendspin.noise.keys import Identity, generate_psk
from aiosendspin.noise.pairing import PairingAttempt
from aiosendspin.noise.trust_store import InMemoryServerPairingStore
from aiosendspin.server.connection import SendspinConnection
from aiosendspin.server.server import SendspinServer


class _FailingInitialConnectSession:
    """Client session whose ws_connect fails immediately."""

    def __init__(self) -> None:
        self.closed = False
        self.calls = 0

    def ws_connect(self, *_args: object, **_kwargs: object) -> object:
        """Raise an initial connection error."""
        self.calls += 1
        raise ClientConnectionError("boom")

    async def close(self) -> None:
        """Close session."""
        self.closed = True


class _SuccessfulConnectContext:
    """Async context manager returning a websocket stub."""

    async def __aenter__(self) -> object:
        return MagicMock()

    async def __aexit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        return None


class _SuccessfulInitialConnectSession:
    """Client session whose first connection succeeds."""

    def __init__(self) -> None:
        self.closed = True
        self.calls = 0

    def ws_connect(self, *_args: object, **_kwargs: object) -> _SuccessfulConnectContext:
        """Return a successful websocket context manager."""
        self.calls += 1
        return _SuccessfulConnectContext()

    async def close(self) -> None:
        """Close session."""
        self.closed = True


class _PersistentSuccessfulSession:
    """Client session with successful connects and open lifecycle."""

    def __init__(self) -> None:
        self.closed = False
        self.calls = 0

    def ws_connect(self, *_args: object, **_kwargs: object) -> _SuccessfulConnectContext:
        self.calls += 1
        return _SuccessfulConnectContext()

    async def close(self) -> None:
        self.closed = True


class _FailOnceThenConnectSession:
    """Client session whose first connection fails and second succeeds."""

    def __init__(self) -> None:
        self.closed = True
        self.calls = 0

    def ws_connect(self, *_args: object, **_kwargs: object) -> _SuccessfulConnectContext:
        self.calls += 1
        if self.calls == 1:
            raise ClientConnectionError("offline")
        return _SuccessfulConnectContext()

    async def close(self) -> None:
        self.closed = True


class _FakeAsyncServiceInfo:
    """Configurable AsyncServiceInfo test double."""

    addresses: ClassVar[list[str]] = []
    port: ClassVar[int | None] = None
    properties: ClassVar[dict[bytes, bytes] | None] = None

    def __init__(self, _service_type: str, _name: str) -> None:
        pass

    def load_from_cache(self, _zeroconf: object) -> bool:
        return True

    async def async_request(self, _zeroconf: object, _timeout_ms: int) -> None:
        return

    def parsed_addresses(self) -> list[str]:
        return list(self.addresses)


def _make_server(client_session: object) -> SendspinServer:
    """Create server with injected client session test double."""
    loop = asyncio.get_running_loop()
    return SendspinServer(
        loop=loop,
        identity=Identity.generate(),
        server_name="server",
        client_session=client_session,
        pairing_store=InMemoryServerPairingStore(),
    )


async def _wait_for_connection_task_cleanup(server: SendspinServer, url: str) -> None:
    """Wait until a connection task is removed from bookkeeping."""
    for _ in range(50):
        task = server._connection_tasks.get(url)  # noqa: SLF001
        if task is None:
            return
        if task.done():
            await asyncio.sleep(0)
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_connect_to_client_and_wait_raises_on_initial_connection_failure() -> None:
    """Initial connection failure should propagate to waiting caller."""
    session = _FailingInitialConnectSession()
    server = _make_server(session)
    url = "ws://127.0.0.1:9999/sendspin"

    with pytest.raises(ClientConnectionError):
        await server.connect_to_client_and_wait(url)

    await _wait_for_connection_task_cleanup(server, url)
    assert session.calls == 1
    assert url not in server._connection_tasks  # noqa: SLF001
    assert url not in server._retry_events  # noqa: SLF001


@pytest.mark.asyncio
async def test_connect_to_client_stops_after_initial_failure_without_retry() -> None:
    """Background connection should stop when first attempt fails."""
    session = _FailingInitialConnectSession()
    server = _make_server(session)
    url = "ws://127.0.0.1:9999/sendspin"

    server.connect_to_client(url)
    await _wait_for_connection_task_cleanup(server, url)

    assert session.calls == 1
    assert url not in server._connection_tasks  # noqa: SLF001


@pytest.mark.asyncio
async def test_connect_to_client_and_wait_can_retry_initial_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opt-in initial retry should wait until an offline client comes online."""
    session = _FailOnceThenConnectSession()
    server = _make_server(session)
    url = "ws://127.0.0.1:9999/sendspin"

    class _FakeConnection:
        """Connection double used to bypass full websocket lifecycle."""

        closing = False
        goodbye_reason = None
        should_retry_server_initiated_connection = False

        def __init__(
            self,
            _server: SendspinServer,
            *,
            wsock_client: object,  # noqa: ARG002
            url: str | None = None,  # noqa: ARG002
            expected_client_id: str | None = None,  # noqa: ARG002
            pairing_attempt: object | None = None,  # noqa: ARG002
        ) -> None:
            return

        async def handle_client(self) -> None:
            return

    monkeypatch.setattr("aiosendspin.server.server.SendspinConnection", _FakeConnection)

    wait_task = asyncio.create_task(
        server.connect_to_client_and_wait(url, retry_initial_connection=True)
    )
    for _ in range(20):
        if session.calls == 1 and url in server._retry_events:  # noqa: SLF001
            break
        await asyncio.sleep(0)

    assert not wait_task.done()
    server._retry_events[url].set()  # noqa: SLF001

    await wait_task

    assert session.calls == 2


@pytest.mark.asyncio
async def test_connect_to_client_and_wait_raises_when_initial_retry_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bounded initial retry should not leave waiters unresolved."""
    session = _FailingInitialConnectSession()
    server = _make_server(session)
    url = "ws://127.0.0.1:9999/sendspin"
    monkeypatch.setattr("aiosendspin.server.server.MAX_RECONNECT_BACKOFF_S", 1.0)

    with pytest.raises(TimeoutError, match="Initial connection did not succeed"):
        await server.connect_to_client_and_wait(url, retry_initial_connection=True)

    assert session.calls == 1


@pytest.mark.asyncio
async def test_connect_to_client_and_wait_returns_on_initial_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Waiting connect call should return once first connection succeeds."""
    session = _SuccessfulInitialConnectSession()
    server = _make_server(session)
    url = "ws://127.0.0.1:9999/sendspin"

    class _FakeConnection:
        """Connection double used to bypass full websocket lifecycle."""

        closing = False

        def __init__(
            self,
            _server: SendspinServer,
            *,
            wsock_client: object,
            url: str | None = None,  # noqa: ARG002
            expected_client_id: str | None = None,  # noqa: ARG002
            pairing_attempt: object | None = None,  # noqa: ARG002
        ) -> None:
            self._wsock_client = wsock_client

        async def handle_client(self) -> None:
            return

    monkeypatch.setattr("aiosendspin.server.server.SendspinConnection", _FakeConnection)

    await server.connect_to_client_and_wait(url)

    assert session.calls == 1


@pytest.mark.asyncio
async def test_server_initiated_stops_retrying_on_another_server_goodbye(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Server-initiated loop must stop when client disconnects with ANOTHER_SERVER."""
    session = _PersistentSuccessfulSession()
    server = _make_server(session)
    url = "ws://127.0.0.1:9999/sendspin"

    class _FakeConnection:
        closing = False

        def __init__(
            self,
            _server: SendspinServer,
            *,
            wsock_client: object,  # noqa: ARG002
            url: str | None = None,  # noqa: ARG002
            expected_client_id: str | None = None,  # noqa: ARG002
            pairing_attempt: object | None = None,  # noqa: ARG002
        ) -> None:
            self.goodbye_reason = GoodbyeReason.ANOTHER_SERVER
            self.should_retry_server_initiated_connection = False

        async def handle_client(self) -> None:
            return

    monkeypatch.setattr("aiosendspin.server.server.SendspinConnection", _FakeConnection)

    server.connect_to_client(url)

    for _ in range(40):
        if url not in server._connection_tasks:  # noqa: SLF001
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("Connection task was not cleaned up after ANOTHER_SERVER goodbye")

    assert session.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("monotonic_values", "expected_sleeps"),
    [
        # Repeated unstable sessions should grow backoff: 1s -> 2s -> 4s.
        ([0.0, 5.0, 6.0, 9.0, 10.0, 13.0, 14.0, 16.0], [1.0, 2.0, 4.0]),
        # Unstable then stable should reset, then unstable grows again.
        ([0.0, 5.0, 6.0, 18.0, 19.0, 23.0, 24.0, 27.0], [1.0, 1.0, 2.0]),
        # Repeated stable sessions should keep reconnect delay at 1s.
        ([0.0, 12.0, 13.0, 24.0, 25.0, 35.0, 36.0, 49.0], [1.0, 1.0, 1.0]),
    ],
)
async def test_server_initiated_backoff_resets_only_after_stable_session(
    monkeypatch: pytest.MonkeyPatch,
    monotonic_values: list[float],
    expected_sleeps: list[float],
) -> None:
    """Reconnect backoff should reset only after sufficiently long sessions."""
    session = _PersistentSuccessfulSession()
    server = _make_server(session)
    url = "ws://127.0.0.1:9999/sendspin"
    max_attempts = len(monotonic_values) // 2
    attempts = 0
    sleep_calls: list[float] = []
    monotonic_iter = iter(monotonic_values)

    class _FakeConnection:
        closing = False

        def __init__(
            self,
            _server: SendspinServer,
            *,
            wsock_client: object,  # noqa: ARG002
            url: str | None = None,  # noqa: ARG002
            expected_client_id: str | None = None,  # noqa: ARG002
            pairing_attempt: object | None = None,  # noqa: ARG002
        ) -> None:
            nonlocal attempts
            attempts += 1
            self.should_retry_server_initiated_connection = attempts < max_attempts
            self.goodbye_reason = None

        async def handle_client(self) -> None:
            return

    async def _fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    def _fake_monotonic() -> float:
        return next(monotonic_iter)

    monkeypatch.setattr("aiosendspin.server.server.SendspinConnection", _FakeConnection)
    monkeypatch.setattr(
        "aiosendspin.server.server.time",
        SimpleNamespace(monotonic=_fake_monotonic),
    )
    monkeypatch.setattr("aiosendspin.server.server.asyncio.sleep", _fake_sleep)

    await server._handle_client_connection(url)  # noqa: SLF001

    assert sleep_calls == expected_sleeps


@pytest.mark.asyncio
async def test_retry_indefinitely_keeps_reconnecting_after_backoff_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistent address connections should cap backoff without giving up."""
    session = _PersistentSuccessfulSession()
    server = _make_server(session)
    url = "ws://127.0.0.1:9999/sendspin"
    attempts = 0
    sleep_calls: list[float] = []
    monotonic_iter = iter([0.0, 0.1, 0.2, 0.3])

    class _FakeConnection:
        closing = False
        goodbye_reason = None

        def __init__(
            self,
            _server: SendspinServer,
            *,
            wsock_client: object,  # noqa: ARG002
            url: str | None = None,  # noqa: ARG002
            expected_client_id: str | None = None,  # noqa: ARG002
            pairing_attempt: object | None = None,  # noqa: ARG002
        ) -> None:
            nonlocal attempts
            attempts += 1
            self.should_retry_server_initiated_connection = attempts < 2

        async def handle_client(self) -> None:
            return

    async def _fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    def _fake_monotonic() -> float:
        return next(monotonic_iter)

    monkeypatch.setattr("aiosendspin.server.server.SendspinConnection", _FakeConnection)
    monkeypatch.setattr("aiosendspin.server.server.MAX_RECONNECT_BACKOFF_S", 1.0)
    monkeypatch.setattr(
        "aiosendspin.server.server.time",
        SimpleNamespace(monotonic=_fake_monotonic),
    )
    monkeypatch.setattr("aiosendspin.server.server.asyncio.sleep", _fake_sleep)
    server._set_connection_options(  # noqa: SLF001
        url,
        retry_initial_connection=False,
        retry_indefinitely=True,
    )

    await server._handle_client_connection(url)  # noqa: SLF001

    assert session.calls == 2
    assert sleep_calls == [1.0]


def test_should_retry_false_when_closing() -> None:
    """should_retry_server_initiated_connection must be False after disconnect(retry=False)."""
    conn = SendspinConnection(MagicMock(), wsock_client=MagicMock(spec=ClientWebSocketResponse))
    assert conn.should_retry_server_initiated_connection is True
    conn._closing = True  # noqa: SLF001
    assert conn.should_retry_server_initiated_connection is False


@pytest.mark.asyncio
async def test_mdns_removal_cleans_retained_another_server_client() -> None:
    """Removing mDNS entry should clean retained ANOTHER_SERVER clients."""
    session = _PersistentSuccessfulSession()
    server = _make_server(session)
    url = "ws://127.0.0.1:9999/sendspin"

    retained_client = MagicMock()
    retained_client.is_connected = False
    retained_client.cleanup_on_mdns_removal = True

    server._clients = {"client-1": retained_client}  # noqa: SLF001
    server._client_urls = {"client-1": url}  # noqa: SLF001
    server._mdns_client_urls = {"service._sendspin._tcp.local.": url}  # noqa: SLF001
    server.remove_client = AsyncMock()  # type: ignore[method-assign]

    server._handle_service_removed("service._sendspin._tcp.local.")  # noqa: SLF001
    await asyncio.sleep(0)

    server.remove_client.assert_awaited_once_with("client-1")
    assert "client-1" not in server._client_urls  # noqa: SLF001


@pytest.mark.asyncio
async def test_register_client_url_disconnects_stale_url() -> None:
    """Registering a new URL for a client should cancel the connection to the old URL."""
    session = _PersistentSuccessfulSession()
    server = _make_server(session)
    old_url = "ws://127.0.0.1:9998/sendspin"
    new_url = "ws://127.0.0.1:9999/sendspin"

    server.connect_to_client(old_url)
    server.register_client_url("client-1", old_url)
    old_task = server._connection_tasks.get(old_url)  # noqa: SLF001
    assert old_task is not None

    server.register_client_url("client-1", new_url)
    assert old_url not in server._connection_tasks  # noqa: SLF001
    assert old_task.cancelling() > 0


@pytest.mark.asyncio
async def test_register_client_url_keeps_connection_when_url_unchanged() -> None:
    """Re-registering the same URL should not cancel the existing connection."""
    session = _PersistentSuccessfulSession()
    server = _make_server(session)
    url = "ws://127.0.0.1:9999/sendspin"

    server.connect_to_client(url)
    task = server._connection_tasks.get(url)  # noqa: SLF001
    assert task is not None

    server.register_client_url("client-1", url)
    assert server._connection_tasks.get(url) is task  # noqa: SLF001
    assert not task.cancelled()


@pytest.mark.asyncio
async def test_mdns_removal_keeps_non_retained_client() -> None:
    """MDNS removal should not remove clients that are not marked for mDNS cleanup."""
    session = _PersistentSuccessfulSession()
    server = _make_server(session)
    url = "ws://127.0.0.1:9999/sendspin"

    retained_client = MagicMock()
    retained_client.is_connected = False
    retained_client.cleanup_on_mdns_removal = False

    server._clients = {"client-1": retained_client}  # noqa: SLF001
    server._client_urls = {"client-1": url}  # noqa: SLF001
    server._mdns_client_urls = {"service._sendspin._tcp.local.": url}  # noqa: SLF001
    server.remove_client = AsyncMock()  # type: ignore[method-assign]

    server._handle_service_removed("service._sendspin._tcp.local.")  # noqa: SLF001
    await asyncio.sleep(0)

    server.remove_client.assert_not_awaited()


@pytest.mark.asyncio
async def test_mdns_removal_keeps_connection_task_for_persistent_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A persistent client must keep its retry task alive after mDNS withdrawal.

    Without this guarantee, a transient mDNS drop (network blip while the
    device is still alive) silently cancels the reconnect loop and the device
    stays unavailable until MA restart.
    """
    session = _PersistentSuccessfulSession()
    server = _make_server(session)
    url = "ws://127.0.0.1:9999/sendspin"

    class _HangingConnection:
        def __init__(
            self,
            _server: SendspinServer,
            *,
            wsock_client: object,  # noqa: ARG002
            url: str | None = None,  # noqa: ARG002
            expected_client_id: str | None = None,  # noqa: ARG002
            pairing_attempt: object | None = None,  # noqa: ARG002
        ) -> None:
            self.goodbye_reason = None
            self.should_retry_server_initiated_connection = True

        async def handle_client(self) -> None:
            await asyncio.Event().wait()

    monkeypatch.setattr("aiosendspin.server.server.SendspinConnection", _HangingConnection)

    persistent_client = MagicMock()
    persistent_client.is_connected = False
    persistent_client.cleanup_on_mdns_removal = False

    server.connect_to_client(url)
    task = server._connection_tasks.get(url)  # noqa: SLF001
    assert task is not None

    server._clients = {"client-1": persistent_client}  # noqa: SLF001
    server._client_urls = {"client-1": url}  # noqa: SLF001
    server._mdns_client_urls = {"service._sendspin._tcp.local.": url}  # noqa: SLF001

    server._handle_service_removed("service._sendspin._tcp.local.")  # noqa: SLF001
    await asyncio.sleep(0)

    assert server._connection_tasks.get(url) is task  # noqa: SLF001
    assert task.cancelling() == 0
    assert server._client_urls.get("client-1") == url  # noqa: SLF001


@pytest.mark.asyncio
async def test_mdns_removal_cancels_connection_when_no_persistent_client() -> None:
    """With no persistent client (or only an ANOTHER_SERVER retainer), tear the retry task down."""
    session = _PersistentSuccessfulSession()
    server = _make_server(session)
    url = "ws://127.0.0.1:9999/sendspin"

    retained_client = MagicMock()
    retained_client.is_connected = False
    retained_client.cleanup_on_mdns_removal = True

    server.connect_to_client(url)
    task = server._connection_tasks.get(url)  # noqa: SLF001
    assert task is not None

    server._clients = {"client-1": retained_client}  # noqa: SLF001
    server._client_urls = {"client-1": url}  # noqa: SLF001
    server._mdns_client_urls = {"service._sendspin._tcp.local.": url}  # noqa: SLF001
    server.remove_client = AsyncMock()  # type: ignore[method-assign]

    server._handle_service_removed("service._sendspin._tcp.local.")  # noqa: SLF001
    await asyncio.sleep(0)

    assert url not in server._connection_tasks  # noqa: SLF001
    assert task.cancelling() > 0
    assert "client-1" not in server._client_urls  # noqa: SLF001


@pytest.mark.parametrize(
    ("addresses", "port", "path", "has_task", "expected_url", "expect_reconnect"),
    [
        # Address reorder, old IP still advertised, task alive -> keep existing.
        (["10.0.0.3", "10.0.0.2"], 9999, "/sendspin", True, "ws://10.0.0.2:9999/sendspin", False),
        # Same IP, port/path changed -> reconnect.
        (["10.0.0.2"], 10000, "/other", True, "ws://10.0.0.2:10000/other", True),
        # Old IP gone from advertised set -> reconnect.
        (["10.0.0.5"], 9999, "/sendspin", True, "ws://10.0.0.5:9999/sendspin", True),
        # URL changed but no active task -> reconnect.
        (["10.0.0.3"], 9999, "/sendspin", False, "ws://10.0.0.3:9999/sendspin", True),
    ],
)
@pytest.mark.asyncio
async def test_mdns_update_reconnect_decision(
    monkeypatch: pytest.MonkeyPatch,
    addresses: list[str],
    port: int,
    path: str,
    has_task: bool,  # noqa: FBT001
    expected_url: str,
    expect_reconnect: bool,  # noqa: FBT001
) -> None:
    """Reconnect only when old endpoint no longer matches an active connection."""
    server = _make_server(_PersistentSuccessfulSession())
    service_name = "service._sendspin._tcp.local."
    old_url = "ws://10.0.0.2:9999/sendspin"

    _FakeAsyncServiceInfo.addresses = addresses
    _FakeAsyncServiceInfo.port = port
    _FakeAsyncServiceInfo.properties = {b"path": path.encode()}
    monkeypatch.setattr("aiosendspin.server.server.AsyncServiceInfo", _FakeAsyncServiceInfo)

    server._mdns_client_urls[service_name] = old_url  # noqa: SLF001
    connection_task: asyncio.Task[None] | None = None
    if has_task:
        connection_task = asyncio.create_task(asyncio.sleep(3600))
        server._connection_tasks[old_url] = connection_task  # noqa: SLF001
    server.connect_to_client = MagicMock()  # type: ignore[method-assign]

    try:
        await server._handle_service_added(MagicMock(), "_sendspin._tcp.local.", service_name)  # noqa: SLF001
    finally:
        if connection_task is not None:
            connection_task.cancel()
            with suppress(asyncio.CancelledError):
                await connection_task

    assert server._mdns_client_urls[service_name] == expected_url  # noqa: SLF001
    if expect_reconnect:
        server.connect_to_client.assert_called_once_with(expected_url)
    else:
        server.connect_to_client.assert_not_called()


@pytest.mark.asyncio
async def test_pairing_attempt_queued_for_existing_dial_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pairing_attempt passed while a dial task exists rides that task's next dial."""
    session = _PersistentSuccessfulSession()
    server = _make_server(session)
    url = "ws://127.0.0.1:9999/sendspin"
    attempt = PairingAttempt(
        method=PairMethod.PAIRING_PSK, pairing_psk=generate_psk(), client_id="client-A"
    )
    received: list[object] = []

    class _FakeConnection:
        """Connection double recording the pairing_attempt of each dial."""

        closing = False
        goodbye_reason = None

        def __init__(
            self,
            _server: SendspinServer,
            *,
            wsock_client: object,  # noqa: ARG002
            url: str | None = None,  # noqa: ARG002
            expected_client_id: str | None = None,  # noqa: ARG002
            pairing_attempt: object | None = None,
        ) -> None:
            received.append(pairing_attempt)

        async def handle_client(self) -> None:
            return

        @property
        def should_retry_server_initiated_connection(self) -> bool:
            return len(received) < 2

    monkeypatch.setattr("aiosendspin.server.server.SendspinConnection", _FakeConnection)

    server.connect_to_client(url)
    for _ in range(50):
        if received:
            break
        await asyncio.sleep(0)
    assert received == [None]

    server.connect_to_client(url, pairing_attempt=attempt)
    await _wait_for_connection_task_cleanup(server, url)

    assert received == [None, attempt]
