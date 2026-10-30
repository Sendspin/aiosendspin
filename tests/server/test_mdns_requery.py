"""mDNS discovery keeps querying while no Sendspin client is known."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiosendspin.noise.keys import Identity
from aiosendspin.noise.trust_store import InMemoryServerPairingStore
from aiosendspin.server.server import SendspinServer

_SERVICE = "device._sendspin._tcp.local."


class _FakeAsyncServiceBrowser:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.types = {"_sendspin._tcp.local."}
        self.query_scheduler = MagicMock()
        self.async_cancel = AsyncMock()


@pytest.fixture
def browsing_server(monkeypatch: pytest.MonkeyPatch, mock_loop: MagicMock) -> SendspinServer:
    """Server on a mock loop with a fake zeroconf and service browser."""
    monkeypatch.setattr("aiosendspin.server.server.AsyncServiceBrowser", _FakeAsyncServiceBrowser)
    client_session = MagicMock()
    client_session.closed = True
    server = SendspinServer(
        loop=mock_loop,
        identity=Identity.generate(),
        server_name="server",
        client_session=client_session,
        pairing_store=InMemoryServerPairingStore(),
    )
    zc = MagicMock()
    zc.async_close = AsyncMock()
    server._zc = zc  # noqa: SLF001
    return server


def _fire_requery(mock_loop: MagicMock) -> float:
    """Run the pending re-query timer and return the delay it was scheduled with."""
    delay, callback, *args = mock_loop.call_later.call_args.args
    callback(*args)
    return delay


async def test_requery_backs_off_to_rfc_cap_while_empty(
    browsing_server: SendspinServer, mock_loop: MagicMock
) -> None:
    """Each re-query doubles the wait from 30 s and holds at 60 minutes."""
    await browsing_server._start_mdns_discovery()  # noqa: SLF001
    browser = browsing_server._mdns_browser  # noqa: SLF001
    assert isinstance(browser, _FakeAsyncServiceBrowser)

    delays = [_fire_requery(mock_loop) for _ in range(9)]

    assert delays == [30.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 3600.0, 3600.0]
    send = browser.query_scheduler.async_send_ready_queries
    assert send.call_count == 9
    assert send.call_args.kwargs["first_request"] is False
    assert send.call_args.kwargs["ready_types"] == browser.types


async def test_no_requery_while_known_and_restart_when_last_removed(
    browsing_server: SendspinServer, mock_loop: MagicMock
) -> None:
    """A known client stops the re-queries, and losing the last one restarts them at 30 s."""
    await browsing_server._start_mdns_discovery()  # noqa: SLF001
    browser = browsing_server._mdns_browser  # noqa: SLF001
    assert isinstance(browser, _FakeAsyncServiceBrowser)
    _fire_requery(mock_loop)
    browsing_server._mdns_client_urls[_SERVICE] = "ws://192.0.2.1:8928/sendspin"  # noqa: SLF001
    scheduled = mock_loop.call_later.call_count
    _fire_requery(mock_loop)

    browser.query_scheduler.async_send_ready_queries.assert_called_once()
    assert mock_loop.call_later.call_count == scheduled

    browsing_server._handle_service_removed(_SERVICE)  # noqa: SLF001

    assert mock_loop.call_later.call_args.args[0] == 30.0


async def test_stop_cancels_pending_requery(
    browsing_server: SendspinServer, mock_loop: MagicMock
) -> None:
    """Stopping the server cancels the pending re-query."""
    await browsing_server._start_mdns_discovery()  # noqa: SLF001

    await browsing_server.stop_server()

    mock_loop.call_later.return_value.cancel.assert_called_once()
