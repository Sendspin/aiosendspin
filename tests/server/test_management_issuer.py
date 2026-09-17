"""Unit tests for the server-side management request/reply correlation."""
# DEPRECATED(spec-pr-183): remove in aiosendspin <version>

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

from aiosendspin.models.management import (
    ManagementListRecordsMessage,
    ManagementResultPayload,
)
from aiosendspin.models.types import ManagementResult
from aiosendspin.noise.keys import generate_psk, psk_id_for
from aiosendspin.noise.trust_store import PskCategory, ResolvedPsk
from aiosendspin.server.connection import SendspinConnection
from tests.conftest import recorded_deprecations


class _OtherResultPayload(ManagementResultPayload):
    """A distinct reply subtype used to exercise the wrong-reply-type guard."""


def _bare_connection() -> SendspinConnection:
    """Build a connection with only the attributes _resolve_management touches."""
    conn = SendspinConnection.__new__(SendspinConnection)
    conn._logger = logging.getLogger("test")  # noqa: SLF001
    conn._management_waiter = None  # noqa: SLF001
    conn._client = None  # noqa: SLF001
    conn._hello_description = ""  # noqa: SLF001
    conn._server = SimpleNamespace(allow_noncompliant_clients=True)  # type: ignore[assignment]  # noqa: SLF001
    return conn


def test_resolve_management_delivers_to_active_waiter() -> None:
    """A reply for an active waiter sets its result and drains the slot."""
    conn = _bare_connection()
    loop = asyncio.new_event_loop()
    try:
        waiter: asyncio.Future[ManagementResultPayload] = loop.create_future()
        conn._management_waiter = waiter  # noqa: SLF001
        payload = ManagementResultPayload(result=ManagementResult.OK)
        conn._resolve_management(payload)  # noqa: SLF001
        assert waiter.result() is payload
        assert conn._management_waiter is None  # noqa: SLF001
    finally:
        loop.close()


def test_resolve_management_drains_abandoned_waiter() -> None:
    """A late reply for an abandoned (cancelled) waiter still drains the slot."""
    conn = _bare_connection()
    loop = asyncio.new_event_loop()
    try:
        waiter: asyncio.Future[ManagementResultPayload] = loop.create_future()
        waiter.cancel()  # caller gave up on the wait
        conn._management_waiter = waiter  # noqa: SLF001
        # No InvalidStateError despite the cancelled future, and the slot is freed.
        conn._resolve_management(ManagementResultPayload(result=ManagementResult.OK))  # noqa: SLF001
        assert conn._management_waiter is None  # noqa: SLF001
    finally:
        loop.close()


def test_resolve_management_ignores_unsolicited_reply() -> None:
    """A reply with no outstanding waiter is ignored without error."""
    conn = _bare_connection()
    conn._resolve_management(ManagementResultPayload(result=ManagementResult.OK))  # noqa: SLF001
    assert conn._management_waiter is None  # noqa: SLF001


async def test_management_request_rejects_wrong_reply_type() -> None:
    """A reply of the wrong type fails the request rather than asserting opaquely."""
    conn = _bare_connection()
    conn._transport = object()  # type: ignore[assignment]  # noqa: SLF001 - non-None sentinel
    conn._disconnecting = False  # noqa: SLF001
    conn._management_active = True  # noqa: SLF001
    conn._noise_psk = _long_term_psk()  # noqa: SLF001

    def _send(_message: object) -> None:
        pass

    conn.send_priority_message = _send  # type: ignore[method-assign]
    task = asyncio.ensure_future(
        conn._management_request(  # noqa: SLF001
            ManagementListRecordsMessage(), _OtherResultPayload
        )
    )
    await asyncio.sleep(0)  # let the request register its waiter
    # The client answers with the wrong reply type.
    conn._resolve_management(ManagementResultPayload(result=ManagementResult.OK))  # noqa: SLF001
    with pytest.raises(RuntimeError, match="expected a"):
        await task


def _long_term_psk() -> ResolvedPsk:
    psk = generate_psk()
    return ResolvedPsk(psk_id_for(psk), psk, PskCategory.LONG_TERM)


@pytest.mark.parametrize(
    ("active", "category"),
    [(False, PskCategory.LONG_TERM), (True, PskCategory.SENTINEL)],
)
async def test_management_request_refused_unless_enabled(
    *, active: bool, category: PskCategory
) -> None:
    """No management/* request is sent on a connection the embedder has not enabled."""
    conn = _bare_connection()
    conn._transport = object()  # type: ignore[assignment]  # noqa: SLF001 - non-None sentinel
    conn._disconnecting = False  # noqa: SLF001
    conn._management_active = active  # noqa: SLF001
    psk = generate_psk()
    conn._noise_psk = ResolvedPsk(psk_id_for(psk), psk, category)  # noqa: SLF001
    sent: list[object] = []
    conn.send_priority_message = sent.append  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="management is not enabled"):
        await conn._management_request(  # noqa: SLF001
            ManagementListRecordsMessage(), ManagementResultPayload
        )
    assert sent == []


def _conn_with_psk(category: PskCategory | None) -> SendspinConnection:
    """Build a connection with just the PSK/management attributes the gate touches."""
    conn = SendspinConnection.__new__(SendspinConnection)
    if category is None:
        conn._noise_psk = None  # noqa: SLF001
    else:
        psk = generate_psk()
        conn._noise_psk = ResolvedPsk(psk_id_for(psk), psk, category)  # noqa: SLF001
    conn._management_active = False  # noqa: SLF001
    # Keep _pairing_in_progress False.
    conn._in_pairing = False  # noqa: SLF001
    conn._pairing_message_queue = None  # noqa: SLF001
    conn._declared_activities = None  # noqa: SLF001 — _refresh_activities then no-ops
    return conn


@pytest.mark.parametrize(
    ("category", "capable"),
    [
        (PskCategory.LONG_TERM, True),
        (PskCategory.SENTINEL, False),
        (PskCategory.PAIRING, False),
        (None, False),
    ],
)
def test_management_capable_requires_long_term(
    category: PskCategory | None,
    capable: bool,  # noqa: FBT001
) -> None:
    """A connection may carry management only on a long-term Sendspin PSK."""
    assert _conn_with_psk(category)._management_capable is capable  # noqa: SLF001


@pytest.mark.parametrize("category", [PskCategory.SENTINEL, PskCategory.PAIRING, None])
def test_enable_management_rejects_non_long_term(category: PskCategory | None) -> None:
    """enable_management refuses a connection that is not paired (long-term PSK)."""
    conn = _conn_with_psk(category)
    with pytest.raises(RuntimeError, match="paired"):
        conn.enable_management()
    assert conn._management_active is False  # noqa: SLF001


def test_connection_management_toggles_warn_once_each(caplog: pytest.LogCaptureFixture) -> None:
    """The connection's enable/disable each warn once, however often they are called."""
    conn = _conn_with_psk(PskCategory.LONG_TERM)
    with caplog.at_level(logging.WARNING), recorded_deprecations() as deprecations:
        for _ in range(2):
            conn.enable_management()
            conn.disable_management()

    apis = ["SendspinConnection.enable_management", "SendspinConnection.disable_management"]
    assert [m.split(" is deprecated")[0] for m in deprecations] == apis
    logged = [r.message for r in caplog.records if r.name == "aiosendspin.util"]
    assert [m.split(" is deprecated")[0] for m in logged] == apis


def test_enable_management_allows_long_term() -> None:
    """enable_management activates management on a long-term (paired) connection."""
    conn = _conn_with_psk(PskCategory.LONG_TERM)
    conn.enable_management()
    assert conn._management_active is True  # noqa: SLF001
