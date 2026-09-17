"""Tests for the client spec-compliance signalling helpers."""

from __future__ import annotations

import asyncio
import logging
import warnings
from contextlib import suppress
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiosendspin.models.types import ConnectionReason
from aiosendspin.noise.keys import Identity
from aiosendspin.noise.trust_store import InMemoryServerPairingStore
from aiosendspin.server import SendspinServer
from aiosendspin.server.compliance import ClientComplianceError


def _make_server(*, allow_noncompliant_clients: bool = True) -> SendspinServer:
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
        allow_noncompliant_clients=allow_noncompliant_clients,
    )


@pytest.mark.asyncio
async def test_allow_noncompliant_clients_defaults_to_true() -> None:
    """Rejecting non-compliant clients is opt-in."""
    server = _make_server()
    assert server.allow_noncompliant_clients is True


@pytest.mark.asyncio
async def test_flag_noncompliance_lenient_dedups_per_reason(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Lenient mode logs each distinct reason once at warning, not on every occurrence."""
    server = _make_server()
    client = server.get_or_create_client("dev")
    with caplog.at_level(logging.INFO):
        client.flag_noncompliance("legacy thing")
        client.flag_noncompliance("legacy thing")
        client.flag_noncompliance("other thing")
    hits = [r for r in caplog.records if "non-compliant client" in r.message]
    assert [r.message for r in hits] == [
        "non-compliant client: legacy thing",
        "non-compliant client: other thing",
    ]
    assert all(r.levelno == logging.WARNING for r in hits)


@pytest.mark.asyncio
async def test_flag_noncompliance_strict_raises_and_logs_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Strict mode logs the reason at error and raises ClientComplianceError."""
    server = _make_server(allow_noncompliant_clients=False)
    client = server.get_or_create_client("dev")
    with caplog.at_level(logging.INFO), pytest.raises(ClientComplianceError, match="legacy thing"):
        client.flag_noncompliance("legacy thing")
    hits = [r for r in caplog.records if "non-compliant client" in r.message]
    assert len(hits) == 1
    assert hits[0].levelno == logging.ERROR


# DEPRECATED(spec-pr-183): remove in aiosendspin <version>
@pytest.mark.asyncio
async def test_management_connection_reason_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    """Dialing with ConnectionReason.MANAGEMENT warns once, naming the embedder's call."""
    server = _make_server()
    url = "ws://127.0.0.1:9/sendspin"
    try:
        with caplog.at_level(logging.WARNING), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            server.connect_to_client(url, connection_reason=ConnectionReason.MANAGEMENT)
            server.connect_to_client(url, connection_reason=ConnectionReason.MANAGEMENT)
            server.connect_to_client(url, connection_reason=ConnectionReason.PLAYBACK)
    finally:
        await server.close()

    deprecations = [w for w in caught if w.category is DeprecationWarning]
    assert len(deprecations) == 1
    assert str(deprecations[0].message).startswith("ConnectionReason.MANAGEMENT is deprecated")
    assert deprecations[0].filename == __file__
    logged = [r for r in caplog.records if r.message.startswith("ConnectionReason.MANAGEMENT")]
    assert len(logged) == 1


# DEPRECATED(spec-pr-183): remove in aiosendspin <version>
@pytest.mark.asyncio
async def test_management_connection_reason_warns_on_wait_too() -> None:
    """connect_to_client_and_wait also names the embedder's call in its warning."""
    server = _make_server()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with suppress(OSError, TimeoutError):
                async with asyncio.timeout(5):
                    await server.connect_to_client_and_wait(
                        "ws://127.0.0.1:9/sendspin",
                        connection_reason=ConnectionReason.MANAGEMENT,
                    )
    finally:
        await server.close()

    deprecations = [w for w in caught if w.category is DeprecationWarning]
    assert [w.filename for w in deprecations] == [__file__]
