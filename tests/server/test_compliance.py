"""Tests for the client spec-compliance signalling helpers."""

from __future__ import annotations

import asyncio
import logging
import warnings
from contextlib import suppress
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest

from aiosendspin.models.core import ClientHelloPayload, DeviceInfo
from aiosendspin.models.types import ConnectionReason
from aiosendspin.noise.keys import Identity, generate_psk, psk_id_for
from aiosendspin.noise.trust_store import (
    InMemoryServerPairingStore,
    PskCategory,
    ResolvedPsk,
)
from aiosendspin.server import SendspinServer
from aiosendspin.server.compliance import (
    ClientComplianceError,
    describe_client,
    noncompliance_subject,
)
from aiosendspin.server.connection import SendspinConnection


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


def _hello(name: str, device_info: DeviceInfo | None = None) -> ClientHelloPayload:
    return ClientHelloPayload(name=name, supported_roles=[], device_info=device_info)


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
        "non-compliant client dev: legacy thing",
        "non-compliant client dev: other thing",
    ]
    assert all(r.levelno == logging.WARNING for r in hits)


@pytest.mark.asyncio
async def test_flag_noncompliance_strict_raises_and_logs_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Strict mode logs the reason at error and raises ClientComplianceError."""
    server = _make_server(allow_noncompliant_clients=False)
    client = server.get_or_create_client("dev")
    client.preload_hello(_hello("Kitchen", DeviceInfo(manufacturer="Acme")))
    with caplog.at_level(logging.INFO), pytest.raises(ClientComplianceError, match="legacy thing"):
        client.flag_noncompliance("legacy thing")
    hits = [r for r in caplog.records if "non-compliant client" in r.message]
    assert len(hits) == 1
    assert hits[0].levelno == logging.ERROR
    assert hits[0].message == "rejecting non-compliant client Kitchen (Acme): legacy thing"


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


@pytest.mark.parametrize(
    ("client_info", "client_id", "expected"),
    [
        (
            _hello("Kitchen", DeviceInfo("Speaker One", "Acme", "1.2.3", "aa:bb:cc:dd:ee:ff")),
            "dev-1",
            "Kitchen (Acme Speaker One, software 1.2.3)",
        ),
        (_hello("Kitchen", DeviceInfo(manufacturer="Acme")), "dev-1", "Kitchen (Acme)"),
        (
            _hello("Kitchen", DeviceInfo(product_name="Speaker One")),
            "dev-1",
            "Kitchen (Speaker One)",
        ),
        (
            _hello("Kitchen", DeviceInfo(software_version="1.2.3")),
            "dev-1",
            "Kitchen (software 1.2.3)",
        ),
        (_hello("Kitchen", DeviceInfo(mac_address="aa:bb:cc:dd:ee:ff")), "dev-1", "Kitchen"),
        (_hello("Kitchen", DeviceInfo()), "dev-1", "Kitchen"),
        (_hello("Kitchen"), "dev-1", "Kitchen"),
        (None, "dev-1", "dev-1"),
        (None, None, ""),
        # A blank or whitespace-only name falls through to whichever id is known.
        (_hello("", DeviceInfo(manufacturer="Acme")), "dev-1", "dev-1 (Acme)"),
        (_hello("   "), "dev-1", "dev-1"),
        (ClientHelloPayload(name="", supported_roles=[], client_id="legacy-1"), None, "legacy-1"),
        (_hello(""), None, ""),
        # Newlines would let a client forge whole log records; they are collapsed away.
        (_hello("Kitchen\nERROR forged"), "dev-1", "Kitchen ERROR forged"),
        (
            _hello("Kitchen", DeviceInfo(manufacturer="Acme\r\nERROR forged")),
            "dev-1",
            "Kitchen (Acme ERROR forged)",
        ),
        # Terminal escapes and NUL would let a client rewrite or truncate what an operator sees.
        (_hello("Kitchen\x1b[2K\x1b[1A"), "dev-1", "Kitchen[2K[1A"),
        (_hello("Kitchen", DeviceInfo(manufacturer="Acme\x00Corp")), "dev-1", "Kitchen (AcmeCorp)"),
        # A part of nothing but control characters and spaces drops out entirely.
        (_hello("\x00 \x00", DeviceInfo(manufacturer="Acme")), "dev-1", "dev-1 (Acme)"),
        # Unbounded client input is capped per part.
        (_hello("N" * 200), "dev-1", "N" * 64),
    ],
)
def test_describe_client_renders_only_the_parts_it_has(
    client_info: ClientHelloPayload | None, client_id: str | None, expected: str
) -> None:
    """Absent device details are dropped, never rendered as 'None' or empty parentheses."""
    assert describe_client(client_info, client_id) == expected


@pytest.mark.asyncio
async def test_flag_noncompliance_names_the_device_and_its_software(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A hello with full device_info puts every identifying field in the warning."""
    server = _make_server()
    client = server.get_or_create_client("dev")
    client.preload_hello(
        _hello("Kitchen", DeviceInfo("Speaker One", "Acme", "1.2.3", "aa:bb:cc:dd:ee:ff"))
    )
    with caplog.at_level(logging.WARNING):
        client.flag_noncompliance("legacy thing")
    assert caplog.messages == [
        "non-compliant client Kitchen (Acme Speaker One, software 1.2.3): legacy thing"
    ]


@pytest.mark.asyncio
async def test_flag_noncompliance_without_device_info_stays_clean(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A hello carrying no device_info names the device alone, with no 'None' filler."""
    server = _make_server()
    client = server.get_or_create_client("dev")
    client.preload_hello(_hello("Kitchen"))
    with caplog.at_level(logging.WARNING):
        client.flag_noncompliance("legacy thing")
    assert caplog.messages == ["non-compliant client Kitchen: legacy thing"]


@pytest.mark.asyncio
async def test_flag_noncompliance_with_partial_device_info_stays_clean(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A half-populated device_info yields no empty parentheses, stray comma or 'None'."""
    server = _make_server()
    client = server.get_or_create_client("dev")
    client.preload_hello(_hello("Kitchen", DeviceInfo(manufacturer="Acme")))
    with caplog.at_level(logging.WARNING):
        client.flag_noncompliance("legacy thing")
    assert caplog.messages == ["non-compliant client Kitchen (Acme): legacy thing"]


@pytest.mark.asyncio
async def test_flag_noncompliance_before_any_hello_names_the_client_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Before a hello arrives the client_id stands in for the device description."""
    server = _make_server()
    client = server.get_or_create_client("dev")
    with caplog.at_level(logging.WARNING):
        client.flag_noncompliance("legacy thing")
    assert caplog.messages == ["non-compliant client dev: legacy thing"]


def test_noncompliance_subject_degrades_without_a_description() -> None:
    """With nothing to name, the subject is the unqualified one."""
    assert noncompliance_subject("") == "non-compliant client"
    assert noncompliance_subject("Kitchen") == "non-compliant client Kitchen"


@pytest.mark.asyncio
async def test_flag_noncompliance_reports_the_latest_hello(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A device correcting its device_info on a second hello is reported with the new values."""
    server = _make_server()
    client = server.get_or_create_client("dev")
    client.preload_hello(_hello("Kitchen", DeviceInfo(software_version="1.2.3")))
    client.preload_hello(_hello("Kitchen", DeviceInfo(software_version="2.0.0")))
    with caplog.at_level(logging.WARNING):
        client.flag_noncompliance("legacy thing")
    assert caplog.messages == ["non-compliant client Kitchen (software 2.0.0): legacy thing"]


@pytest.mark.asyncio
async def test_hello_time_deviation_names_the_device_before_attach(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A deviation flagged during the hello exchange carries the description too."""
    server = _make_server()
    raw = orjson.dumps(
        {
            "type": "client/hello",
            "payload": {
                "name": "Kitchen",
                "supported_roles": ["player@v1"],
                "device_info": {"manufacturer": "Acme", "software_version": "1.2.3"},
                "player_support": {
                    "supported_formats": [
                        {"codec": "pcm", "channels": 2, "sample_rate": 48000, "bit_depth": 16}
                    ],
                    "buffer_capacity": 100_000,
                    "supported_commands": [],
                },
            },
        }
    ).decode()
    conn = SendspinConnection(server, wsock_client=AsyncMock())
    psk = generate_psk()
    conn._client_id = "dev"  # noqa: SLF001
    conn._noise_psk = ResolvedPsk(  # noqa: SLF001
        psk_id=psk_id_for(psk),
        psk=psk,
        category=PskCategory.LONG_TERM,
        counterparty_id="dev",
    )

    with caplog.at_level(logging.INFO):
        assert await conn._ingest_client_hello_checked(raw) is True  # noqa: SLF001
    await server.close()

    assert any(
        r.message.startswith(
            "non-compliant client Kitchen (Acme, software 1.2.3): "
            "client/hello used unversioned support keys"
        )
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_unimplemented_roles_notice_names_the_device(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The unimplemented-roles notice names the device, sanitized like every other line."""
    server = _make_server()
    raw = orjson.dumps(
        {
            "type": "client/hello",
            "payload": {
                "name": "Kitchen\nERROR forged",
                "supported_roles": ["player@v1", "player@v99"],
                "device_info": {"manufacturer": "Acme", "software_version": "1.2.3"},
                "player_support": {
                    "supported_formats": [
                        {"codec": "pcm", "channels": 2, "sample_rate": 48000, "bit_depth": 16}
                    ],
                    "buffer_capacity": 100_000,
                    "supported_commands": [],
                },
            },
        }
    ).decode()
    conn = SendspinConnection(server, wsock_client=AsyncMock())
    psk = generate_psk()
    conn._client_id = "dev"  # noqa: SLF001
    conn._noise_psk = ResolvedPsk(  # noqa: SLF001
        psk_id=psk_id_for(psk),
        psk=psk,
        category=PskCategory.LONG_TERM,
        counterparty_id="dev",
    )

    with caplog.at_level(logging.INFO):
        assert await conn._ingest_client_hello_checked(raw) is True  # noqa: SLF001
    await server.close()

    assert (
        "Client Kitchen ERROR forged (Acme, software 1.2.3) offered roles/versions "
        "this server does not implement: ['player@v99']"
    ) in caplog.messages
