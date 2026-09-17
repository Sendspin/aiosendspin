"""Tests for :mod:`aiosendspin.noise.driver`."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import replace
from typing import TYPE_CHECKING
from unittest.mock import patch

import orjson
import pytest
from aiohttp import WSMessage, WSMsgType

from aiosendspin.models.types import ServerErrorReason
from aiosendspin.noise.constants import PROTOCOL_VERSION
from aiosendspin.noise.driver import (
    HandshakeAbortedError,
    HandshakeResult,
    InitRejectedError,
    PskProvider,
    PskResolver,
    _exchange_as_responder,
    run_handshake_client,
    run_handshake_server,
    run_rehandshake_client,
    run_rehandshake_server,
)
from aiosendspin.noise.keys import (
    PEER_ID_SIZE,
    Identity,
    b64url_decode,
    b64url_encode,
    generate_psk,
    psk_id_for,
)
from aiosendspin.noise.models import (
    ClientInitMessage,
    ClientInitPayload,
    NoiseHandshakeMessage,
    NoiseHandshakePayload,
    ServerErrorMessage,
    ServerErrorPayload,
    ServerInitMessage,
    ServerInitPayload,
)
from aiosendspin.noise.session import NoiseCipherSuite, NoiseSession
from aiosendspin.noise.trust_store import PskCategory, ResolvedPsk
from tests.noise.conftest import FakeWebSocket, make_ws_pair

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


def _resolver(known: dict[str, ResolvedPsk]) -> PskResolver:
    """Build a store that answers only within the category message 1 declared."""

    async def resolve(psk_id: str, category: PskCategory) -> ResolvedPsk | None:
        found = known.get(psk_id)
        if found is None or found.category is not category:
            return None
        return found

    return resolve


def _provider(chosen: ResolvedPsk | None) -> PskProvider:
    async def provide(_client_id: str) -> ResolvedPsk | None:
        return chosen

    return provide


@pytest.mark.parametrize("suite", [NoiseCipherSuite.CHACHAPOLY, NoiseCipherSuite.AESGCM])
async def test_full_handshake_yields_paired_encrypted_websockets(
    suite: NoiseCipherSuite,
) -> None:
    """Server+client handshake produces two EncryptedWebSockets that talk to each other.

    Parametrized over both spec suites: the server admits whichever the client
    announces, so this also covers the server's AES-GCM acceptance path.
    """
    server_id = Identity.generate()
    client_id = Identity.generate()
    psk = generate_psk()
    psk_id = psk_id_for(psk)
    # counterparty_id is directional: the server's record names the client; the
    # client's record names the server.
    server_resolved = ResolvedPsk(
        psk_id=psk_id,
        psk=psk,
        category=PskCategory.LONG_TERM,
        counterparty_id=client_id.peer_id,
    )
    client_resolved = ResolvedPsk(
        psk_id=psk_id,
        psk=psk,
        category=PskCategory.LONG_TERM,
        counterparty_id=server_id.peer_id,
    )

    server_ws, client_ws = make_ws_pair()

    server_result, client_result = await asyncio.gather(
        run_handshake_server(
            server_ws,
            local_identity=server_id,
            psk_provider=_provider(server_resolved),
        ),
        run_handshake_client(
            client_ws,
            local_identity=client_id,
            suite=suite,
            psk_resolver=_resolver({client_resolved.psk_id: client_resolved}),
        ),
    )

    assert server_result.peer_id == client_id.peer_id
    assert client_result.peer_id == server_id.peer_id
    assert server_result.suite is suite
    assert client_result.suite is suite
    assert server_result.psk.psk == client_result.psk.psk
    # Both sides agree on the Noise handshake hash (used by pairing-code pairing / CPace).
    assert server_result.handshake_hash == client_result.handshake_hash
    assert len(server_result.handshake_hash) == 32

    # Smoke-test a roundtrip over the encrypted channel: server → client.
    await server_result.encrypted_ws.send_str('{"type":"server/hello"}')
    await server_ws.close_outbound()  # terminate client's iteration
    seen = [msg async for msg in client_result.encrypted_ws]
    assert seen[0].type is WSMsgType.TEXT
    assert seen[0].data == '{"type":"server/hello"}'


async def test_expected_server_id_match_passes() -> None:
    """run_client accepts a matching ``expected_server_id``."""
    server_id = Identity.generate()
    client_id = Identity.generate()
    psk = generate_psk()
    resolved = ResolvedPsk(
        psk_id=psk_id_for(psk),
        psk=psk,
        category=PskCategory.LONG_TERM,
    )

    server_ws, client_ws = make_ws_pair()
    await asyncio.gather(
        run_handshake_server(
            server_ws,
            local_identity=server_id,
            psk_provider=_provider(resolved),
        ),
        run_handshake_client(
            client_ws,
            local_identity=client_id,
            suite=NoiseCipherSuite.CHACHAPOLY,
            psk_resolver=_resolver({resolved.psk_id: resolved}),
            expected_server_id=server_id.peer_id,
        ),
    )


async def test_expected_server_id_mismatch_aborts_client() -> None:
    """run_client raises HandshakeAbortedError when expected_server_id doesn't match."""
    server_id = Identity.generate()
    client_id = Identity.generate()
    impostor = Identity.generate()
    psk = generate_psk()
    resolved = ResolvedPsk(psk_id=psk_id_for(psk), psk=psk, category=PskCategory.LONG_TERM)

    server_ws, client_ws = make_ws_pair()

    async def server_side() -> None:
        # Server may or may not complete depending on timing; if it errors we don't care here.
        with contextlib.suppress(HandshakeAbortedError):
            await run_handshake_server(
                server_ws,
                local_identity=server_id,
                psk_provider=_provider(resolved),
            )

    server_task = asyncio.create_task(server_side())
    with pytest.raises(HandshakeAbortedError, match="server_id mismatch"):
        await run_handshake_client(
            client_ws,
            local_identity=client_id,
            suite=NoiseCipherSuite.CHACHAPOLY,
            psk_resolver=_resolver({resolved.psk_id: resolved}),
            expected_server_id=impostor.peer_id,
        )
    server_task.cancel()


async def test_expected_client_id_mismatch_aborts_server() -> None:
    """run_server raises HandshakeAbortedError when expected_client_id doesn't match.

    Mirrors the client-side check for the server-initiated connection case.
    """
    server_id = Identity.generate()
    client_id = Identity.generate()
    other_client = Identity.generate()
    psk = generate_psk()
    resolved = ResolvedPsk(psk_id=psk_id_for(psk), psk=psk, category=PskCategory.SENTINEL)

    server_ws, client_ws = make_ws_pair()

    async def client_side() -> None:
        with contextlib.suppress(Exception):
            await run_handshake_client(
                client_ws,
                local_identity=client_id,
                suite=NoiseCipherSuite.CHACHAPOLY,
                psk_resolver=_resolver({resolved.psk_id: resolved}),
            )

    client_task = asyncio.create_task(client_side())
    with pytest.raises(HandshakeAbortedError, match="client_id mismatch"):
        await run_handshake_server(
            server_ws,
            local_identity=server_id,
            psk_provider=_provider(resolved),
            expected_client_id=other_client.peer_id,
        )
    client_task.cancel()


async def test_server_psk_provider_returning_none_aborts() -> None:
    """run_server raises HandshakeAbortedError if psk_provider returns None for the client."""
    server_id = Identity.generate()
    client_id = Identity.generate()

    server_ws, client_ws = make_ws_pair()

    async def client_side() -> None:
        with contextlib.suppress(HandshakeAbortedError):
            await run_handshake_client(
                client_ws,
                local_identity=client_id,
                suite=NoiseCipherSuite.CHACHAPOLY,
                psk_resolver=_resolver({}),
            )

    client_task = asyncio.create_task(client_side())
    with pytest.raises(HandshakeAbortedError, match="no PSK admits"):
        await run_handshake_server(
            server_ws,
            local_identity=server_id,
            psk_provider=_provider(None),
        )
    client_task.cancel()


async def test_psk_lookup_miss_completes_under_the_sentinel() -> None:
    """A client that cannot resolve the referenced psk_id answers under the Sentinel.

    Both sides end up on the Sentinel and report the mismatch, so the session survives
    to carry a re-pairing instead of failing at the handshake.
    """
    server_id = Identity.generate()
    client_id = Identity.generate()
    psk = generate_psk()
    resolved = ResolvedPsk(psk_id=psk_id_for(psk), psk=psk, category=PskCategory.LONG_TERM)

    server_ws, client_ws = make_ws_pair()
    server_result, client_result = await asyncio.gather(
        run_handshake_server(
            server_ws,
            local_identity=server_id,
            psk_provider=_provider(resolved),
        ),
        run_handshake_client(
            client_ws,
            local_identity=client_id,
            suite=NoiseCipherSuite.CHACHAPOLY,
            psk_resolver=_resolver({}),  # the record is gone
        ),
    )

    assert server_result.credential_mismatch is True
    assert client_result.credential_mismatch is True
    assert server_result.psk.category is PskCategory.SENTINEL
    assert client_result.psk.category is PskCategory.SENTINEL
    assert server_result.handshake_hash == client_result.handshake_hash


_CLIENT_ID = Identity.generate().peer_id
_SUITE = NoiseCipherSuite.CHACHAPOLY.value


def _client_init(**payload: object) -> str:
    return orjson.dumps({"type": "client/init", "payload": payload}).decode()


_MALFORMED = ServerErrorReason.MALFORMED
_UNSUPPORTED_VERSION = ServerErrorReason.UNSUPPORTED_VERSION
_UNSUPPORTED_SUITE = ServerErrorReason.UNSUPPORTED_SUITE


@pytest.mark.parametrize(
    ("client_init_text", "reason"),
    [
        pytest.param("this is not json", _MALFORMED, id="not-json"),
        pytest.param("[]", _MALFORMED, id="not-object"),
        pytest.param('{"type":"client/hello","payload":{}}', _MALFORMED, id="wrong-type"),
        pytest.param('{"type":"client/init"}', _MALFORMED, id="no-payload"),
        pytest.param('{"type":"client/init","payload":[]}', _MALFORMED, id="list-payload"),
        pytest.param(_client_init(client_id=_CLIENT_ID, suite=_SUITE), _MALFORMED, id="no-version"),
        pytest.param(
            _client_init(client_id=_CLIENT_ID, version="1", suite=_SUITE),
            _MALFORMED,
            id="string-version",
        ),
        pytest.param(
            _client_init(client_id=_CLIENT_ID, version=True, suite=_SUITE),
            _MALFORMED,
            id="bool-version",
        ),
        pytest.param(
            _client_init(client_id=_CLIENT_ID, version=1.0, suite=_SUITE),
            _MALFORMED,
            id="float-version",
        ),
        pytest.param(
            _client_init(client_id=_CLIENT_ID, version=None, suite=_SUITE),
            _MALFORMED,
            id="null-version",
        ),
        pytest.param(_client_init(version=2), _UNSUPPORTED_VERSION, id="future-version-bare"),
        pytest.param(
            _client_init(client_id=5, version=0, suite=5),
            _UNSUPPORTED_VERSION,
            id="other-version-bad-fields",
        ),
        pytest.param(
            _client_init(client_id=_CLIENT_ID, version=1, suite="25519_AESGCM_SHA512"),
            _UNSUPPORTED_SUITE,
            id="unknown-suite",
        ),
        pytest.param(
            _client_init(version=1, suite="25519_AESGCM_SHA512"),
            _UNSUPPORTED_SUITE,
            id="unknown-suite-no-client-id",
        ),
        pytest.param(_client_init(client_id=_CLIENT_ID, version=1), _MALFORMED, id="no-suite"),
        pytest.param(
            _client_init(client_id=_CLIENT_ID, version=1, suite=1),
            _MALFORMED,
            id="non-string-suite",
        ),
        pytest.param(_client_init(version=1, suite=_SUITE), _MALFORMED, id="no-client-id"),
        pytest.param(
            _client_init(client_id=5, version=1, suite=_SUITE),
            _MALFORMED,
            id="non-string-client-id",
        ),
        pytest.param(
            _client_init(client_id="tooshort", version=1, suite=_SUITE),
            _MALFORMED,
            id="short-client-id",
        ),
        pytest.param(
            _client_init(client_id="*" * PEER_ID_SIZE, version=1, suite=_SUITE),
            _MALFORMED,
            id="undecodable-client-id",
        ),
    ],
)
async def test_server_rejects_client_init_with_server_error(
    client_init_text: str,
    reason: ServerErrorReason,
) -> None:
    """An unacceptable client/init gets exactly one server/error with the spec-order reason."""
    server_ws = FakeWebSocket()
    with pytest.raises(InitRejectedError) as exc_info:
        await run_handshake_server(
            server_ws,
            local_identity=Identity.generate(),
            psk_provider=_provider(None),
            client_init_text=client_init_text,
        )
    assert exc_info.value.reason is reason
    assert server_ws.sent == [
        ServerErrorMessage(payload=ServerErrorPayload(reason=reason)).to_json(),
    ]
    assert orjson.loads(server_ws.sent[0]) == {
        "type": "server/error",
        "payload": {"reason": reason.value},
    }


async def test_server_receives_client_init_before_rejecting_it() -> None:
    """A rejected client/init read from the socket is answered with server/error."""
    server_ws, client_ws = make_ws_pair()
    await client_ws.send_str(_client_init(version=2))
    with pytest.raises(InitRejectedError):
        await run_handshake_server(
            server_ws,
            local_identity=Identity.generate(),
            psk_provider=_provider(None),
        )
    reply = await client_ws.receive()
    assert reply.type is WSMsgType.TEXT
    assert ServerErrorMessage.from_json(reply.data).payload.reason is _UNSUPPORTED_VERSION


async def test_server_rejection_survives_a_dropped_peer() -> None:
    """A peer gone before server/error is sent still yields the typed rejection."""
    server_ws = FakeWebSocket()

    async def send_str(_data: str) -> None:
        raise ConnectionResetError

    with (
        patch.object(server_ws, "send_str", send_str),
        pytest.raises(InitRejectedError) as exc_info,
    ):
        await run_handshake_server(
            server_ws,
            local_identity=Identity.generate(),
            psk_provider=_provider(None),
            client_init_text=_client_init(version=2),
        )
    assert exc_info.value.reason is _UNSUPPORTED_VERSION


async def _abort_non_text_first(server_ws: FakeWebSocket) -> None:
    await server_ws.push(WSMessage(WSMsgType.BINARY, b"\x00", ""))
    await run_handshake_server(
        server_ws, local_identity=Identity.generate(), psk_provider=_provider(None)
    )


async def _abort_closed_first(server_ws: FakeWebSocket) -> None:
    await server_ws.push(None)
    await run_handshake_server(
        server_ws, local_identity=Identity.generate(), psk_provider=_provider(None)
    )


async def _abort_timeout(server_ws: FakeWebSocket) -> None:
    await run_handshake_server(
        server_ws,
        local_identity=Identity.generate(),
        psk_provider=_provider(None),
        timeout_s=0.01,
    )


async def _abort_client_id_mismatch(server_ws: FakeWebSocket) -> None:
    await run_handshake_server(
        server_ws,
        local_identity=Identity.generate(),
        psk_provider=_provider(None),
        client_init_text=_client_init(client_id=_CLIENT_ID, version=1, suite=_SUITE),
        expected_client_id=Identity.generate().peer_id,
    )


async def _abort_psk_miss(server_ws: FakeWebSocket) -> None:
    await run_handshake_server(
        server_ws,
        local_identity=Identity.generate(),
        psk_provider=_provider(None),
        client_init_text=_client_init(client_id=_CLIENT_ID, version=1, suite=_SUITE),
    )


@pytest.mark.parametrize(
    "abort",
    [
        _abort_non_text_first,
        _abort_closed_first,
        _abort_timeout,
        _abort_client_id_mismatch,
        _abort_psk_miss,
    ],
)
async def test_server_non_init_failures_send_nothing(
    abort: Callable[[FakeWebSocket], Awaitable[None]],
) -> None:
    """Handshake-phase failures other than init failures close without a message."""
    server_ws = FakeWebSocket()
    with pytest.raises(HandshakeAbortedError) as exc_info:
        await abort(server_ws)
    assert not isinstance(exc_info.value, InitRejectedError)
    assert server_ws.sent == []


async def test_handshake_timeout_aborts() -> None:
    """If the peer never sends, run_server raises HandshakeAbortedError on timeout."""
    server_id = Identity.generate()
    server_ws, _client_ws = make_ws_pair()
    with pytest.raises(HandshakeAbortedError, match="timed out"):
        await run_handshake_server(
            server_ws,
            local_identity=server_id,
            psk_provider=_provider(None),
            timeout_s=0.05,
        )


async def test_client_post_match_check_rejects_wrong_bound_server_id() -> None:
    """A stored-pubkey PSK whose counterparty_id != the connected server_id is rejected.

    This is the spec's stored-pubkey post-match check: the PSK record stores the
    server's identity, and the client must confirm it reached that very server.
    """
    server_id = Identity.generate()
    client_id = Identity.generate()
    psk = generate_psk()
    server_resolved = ResolvedPsk(
        psk_id=psk_id_for(psk),
        psk=psk,
        category=PskCategory.LONG_TERM,
        counterparty_id=client_id.peer_id,
    )
    # Client's record claims the PSK belongs to a *different* server.
    other_server = Identity.generate()
    client_resolved = ResolvedPsk(
        psk_id=psk_id_for(psk),
        psk=psk,
        category=PskCategory.LONG_TERM,
        counterparty_id=other_server.peer_id,
    )

    server_ws, client_ws = make_ws_pair()

    async def server_side() -> None:
        with contextlib.suppress(Exception):
            await run_handshake_server(
                server_ws,
                local_identity=server_id,
                psk_provider=_provider(server_resolved),
            )

    server_task = asyncio.create_task(server_side())
    with pytest.raises(HandshakeAbortedError, match="bound to server_id"):
        await run_handshake_client(
            client_ws,
            local_identity=client_id,
            suite=NoiseCipherSuite.CHACHAPOLY,
            psk_resolver=_resolver({client_resolved.psk_id: client_resolved}),
        )
    server_task.cancel()


# DEPRECATED(spec-pr-183): remove in aiosendspin <version>
async def test_shared_psk_record_admits_any_server() -> None:
    """A shared-PSK record (counterparty_id=None) skips the post-match check.

    The client accepts the advertised server_id at face value, so the handshake
    completes even though the record is not bound to this server (spec's
    shared-PSK model).
    """
    server_id = Identity.generate()
    client_id = Identity.generate()
    psk = generate_psk()
    psk_id = psk_id_for(psk)
    server_resolved = ResolvedPsk(
        psk_id=psk_id,
        psk=psk,
        category=PskCategory.LONG_TERM,
        counterparty_id=client_id.peer_id,
    )
    # Shared-PSK record: no bound server_id.
    client_resolved = ResolvedPsk(
        psk_id=psk_id,
        psk=psk,
        category=PskCategory.LONG_TERM,
        counterparty_id=None,
    )

    server_ws, client_ws = make_ws_pair()

    server_result, client_result = await asyncio.gather(
        run_handshake_server(
            server_ws,
            local_identity=server_id,
            psk_provider=_provider(server_resolved),
        ),
        run_handshake_client(
            client_ws,
            local_identity=client_id,
            suite=NoiseCipherSuite.CHACHAPOLY,
            psk_resolver=_resolver({client_resolved.psk_id: client_resolved}),
        ),
    )

    assert client_result.peer_id == server_id.peer_id
    assert client_result.psk.counterparty_id is None
    assert server_result.psk.psk == client_result.psk.psk


async def test_psk_mismatch_after_lookup_aborts_initiator() -> None:
    """Responder returning a wrong PSK causes the initiator's msg2 AEAD to fail."""
    server_id = Identity.generate()
    client_id = Identity.generate()
    real_psk = generate_psk()
    wrong_psk = generate_psk()
    server_resolved = ResolvedPsk(
        psk_id=psk_id_for(real_psk),
        psk=real_psk,
        category=PskCategory.LONG_TERM,
    )
    # Client resolves the (correct) psk_id but returns the WRONG psk bytes.
    client_resolved = ResolvedPsk(
        psk_id=psk_id_for(real_psk),
        psk=wrong_psk,
        category=PskCategory.LONG_TERM,
    )

    server_ws, client_ws = make_ws_pair()

    async def client_side() -> None:
        with contextlib.suppress(HandshakeAbortedError):
            await run_handshake_client(
                client_ws,
                local_identity=client_id,
                suite=NoiseCipherSuite.CHACHAPOLY,
                psk_resolver=_resolver({server_resolved.psk_id: client_resolved}),
            )

    client_task = asyncio.create_task(client_side())
    # The initiator authenticates msg2 with real_psk; the responder mixed the
    # wrong PSK, so the AEAD tag fails — the driver wraps that as
    # HandshakeAbortedError (uniform handshake-failure contract).
    with pytest.raises(HandshakeAbortedError, match="failed Noise authentication"):
        await run_handshake_server(
            server_ws,
            local_identity=server_id,
            psk_provider=_provider(server_resolved),
        )
    await asyncio.wait_for(client_task, timeout=1.0)


async def test_rehandshake_swaps_keys_in_transport_mode() -> None:
    """A re-handshake over the encrypted channel installs a new session both sides.

    After the initial handshake, the server re-runs the handshake in transport
    mode to swap to a different PSK (mirroring trust promotion after pairing).
    The two ``noise/handshake`` messages travel doubly encrypted under the
    current keys; once swapped, traffic flows under the new keys and both sides
    agree on a fresh handshake hash.
    """
    server_id = Identity.generate()
    client_id = Identity.generate()

    # Initial session keyed by PSK #1.
    psk1 = generate_psk()
    server_psk1 = ResolvedPsk(
        psk_id=psk_id_for(psk1),
        psk=psk1,
        category=PskCategory.SENTINEL,
    )
    client_psk1 = ResolvedPsk(psk_id=psk_id_for(psk1), psk=psk1, category=PskCategory.SENTINEL)

    server_ws, client_ws = make_ws_pair()
    server_init, client_init = await asyncio.gather(
        run_handshake_server(
            server_ws,
            local_identity=server_id,
            psk_provider=_provider(server_psk1),
        ),
        run_handshake_client(
            client_ws,
            local_identity=client_id,
            suite=NoiseCipherSuite.CHACHAPOLY,
            psk_resolver=_resolver({client_psk1.psk_id: client_psk1}),
        ),
    )

    # New long-term PSK #2 to re-handshake into.
    psk2 = generate_psk()
    server_psk2 = ResolvedPsk(
        psk_id=psk_id_for(psk2),
        psk=psk2,
        category=PskCategory.LONG_TERM,
        counterparty_id=client_id.peer_id,
    )
    client_psk2 = ResolvedPsk(
        psk_id=psk_id_for(psk2),
        psk=psk2,
        category=PskCategory.LONG_TERM,
        counterparty_id=server_id.peer_id,
    )

    server_re, client_re = await asyncio.gather(
        run_rehandshake_server(
            server_init.encrypted_ws,
            local_identity=server_id,
            client_id=client_id.peer_id,
            suite=server_init.suite,
            prologue=server_init.handshake_hash,
            psk=server_psk2,
        ),
        run_rehandshake_client(
            client_init.encrypted_ws,
            local_identity=client_id,
            server_id=server_id.peer_id,
            suite=client_init.suite,
            prologue=client_init.handshake_hash,
            psk_resolver=_resolver({client_psk2.psk_id: client_psk2}),
        ),
    )

    # Same wrapper object, new session: both sides agree on a fresh hash that
    # differs from the initial one.
    assert server_re.encrypted_ws is server_init.encrypted_ws
    assert client_re.encrypted_ws is client_init.encrypted_ws
    assert server_re.handshake_hash == client_re.handshake_hash
    assert server_re.handshake_hash != server_init.handshake_hash
    assert server_re.psk.psk == psk2

    # Traffic now flows under the new keys.
    await server_re.encrypted_ws.send_str('{"type":"server/activate"}')
    await server_ws.close_outbound()
    seen = [msg async for msg in client_re.encrypted_ws]
    assert seen[0].type is WSMsgType.TEXT
    assert seen[0].data == '{"type":"server/activate"}'


async def _established_sessions() -> tuple[
    Identity, Identity, HandshakeResult, HandshakeResult, FakeWebSocket
]:
    """Return both identities, both initial handshake results, and the client's raw socket."""
    server_id = Identity.generate()
    client_id = Identity.generate()
    psk = generate_psk()
    resolved = ResolvedPsk(psk_id=psk_id_for(psk), psk=psk, category=PskCategory.SENTINEL)
    server_ws, client_ws = make_ws_pair()
    server_init, client_init = await asyncio.gather(
        run_handshake_server(server_ws, local_identity=server_id, psk_provider=_provider(resolved)),
        run_handshake_client(
            client_ws,
            local_identity=client_id,
            suite=NoiseCipherSuite.CHACHAPOLY,
            psk_resolver=_resolver({resolved.psk_id: resolved}),
        ),
    )
    return server_id, client_id, server_init, client_init, client_ws


def _long_term_psks(server_id: Identity, client_id: Identity) -> tuple[ResolvedPsk, PskResolver]:
    """Return a fresh long-term PSK as the server holds it and a client store holding it."""
    psk = generate_psk()
    server_psk = ResolvedPsk(
        psk_id=psk_id_for(psk),
        psk=psk,
        category=PskCategory.LONG_TERM,
        counterparty_id=client_id.peer_id,
    )
    client_psk = replace(server_psk, counterparty_id=server_id.peer_id)
    return server_psk, _resolver({client_psk.psk_id: client_psk})


async def test_rehandshake_discards_old_key_messages_before_message_2() -> None:
    """Application messages the client sent before message 1 do not break the re-handshake."""
    server_id, client_id, server_init, client_init, _ = await _established_sessions()
    server_psk, client_resolver = _long_term_psks(server_id, client_id)

    # In flight when the server starts: they reach it ahead of message 2.
    await client_init.encrypted_ws.send_str('{"type":"client/state","payload":{}}')
    await client_init.encrypted_ws.send_bytes(b"\x04audio")
    await client_init.encrypted_ws.send_str('{"type":"client/from-the-future","payload":{}}')
    await client_init.encrypted_ws.send_bytes(b"\x64unknown")

    server_re, client_re = await asyncio.gather(
        run_rehandshake_server(
            server_init.encrypted_ws,
            local_identity=server_id,
            client_id=client_id.peer_id,
            suite=server_init.suite,
            prologue=server_init.handshake_hash,
            psk=server_psk,
            timeout_s=1.0,
        ),
        run_rehandshake_client(
            client_init.encrypted_ws,
            local_identity=client_id,
            server_id=server_id.peer_id,
            suite=client_init.suite,
            prologue=client_init.handshake_hash,
            psk_resolver=client_resolver,
        ),
    )

    assert server_re.handshake_hash == client_re.handshake_hash
    assert server_re.psk.psk == server_psk.psk
    await client_re.encrypted_ws.send_str('{"type":"client/hello"}')
    msg = await server_re.encrypted_ws.receive()
    assert msg.type is WSMsgType.TEXT
    assert msg.data == '{"type":"client/hello"}'


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("this is not json", id="not-json"),
        pytest.param("[1]", id="not-an-object"),
        pytest.param('{"payload":{}}', id="no-type"),
        pytest.param('{"type":1}', id="non-string-type"),
    ],
)
async def test_rehandshake_aborts_on_untyped_text_before_message_2(text: str) -> None:
    """Only typed application messages are discarded; anything else aborts."""
    server_id, client_id, server_init, client_init, _ = await _established_sessions()
    server_psk, _ = _long_term_psks(server_id, client_id)

    await client_init.encrypted_ws.send_str(text)
    with pytest.raises(HandshakeAbortedError, match="malformed message while awaiting"):
        await run_rehandshake_server(
            server_init.encrypted_ws,
            local_identity=server_id,
            client_id=client_id.peer_id,
            suite=server_init.suite,
            prologue=server_init.handshake_hash,
            psk=server_psk,
            timeout_s=1.0,
        )


async def _send_closed(client_ws: FakeWebSocket) -> None:
    await client_ws.close_outbound()


async def _send_unauthenticated_frame(client_ws: FakeWebSocket) -> None:
    await client_ws.send_bytes(b"\x00" * 32)


@pytest.mark.parametrize(
    ("send", "match"),
    [
        pytest.param(_send_closed, "got CLOSED", id="closed"),
        pytest.param(_send_unauthenticated_frame, "got ERROR", id="error"),
    ],
)
async def test_rehandshake_aborts_on_error_or_close_before_message_2(
    send: Callable[[FakeWebSocket], Awaitable[None]], match: str
) -> None:
    """Discarding stops at transport failures: they still abort the re-handshake."""
    server_id, client_id, server_init, client_init, client_ws = await _established_sessions()
    server_psk, _ = _long_term_psks(server_id, client_id)

    await client_init.encrypted_ws.send_str('{"type":"client/state","payload":{}}')
    await send(client_ws)
    with pytest.raises(HandshakeAbortedError, match=match):
        await run_rehandshake_server(
            server_init.encrypted_ws,
            local_identity=server_id,
            client_id=client_id.peer_id,
            suite=server_init.suite,
            prologue=server_init.handshake_hash,
            psk=server_psk,
            timeout_s=1.0,
        )


async def test_rehandshake_discarded_messages_do_not_extend_the_deadline() -> None:
    """A client that keeps sending but never answers is cut off after one ``timeout_s``."""
    server_id, client_id, server_init, client_init, _ = await _established_sessions()
    server_psk, _ = _long_term_psks(server_id, client_id)

    async def chatter() -> None:
        while True:
            await client_init.encrypted_ws.send_str('{"type":"client/time","payload":{}}')
            await asyncio.sleep(0.005)

    chatter_task = asyncio.create_task(chatter())
    try:
        async with asyncio.timeout(1.0):
            with pytest.raises(HandshakeAbortedError, match="timed out awaiting Noise message 2"):
                await run_rehandshake_server(
                    server_init.encrypted_ws,
                    local_identity=server_id,
                    client_id=client_id.peer_id,
                    suite=server_init.suite,
                    prologue=server_init.handshake_hash,
                    psk=server_psk,
                    timeout_s=0.1,
                )
    finally:
        chatter_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await chatter_task


async def test_initial_handshake_aborts_on_application_message_before_message_2() -> None:
    """Only the re-handshake discards: the cleartext handshake has no earlier session."""
    server_id = Identity.generate()
    client_id = Identity.generate()
    psk = generate_psk()
    resolved = ResolvedPsk(psk_id=psk_id_for(psk), psk=psk, category=PskCategory.SENTINEL)
    server_ws, client_ws = make_ws_pair()

    async def chatty_client() -> None:
        await client_ws.send_str(
            ClientInitMessage(
                payload=ClientInitPayload(
                    client_id=client_id.peer_id,
                    version=PROTOCOL_VERSION,
                    suite=NoiseCipherSuite.CHACHAPOLY.value,
                ),
            ).to_json(),
        )
        await client_ws.receive()  # server/init
        await client_ws.receive()  # Noise message 1
        await client_ws.send_str('{"type":"client/state","payload":{}}')

    client_task = asyncio.create_task(chatty_client())
    with pytest.raises(HandshakeAbortedError, match="malformed noise/handshake"):
        await run_handshake_server(
            server_ws,
            local_identity=server_id,
            psk_provider=_provider(resolved),
            timeout_s=1.0,
        )
    await client_task


# --- adversarial: the client rejecting a malicious / buggy server ----------
#
# The pairing trust flows client -> server, so the client must treat every frame
# the server sends as hostile input. Each malformed server frame must abort the
# handshake as HandshakeAbortedError (never a raw exception, or a hang), because
# only HandshakeAbortedError is caught by the client's connection bring-up.


def _valid_server_init(server_id: str, *, version: int = PROTOCOL_VERSION) -> str:
    return ServerInitMessage(
        payload=ServerInitPayload(server_id=server_id, version=version),
    ).to_json()


async def _send_non_text_first(server_ws: FakeWebSocket, _server_id: Identity) -> None:
    await server_ws.send_bytes(b"\x00not-a-text-frame")


async def _send_malformed_init(server_ws: FakeWebSocket, _server_id: Identity) -> None:
    await server_ws.send_str("this is not json")


async def _send_wrong_type_init(server_ws: FakeWebSocket, _server_id: Identity) -> None:
    await server_ws.send_str('{"type":"server/hello","payload":{}}')


async def _send_bad_version_init(server_ws: FakeWebSocket, server_id: Identity) -> None:
    await server_ws.send_str(_valid_server_init(server_id.peer_id, version=PROTOCOL_VERSION + 1))


def _send_server_init_version(
    version: object,
) -> Callable[[FakeWebSocket, Identity], Awaitable[None]]:
    async def send(server_ws: FakeWebSocket, server_id: Identity) -> None:
        payload = {"server_id": server_id.peer_id, "version": version}
        await server_ws.send_str(orjson.dumps({"type": "server/init", "payload": payload}).decode())

    return send


async def _send_short_server_id(server_ws: FakeWebSocket, _server_id: Identity) -> None:
    await server_ws.send_str(
        '{"type":"server/init","payload":{"server_id":"tooshort","version":1}}',
    )


async def _send_undecodable_server_id(server_ws: FakeWebSocket, _server_id: Identity) -> None:
    # 43 chars (the right length) but decodes to the wrong key size.
    await server_ws.send_str(
        '{"type":"server/init","payload":{"server_id":"' + ("*" * PEER_ID_SIZE) + '","version":1}}',
    )


async def _send_undecodable_msg1(server_ws: FakeWebSocket, server_id: Identity) -> None:
    await server_ws.send_str(_valid_server_init(server_id.peer_id))
    await server_ws.send_str(
        NoiseHandshakeMessage(payload=NoiseHandshakePayload(data="!!not base64!!")).to_json(),
    )


async def _send_wrong_type_msg1(server_ws: FakeWebSocket, server_id: Identity) -> None:
    await server_ws.send_str(_valid_server_init(server_id.peer_id))
    await server_ws.send_str(
        '{"type":"noise/rehandshake","payload":{"data":"' + b64url_encode(b"x" * 48) + '"}}',
    )


async def _send_garbage_msg1(server_ws: FakeWebSocket, server_id: Identity) -> None:
    await server_ws.send_str(_valid_server_init(server_id.peer_id))
    payload = NoiseHandshakePayload(data=b64url_encode(b"x" * 48))  # valid base64, wrong length
    await server_ws.send_str(NoiseHandshakeMessage(payload=payload).to_json())


async def _send_malformed_msg1(server_ws: FakeWebSocket, server_id: Identity) -> None:
    await server_ws.send_str(_valid_server_init(server_id.peer_id))
    await server_ws.send_str("this is not json either")


@pytest.mark.parametrize(
    ("server_send", "match"),
    [
        (_send_non_text_first, "server/init"),
        (_send_malformed_init, "malformed server/init"),
        (_send_wrong_type_init, "server/init"),
        (_send_bad_version_init, "unsupported protocol version"),
        (_send_server_init_version("1"), "malformed server/init version"),
        (_send_server_init_version(True), "malformed server/init version"),  # noqa: FBT003
        (_send_server_init_version(1.0), "malformed server/init version"),
        (_send_short_server_id, "invalid server_id length"),
        (_send_undecodable_server_id, "invalid server_id"),
        (_send_undecodable_msg1, "payload encoding"),
        (_send_wrong_type_msg1, "malformed noise/handshake"),
        (_send_garbage_msg1, "failed Noise authentication"),
        (_send_malformed_msg1, "malformed noise/handshake"),
    ],
)
async def test_client_rejects_malicious_server_frame(
    server_send: Callable[[FakeWebSocket, Identity], Awaitable[None]],
    match: str,
) -> None:
    """Every malformed server frame aborts the client handshake as HandshakeAbortedError."""
    server_id = Identity.generate()
    client_id = Identity.generate()
    psk = generate_psk()
    resolved = ResolvedPsk(
        psk_id=psk_id_for(psk),
        psk=psk,
        category=PskCategory.LONG_TERM,
        counterparty_id=server_id.peer_id,
    )
    server_ws, client_ws = make_ws_pair()

    async def bogus_server() -> None:
        await server_ws.receive()  # consume client/init
        await server_send(server_ws, server_id)

    server_task = asyncio.create_task(bogus_server())
    with pytest.raises(HandshakeAbortedError, match=match):
        await run_handshake_client(
            client_ws,
            local_identity=client_id,
            suite=NoiseCipherSuite.CHACHAPOLY,
            psk_resolver=_resolver({resolved.psk_id: resolved}),
            timeout_s=1.0,
        )
    await server_task


@pytest.mark.parametrize(
    ("raw_reason", "reason"),
    [
        ("unsupported_version", ServerErrorReason.UNSUPPORTED_VERSION),
        ("unsupported_suite", ServerErrorReason.UNSUPPORTED_SUITE),
        ("malformed", ServerErrorReason.MALFORMED),
        ("from_the_future", None),
        (None, None),
    ],
)
async def test_client_raises_init_rejected_on_server_error(
    raw_reason: str | None,
    reason: ServerErrorReason | None,
) -> None:
    """A server/error in place of server/init aborts the client with its reason."""
    server_ws, client_ws = make_ws_pair()
    await server_ws.send_str(
        orjson.dumps({"type": "server/error", "payload": {"reason": raw_reason}}).decode()
    )
    with pytest.raises(InitRejectedError) as exc_info:
        await run_handshake_client(
            client_ws,
            local_identity=Identity.generate(),
            suite=NoiseCipherSuite.CHACHAPOLY,
            psk_resolver=_resolver({}),
            timeout_s=1.0,
        )
    assert exc_info.value.reason is reason
    assert repr(raw_reason) in str(exc_info.value)
    # The client sent only its client/init and nothing in reply.
    assert len(client_ws.sent) == 1


async def test_server_rejects_bad_base64_msg2() -> None:
    """A client that sends an unstructured Noise message 2 aborts the server handshake."""
    server_id = Identity.generate()
    client_id = Identity.generate()
    psk = generate_psk()
    resolved = ResolvedPsk(psk_id=psk_id_for(psk), psk=psk, category=PskCategory.SENTINEL)
    server_ws, client_ws = make_ws_pair()

    async def bogus_client() -> None:
        await client_ws.send_str(
            ClientInitMessage(
                payload=ClientInitPayload(
                    client_id=client_id.peer_id,
                    version=PROTOCOL_VERSION,
                    suite=NoiseCipherSuite.CHACHAPOLY.value,
                ),
            ).to_json(),
        )
        await client_ws.receive()  # server/init
        await client_ws.receive()  # Noise message 1
        await client_ws.send_str(
            NoiseHandshakeMessage(payload=NoiseHandshakePayload(data="!!not base64!!")).to_json(),
        )

    client_task = asyncio.create_task(bogus_client())
    with pytest.raises(HandshakeAbortedError, match="Noise message 2"):
        await run_handshake_server(
            server_ws,
            local_identity=server_id,
            psk_provider=_provider(resolved),
            timeout_s=1.0,
        )
    await client_task


async def test_server_rejects_msg2_with_malformed_payload() -> None:
    """A structurally valid Noise message 2 whose plaintext isn't ``{}`` aborts the server.

    The client here runs a real responder session (so message 2 decrypts), but
    writes a non-empty-object payload — exercising the msg2 payload validation.
    """
    server_id = Identity.generate()
    client_id = Identity.generate()
    psk = generate_psk()
    resolved = ResolvedPsk(psk_id=psk_id_for(psk), psk=psk, category=PskCategory.SENTINEL)
    server_ws, client_ws = make_ws_pair()

    async def bogus_client() -> None:
        client_init = ClientInitMessage(
            payload=ClientInitPayload(
                client_id=client_id.peer_id,
                version=PROTOCOL_VERSION,
                suite=NoiseCipherSuite.CHACHAPOLY.value,
            ),
        ).to_json()
        await client_ws.send_str(client_init)
        server_init = (await client_ws.receive()).data
        prologue = client_init.encode("utf-8") + server_init.encode("utf-8")
        session = NoiseSession.as_responder(
            suite=NoiseCipherSuite.CHACHAPOLY,
            local_static_priv=client_id.private_bytes,
            remote_static_pub=server_id.public_bytes,
            prologue=prologue,
        )
        hs1 = NoiseHandshakeMessage.from_json((await client_ws.receive()).data)
        session.read_message(b64url_decode(hs1.payload.data))
        session.mix_psk(psk)
        bad = session.write_message(b"not json")  # valid Noise, invalid payload
        await client_ws.send_str(
            NoiseHandshakeMessage(payload=NoiseHandshakePayload(data=b64url_encode(bad))).to_json(),
        )

    client_task = asyncio.create_task(bogus_client())
    with pytest.raises(HandshakeAbortedError, match="malformed Noise message 2 payload"):
        await run_handshake_server(
            server_ws,
            local_identity=server_id,
            psk_provider=_provider(resolved),
            timeout_s=1.0,
        )
    await client_task


async def test_psk_held_under_another_category_falls_back_to_the_sentinel() -> None:
    """The declared category binds: the same psk_id under another category is a miss."""
    server_id = Identity.generate()
    client_id = Identity.generate()
    psk = generate_psk()
    # The server references the PSK as long-term; the client holds it as a pairing PSK.
    server_resolved = ResolvedPsk(psk_id=psk_id_for(psk), psk=psk, category=PskCategory.LONG_TERM)
    client_resolved = ResolvedPsk(psk_id=psk_id_for(psk), psk=psk, category=PskCategory.PAIRING)

    server_ws, client_ws = make_ws_pair()
    server_result, client_result = await asyncio.gather(
        run_handshake_server(
            server_ws,
            local_identity=server_id,
            psk_provider=_provider(server_resolved),
        ),
        run_handshake_client(
            client_ws,
            local_identity=client_id,
            suite=NoiseCipherSuite.CHACHAPOLY,
            psk_resolver=_resolver({client_resolved.psk_id: client_resolved}),
        ),
    )

    assert client_result.credential_mismatch is True
    assert server_result.credential_mismatch is True
    assert server_result.psk.category is PskCategory.SENTINEL


async def test_message_1_without_a_category_is_rejected() -> None:
    """The category is required: a server that omits it does not get a handshake."""
    server_id = Identity.generate()
    client_id = Identity.generate()
    psk = generate_psk()
    resolved = ResolvedPsk(
        psk_id=psk_id_for(psk),
        psk=psk,
        category=PskCategory.LONG_TERM,
        counterparty_id=server_id.peer_id,
    )
    server_ws, client_ws = make_ws_pair()

    async def server_without_a_category() -> None:
        """Drive the server side, writing a message 1 payload that names no category."""
        client_init = (await server_ws.receive()).data
        server_init = ServerInitMessage(
            payload=ServerInitPayload(server_id=server_id.peer_id, version=PROTOCOL_VERSION),
        ).to_json()
        prologue = client_init.encode("utf-8") + server_init.encode("utf-8")
        session = NoiseSession.as_initiator(
            suite=NoiseCipherSuite.CHACHAPOLY,
            local_static_priv=server_id.private_bytes,
            remote_static_pub=client_id.public_bytes,
            prologue=prologue,
            psk=psk,
        )
        await server_ws.send_str(server_init)
        msg1 = session.write_message(f'{{"psk_id":"{resolved.psk_id}"}}'.encode())
        await server_ws.send_str(
            NoiseHandshakeMessage(payload=NoiseHandshakePayload(data=b64url_encode(msg1))).to_json()
        )

    server_task = asyncio.create_task(server_without_a_category())
    with pytest.raises(HandshakeAbortedError, match="malformed Noise message 1 payload"):
        await run_handshake_client(
            client_ws,
            local_identity=client_id,
            suite=NoiseCipherSuite.CHACHAPOLY,
            psk_resolver=_resolver({resolved.psk_id: resolved}),
        )
    await asyncio.gather(server_task, return_exceptions=True)


@pytest.mark.parametrize(
    "category",
    ["zz", [], {}, 1, None],
    ids=["unknown-code", "list", "object", "number", "null"],
)
async def test_message_1_with_an_unknown_category_is_rejected(category: object) -> None:
    """An undefined category is malformed input, not a miss the Sentinel could answer."""
    server_id = Identity.generate()
    client_id = Identity.generate()
    psk = generate_psk()
    resolved = ResolvedPsk(psk_id=psk_id_for(psk), psk=psk, category=PskCategory.LONG_TERM)
    server_ws, client_ws = make_ws_pair()
    lookups: list[tuple[str, PskCategory]] = []

    async def recording_resolver(psk_id: str, category: PskCategory) -> ResolvedPsk | None:
        lookups.append((psk_id, category))
        return resolved

    async def server_with_an_unknown_category() -> None:
        """Drive the server side, naming a held psk_id under a category no revision defines."""
        client_init = (await server_ws.receive()).data
        server_init = ServerInitMessage(
            payload=ServerInitPayload(server_id=server_id.peer_id, version=PROTOCOL_VERSION),
        ).to_json()
        prologue = client_init.encode("utf-8") + server_init.encode("utf-8")
        session = NoiseSession.as_initiator(
            suite=NoiseCipherSuite.CHACHAPOLY,
            local_static_priv=server_id.private_bytes,
            remote_static_pub=client_id.public_bytes,
            prologue=prologue,
            psk=psk,
        )
        await server_ws.send_str(server_init)
        payload = orjson.dumps({"psk_id": resolved.psk_id, "psk_category": category})
        msg1 = session.write_message(payload)
        await server_ws.send_str(
            NoiseHandshakeMessage(payload=NoiseHandshakePayload(data=b64url_encode(msg1))).to_json()
        )

    server_task = asyncio.create_task(server_with_an_unknown_category())
    with pytest.raises(
        HandshakeAbortedError, match="malformed Noise message 1 payload: unknown psk_category"
    ):
        await run_handshake_client(
            client_ws,
            local_identity=client_id,
            suite=NoiseCipherSuite.CHACHAPOLY,
            psk_resolver=recording_resolver,
        )
    await server_task

    assert lookups == []
    # Only client/init went out: no message 2, under the Sentinel or otherwise.
    assert len(client_ws.sent) == 1
    assert server_ws.incoming.empty()


async def test_rehandshake_category_mismatch_aborts() -> None:
    """A category mismatch during a re-handshake is a miss, and a miss there aborts."""
    server_id = Identity.generate()
    client_id = Identity.generate()

    psk1 = generate_psk()
    psk1_resolved = ResolvedPsk(psk_id=psk_id_for(psk1), psk=psk1, category=PskCategory.SENTINEL)

    server_ws, client_ws = make_ws_pair()
    server_init, client_init = await asyncio.gather(
        run_handshake_server(
            server_ws,
            local_identity=server_id,
            psk_provider=_provider(psk1_resolved),
        ),
        run_handshake_client(
            client_ws,
            local_identity=client_id,
            suite=NoiseCipherSuite.CHACHAPOLY,
            psk_resolver=_resolver({psk1_resolved.psk_id: psk1_resolved}),
        ),
    )

    # The server re-handshakes referencing psk2 as long-term; the client holds it as pairing.
    psk2 = generate_psk()
    server_psk2 = ResolvedPsk(
        psk_id=psk_id_for(psk2),
        psk=psk2,
        category=PskCategory.LONG_TERM,
        counterparty_id=client_id.peer_id,
    )
    client_psk2 = ResolvedPsk(psk_id=psk_id_for(psk2), psk=psk2, category=PskCategory.PAIRING)

    server_task = asyncio.create_task(
        run_rehandshake_server(
            server_init.encrypted_ws,
            local_identity=server_id,
            client_id=client_id.peer_id,
            suite=server_init.suite,
            prologue=server_init.handshake_hash,
            psk=server_psk2,
            timeout_s=1.0,
        )
    )
    with pytest.raises(HandshakeAbortedError, match="no PSK matches psk_id"):
        await run_rehandshake_client(
            client_init.encrypted_ws,
            local_identity=client_id,
            server_id=server_id.peer_id,
            suite=client_init.suite,
            prologue=client_init.handshake_hash,
            psk_resolver=_resolver({client_psk2.psk_id: client_psk2}),
        )
    await asyncio.gather(server_task, return_exceptions=True)


async def test_rehandshake_with_an_unknown_category_is_rejected() -> None:
    """An undefined category aborts a re-handshake as malformed, not as a miss."""
    server_id, client_id, server_init, client_init, client_ws = await _established_sessions()
    server_psk, client_resolver = _long_term_psks(server_id, client_id)
    session = NoiseSession.as_initiator(
        suite=server_init.suite,
        local_static_priv=server_id.private_bytes,
        remote_static_pub=client_id.public_bytes,
        prologue=server_init.handshake_hash,
        psk=server_psk.psk,
    )
    payload = orjson.dumps({"psk_id": server_psk.psk_id, "psk_category": "zz"})
    msg1 = session.write_message(payload)
    await server_init.encrypted_ws.send_str(
        NoiseHandshakeMessage(payload=NoiseHandshakePayload(data=b64url_encode(msg1))).to_json()
    )
    sent_before = len(client_ws.sent)

    with pytest.raises(
        HandshakeAbortedError, match="malformed Noise message 1 payload: unknown psk_category"
    ):
        await run_rehandshake_client(
            client_init.encrypted_ws,
            local_identity=client_id,
            server_id=server_id.peer_id,
            suite=client_init.suite,
            prologue=client_init.handshake_hash,
            psk_resolver=client_resolver,
            timeout_s=1.0,
        )
    assert len(client_ws.sent) == sent_before


async def test_rehandshake_referencing_an_unusable_psk_aborts_the_server() -> None:
    """The initiator half of the re-handshake rule: no Sentinel rescue there either.

    The client is driven with the fallback deliberately enabled, so it really does answer
    message 2 under the Sentinel. The server must refuse that rather than accept it, which
    is what keeps a live session from being demoted mid-connection.
    """
    server_id = Identity.generate()
    client_id = Identity.generate()

    psk1 = generate_psk()
    psk1_resolved = ResolvedPsk(psk_id=psk_id_for(psk1), psk=psk1, category=PskCategory.SENTINEL)

    server_ws, client_ws = make_ws_pair()
    server_init, client_init = await asyncio.gather(
        run_handshake_server(
            server_ws, local_identity=server_id, psk_provider=_provider(psk1_resolved)
        ),
        run_handshake_client(
            client_ws,
            local_identity=client_id,
            suite=NoiseCipherSuite.CHACHAPOLY,
            psk_resolver=_resolver({psk1_resolved.psk_id: psk1_resolved}),
        ),
    )

    # The server re-handshakes on a record the client does not hold at all.
    psk2 = generate_psk()
    server_psk2 = ResolvedPsk(
        psk_id=psk_id_for(psk2),
        psk=psk2,
        category=PskCategory.LONG_TERM,
        counterparty_id=client_id.peer_id,
    )

    async def falling_back_client() -> None:
        """Answer the re-handshake under the Sentinel, as a conformant client never would."""
        session = NoiseSession.as_responder(
            suite=client_init.suite,
            local_static_priv=client_id.private_bytes,
            remote_static_pub=server_id.public_bytes,
            prologue=client_init.handshake_hash,
        )
        await _exchange_as_responder(
            client_init.encrypted_ws,
            session=session,
            psk_resolver=_resolver({}),
            expected_peer_id=server_id.peer_id,
            timeout_s=1.0,
            allow_sentinel_fallback=True,
        )

    client_task = asyncio.create_task(falling_back_client())
    with pytest.raises(HandshakeAbortedError):
        await run_rehandshake_server(
            server_init.encrypted_ws,
            local_identity=server_id,
            client_id=client_id.peer_id,
            suite=server_init.suite,
            prologue=server_init.handshake_hash,
            psk=server_psk2,
            timeout_s=1.0,
        )
    await asyncio.gather(client_task, return_exceptions=True)


async def test_fork_requires_a_written_message_1() -> None:
    """The fork is only meaningful mid-handshake, and says so rather than misbehaving."""
    server_id = Identity.generate()
    client_id = Identity.generate()
    session = NoiseSession.as_initiator(
        suite=NoiseCipherSuite.CHACHAPOLY,
        local_static_priv=server_id.private_bytes,
        remote_static_pub=client_id.public_bytes,
        prologue=b"prologue",
        psk=generate_psk(),
    )

    with pytest.raises(RuntimeError, match="written message 1"):
        session.fork_at_message_2(generate_psk())


async def test_fork_refuses_a_message_1_it_cannot_reproduce() -> None:
    """A replay that differs from what the peer answered is drift, and must be loud."""
    server_id = Identity.generate()
    client_id = Identity.generate()
    session = NoiseSession.as_initiator(
        suite=NoiseCipherSuite.CHACHAPOLY,
        local_static_priv=server_id.private_bytes,
        remote_static_pub=client_id.public_bytes,
        prologue=b"prologue",
        psk=generate_psk(),
    )
    session.write_message(b'{"psk_id":"x"}')
    # Stand in for a library whose ephemeral handling has drifted under us.
    with (
        patch(
            "aiosendspin.noise.session._ephemeral_private_bytes",
            return_value=Identity.generate().private_bytes,
        ),
        pytest.raises(RuntimeError, match="differs from the one sent"),
    ):
        session.fork_at_message_2(generate_psk())
