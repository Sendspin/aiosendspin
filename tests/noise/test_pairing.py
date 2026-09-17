"""Tests for :mod:`aiosendspin.noise.pairing`."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime

import pytest
from cpace import CPace, CPaceRole
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

from aiosendspin.models.core import ServerActivateMessage, ServerActivatePayload
from aiosendspin.models.types import Activity, PairAbortReason, PairingCodeFormat, PairMethod
from aiosendspin.noise import pairing_code as pairing_code_mod
from aiosendspin.noise.keys import b64url_decode, b64url_encode, generate_psk, psk_id_for
from aiosendspin.noise.models import (
    ClientPairAuthMessage,
    ClientPairAuthPayload,
    ClientPairConfirmMessage,
    ClientPairConfirmPayload,
    ClientPairFinalizeMessage,
    ClientPairFinalizePayload,
    ClientPairInitMessage,
    ClientPairInitPayload,
    ClientPairPendingMessage,
    ClientPairPendingPayload,
    ClientPairRetryMessage,
    ServerPairAuthMessage,
    ServerPairAuthPayload,
    ServerPairConfirmMessage,
    ServerPairConfirmPayload,
    ServerPairFinalizeMessage,
    ServerPairInitMessage,
    ServerPairInitPayload,
)
from aiosendspin.noise.pairing import (
    InvalidPairingCodeError,
    PairingAbortError,
    PairingAttempt,
    PairingError,
    PairingTimeoutError,
    _legacy_pake_sid,
    _pake_sid,
    run_dynamic_pairing_code_client,
    run_dynamic_pairing_code_server,
    run_pairing_psk_client,
    run_pairing_psk_server,
    run_static_pairing_code_client,
    run_static_pairing_code_server,
)
from aiosendspin.noise.trust_store import (
    PAIRING_ROUND_LIMIT,
    ClientPairingRecord,
    InMemoryClientPairingStore,
    InMemoryServerPairingStore,
    ServerPairingRecord,
)
from aiosendspin.noise.wire import EncryptedWebSocket
from tests.noise.conftest import make_paired_encrypted_ws
from tests.pairing_stores import ExhaustedClientStore


def _added_records(records: Sequence[ClientPairingRecord]) -> list[ClientPairingRecord]:
    """Stored-pubkey records added by pairing (excludes the pre-provisioned shared record)."""
    return [r for r in records if r.server_id is not None]


async def _code() -> str:
    return "000000"


def test_pairing_attempt_verify_is_code_only() -> None:
    """Verify is a pairing code-only re-authentication flag; PAIRING_PSK rejects it."""
    with pytest.raises(ValueError, match="does not support verification"):
        PairingAttempt(method=PairMethod.PAIRING_PSK, pairing_psk=generate_psk(), verify=True)
    assert PairingAttempt(
        method=PairMethod.STATIC_PAIRING_CODE, pairing_code_provider=_code, verify=True
    ).verify
    assert PairingAttempt(
        method=PairMethod.DYNAMIC_PAIRING_CODE,
        pairing_code_provider=_code,
        verify=True,
        pairing_format=PairingCodeFormat.DIGITS,
    ).verify


def test_pairing_attempt_pairing_psk_requires_material() -> None:
    """PAIRING_PSK must carry a 32-byte pairing_psk and no pairing-code flow hooks."""
    with pytest.raises(ValueError, match="requires pairing_psk"):
        PairingAttempt(method=PairMethod.PAIRING_PSK)
    with pytest.raises(ValueError, match="must be 32 bytes"):
        PairingAttempt(method=PairMethod.PAIRING_PSK, pairing_psk=b"\x01" * 16)
    with pytest.raises(ValueError, match="does not use code pairing fields"):
        PairingAttempt(
            method=PairMethod.PAIRING_PSK, pairing_psk=generate_psk(), pairing_code_provider=_code
        )
    with pytest.raises(ValueError, match="does not use on_pair_pending"):
        PairingAttempt(
            method=PairMethod.PAIRING_PSK,
            pairing_psk=generate_psk(),
            on_pair_pending=lambda: None,
        )


@pytest.mark.parametrize(
    "method", [PairMethod.DYNAMIC_PAIRING_CODE, PairMethod.STATIC_PAIRING_CODE]
)
def test_pairing_attempt_code_methods_require_pairing_code_provider(method: PairMethod) -> None:
    """pairing-code methods must carry a pairing_code_provider and must not carry a pairing_psk."""
    with pytest.raises(ValueError, match="requires pairing_code_provider"):
        PairingAttempt(method=method)
    with pytest.raises(ValueError, match="does not use pairing_psk"):
        PairingAttempt(method=method, pairing_code_provider=_code, pairing_psk=generate_psk())


def test_pairing_attempt_pairing_format_is_dynamic_only() -> None:
    """The emission format is required for dynamic pairing code and rejected for the rest."""
    with pytest.raises(ValueError, match="requires pairing_format"):
        PairingAttempt(method=PairMethod.DYNAMIC_PAIRING_CODE, pairing_code_provider=_code)
    with pytest.raises(ValueError, match="does not use pairing_format"):
        PairingAttempt(
            method=PairMethod.STATIC_PAIRING_CODE,
            pairing_code_provider=_code,
            pairing_format=PairingCodeFormat.DIGITS,
        )
    with pytest.raises(ValueError, match="does not use code pairing fields"):
        PairingAttempt(
            method=PairMethod.PAIRING_PSK,
            pairing_psk=generate_psk(),
            pairing_format=PairingCodeFormat.DIGITS,
        )


_paired_encrypted_ws = make_paired_encrypted_ws


async def test_pairing_psk_finalize_round_trip() -> None:
    """Both sides persist matching records carrying the client-generated long-term PSK."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()
    server_store = InMemoryServerPairingStore()

    _client_ret, server_record = await asyncio.gather(
        run_pairing_psk_client(
            client_ews,
            pairing_index=0,
            server_id="server-X",
            store=client_store,
        ),
        run_pairing_psk_server(
            server_ews, pairing_index=0, client_id="client-A", store=server_store
        ),
    )

    client_record = await client_store.record_by_server_id("server-X")
    assert client_record is not None
    # The same long-term PSK is recorded on both sides.
    assert client_record.psk == server_record.psk
    assert client_record.psk_id == server_record.psk_id
    # Directional counterparties.
    assert client_record.server_id == "server-X"
    assert server_record.client_id == "client-A"
    # The server persisted its record too.
    assert await server_store.record_by_client_id("client-A") == server_record
    # A first pairing records the establishing method.
    assert server_record.pair_methods == [PairMethod.PAIRING_PSK]


async def test_pairing_psk_client_sends_pair_init_then_finalize() -> None:
    """The client starts the attempt with its indexed pair-init, then finalizes unprompted."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()

    async def server() -> None:
        init = ClientPairInitMessage.from_json((await server_ews.receive()).data)
        assert init.payload == ClientPairInitPayload(pairing_index=3)
        assert "commit_B" not in init.to_dict()["payload"]
        finalize = ClientPairFinalizeMessage.from_json((await server_ews.receive()).data)
        assert finalize.payload.long_term_psk is not None
        await server_ews.send_str(ServerPairFinalizeMessage().to_json())

    client_ret, _ = await asyncio.gather(
        run_pairing_psk_client(
            client_ews, pairing_index=3, server_id="server-X", store=client_store
        ),
        server(),
    )
    assert client_ret is None
    assert await client_store.record_by_server_id("server-X") is not None


async def test_pairing_psk_client_times_out_without_finalize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Pairing PSK client aborts with attempt_timeout if the server never finalizes."""
    monkeypatch.setattr("aiosendspin.noise.pairing._CLIENT_ATTEMPT_TIMEOUT_S", 0.05)
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()

    async def silent_server() -> None:
        await server_ews.receive()  # consume client/pair-init
        await server_ews.receive()  # consume client/pair-finalize, then never reply
        await asyncio.sleep(0.5)

    with pytest.raises(PairingAbortError) as excinfo:
        await asyncio.gather(
            run_pairing_psk_client(
                client_ews, pairing_index=0, server_id="server-X", store=client_store
            ),
            silent_server(),
        )
    assert excinfo.value.reason is PairAbortReason.ATTEMPT_TIMEOUT
    assert await client_store.record_by_server_id("server-X") is None


async def test_pairing_psk_server_times_out_without_pair_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Pairing PSK server times out locally, sending nothing, without a client/pair-init."""
    monkeypatch.setattr("aiosendspin.noise.pairing.SERVER_FIRST_MESSAGE_TIMEOUT_S", 0.05)
    _client_ews, server_ews, _client_raw, server_raw = _paired_encrypted_ws()
    server_store = InMemoryServerPairingStore()

    with pytest.raises(PairingTimeoutError, match="client/pair-init did not arrive"):
        await run_pairing_psk_server(
            server_ews, pairing_index=0, client_id="client-X", store=server_store
        )
    assert server_raw.sent == []
    assert await server_store.record_by_client_id("client-X") is None


async def test_static_pairing_code_server_first_message_wait_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The static-pairing-code server times out awaiting the client's first message."""
    monkeypatch.setattr("aiosendspin.noise.pairing.SERVER_FIRST_MESSAGE_TIMEOUT_S", 0.05)
    _client_ews, server_ews, _client_raw, server_raw = _paired_encrypted_ws()
    server_store = InMemoryServerPairingStore()

    with pytest.raises(PairingTimeoutError):
        await run_static_pairing_code_server(
            server_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=0,
            pairing_code_provider=_code,
            client_id="client-X",
            store=server_store,
        )
    assert server_raw.sent == []


async def test_pair_pending_extends_the_first_message_wait() -> None:
    """A matching pair-pending switches the server to the gesture timeout; pairing completes."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()
    server_store = InMemoryServerPairingStore()
    pending_signals = 0

    def on_pending() -> None:
        nonlocal pending_signals
        pending_signals += 1

    async def gated_client() -> None:
        await client_ews.send_str(
            ClientPairPendingMessage(payload=ClientPairPendingPayload(pairing_index=0)).to_json()
        )
        await run_static_pairing_code_client(
            client_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=0,
            static_pairing_code=_STATIC_PAIRING_CODE,
            server_id="server-X",
            store=client_store,
        )

    async def provide() -> str:
        return _STATIC_PAIRING_CODE

    _client_ret, server_record = await asyncio.gather(
        gated_client(),
        run_static_pairing_code_server(
            server_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=0,
            pairing_code_provider=provide,
            client_id="client-A",
            store=server_store,
            on_pair_pending=on_pending,
        ),
    )
    assert server_record is not None
    assert await server_store.record_by_client_id("client-A") == server_record
    assert pending_signals == 1


async def test_gesture_wait_times_out_after_pair_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After pair-pending the gesture bound applies; its expiry raises locally."""
    monkeypatch.setattr("aiosendspin.noise.pairing.SERVER_GESTURE_TIMEOUT_S", 0.05)
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    server_store = InMemoryServerPairingStore()

    await client_ews.send_str(
        ClientPairPendingMessage(payload=ClientPairPendingPayload(pairing_index=0)).to_json()
    )
    with pytest.raises(PairingTimeoutError):
        await run_static_pairing_code_server(
            server_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=0,
            pairing_code_provider=_code,
            client_id="client-X",
            store=server_store,
        )


async def test_stale_pair_pending_is_discarded() -> None:
    """A pair-pending left over from a superseded activate is ignored; the fresh init pairs."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()
    server_store = InMemoryServerPairingStore()
    pending_signals = 0

    def on_pending() -> None:
        nonlocal pending_signals
        pending_signals += 1

    await client_ews.send_str(
        ClientPairPendingMessage(payload=ClientPairPendingPayload(pairing_index=0)).to_json()
    )

    async def provide() -> str:
        return _STATIC_PAIRING_CODE

    _client_ret, server_record = await asyncio.gather(
        run_static_pairing_code_client(
            client_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=1,
            static_pairing_code=_STATIC_PAIRING_CODE,
            server_id="server-X",
            store=client_store,
        ),
        run_static_pairing_code_server(
            server_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=1,
            pairing_code_provider=provide,
            client_id="client-A",
            store=server_store,
            on_pair_pending=on_pending,
        ),
    )
    assert server_record is not None
    assert pending_signals == 0  # the stale pending is not surfaced


async def test_repeated_pair_pending_is_protocol_error() -> None:
    """A second matching pair-pending within one attempt is a protocol error."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    server_store = InMemoryServerPairingStore()

    for _ in range(2):
        await client_ews.send_str(
            ClientPairPendingMessage(payload=ClientPairPendingPayload(pairing_index=0)).to_json()
        )
    with pytest.raises(PairingError, match="expected ClientPairInitMessage"):
        await run_static_pairing_code_server(
            server_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=0,
            pairing_code_provider=_code,
            client_id="client-A",
            store=server_store,
        )


async def test_pair_init_index_mismatch_after_pending_is_protocol_error() -> None:
    """After the matching pair-pending, an init for a different attempt is a protocol error."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    server_store = InMemoryServerPairingStore()

    await client_ews.send_str(
        ClientPairPendingMessage(payload=ClientPairPendingPayload(pairing_index=0)).to_json()
    )
    await client_ews.send_str(
        ClientPairInitMessage(payload=ClientPairInitPayload(pairing_index=1)).to_json()
    )
    with pytest.raises(PairingError, match="does not match the attempt"):
        await run_static_pairing_code_server(
            server_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=0,
            pairing_code_provider=_code,
            client_id="client-A",
            store=server_store,
        )


async def test_pair_pending_ahead_of_server_count_is_protocol_error() -> None:
    """A pair-pending with a pairing_index the server has not reached yet is a protocol error."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    server_store = InMemoryServerPairingStore()

    await client_ews.send_str(
        ClientPairPendingMessage(payload=ClientPairPendingPayload(pairing_index=1)).to_json()
    )
    with pytest.raises(PairingError, match="ahead of the server's count"):
        await run_static_pairing_code_server(
            server_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=0,
            pairing_code_provider=_code,
            client_id="client-A",
            store=server_store,
        )


async def test_static_pairing_code_server_rejects_non_8_digit_operator_code() -> None:
    """A non-8-digit operator pairing code aborts the server before it emits its PAKE share."""
    client_ews, server_ews, _client_raw, server_raw = _paired_encrypted_ws()
    server_store = InMemoryServerPairingStore()

    async def bad_code() -> str:
        return "12345"

    await client_ews.send_str(
        ClientPairInitMessage(payload=ClientPairInitPayload(pairing_index=0)).to_json(),
    )
    with pytest.raises(InvalidPairingCodeError, match="8 decimal digits"):
        await run_static_pairing_code_server(
            server_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=0,
            pairing_code_provider=bad_code,
            client_id="client-X",
            store=server_store,
        )
    assert server_raw.sent == []
    assert await server_store.record_by_client_id("client-X") is None


@pytest.mark.parametrize(
    ("pairing_format", "entered"),
    [
        pytest.param(PairingCodeFormat.DIGITS, "12345", id="short-digits"),
        pytest.param(PairingCodeFormat.DIGITS, "12a456", id="non-digit"),
        pytest.param(PairingCodeFormat.DIGITS, "123_456", id="unknown-separator"),
        pytest.param(PairingCodeFormat.QR_CODE, "SP:0AAAA", id="wrong-token-version"),
    ],
)
async def test_dynamic_pairing_code_server_rejects_malformed_operator_input(
    pairing_format: PairingCodeFormat, entered: str
) -> None:
    """Malformed operator input raises before the server emits its PAKE share."""
    client_ews, server_ews, _client_raw, server_raw = _paired_encrypted_ws()
    server_store = InMemoryServerPairingStore()

    async def bad_code() -> str:
        return entered

    await client_ews.send_str(
        ClientPairInitMessage(
            payload=ClientPairInitPayload(
                pairing_index=0,
                commit_B=b64url_encode(pairing_code_mod.commit(pairing_code_mod.generate_nonce())),
            ),
        ).to_json(),
    )
    with pytest.raises(InvalidPairingCodeError):
        await run_dynamic_pairing_code_server(
            server_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=0,
            pairing_code_provider=bad_code,
            pairing_format=pairing_format,
            client_id="client-X",
            store=server_store,
        )
    assert len(server_raw.sent) == 1  # server/pair-init only
    assert await server_store.record_by_client_id("client-X") is None


async def test_static_pairing_code_server_rejects_dynamic_only_commit_b() -> None:
    """A static-pairing-code pair-init carrying commit_B is a protocol error."""
    client_ews, server_ews, _client_raw, server_raw = _paired_encrypted_ws()
    server_store = InMemoryServerPairingStore()

    await client_ews.send_str(
        ClientPairInitMessage(
            payload=ClientPairInitPayload(
                pairing_index=0,
                commit_B=b64url_encode(pairing_code_mod.commit(pairing_code_mod.generate_nonce())),
            )
        ).to_json(),
    )
    with pytest.raises(PairingError, match="commit_B for static pairing code"):
        await run_static_pairing_code_server(
            server_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=0,
            pairing_code_provider=_code,
            client_id="client-X",
            store=server_store,
        )
    assert server_raw.sent == []
    assert await server_store.record_by_client_id("client-X") is None


async def test_dynamic_pairing_code_server_times_out_mid_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dynamic server times out locally, with no abort on the wire, if the client stalls."""
    monkeypatch.setattr("aiosendspin.noise.pairing.SERVER_ATTEMPT_TIMEOUT_S", 0.05)
    client_ews, server_ews, client_raw, _server_raw = _paired_encrypted_ws()
    server_store = InMemoryServerPairingStore()

    await client_ews.send_str(
        ClientPairInitMessage(
            payload=ClientPairInitPayload(
                pairing_index=0,
                commit_B=b64url_encode(pairing_code_mod.commit(pairing_code_mod.generate_nonce())),
            ),
        ).to_json(),
    )
    with pytest.raises(PairingTimeoutError):
        await run_dynamic_pairing_code_server(
            server_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=0,
            pairing_code_provider=_code,
            pairing_format=PairingCodeFormat.DIGITS,
            client_id="client-X",
            store=server_store,
        )
    await client_ews.receive()  # server/pair-init
    await client_ews.receive()  # server/pair-auth
    assert client_raw.incoming.qsize() == 0


async def test_finalize_rotate_preserves_birth_and_appends_method() -> None:
    """Re-pairing rotates the PSK but carries over created_at and appends the method."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()
    server_store = InMemoryServerPairingStore()
    born = datetime(2020, 1, 1, tzinfo=UTC)
    seeded = ServerPairingRecord(
        psk_id="old",
        psk=generate_psk(),
        client_id="client-A",
        created_at=born,
        pair_methods=[PairMethod.DYNAMIC_PAIRING_CODE],
    )
    await server_store.store_record(seeded)

    _client_ret, rotated = await asyncio.gather(
        run_pairing_psk_client(
            client_ews, pairing_index=0, server_id="server-X", store=client_store
        ),
        run_pairing_psk_server(
            server_ews, pairing_index=0, client_id="client-A", store=server_store
        ),
    )

    assert rotated.psk != seeded.psk  # rotated onto a fresh PSK
    assert rotated.created_at == born  # birth time carried over
    assert rotated.pair_methods == [PairMethod.DYNAMIC_PAIRING_CODE, PairMethod.PAIRING_PSK]


async def test_finalize_stamps_owner_on_a_fresh_record() -> None:
    """A pairing run with an owner binds the persisted record to that owner."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()
    server_store = InMemoryServerPairingStore()

    _client_ret, record = await asyncio.gather(
        run_pairing_psk_client(
            client_ews, pairing_index=0, server_id="server-X", store=client_store
        ),
        run_pairing_psk_server(
            server_ews, pairing_index=0, client_id="client-A", store=server_store, owner="user-1"
        ),
    )

    assert record.owner == "user-1"
    assert await server_store.record_by_client_id("client-A") == record


async def test_finalize_rotate_restamps_owner() -> None:
    """Re-pairing re-stamps ownership from the new attempt, superseding the old owner."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()
    server_store = InMemoryServerPairingStore()
    seeded = ServerPairingRecord(
        psk_id="old",
        psk=generate_psk(),
        client_id="client-A",
        pair_methods=[PairMethod.PAIRING_PSK],
        owner="user-1",
    )
    await server_store.store_record(seeded)

    _client_ret, rotated = await asyncio.gather(
        run_pairing_psk_client(
            client_ews, pairing_index=0, server_id="server-X", store=client_store
        ),
        run_pairing_psk_server(
            server_ews, pairing_index=0, client_id="client-A", store=server_store
        ),
    )

    assert rotated.owner is None  # an unowned re-pair promotes the record to durable


async def test_client_finalize_raises_if_server_closes_before_ack() -> None:
    """If the server closes before sending server/pair-finalize, the client raises."""
    client_ews, _server_ews, _client_raw, server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()

    await server_raw.close_outbound()  # server never acks
    with pytest.raises(PairingError, match="closed while awaiting ServerPairFinalizeMessage"):
        await run_pairing_psk_client(
            client_ews,
            pairing_index=0,
            server_id="server-X",
            store=client_store,
        )
    # Nothing persisted on failure (only the pre-provisioned shared record remains).
    assert _added_records(await client_store.list_records()) == []


async def test_server_finalize_raises_if_client_closes_first() -> None:
    """If the client closes before starting the attempt, the server raises."""
    _client_ews, server_ews, client_raw, _server_raw = _paired_encrypted_ws()
    server_store = InMemoryServerPairingStore()

    await client_raw.close_outbound()  # client never sends client/pair-init
    with pytest.raises(PairingError, match="closed while awaiting ClientPairInitMessage"):
        await run_pairing_psk_server(
            server_ews, pairing_index=0, client_id="client-A", store=server_store
        )
    assert await server_store.record_by_client_id("client-A") is None


def _psk_finalize(psk: bytes) -> str:
    return ClientPairFinalizeMessage(
        payload=ClientPairFinalizePayload(long_term_psk=b64url_encode(psk))
    ).to_json()


def _unexpected_legacy_finalize() -> None:
    pytest.fail("a leftover finalize was accepted as a legacy attempt")


async def test_pairing_psk_server_discards_a_cancelled_attempts_messages() -> None:
    """A late init and finalize from a cancelled attempt do not finalize the next attempt."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()
    server_store = InMemoryServerPairingStore()
    stale_psk = generate_psk()
    pair_inits = 0

    def on_pair_init() -> None:
        nonlocal pair_inits
        pair_inits += 1

    await client_ews.send_str(
        ClientPairInitMessage(payload=ClientPairInitPayload(pairing_index=1)).to_json()
    )
    await client_ews.send_str(_psk_finalize(stale_psk))

    _client_ret, server_record = await asyncio.gather(
        run_pairing_psk_client(
            client_ews, pairing_index=2, server_id="server-X", store=client_store
        ),
        run_pairing_psk_server(
            server_ews,
            pairing_index=2,
            client_id="client-A",
            store=server_store,
            on_pair_init=on_pair_init,
            on_legacy_finalize=_unexpected_legacy_finalize,
        ),
    )

    client_record = await client_store.record_by_server_id("server-X")
    assert client_record is not None
    assert server_record.psk == client_record.psk
    assert server_record.psk != stale_psk
    assert pair_inits == 2


async def test_pairing_psk_server_discards_a_leading_finalize_without_the_legacy_hook() -> None:
    """Without ``on_legacy_finalize``, a finalize ahead of the matching init is a leftover."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()
    server_store = InMemoryServerPairingStore()
    stale_psk = generate_psk()

    await client_ews.send_str(_psk_finalize(stale_psk))

    _client_ret, server_record = await asyncio.gather(
        run_pairing_psk_client(
            client_ews, pairing_index=1, server_id="server-X", store=client_store
        ),
        run_pairing_psk_server(
            server_ews, pairing_index=1, client_id="client-A", store=server_store
        ),
    )

    client_record = await client_store.record_by_server_id("server-X")
    assert client_record is not None
    assert server_record.psk == client_record.psk != stale_psk


@pytest.mark.parametrize("long_term_psk", [None, b64url_encode(bytes(32))])
async def test_pairing_psk_server_always_discards_a_wrapped_finalize(
    long_term_psk: str | None,
) -> None:
    """A finalize carrying ``wrapped_psk`` is a leftover, even with the legacy hook set."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()
    server_store = InMemoryServerPairingStore()

    await client_ews.send_str(
        ClientPairFinalizeMessage(
            payload=ClientPairFinalizePayload(
                long_term_psk=long_term_psk, wrapped_psk=b64url_encode(bytes(48))
            )
        ).to_json()
    )

    _client_ret, server_record = await asyncio.gather(
        run_pairing_psk_client(
            client_ews, pairing_index=1, server_id="server-X", store=client_store
        ),
        run_pairing_psk_server(
            server_ews,
            pairing_index=1,
            client_id="client-A",
            store=server_store,
            on_legacy_finalize=_unexpected_legacy_finalize,
        ),
    )

    client_record = await client_store.record_by_server_id("server-X")
    assert client_record is not None
    assert server_record.psk == client_record.psk


async def test_pairing_psk_finalize_with_both_psk_fields_is_protocol_error() -> None:
    """A Pairing PSK finalize carrying ``wrapped_psk`` too persists nothing."""
    client_ews, server_ews, _client_raw, server_raw = _paired_encrypted_ws()
    server_store = InMemoryServerPairingStore()

    await client_ews.send_str(
        ClientPairInitMessage(payload=ClientPairInitPayload(pairing_index=1)).to_json()
    )
    await client_ews.send_str(
        ClientPairFinalizeMessage(
            payload=ClientPairFinalizePayload(
                long_term_psk=b64url_encode(generate_psk()), wrapped_psk=b64url_encode(bytes(48))
            )
        ).to_json()
    )
    with pytest.raises(PairingError, match="both long_term_psk and wrapped_psk") as excinfo:
        await run_pairing_psk_server(
            server_ews, pairing_index=1, client_id="client-A", store=server_store
        )

    assert not isinstance(excinfo.value, PairingAbortError)
    assert await server_store.record_by_client_id("client-A") is None
    assert server_raw.sent == []


# DEPRECATED(spec-pr-247): remove in aiosendspin <version>
async def test_pairing_psk_server_accepts_a_legacy_finalize_first_attempt() -> None:
    """With ``on_legacy_finalize`` set, a finalize before any init is this attempt's."""
    client_ews, server_ews, _client_raw, server_raw = _paired_encrypted_ws()
    server_store = InMemoryServerPairingStore()
    psk = generate_psk()
    legacy_calls = 0

    def on_legacy_finalize() -> None:
        nonlocal legacy_calls
        legacy_calls += 1

    await client_ews.send_str(_psk_finalize(psk))
    record = await run_pairing_psk_server(
        server_ews,
        pairing_index=1,
        client_id="client-A",
        store=server_store,
        on_legacy_finalize=on_legacy_finalize,
    )

    assert record.psk == psk
    assert legacy_calls == 1
    assert await server_store.record_by_client_id("client-A") == record
    assert len(server_raw.sent) == 1  # server/pair-finalize


# DEPRECATED(spec-pr-247): remove in aiosendspin <version>
async def test_pairing_psk_server_legacy_hook_rejection_stores_nothing() -> None:
    """An ``on_legacy_finalize`` that raises ends the attempt before anything is persisted."""
    client_ews, server_ews, _client_raw, server_raw = _paired_encrypted_ws()
    server_store = InMemoryServerPairingStore()

    class RejectedError(Exception):
        pass

    def reject() -> None:
        raise RejectedError

    await client_ews.send_str(_psk_finalize(generate_psk()))
    with pytest.raises(RejectedError):
        await run_pairing_psk_server(
            server_ews,
            pairing_index=1,
            client_id="client-A",
            store=server_store,
            on_legacy_finalize=reject,
        )
    assert server_raw.sent == []
    assert await server_store.record_by_client_id("client-A") is None


@pytest.mark.parametrize(
    ("message", "error"),
    [
        (
            ClientPairInitMessage(payload=ClientPairInitPayload(pairing_index=2)),
            "ClientPairInitMessage pairing_index is ahead of the server's count",
        ),
        (
            ClientPairPendingMessage(payload=ClientPairPendingPayload(pairing_index=2)),
            "ClientPairPendingMessage pairing_index is ahead of the server's count",
        ),
        (
            ClientPairPendingMessage(payload=ClientPairPendingPayload(pairing_index=1)),
            "client/pair-pending is not part of the Pairing PSK flow",
        ),
        (
            ClientPairInitMessage(
                payload=ClientPairInitPayload(pairing_index=1, commit_B=b64url_encode(bytes(32)))
            ),
            "client/pair-init carries commit_B for Pairing PSK",
        ),
    ],
)
async def test_pairing_psk_server_rejects_out_of_sequence_messages(
    message: ClientPairInitMessage | ClientPairPendingMessage, error: str
) -> None:
    """A higher index, a matching pair-pending, or a commit_B is a protocol error."""
    client_ews, server_ews, _client_raw, server_raw = _paired_encrypted_ws()
    server_store = InMemoryServerPairingStore()

    await client_ews.send_str(message.to_json())
    with pytest.raises(PairingError, match=error):
        await run_pairing_psk_server(
            server_ews, pairing_index=1, client_id="client-A", store=server_store
        )
    assert server_raw.sent == []
    assert await server_store.record_by_client_id("client-A") is None


async def test_pairing_psk_server_discards_a_stale_pair_pending() -> None:
    """A pair-pending from a superseded activate is discarded; the matching init pairs."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()
    server_store = InMemoryServerPairingStore()

    await client_ews.send_str(
        ClientPairPendingMessage(payload=ClientPairPendingPayload(pairing_index=1)).to_json()
    )

    _client_ret, server_record = await asyncio.gather(
        run_pairing_psk_client(
            client_ews, pairing_index=2, server_id="server-X", store=client_store
        ),
        run_pairing_psk_server(
            server_ews, pairing_index=2, client_id="client-A", store=server_store
        ),
    )

    client_record = await client_store.record_by_server_id("server-X")
    assert client_record is not None
    assert server_record.psk == client_record.psk


_HANDSHAKE_HASH = bytes(range(32))


@pytest.mark.parametrize("separator", ["", "-", " "])
async def test_dynamic_pairing_code_round_trip(separator: str) -> None:
    """A matching pairing code authenticates the PAKE and both sides persist the record.

    Separators the operator types between the digit groups are ignored.
    """
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()
    server_store = InMemoryServerPairingStore()
    shown: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def emit(pairing_code: str) -> None:
        shown.set_result(pairing_code)

    async def provide() -> str:
        pairing_code = await shown  # operator types the pairing code the client displayed
        return pairing_code[:3] + separator + pairing_code[3:]

    _client_ret, server_record = await asyncio.gather(
        run_dynamic_pairing_code_client(
            client_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=0,
            pairing_format=PairingCodeFormat.DIGITS,
            pairing_code_emitter=emit,
            server_id="server-X",
            store=client_store,
        ),
        run_dynamic_pairing_code_server(
            server_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=0,
            pairing_format=PairingCodeFormat.DIGITS,
            pairing_code_provider=provide,
            client_id="client-A",
            store=server_store,
        ),
    )

    assert server_record is not None  # a finalized pairing returns a record
    client_record = await client_store.record_by_server_id("server-X")
    assert client_record is not None
    assert client_record.psk == server_record.psk
    assert client_record.psk_id == server_record.psk_id
    assert client_record.server_id == "server-X"
    assert server_record.client_id == "client-A"
    assert await server_store.record_by_client_id("client-A") == server_record


async def test_dynamic_pairing_code_qr_round_trip() -> None:
    """A scanned SP:1 token authenticates the PAKE and both sides persist the record."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()
    server_store = InMemoryServerPairingStore()
    shown: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def emit(pairing_code: str) -> None:
        shown.set_result(pairing_code)

    async def provide() -> str:
        return await shown  # operator scans the token the client rendered

    _client_ret, server_record = await asyncio.gather(
        run_dynamic_pairing_code_client(
            client_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=0,
            pairing_format=PairingCodeFormat.QR_CODE,
            pairing_code_emitter=emit,
            server_id="server-X",
            store=client_store,
        ),
        run_dynamic_pairing_code_server(
            server_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=0,
            pairing_format=PairingCodeFormat.QR_CODE,
            pairing_code_provider=provide,
            client_id="client-A",
            store=server_store,
        ),
    )

    assert shown.result().startswith("SP:1")
    assert server_record is not None
    client_record = await client_store.record_by_server_id("server-X")
    assert client_record is not None
    assert client_record.psk == server_record.psk


async def test_code_server_discards_a_superseded_attempts_pair_retry() -> None:
    """A client/pair-retry still in flight from a superseded attempt does not fail the next one."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()
    server_store = InMemoryServerPairingStore()
    shown: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def emit(pairing_code: str) -> None:
        shown.set_result(pairing_code)

    async def provide() -> str:
        return await shown

    await client_ews.send_str(ClientPairRetryMessage().to_json())

    _client_ret, server_record = await asyncio.gather(
        run_dynamic_pairing_code_client(
            client_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=1,
            pairing_format=PairingCodeFormat.DIGITS,
            pairing_code_emitter=emit,
            server_id="server-X",
            store=client_store,
        ),
        run_dynamic_pairing_code_server(
            server_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=1,
            pairing_format=PairingCodeFormat.DIGITS,
            pairing_code_provider=provide,
            client_id="client-A",
            store=server_store,
        ),
    )
    assert server_record is not None
    assert await server_store.record_by_client_id("client-A") == server_record


async def test_pairing_psk_server_discards_a_superseded_attempts_pair_retry() -> None:
    """A client/pair-retry in flight from a dynamic attempt does not fail a Pairing PSK attempt."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()
    await client_ews.send_str(ClientPairRetryMessage().to_json())

    _client_ret, server_record = await asyncio.gather(
        run_pairing_psk_client(
            client_ews, pairing_index=1, server_id="server-X", store=client_store
        ),
        run_pairing_psk_server(
            server_ews,
            pairing_index=1,
            client_id="client-A",
            store=InMemoryServerPairingStore(),
            on_legacy_finalize=_unexpected_legacy_finalize,
        ),
    )
    client_record = await client_store.record_by_server_id("server-X")
    assert client_record is not None
    assert server_record.psk == client_record.psk


async def test_dynamic_pairing_code_server_discards_stale_pair_init() -> None:
    """A pair-init left over from a superseded activate is discarded; the fresh one pairs."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()
    server_store = InMemoryServerPairingStore()
    shown: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def emit(pairing_code: str) -> None:
        shown.set_result(pairing_code)

    async def provide() -> str:
        return await shown

    stale = ClientPairInitMessage(
        payload=ClientPairInitPayload(
            pairing_index=0,
            commit_B=b64url_encode(pairing_code_mod.commit(pairing_code_mod.generate_nonce())),
        ),
    )
    await client_ews.send_str(stale.to_json())

    _client_ret, server_record = await asyncio.gather(
        run_dynamic_pairing_code_client(
            client_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=1,
            pairing_format=PairingCodeFormat.DIGITS,
            pairing_code_emitter=emit,
            server_id="server-X",
            store=client_store,
        ),
        run_dynamic_pairing_code_server(
            server_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=1,
            pairing_format=PairingCodeFormat.DIGITS,
            pairing_code_provider=provide,
            client_id="client-A",
            store=server_store,
        ),
    )
    assert server_record is not None
    assert await server_store.record_by_client_id("client-A") == server_record


async def test_pair_init_ahead_of_server_count_is_protocol_error() -> None:
    """A pair-init with a pairing_index the server has not reached yet is a protocol error."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    server_store = InMemoryServerPairingStore()

    await client_ews.send_str(
        ClientPairInitMessage(payload=ClientPairInitPayload(pairing_index=1)).to_json()
    )
    with pytest.raises(PairingError, match="ahead of the server's count"):
        await run_static_pairing_code_server(
            server_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=0,
            pairing_code_provider=_code,
            client_id="client-A",
            store=server_store,
        )


def test_pake_sid_known_answer() -> None:
    """The sid is label || h || u32be(pairing_index) || u32be(round)."""
    prefix = (
        "73656e647370696e2d706169722d70616b652d7631"
        "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"
        "00000002"
    )
    assert _pake_sid(_HANDSHAKE_HASH, 2, 1).hex() == prefix + "00000001"
    assert _pake_sid(_HANDSHAKE_HASH, 2, 3).hex() == prefix + "00000003"


# DEPRECATED(spec-pr-237): remove in aiosendspin <version>
def test_legacy_pake_sid_known_answer() -> None:
    """The sid for a client predating rounds carries no round number."""
    assert _legacy_pake_sid(_HANDSHAKE_HASH, 2).hex() == (
        "73656e647370696e2d706169722d70616b652d7631"
        "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"
        "00000002"
    )


class _RecordingWS:
    """Wraps an ``EncryptedWebSocket``, recording the pairing messages sent through it."""

    def __init__(self, ws: EncryptedWebSocket) -> None:
        self._ws = ws
        self.session = ws.session
        self.sent: list[str] = []

    async def send_str(self, data: str) -> None:
        self.sent.append(data)
        await self._ws.send_str(data)

    async def receive(self) -> object:
        return await self._ws.receive()

    def types(self) -> list[str]:
        return [json.loads(frame)["type"] for frame in self.sent]


async def test_dynamic_pairing_code_retries_a_wrong_round() -> None:
    """A wrong code fails one round; the next round runs on the same code and pairs."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_rec, server_rec = _RecordingWS(client_ews), _RecordingWS(server_ews)
    client_store = InMemoryClientPairingStore()
    server_store = InMemoryServerPairingStore()
    shown: asyncio.Queue[str] = asyncio.Queue()
    entered: list[str] = []

    async def emit(pairing_code: str) -> None:
        shown.put_nowait(pairing_code)

    async def provide() -> str:
        pairing_code = await shown.get()
        if not entered:
            pairing_code = ("2" if pairing_code[0] == "1" else "1") + pairing_code[1:]
        entered.append(pairing_code)
        return pairing_code

    _client_ret, server_record = await asyncio.gather(
        run_dynamic_pairing_code_client(
            client_rec,  # type: ignore[arg-type]
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=0,
            pairing_format=PairingCodeFormat.DIGITS,
            pairing_code_emitter=emit,
            server_id="server-X",
            store=client_store,
        ),
        run_dynamic_pairing_code_server(
            server_rec,  # type: ignore[arg-type]
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=0,
            pairing_format=PairingCodeFormat.DIGITS,
            pairing_code_provider=provide,
            client_id="client-A",
            store=server_store,
        ),
    )

    assert server_record is not None
    client_record = await client_store.record_by_server_id("server-X")
    assert client_record is not None
    assert client_record.psk == server_record.psk
    assert len(entered) == 2
    assert entered[0] != entered[1]
    assert client_rec.types() == [
        "client/pair-init",
        "client/pair-auth",
        "client/pair-retry",
        "client/pair-auth",
        "client/pair-confirm",
        "client/pair-finalize",
    ]
    inits = [
        ServerPairInitMessage.from_json(frame)
        for frame in server_rec.sent
        if json.loads(frame)["type"] == "server/pair-init"
    ]
    assert len(inits) == 2
    assert inits[0].payload.nonce_A is not None
    assert inits[1].payload.nonce_A is None
    assert await client_store.pairing_round_count() == 0


async def test_dynamic_pairing_code_aborts_at_round_limit_and_persists_nothing() -> None:
    """A code that never matches runs rounds up to the limit, then the client aborts."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_rec = _RecordingWS(client_ews)
    client_store = InMemoryClientPairingStore()
    server_store = InMemoryServerPairingStore()
    shown: asyncio.Queue[str] = asyncio.Queue()
    emitted: list[str] = []

    async def emit(pairing_code: str) -> None:
        emitted.append(pairing_code)
        shown.put_nowait(pairing_code)

    async def provide_wrong() -> str:
        pairing_code = await shown.get()
        return ("2" if pairing_code[0] == "1" else "1") + pairing_code[1:]

    with pytest.raises(PairingAbortError) as excinfo:
        await asyncio.gather(
            run_dynamic_pairing_code_client(
                client_rec,  # type: ignore[arg-type]
                handshake_hash=_HANDSHAKE_HASH,
                pairing_index=0,
                pairing_format=PairingCodeFormat.DIGITS,
                pairing_code_emitter=emit,
                server_id="server-X",
                store=client_store,
            ),
            run_dynamic_pairing_code_server(
                server_ews,
                handshake_hash=_HANDSHAKE_HASH,
                pairing_index=0,
                pairing_format=PairingCodeFormat.DIGITS,
                pairing_code_provider=provide_wrong,
                client_id="client-A",
                store=server_store,
            ),
        )

    assert excinfo.value.reason is PairAbortReason.PAIRING_CODE_MISMATCH
    assert len(emitted) == PAIRING_ROUND_LIMIT
    assert len(set(emitted)) == 1  # the code is stable across rounds
    assert client_rec.types().count("client/pair-retry") == PAIRING_ROUND_LIMIT - 1
    assert client_rec.types()[-1] == "pair/abort"
    assert await client_store.pairing_round_count() == PAIRING_ROUND_LIMIT
    assert _added_records(await client_store.list_records()) == []
    assert await server_store.record_by_client_id("client-A") is None


async def test_dynamic_pairing_code_client_aborts_early_when_rounds_carry_over() -> None:
    """Rounds from earlier attempts count: one round short of the limit leaves no retry."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_rec = _RecordingWS(client_ews)
    client_store = InMemoryClientPairingStore()
    for _ in range(PAIRING_ROUND_LIMIT - 1):
        await client_store.record_pairing_round()
    shown: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def emit(pairing_code: str) -> None:
        shown.set_result(pairing_code)

    async def provide_wrong() -> str:
        pairing_code = await shown
        return ("2" if pairing_code[0] == "1" else "1") + pairing_code[1:]

    with pytest.raises(PairingAbortError):
        await asyncio.gather(
            run_dynamic_pairing_code_client(
                client_rec,  # type: ignore[arg-type]
                handshake_hash=_HANDSHAKE_HASH,
                pairing_index=0,
                pairing_format=PairingCodeFormat.DIGITS,
                pairing_code_emitter=emit,
                server_id="server-X",
                store=client_store,
            ),
            run_dynamic_pairing_code_server(
                server_ews,
                handshake_hash=_HANDSHAKE_HASH,
                pairing_index=0,
                pairing_format=PairingCodeFormat.DIGITS,
                pairing_code_provider=provide_wrong,
                client_id="client-A",
                store=InMemoryServerPairingStore(),
            ),
        )

    assert "client/pair-retry" not in client_rec.types()
    assert await client_store.pairing_round_count() == PAIRING_ROUND_LIMIT


async def test_dynamic_pairing_code_counts_a_round_once_emitted() -> None:
    """A round counts as soon as its code is emitted, even if the attempt ends there."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()

    async def emit(_pairing_code: str) -> None:
        raise asyncio.CancelledError

    await server_ews.send_str(
        ServerPairInitMessage(
            payload=ServerPairInitPayload(nonce_A=b64url_encode(bytes(32)))
        ).to_json()
    )
    with pytest.raises(asyncio.CancelledError):
        await run_dynamic_pairing_code_client(
            client_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=0,
            pairing_format=PairingCodeFormat.DIGITS,
            pairing_code_emitter=emit,
            server_id="server-X",
            store=client_store,
        )
    assert await client_store.pairing_round_count() == 1


async def _emitting_client(
    client_ews: EncryptedWebSocket, client_store: InMemoryClientPairingStore
) -> None:
    async def emit(_pairing_code: str) -> None:
        pass

    await run_dynamic_pairing_code_client(
        client_ews,
        handshake_hash=_HANDSHAKE_HASH,
        pairing_index=0,
        pairing_format=PairingCodeFormat.DIGITS,
        pairing_code_emitter=emit,
        server_id="server-X",
        store=client_store,
    )


async def test_dynamic_pairing_code_client_requires_nonce_a_in_first_round() -> None:
    """A first server/pair-init without nonce_A is a protocol error."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()
    await server_ews.send_str(ServerPairInitMessage(payload=ServerPairInitPayload()).to_json())

    with pytest.raises(PairingError, match="missing nonce_A") as excinfo:
        await _emitting_client(client_ews, client_store)

    assert not isinstance(excinfo.value, PairingAbortError)
    assert await client_store.pairing_round_count() == 0


async def test_dynamic_pairing_code_client_rejects_nonce_a_in_later_round() -> None:
    """A server/pair-init carrying nonce_A after a retry is a protocol error."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()

    async def server() -> None:
        await server_ews.receive()  # client/pair-init
        nonce_a = b64url_encode(pairing_code_mod.generate_nonce())
        await server_ews.send_str(
            ServerPairInitMessage(payload=ServerPairInitPayload(nonce_A=nonce_a)).to_json()
        )
        # A share for the wrong code, so the client's server_kc check fails.
        cpace = CPace.start(
            role=CPaceRole.INITIATOR,
            prs=b"not-the-code",
            sid=_pake_sid(_HANDSHAKE_HASH, 0, 1),
            ad=b"server",
        )
        await server_ews.send_str(
            ServerPairAuthMessage(
                payload=ServerPairAuthPayload(pake_msg_1=b64url_encode(cpace.public_share))
            ).to_json()
        )
        auth = ClientPairAuthMessage.from_json((await server_ews.receive()).data)
        cpace.derive(b64url_decode(auth.payload.pake_msg_2), b"client")
        await server_ews.send_str(
            ServerPairConfirmMessage(
                payload=ServerPairConfirmPayload(server_kc=b64url_encode(cpace.tag()))
            ).to_json()
        )
        retry = (await server_ews.receive()).data
        assert ClientPairRetryMessage.from_json(retry) == ClientPairRetryMessage()
        await server_ews.send_str(
            ServerPairInitMessage(payload=ServerPairInitPayload(nonce_A=nonce_a)).to_json()
        )

    with pytest.raises(PairingError, match="nonce_A after the first round") as excinfo:
        await asyncio.gather(_emitting_client(client_ews, client_store), server())

    assert not isinstance(excinfo.value, PairingAbortError)


async def test_client_stores_nothing_without_the_finalize_ack() -> None:
    """A non-pairing frame in place of the finalize ack fails the attempt and stores nothing."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()
    server_store = InMemoryServerPairingStore()
    await client_store.record_pairing_round()  # a prior round to be reset
    shown: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def emit(pairing_code: str) -> None:
        shown.set_result(pairing_code)

    async def provide() -> str:
        return await shown

    async def server_leaves_pairing() -> None:
        # Receive client/pair-finalize without finalizing, then leave pairing with a
        # server/activate (what the connection layer sends in place of an ack).
        await run_dynamic_pairing_code_server(
            server_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=0,
            pairing_format=PairingCodeFormat.DIGITS,
            pairing_code_provider=provide,
            client_id="client-A",
            store=server_store,
            verify=True,
        )
        await server_ews.send_str(
            ServerActivateMessage(
                payload=ServerActivatePayload(activities=[Activity.PLAYBACK], active_roles=[]),
            ).to_json(),
        )

    with pytest.raises(PairingError, match="malformed message awaiting ServerPairFinalizeMessage"):
        await asyncio.gather(
            run_dynamic_pairing_code_client(
                client_ews,
                handshake_hash=_HANDSHAKE_HASH,
                pairing_index=0,
                pairing_format=PairingCodeFormat.DIGITS,
                pairing_code_emitter=emit,
                server_id="server-X",
                store=client_store,
            ),
            server_leaves_pairing(),
        )

    # Neither side stored a record.
    assert _added_records(await client_store.list_records()) == []
    assert await server_store.record_by_client_id("client-A") is None
    # server_kc verified, so the round count resets like any other attempt.
    assert await client_store.pairing_round_count() == 0


_STATIC_PAIRING_CODE = "12345678"


@pytest.mark.parametrize("separator", ["", "-", " "])
async def test_static_pairing_code_round_trip(separator: str) -> None:
    """A matching static pairing code authenticates the PAKE and both sides persist the record.

    Separators the operator types between the digit groups are ignored.
    """
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()
    server_store = InMemoryServerPairingStore()
    await client_store.record_pairing_round()  # a dynamic-pairing-code round static pairing ignores

    async def provide() -> str:
        return _STATIC_PAIRING_CODE[:4] + separator + _STATIC_PAIRING_CODE[4:]

    _client_ret, server_record = await asyncio.gather(
        run_static_pairing_code_client(
            client_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=0,
            static_pairing_code=_STATIC_PAIRING_CODE,
            server_id="server-X",
            store=client_store,
        ),
        run_static_pairing_code_server(
            server_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=0,
            pairing_code_provider=provide,
            client_id="client-A",
            store=server_store,
        ),
    )

    client_record = await client_store.record_by_server_id("server-X")
    assert client_record is not None
    assert client_record.psk == server_record.psk
    assert client_record.psk_id == server_record.psk_id
    assert server_record.client_id == "client-A"
    assert await server_store.record_by_client_id("client-A") == server_record
    # The static flow leaves the dynamic-pairing-code round count alone.
    assert await client_store.pairing_round_count() == 1


async def test_static_pairing_code_wrong_code_aborts_and_persists_nothing() -> None:
    """A static-pairing-code mismatch aborts and stores nothing; the counter stays untouched."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()
    server_store = InMemoryServerPairingStore()

    async def provide_wrong() -> str:
        return "87654321"

    with pytest.raises(PairingAbortError) as excinfo:
        await asyncio.gather(
            run_static_pairing_code_client(
                client_ews,
                handshake_hash=_HANDSHAKE_HASH,
                pairing_index=0,
                static_pairing_code=_STATIC_PAIRING_CODE,
                server_id="server-X",
                store=client_store,
            ),
            run_static_pairing_code_server(
                server_ews,
                handshake_hash=_HANDSHAKE_HASH,
                pairing_index=0,
                pairing_code_provider=provide_wrong,
                client_id="client-A",
                store=server_store,
            ),
        )

    assert excinfo.value.reason is PairAbortReason.PAIRING_CODE_MISMATCH
    assert await client_store.pairing_round_count() == 0
    assert _added_records(await client_store.list_records()) == []
    assert await server_store.record_by_client_id("client-A") is None


async def test_static_pairing_code_pair_retry_is_protocol_error() -> None:
    """The static flow has no rounds: a client/pair-retry is a protocol error."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()

    async def provide() -> str:
        return _STATIC_PAIRING_CODE

    async def retrying_client() -> None:
        await client_ews.send_str(
            ClientPairInitMessage(payload=ClientPairInitPayload(pairing_index=0)).to_json()
        )
        cpace = CPace.start(
            role=CPaceRole.RESPONDER,
            prs=b"87654321",
            sid=_pake_sid(_HANDSHAKE_HASH, 0, 1),
            ad=b"client",
        )
        await client_ews.receive()  # server/pair-auth
        await client_ews.send_str(
            ClientPairAuthMessage(
                payload=ClientPairAuthPayload(pake_msg_2=b64url_encode(cpace.public_share)),
            ).to_json(),
        )
        await client_ews.receive()  # server/pair-confirm
        await client_ews.send_str(ClientPairRetryMessage().to_json())

    with pytest.raises(PairingError, match="got ClientPairRetryMessage") as excinfo:
        await asyncio.gather(
            run_static_pairing_code_server(
                server_ews,
                handshake_hash=_HANDSHAKE_HASH,
                pairing_index=0,
                pairing_code_provider=provide,
                client_id="client-A",
                store=InMemoryServerPairingStore(),
            ),
            retrying_client(),
        )
    assert not isinstance(excinfo.value, PairingAbortError)


@pytest.mark.parametrize(
    "pake_msg_1",
    [
        pytest.param("!!!notbase64!!!", id="not-base64"),
        pytest.param(b64url_encode(bytes(31)), id="wrong-length"),
        pytest.param(b64url_encode(bytes(32)), id="low-order"),
    ],
)
async def test_static_pairing_code_invalid_server_share_is_protocol_error(pake_msg_1: str) -> None:
    """An invalid CPace share from the server is a protocol error, not a pairing code guess."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()

    async def malicious_server() -> None:
        await server_ews.receive()  # client/pair-init
        await server_ews.send_str(
            ServerPairAuthMessage(
                payload=ServerPairAuthPayload(pake_msg_1=pake_msg_1),
            ).to_json(),
        )

    with pytest.raises(PairingError) as excinfo:
        await asyncio.gather(
            run_static_pairing_code_client(
                client_ews,
                handshake_hash=_HANDSHAKE_HASH,
                pairing_index=0,
                static_pairing_code=_STATIC_PAIRING_CODE,
                server_id="server-X",
                store=client_store,
            ),
            malicious_server(),
        )

    assert not isinstance(excinfo.value, PairingAbortError)
    assert await client_store.pairing_round_count() == 0
    assert _added_records(await client_store.list_records()) == []


async def test_static_pairing_code_malformed_client_share_raises() -> None:
    """A non-base64 CPace share from the client aborts the server without persisting a record."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    server_store = InMemoryServerPairingStore()

    async def provide() -> str:
        return _STATIC_PAIRING_CODE

    async def malicious_client() -> None:
        await client_ews.send_str(
            ClientPairInitMessage(payload=ClientPairInitPayload(pairing_index=0)).to_json()
        )
        await client_ews.receive()  # server/pair-auth
        await client_ews.send_str(
            ClientPairAuthMessage(
                payload=ClientPairAuthPayload(pake_msg_2="!!!notbase64!!!"),
            ).to_json(),
        )

    with pytest.raises(PairingError) as excinfo:
        await asyncio.gather(
            run_static_pairing_code_server(
                server_ews,
                handshake_hash=_HANDSHAKE_HASH,
                pairing_index=0,
                pairing_code_provider=provide,
                client_id="client-A",
                store=server_store,
            ),
            malicious_client(),
        )

    assert not isinstance(excinfo.value, PairingAbortError)
    assert await server_store.record_by_client_id("client-A") is None


def _wrap_nonce_b(sid: bytes, cpace: CPace, nonce: bytes) -> str:
    """Independently wrap ``nonce`` as ``wrapped_nonce_B`` (tests run the chachapoly suite)."""
    key = hashlib.sha256(b"sendspin-pair-nonce-wrap-v1" + sid + cpace.isk).digest()
    return b64url_encode(ChaCha20Poly1305(key).encrypt(bytes(12), nonce, None))


async def _honest_pake_to_finalize(
    client_ews: EncryptedWebSocket,
    *,
    wrapped_nonce_b: str | None = None,
    sid: bytes = _pake_sid(_HANDSHAKE_HASH, 0, 1),
) -> CPace:
    """Drive an honest static PAKE round, stopping before ``client/pair-finalize``."""
    await client_ews.send_str(
        ClientPairInitMessage(payload=ClientPairInitPayload(pairing_index=0)).to_json()
    )
    cpace = CPace.start(
        role=CPaceRole.RESPONDER, prs=_STATIC_PAIRING_CODE.encode("ascii"), sid=sid, ad=b"client"
    )
    auth = ServerPairAuthMessage.from_json((await client_ews.receive()).data)
    await client_ews.send_str(
        ClientPairAuthMessage(
            payload=ClientPairAuthPayload(pake_msg_2=b64url_encode(cpace.public_share)),
        ).to_json(),
    )
    cpace.derive(b64url_decode(auth.payload.pake_msg_1), b"server")
    await client_ews.receive()  # server/pair-confirm
    await client_ews.send_str(
        ClientPairConfirmMessage(
            payload=ClientPairConfirmPayload(
                client_kc=b64url_encode(cpace.tag()), wrapped_nonce_B=wrapped_nonce_b
            ),
        ).to_json(),
    )
    return cpace


def _psk_finalize_wrapped(sid: bytes, cpace: CPace, psk: bytes) -> str:
    """Independently build a ``client/pair-finalize`` wrapping ``psk`` under ``sid``."""
    key = hashlib.sha256(b"sendspin-pair-psk-wrap-v1" + sid + cpace.isk).digest()
    wrapped = ChaCha20Poly1305(key).encrypt(bytes(12), psk, None)
    return ClientPairFinalizeMessage(
        payload=ClientPairFinalizePayload(wrapped_psk=b64url_encode(wrapped))
    ).to_json()


async def test_static_pairing_code_wraps_under_the_round_one_sid() -> None:
    """The static flow keys and wraps under the sid of round 1."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    psk = generate_psk()
    sid = _pake_sid(_HANDSHAKE_HASH, 0, 1)

    async def provide() -> str:
        return _STATIC_PAIRING_CODE

    async def client() -> None:
        cpace = await _honest_pake_to_finalize(client_ews, sid=sid)
        await client_ews.send_str(_psk_finalize_wrapped(sid, cpace, psk))

    server_record, _ = await asyncio.gather(
        run_static_pairing_code_server(
            server_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=0,
            pairing_code_provider=provide,
            client_id="client-A",
            store=InMemoryServerPairingStore(),
        ),
        client(),
    )
    assert server_record is not None
    assert server_record.psk == psk


async def test_pairing_code_finalize_with_both_psk_fields_is_protocol_error() -> None:
    """A validly wrapped finalize that also carries ``long_term_psk`` persists nothing."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    server_store = InMemoryServerPairingStore()
    sid = _pake_sid(_HANDSHAKE_HASH, 0, 1)

    async def provide() -> str:
        return _STATIC_PAIRING_CODE

    async def client() -> None:
        cpace = await _honest_pake_to_finalize(client_ews, sid=sid)
        finalize = ClientPairFinalizeMessage.from_json(
            _psk_finalize_wrapped(sid, cpace, generate_psk())
        )
        finalize.payload.long_term_psk = b64url_encode(generate_psk())
        await client_ews.send_str(finalize.to_json())

    with pytest.raises(PairingError, match="both long_term_psk and wrapped_psk") as excinfo:
        await asyncio.gather(
            run_static_pairing_code_server(
                server_ews,
                handshake_hash=_HANDSHAKE_HASH,
                pairing_index=0,
                pairing_code_provider=provide,
                client_id="client-A",
                store=server_store,
            ),
            client(),
        )

    assert not isinstance(excinfo.value, PairingAbortError)
    assert await server_store.record_by_client_id("client-A") is None


# DEPRECATED(spec-pr-237): remove in aiosendspin <version>
async def test_legacy_rounds_static_server_pairs_under_the_pre_round_sid() -> None:
    """A static server serving a client predating rounds keys and wraps with the old sid."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    psk = generate_psk()
    sid = _legacy_pake_sid(_HANDSHAKE_HASH, 0)

    async def provide() -> str:
        return _STATIC_PAIRING_CODE

    async def legacy_client() -> None:
        cpace = await _honest_pake_to_finalize(client_ews, sid=sid)
        await client_ews.send_str(_psk_finalize_wrapped(sid, cpace, psk))

    server_record, _ = await asyncio.gather(
        run_static_pairing_code_server(
            server_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=0,
            pairing_code_provider=provide,
            client_id="client-A",
            store=InMemoryServerPairingStore(),
            legacy_rounds=True,
        ),
        legacy_client(),
    )
    assert server_record is not None
    assert server_record.psk == psk


async def test_static_pairing_code_server_rejects_dynamic_only_wrapped_nonce_b() -> None:
    """A static-pairing-code pair-confirm carrying wrapped_nonce_B is a protocol error."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    server_store = InMemoryServerPairingStore()

    async def provide() -> str:
        return _STATIC_PAIRING_CODE

    with pytest.raises(PairingError, match="wrapped_nonce_B for static pairing code"):
        await asyncio.gather(
            run_static_pairing_code_server(
                server_ews,
                handshake_hash=_HANDSHAKE_HASH,
                pairing_index=0,
                pairing_code_provider=provide,
                client_id="client-A",
                store=server_store,
            ),
            _honest_pake_to_finalize(client_ews, wrapped_nonce_b=b64url_encode(bytes(48))),
        )

    assert await server_store.record_by_client_id("client-A") is None


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            ClientPairFinalizePayload(long_term_psk=b64url_encode(bytes(32))),
            id="unwrapped_psk",
        ),
        pytest.param(
            ClientPairFinalizePayload(wrapped_psk=b64url_encode(bytes(48))),
            id="undecryptable_wrap",
        ),
    ],
)
async def test_pairing_code_finalize_without_valid_wrap_is_protocol_error(
    payload: ClientPairFinalizePayload,
) -> None:
    """A finalize whose PSK isn't wrapped under the CPace output is a protocol error."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    server_store = InMemoryServerPairingStore()

    async def provide() -> str:
        return _STATIC_PAIRING_CODE

    async def client_with_bad_finalize() -> None:
        await _honest_pake_to_finalize(client_ews)
        await client_ews.send_str(ClientPairFinalizeMessage(payload=payload).to_json())

    with pytest.raises(PairingError) as excinfo:
        await asyncio.gather(
            run_static_pairing_code_server(
                server_ews,
                handshake_hash=_HANDSHAKE_HASH,
                pairing_index=0,
                pairing_code_provider=provide,
                client_id="client-A",
                store=server_store,
            ),
            client_with_bad_finalize(),
        )

    assert not isinstance(excinfo.value, PairingAbortError)
    assert await server_store.record_by_client_id("client-A") is None


async def _dynamic_pake_client(
    client_ews: EncryptedWebSocket,
    pairing_code_future: asyncio.Future[str],
    *,
    sid: bytes = _pake_sid(_HANDSHAKE_HASH, 0, 1),
    mangle_pairing_code: bool = False,
    mangle_nonce: bool = False,
    mangle_wrap: bool = False,
    omit_wrap: bool = False,
) -> CPace:
    """Drive a dynamic PAKE round through ``client/pair-confirm``, optionally cheating.

    ``mangle_pairing_code`` emits (and uses) a pairing code not bound to the handshake;
    ``mangle_nonce`` reveals a nonce that does not match the commitment; ``mangle_wrap``
    sends an undecryptable ``wrapped_nonce_B``; ``omit_wrap`` sends none at all.
    """
    nonce_b = pairing_code_mod.generate_nonce()
    await client_ews.send_str(
        ClientPairInitMessage(
            payload=ClientPairInitPayload(
                pairing_index=0, commit_B=b64url_encode(pairing_code_mod.commit(nonce_b))
            ),
        ).to_json(),
    )
    init = ServerPairInitMessage.from_json((await client_ews.receive()).data)
    nonce_a = b64url_decode(init.payload.nonce_A)
    pairing_code = pairing_code_mod.derive_digits(_HANDSHAKE_HASH, nonce_a, nonce_b)
    if mangle_pairing_code:
        pairing_code = ("2" if pairing_code[0] == "1" else "1") + pairing_code[1:]
    pairing_code_future.set_result(pairing_code)
    cpace = CPace.start(
        role=CPaceRole.RESPONDER, prs=pairing_code.encode("ascii"), sid=sid, ad=b"client"
    )
    auth = ServerPairAuthMessage.from_json((await client_ews.receive()).data)
    await client_ews.send_str(
        ClientPairAuthMessage(
            payload=ClientPairAuthPayload(pake_msg_2=b64url_encode(cpace.public_share)),
        ).to_json(),
    )
    cpace.derive(b64url_decode(auth.payload.pake_msg_1), b"server")
    await client_ews.receive()  # server/pair-confirm
    revealed = pairing_code_mod.generate_nonce() if mangle_nonce else nonce_b
    wrapped_nonce_b: str | None = _wrap_nonce_b(sid, cpace, revealed)
    if mangle_wrap:
        wrapped_nonce_b = b64url_encode(bytes(48))
    if omit_wrap:
        wrapped_nonce_b = None
    await client_ews.send_str(
        ClientPairConfirmMessage(
            payload=ClientPairConfirmPayload(
                client_kc=b64url_encode(cpace.tag()),
                wrapped_nonce_B=wrapped_nonce_b,
            ),
        ).to_json(),
    )
    return cpace


# DEPRECATED(spec-pr-237): remove in aiosendspin <version>
async def test_legacy_rounds_server_pairs_under_the_pre_round_sid() -> None:
    """A server serving a client predating rounds keys the PAKE and wraps with the old sid."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    server_store = InMemoryServerPairingStore()
    pairing_code_future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    psk = generate_psk()
    sid = _legacy_pake_sid(_HANDSHAKE_HASH, 0)

    async def legacy_client() -> None:
        cpace = await _dynamic_pake_client(client_ews, pairing_code_future, sid=sid)
        await client_ews.send_str(_psk_finalize_wrapped(sid, cpace, psk))

    server_record, _ = await asyncio.gather(
        run_dynamic_pairing_code_server(
            server_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=0,
            pairing_format=PairingCodeFormat.DIGITS,
            pairing_code_provider=lambda: pairing_code_future,
            client_id="client-A",
            store=server_store,
            legacy_rounds=True,
        ),
        legacy_client(),
    )
    assert server_record is not None
    assert server_record.psk == psk


# DEPRECATED(spec-pr-237): remove in aiosendspin <version>
async def test_legacy_rounds_server_rejects_pair_retry() -> None:
    """A server serving a client predating rounds runs no retry loop."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    pairing_code_future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    sid = _legacy_pake_sid(_HANDSHAKE_HASH, 0)

    async def client() -> None:
        nonce_b = pairing_code_mod.generate_nonce()
        await client_ews.send_str(
            ClientPairInitMessage(
                payload=ClientPairInitPayload(
                    pairing_index=0, commit_B=b64url_encode(pairing_code_mod.commit(nonce_b))
                ),
            ).to_json(),
        )
        await client_ews.receive()  # server/pair-init
        pairing_code_future.set_result("123456")
        cpace = CPace.start(role=CPaceRole.RESPONDER, prs=b"123456", sid=sid, ad=b"client")
        await client_ews.receive()  # server/pair-auth
        await client_ews.send_str(
            ClientPairAuthMessage(
                payload=ClientPairAuthPayload(pake_msg_2=b64url_encode(cpace.public_share)),
            ).to_json(),
        )
        await client_ews.receive()  # server/pair-confirm
        await client_ews.send_str(ClientPairRetryMessage().to_json())

    with pytest.raises(PairingError, match="got ClientPairRetryMessage") as excinfo:
        await asyncio.gather(
            run_dynamic_pairing_code_server(
                server_ews,
                handshake_hash=_HANDSHAKE_HASH,
                pairing_index=0,
                pairing_format=PairingCodeFormat.DIGITS,
                pairing_code_provider=lambda: pairing_code_future,
                client_id="client-A",
                store=InMemoryServerPairingStore(),
                legacy_rounds=True,
            ),
            client(),
        )
    assert not isinstance(excinfo.value, PairingAbortError)


async def test_dynamic_pairing_code_mismatched_commit_is_protocol_error() -> None:
    """A revealed nonce_B not matching commit_B is a protocol error, not a mismatch."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    server_store = InMemoryServerPairingStore()
    pairing_code_future: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    with pytest.raises(PairingError) as excinfo:
        await asyncio.gather(
            run_dynamic_pairing_code_server(
                server_ews,
                handshake_hash=_HANDSHAKE_HASH,
                pairing_index=0,
                pairing_format=PairingCodeFormat.DIGITS,
                pairing_code_provider=lambda: pairing_code_future,
                client_id="client-A",
                store=server_store,
            ),
            _dynamic_pake_client(client_ews, pairing_code_future, mangle_nonce=True),
        )

    assert not isinstance(excinfo.value, PairingAbortError)
    assert await server_store.record_by_client_id("client-A") is None


async def test_dynamic_pairing_code_unbound_code_is_protocol_error() -> None:
    """A code not derived from the handshake fails the binding check with a protocol error."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    server_store = InMemoryServerPairingStore()
    pairing_code_future: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    with pytest.raises(PairingError, match="not bound to this connection") as excinfo:
        await asyncio.gather(
            run_dynamic_pairing_code_server(
                server_ews,
                handshake_hash=_HANDSHAKE_HASH,
                pairing_index=0,
                pairing_format=PairingCodeFormat.DIGITS,
                pairing_code_provider=lambda: pairing_code_future,
                client_id="client-A",
                store=server_store,
            ),
            _dynamic_pake_client(client_ews, pairing_code_future, mangle_pairing_code=True),
        )

    assert not isinstance(excinfo.value, PairingAbortError)
    assert await server_store.record_by_client_id("client-A") is None


async def test_dynamic_pairing_code_undecryptable_wrapped_nonce_is_protocol_error() -> None:
    """A wrapped_nonce_B not sealed under the CPace output is a protocol error."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    server_store = InMemoryServerPairingStore()
    pairing_code_future: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    with pytest.raises(PairingError, match="AEAD failure") as excinfo:
        await asyncio.gather(
            run_dynamic_pairing_code_server(
                server_ews,
                handshake_hash=_HANDSHAKE_HASH,
                pairing_index=0,
                pairing_format=PairingCodeFormat.DIGITS,
                pairing_code_provider=lambda: pairing_code_future,
                client_id="client-A",
                store=server_store,
            ),
            _dynamic_pake_client(client_ews, pairing_code_future, mangle_wrap=True),
        )

    assert not isinstance(excinfo.value, PairingAbortError)
    assert await server_store.record_by_client_id("client-A") is None


async def test_dynamic_pairing_code_missing_wrapped_nonce_is_protocol_error() -> None:
    """A dynamic pair-confirm without wrapped_nonce_B is a protocol error."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    server_store = InMemoryServerPairingStore()
    pairing_code_future: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    with pytest.raises(PairingError, match="missing wrapped_nonce_B") as excinfo:
        await asyncio.gather(
            run_dynamic_pairing_code_server(
                server_ews,
                handshake_hash=_HANDSHAKE_HASH,
                pairing_index=0,
                pairing_format=PairingCodeFormat.DIGITS,
                pairing_code_provider=lambda: pairing_code_future,
                client_id="client-A",
                store=server_store,
            ),
            _dynamic_pake_client(client_ews, pairing_code_future, omit_wrap=True),
        )

    assert not isinstance(excinfo.value, PairingAbortError)
    assert await server_store.record_by_client_id("client-A") is None


async def test_pairing_psk_falls_back_to_shared_when_storage_exhausted() -> None:
    """On storage exhaustion the client hands the server its configured shared PSK.

    No new record is created on the client; the server stores the shared PSK as
    its own long-term record keyed by client_id.
    """
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = ExhaustedClientStore()
    server_store = InMemoryServerPairingStore()

    shared_psk = generate_psk()
    shared = ClientPairingRecord(psk_id=psk_id_for(shared_psk), psk=shared_psk, server_id=None)
    await client_store.store_record(shared)
    await client_store.set_record_mode_psk_id(shared.psk_id)

    _client_ret, server_record = await asyncio.gather(
        run_pairing_psk_client(
            client_ews,
            pairing_index=0,
            server_id="server-X",
            store=client_store,
        ),
        run_pairing_psk_server(
            server_ews, pairing_index=0, client_id="client-A", store=server_store
        ),
    )

    # The client admitted the server under the shared record: no new stored-pubkey record.
    assert _added_records(await client_store.list_records()) == []
    assert await client_store.record_by_psk_id(shared.psk_id) is not None
    # The server received and stored the shared PSK.
    assert server_record.psk == shared_psk
    assert await server_store.record_by_client_id("client-A") == server_record
