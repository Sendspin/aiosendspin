"""Pairing exchanges that run over the encrypted channel."""

from __future__ import annotations

import asyncio
import hashlib
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, NoReturn, Protocol, cast, overload

from aiohttp import WSMsgType
from cpace import CPace, CPaceError, CPaceRole
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305

from aiosendspin.models.types import PairAbortReason, PairingCodeFormat, PairMethod

from . import pairing_code as pairing_code_mod
from .keys import PSK_SIZE, b64url_decode, b64url_encode, psk_id_for
from .models import (
    ClientPairAuthMessage,
    ClientPairAuthPayload,
    ClientPairConfirmMessage,
    ClientPairConfirmPayload,
    ClientPairFinalizeMessage,
    ClientPairFinalizePayload,
    ClientPairInitMessage,
    ClientPairInitPayload,
    ClientPairPendingMessage,
    PairAbortMessage,
    PairAbortPayload,
    PairingMessage,
    ServerPairAuthMessage,
    ServerPairAuthPayload,
    ServerPairConfirmMessage,
    ServerPairConfirmPayload,
    ServerPairFinalizeMessage,
    ServerPairInitMessage,
    ServerPairInitPayload,
)
from .pairing_token import decode_pairing_code_token, encode_pairing_code_token
from .session import NoiseCipherSuite
from .trust_store import ServerPairingRecord

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

    from .trust_store import ClientPairingStore, ServerPairingStore
    from .wire import EncryptedWebSocket

_PAKE_SID_LABEL = b"sendspin-pair-pake-v1"
_PAKE_AD_SERVER = b"server"  # CPace ADa (initiator)
_PAKE_AD_CLIENT = b"client"  # CPace ADb (responder)
_PAKE_SHARE_SIZE = 32
_KC_TAG_SIZE = 64
_PSK_WRAP_LABEL = b"sendspin-pair-psk-wrap-v1"
_NONCE_WRAP_LABEL = b"sendspin-pair-nonce-wrap-v1"
_WRAP_NONCE = bytes(12)  # zero nonce is safe: each wrap key is per-field and used once
_AEAD_TAG_SIZE = 16
_CLIENT_ATTEMPT_TIMEOUT_S: float = 120.0
# Server bounds raise PairingTimeoutError, sending no pairing message: no pair/abort reason is
# available to a server for its own timeout. They exceed the client's attempt timeout so the
# client's in-band abort wins when both sides are live.
# Public so operator UIs can mirror the enforced bounds (countdowns etc.).
SERVER_ATTEMPT_TIMEOUT_S: float = 180.0
SERVER_FIRST_MESSAGE_TIMEOUT_S: float = 60.0
SERVER_GESTURE_TIMEOUT_S: float = 360.0


class PairingError(Exception):
    """A pairing attempt could not complete (malformed message, early close, etc.)."""


class PairingTimeoutError(PairingError):
    """A server-side bound on waiting for a client pairing message expired."""


class PairingAbortError(PairingError):
    """A pairing attempt ended with a ``pair/abort`` carrying ``reason`` (base)."""

    def __init__(self, reason: PairAbortReason) -> None:
        """Record the abort ``reason``."""
        super().__init__(f"pairing aborted: {reason.value}")
        self.reason = reason


class LocalPairingAbortError(PairingAbortError):
    """This side aborted the pairing and sent the ``pair/abort``."""


class RemotePairingAbortError(PairingAbortError):
    """The peer aborted the pairing; its ``pair/abort`` was received."""


class PairingCodeProvider(Protocol):
    """Supplies the pairing code the operator entered into the server."""

    def __call__(self) -> Awaitable[str]:
        """Return the operator-entered pairing code as an awaitable."""


@dataclass(frozen=True, slots=True)
class PairingAttempt:
    """Operator-initiated pairing intent attached to a server-side dial."""

    method: PairMethod
    pairing_code_provider: PairingCodeProvider | None = None
    """Required for code methods; supplies the operator-entered pairing code or token."""
    pairing_format: PairingCodeFormat | None = None
    """Emission format for the dynamic pairing code; absent for the other methods."""
    pairing_psk: bytes | None = None
    """Required for the Pairing PSK method; the live PSK pasted from a token."""
    verify: bool = False
    """Re-verify an already-paired client instead of pairing anew."""
    on_pair_pending: Callable[[], None] | None = None
    """Called when the client reports the attempt gesture-gated."""
    owner: str | None = None
    """Application-defined authorization id the resulting record is bound to."""

    def __post_init__(self) -> None:
        """Validate ``method`` / material agree."""
        if self.method is PairMethod.PAIRING_PSK:
            if self.pairing_psk is None:
                msg = "PAIRING_PSK requires pairing_psk"
                raise ValueError(msg)
            if len(self.pairing_psk) != PSK_SIZE:
                msg = f"pairing_psk must be {PSK_SIZE} bytes, got {len(self.pairing_psk)}"
                raise ValueError(msg)
            if self.pairing_code_provider is not None or self.pairing_format is not None:
                msg = "PAIRING_PSK does not use code pairing fields"
                raise ValueError(msg)
            if self.verify:
                msg = "PAIRING_PSK does not support verification"
                raise ValueError(msg)
            if self.on_pair_pending is not None:
                msg = "PAIRING_PSK does not use on_pair_pending"
                raise ValueError(msg)
        else:  # Pairing-code methods
            if self.pairing_code_provider is None:
                msg = f"{self.method.value} requires pairing_code_provider"
                raise ValueError(msg)
            if self.pairing_psk is not None:
                msg = f"{self.method.value} does not use pairing_psk"
                raise ValueError(msg)
            if self.method is PairMethod.DYNAMIC_PAIRING_CODE:
                if self.pairing_format is None:
                    msg = "dynamic_pairing_code requires pairing_format"
                    raise ValueError(msg)
            elif self.pairing_format is not None:
                msg = f"{self.method.value} does not use pairing_format"
                raise ValueError(msg)


if TYPE_CHECKING:
    PairingCodeEmitter = Callable[[str], Awaitable[None]]


async def run_pairing_psk_client(
    ws: EncryptedWebSocket,
    *,
    pairing_index: int,
    server_id: str,
    store: ClientPairingStore,
) -> str | None:
    """Run the client side of the Pairing PSK flow.

    Returns ``None`` on finalize, else the raw ``server/activate`` leave frame.
    """
    async with _client_timeout(ws):
        await ws.send_str(
            ClientPairInitMessage(
                payload=ClientPairInitPayload(pairing_index=pairing_index),
            ).to_json(),
        )
        return await _finalize_client(ws, server_id=server_id, store=store)


async def run_pairing_psk_server(
    ws: EncryptedWebSocket,
    *,
    pairing_index: int,
    client_id: str,
    store: ServerPairingStore,
    owner: str | None = None,
    on_pair_init: Callable[[], None] | None = None,
    # DEPRECATED(spec-pr-247): remove in aiosendspin <version>
    on_legacy_finalize: Callable[[], None] | None = None,
) -> ServerPairingRecord:
    """Run the server side of the Pairing PSK flow.

    ``on_pair_init`` is called for every ``client/pair-init`` received, whatever its index.
    ``client/pair-finalize`` messages preceding the matching ``client/pair-init`` are discarded
    as leftovers, except that with ``on_legacy_finalize`` set, a ``long_term_psk`` finalize
    arriving before any ``client/pair-init`` is accepted as this attempt's unless it raises.
    """
    finalize: ClientPairFinalizeMessage | None = None
    pair_init_seen = False
    async with _server_timeout(SERVER_FIRST_MESSAGE_TIMEOUT_S, "client/pair-init"):
        while True:
            message = await _receive_pairing(
                ws, (ClientPairInitMessage, ClientPairPendingMessage, ClientPairFinalizeMessage)
            )
            if isinstance(message, ClientPairFinalizeMessage):
                # DEPRECATED(spec-pr-247): remove in aiosendspin <version>
                if (
                    on_legacy_finalize is not None
                    and not pair_init_seen
                    and message.payload.long_term_psk is not None
                ):
                    on_legacy_finalize()
                    finalize = message
                    break
                # A leftover from a cancelled attempt: discard silently.
                continue
            if isinstance(message, ClientPairInitMessage):
                pair_init_seen = True
                if on_pair_init is not None:
                    on_pair_init()
            if message.payload.pairing_index > pairing_index:
                raise PairingError(
                    f"{type(message).__name__} pairing_index is ahead of the server's count"
                )
            if message.payload.pairing_index < pairing_index:
                # A leftover from a superseded pairing server/activate: discard silently.
                continue
            if isinstance(message, ClientPairPendingMessage):
                raise PairingError("client/pair-pending is not part of the Pairing PSK flow")
            if message.payload.commit_B is not None:
                raise PairingError("client/pair-init carries commit_B for Pairing PSK")
            break
    async with _server_timeout(SERVER_ATTEMPT_TIMEOUT_S, "the rest of the attempt"):
        record = await _finalize_server(
            ws,
            client_id=client_id,
            store=store,
            method=PairMethod.PAIRING_PSK,
            owner=owner,
            finalize=finalize,
        )
    assert record is not None
    return record


async def run_dynamic_pairing_code_client(
    ws: EncryptedWebSocket,
    *,
    handshake_hash: bytes,
    pairing_index: int,
    pairing_format: PairingCodeFormat,
    pairing_code_emitter: PairingCodeEmitter,
    server_id: str,
    store: ClientPairingStore,
) -> str | None:
    """Run the client side of the dynamic-pairing-code flow.

    Returns ``None`` on finalize, else the raw ``server/activate`` leave frame.
    """
    sid = _pake_sid(handshake_hash, pairing_index)
    nonce_b = pairing_code_mod.generate_nonce()
    async with _client_timeout(ws):
        await ws.send_str(
            ClientPairInitMessage(
                payload=ClientPairInitPayload(
                    pairing_index=pairing_index,
                    commit_B=b64url_encode(pairing_code_mod.commit(nonce_b)),
                ),
            ).to_json(),
        )

        init = await _receive_pairing(ws, ServerPairInitMessage)
        nonce_a = _decode_field(
            init.payload.nonce_A, "nonce_A", expect_len=pairing_code_mod.NONCE_SIZE
        )
        if pairing_format is PairingCodeFormat.DIGITS:
            pairing_code = pairing_code_mod.derive_digits(handshake_hash, nonce_a, nonce_b)
            prs = pairing_code.encode("ascii")
        else:
            prs = pairing_code_mod.derive_qr_code(handshake_hash, nonce_a, nonce_b)
            pairing_code = encode_pairing_code_token(prs)
        await pairing_code_emitter(pairing_code)
        try:
            cpace = CPace.start(role=CPaceRole.RESPONDER, prs=prs, sid=sid, ad=_PAKE_AD_CLIENT)
        except CPaceError as exc:
            raise PairingError("CPace initialization failed") from exc

        auth = await _receive_pairing(ws, ServerPairAuthMessage)
        await ws.send_str(
            ClientPairAuthMessage(
                payload=ClientPairAuthPayload(pake_msg_2=b64url_encode(cpace.public_share)),
            ).to_json(),
        )
        peer_share = _decode_field(
            auth.payload.pake_msg_1, "pake_msg_1", expect_len=_PAKE_SHARE_SIZE
        )
        try:
            cpace.derive(peer_share, _PAKE_AD_SERVER)
        except CPaceError as exc:
            raise PairingError("malformed pake_msg_1: invalid CPace share") from exc

        confirm = await _receive_pairing(ws, ServerPairConfirmMessage)
        if not cpace.verify(
            _decode_field(confirm.payload.server_kc, "server_kc", expect_len=_KC_TAG_SIZE)
        ):
            await store.record_pairing_code_failure()
            await abort_pairing(ws, PairAbortReason.PAIRING_CODE_MISMATCH)
        await store.reset_pairing_code_failures()
        wrapped_nonce = _wrap_aead(
            ws.session.suite, _wrap_key(_NONCE_WRAP_LABEL, sid, cpace)
        ).encrypt(_WRAP_NONCE, nonce_b, None)
        await ws.send_str(
            ClientPairConfirmMessage(
                payload=ClientPairConfirmPayload(
                    client_kc=b64url_encode(cpace.tag()),
                    wrapped_nonce_B=b64url_encode(wrapped_nonce),
                ),
            ).to_json(),
        )

        return await _finalize_client(
            ws,
            server_id=server_id,
            store=store,
            wrap_key=_wrap_key(_PSK_WRAP_LABEL, sid, cpace),
        )


async def run_dynamic_pairing_code_server(
    ws: EncryptedWebSocket,
    *,
    handshake_hash: bytes,
    pairing_index: int,
    pairing_code_provider: PairingCodeProvider,
    pairing_format: PairingCodeFormat,
    client_id: str,
    store: ServerPairingStore,
    verify: bool = False,
    on_pair_pending: Callable[[], None] | None = None,
    owner: str | None = None,
) -> ServerPairingRecord | None:
    """Run the server side of the dynamic-pairing-code flow.

    Returns the persisted record, or ``None`` when ``verify`` is set (re-verified, left pairing).
    """
    sid = _pake_sid(handshake_hash, pairing_index)
    init = await _receive_pair_init(ws, pairing_index, on_pending=on_pair_pending)
    if init.payload.commit_B is None:
        raise PairingError("client/pair-init missing commit_B for dynamic pairing code")
    commit_b = _decode_field(
        init.payload.commit_B, "commit_B", expect_len=pairing_code_mod.COMMIT_SIZE
    )
    async with _server_timeout(SERVER_ATTEMPT_TIMEOUT_S, "the rest of the attempt"):
        nonce_a = pairing_code_mod.generate_nonce()
        await ws.send_str(
            ServerPairInitMessage(
                payload=ServerPairInitPayload(nonce_A=b64url_encode(nonce_a)),
            ).to_json(),
        )
        entered = await pairing_code_provider()
        if pairing_format is PairingCodeFormat.DIGITS:
            if (
                not entered.isascii()
                or not entered.isdigit()
                or len(entered) != pairing_code_mod.DYNAMIC_DIGITS
            ):
                raise PairingError("dynamic pairing code must be exactly 6 ASCII digits")
            prs = entered.encode("ascii")
        else:
            try:
                prs = decode_pairing_code_token(entered)
            except ValueError as exc:
                raise PairingError("malformed pairing token") from exc
        try:
            cpace = CPace.start(role=CPaceRole.INITIATOR, prs=prs, sid=sid, ad=_PAKE_AD_SERVER)
        except CPaceError as exc:
            raise PairingError("CPace initialization failed") from exc
        await ws.send_str(
            ServerPairAuthMessage(
                payload=ServerPairAuthPayload(pake_msg_1=b64url_encode(cpace.public_share)),
            ).to_json(),
        )

        auth = await _receive_pairing(ws, ClientPairAuthMessage)
        peer_share = _decode_field(
            auth.payload.pake_msg_2, "pake_msg_2", expect_len=_PAKE_SHARE_SIZE
        )
        try:
            cpace.derive(peer_share, _PAKE_AD_CLIENT)
        except CPaceError as exc:
            raise PairingError("malformed pake_msg_2: invalid CPace share") from exc
        await ws.send_str(
            ServerPairConfirmMessage(
                payload=ServerPairConfirmPayload(server_kc=b64url_encode(cpace.tag())),
            ).to_json(),
        )

        confirm = await _receive_pairing(ws, ClientPairConfirmMessage)
        if not cpace.verify(
            _decode_field(confirm.payload.client_kc, "client_kc", expect_len=_KC_TAG_SIZE)
        ):
            await abort_pairing(ws, PairAbortReason.PAIRING_CODE_MISMATCH)
        if confirm.payload.wrapped_nonce_B is None:
            raise PairingError(
                "client/pair-confirm missing wrapped_nonce_B for dynamic pairing code"
            )
        wrapped_nonce = _decode_field(
            confirm.payload.wrapped_nonce_B,
            "wrapped_nonce_B",
            expect_len=pairing_code_mod.NONCE_SIZE + _AEAD_TAG_SIZE,
        )
        try:
            nonce_b = _wrap_aead(
                ws.session.suite, _wrap_key(_NONCE_WRAP_LABEL, sid, cpace)
            ).decrypt(_WRAP_NONCE, wrapped_nonce, None)
        except InvalidTag as exc:
            raise PairingError("malformed wrapped_nonce_B: AEAD failure") from exc
        if not pairing_code_mod.verify_commit(nonce_b, commit_b):
            raise PairingError("revealed nonce_B does not match commit_B")
        derived_prs = (
            pairing_code_mod.derive_digits(handshake_hash, nonce_a, nonce_b).encode("ascii")
            if pairing_format is PairingCodeFormat.DIGITS
            else pairing_code_mod.derive_qr_code(handshake_hash, nonce_a, nonce_b)
        )
        if prs != derived_prs:
            raise PairingError("entered pairing code is not bound to this connection")

        return await _finalize_server(
            ws,
            client_id=client_id,
            store=store,
            method=PairMethod.DYNAMIC_PAIRING_CODE,
            verify=verify,
            wrap_key=_wrap_key(_PSK_WRAP_LABEL, sid, cpace),
            owner=owner,
        )


async def run_static_pairing_code_client(
    ws: EncryptedWebSocket,
    *,
    handshake_hash: bytes,
    pairing_index: int,
    static_pairing_code: str,
    server_id: str,
    store: ClientPairingStore,
) -> str | None:
    """Run the client side of the static-pairing-code flow.

    The caller has opened the pairing window. Returns ``None`` on finalize,
    else the raw ``server/activate`` leave frame.
    """
    sid = _pake_sid(handshake_hash, pairing_index)
    async with _client_timeout(ws):
        await ws.send_str(
            ClientPairInitMessage(
                payload=ClientPairInitPayload(pairing_index=pairing_index),
            ).to_json(),
        )
        try:
            cpace = CPace.start(
                role=CPaceRole.RESPONDER,
                prs=static_pairing_code.encode("ascii"),
                sid=sid,
                ad=_PAKE_AD_CLIENT,
            )
        except CPaceError as exc:
            raise PairingError("CPace initialization failed") from exc

        auth = await _receive_pairing(ws, ServerPairAuthMessage)
        await ws.send_str(
            ClientPairAuthMessage(
                payload=ClientPairAuthPayload(pake_msg_2=b64url_encode(cpace.public_share)),
            ).to_json(),
        )
        peer_share = _decode_field(
            auth.payload.pake_msg_1, "pake_msg_1", expect_len=_PAKE_SHARE_SIZE
        )
        try:
            cpace.derive(peer_share, _PAKE_AD_SERVER)
        except CPaceError as exc:
            raise PairingError("malformed pake_msg_1: invalid CPace share") from exc

        confirm = await _receive_pairing(ws, ServerPairConfirmMessage)
        if not cpace.verify(
            _decode_field(confirm.payload.server_kc, "server_kc", expect_len=_KC_TAG_SIZE)
        ):
            await abort_pairing(ws, PairAbortReason.PAIRING_CODE_MISMATCH)
        await ws.send_str(
            ClientPairConfirmMessage(
                payload=ClientPairConfirmPayload(client_kc=b64url_encode(cpace.tag())),
            ).to_json(),
        )

        return await _finalize_client(
            ws,
            server_id=server_id,
            store=store,
            wrap_key=_wrap_key(_PSK_WRAP_LABEL, sid, cpace),
        )


async def run_static_pairing_code_server(
    ws: EncryptedWebSocket,
    *,
    handshake_hash: bytes,
    pairing_index: int,
    pairing_code_provider: PairingCodeProvider,
    client_id: str,
    store: ServerPairingStore,
    verify: bool = False,
    on_pair_pending: Callable[[], None] | None = None,
    owner: str | None = None,
) -> ServerPairingRecord | None:
    """Run the server side of the static-pairing-code flow.

    Returns the persisted record, or ``None`` when ``verify`` is set (re-verified, left pairing).
    """
    sid = _pake_sid(handshake_hash, pairing_index)
    init = await _receive_pair_init(ws, pairing_index, on_pending=on_pair_pending)
    if init.payload.commit_B is not None:
        raise PairingError("client/pair-init carries commit_B for static pairing code")
    async with _server_timeout(SERVER_ATTEMPT_TIMEOUT_S, "the rest of the attempt"):
        pairing_code = await pairing_code_provider()
        if not pairing_code_mod.is_valid_static_pairing_code(pairing_code):
            raise PairingError("static pairing code must be exactly 8 decimal digits")
        try:
            cpace = CPace.start(
                role=CPaceRole.INITIATOR,
                prs=pairing_code.encode("ascii"),
                sid=sid,
                ad=_PAKE_AD_SERVER,
            )
        except CPaceError as exc:
            raise PairingError("CPace initialization failed") from exc
        await ws.send_str(
            ServerPairAuthMessage(
                payload=ServerPairAuthPayload(pake_msg_1=b64url_encode(cpace.public_share)),
            ).to_json(),
        )

        auth = await _receive_pairing(ws, ClientPairAuthMessage)
        peer_share = _decode_field(
            auth.payload.pake_msg_2, "pake_msg_2", expect_len=_PAKE_SHARE_SIZE
        )
        try:
            cpace.derive(peer_share, _PAKE_AD_CLIENT)
        except CPaceError as exc:
            raise PairingError("malformed pake_msg_2: invalid CPace share") from exc
        await ws.send_str(
            ServerPairConfirmMessage(
                payload=ServerPairConfirmPayload(server_kc=b64url_encode(cpace.tag())),
            ).to_json(),
        )

        confirm = await _receive_pairing(ws, ClientPairConfirmMessage)
        if confirm.payload.wrapped_nonce_B is not None:
            raise PairingError(
                "client/pair-confirm carries wrapped_nonce_B for static pairing code"
            )
        if not cpace.verify(
            _decode_field(confirm.payload.client_kc, "client_kc", expect_len=_KC_TAG_SIZE)
        ):
            await abort_pairing(ws, PairAbortReason.PAIRING_CODE_MISMATCH)

        return await _finalize_server(
            ws,
            client_id=client_id,
            store=store,
            method=PairMethod.STATIC_PAIRING_CODE,
            verify=verify,
            wrap_key=_wrap_key(_PSK_WRAP_LABEL, sid, cpace),
            owner=owner,
        )


async def _finalize_client(
    ws: EncryptedWebSocket,
    *,
    server_id: str,
    store: ClientPairingStore,
    wrap_key: bytes | None = None,
) -> str | None:
    """Send ``client/pair-finalize``, wrapping the PSK when ``wrap_key`` is set.

    Pairing-code flows set ``wrap_key``. Returns ``None`` after persisting on the
    server's ack, else its raw leave frame.
    """
    psk, record = await store.resolve_pairing_outcome(server_id=server_id)
    if wrap_key is None:
        payload = ClientPairFinalizePayload(long_term_psk=b64url_encode(psk))
    else:
        wrapped = _wrap_aead(ws.session.suite, wrap_key).encrypt(_WRAP_NONCE, psk, None)
        payload = ClientPairFinalizePayload(wrapped_psk=b64url_encode(wrapped))
    await ws.send_str(ClientPairFinalizeMessage(payload=payload).to_json())
    reply = await _receive_pairing_frame(ws, ServerPairFinalizeMessage)
    if isinstance(reply, str):
        return reply  # server left pairing without finalizing; nothing stored
    if record is not None:
        await store.replace_record_for_server_id(record)
    return None


async def _finalize_server(
    ws: EncryptedWebSocket,
    *,
    client_id: str,
    store: ServerPairingStore,
    method: PairMethod,
    verify: bool = False,
    wrap_key: bytes | None = None,
    owner: str | None = None,
    finalize: ClientPairFinalizeMessage | None = None,
) -> ServerPairingRecord | None:
    """Consume ``client/pair-finalize``: finalize a record, or re-verify (returns ``None``).

    A ``finalize`` already received is consumed instead of reading the next frame.
    """
    if finalize is None:
        finalize = await _receive_pairing(ws, ClientPairFinalizeMessage)
    existing = await store.record_by_client_id(client_id)
    record = existing.with_method(method) if existing is not None else None
    if not verify:
        psk = _unwrap_psk(ws.session.suite, finalize.payload, wrap_key)
        if record is None:
            record = ServerPairingRecord(
                psk_id=psk_id_for(psk),
                psk=psk,
                client_id=client_id,
                pair_methods=[method],
                owner=owner,
            )
        else:
            # Ownership tracks the latest authorization that minted the credential.
            record = replace(record, psk_id=psk_id_for(psk), psk=psk, owner=owner)
    if record is not None and record is not existing:
        await store.store_record(record)  # persist before acking
    if verify:
        return None
    # The new record supersedes the client's lesser grants.
    await store.unstage_pairing_psk(client_id)
    await store.remove_trusted_unpaired(client_id)
    await ws.send_str(ServerPairFinalizeMessage().to_json())
    return record


def _unwrap_psk(
    suite: NoiseCipherSuite, payload: ClientPairFinalizePayload, wrap_key: bytes | None
) -> bytes:
    """Extract the PSK from ``client/pair-finalize``, unwrapping when ``wrap_key`` is set."""
    if wrap_key is None:
        if payload.long_term_psk is None:
            raise PairingError("client/pair-finalize is missing long_term_psk")
        return _decode_field(payload.long_term_psk, "long_term_psk", expect_len=PSK_SIZE)
    if payload.wrapped_psk is None:
        raise PairingError("client/pair-finalize is missing wrapped_psk")
    wrapped = _decode_field(
        payload.wrapped_psk, "wrapped_psk", expect_len=PSK_SIZE + _AEAD_TAG_SIZE
    )
    try:
        psk = _wrap_aead(suite, wrap_key).decrypt(_WRAP_NONCE, wrapped, None)
    except InvalidTag as exc:
        raise PairingError("malformed wrapped_psk: AEAD failure") from exc
    return psk


def _decode_field(value: str, what: str, *, expect_len: int | None = None) -> bytes:
    """Base64url-decode a received pairing field, raising ``PairingError`` if malformed."""
    try:
        raw = b64url_decode(value)
    except ValueError as exc:
        raise PairingError(f"malformed {what}: not valid base64url") from exc
    if expect_len is not None and len(raw) != expect_len:
        raise PairingError(f"malformed {what}: expected {expect_len} bytes, got {len(raw)}")
    return raw


@asynccontextmanager
async def _client_timeout(ws: EncryptedWebSocket) -> AsyncGenerator[None]:
    """Bound a client attempt, aborting in band with ``attempt_timeout`` on expiry."""
    try:
        async with asyncio.timeout(_CLIENT_ATTEMPT_TIMEOUT_S):
            yield
    except TimeoutError:
        await abort_pairing(ws, PairAbortReason.ATTEMPT_TIMEOUT)


@asynccontextmanager
async def _server_timeout(timeout_s: float, what: str) -> AsyncGenerator[None]:
    """Bound a server-side wait for ``what``, reporting expiry as a pairing timeout."""
    try:
        async with asyncio.timeout(timeout_s):
            yield
    except TimeoutError as exc:
        raise PairingTimeoutError(f"{what} did not arrive in time") from exc


async def abort_pairing(ws: EncryptedWebSocket, reason: PairAbortReason) -> NoReturn:
    """Send ``pair/abort`` (best-effort) and raise ``LocalPairingAbortError``."""
    with suppress(Exception):
        await ws.send_str(PairAbortMessage(payload=PairAbortPayload(reason=reason)).to_json())
    raise LocalPairingAbortError(reason)


@overload
async def _receive_pairing_frame[T: PairingMessage](
    ws: EncryptedWebSocket, expected: type[T]
) -> T | str: ...


@overload
async def _receive_pairing_frame[T: PairingMessage, U: PairingMessage](
    ws: EncryptedWebSocket, expected: tuple[type[T], type[U]]
) -> T | U | str: ...


@overload
async def _receive_pairing_frame[T: PairingMessage, U: PairingMessage, V: PairingMessage](
    ws: EncryptedWebSocket, expected: tuple[type[T], type[U], type[V]]
) -> T | U | V | str: ...


async def _receive_pairing_frame(
    ws: EncryptedWebSocket, expected: type[PairingMessage] | tuple[type[PairingMessage], ...]
) -> PairingMessage | str:
    """Receive a frame: a parsed ``expected`` message, or the raw text if it isn't pairing."""
    kinds = expected if isinstance(expected, tuple) else (expected,)
    expected_names = _expected_names(expected)
    msg = await ws.receive()
    if msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED):
        raise PairingError(f"connection closed while awaiting {expected_names}")
    if msg.type is not WSMsgType.TEXT:
        raise PairingError(f"expected a JSON frame ({expected_names}), got {msg.type.name}")
    data = cast("str", msg.data)
    try:
        message = PairingMessage.from_json(data)
    except (ValueError, LookupError):
        return data
    if isinstance(message, PairAbortMessage):
        raise RemotePairingAbortError(message.payload.reason)
    if not isinstance(message, kinds):
        raise PairingError(f"expected {expected_names}, got {type(message).__name__}")
    return message


@overload
async def _receive_pairing[T: PairingMessage](ws: EncryptedWebSocket, expected: type[T]) -> T: ...


@overload
async def _receive_pairing[T: PairingMessage, U: PairingMessage](
    ws: EncryptedWebSocket, expected: tuple[type[T], type[U]]
) -> T | U: ...


@overload
async def _receive_pairing[T: PairingMessage, U: PairingMessage, V: PairingMessage](
    ws: EncryptedWebSocket, expected: tuple[type[T], type[U], type[V]]
) -> T | U | V: ...


async def _receive_pairing(
    ws: EncryptedWebSocket,
    expected: type[PairingMessage]
    | tuple[type[PairingMessage], type[PairingMessage]]
    | tuple[type[PairingMessage], type[PairingMessage], type[PairingMessage]],
) -> PairingMessage:
    """Receive the next pairing frame, requiring it to be of an ``expected`` type."""
    message = await _receive_pairing_frame(ws, expected)
    if isinstance(message, str):
        raise PairingError(f"malformed message awaiting {_expected_names(expected)}")
    return message


def _expected_names(expected: type[PairingMessage] | tuple[type[PairingMessage], ...]) -> str:
    """Human-readable name(s) of the expected message type(s)."""
    kinds = expected if isinstance(expected, tuple) else (expected,)
    return " or ".join(kind.__name__ for kind in kinds)


async def receive_pairing_abort(ws: EncryptedWebSocket) -> str:
    """Await the ``pair/abort`` ending an unstarted attempt.

    A ``pair/abort`` (or close, or another pairing frame) raises. A non-pairing
    JSON frame, such as the ``server/activate`` leaving pairing, is returned raw
    for the caller to interpret.
    """
    frame = await _receive_pairing_frame(ws, PairAbortMessage)
    assert isinstance(frame, str)
    return frame


async def _receive_pair_init(
    ws: EncryptedWebSocket,
    pairing_index: int,
    *,
    on_pending: Callable[[], None] | None = None,
) -> ClientPairInitMessage:
    """Receive this attempt's ``client/pair-init``.

    It allows one gesture-extending ``client/pair-pending``.
    It also discards any leftover pair-init/pair-pending from a superseded attempt.
    """
    async with _server_timeout(SERVER_FIRST_MESSAGE_TIMEOUT_S, "client/pair-init"):
        while True:
            message = await _receive_pairing(ws, (ClientPairInitMessage, ClientPairPendingMessage))
            if message.payload.pairing_index > pairing_index:
                raise PairingError(
                    f"{type(message).__name__} pairing_index is ahead of the server's count"
                )
            if message.payload.pairing_index == pairing_index:
                break
            # A leftover from a superseded pairing server/activate: discard silently.
    if isinstance(message, ClientPairInitMessage):
        return message
    if on_pending is not None:
        on_pending()
    # In-order delivery leaves no room for leftovers after the matching pair-pending:
    # the next pairing frame must be this attempt's client/pair-init.
    async with _server_timeout(SERVER_GESTURE_TIMEOUT_S, "gesture-gated client/pair-init"):
        init = await _receive_pairing(ws, ClientPairInitMessage)
    if init.payload.pairing_index != pairing_index:
        raise PairingError("client/pair-init pairing_index does not match the attempt")
    return init


def _pake_sid(handshake_hash: bytes, pairing_index: int) -> bytes:
    """CPace session id binding the PAKE to the Noise handshake and pairing-code pairing attempt."""
    return _PAKE_SID_LABEL + handshake_hash + pairing_index.to_bytes(4, "big")


def _wrap_key(label: bytes, sid: bytes, cpace: CPace) -> bytes:
    """Derive a per-field wrap key from the CPace output."""
    return hashlib.sha256(label + sid + cpace.isk).digest()


def _wrap_aead(suite: NoiseCipherSuite, wrap_key: bytes) -> AESGCM | ChaCha20Poly1305:
    """Build the negotiated suite's AEAD, keyed for wrapping."""
    if suite is NoiseCipherSuite.AESGCM:
        return AESGCM(wrap_key)
    return ChaCha20Poly1305(wrap_key)
