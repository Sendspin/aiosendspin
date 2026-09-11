"""Async driver for the cleartext init exchange and KKpsk2 handshake."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Final, Protocol, cast

from aiohttp import WSMsgType
from noise.exceptions import (
    NoiseHandshakeError,
    NoiseInvalidMessage,
    NoiseValueError,
)

from .constants import PROTOCOL_VERSION
from .keys import (
    PEER_ID_SIZE,
    X25519_KEY_SIZE,
    Identity,
    b64url_decode,
    b64url_encode,
)
from .models import (
    ClientInitMessage,
    ClientInitPayload,
    NoiseHandshakeMessage,
    NoiseHandshakePayload,
    NoiseMsg1Payload,
    NoiseMsg2Payload,
    ServerInitMessage,
    ServerInitPayload,
)
from .session import NoiseCipherSuite, NoiseSession
from .trust_store import PskCategory, ResolvedPsk
from .wire import EncryptedWebSocket, RawWebSocket

# Per-message timeout during cleartext init and Noise handshake.
DEFAULT_HANDSHAKE_TIMEOUT_S: Final[float] = 30.0

# Client callback: given a psk_id, return the matching PSK record, or None.
PskResolver = Callable[[str, "PskCategory"], Awaitable[ResolvedPsk | None]]

# Server callback: given a client_id, return a PSK to admit it, or None.
PskProvider = Callable[[str], Awaitable[ResolvedPsk | None]]


class HandshakeWebSocket(RawWebSocket, Protocol):
    """WS surface needed by the handshake driver (extends ``RawWebSocket``)."""

    async def send_str(self, data: str) -> None:
        """Send a text WebSocket frame."""


class HandshakeAbortedError(Exception):
    """Raised when the Noise handshake cannot complete.

    The caller is expected to close the underlying WebSocket without sending
    any application-level error message.
    """


@dataclass(frozen=True, slots=True)
class HandshakeResult:
    """Outcome of a successful Noise handshake."""

    encrypted_ws: EncryptedWebSocket
    """Transport-mode wrapper for all subsequent application I/O."""
    peer_id: str
    """``client_id`` on the server side, ``server_id`` on the client side."""
    suite: NoiseCipherSuite
    """The negotiated cipher suite."""
    psk: ResolvedPsk
    """The PSK that admitted the connection, with its trust metadata."""
    handshake_hash: bytes
    """The Noise handshake hash ``h`` for the completed handshake."""


async def run_handshake_server(
    ws: HandshakeWebSocket,
    *,
    local_identity: Identity,
    psk_provider: PskProvider,
    expected_client_id: str | None = None,
    client_init_text: str | None = None,
    timeout_s: float = DEFAULT_HANDSHAKE_TIMEOUT_S,
) -> HandshakeResult:
    """Run the server-side (Noise initiator) handshake."""
    if client_init_text is None:
        client_init_text = await receive_text_frame(ws, what="client/init", timeout_s=timeout_s)
    client_init = _parse_client_init(client_init_text)
    suite = _select_suite(client_init.payload.suite)
    client_id = client_init.payload.client_id
    if expected_client_id is not None and client_id != expected_client_id:
        raise HandshakeAbortedError(
            f"client_id mismatch: expected {expected_client_id!r}, got {client_id!r}",
        )
    client_static_pub = _peer_pub_bytes(client_id, "client_id")

    # Resolve the PSK and build message 1 *before* sending anything, so
    # server/init and noise/handshake go out back-to-back, and so a client
    # we won't admit is rejected without ever seeing server/init.
    resolved = await psk_provider(client_id)
    if resolved is None:
        raise HandshakeAbortedError(f"no PSK admits client_id={client_id!r}")

    server_init_text = ServerInitMessage(
        payload=ServerInitPayload(
            server_id=local_identity.peer_id,
            version=PROTOCOL_VERSION,
        ),
    ).to_json()
    prologue = client_init_text.encode("utf-8") + server_init_text.encode("utf-8")
    session = NoiseSession.as_initiator(
        suite=suite,
        local_static_priv=local_identity.private_bytes,
        remote_static_pub=client_static_pub,
        prologue=prologue,
        psk=resolved.psk,
    )
    # server/init immediately followed by Noise message 1 (spec: no client
    # message is awaited in between).
    await ws.send_str(server_init_text)
    await _exchange_as_initiator(ws, session=session, psk=resolved, timeout_s=timeout_s)

    return HandshakeResult(
        encrypted_ws=EncryptedWebSocket(ws, session),
        peer_id=client_id,
        suite=suite,
        psk=resolved,
        handshake_hash=session.handshake_hash,
    )


async def run_handshake_client(
    ws: HandshakeWebSocket,
    *,
    local_identity: Identity,
    suite: NoiseCipherSuite,
    psk_resolver: PskResolver,
    expected_server_id: str | None = None,
    timeout_s: float = DEFAULT_HANDSHAKE_TIMEOUT_S,
) -> HandshakeResult:
    """Run the client-side (Noise responder) handshake."""
    client_init_text = ClientInitMessage(
        payload=ClientInitPayload(
            client_id=local_identity.peer_id,
            version=PROTOCOL_VERSION,
            suite=suite.value,
        ),
    ).to_json()
    await ws.send_str(client_init_text)

    server_init_text = await receive_text_frame(ws, what="server/init", timeout_s=timeout_s)
    server_init = _parse_server_init(server_init_text)
    server_id = server_init.payload.server_id
    if expected_server_id is not None and server_id != expected_server_id:
        raise HandshakeAbortedError(
            f"server_id mismatch: expected {expected_server_id!r}, got {server_id!r}",
        )
    server_static_pub = _peer_pub_bytes(server_id, "server_id")

    prologue = client_init_text.encode("utf-8") + server_init_text.encode("utf-8")
    session = NoiseSession.as_responder(
        suite=suite,
        local_static_priv=local_identity.private_bytes,
        remote_static_pub=server_static_pub,
        prologue=prologue,
    )

    resolved = await _exchange_as_responder(
        ws,
        session=session,
        psk_resolver=psk_resolver,
        expected_peer_id=server_id,
        timeout_s=timeout_s,
    )

    return HandshakeResult(
        encrypted_ws=EncryptedWebSocket(ws, session),
        peer_id=server_id,
        suite=suite,
        psk=resolved,
        handshake_hash=session.handshake_hash,
    )


async def run_rehandshake_server(
    enc_ws: EncryptedWebSocket,
    *,
    local_identity: Identity,
    client_id: str,
    suite: NoiseCipherSuite,
    prologue: bytes,
    psk: ResolvedPsk,
    timeout_s: float = DEFAULT_HANDSHAKE_TIMEOUT_S,
) -> HandshakeResult:
    """Re-run the handshake as initiator over ``enc_ws`` and swap to ``psk``."""
    client_static_pub = _peer_pub_bytes(client_id, "client_id")
    session = NoiseSession.as_initiator(
        suite=suite,
        local_static_priv=local_identity.private_bytes,
        remote_static_pub=client_static_pub,
        prologue=prologue,
        psk=psk.psk,
    )
    await _exchange_as_initiator(enc_ws, session=session, psk=psk, timeout_s=timeout_s)
    enc_ws.swap_session(session)
    return HandshakeResult(
        encrypted_ws=enc_ws,
        peer_id=client_id,
        suite=suite,
        psk=psk,
        handshake_hash=session.handshake_hash,
    )


async def run_rehandshake_client(
    enc_ws: EncryptedWebSocket,
    *,
    local_identity: Identity,
    server_id: str,
    suite: NoiseCipherSuite,
    prologue: bytes,
    psk_resolver: PskResolver,
    hs1_text: str | None = None,
    timeout_s: float = DEFAULT_HANDSHAKE_TIMEOUT_S,
) -> HandshakeResult:
    """Re-run the handshake as responder over ``enc_ws`` and swap to the resolved PSK."""
    server_static_pub = _peer_pub_bytes(server_id, "server_id")
    session = NoiseSession.as_responder(
        suite=suite,
        local_static_priv=local_identity.private_bytes,
        remote_static_pub=server_static_pub,
        prologue=prologue,
    )
    resolved = await _exchange_as_responder(
        enc_ws,
        session=session,
        psk_resolver=psk_resolver,
        expected_peer_id=server_id,
        timeout_s=timeout_s,
        hs1_text=hs1_text,
    )
    enc_ws.swap_session(session)
    return HandshakeResult(
        encrypted_ws=enc_ws,
        peer_id=server_id,
        suite=suite,
        psk=resolved,
        handshake_hash=session.handshake_hash,
    )


# --- private helpers -----------------------------------------------------


async def receive_text_frame(
    ws: HandshakeWebSocket,
    *,
    what: str,
    timeout_s: float = DEFAULT_HANDSHAKE_TIMEOUT_S,
) -> str:
    """Receive one TEXT frame within ``timeout_s``, or abort the handshake."""
    try:
        async with asyncio.timeout(timeout_s):
            msg = await ws.receive()
    except TimeoutError as exc:
        raise HandshakeAbortedError(f"timed out awaiting {what}") from exc
    if msg.type is not WSMsgType.TEXT:
        raise HandshakeAbortedError(f"expected {what} (TEXT), got {msg.type.name}")
    return cast("str", msg.data)


async def _exchange_as_initiator(
    transport: HandshakeWebSocket,
    *,
    session: NoiseSession,
    psk: ResolvedPsk,
    timeout_s: float,
) -> None:
    """Exchange the two ``noise/handshake`` messages as the initiator (server)."""
    msg1 = NoiseMsg1Payload(psk_id=psk.psk_id, psk_category=psk.category.code)
    msg1_pt = msg1.to_json().encode("utf-8")
    msg1_ct = session.write_message(msg1_pt)
    await transport.send_str(_pack_handshake(msg1_ct))
    hs2_text = await receive_text_frame(transport, what="Noise message 2", timeout_s=timeout_s)
    msg2_pt = _read_handshake_message(session, hs2_text, "Noise message 2")
    _validate_msg2_payload(msg2_pt)


async def _exchange_as_responder(
    transport: HandshakeWebSocket,
    *,
    session: NoiseSession,
    psk_resolver: PskResolver,
    expected_peer_id: str,
    timeout_s: float,
    hs1_text: str | None = None,
) -> ResolvedPsk:
    """Exchange the two ``noise/handshake`` messages as the responder (client); returns the PSK."""
    hs1_text = (
        hs1_text
        if hs1_text is not None
        else await receive_text_frame(transport, what="Noise message 1", timeout_s=timeout_s)
    )
    msg1_pt = _read_handshake_message(session, hs1_text, "Noise message 1")
    msg1_obj = _parse_msg1_payload(msg1_pt)

    declared = _declared_category(msg1_obj.psk_category)
    if declared is _UNKNOWN_CATEGORY:
        # A category from a newer revision names nothing this client holds.
        resolved = None
    else:
        resolved = await psk_resolver(msg1_obj.psk_id, declared)
    if resolved is None:
        raise HandshakeAbortedError(f"no PSK matches psk_id={msg1_obj.psk_id!r}")
    # Stored-pubkey post-match check: the record's bound server_id must be the
    # server we actually reached.
    if resolved.counterparty_id is not None and resolved.counterparty_id != expected_peer_id:
        raise HandshakeAbortedError(
            f"PSK bound to server_id {resolved.counterparty_id!r}, "
            f"but connected to {expected_peer_id!r}",
        )
    session.mix_psk(resolved.psk)

    msg2_pt = NoiseMsg2Payload().to_json().encode("utf-8")
    msg2_ct = session.write_message(msg2_pt)
    await transport.send_str(_pack_handshake(msg2_ct))
    return resolved


def _parse_client_init(text: str) -> ClientInitMessage:
    try:
        msg = ClientInitMessage.from_json(text)
    except Exception as exc:
        raise HandshakeAbortedError(f"malformed client/init: {exc}") from exc
    if msg.type != "client/init":
        raise HandshakeAbortedError(f"expected client/init, got {msg.type!r}")
    _check_version(msg.payload.version)
    return msg


def _parse_server_init(text: str) -> ServerInitMessage:
    try:
        msg = ServerInitMessage.from_json(text)
    except Exception as exc:
        raise HandshakeAbortedError(f"malformed server/init: {exc}") from exc
    if msg.type != "server/init":
        raise HandshakeAbortedError(f"expected server/init, got {msg.type!r}")
    _check_version(msg.payload.version)
    return msg


def _read_handshake_message(session: NoiseSession, text: str, what: str) -> bytes:
    """Parse a ``noise/handshake`` frame and decrypt it through ``session``."""
    try:
        hs = NoiseHandshakeMessage.from_json(text)
    except Exception as exc:
        raise HandshakeAbortedError(f"malformed noise/handshake ({what}): {exc}") from exc
    if hs.type != "noise/handshake":
        raise HandshakeAbortedError(f"expected noise/handshake, got {hs.type!r}")
    try:
        ciphertext = b64url_decode(hs.payload.data)  # binascii.Error subclasses ValueError
    except ValueError as exc:
        raise HandshakeAbortedError(f"malformed {what} payload encoding") from exc
    try:
        return session.read_message(ciphertext)
    except (NoiseInvalidMessage, NoiseHandshakeError, NoiseValueError) as exc:
        raise HandshakeAbortedError(f"{what} failed Noise authentication") from exc


# Stands for a declared category this implementation does not know, which no PSK answers.
_UNKNOWN_CATEGORY: Final[PskCategory] = cast("PskCategory", object())


def _declared_category(code: str) -> PskCategory:
    """Return the category message 1 declared; a code naming none we know matches nothing."""
    return PskCategory.from_code(code) or _UNKNOWN_CATEGORY


def _parse_msg1_payload(plaintext: bytes) -> NoiseMsg1Payload:
    """Parse the decrypted Noise-message-1 payload, wrapping malformed input."""
    try:
        return NoiseMsg1Payload.from_json(plaintext.decode("utf-8"))
    except Exception as exc:
        raise HandshakeAbortedError(f"malformed Noise message 1 payload: {exc}") from exc


def _pack_handshake(noise_bytes: bytes) -> str:
    return NoiseHandshakeMessage(
        payload=NoiseHandshakePayload(data=b64url_encode(noise_bytes)),
    ).to_json()


def _select_suite(name: str) -> NoiseCipherSuite:
    try:
        return NoiseCipherSuite(name)
    except ValueError as exc:
        raise HandshakeAbortedError(f"unsupported suite {name!r}") from exc


def _check_version(version: int) -> None:
    if version != PROTOCOL_VERSION:
        raise HandshakeAbortedError(f"unsupported protocol version {version}")


def _peer_pub_bytes(peer_id: str, what: str) -> bytes:
    if len(peer_id) != PEER_ID_SIZE:
        raise HandshakeAbortedError(
            f"invalid {what} length: {len(peer_id)} (expected {PEER_ID_SIZE})",
        )
    try:
        decoded = b64url_decode(peer_id)
    except Exception as exc:
        raise HandshakeAbortedError(f"invalid {what} encoding") from exc
    if len(decoded) != X25519_KEY_SIZE:
        raise HandshakeAbortedError(
            f"invalid {what}: decoded to {len(decoded)} bytes (expected {X25519_KEY_SIZE})",
        )
    return decoded


def _validate_msg2_payload(payload: bytes) -> None:
    """Validate Noise message 2's plaintext payload (an empty object ``{}``)."""
    try:
        NoiseMsg2Payload.from_json(payload.decode("utf-8"))
    except Exception as exc:
        raise HandshakeAbortedError(f"malformed Noise message 2 payload: {exc}") from exc
