"""Pure Noise KKpsk2 protocol object — handshake and transport, no I/O."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, cast

from cryptography.exceptions import InvalidTag
from noise.connection import Keypair, NoiseConnection
from noise.exceptions import NoiseInvalidMessage

from .keys import PSK_SIZE, X25519_KEY_SIZE


class NoiseCipherSuite(StrEnum):
    """The Noise KKpsk2 cipher suites this implementation supports."""

    CHACHAPOLY = "25519_ChaChaPoly_SHA256"
    AESGCM = "25519_AESGCM_SHA256"

    @property
    def pattern_name(self) -> bytes:
        """The full Noise pattern string fed into the underlying library."""
        return f"Noise_KKpsk2_{self.value}".encode("ascii")


class NoiseSession:
    """A Noise KKpsk2 session — handshake-and-transport state machine."""

    def __init__(self, conn: NoiseConnection, *, suite: NoiseCipherSuite) -> None:
        """Wrap an already-configured ``NoiseConnection``; prefer the factories."""
        self._conn = conn
        self._suite = suite
        self._rebuild: _InitiatorRebuild | None = None
        self._message_1: tuple[bytes, bytes] | None = None

    @property
    def suite(self) -> NoiseCipherSuite:
        """The cipher suite negotiated for this session."""
        return self._suite

    @classmethod
    def as_initiator(
        cls,
        *,
        suite: NoiseCipherSuite,
        local_static_priv: bytes,
        remote_static_pub: bytes,
        prologue: bytes,
        psk: bytes,
    ) -> NoiseSession:
        """Build the **server-side** session and start the handshake."""
        _check_key_size("local_static_priv", local_static_priv)
        _check_key_size("remote_static_pub", remote_static_pub)
        _check_psk_size(psk)
        conn = _build_conn(
            suite=suite,
            initiator=True,
            local_static_priv=local_static_priv,
            remote_static_pub=remote_static_pub,
            prologue=prologue,
            psk=psk,
        )
        session = cls(conn, suite=suite)
        # Keep what it takes to replay this handshake, so message 2 can be verified under
        # a second PSK if the first fails. The ephemeral joins it once message 1 is written.
        session._rebuild = _InitiatorRebuild(
            suite=suite,
            local_static_priv=local_static_priv,
            remote_static_pub=remote_static_pub,
            prologue=prologue,
        )
        return session

    @classmethod
    def as_responder(
        cls,
        *,
        suite: NoiseCipherSuite,
        local_static_priv: bytes,
        remote_static_pub: bytes,
        prologue: bytes,
    ) -> NoiseSession:
        """Build the **client-side** session and start the handshake."""
        _check_key_size("local_static_priv", local_static_priv)
        _check_key_size("remote_static_pub", remote_static_pub)
        conn = _build_conn(
            suite=suite,
            initiator=False,
            local_static_priv=local_static_priv,
            remote_static_pub=remote_static_pub,
            prologue=prologue,
            psk=_PLACEHOLDER_PSK,
        )
        return cls(conn, suite=suite)

    def write_message(self, payload: bytes = b"") -> bytes:
        """Produce the next outgoing handshake message containing ``payload``."""
        message = bytes(self._conn.write_message(payload))
        if self._rebuild is not None and self._message_1 is None:
            self._message_1 = (payload, message)
        return message

    def fork_at_message_2(self, psk: bytes) -> NoiseSession:
        """Return this initiator's state as it stood after message 1, keyed by ``psk``.

        In KKpsk2 the PSK is mixed in at message 2, so verifying that message under a
        second PSK means replaying message 1 on the same ephemeral rather than starting
        a new session. The returned session is ready to read message 2; this one is
        spent either way, its symmetric state already advanced by the failed read.

        Only callable between writing message 1 and completing the handshake, which is
        where the ephemeral is still reachable and reusing it is a replay of one exchange
        rather than reuse across two.
        """
        if self._rebuild is None or self._message_1 is None:
            msg = "only an initiator mid-handshake, having written message 1, can be forked"
            raise RuntimeError(msg)
        _check_psk_size(psk)
        payload, ciphertext = self._message_1
        conn = self._rebuild.build(psk, ephemeral_priv=_ephemeral_private_bytes(self._conn))
        forked = NoiseSession(conn, suite=self._suite)
        if forked.write_message(payload) != ciphertext:
            # Replaying the same inputs must reproduce the message the peer answered;
            # anything else means the rebuild no longer mirrors the original.
            msg = "replayed Noise message 1 differs from the one sent"
            raise RuntimeError(msg)
        return forked

    def read_message(self, ciphertext: bytes) -> bytes:
        """Consume the next incoming handshake message; return the decrypted payload."""
        try:
            plaintext = bytes(self._conn.read_message(ciphertext))
        except InvalidTag as exc:
            raise NoiseInvalidMessage("Failed authentication of handshake message") from exc
        if self.handshake_complete:
            # The library drops its handshake state here. Drop the replay material with
            # it, so nothing that could rebuild this exchange outlives the handshake.
            self._rebuild = None
            self._message_1 = None
        return plaintext

    def mix_psk(self, psk: bytes) -> None:
        """Swap in the real PSK between reading message 1 and writing message 2."""
        _check_psk_size(psk)
        self._conn.set_psks(psks=[psk])

    @property
    def handshake_complete(self) -> bool:
        """``True`` once both handshake messages have been processed."""
        return cast("bool", self._conn.handshake_finished)

    @property
    def handshake_hash(self) -> bytes:
        """The 32-byte Noise handshake hash ``h``."""
        if not self.handshake_complete:
            msg = "handshake_hash is only available after the handshake completes"
            raise RuntimeError(msg)
        return cast("bytes", self._conn.get_handshake_hash())

    def encrypt(self, plaintext: bytes) -> bytes:
        """Encrypt and authenticate ``plaintext`` for transport mode."""
        return cast("bytes", self._conn.encrypt(plaintext))

    def decrypt(self, ciphertext: bytes) -> bytes:
        """Decrypt and authenticate ``ciphertext`` for transport mode."""
        return cast("bytes", self._conn.decrypt(ciphertext))


# --- private helpers -----------------------------------------------------

_PLACEHOLDER_PSK: Final[bytes] = b"\x00" * PSK_SIZE


@dataclass(frozen=True, slots=True)
class _InitiatorRebuild:
    """Everything needed to rebuild an initiator session on the same ephemeral."""

    suite: NoiseCipherSuite
    local_static_priv: bytes
    remote_static_pub: bytes
    prologue: bytes

    def build(self, psk: bytes, *, ephemeral_priv: bytes) -> NoiseConnection:
        """Start a fresh initiator handshake keyed by ``psk`` on the original ephemeral."""
        with warnings.catch_warnings():
            # The library warns that a pre-set ephemeral must never happen in production.
            # It guards against reusing one across handshakes; this replays a single
            # handshake the peer has already answered, and the session is discarded with it.
            warnings.filterwarnings("ignore", message=".*ephemeral keypairs is already set.*")
            return _build_conn(
                suite=self.suite,
                initiator=True,
                local_static_priv=self.local_static_priv,
                remote_static_pub=self.remote_static_pub,
                prologue=self.prologue,
                psk=psk,
                ephemeral_priv=ephemeral_priv,
            )


def _ephemeral_private_bytes(conn: NoiseConnection) -> bytes:
    """Return the ephemeral private key the library generated for ``conn``."""
    keypair = conn.noise_protocol.handshake_state.e
    return cast("bytes", keypair.private.private_bytes_raw())


def _check_key_size(name: str, value: bytes) -> None:
    if len(value) != X25519_KEY_SIZE:
        msg = f"{name} must be {X25519_KEY_SIZE} bytes, got {len(value)}"
        raise ValueError(msg)


def _check_psk_size(psk: bytes) -> None:
    if len(psk) != PSK_SIZE:
        msg = f"PSK must be {PSK_SIZE} bytes, got {len(psk)}"
        raise ValueError(msg)


def _build_conn(
    *,
    suite: NoiseCipherSuite,
    initiator: bool,
    local_static_priv: bytes,
    remote_static_pub: bytes,
    prologue: bytes,
    psk: bytes,
    ephemeral_priv: bytes | None = None,
) -> NoiseConnection:
    conn = NoiseConnection.from_name(suite.pattern_name)
    if initiator:
        conn.set_as_initiator()
    else:
        conn.set_as_responder()
    conn.set_keypair_from_private_bytes(Keypair.STATIC, local_static_priv)
    if ephemeral_priv is not None:
        # Pre-set so the library reuses it instead of generating its own.
        conn.set_keypair_from_private_bytes(Keypair.EPHEMERAL, ephemeral_priv)
    conn.set_keypair_from_public_bytes(Keypair.REMOTE_STATIC, remote_static_pub)
    conn.set_prologue(prologue)
    conn.set_psks(psks=[psk])
    conn.start_handshake()
    return conn
