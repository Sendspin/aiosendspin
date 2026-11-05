"""X25519 identity keys, base64url helpers, and ``psk_id`` derivation."""

from __future__ import annotations

import base64
import hashlib
import secrets
from dataclasses import dataclass, field
from typing import Final

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import x25519

from .constants import PSK_ID_LABEL

# Sizes (bytes).
PSK_SIZE: Final[int] = 32
X25519_KEY_SIZE: Final[int] = 32

# Length of a base64url-encoded peer identifier (32-byte X25519 public key,
# no padding).
PEER_ID_SIZE: Final[int] = 43


def b64url_encode(data: bytes) -> str:
    """Return ``data`` base64url-encoded with no trailing ``=`` padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def b64url_decode(value: str) -> bytes:
    """Base64url-decode ``value``, tolerating missing ``=`` padding."""
    pad = (-len(value)) % 4
    return base64.urlsafe_b64decode(value + "=" * pad)


def psk_id_for(psk: bytes) -> str:
    """Return the base64url-encoded ``psk_id`` for ``psk``."""
    if len(psk) != PSK_SIZE:
        msg = f"PSK must be {PSK_SIZE} bytes, got {len(psk)}"
        raise ValueError(msg)
    digest = hashlib.sha256(PSK_ID_LABEL + psk).digest()
    return b64url_encode(digest)


def generate_psk() -> bytes:
    """Return a fresh 32-byte PSK from a cryptographically secure RNG."""
    return secrets.token_bytes(PSK_SIZE)


@dataclass(frozen=True, slots=True)
class Identity:
    """Sendspin static identity — a long-term X25519 keypair."""

    private_bytes: bytes = field(repr=False)
    public_bytes: bytes

    @classmethod
    def generate(cls) -> Identity:
        """Return a new randomly generated identity."""
        return cls._from_priv_obj(x25519.X25519PrivateKey.generate())

    @classmethod
    def from_private_bytes(cls, private_bytes: bytes) -> Identity:
        """Reconstruct an identity from its 32-byte raw private key."""
        if len(private_bytes) != X25519_KEY_SIZE:
            msg = f"X25519 private key must be {X25519_KEY_SIZE} bytes"
            raise ValueError(msg)
        priv = x25519.X25519PrivateKey.from_private_bytes(private_bytes)
        return cls._from_priv_obj(priv)

    @classmethod
    def _from_priv_obj(cls, priv: x25519.X25519PrivateKey) -> Identity:
        pub = priv.public_key()
        return cls(
            private_bytes=priv.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption(),
            ),
            public_bytes=pub.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            ),
        )

    @property
    def peer_id(self) -> str:
        """The base64url public key — i.e. the Sendspin ``client_id``/``server_id``."""
        return b64url_encode(self.public_bytes)

    @property
    def private_b64u(self) -> str:
        """Return the base64url-encoded private key (for persistence)."""
        return b64url_encode(self.private_bytes)
