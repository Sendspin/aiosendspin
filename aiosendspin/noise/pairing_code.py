"""Pairing-code derivation and commitment helpers."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Final

PAIRING_CODE_DERIVE_LABEL: Final[bytes] = b"sendspin-pairing-code-derive-v1"
COMMIT_LABEL: Final[bytes] = b"sendspin-pair-commit-v1"
NONCE_SIZE: Final[int] = 32
COMMIT_SIZE: Final[int] = 32
DYNAMIC_DIGITS: Final[int] = 6
STATIC_DIGITS: Final[int] = 8
QR_CODE_SIZE: Final[int] = 24


def is_valid_static_pairing_code(code: str) -> bool:
    """Return whether ``code`` is exactly 8 ASCII decimal digits."""
    return len(code) == STATIC_DIGITS and code.isascii() and code.isdigit()


def format_pairing_code(code: str) -> str:
    """Return ``code`` grouped for presentation: ``123-456`` or ``1234-5678``.

    Raises ``ValueError`` unless ``code`` is 6 or 8 ASCII decimal digits.
    """
    if len(code) not in (DYNAMIC_DIGITS, STATIC_DIGITS) or not (code.isascii() and code.isdigit()):
        msg = f"pairing code must be {DYNAMIC_DIGITS} or {STATIC_DIGITS} ASCII digits"
        raise ValueError(msg)
    half = len(code) // 2
    return f"{code[:half]}-{code[half:]}"


def strip_separators(code: str) -> str:
    """Return ``code`` without the spaces and hyphens an operator may type between groups."""
    return code.replace(" ", "").replace("-", "")


def generate_nonce() -> bytes:
    """Return a fresh 32-byte CSPRNG nonce (``nonce_A`` or ``nonce_B``)."""
    return secrets.token_bytes(NONCE_SIZE)


def commit(nonce: bytes) -> bytes:
    """Return the commitment to ``nonce_B``."""
    _check_size(nonce, NONCE_SIZE, "nonce")
    return hashlib.sha256(COMMIT_LABEL + nonce).digest()


def verify_commit(nonce: bytes, commitment: bytes) -> bool:
    """Return whether ``commitment`` is ``commit(nonce)`` (constant-time)."""
    return hmac.compare_digest(commit(nonce), commitment)


def derive_digest(handshake_hash: bytes, nonce_a: bytes, nonce_b: bytes) -> bytes:
    """Derive the common dynamic pairing-code digest."""
    _check_size(handshake_hash, 32, "handshake_hash")
    _check_size(nonce_a, NONCE_SIZE, "nonce_a")
    _check_size(nonce_b, NONCE_SIZE, "nonce_b")
    return hashlib.sha256(PAIRING_CODE_DERIVE_LABEL + handshake_hash + nonce_a + nonce_b).digest()


def derive_digits(handshake_hash: bytes, nonce_a: bytes, nonce_b: bytes) -> str:
    """Derive the six-digit dynamic pairing code."""
    value = int.from_bytes(derive_digest(handshake_hash, nonce_a, nonce_b), "big") % 1_000_000
    return f"{value:06d}"


def derive_qr_code(handshake_hash: bytes, nonce_a: bytes, nonce_b: bytes) -> bytes:
    """Derive the 24-byte binary dynamic pairing code for QR emission."""
    return derive_digest(handshake_hash, nonce_a, nonce_b)[:QR_CODE_SIZE]


def _check_size(value: bytes, size: int, name: str) -> None:
    if len(value) != size:
        msg = f"{name} must be {size} bytes, got {len(value)}"
        raise ValueError(msg)
