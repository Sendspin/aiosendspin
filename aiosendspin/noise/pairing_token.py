"""Pairing-token encoding for the Pairing PSK and dynamic pairing-code flows."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Final

from .keys import (
    PSK_SIZE,
    X25519_KEY_SIZE,
    b64url_decode,
    b64url_encode,
)
from .pairing_code import QR_CODE_SIZE

_TOKEN_PREFIX: Final[str] = "SP:"  # noqa: S105 - format marker, not a credential
_PSK_TOKEN_VERSION: Final[str] = "0"  # noqa: S105 - format version, not a credential
_PAIRING_CODE_TOKEN_VERSION: Final[str] = "1"  # noqa: S105 - format version, not a credential
_PSK_TOKEN_SIZE: Final[int] = X25519_KEY_SIZE + PSK_SIZE


@dataclass(frozen=True, slots=True)
class PSKPairingToken:
    """Pairing Token."""

    client_id: str
    pairing_psk: bytes


def encode_psk_token(token: PSKPairingToken) -> str:
    """Encode a ``PSKPairingToken`` as a version-0 pairing token."""
    try:
        client_id = b64url_decode(token.client_id)
    except ValueError as exc:
        raise ValueError("client_id is not valid base64url") from exc
    if len(client_id) != X25519_KEY_SIZE:
        msg = f"client_id must decode to {X25519_KEY_SIZE} bytes"
        raise ValueError(msg)
    if len(token.pairing_psk) != PSK_SIZE:
        msg = f"pairing_psk must be {PSK_SIZE} bytes"
        raise ValueError(msg)
    return _encode(_PSK_TOKEN_VERSION, client_id + token.pairing_psk)


def decode_psk_token(value: str) -> PSKPairingToken:
    """Decode a version-0 pairing token (leniently), raising ``ValueError`` if malformed."""
    payload = _decode(value, expect_version=_PSK_TOKEN_VERSION)
    if len(payload) < _PSK_TOKEN_SIZE:
        raise ValueError("malformed pairing token")
    return PSKPairingToken(
        client_id=b64url_encode(payload[:X25519_KEY_SIZE]),
        pairing_psk=payload[X25519_KEY_SIZE:_PSK_TOKEN_SIZE],
    )


def encode_pairing_code_token(code: bytes) -> str:
    """Encode a 24-byte dynamic pairing code as a version-1 pairing token."""
    if len(code) != QR_CODE_SIZE:
        msg = f"pairing code must be {QR_CODE_SIZE} bytes, got {len(code)}"
        raise ValueError(msg)
    return _encode(_PAIRING_CODE_TOKEN_VERSION, code)


def decode_pairing_code_token(value: str) -> bytes:
    """Decode a version-1 pairing token (leniently) into its 24-byte pairing code."""
    payload = _decode(value, expect_version=_PAIRING_CODE_TOKEN_VERSION)
    if len(payload) < QR_CODE_SIZE:
        raise ValueError("malformed pairing token")
    return payload[:QR_CODE_SIZE]


def _encode(version: str, payload: bytes) -> str:
    body = base64.b32encode(payload).decode("ascii").rstrip("=").replace("2", "9")
    return f"{_TOKEN_PREFIX}{version}{body}"


def _decode(value: str, *, expect_version: str) -> bytes:
    text = value.strip()
    # Checked before upper-casing, which maps some non-ASCII letters to ASCII ones.
    if not text.isascii():
        raise ValueError("malformed pairing token")
    text = text.upper().removeprefix(_TOKEN_PREFIX)
    if not text:
        raise ValueError("malformed pairing token")
    version, body = text[0], text[1:]
    if version != expect_version:
        raise ValueError("unsupported pairing token version")
    pad = (-len(body)) % 8
    try:
        return base64.b32decode(body.replace("9", "2") + "=" * pad)
    except ValueError as exc:
        raise ValueError("malformed pairing token") from exc
