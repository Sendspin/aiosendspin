"""Tests for :mod:`aiosendspin.noise.pairing_token`."""

import base64

import pytest

from aiosendspin.noise.keys import b64url_encode
from aiosendspin.noise.pairing_token import (
    PSKPairingToken,
    decode_pairing_code_token,
    decode_psk_token,
    encode_pairing_code_token,
    encode_psk_token,
)

# Reference vector for client_id=b64url(bytes(range(32))), pairing_psk=bytes(range(32, 64))
# — guards the wire format against accidental changes.
REFERENCE_TOKEN = (
    "SP:0AAAQEAYEAUDAOCAJBIFQYDIOB4IBCEQTCQKRMFYYDENBWHA5DYPSAIJCEMSCKJRHFAUSUKZ"  # noqa: S105 - fixed test vector, not a credential
    "MFUXC6MBRGIZTINJWG44DSOR3HQ6T4PY"
)
REFERENCE_PAIRING_TOKEN = PSKPairingToken(
    client_id=b64url_encode(bytes(range(32))),
    pairing_psk=bytes(range(32, 64)),
)

# Version-1 (dynamic pairing code) reference vector. Provenance:
# SendspinKit/Tests/SendspinKitTests/Resources/cpace-mcf-known-answer.json, dynamic_transcript;
# independently verified against cpace-py and the Sendspin spec.
REFERENCE_PAIRING_CODE = bytes.fromhex("3e2e937a82ea414f686a6155b3628640ee30d3fda85dc931")
REFERENCE_PAIRING_CODE_TOKEN = "SP:1HYXJG6UC5JAU69DKMFK3GYUGIDXDBU75VBO4SMI"  # noqa: S105 - test vector


def test_roundtrip() -> None:
    """A token survives an encode/decode round-trip."""
    token = PSKPairingToken(client_id=b64url_encode(b"\x01" * 32), pairing_psk=b"\x02" * 32)
    assert decode_psk_token(encode_psk_token(token)) == token


def test_reference_vector() -> None:
    """The codec decodes and reproduces the reference token byte-for-byte."""
    assert decode_psk_token(REFERENCE_TOKEN) == REFERENCE_PAIRING_TOKEN
    assert encode_psk_token(REFERENCE_PAIRING_TOKEN) == REFERENCE_TOKEN


def test_encoded_charset_is_qr_alphanumeric_safe() -> None:
    """Encoded tokens stay uppercase with 2 transliterated to 9 and no padding."""
    assert REFERENCE_TOKEN.upper() == REFERENCE_TOKEN
    assert "2" not in REFERENCE_TOKEN
    assert "=" not in REFERENCE_TOKEN


@pytest.mark.parametrize(
    "value",
    [
        REFERENCE_TOKEN.lower(),
        f"  {REFERENCE_TOKEN}\n",
        REFERENCE_TOKEN.removeprefix("SP:"),
        REFERENCE_TOKEN.removeprefix("SP:").lower(),
    ],
)
def test_decode_leniency(value: str) -> None:
    """Case, surrounding whitespace, and a missing SP: prefix are tolerated."""
    assert decode_psk_token(value) == REFERENCE_PAIRING_TOKEN


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ("", "malformed"),
        ("SP:", "malformed"),
        ("SP:2" + REFERENCE_TOKEN[4:], "unsupported"),
        ("SP:0NOT!VALID", "malformed"),
        ("SP:0" + REFERENCE_TOKEN[4:40], "malformed"),  # truncated payload
        ("\N{LATIN SMALL LETTER LONG S}P:0" + REFERENCE_TOKEN[4:], "malformed"),
    ],
)
def test_decode_rejects_malformed(value: str, match: str) -> None:
    """Malformed or unsupported tokens raise ValueError."""
    with pytest.raises(ValueError, match=match):
        decode_psk_token(value)


def test_decode_ignores_trailing_payload_bytes() -> None:
    """Payload bytes beyond the 64 this version defines are ignored, per the spec."""
    extended = bytes(range(32)) + bytes(range(32, 64)) + b"\xff" * 8
    body = base64.b32encode(extended).decode("ascii").rstrip("=").replace("2", "9")
    assert decode_psk_token(f"SP:0{body}") == REFERENCE_PAIRING_TOKEN


def test_encode_validates_field_sizes() -> None:
    """encode_psk_token rejects wrong-size client_id and pairing_psk."""
    with pytest.raises(ValueError, match="client_id"):
        encode_psk_token(
            PSKPairingToken(client_id=b64url_encode(b"\x01" * 16), pairing_psk=b"\x02" * 32)
        )
    with pytest.raises(ValueError, match="pairing_psk"):
        encode_psk_token(
            PSKPairingToken(client_id=b64url_encode(b"\x01" * 32), pairing_psk=b"\x02" * 16)
        )


def test_encode_rejects_non_base64url_client_id() -> None:
    """A client_id that isn't valid base64url is rejected before size checks."""
    # A single character is an invalid base64 length (1 more than a multiple of 4).
    with pytest.raises(ValueError, match="not valid base64url"):
        encode_psk_token(PSKPairingToken(client_id="A", pairing_psk=b"\x02" * 32))


def test_pairing_code_roundtrip() -> None:
    """A 24-byte pairing code survives a version-1 encode/decode round-trip."""
    code = bytes(range(24))
    assert decode_pairing_code_token(encode_pairing_code_token(code)) == code


def test_pairing_code_reference_vector() -> None:
    """The version-1 codec reproduces the reference token byte-for-byte."""
    assert encode_pairing_code_token(REFERENCE_PAIRING_CODE) == REFERENCE_PAIRING_CODE_TOKEN
    assert decode_pairing_code_token(REFERENCE_PAIRING_CODE_TOKEN) == REFERENCE_PAIRING_CODE


@pytest.mark.parametrize(
    "value",
    [
        REFERENCE_PAIRING_CODE_TOKEN.lower(),
        f"  {REFERENCE_PAIRING_CODE_TOKEN}\n",
        REFERENCE_PAIRING_CODE_TOKEN.removeprefix("SP:"),
    ],
)
def test_pairing_code_decode_leniency(value: str) -> None:
    """Case, surrounding whitespace, and a missing SP: prefix are tolerated."""
    assert decode_pairing_code_token(value) == REFERENCE_PAIRING_CODE


def test_pairing_code_decode_rejects_other_versions_and_short_payloads() -> None:
    """A version-0 token, a truncated payload, and a wrong-size code are all rejected."""
    with pytest.raises(ValueError, match="unsupported"):
        decode_pairing_code_token(REFERENCE_TOKEN)
    with pytest.raises(ValueError, match="malformed"):
        decode_pairing_code_token("SP:1AAAA")
    with pytest.raises(ValueError, match="pairing code must be"):
        encode_pairing_code_token(b"short")


def test_pairing_code_decode_ignores_trailing_payload_bytes() -> None:
    """Payload bytes beyond the 24 this version defines are ignored, per the spec."""
    extended = REFERENCE_PAIRING_CODE + b"\xff" * 8
    body = base64.b32encode(extended).decode("ascii").rstrip("=").replace("2", "9")
    assert decode_pairing_code_token(f"SP:1{body}") == REFERENCE_PAIRING_CODE


def test_decode_fuzz_single_char_mutations_never_crash() -> None:
    """Every single-character mutation of a valid token decodes cleanly or raises ValueError.

    QR/paste input is attacker-influenced, so ``decode_psk_token`` must only ever raise
    ``ValueError`` (its documented contract) — never an unhandled binascii/base64/index
    error. Deterministic: walks every position against a fixed replacement alphabet.
    """
    replacements = "ABYZ2679!@ \t/=+-.0"  # b32 chars, transliterated digits, and junk
    for pos in range(len(REFERENCE_TOKEN)):
        for repl in replacements:
            if repl == REFERENCE_TOKEN[pos]:
                continue
            mutated = REFERENCE_TOKEN[:pos] + repl + REFERENCE_TOKEN[pos + 1 :]
            try:
                result = decode_psk_token(mutated)
            except ValueError:
                continue
            # A mutation that still decodes must yield a well-formed token.
            assert len(result.pairing_psk) == 32, mutated
