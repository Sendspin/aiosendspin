"""Tests for pairing-code derivation and commitment (:mod:`aiosendspin.noise.pairing_code`).

The known-answer values are computed from the spec formula over fixed inputs
(``h = 0x00..1f``, ``nonce_A = 0x01*32``, ``nonce_B = 0x02*32``) and pinned here
to catch regressions in the derivation.
"""

from __future__ import annotations

import pytest

from aiosendspin.noise.pairing_code import (
    COMMIT_SIZE,
    NONCE_SIZE,
    QR_CODE_SIZE,
    commit,
    derive_digits,
    derive_qr_code,
    format_pairing_code,
    generate_nonce,
    verify_commit,
)

H = bytes(range(32))
NONCE_A = bytes([1]) * 32
NONCE_B = bytes([2]) * 32


def test_derive_digits_known_answer() -> None:
    """Dynamic digits are always the six-digit spec derivation."""
    assert derive_digits(H, NONCE_A, NONCE_B) == "305673"


def test_derive_digits_is_fixed_width_and_input_sensitive() -> None:
    """The six-digit code changes when any derivation input changes."""
    base = derive_digits(H, NONCE_A, NONCE_B)
    assert len(base) == 6
    assert base.isascii()
    assert base.isdigit()
    assert derive_digits(bytes([9]) * 32, NONCE_A, NONCE_B) != base
    assert derive_digits(H, bytes([9]) * 32, NONCE_B) != base
    assert derive_digits(H, NONCE_A, bytes([9]) * 32) != base


def test_derive_qr_code_is_the_digest_prefix() -> None:
    """The qr_code form is the first 24 bytes of the same digest."""
    code = derive_qr_code(H, NONCE_A, NONCE_B)
    assert len(code) == QR_CODE_SIZE
    assert derive_qr_code(H, NONCE_A, bytes([9]) * 32) != code


# Provenance: SendspinKit/Tests/SendspinKitTests/Resources/cpace-mcf-known-answer.json,
# dynamic_transcript; independently verified against cpace-py and the Sendspin spec.
def test_sendspinkit_derivation_vector() -> None:
    """Both forms match the independently generated SendspinKit dynamic transcript."""
    handshake_hash = bytes.fromhex(
        "00112233445566778899aabbccddeeff102132435465768798a9bacbdcedfe0f"
    )
    nonce_a = bytes.fromhex("101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f")
    nonce_b = bytes.fromhex("303132333435363738393a3b3c3d3e3f404142434445464748494a4b4c4d4e4f")

    assert derive_digits(handshake_hash, nonce_a, nonce_b) == "268386"
    qr_code = derive_qr_code(handshake_hash, nonce_a, nonce_b)
    assert qr_code.hex() == "3e2e937a82ea414f686a6155b3628640ee30d3fda85dc931"


def test_commit_known_answer_and_size() -> None:
    """Commit returns the pinned SHA-256 digest of the labeled nonce."""
    digest = commit(NONCE_B)
    assert len(digest) == COMMIT_SIZE
    assert digest.hex() == "03882b6c6c622d0d347626dad0e7957853e3cd5830163a2688823bba9443dbca"


def test_verify_commit_round_trip() -> None:
    """verify_commit accepts the matching nonce and rejects others."""
    nonce = generate_nonce()
    assert len(nonce) == NONCE_SIZE
    assert verify_commit(nonce, commit(nonce))
    assert not verify_commit(generate_nonce(), commit(nonce))


def test_size_validation() -> None:
    """Nonces of the wrong length are rejected by the derivation helpers."""
    with pytest.raises(ValueError, match="nonce"):
        commit(b"short")
    with pytest.raises(ValueError, match="nonce_a"):
        derive_digits(H, b"short", NONCE_B)
    with pytest.raises(ValueError, match="nonce_b"):
        derive_qr_code(H, NONCE_A, b"short")


@pytest.mark.parametrize(
    ("code", "grouped"),
    [("123456", "123-456"), ("12345678", "1234-5678"), ("000000", "000-000")],
)
def test_format_pairing_code_groups_the_digits(code: str, grouped: str) -> None:
    """The dynamic code is presented 3-3 and the static one 4-4."""
    assert format_pairing_code(code) == grouped


@pytest.mark.parametrize(
    "code",
    [
        "",
        "12345",
        "1234567",
        "123456789",
        "123-456",
        "12345a",
        "\u0661\u0662\u0663\u0664\u0665\u0666",  # Arabic-Indic digits
        "\uff11\uff12\uff13\uff14\uff15\uff16",  # fullwidth digits
    ],
)
def test_format_pairing_code_rejects_anything_but_6_or_8_ascii_digits(code: str) -> None:
    """Other lengths, separators and non-ASCII digits are not a pairing code."""
    with pytest.raises(ValueError, match="ASCII digits"):
        format_pairing_code(code)
