"""The source role may only be activated on a long-term paired connection or once approved."""

from __future__ import annotations

from aiosendspin.noise.trust_store import PskCategory, ResolvedPsk
from aiosendspin.server.connection import SendspinConnection
from aiosendspin.server.roles.registry import role_requires_pairing


def _connection_with_psk(category: PskCategory | None) -> SendspinConnection:
    # The activation filter only reads _noise_psk and the approved roles, so build a bare instance.
    conn = SendspinConnection.__new__(SendspinConnection)
    conn._noise_psk = (  # noqa: SLF001
        None if category is None else ResolvedPsk("id", b"\x00" * 32, category)
    )
    conn._approved_role_families = frozenset()  # noqa: SLF001
    return conn


def test_source_role_is_marked_pairing_required() -> None:
    """The registry records source@v1 as pairing-required (player is not)."""
    assert role_requires_pairing("source@v1") is True
    assert role_requires_pairing("player@v1") is False


def test_long_term_paired_keeps_source() -> None:
    """A long-term paired connection may activate the source role."""
    conn = _connection_with_psk(PskCategory.LONG_TERM)
    assert conn._filter_pairing_roles(["source@v1", "player@v1"]) == [  # noqa: SLF001
        "source@v1",
        "player@v1",
    ]


def test_sentinel_connection_drops_source() -> None:
    """An unpaired (sentinel) connection cannot activate the source role."""
    conn = _connection_with_psk(PskCategory.SENTINEL)
    assert conn._filter_pairing_roles(["source@v1", "player@v1"]) == ["player@v1"]  # noqa: SLF001


def test_legacy_unencrypted_connection_drops_source() -> None:
    """A legacy unencrypted connection (no PSK) cannot activate the source role."""
    conn = _connection_with_psk(None)
    assert conn._filter_pairing_roles(["source@v1", "player@v1"]) == ["player@v1"]  # noqa: SLF001
