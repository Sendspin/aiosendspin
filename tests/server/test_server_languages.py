"""Tests for the SendspinServer operator languages option."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest

from aiosendspin.noise.keys import Identity
from aiosendspin.noise.trust_store import InMemoryServerPairingStore
from aiosendspin.server.server import SendspinServer


def _make_server(languages: Sequence[str] | None) -> SendspinServer:
    return SendspinServer(
        loop=asyncio.get_running_loop(),
        identity=Identity.generate(),
        server_name="server",
        pairing_store=InMemoryServerPairingStore(),
        languages=languages,
    )


@pytest.mark.parametrize(
    ("languages", "expected"),
    [(["ca", "es", "en"], ("ca", "es", "en")), (None, None)],
)
async def test_languages_are_stored_in_preference_order(
    languages: list[str] | None, expected: tuple[str, ...] | None
) -> None:
    """The declared languages are kept as given, and default to undeclared."""
    server = _make_server(languages)
    try:
        assert server.languages == expected
    finally:
        await server.close()


@pytest.mark.parametrize(
    ("languages", "match"),
    [([], "must not be empty"), (["en", ""], "blank tag"), (["en", " "], "blank tag")],
)
async def test_invalid_languages_are_rejected(languages: list[str], match: str) -> None:
    """The spec requires a non-empty list of BCP 47 tags."""
    with pytest.raises(ValueError, match=match):
        _make_server(languages)


async def test_a_single_string_is_not_a_language_list() -> None:
    """A bare tag would otherwise be split into one-character tags."""
    with pytest.raises(TypeError, match="not a string"):
        _make_server("en")
