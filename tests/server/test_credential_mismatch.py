"""What a credential-mismatch session may do while the server still holds the record."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

from aiosendspin.models.core import ClientHelloPayload, UnpairedAccess
from aiosendspin.models.types import Activity, PlaybackStateType
from aiosendspin.noise.keys import generate_psk, psk_id_for
from aiosendspin.noise.trust_store import PskCategory, ResolvedPsk
from aiosendspin.server.clock import LoopClock
from aiosendspin.server.connection import SendspinConnection


@dataclass(slots=True)
class _DummyServer:
    loop: asyncio.AbstractEventLoop
    clock: Any
    id: str = "srv"
    name: str = "server"


def _long_term_connection(*, credential_mismatch: bool) -> SendspinConnection:
    """Build a connection the server keyed to a long-term record, optionally mismatched."""
    loop = asyncio.get_running_loop()
    conn = SendspinConnection(
        _DummyServer(loop=loop, clock=LoopClock(loop)), wsock_client=MagicMock()
    )
    psk = generate_psk()
    conn._noise_psk = ResolvedPsk(  # noqa: SLF001
        psk_id=psk_id_for(psk), psk=psk, category=PskCategory.LONG_TERM
    )
    conn._client_info = ClientHelloPayload(  # noqa: SLF001
        name="c",
        supported_roles=["controller@v1"],
        unpaired_access=UnpairedAccess(enabled=True),
    )
    conn._negotiated_roles = ["controller@v1"]  # noqa: SLF001
    conn._credential_mismatch = credential_mismatch  # noqa: SLF001
    return conn


async def test_matched_record_activates_roles() -> None:
    """The baseline: an ordinary long-term session is playback-capable."""
    conn = _long_term_connection(credential_mismatch=False)

    assert conn._playback_capable is True  # noqa: SLF001
    assert conn._roles_to_activate == ["controller@v1"]  # noqa: SLF001


async def test_credential_mismatch_activates_no_roles() -> None:
    """A client that could not use the record gets no roles while the record stands."""
    conn = _long_term_connection(credential_mismatch=True)

    assert conn._playback_capable is False  # noqa: SLF001
    assert conn._roles_to_activate == []  # noqa: SLF001


def _put_group_in_playback(conn: SendspinConnection) -> None:
    """Give the connection a client whose group is playing, so playback is warranted."""
    client = MagicMock()
    client.group.state = PlaybackStateType.PLAYING
    conn._client = client  # noqa: SLF001


async def test_playing_group_declares_playback_when_the_record_matches() -> None:
    """The baseline: a playing group warrants the playback activity."""
    conn = _long_term_connection(credential_mismatch=False)
    _put_group_in_playback(conn)

    assert conn._client_in_playback is True  # noqa: SLF001
    assert Activity.PLAYBACK in conn._desired_activities  # noqa: SLF001


async def test_credential_mismatch_declares_no_playback() -> None:
    """Even with the group playing, a mismatched session may not declare playback."""
    conn = _long_term_connection(credential_mismatch=True)
    _put_group_in_playback(conn)

    assert conn._client_in_playback is True  # noqa: SLF001
    assert Activity.PLAYBACK not in conn._desired_activities  # noqa: SLF001


async def test_trusted_unpaired_does_not_reactivate_a_mismatched_session() -> None:
    """Granting trusted-unpaired must not route around the constraint.

    The mismatched session is Sentinel-keyed from the client's side, which is exactly
    what ``refresh_trusted_unpaired`` admits, so the gate has to hold here too.
    """
    conn = _long_term_connection(credential_mismatch=True)
    # A mismatched session is keyed to the Sentinel, which is what trusted-unpaired admits.
    psk = generate_psk()
    conn._noise_psk = ResolvedPsk(  # noqa: SLF001
        psk_id=psk_id_for(psk), psk=psk, category=PskCategory.SENTINEL
    )
    conn._trusted_unpaired = True  # noqa: SLF001

    # Without the mismatch this exact state would be playback-capable.
    assert conn._client_info is not None  # noqa: SLF001
    assert conn._client_info.unpaired_access.enabled is True  # noqa: SLF001
    assert conn._playback_capable is False  # noqa: SLF001
    assert conn._roles_to_activate == []  # noqa: SLF001


async def test_forgetting_the_client_releases_the_constraint() -> None:
    """The gate holds only while the record does: unpairing frees the session.

    An operator who answers the mismatch by forgetting the client rather than re-pairing
    leaves an ordinary unpaired client, which trusted-unpaired access may then admit.
    """
    conn = _long_term_connection(credential_mismatch=True)
    psk = generate_psk()
    conn._noise_psk = ResolvedPsk(  # noqa: SLF001
        psk_id=psk_id_for(psk), psk=psk, category=PskCategory.SENTINEL
    )
    conn._trusted_unpaired = True  # noqa: SLF001
    assert conn._playback_capable is False  # noqa: SLF001

    conn.forget_credential_mismatch()

    assert conn._playback_capable is True  # noqa: SLF001
    assert conn._roles_to_activate == ["controller@v1"]  # noqa: SLF001
