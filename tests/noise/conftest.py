"""Shared fakes and helpers for the Noise test package."""

from __future__ import annotations

import asyncio
import string

from aiohttp import WSMessage, WSMsgType

from aiosendspin.noise.keys import Identity, generate_psk
from aiosendspin.noise.session import NoiseCipherSuite, NoiseSession
from aiosendspin.noise.wire import EncryptedWebSocket

_B64URL_ALPHABET = string.ascii_uppercase + string.ascii_lowercase + string.digits + "-_"


def non_canonical_peer_id(peer_id: str) -> str:
    """Return ``peer_id`` with an unused trailing bit set, so it decodes to the same key."""
    return peer_id[:-1] + _B64URL_ALPHABET[_B64URL_ALPHABET.index(peer_id[-1]) | 1]


class FakeWebSocket:
    """In-memory ``RawWebSocket`` / ``HandshakeWebSocket`` fake.

    Standalone: sends record onto :attr:`sent` and :meth:`push` injects inbound
    frames that :meth:`receive` later drains. Linked (via :func:`make_ws_pair`):
    every send is also delivered to the peer's inbound queue, so two fakes form
    one bidirectional channel.
    """

    def __init__(self) -> None:
        """Initialize an unlinked fake with empty send log and inbound queue."""
        self.sent: list[bytes | str] = []
        self.incoming: asyncio.Queue[WSMessage | None] = asyncio.Queue()
        self.closed = False
        self.close_code: int | None = None
        self._peer: FakeWebSocket | None = None

    async def send_str(self, data: str) -> None:
        """Record a TEXT send and, if linked, deliver it to the peer."""
        self.sent.append(data)
        if self._peer is not None:
            await self._peer.push(WSMessage(WSMsgType.TEXT, data, ""))

    async def send_bytes(self, data: bytes) -> None:
        """Record a BINARY send and, if linked, deliver it to the peer."""
        self.sent.append(data)
        if self._peer is not None:
            await self._peer.push(WSMessage(WSMsgType.BINARY, data, ""))

    async def push(self, msg: WSMessage | None) -> None:
        """Inject an inbound frame; ``None`` yields a CLOSED frame on receive."""
        await self.incoming.put(msg)

    async def close_outbound(self) -> None:
        """Signal end-of-stream: a CLOSED frame on the peer's (or own) receive."""
        target = self._peer if self._peer is not None else self
        await target.push(None)

    async def receive(self) -> WSMessage:
        """Return the next inbound frame, or a CLOSED frame at end-of-stream."""
        msg = await self.incoming.get()
        if msg is None:
            return WSMessage(WSMsgType.CLOSED, None, "")
        return msg

    async def close(self) -> bool:
        """Mark the transport closed."""
        self.closed = True
        return True

    def exception(self) -> BaseException | None:
        """Return the transport exception (never set by this fake)."""
        return None


def make_ws_pair() -> tuple[FakeWebSocket, FakeWebSocket]:
    """Return two linked ``FakeWebSocket``s; each one's sends reach the other."""
    a, b = FakeWebSocket(), FakeWebSocket()
    a._peer = b  # noqa: SLF001
    b._peer = a  # noqa: SLF001
    return a, b


def make_paired_sessions(
    *,
    suite: NoiseCipherSuite = NoiseCipherSuite.CHACHAPOLY,
    prologue: bytes = b"prologue",
) -> tuple[NoiseSession, NoiseSession]:
    """Return (initiator, responder) ``NoiseSession``s both in transport mode."""
    server, client = Identity.generate(), Identity.generate()
    psk = generate_psk()
    initiator = NoiseSession.as_initiator(
        suite=suite,
        local_static_priv=server.private_bytes,
        remote_static_pub=client.public_bytes,
        prologue=prologue,
        psk=psk,
    )
    responder = NoiseSession.as_responder(
        suite=suite,
        local_static_priv=client.private_bytes,
        remote_static_pub=server.public_bytes,
        prologue=prologue,
    )
    responder.read_message(initiator.write_message(b""))
    responder.mix_psk(psk)
    initiator.read_message(responder.write_message(b""))
    return initiator, responder


def make_paired_encrypted_ws() -> tuple[
    EncryptedWebSocket,
    EncryptedWebSocket,
    FakeWebSocket,
    FakeWebSocket,
]:
    """Return (client_ews, server_ews, client_raw, server_raw), transport-mode wrappers.

    The raw ends are returned so a test can inject a poisoned frame or close a
    side's outbound to simulate an early disconnect.
    """
    initiator, responder = make_paired_sessions()
    client_raw, server_raw = make_ws_pair()
    return (
        EncryptedWebSocket(client_raw, responder),
        EncryptedWebSocket(server_raw, initiator),
        client_raw,
        server_raw,
    )
