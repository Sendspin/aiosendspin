"""Post-handshake encrypted WebSocket wrapper."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Protocol

from aiohttp import WSMessage, WSMsgType
from noise.exceptions import NoiseInvalidMessage

from .constants import (
    FRAGMENT_FLAG_FIRST,
    FRAGMENT_FLAG_LAST,
    FRAGMENT_FLAGS_RESERVED,
    MAX_TRANSPORT_PLAINTEXT,
    MSG_TYPE_FRAGMENT,
    MSG_TYPE_FRAGMENT_END,
    MSG_TYPE_FRAGMENT_MORE,
    MSG_TYPE_JSON_BODY,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from .session import NoiseSession

# Bounds a single connection's reassembly buffer against a peer streaming endless fragments.
MAX_REASSEMBLED_MESSAGE_BYTES = 64 * 1024 * 1024

# Binary types owned by the transport: 1 is a fragment and 2-3 are reserved.
_TRANSPORT_BINARY_TYPES = frozenset({MSG_TYPE_FRAGMENT, 2, 3})


class RawWebSocket(Protocol):
    """Structural typing for the subset of aiohttp WS APIs we use.

    Satisfied structurally by both ``aiohttp.web.WebSocketResponse``
    (server-side) and ``aiohttp.ClientWebSocketResponse`` (client-side); also
    by lightweight in-memory fakes in tests.
    """

    async def send_bytes(self, data: bytes) -> None:
        """Send a binary WebSocket frame."""

    async def receive(self) -> WSMessage:
        """Receive a single WebSocket message."""

    @property
    def closed(self) -> bool:
        """Whether the underlying connection has been closed."""

    @property
    def close_code(self) -> int | None:
        """The WebSocket close code once closed, or ``None``."""

    async def close(self) -> bool:
        """Close the underlying connection."""

    def exception(self) -> BaseException | None:
        """Return the exception that closed the connection, if any."""


class EncryptedWebSocket:
    """Wraps a raw WebSocket post-handshake; routes I/O through a ``NoiseSession``.

    Construct only after ``NoiseSession.handshake_complete`` is true. A message
    larger than one Noise frame is sent as binary type ``1`` fragments and
    reassembled on receive; a malformed fragment sequence surfaces as an ERROR
    message. Concurrent sends are serialized, so a fragmented message goes out whole.

    Two opt-ins keep pre-spec-#172 peers working; both are off by default:

    - ``on_legacy_fragment``: when set, legacy fragment types ``2``/``3`` are
      accepted and the callback runs once per such message. When unset they are
      delivered as ordinary binary messages, for the owner to ignore.
    - ``legacy_fragment_framing``: when true, oversized messages are sent with
      the legacy type ``2``/``3`` framing.
    """

    def __init__(self, ws: RawWebSocket, session: NoiseSession) -> None:
        """Initialize the wrapper; ``session`` must already be in transport mode."""
        if not session.handshake_complete:
            msg = "NoiseSession must be in transport mode before wrapping a WebSocket"
            raise RuntimeError(msg)
        self._ws = ws
        self._session = session
        self._reasm_buf: bytearray | None = None
        self._reasm_type: int | None = None
        self._reasm_legacy = False
        self._send_lock = asyncio.Lock()
        # DEPRECATED(spec-pr-172): remove in aiosendspin <version>
        self.on_legacy_fragment: Callable[[], None] | None = None
        # DEPRECATED(spec-pr-172): remove in aiosendspin <version>
        self.legacy_fragment_framing = False

    @property
    def session(self) -> NoiseSession:
        """The underlying ``NoiseSession`` (transport-mode)."""
        return self._session

    def swap_session(self, new_session: NoiseSession) -> None:
        """Replace the transport session with ``new_session`` after a re-handshake."""
        if not new_session.handshake_complete:
            msg = "new NoiseSession must be in transport mode before swapping"
            raise RuntimeError(msg)
        self._session = new_session

    @property
    def closed(self) -> bool:
        """Whether the underlying transport is closed."""
        return self._ws.closed

    @property
    def close_code(self) -> int | None:
        """The underlying transport's close code."""
        return self._ws.close_code

    async def close(self) -> bool:
        """Close the underlying transport."""
        return await self._ws.close()

    def exception(self) -> BaseException | None:
        """Return the underlying transport's exception, if any."""
        return self._ws.exception()

    async def send_str(self, data: str) -> None:
        """Encrypt and send a JSON control body (transport type byte ``0``)."""
        await self._send_plaintext(bytes([MSG_TYPE_JSON_BODY]) + data.encode("utf-8"))

    async def send_bytes(self, data: bytes) -> None:
        """Encrypt and send pre-typed binary data (caller has prefixed the role type byte).

        Raises ``ValueError`` for an empty payload or a type byte the transport reserves (1-3).
        """
        if not data:
            msg = "binary payload must include a leading type byte"
            raise ValueError(msg)
        if data[0] in _TRANSPORT_BINARY_TYPES:
            msg = f"binary type {data[0]} is reserved for the transport"
            raise ValueError(msg)
        await self._send_plaintext(data)

    async def _send_plaintext(self, plaintext: bytes) -> None:
        """Encrypt and send ``plaintext``, fragmenting it if it exceeds one frame."""
        async with self._send_lock:
            if len(plaintext) <= MAX_TRANSPORT_PLAINTEXT:
                await self._ws.send_bytes(self._session.encrypt(plaintext))
                return
            # DEPRECATED(spec-pr-172): remove in aiosendspin <version>
            fragment = _fragment_legacy if self.legacy_fragment_framing else _fragment
            for frame in fragment(plaintext):
                await self._ws.send_bytes(self._session.encrypt(frame))

    def __aiter__(self) -> EncryptedWebSocket:
        """Iterate decrypted messages (the wrapper is its own async iterator)."""
        return self

    async def __anext__(self) -> WSMessage:
        """Return the next decrypted ``WSMessage``, raising on close."""
        msg = await self.receive()
        if msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED):
            raise StopAsyncIteration
        return msg

    async def receive(self) -> WSMessage:
        """Return the next decrypted message, reassembling fragments across frames.

        Protocol violations are returned as an ERROR message. An exception raised by
        the pre-spec-#172 ``on_legacy_fragment`` callback propagates to the caller.
        """
        while True:
            raw = await self._ws.receive()
            if raw.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED):
                return raw
            decoded = self._decode(raw)
            if decoded is not None:
                return decoded

    def _decode(self, msg: WSMessage) -> WSMessage | None:
        """Decrypt one frame; return a ``WSMessage`` or ``None`` to await more fragments."""
        if msg.type is WSMsgType.BINARY:
            try:
                plaintext = self._session.decrypt(msg.data)
            except NoiseInvalidMessage:
                return self._error("Noise transport frame failed authentication")
            if not plaintext:
                return self._error("empty plaintext after Noise decrypt")
            type_byte = plaintext[0]
            if type_byte == MSG_TYPE_FRAGMENT:
                return self._on_fragment(plaintext)
            # DEPRECATED(spec-pr-172): remove in aiosendspin <version>
            if (
                type_byte in (MSG_TYPE_FRAGMENT_MORE, MSG_TYPE_FRAGMENT_END)
                and self.on_legacy_fragment is not None
            ):
                return self._on_legacy_fragment(plaintext)
            if self._reasm_buf is not None:
                return self._error("non-fragment frame while a fragmented message is in flight")
            return self._dispatch(plaintext)
        if msg.type is WSMsgType.ERROR:
            return msg
        return self._error(f"unexpected {msg.type.name} frame after Noise handshake")

    def _on_fragment(self, plaintext: bytes) -> WSMessage | None:
        """Handle a type ``1`` fragment frame; return the message once its last frame arrives."""
        if len(plaintext) < 2:
            return self._error("fragment frame missing flags byte")
        flags = plaintext[1]
        if flags & FRAGMENT_FLAGS_RESERVED:
            return self._error(f"fragment flags 0x{flags:02x} set reserved bits")
        if flags & FRAGMENT_FLAG_FIRST:
            if self._reasm_buf is not None:
                return self._error("first fragment while a fragmented message is in flight")
            if len(plaintext) < 3:
                return self._error("first fragment missing orig_type")
            if plaintext[2] == MSG_TYPE_FRAGMENT:
                return self._error("first fragment has orig_type 1")
            self._start_reassembly(plaintext[2], legacy=False)
            data = plaintext[3:]
        elif self._reasm_buf is None:
            return self._error("continuation fragment with no fragmented message in flight")
        elif self._reasm_legacy:
            return self._error("type 1 fragment inside a legacy fragmented message")
        else:
            data = plaintext[2:]
        return self._append_fragment(data, last=bool(flags & FRAGMENT_FLAG_LAST))

    # DEPRECATED(spec-pr-172): remove in aiosendspin <version>
    def _on_legacy_fragment(self, plaintext: bytes) -> WSMessage | None:
        """Handle a legacy type ``2``/``3`` fragment frame; the owner has opted in."""
        assert self.on_legacy_fragment is not None
        last = plaintext[0] == MSG_TYPE_FRAGMENT_END
        if self._reasm_buf is None:
            if last:
                return self._error("legacy fragment-end frame with no fragmented message in flight")
            if len(plaintext) < 2:
                return self._error("legacy fragment start frame missing orig_type")
            if plaintext[1] in _TRANSPORT_BINARY_TYPES:
                return self._error(f"legacy fragment has reserved orig_type {plaintext[1]}")
            self.on_legacy_fragment()
            self._start_reassembly(plaintext[1], legacy=True)
            return self._append_fragment(plaintext[2:], last=False)
        if not self._reasm_legacy:
            return self._error("legacy fragment inside a type 1 fragmented message")
        return self._append_fragment(plaintext[1:], last=last)

    def _start_reassembly(self, orig_type: int, *, legacy: bool) -> None:
        """Begin buffering a fragmented message of ``orig_type``."""
        self._reasm_buf = bytearray()
        self._reasm_type = orig_type
        self._reasm_legacy = legacy

    def _append_fragment(self, data: bytes, *, last: bool) -> WSMessage | None:
        """Append to the in-flight message; dispatch it when ``last`` is set."""
        assert self._reasm_buf is not None
        assert self._reasm_type is not None
        if len(self._reasm_buf) + len(data) > MAX_REASSEMBLED_MESSAGE_BYTES:
            return self._error("fragmented message exceeds maximum reassembly size")
        self._reasm_buf += data
        if not last:
            return None
        reassembled = bytes([self._reasm_type]) + bytes(self._reasm_buf)
        self._reset_reassembly()
        return self._dispatch(reassembled)

    def _reset_reassembly(self) -> None:
        """Drop any in-flight fragmented message."""
        self._reasm_buf = None
        self._reasm_type = None
        self._reasm_legacy = False

    def _dispatch(self, plaintext: bytes) -> WSMessage:
        """Synthesize a ``WSMessage`` from a complete, type-prefixed plaintext."""
        if plaintext[0] == MSG_TYPE_JSON_BODY:
            return WSMessage(WSMsgType.TEXT, plaintext[1:].decode("utf-8"), "")
        return WSMessage(WSMsgType.BINARY, plaintext, "")

    def _error(self, message: str) -> WSMessage:
        """Discard any reassembly state and build an ERROR ``WSMessage``."""
        self._reset_reassembly()
        return WSMessage(WSMsgType.ERROR, RuntimeError(message), "")


class QueuedEncryptedWebSocket(EncryptedWebSocket):
    """View of an ``EncryptedWebSocket`` that receives messages its owner's reader routed to it.

    Sends and session swaps go through ``base``.
    """

    def __init__(self, base: EncryptedWebSocket, queue: asyncio.Queue[WSMessage]) -> None:
        """Initialize the view; ``queue`` supplies the messages ``receive()`` returns."""
        super().__init__(base._ws, base._session)  # noqa: SLF001
        self._base = base
        self._queue = queue

    @property
    def session(self) -> NoiseSession:
        """The base transport's current session."""
        return self._base.session

    def swap_session(self, new_session: NoiseSession) -> None:
        """Swap the base transport's session."""
        self._base.swap_session(new_session)

    async def _send_plaintext(self, plaintext: bytes) -> None:
        await self._base._send_plaintext(plaintext)  # noqa: SLF001

    async def receive(self) -> WSMessage:
        """Return the next routed message."""
        return await self._queue.get()


def _fragment(plaintext: bytes) -> list[bytes]:
    """Split an oversized type-prefixed plaintext into type ``1`` fragment frames."""
    orig_type = plaintext[0]
    data = memoryview(plaintext)[1:]
    first_cap = MAX_TRANSPORT_PLAINTEXT - 3
    cont_cap = MAX_TRANSPORT_PLAINTEXT - 2

    chunks = [data[:first_cap]]
    chunks += [data[i : i + cont_cap] for i in range(first_cap, len(data), cont_cap)]
    last = len(chunks) - 1
    frames = []
    for index, chunk in enumerate(chunks):
        flags = FRAGMENT_FLAG_LAST if index == last else 0
        if index == 0:
            header = bytes([MSG_TYPE_FRAGMENT, flags | FRAGMENT_FLAG_FIRST, orig_type])
        else:
            header = bytes([MSG_TYPE_FRAGMENT, flags])
        frames.append(header + bytes(chunk))
    return frames


# DEPRECATED(spec-pr-172): remove in aiosendspin <version>
def _fragment_legacy(plaintext: bytes) -> list[bytes]:
    """Split an oversized type-prefixed plaintext into legacy type ``2``/``3`` frames."""
    orig_type = plaintext[0]
    data = memoryview(plaintext)[1:]
    first_cap = MAX_TRANSPORT_PLAINTEXT - 2
    cont_cap = MAX_TRANSPORT_PLAINTEXT - 1

    frames = [bytes([MSG_TYPE_FRAGMENT_MORE, orig_type]) + bytes(data[:first_cap])]
    rest = data[first_cap:]
    chunks = [rest[i : i + cont_cap] for i in range(0, len(rest), cont_cap)]
    last = len(chunks) - 1
    for index, chunk in enumerate(chunks):
        tag = MSG_TYPE_FRAGMENT_END if index == last else MSG_TYPE_FRAGMENT_MORE
        frames.append(bytes([tag]) + bytes(chunk))
    return frames
