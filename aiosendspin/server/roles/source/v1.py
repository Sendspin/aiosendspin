"""Source role implementation: decode audio captured by a source client."""

from __future__ import annotations

import base64
import binascii
import logging
from typing import TYPE_CHECKING

from aiosendspin.audio.codecs import create_decoder
from aiosendspin.audio.format import AudioFormat
from aiosendspin.models.core import ServerCommandMessage, ServerCommandPayload
from aiosendspin.models.source import SourceCommandServerPayload
from aiosendspin.models.types import AudioCodec, BinaryMessageType
from aiosendspin.server.roles.base import Role

from .events import (
    SourceSignalChangedEvent,
    SourceStreamEndedEvent,
    SourceStreamStartedEvent,
)
from .stream import SourceStream

if TYPE_CHECKING:
    from aiosendspin.models.core import ClientStatePayload
    from aiosendspin.models.source import ClientStreamStartPayload
    from aiosendspin.models.types import SignalState
    from aiosendspin.server.client import SendspinClient

logger = logging.getLogger(__name__)


class SourceV1Role(Role):
    """Per-connection role that decodes audio streamed up by a source client."""

    def __init__(self, client: SendspinClient | None = None) -> None:
        """Initialize the source role."""
        if client is None:
            raise ValueError("SourceV1Role requires a client")
        self._client = client
        self._group_role = None
        self._decoder: object | None = None
        self._stream: SourceStream | None = None
        self._stream_active = False
        self._initial_state_received = False
        # Require a fresh start request after reconnecting.
        self._start_requested = False
        # Stamp decoder output produced during flush.
        self._last_timestamp_us = 0
        self._signal: SignalState | None = None

    @property
    def role_id(self) -> str:
        """Versioned role identifier."""
        return "source@v1"

    @property
    def stream_active(self) -> bool:
        """Whether the client currently has an open input stream."""
        return self._stream_active

    def handles_inbound_binary(self, message_type: int) -> bool:
        """Return whether this role consumes the source audio message type."""
        return message_type == BinaryMessageType.SOURCE_AUDIO_CHUNK.value

    @property
    def role_family(self) -> str:
        """Role family name for protocol messages."""
        return "source"

    def on_connect(self) -> None:
        """Connect without a group role."""

    def on_disconnect(self) -> None:
        """End any active stream so a waiting consumer is released."""
        self._end_stream()
        self._initial_state_received = False
        self._start_requested = False
        self._signal = None

    def on_deactivate(self) -> None:
        """End any active stream when the role leaves active_roles."""
        self._end_stream()
        self._initial_state_received = False
        self._start_requested = False
        self._signal = None
        super().on_deactivate()

    def requires_initial_state(self) -> bool:
        """Require synchronized client state before accepting captured audio."""
        return True

    def request_start(self) -> None:
        """Ask the source client to begin streaming (server/command: start)."""
        self._start_requested = True
        self.send_message(
            ServerCommandMessage(
                payload=ServerCommandPayload(source=SourceCommandServerPayload(command="start"))
            )
        )

    def request_stop(self) -> None:
        """Ask the source client to stop streaming (server/command: stop)."""
        self._start_requested = False
        self.send_message(
            ServerCommandMessage(
                payload=ServerCommandPayload(source=SourceCommandServerPayload(command="stop"))
            )
        )

    def on_client_stream_start(self, payload: ClientStreamStartPayload) -> None:
        """Build a decoder and a fresh stream handle, then announce it."""
        if not self._start_requested:
            # Let server compliance policy decide whether to disconnect.
            self._client.flag_noncompliance(
                "client-stream/start sent without a preceding source start command"
            )
            return
        source = payload.source
        if self._stream_active:
            self._end_stream()

        # The spec ignores bit_depth for opus, so decode at the canonical 16 bits.
        bit_depth = 16 if source.codec is AudioCodec.OPUS else source.bit_depth
        audio_format = AudioFormat(
            sample_rate=source.sample_rate,
            bit_depth=bit_depth,
            channels=source.channels,
        )
        header = None
        if source.codec_header is not None:
            try:
                header = base64.b64decode(source.codec_header, validate=True)
            except (binascii.Error, ValueError):
                self._client.flag_noncompliance(
                    "client-stream/start codec_header is not valid Base64"
                )
                return
        if source.codec is AudioCodec.FLAC and (
            header is None
            or len(header) < 42
            or header[:4] != b"fLaC"
            or header[4] & 0x7F
            or int.from_bytes(header[5:8], "big") != 34
        ):
            self._client.flag_noncompliance(
                "client-stream/start FLAC codec_header must contain STREAMINFO"
            )
            return
        try:
            # Validate formats before exposing a stream handle.
            if source.sample_rate <= 0:
                msg = f"Unsupported sample rate: {source.sample_rate}"
                raise ValueError(msg)  # noqa: TRY301
            audio_format.resolve_av_format()
            self._decoder = create_decoder(
                source.codec.value,
                sample_rate=source.sample_rate,
                bit_depth=bit_depth,
                channels=source.channels,
                codec_header=header,
            )
        except (ValueError, ImportError):
            logger.exception("Failed to build source decoder for codec %r", source.codec)
            self._decoder = None
            return

        self._stream = SourceStream(audio_format)
        self._stream_active = True
        self._client._signal_event(  # noqa: SLF001
            SourceStreamStartedEvent(audio_format=audio_format, handle=self._stream)
        )

    def on_binary_chunk(self, message_type: int, timestamp_us: int, data: bytes) -> None:  # noqa: ARG002
        """Decode a source audio chunk into the active stream."""
        if (
            not self._initial_state_received
            or not self._stream_active
            or self._stream is None
            or self._decoder is None
        ):
            return
        if not self._client.available:
            return
        try:
            pcm = self._decoder.decode(data)  # type: ignore[attr-defined]
        except Exception:
            logger.exception("Failed to decode source audio chunk")
            return
        # Keep the flush-tail stamp monotonic even if a chunk arrives out of order.
        self._last_timestamp_us = max(self._last_timestamp_us, timestamp_us)
        self._stream._push(pcm, timestamp_us)  # noqa: SLF001

    def on_client_stream_end(self) -> None:
        """End the active stream and release its decoder."""
        self._end_stream()

    def _end_stream(self) -> None:
        """Drain and close the active stream."""
        was_active = self._stream_active
        if self._stream is not None and self._decoder is not None:
            try:
                tail = self._decoder.flush()  # type: ignore[attr-defined]
            except Exception:
                logger.exception("Failed to flush source decoder")
                tail = b""
            self._stream._push(tail, self._last_timestamp_us)  # noqa: SLF001
            self._stream._end()  # noqa: SLF001
        self._stream = None
        self._decoder = None
        self._stream_active = False
        self._last_timestamp_us = 0
        if was_active:
            self._client._signal_event(SourceStreamEndedEvent())  # noqa: SLF001

    def on_client_state(self, payload: ClientStatePayload) -> None:
        """Surface a source's signal presence, only when it advertised line_sense."""
        if payload.available is not None:
            self._initial_state_received = True
        source = payload.source
        if source is None or source.signal is None or not self._line_sense_supported():
            return
        if source.signal == self._signal:
            return
        self._signal = source.signal
        self._client._signal_event(  # noqa: SLF001
            SourceSignalChangedEvent(signal=source.signal)
        )

    def on_availability_changed(
        self,
        old_available: bool,  # noqa: ARG002, FBT001
        new_available: bool,  # noqa: FBT001
    ) -> None:
        """Treat becoming unavailable as an implicit source stop."""
        if new_available:
            return
        self._start_requested = False
        self._end_stream()

    def _line_sense_supported(self) -> bool:
        """Whether the source advertised the 'line_sense' feature in client/hello."""
        support = self._client.info.source_support
        return (
            support is not None
            and support.features is not None
            and bool(support.features.line_sense)
        )
