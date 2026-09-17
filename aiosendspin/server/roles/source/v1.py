"""Source role implementation: decode audio captured by a source client."""

from __future__ import annotations

import base64
import binascii
import logging
from typing import TYPE_CHECKING

from aiosendspin.audio.codecs import create_decoder, decoded_bit_depth, opus_available
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
        # The source object of the client/state this activation requires.
        self._source_state_received = False
        # A start request waiting for can_start; see request_start().
        self._start_queued = False
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

    @staticmethod
    def accepted_codecs() -> list[AudioCodec]:
        """Codecs accepted in client-stream/start, as listed in server/hello."""
        codecs = [AudioCodec.FLAC, AudioCodec.PCM]
        if opus_available():
            codecs.append(AudioCodec.OPUS)
        return codecs

    def on_connect(self) -> None:
        """Connect without a group role."""

    def on_disconnect(self) -> None:
        """End any active stream so a waiting consumer is released."""
        self._end_stream()
        self._source_state_received = False
        self._start_queued = False
        self._start_requested = False
        self._signal = None

    def on_deactivate(self) -> None:
        """End any active stream when the role leaves active_roles."""
        self._end_stream()
        self._source_state_received = False
        self._start_queued = False
        self._start_requested = False
        self._signal = None
        super().on_deactivate()

    def requires_initial_state(self) -> bool:
        """Require synchronized client state before accepting captured audio."""
        return True

    def initial_state_deviations(self, payload: ClientStatePayload) -> list[str]:
        """Report a client/state that lacks the source object its activation requires."""
        if payload.source is None:
            return ["has an active source role but no source state"]
        return []

    # DEPRECATED(spec-pr-195): remove in aiosendspin <version>
    def on_initial_client_state(self, payload: ClientStatePayload) -> None:  # noqa: ARG002
        """Record the initial client/state, which allows a start even without a source object."""
        # A pre-#195 client may never send the source object. Only its initial client/state
        # stands in, flagged by initial_state_deviations, which a strict server rejects.
        self._source_state_received = True

    def on_hold_released(self) -> None:
        """Send a queued start now that the role's client/state hold is over."""
        self._send_queued_start()

    @property
    def can_start(self) -> bool:
        """Whether `request_start()` may send a start command now."""
        return (
            self._source_state_received
            and self._client.available
            and not self._client.awaits_role_state(self.role_family)
        )

    def request_start(self) -> None:
        """
        Ask the source client to begin streaming (server/command: start).

        While `can_start` is false the request is queued and sent once it turns true.
        `request_stop()`, a disconnect or the role's deactivation cancels it.
        """
        self._start_queued = True
        self._send_queued_start()

    def request_stop(self) -> None:
        """Ask the source client to stop streaming (server/command: stop)."""
        self._start_queued = False
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

        if source.codec not in self.accepted_codecs():
            self._client.flag_noncompliance(
                f"client-stream/start announced codec {source.codec.value!r}, "
                "which server/hello did not list"
            )
            return
        if source.codec is not AudioCodec.OPUS and not 1 <= source.bit_depth <= 32:
            self._client.flag_noncompliance(
                f"client-stream/start announced unsupported bit_depth {source.bit_depth}"
            )
            return
        if source.codec is AudioCodec.PCM and source.bit_depth % 8:
            # The PCM wire convention only packs whole-byte samples.
            self._client.flag_noncompliance(
                f"client-stream/start announced pcm bit_depth {source.bit_depth}, "
                "which is not a whole number of bytes"
            )
            return
        audio_format = AudioFormat(
            sample_rate=source.sample_rate,
            bit_depth=decoded_bit_depth(source.codec.value, source.bit_depth),
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
                bit_depth=source.bit_depth,
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
            not self._source_state_received
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
        """Send a queued start the state allows, and surface a line_sense signal change."""
        source = payload.source
        if source is not None:
            self._source_state_received = True
        self._send_queued_start()
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

    def _send_queued_start(self) -> None:
        """Send the queued start command once `can_start` allows it."""
        if not self._start_queued or not self.can_start:
            return
        self._start_queued = False
        self._start_requested = True
        self.send_message(
            ServerCommandMessage(
                payload=ServerCommandPayload(source=SourceCommandServerPayload(command="start"))
            )
        )

    def _line_sense_supported(self) -> bool:
        """Whether the source advertised the 'line_sense' feature in client/hello."""
        support = self._client.info.source_support
        return (
            support is not None
            and support.features is not None
            and bool(support.features.line_sense)
        )
