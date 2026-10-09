"""Source role implementation: decode audio captured by a source client."""

from __future__ import annotations

import base64
import binascii
import logging
import time
from typing import TYPE_CHECKING

from aiosendspin.audio.codecs import create_decoder, decoded_bit_depth, opus_available
from aiosendspin.audio.format import AudioFormat
from aiosendspin.models.core import ServerCommandMessage, ServerCommandPayload
from aiosendspin.models.source import SourceCommandServerPayload
from aiosendspin.models.types import AudioCodec, BinaryMessageType
from aiosendspin.server.roles.base import Role
from aiosendspin.util import WARN_INTERVAL_S

from .events import (
    SourceSignalChangedEvent,
    SourceStreamEndedEvent,
    SourceStreamStartedEvent,
)
from .stream import SourceStream

if TYPE_CHECKING:
    from collections.abc import Callable

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
        self._pcm_frame_bytes: int | None = None
        self._max_chunk_bytes = 0
        self._stream: SourceStream | None = None
        self._stream_active = False
        # The source object of the client/state this activation requires.
        self._source_state_received = False
        # A start request waiting for can_start; see request_start().
        self._start_queued = False
        # A start was sent after the latest stop, unavailability, removal or disconnect.
        self._stream_wanted = False
        # Stamp decoder output produced during flush.
        self._last_timestamp_us = 0
        self._signal: SignalState | None = None
        self._decode_error_count = 0
        self._last_decode_error_log_s: float | None = None

    @property
    def role_id(self) -> str:
        """Versioned role identifier."""
        return "source@v1"

    @property
    def stream_active(self) -> bool:
        """Whether a stream handle is open for consumers."""
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
        self._stream_wanted = False
        self._signal = None

    def on_deactivate(self) -> None:
        """End any active stream when the role leaves active_roles."""
        self._end_stream()
        self._source_state_received = False
        self._start_queued = False
        self._stream_wanted = False
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
        self._stream_wanted = False
        self.send_message(
            ServerCommandMessage(
                payload=ServerCommandPayload(source=SourceCommandServerPayload(command="stop"))
            )
        )

    @staticmethod
    def check_client_stream_start(
        payload: ClientStreamStartPayload, *, on_noncompliance: Callable[[str], None]
    ) -> tuple[AudioFormat, bytes | None] | None:
        """Flag spec violations and return the format and codec header, or None to discard."""
        source = payload.source
        if source.codec not in SourceV1Role.accepted_codecs():
            on_noncompliance(
                f"client-stream/start announced codec {source.codec.value!r}, "
                "which server/hello did not list"
            )
            return None
        if source.codec is not AudioCodec.OPUS and not 1 <= source.bit_depth <= 32:
            on_noncompliance("client-stream/start announced an unsupported bit_depth")
            return None
        if source.codec is AudioCodec.PCM and source.bit_depth % 8:
            # The PCM wire convention only packs whole-byte samples.
            on_noncompliance(
                "client-stream/start announced a pcm bit_depth that is not a whole number of bytes"
            )
            return None
        if source.sample_rate <= 0:
            on_noncompliance("client-stream/start announced an unsupported sample_rate")
            return None
        if source.channels <= 0:
            on_noncompliance("client-stream/start announced an unsupported channels count")
            return None
        audio_format = AudioFormat(
            sample_rate=source.sample_rate,
            bit_depth=decoded_bit_depth(source.codec.value, source.bit_depth),
            channels=source.channels,
        )
        header = None
        if source.codec_header is not None:
            if source.codec is not AudioCodec.FLAC:
                on_noncompliance(
                    f"client-stream/start sent a codec_header for {source.codec.value}"
                )
            try:
                header = base64.b64decode(source.codec_header, validate=True)
            except (binascii.Error, ValueError):
                on_noncompliance("client-stream/start codec_header is not valid Base64")
                return None
        if source.codec is AudioCodec.FLAC and (
            header is None
            or len(header) < 42
            or header[:4] != b"fLaC"
            or header[4] & 0x7F
            or int.from_bytes(header[5:8], "big") != 34
        ):
            on_noncompliance("client-stream/start FLAC codec_header must contain STREAMINFO")
            return None
        return audio_format, header

    def on_client_stream_start(self, payload: ClientStreamStartPayload) -> None:
        """Build a decoder and a fresh stream handle, then announce it, if a stream is wanted."""
        source = payload.source
        if self._stream_active:
            self._end_stream()

        checked = self.check_client_stream_start(
            payload, on_noncompliance=self._client.flag_noncompliance
        )
        if checked is None:
            return
        audio_format, header = checked
        if not self._stream_wanted:
            # A response to a start that crossed a stop, unavailability or removal: discard it.
            return
        try:
            # Validate formats before exposing a stream handle.
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

        if source.codec is AudioCodec.PCM:
            self._pcm_frame_bytes = source.bit_depth // 8 * source.channels
        else:
            self._pcm_frame_bytes = None
        # The longest chunk allowed is 150 ms of frames, measured on the decoded PCM.
        decoded_frame_bytes = audio_format.bit_depth // 8 * audio_format.channels
        self._max_chunk_bytes = source.sample_rate * 3 // 20 * decoded_frame_bytes
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
        if self._pcm_frame_bytes is not None and len(data) % self._pcm_frame_bytes:
            self._client.flag_noncompliance(
                "sent a pcm source audio chunk that is not a whole number of frames"
            )
        if self._pcm_frame_bytes is None and not data:
            # An empty packet flushes the decoder and ends decoding for the stream.
            self._client.flag_noncompliance("sent an empty flac or opus source audio chunk")
            return
        try:
            pcm = self._decoder.decode(data)  # type: ignore[attr-defined]
        except Exception as err:
            self._client.flag_noncompliance("sent a source audio chunk that failed to decode")
            self._decode_error_count += 1
            now_s = time.monotonic()
            last_log_s = self._last_decode_error_log_s
            if last_log_s is None or now_s - last_log_s >= WARN_INTERVAL_S:
                logger.warning(
                    "Failed to decode %s source audio chunk(s) since last report: %s",
                    self._decode_error_count,
                    err,
                    exc_info=self._decode_error_count == 1,
                )
                self._decode_error_count = 0
                self._last_decode_error_log_s = now_s
            return
        if len(pcm) > self._max_chunk_bytes:
            self._client.flag_noncompliance("sent a source audio chunk longer than 150 ms")
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
        self._decode_error_count = 0
        self._last_decode_error_log_s = None
        if was_active:
            self._client._signal_event(SourceStreamEndedEvent())  # noqa: SLF001

    def client_state_deviations(self, payload: ClientStatePayload) -> list[str]:
        """Report a line_sense signal from a source that did not advertise line_sense."""
        if (
            payload.source is not None
            and payload.source.signal is not None
            and not self._line_sense_supported()
        ):
            return ["source signal sent without line_sense support"]
        return []

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
        """Send a queued start once available, and treat becoming unavailable as a stop."""
        if new_available:
            self._send_queued_start()
            return
        self._stream_wanted = False
        self._end_stream()

    def _send_queued_start(self) -> None:
        """Send the queued start command once `can_start` allows it."""
        if not self._start_queued or not self.can_start:
            return
        self._start_queued = False
        self._stream_wanted = True
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
