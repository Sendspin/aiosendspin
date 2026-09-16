"""Audio codec encoders and decoders."""

from __future__ import annotations

import functools
import logging
import struct
import types
from collections.abc import Mapping
from typing import TYPE_CHECKING

from aiosendspin.audio.format import (
    AudioFormat,
    _convert_s32_to_s24,
    _get_av,
    _validate_pcm_buffer_length,
)

if TYPE_CHECKING:
    import av

logger = logging.getLogger(__name__)


def _require_av() -> types.ModuleType:
    """Return the av module or raise a friendly error if the extra is missing."""
    try:
        return _get_av()
    except ImportError as err:
        raise ImportError(
            "PyAV is required for Opus/FLAC encoding and decoding. "
            "Install the 'source' extra: pip install aiosendspin[source]"
        ) from err


@functools.cache
def opus_available() -> bool:
    """Whether Opus can be encoded and decoded, which needs PyAV built with libopus."""
    try:
        av = _get_av()
        av.codec.Codec("libopus", "w")
        av.codec.Codec("libopus", "r")
    except (ImportError, ValueError):
        # PyAV reports a missing codec as UnknownCodecError, a ValueError.
        return False
    return True


class PcmPassthrough:
    """Chunk PCM into fixed-size frames."""

    def __init__(
        self,
        *,
        sample_rate: int,
        bit_depth: int,
        channels: int,
        chunk_duration_us: int = 25_000,
        options: Mapping[str, str] | None = None,
    ) -> None:
        """Initialize with audio format parameters."""
        self._sample_rate = sample_rate
        self._frame_stride = (bit_depth // 8) * channels
        self._options = options
        self._chunk_samples = int(sample_rate * chunk_duration_us / 1_000_000)
        # Derive duration from the integer sample count.
        self._chunk_duration_us = self._chunk_samples * 1_000_000 // sample_rate
        self._frame_size = self._chunk_samples * self._frame_stride
        self._buffer = bytearray()
        self._pending_timestamp_us: int | None = None
        # Carry fractional sample time to prevent timestamp drift.
        self._ts_residue: int = 0
        self._last_input_timestamp_us: int | None = None

    @property
    def frame_duration_us(self) -> int:
        """Static frame duration used for `TransformKey` identity."""
        return self._chunk_duration_us

    @property
    def frame_samples(self) -> int:
        """Number of PCM samples represented by each output frame."""
        return self._chunk_samples

    @property
    def lookahead_us(self) -> int:
        """Encoder delay ahead of the first input sample (none for this codec)."""
        return 0

    @property
    def pending_timestamp_us(self) -> int | None:
        """Timestamp of the first buffered sample, or None if buffer is empty."""
        return self._pending_timestamp_us

    def process(self, pcm: bytes, timestamp_us: int, duration_us: int) -> list[tuple[bytes, int]]:  # noqa: ARG002
        """Chunk PCM into fixed-size frames."""
        # Reset the timeline after a large input gap.
        if self._last_input_timestamp_us is not None:
            input_gap = timestamp_us - self._last_input_timestamp_us
            if input_gap > 1_500_000:
                self._pending_timestamp_us = timestamp_us
                self._ts_residue = 0
        self._last_input_timestamp_us = timestamp_us

        if self._pending_timestamp_us is None:
            self._pending_timestamp_us = timestamp_us

        self._buffer.extend(pcm)
        frames: list[tuple[bytes, int]] = []

        while len(self._buffer) >= self._frame_size:
            frame = bytes(memoryview(self._buffer)[: self._frame_size])
            del self._buffer[: self._frame_size]
            self._ts_residue += self._chunk_samples * 1_000_000
            delta_us, self._ts_residue = divmod(self._ts_residue, self._sample_rate)
            if self._pending_timestamp_us is not None:
                self._pending_timestamp_us += delta_us
            frames.append((frame, delta_us))

        return frames

    def flush(self) -> list[tuple[bytes, int]]:
        """Flush remaining buffered audio, padded with silence."""
        if not self._buffer:
            return []

        padding_needed = self._frame_size - len(self._buffer)
        self._buffer.extend(bytes(padding_needed))
        frame = bytes(self._buffer)
        self._buffer.clear()
        self._ts_residue += self._chunk_samples * 1_000_000
        delta_us, self._ts_residue = divmod(self._ts_residue, self._sample_rate)
        self._pending_timestamp_us = None
        self._ts_residue = 0
        return [(frame, delta_us)]

    def get_header(self) -> bytes | None:
        """No codec header for raw PCM."""
        return None

    def get_codec_header(self) -> bytes | None:
        """No codec header for raw PCM (source wire)."""
        return None

    def reset(self) -> None:
        """Reset internal buffer."""
        self._buffer.clear()
        self._pending_timestamp_us = None
        self._ts_residue = 0
        self._last_input_timestamp_us = None


class FlacEncoder:
    """FLAC audio encoder transformer."""

    # ffmpeg encodes wider than 24 bits only under `-strict experimental`, and decoders
    # older than libFLAC 1.4 cannot read 32-bit FLAC.
    VALID_BIT_DEPTHS = frozenset({16, 24})

    def __init__(
        self,
        *,
        sample_rate: int,
        bit_depth: int,
        channels: int,
        chunk_duration_us: int = 25_000,
        options: Mapping[str, str] | None = None,
    ) -> None:
        """Initialize FLAC encoder with audio format parameters."""
        if bit_depth not in self.VALID_BIT_DEPTHS:
            valid = sorted(self.VALID_BIT_DEPTHS)
            msg = f"FLAC only supports bit depths {valid}, got {bit_depth}"
            raise ValueError(msg)

        self._sample_rate = sample_rate
        self._bit_depth = bit_depth
        self._channels = channels
        self._chunk_duration_us = chunk_duration_us
        self._options = options or {}
        self._encoder: av.AudioCodecContext | None = None
        self._codec_header: bytes | None = None
        self._av_format: str | None = None
        self._av_layout: str | None = None
        self._frame_stride: int = (bit_depth // 8) * channels
        self._chunk_samples = int(sample_rate * chunk_duration_us / 1_000_000)
        self._buffer = bytearray()
        self._initialized = False
        # FLAC has internal buffering, so output is tracked separately from input.
        self._stream_start_timestamp_us: int | None = None
        self._output_frame_count: int = 0
        self._first_input_timestamp_us: int | None = None
        self._chunks_encoded_total: int = 0
        self._last_input_timestamp_us: int | None = None
        self._dur_residue: int = 0

    @property
    def frame_duration_us(self) -> int:
        """Static frame duration used for `TransformKey` identity."""
        return self._chunk_duration_us

    @property
    def frame_samples(self) -> int:
        """Number of PCM samples represented by each output frame."""
        return self._chunk_samples

    @property
    def lookahead_us(self) -> int:
        """Encoder delay ahead of the first input sample (none for this codec)."""
        return 0

    @property
    def pending_timestamp_us(self) -> int | None:
        """Timestamp of the next output frame, or None if stream not started."""
        if self._stream_start_timestamp_us is None:
            return None
        cumulative_samples = self._output_frame_count * self._chunk_samples
        return self._stream_start_timestamp_us + (
            cumulative_samples * 1_000_000 // self._sample_rate
        )

    def _ensure_initialized(self) -> None:
        """Lazily initialize encoder on first use."""
        if self._initialized:
            return

        av = _require_av()
        audio_format = AudioFormat(
            sample_rate=self._sample_rate,
            bit_depth=self._bit_depth,
            channels=self._channels,
        )
        _, self._av_format, self._av_layout, av_bytes_per_sample = audio_format.resolve_av_format()
        self._frame_stride = av_bytes_per_sample * self._channels

        self._encoder = av.AudioCodecContext.create("flac", "w")
        self._encoder.sample_rate = self._sample_rate
        self._encoder.layout = self._av_layout
        self._encoder.format = self._av_format
        self._encoder.options = {"compression_level": self._options.get("compression_level", "5")}

        with av.logging.Capture():
            self._encoder.open()

        # FLAC selects its own block size.
        if self._encoder.frame_size:
            self._chunk_samples = self._encoder.frame_size
            self._chunk_duration_us = self._chunk_samples * 1_000_000 // self._sample_rate

        header = bytes(self._encoder.extradata) if self._encoder.extradata else b""
        if header:
            self._codec_header = b"fLaC\x80" + len(header).to_bytes(3, "big") + header
        else:
            self._codec_header = None

        self._initialized = True

    def _encode_chunk(self, chunk_pcm: bytes) -> bytes:
        """Encode a single chunk of PCM to FLAC."""
        assert self._encoder is not None
        av = _require_av()
        _validate_pcm_buffer_length(
            chunk_pcm,
            expected=self._chunk_samples * self._frame_stride,
            context="FLAC encoder input",
        )

        frame = av.AudioFrame(
            format=self._av_format,
            layout=self._av_layout,
            samples=self._chunk_samples,
        )
        frame.sample_rate = self._sample_rate
        frame.planes[0].update(chunk_pcm)

        output = bytearray()
        packets = self._encoder.encode(frame)
        for packet in packets:
            output.extend(bytes(packet))
        return bytes(output)

    def process(self, pcm: bytes, timestamp_us: int, duration_us: int) -> list[tuple[bytes, int]]:  # noqa: ARG002
        """Encode PCM to FLAC frames."""
        self._ensure_initialized()

        if self._last_input_timestamp_us is not None:
            input_gap = timestamp_us - self._last_input_timestamp_us
            if input_gap > 1_500_000:
                self._stream_start_timestamp_us = None
                self._output_frame_count = 0
                self._first_input_timestamp_us = timestamp_us
                self._chunks_encoded_total = 0
                self._dur_residue = 0
        self._last_input_timestamp_us = timestamp_us

        if self._first_input_timestamp_us is None:
            self._first_input_timestamp_us = timestamp_us

        self._buffer.extend(pcm)
        frames: list[tuple[bytes, int]] = []
        chunk_size = self._chunk_samples * self._frame_stride

        while len(self._buffer) >= chunk_size:
            chunk_pcm = bytes(memoryview(self._buffer)[:chunk_size])
            del self._buffer[:chunk_size]
            encoded = self._encode_chunk(chunk_pcm)
            self._chunks_encoded_total += 1
            if encoded:
                if self._stream_start_timestamp_us is None:
                    assert self._first_input_timestamp_us is not None
                    encoder_delay_chunks = max(self._chunks_encoded_total - 1, 0)
                    # Derive delay from samples consumed before the first packet.
                    delay_samples = encoder_delay_chunks * self._chunk_samples
                    self._stream_start_timestamp_us = self._first_input_timestamp_us + (
                        delay_samples * 1_000_000 // self._sample_rate
                    )
                self._dur_residue += self._chunk_samples * 1_000_000
                delta_us, self._dur_residue = divmod(self._dur_residue, self._sample_rate)
                frames.append((encoded, delta_us))
                self._output_frame_count += 1

        return frames

    def flush(self) -> list[tuple[bytes, int]]:
        """Flush remaining buffered audio, padded with silence."""
        if not self._buffer:
            return []

        self._ensure_initialized()
        chunk_size = self._chunk_samples * self._frame_stride

        padding_needed = chunk_size - len(self._buffer)
        self._buffer.extend(bytes(padding_needed))
        chunk_pcm = bytes(self._buffer)
        self._buffer.clear()

        encoded = self._encode_chunk(chunk_pcm)
        if encoded:
            self._output_frame_count += 1
            self._dur_residue += self._chunk_samples * 1_000_000
            delta_us, self._dur_residue = divmod(self._dur_residue, self._sample_rate)
            return [(encoded, delta_us)]
        return []

    def get_header(self) -> bytes | None:
        """Return FLAC streaminfo header."""
        self._ensure_initialized()
        return self._codec_header

    def get_codec_header(self) -> bytes | None:
        """Return the FLAC codec header for the source wire."""
        return self.get_header()

    def reset(self) -> None:
        """Reset encoder state."""
        self._encoder = None
        self._codec_header = None
        self._buffer.clear()
        self._initialized = False
        self._stream_start_timestamp_us = None
        self._output_frame_count = 0
        self._first_input_timestamp_us = None
        self._chunks_encoded_total = 0
        self._last_input_timestamp_us = None
        self._dur_residue = 0


class OpusEncoder:
    """Opus audio encoder transformer."""

    VALID_SAMPLE_RATES = frozenset({8000, 12000, 16000, 24000, 48000})

    def __init__(
        self,
        *,
        sample_rate: int,
        bit_depth: int,  # noqa: ARG002 - Opus uses s16 internally
        channels: int,
        chunk_duration_us: int = 25_000,
        options: Mapping[str, str] | None = None,  # noqa: ARG002 - uses libopus defaults
    ) -> None:
        """Initialize Opus encoder with audio format parameters."""
        if sample_rate not in self.VALID_SAMPLE_RATES:
            valid = sorted(self.VALID_SAMPLE_RATES)
            msg = f"Opus only supports sample rates {valid}, got {sample_rate}"
            raise ValueError(msg)

        self._sample_rate = sample_rate
        self._channels = channels
        self._chunk_duration_us = chunk_duration_us
        self._encoder: av.AudioCodecContext | None = None
        self._frame_stride: int = 2 * channels
        self._chunk_samples = int(sample_rate * chunk_duration_us / 1_000_000)
        self._buffer = bytearray()
        self._initialized = False
        self._stream_start_timestamp_us: int | None = None
        self._output_frame_count: int = 0
        self._first_input_timestamp_us: int | None = None
        self._chunks_encoded_total: int = 0
        self._last_input_timestamp_us: int | None = None
        # Read encoder lookahead from OpusHead after opening.
        self._lookahead_us: int = 0
        self._dur_residue: int = 0

    @property
    def frame_duration_us(self) -> int:
        """Static frame duration used for `TransformKey` identity."""
        return self._chunk_duration_us

    @property
    def frame_samples(self) -> int:
        """Number of PCM samples represented by each output frame."""
        return self._chunk_samples

    @property
    def lookahead_us(self) -> int:
        """Encoder pre-skip, valid once the encoder has been initialized."""
        return self._lookahead_us

    @property
    def pending_timestamp_us(self) -> int | None:
        """Timestamp of the next output frame, or None if stream not started."""
        if self._stream_start_timestamp_us is None:
            return None
        cumulative_samples = self._output_frame_count * self._chunk_samples
        return self._stream_start_timestamp_us + (
            cumulative_samples * 1_000_000 // self._sample_rate
        )

    def _ensure_initialized(self) -> None:
        """Lazily initialize encoder on first use."""
        if self._initialized:
            return

        av = _require_av()

        self._encoder = av.AudioCodecContext.create("libopus", "w")
        self._encoder.sample_rate = self._sample_rate
        self._encoder.layout = "stereo" if self._channels == 2 else "mono"
        self._encoder.format = "s16"

        with av.logging.Capture():
            self._encoder.open()

        if self._encoder.frame_size:
            self._chunk_samples = self._encoder.frame_size
            self._chunk_duration_us = self._chunk_samples * 1_000_000 // self._sample_rate

        # Shift the stream anchor by the OpusHead pre-skip.
        extradata = self._encoder.extradata
        if extradata and len(extradata) >= 12 and extradata[:8] == b"OpusHead":
            pre_skip_samples = struct.unpack_from("<H", extradata, 10)[0]
            self._lookahead_us = pre_skip_samples * 1_000_000 // 48_000
        else:
            logger.debug(
                "Opus extradata missing or unrecognized; skipping lookahead "
                "compensation (extradata=%r)",
                extradata,
            )

        self._initialized = True

    def _encode_chunk(self, chunk_pcm: bytes) -> bytes:
        """Encode a single chunk of PCM to Opus."""
        assert self._encoder is not None
        av = _require_av()
        _validate_pcm_buffer_length(
            chunk_pcm,
            expected=self._chunk_samples * self._frame_stride,
            context="Opus encoder input",
        )

        frame = av.AudioFrame(
            format="s16",
            layout="stereo" if self._channels == 2 else "mono",
            samples=self._chunk_samples,
        )
        frame.sample_rate = self._sample_rate
        frame.planes[0].update(chunk_pcm)

        output = bytearray()
        packets = self._encoder.encode(frame)
        for packet in packets:
            output.extend(bytes(packet))
        return bytes(output)

    def process(self, pcm: bytes, timestamp_us: int, duration_us: int) -> list[tuple[bytes, int]]:  # noqa: ARG002
        """Encode PCM to Opus frames."""
        self._ensure_initialized()

        if self._last_input_timestamp_us is not None:
            input_gap = timestamp_us - self._last_input_timestamp_us
            if input_gap > 1_500_000:
                self._stream_start_timestamp_us = None
                self._output_frame_count = 0
                self._first_input_timestamp_us = timestamp_us
                self._chunks_encoded_total = 0
                self._dur_residue = 0
        self._last_input_timestamp_us = timestamp_us

        if self._first_input_timestamp_us is None:
            self._first_input_timestamp_us = timestamp_us

        self._buffer.extend(pcm)
        frames: list[tuple[bytes, int]] = []
        chunk_size = self._chunk_samples * self._frame_stride

        while len(self._buffer) >= chunk_size:
            chunk_pcm = bytes(memoryview(self._buffer)[:chunk_size])
            del self._buffer[:chunk_size]
            encoded = self._encode_chunk(chunk_pcm)
            self._chunks_encoded_total += 1
            if encoded:
                if self._stream_start_timestamp_us is None:
                    assert self._first_input_timestamp_us is not None
                    encoder_delay_chunks = max(self._chunks_encoded_total - 1, 0)
                    delay_samples = encoder_delay_chunks * self._chunk_samples
                    self._stream_start_timestamp_us = (
                        self._first_input_timestamp_us
                        + (delay_samples * 1_000_000 // self._sample_rate)
                        - self._lookahead_us
                    )
                self._dur_residue += self._chunk_samples * 1_000_000
                delta_us, self._dur_residue = divmod(self._dur_residue, self._sample_rate)
                frames.append((encoded, delta_us))
                self._output_frame_count += 1

        return frames

    def flush(self) -> list[tuple[bytes, int]]:
        """Flush remaining buffered audio, padded with silence."""
        if not self._buffer:
            return []

        self._ensure_initialized()
        chunk_size = self._chunk_samples * self._frame_stride

        padding_needed = chunk_size - len(self._buffer)
        self._buffer.extend(bytes(padding_needed))
        chunk_pcm = bytes(self._buffer)
        self._buffer.clear()

        encoded = self._encode_chunk(chunk_pcm)
        if encoded:
            self._output_frame_count += 1
            self._dur_residue += self._chunk_samples * 1_000_000
            delta_us, self._dur_residue = divmod(self._dur_residue, self._sample_rate)
            return [(encoded, delta_us)]
        return []

    def get_header(self) -> bytes | None:
        """Opus doesn't need a header for raw packets (player stream/start)."""
        return None

    def get_codec_header(self) -> bytes | None:
        """Return no source-wire header for raw Opus packets."""
        return None

    def reset(self) -> None:
        """Reset encoder state."""
        self._encoder = None
        self._buffer.clear()
        self._initialized = False
        self._stream_start_timestamp_us = None
        self._output_frame_count = 0
        self._first_input_timestamp_us = None
        self._chunks_encoded_total = 0
        self._last_input_timestamp_us = None
        self._lookahead_us = 0
        self._dur_residue = 0


class PcmDecoder:
    """Identity decoder for raw PCM source frames (no transcoding)."""

    def __init__(
        self, *, sample_rate: int, bit_depth: int, channels: int, codec_header: bytes | None = None
    ) -> None:
        """Accept the declared format without setup."""
        del sample_rate, bit_depth, channels, codec_header

    def decode(self, data: bytes) -> bytes:
        """Return the PCM frame unchanged."""
        return data

    def flush(self) -> bytes:
        """No buffered audio for raw PCM."""
        return b""


class _AvDecoder:
    """Decode audio to packed PCM at the declared format."""

    def __init__(
        self,
        codec_name: str,
        *,
        sample_rate: int,
        bit_depth: int,
        channels: int,
        codec_header: bytes | None = None,
    ) -> None:
        """Create a decoder context for ``codec_name`` at the declared format."""
        av = _require_av()
        fmt = AudioFormat(sample_rate=sample_rate, bit_depth=bit_depth, channels=channels)
        _, self._av_format, self._av_layout, _ = fmt.resolve_av_format()
        self._bit_depth = bit_depth
        self._sample_rate = sample_rate
        self._channels = channels

        self._ctx = av.CodecContext.create(codec_name, "r")
        self._ctx.sample_rate = sample_rate
        self._ctx.layout = self._av_layout
        if codec_header:
            self._ctx.extradata = self._normalize_header(codec_name, codec_header)

        # Normalize format and layout without changing sample rate.
        self._resampler = av.AudioResampler(
            format="s32" if bit_depth == 24 else self._av_format,
            layout=self._av_layout,
            rate=sample_rate,
        )

    @staticmethod
    def _normalize_header(codec_name: str, header: bytes) -> bytes:
        """Strip the ``fLaC``-block wrapper so ffmpeg gets the bare STREAMINFO."""
        if codec_name == "flac" and header[:4] == b"fLaC" and len(header) >= 8:
            block_len = int.from_bytes(header[5:8], "big")
            return header[8 : 8 + block_len]
        return header

    def _frames_to_pcm(self, frames: list[av.AudioFrame]) -> bytes:
        out = bytearray()
        for frame in frames:
            for resampled in self._resampler.resample(frame):
                pcm = bytes(resampled.planes[0])[: resampled.samples * self._stride()]
                out += _convert_s32_to_s24(pcm) if self._bit_depth == 24 else pcm
        return bytes(out)

    def _stride(self) -> int:
        bytes_per_sample = 4 if self._bit_depth in (24, 32) else 2
        return bytes_per_sample * self._channels

    def decode(self, data: bytes) -> bytes:
        """Decode one encoded frame to native packed PCM (may be empty while buffering)."""
        av = _require_av()
        packet = av.Packet(data)
        return self._frames_to_pcm(self._ctx.decode(packet))

    def flush(self) -> bytes:
        """Flush the decoder, returning any trailing PCM."""
        av = _require_av()
        return self._frames_to_pcm(self._ctx.decode(av.Packet(None)))


def create_decoder(
    codec: str, *, sample_rate: int, bit_depth: int, channels: int, codec_header: bytes | None
) -> PcmDecoder | _AvDecoder:
    """Build a decoder for ``codec`` ('pcm' | 'flac' | 'opus')."""
    if codec == "pcm":
        return PcmDecoder(
            sample_rate=sample_rate,
            bit_depth=bit_depth,
            channels=channels,
            codec_header=codec_header,
        )
    if codec in ("flac", "opus"):
        codec_name = "flac" if codec == "flac" else "libopus"
        return _AvDecoder(
            codec_name,
            sample_rate=sample_rate,
            bit_depth=bit_depth,
            channels=channels,
            codec_header=codec_header,
        )
    raise ValueError(f"Unsupported source codec: {codec!r}")


def create_encoder(
    codec: str, *, sample_rate: int, bit_depth: int, channels: int, chunk_duration_us: int = 25_000
) -> PcmPassthrough | FlacEncoder | OpusEncoder:
    """Build an encoder for ``codec`` ('pcm' | 'flac' | 'opus')."""
    cls: type[PcmPassthrough | FlacEncoder | OpusEncoder]
    if codec == "pcm":
        cls = PcmPassthrough
    elif codec == "flac":
        cls = FlacEncoder
    elif codec == "opus":
        cls = OpusEncoder
    else:
        raise ValueError(f"Unsupported source codec: {codec!r}")
    return cls(
        sample_rate=sample_rate,
        bit_depth=bit_depth,
        channels=channels,
        chunk_duration_us=chunk_duration_us,
    )


__all__ = [
    "FlacEncoder",
    "OpusEncoder",
    "PcmDecoder",
    "PcmPassthrough",
    "create_decoder",
    "create_encoder",
    "opus_available",
]
