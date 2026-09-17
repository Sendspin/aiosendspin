"""Server audio buffers and shared PCM helpers."""

from __future__ import annotations

import asyncio
from collections import deque
from typing import TYPE_CHECKING, NamedTuple

from aiosendspin.audio.format import (
    AudioFormat,
    _convert_s24_to_s32,
    _convert_s32_to_s24,
    _get_av,
    _get_numpy,
    _validate_pcm_buffer_length,
)

if TYPE_CHECKING:
    from aiosendspin.clock import Clock

# Wait before rechecking capacity held by a chunk whose transmission has not finished.
TRANSMISSION_RECHECK_US = 1_000


class BufferedChunk(NamedTuple):
    """Buffered chunk metadata tracked by BufferTracker for backpressure control."""

    end_time_us: int
    """Timestamp plus duration, before the tracker's output delay is applied."""
    byte_count: int
    """Bytes the chunk occupies in the device buffer."""
    duration_us: int
    """Duration of audio in microseconds (independent of compression)."""


class BufferTracker:
    """
    Track the bytes a client holds in its buffer and apply backpressure when needed.

    A registered chunk counts in full while its transmission is unfinished or its
    completion time (end time minus ``output_delay_us``) is in the future. Changing
    ``output_delay_us`` recalculates the completion time of every chunk registered
    since the last reset. Callers must not send a chunk larger than
    ``capacity_bytes``: it can never fit.
    """

    def __init__(
        self,
        *,
        clock: Clock,
        client_id: str,
        capacity_bytes: int,
        max_duration_us: int = 0,
    ) -> None:
        """
        Initialize the buffer tracker for a client.

        Args:
            clock: Time source used for timing calculations.
            client_id: Identifier for the client being tracked.
            capacity_bytes: Maximum buffer capacity in bytes reported by the client.
            max_duration_us: Maximum buffer duration in microseconds. If 0, duration
                is not tracked and has_duration_capacity() always returns True.
        """
        self._clock = clock
        self.client_id = client_id
        self.capacity_bytes = capacity_bytes
        self.max_duration_us = max_duration_us
        self.buffered_chunks: deque[BufferedChunk] = deque()
        """Chunks that currently count, in registration order."""
        self.buffered_bytes = 0
        self.buffered_duration_us = 0
        # Completed chunks that a lower output delay could make count again.
        self._completed_chunks: deque[BufferedChunk] = deque()
        self._output_delay_us = 0
        self._transmitting: BufferedChunk | None = None
        self.oversize_logged = False
        """Whether a chunk larger than the capacity was reported since the last reset."""

    @property
    def output_delay_us(self) -> int:
        """Output delay subtracted from each chunk's end time to get its completion time."""
        return self._output_delay_us

    @output_delay_us.setter
    def output_delay_us(self, value: int) -> None:
        if value == self._output_delay_us:
            return
        self._output_delay_us = value
        # Recount every chunk since the last reset, then drop those complete under the new delay.
        chunks = [*self._completed_chunks, *self.buffered_chunks]
        self._completed_chunks.clear()
        self.buffered_chunks = deque(chunks)
        self.buffered_bytes = sum(chunk.byte_count for chunk in chunks)
        self.buffered_duration_us = sum(chunk.duration_us for chunk in chunks)
        self.prune_consumed()

    def prune_consumed(self, now_us: int | None = None) -> int:
        """Stop counting completed chunks and return the timestamp used for the calculation."""
        if now_us is None:
            now_us = self._clock.now_us()
        while self.buffered_chunks and self._completion_us(self.buffered_chunks[0]) <= now_us:
            if self.buffered_chunks[0] is self._transmitting:
                break
            chunk = self.buffered_chunks.popleft()
            self.buffered_bytes -= chunk.byte_count
            self.buffered_duration_us -= chunk.duration_us
            self._completed_chunks.append(chunk)
        # A chunk whose end time has passed stays complete under any non-negative delay.
        while self._completed_chunks and self._completed_chunks[0].end_time_us <= now_us:
            self._completed_chunks.popleft()
        return now_us

    def buffered_horizon_us(self, now_us: int | None = None) -> int:
        """Return buffer horizon from now until the furthest completion time."""
        now_us = self.prune_consumed(now_us)
        if not self.buffered_chunks:
            return 0
        return max(self._completion_us(self.buffered_chunks[-1]) - now_us, 0)

    def has_capacity_now(self, bytes_needed: int) -> bool:
        """
        Check if buffer can accept bytes_needed without waiting.

        This is a non-blocking version of wait_for_capacity that returns immediately.

        Args:
            bytes_needed: Number of bytes to check capacity for.

        Returns:
            True if the buffer has capacity for bytes_needed, False otherwise.
        """
        if bytes_needed <= 0:
            return True
        self.prune_consumed()
        return self.buffered_bytes + bytes_needed <= self.capacity_bytes

    def has_duration_capacity(self, duration_needed_us: int = 0) -> bool:
        """
        Check if buffer can accept duration_needed_us without exceeding max_duration_us.

        This is independent of byte-based capacity. If max_duration_us is 0 (not configured),
        this always returns True.

        Args:
            duration_needed_us: Duration in microseconds to check capacity for.

        Returns:
            True if the buffer has capacity for duration_needed_us, False otherwise.
        """
        if self.max_duration_us == 0:
            # Duration tracking not configured
            return True
        if duration_needed_us <= 0:
            return True

        self.prune_consumed()
        projected_duration = self.buffered_duration_us + duration_needed_us
        return projected_duration <= self.max_duration_us

    def time_until_duration_capacity(self, duration_needed_us: int = 0) -> int:
        """
        Calculate time in microseconds until the buffer can accept duration_needed_us more.

        Since audio drains at 1x real time, the wait time equals the excess duration.
        Returns 0 if max_duration_us is 0 (not configured) or if there's already capacity.

        Args:
            duration_needed_us: Duration in microseconds to check capacity for.

        Returns:
            Time in microseconds to wait, or 0 if capacity is immediately available.
        """
        if self.max_duration_us == 0:
            return 0
        if duration_needed_us <= 0:
            return 0

        self.prune_consumed()
        projected_duration = self.buffered_duration_us + duration_needed_us
        if projected_duration <= self.max_duration_us:
            return 0

        # Wait for the excess duration to drain (audio plays at 1x real time)
        return projected_duration - self.max_duration_us

    def time_until_end_time_capacity(self, end_time_us: int) -> int:
        """
        Calculate wait time until the buffer horizon can extend to end_time_us.

        This preserves effective playback headroom based on the furthest completion
        time, which is more accurate than summing durations when chunks are shifted
        on the timeline by the output delay. ``end_time_us`` is given before the
        output delay is applied.
        """
        if self.max_duration_us == 0:
            return 0

        now_us = self.prune_consumed()
        completion_us = end_time_us - self._output_delay_us
        if completion_us <= now_us:
            return 0

        latest_end_us = now_us
        if self.buffered_chunks:
            # Chunks are appended in timestamp order, so the last entry is the furthest.
            latest_end_us = max(now_us, self._completion_us(self.buffered_chunks[-1]))

        projected_end_us = max(latest_end_us, completion_us)
        projected_horizon_us = projected_end_us - now_us
        if projected_horizon_us <= self.max_duration_us:
            return 0
        return projected_horizon_us - self.max_duration_us

    def time_until_capacity(self, bytes_needed: int) -> int:
        """
        Calculate time in microseconds until the buffer can accept bytes_needed more bytes.

        Returns 0 if bytes_needed <= 0 (immediate capacity). ``bytes_needed`` must not
        exceed ``capacity_bytes``.
        """
        if bytes_needed <= 0:
            return 0

        now_us = self.prune_consumed()
        time_needed_us = 0
        # Simulate chunks completing in order without modifying the tracked state.
        virtual_buffered_bytes = self.buffered_bytes
        for chunk in self.buffered_chunks:
            if virtual_buffered_bytes + bytes_needed <= self.capacity_bytes:
                break
            if chunk is self._transmitting:
                return max(time_needed_us, TRANSMISSION_RECHECK_US)
            time_needed_us = max(time_needed_us, self._completion_us(chunk) - now_us)
            virtual_buffered_bytes -= chunk.byte_count
        return time_needed_us

    def time_until_ready(
        self,
        bytes_needed: int,
        duration_needed_us: int,
        *,
        end_time_us: int | None = None,
    ) -> int:
        """
        Calculate time until buffer can accept both bytes and duration.

        Combines byte-based and duration-based backpressure into a single wait time.
        Returns the maximum of both wait times.

        Args:
            bytes_needed: Number of bytes to check capacity for.
            duration_needed_us: Duration in microseconds to check capacity for.
            end_time_us: End timestamp, before the output delay, for horizon-based
                duration gating.

        Returns:
            Time in microseconds to wait, or 0 if ready immediately.
        """
        byte_wait = self.time_until_capacity(bytes_needed)
        if end_time_us is not None:
            duration_wait = self.time_until_end_time_capacity(end_time_us)
        else:
            duration_wait = self.time_until_duration_capacity(duration_needed_us)
        return max(byte_wait, duration_wait)

    # TODO: if unused delete
    async def wait_for_capacity(self, bytes_needed: int) -> None:
        """Block until the device buffer can accept bytes_needed more bytes."""
        if sleep_time_us := self.time_until_capacity(bytes_needed):
            await asyncio.sleep(sleep_time_us / 1_000_000)

    def register(
        self, end_time_us: int, byte_count: int, duration_us: int = 0
    ) -> BufferedChunk | None:
        """Count a chunk whose transmission is starting, and return it.

        The chunk keeps counting at least until it is passed to finish_transmission().
        Returns None, counting nothing, for an empty chunk.

        Args:
            end_time_us: Timestamp plus duration, before the output delay is applied.
            byte_count: Bytes the chunk occupies in the device buffer.
            duration_us: Duration of audio in microseconds (for duration-based tracking).
        """
        if byte_count <= 0:
            return None
        chunk = BufferedChunk(end_time_us, byte_count, duration_us)
        self.buffered_chunks.append(chunk)
        self.buffered_bytes += byte_count
        self.buffered_duration_us += duration_us
        self._transmitting = chunk
        return chunk

    def finish_transmission(self, chunk: BufferedChunk) -> None:
        """Mark the transmission of a chunk returned by register() as finished."""
        if chunk is self._transmitting:
            self._transmitting = None

    def reset(self) -> None:
        """Clear all tracked chunks and reset counters to zero."""
        self.buffered_chunks.clear()
        self._completed_chunks.clear()
        self.buffered_bytes = 0
        self.buffered_duration_us = 0
        self._transmitting = None
        self.oversize_logged = False

    def _completion_us(self, chunk: BufferedChunk) -> int:
        return chunk.end_time_us - self._output_delay_us


__all__ = [
    "AudioFormat",
    "BufferTracker",
    "_convert_s24_to_s32",
    "_convert_s32_to_s24",
    "_get_av",
    "_get_numpy",
    "_validate_pcm_buffer_length",
]
