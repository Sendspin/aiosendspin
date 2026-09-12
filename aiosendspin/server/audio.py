"""Server audio buffers and shared PCM helpers."""

from __future__ import annotations

import asyncio
import logging
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

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from aiosendspin.clock import Clock


class BufferedChunk(NamedTuple):
    """Buffered chunk metadata tracked by BufferTracker for backpressure control."""

    end_time_us: int
    """Absolute timestamp when these bytes should be fully consumed."""
    byte_count: int
    """Compressed byte count occupying the device buffer."""
    duration_us: int
    """Duration of audio in microseconds (independent of compression)."""


class BufferTracker:
    """
    Track buffered compressed audio for a client and apply backpressure when needed.

    This class monitors the amount of compressed audio data buffered on a client device
    and ensures the server doesn't exceed the client's buffer capacity by applying
    backpressure when necessary.
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
        self.buffered_bytes = 0
        self.buffered_duration_us = 0

    def prune_consumed(self, now_us: int | None = None) -> int:
        """Drop finished chunks and return the timestamp used for the calculation."""
        if now_us is None:
            now_us = self._clock.now_us()
        while self.buffered_chunks and self.buffered_chunks[0].end_time_us <= now_us:
            chunk = self.buffered_chunks.popleft()
            self.buffered_bytes -= chunk.byte_count
            self.buffered_duration_us -= chunk.duration_us
        self.buffered_bytes = max(self.buffered_bytes, 0)
        self.buffered_duration_us = max(self.buffered_duration_us, 0)
        return now_us

    def buffered_horizon_us(self, now_us: int | None = None) -> int:
        """Return buffer horizon from now until the furthest scheduled end time."""
        now_us = self.prune_consumed(now_us)
        if not self.buffered_chunks:
            return 0
        return max(self.buffered_chunks[-1].end_time_us - now_us, 0)

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
        if bytes_needed >= self.capacity_bytes:
            logger.warning(
                "Chunk size %s exceeds reported buffer capacity %s for client %s "
                "— blocking until buffer drains",
                bytes_needed,
                self.capacity_bytes,
                self.client_id,
            )
            self.prune_consumed()
            return self.buffered_bytes == 0

        self.prune_consumed()
        projected_usage = self.buffered_bytes + bytes_needed
        return projected_usage <= self.capacity_bytes

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

        This preserves effective playback headroom based on the furthest buffered end time,
        which is more accurate than summing durations when chunks are intentionally shifted
        on the timeline (for example, player output delay).
        """
        if self.max_duration_us == 0:
            return 0

        now_us = self.prune_consumed()
        if end_time_us <= now_us:
            return 0

        latest_end_us = now_us
        if self.buffered_chunks:
            # Chunks are appended in timestamp order, so the last entry is the furthest.
            latest_end_us = max(now_us, self.buffered_chunks[-1].end_time_us)

        projected_end_us = max(latest_end_us, end_time_us)
        projected_horizon_us = projected_end_us - now_us
        if projected_horizon_us <= self.max_duration_us:
            return 0
        return projected_horizon_us - self.max_duration_us

    def time_until_capacity(self, bytes_needed: int) -> int:
        """
        Calculate time in microseconds until the buffer can accept bytes_needed more bytes.

        Returns 0 if bytes_needed <= 0 (immediate capacity). When bytes_needed exceeds
        capacity_bytes, returns the time needed for the buffer to fully drain so the
        oversize chunk can be admitted alone; returns 0 only if the buffer is already empty.
        """
        if bytes_needed <= 0:
            return 0
        if bytes_needed >= self.capacity_bytes:
            logger.warning(
                "Chunk size %s exceeds reported buffer capacity %s for client %s "
                "— blocking until buffer drains",
                bytes_needed,
                self.capacity_bytes,
                self.client_id,
            )
            cursor_time_us = self.prune_consumed()
            if self.buffered_bytes == 0:
                return 0
            latest_end_us = self.buffered_chunks[-1].end_time_us
            return max(latest_end_us - cursor_time_us, 0)

        # Prune consumed chunks once at the start
        cursor_time_us = self.prune_consumed()
        time_needed_us = 0

        # Simulate state without modifying it to find when capacity is available
        virtual_buffered_bytes = self.buffered_bytes
        cursor_index = 0

        while cursor_index < len(self.buffered_chunks):
            projected_usage = virtual_buffered_bytes + bytes_needed
            if projected_usage <= self.capacity_bytes:
                # We have enough capacity at this point
                break

            chunk = self.buffered_chunks[cursor_index]
            cursor_end_time_us = chunk.end_time_us
            time_needed_us += max(cursor_end_time_us - cursor_time_us, 0)

            # Advance cursor to the next chunk
            cursor_index += 1
            cursor_time_us = cursor_end_time_us
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
            end_time_us: Absolute end timestamp for horizon-based duration gating.

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

    def register(self, end_time_us: int, byte_count: int, duration_us: int = 0) -> None:
        """Record bytes added to the buffer finishing at end_time_us.

        Args:
            end_time_us: Absolute timestamp when these bytes should be fully consumed.
            byte_count: Compressed byte count occupying the device buffer.
            duration_us: Duration of audio in microseconds (for duration-based tracking).
        """
        if byte_count <= 0:
            return
        self.buffered_chunks.append(BufferedChunk(end_time_us, byte_count, duration_us))
        self.buffered_bytes += byte_count
        self.buffered_duration_us += duration_us

    def reset(self) -> None:
        """Clear all tracked chunks and reset counters to zero."""
        self.buffered_chunks.clear()
        self.buffered_bytes = 0
        self.buffered_duration_us = 0


__all__ = [
    "AudioFormat",
    "BufferTracker",
    "_convert_s24_to_s32",
    "_convert_s32_to_s24",
    "_get_av",
    "_get_numpy",
    "_validate_pcm_buffer_length",
]
