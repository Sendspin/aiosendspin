"""Tests for BufferTracker byte and duration accounting."""

from __future__ import annotations

from aiosendspin.server.audio import BufferTracker


class _FakeClock:
    """Fake clock for testing."""

    def __init__(self, now_us: int = 0) -> None:
        self._now_us = now_us

    def now_us(self) -> int:
        return self._now_us

    def set_now(self, now_us: int) -> None:
        self._now_us = now_us


def test_buffer_tracker_tracks_duration() -> None:
    """BufferTracker should track duration when registering chunks."""
    clock = _FakeClock(now_us=0)
    tracker = BufferTracker(
        clock=clock,
        client_id="test",
        capacity_bytes=10000,
        max_duration_us=1_000_000,  # 1 second
    )

    # Register a chunk with duration
    tracker.register(end_time_us=100_000, byte_count=1000, duration_us=100_000)

    assert tracker.buffered_bytes == 1000
    assert tracker.buffered_duration_us == 100_000


def test_buffer_tracker_prune_removes_duration() -> None:
    """prune_consumed() should remove duration from consumed chunks."""
    clock = _FakeClock(now_us=0)
    tracker = BufferTracker(
        clock=clock,
        client_id="test",
        capacity_bytes=10000,
        max_duration_us=1_000_000,
    )

    tracker.register(end_time_us=100_000, byte_count=1000, duration_us=100_000)
    tracker.register(end_time_us=200_000, byte_count=1000, duration_us=100_000)

    assert tracker.buffered_duration_us == 200_000

    # Advance time past first chunk
    clock.set_now(150_000)
    tracker.prune_consumed()

    assert tracker.buffered_bytes == 1000
    assert tracker.buffered_duration_us == 100_000


def test_has_duration_capacity_when_not_configured() -> None:
    """has_duration_capacity() returns True when max_duration_us is 0."""
    clock = _FakeClock(now_us=0)
    tracker = BufferTracker(
        clock=clock,
        client_id="test",
        capacity_bytes=10000,
        # max_duration_us defaults to 0
    )

    # Should always return True when duration tracking not configured
    assert tracker.has_duration_capacity(1_000_000_000) is True


def test_has_duration_capacity_with_space() -> None:
    """has_duration_capacity() returns True when buffer has space."""
    clock = _FakeClock(now_us=0)
    tracker = BufferTracker(
        clock=clock,
        client_id="test",
        capacity_bytes=10000,
        max_duration_us=1_000_000,  # 1 second max
    )

    tracker.register(end_time_us=500_000, byte_count=1000, duration_us=500_000)

    # Has space for another 400ms
    assert tracker.has_duration_capacity(400_000) is True


def test_has_duration_capacity_full() -> None:
    """has_duration_capacity() returns False when buffer is full."""
    clock = _FakeClock(now_us=0)
    tracker = BufferTracker(
        clock=clock,
        client_id="test",
        capacity_bytes=10000,
        max_duration_us=1_000_000,  # 1 second max
    )

    tracker.register(end_time_us=800_000, byte_count=1000, duration_us=800_000)

    # No space for another 300ms (800ms + 300ms > 1000ms)
    assert tracker.has_duration_capacity(300_000) is False


def test_reset_clears_duration() -> None:
    """reset() should clear buffered_duration_us."""
    clock = _FakeClock(now_us=0)
    tracker = BufferTracker(
        clock=clock,
        client_id="test",
        capacity_bytes=10000,
        max_duration_us=1_000_000,
    )

    tracker.register(end_time_us=100_000, byte_count=1000, duration_us=100_000)
    tracker.reset()

    assert tracker.buffered_bytes == 0
    assert tracker.buffered_duration_us == 0


def test_buffered_chunk_includes_duration() -> None:
    """BufferedChunk should store duration_us."""
    clock = _FakeClock(now_us=0)
    tracker = BufferTracker(
        clock=clock,
        client_id="test",
        capacity_bytes=10000,
    )

    tracker.register(end_time_us=100_000, byte_count=1000, duration_us=50_000)

    chunk = tracker.buffered_chunks[0]
    assert chunk.end_time_us == 100_000
    assert chunk.byte_count == 1000
    assert chunk.duration_us == 50_000


def test_time_until_duration_capacity_when_not_configured() -> None:
    """time_until_duration_capacity() returns 0 when max_duration_us is 0."""
    clock = _FakeClock(now_us=0)
    tracker = BufferTracker(
        clock=clock,
        client_id="test",
        capacity_bytes=10000,
        # max_duration_us defaults to 0
    )

    # Should return 0 when duration tracking not configured
    assert tracker.time_until_duration_capacity(1_000_000) == 0


def test_time_until_duration_capacity_with_space() -> None:
    """time_until_duration_capacity() returns 0 when buffer has space."""
    clock = _FakeClock(now_us=0)
    tracker = BufferTracker(
        clock=clock,
        client_id="test",
        capacity_bytes=10000,
        max_duration_us=1_000_000,  # 1 second max
    )

    tracker.register(end_time_us=500_000, byte_count=1000, duration_us=500_000)

    # Has space for another 400ms, no wait needed
    assert tracker.time_until_duration_capacity(400_000) == 0


def test_time_until_duration_capacity_returns_excess() -> None:
    """time_until_duration_capacity() returns excess duration when full."""
    clock = _FakeClock(now_us=0)
    tracker = BufferTracker(
        clock=clock,
        client_id="test",
        capacity_bytes=10000,
        max_duration_us=1_000_000,  # 1 second max
    )

    tracker.register(end_time_us=800_000, byte_count=1000, duration_us=800_000)

    # Need 300ms more, but only 200ms capacity → wait 100ms
    # (800ms + 300ms) - 1000ms = 100ms
    assert tracker.time_until_duration_capacity(300_000) == 100_000


def test_time_until_duration_capacity_prunes_first() -> None:
    """time_until_duration_capacity() prunes consumed chunks before checking."""
    clock = _FakeClock(now_us=0)
    tracker = BufferTracker(
        clock=clock,
        client_id="test",
        capacity_bytes=10000,
        max_duration_us=1_000_000,  # 1 second max
    )

    tracker.register(end_time_us=500_000, byte_count=1000, duration_us=500_000)
    tracker.register(end_time_us=1_000_000, byte_count=1000, duration_us=500_000)

    # Buffer is full (1000ms), but advance time to consume first chunk
    clock.set_now(600_000)

    # Now only 500ms buffered, should have space for 400ms
    assert tracker.time_until_duration_capacity(400_000) == 0


def test_time_until_end_time_capacity_uses_buffer_horizon() -> None:
    """Horizon-based gating should respect the furthest effective end timestamp."""
    clock = _FakeClock(now_us=0)
    tracker = BufferTracker(
        clock=clock,
        client_id="test",
        capacity_bytes=10000,
        max_duration_us=1_000_000,  # 1 second max effective horizon
    )

    tracker.register(end_time_us=600_000, byte_count=1000, duration_us=100_000)

    # Raw duration would fit, but an end timestamp at 1.5s pushes effective horizon to 1.5s.
    assert tracker.time_until_ready(100, 100_000, end_time_us=1_500_000) == 500_000


def test_buffered_horizon_us_tracks_furthest_end_from_now() -> None:
    """buffered_horizon_us() should report furthest scheduled end minus now."""
    clock = _FakeClock(now_us=0)
    tracker = BufferTracker(
        clock=clock,
        client_id="test",
        capacity_bytes=10000,
    )

    tracker.register(end_time_us=200_000, byte_count=1000, duration_us=100_000)
    tracker.register(end_time_us=500_000, byte_count=1000, duration_us=100_000)

    assert tracker.buffered_horizon_us() == 500_000

    clock.set_now(250_000)
    # First chunk is pruned; horizon is from now to the second chunk's end.
    assert tracker.buffered_horizon_us() == 250_000


def _tracker(clock: _FakeClock, capacity_bytes: int = 1000) -> BufferTracker:
    return BufferTracker(clock=clock, client_id="test", capacity_bytes=capacity_bytes)


def _send(tracker: BufferTracker, end_time_us: int, byte_count: int) -> None:
    chunk = tracker.register(end_time_us=end_time_us, byte_count=byte_count, duration_us=100_000)
    assert chunk is not None
    tracker.finish_transmission(chunk)


def test_count_starts_empty() -> None:
    """A new tracker counts nothing and admits a chunk up to the full capacity."""
    tracker = _tracker(_FakeClock())

    assert tracker.buffered_bytes == 0
    assert tracker.has_capacity_now(1000) is True


def test_chunk_may_not_start_if_it_would_exceed_capacity() -> None:
    """A chunk fits only while its size plus the current count stays within capacity."""
    tracker = _tracker(_FakeClock())
    _send(tracker, end_time_us=100_000, byte_count=600)

    assert tracker.has_capacity_now(400) is True
    assert tracker.has_capacity_now(401) is False
    assert tracker.time_until_capacity(401) == 100_000


def test_chunk_counts_while_transmission_is_unfinished() -> None:
    """A chunk past its completion time still counts until its transmission finishes."""
    clock = _FakeClock()
    tracker = _tracker(clock)
    chunk = tracker.register(end_time_us=100_000, byte_count=500, duration_us=100_000)
    assert chunk is not None

    clock.set_now(200_000)
    assert tracker.has_capacity_now(600) is False
    assert tracker.time_until_capacity(600) > 0

    tracker.finish_transmission(chunk)
    assert tracker.has_capacity_now(600) is True
    assert tracker.time_until_capacity(600) == 0


def test_finishing_a_reset_chunk_leaves_a_newer_transmission_counted() -> None:
    """A chunk from before a reset cannot end the transmission of a newer chunk."""
    clock = _FakeClock()
    tracker = _tracker(clock)
    stale = tracker.register(end_time_us=100_000, byte_count=100, duration_us=100_000)
    assert stale is not None
    tracker.reset()
    tracker.register(end_time_us=100_000, byte_count=500, duration_us=100_000)

    tracker.finish_transmission(stale)
    clock.set_now(200_000)

    assert tracker.has_capacity_now(600) is False


def test_completion_time_subtracts_output_delay() -> None:
    """A chunk stops counting at its end time minus the output delay."""
    clock = _FakeClock()
    tracker = _tracker(clock)
    tracker.output_delay_us = 200_000
    _send(tracker, end_time_us=500_000, byte_count=500)

    clock.set_now(299_999)
    assert tracker.buffered_horizon_us() == 1
    assert tracker.time_until_capacity(600) == 1

    clock.set_now(300_000)
    tracker.prune_consumed()
    assert tracker.buffered_bytes == 0


def test_output_delay_update_recounts_completed_chunks() -> None:
    """Lowering the delay counts a chunk again when its new completion time is in the future."""
    clock = _FakeClock()
    tracker = _tracker(clock)
    tracker.output_delay_us = 200_000
    _send(tracker, end_time_us=400_000, byte_count=300)
    _send(tracker, end_time_us=500_000, byte_count=400)

    clock.set_now(350_000)
    tracker.prune_consumed()
    assert tracker.buffered_bytes == 0

    tracker.output_delay_us = 100_000
    assert tracker.buffered_bytes == 400

    tracker.output_delay_us = 0
    assert tracker.buffered_bytes == 700

    # An end time in the past cannot come back under any delay.
    clock.set_now(450_000)
    tracker.output_delay_us = 200_000
    tracker.output_delay_us = 0
    assert tracker.buffered_bytes == 400


def test_raising_output_delay_releases_chunks() -> None:
    """Raising the delay moves completion times earlier."""
    clock = _FakeClock(now_us=150_000)
    tracker = _tracker(clock)
    _send(tracker, end_time_us=200_000, byte_count=500)

    tracker.output_delay_us = 100_000

    assert tracker.buffered_bytes == 0


def test_reset_empties_count_and_retained_chunks() -> None:
    """reset() drops every chunk, including ones a later delay change would count again."""
    clock = _FakeClock()
    tracker = _tracker(clock)
    tracker.output_delay_us = 200_000
    _send(tracker, end_time_us=300_000, byte_count=500)
    clock.set_now(150_000)
    tracker.prune_consumed()
    tracker.oversize_logged = True

    tracker.reset()
    tracker.output_delay_us = 0

    assert tracker.buffered_bytes == 0
    assert tracker.oversize_logged is False
