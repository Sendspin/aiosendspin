"""Clock abstractions for deterministic timing."""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass
from typing import Protocol

if sys.platform == "linux" and hasattr(time, "CLOCK_MONOTONIC_RAW"):
    _RAW_CLOCK_ID = time.CLOCK_MONOTONIC_RAW

    def _raw_now_us() -> int:
        return time.clock_gettime_ns(_RAW_CLOCK_ID) // 1_000
elif sys.platform == "win32":
    # Before Python 3.13, time.monotonic on Windows ticks only every ~15.6 ms.
    def _raw_now_us() -> int:
        return time.perf_counter_ns() // 1_000
else:

    def _raw_now_us() -> int:
        return time.monotonic_ns() // 1_000


class Clock(Protocol):
    """Time source used for timestamping in microseconds."""

    def now_us(self) -> int:
        """Return current time in microseconds (monotonic)."""


@dataclass(frozen=True, slots=True)
class RawMonotonicClock:
    """Clock backed by CLOCK_MONOTONIC_RAW on Linux, perf_counter on Windows, monotonic elsewhere.

    Unlike CLOCK_MONOTONIC, CLOCK_MONOTONIC_RAW is not slewed by NTP/adjtime,
    so elapsed-microsecond measurements reflect hardware ticks. This matters
    for audio sync since NTP slewing poisons the client-side time filter.

    Note that asyncio scheduling primitives (``call_later``, ``asyncio.sleep``,
    ``wait_for`` timeouts) are driven by the event loop's own clock, which on
    Linux is CLOCK_MONOTONIC. Sendspin only passes relative durations to those
    primitives and never compares a loop-clock value against a ``now_us()``
    value, so the two clock domains never mix in a computation. During active
    NTP slewing the kernel caps the rate mismatch at around 500 ppm, so a
    one-second sleep measured in raw microseconds differs by at most ~500 us.
    That is well below the timing tolerances of backpressure and writer-wait
    loops.
    """

    def now_us(self) -> int:
        """Return current raw monotonic time in microseconds."""
        return _raw_now_us()


@dataclass(frozen=True, slots=True)
class LoopClock:
    """Clock implementation backed by an asyncio event loop."""

    loop: asyncio.AbstractEventLoop

    def now_us(self) -> int:
        """Return current loop time in microseconds."""
        return int(self.loop.time() * 1_000_000)


@dataclass(slots=True)
class ManualClock:
    """Manually-advanced clock used in tests."""

    now_us_value: int = 0

    def now_us(self) -> int:
        """Return current manual time in microseconds."""
        return self.now_us_value

    def advance_us(self, delta_us: int) -> None:
        """Advance the clock by delta_us microseconds."""
        self.now_us_value += delta_us
