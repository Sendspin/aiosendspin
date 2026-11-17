"""Behavior tests for the SendspinTimeFilter Kalman filter.

These tests assert invariants that hold for any correct Kalman implementation
of the Sendspin time-sync filter. They deliberately avoid pinning exact float
outputs; for vector-level parity against the C++ reference see
``scripts/compare_time_filter.py``.
"""

# ruff: noqa: SLF001  Tests intentionally probe filter internals

from __future__ import annotations

import math

import pytest

from aiosendspin.client.time_sync import SendspinTimeFilter, TimeElement


def _feed_constant_offset(
    filter_: SendspinTimeFilter,
    offset_us: int,
    count: int,
    max_error_us: int = 1000,
    interval_us: int = 1_000_000,
    start_time_us: int = 1_000_000,
) -> int:
    """Feed ``count`` measurements with constant offset and zero drift."""
    t = start_time_us
    for _ in range(count):
        filter_.update(offset_us, max_error_us, t)
        t += interval_us
    return t


def test_count_zero_branch_seeds_offset_from_first_sample() -> None:
    """First update seeds offset from the single measurement."""
    filter_ = SendspinTimeFilter()

    assert filter_.count == 0

    filter_.update(measurement=12_345, max_error=500, time_added=1_000_000)

    assert filter_.count == 1
    assert filter_.offset == pytest.approx(12_345.0)
    # Drift unknown after one sample
    assert filter_._drift == 0.0


@pytest.mark.parametrize("max_error", [0, -1])
def test_measurement_without_positive_max_error_is_discarded(max_error: int) -> None:
    """A measurement with max_error <= 0 leaves the filter state unchanged."""
    filter_ = SendspinTimeFilter()
    t = _feed_constant_offset(filter_, offset_us=1_000, count=3)

    snapshot_count = filter_.count
    snapshot_offset = filter_.offset
    snapshot_last_update = filter_._last_update

    filter_.update(measurement=9_999, max_error=max_error, time_added=t)

    assert filter_.count == snapshot_count
    assert filter_.offset == snapshot_offset
    assert filter_._last_update == snapshot_last_update


def test_count_one_branch_estimates_drift_from_two_point_slope() -> None:
    """Second update estimates drift from the two-point slope."""
    filter_ = SendspinTimeFilter()

    filter_.update(measurement=1_000, max_error=500, time_added=1_000_000)
    filter_.update(measurement=2_000, max_error=500, time_added=2_000_000)

    assert filter_.count == 2
    assert filter_.offset == pytest.approx(2_000.0)
    # (2000 - 1000) / (2_000_000 - 1_000_000) = 0.001
    assert filter_._drift == pytest.approx(0.001)


def test_late_update_with_non_monotonic_time_is_ignored() -> None:
    """An update with time_added <= _last_update returns without state changes."""
    filter_ = SendspinTimeFilter()
    filter_.update(measurement=1_000, max_error=500, time_added=2_000_000)

    snapshot_count = filter_.count
    snapshot_offset = filter_.offset
    snapshot_last_update = filter_._last_update

    # Equal timestamp
    filter_.update(measurement=9_999, max_error=500, time_added=2_000_000)
    # Backward timestamp
    filter_.update(measurement=9_999, max_error=500, time_added=1_500_000)

    assert filter_.count == snapshot_count
    assert filter_.offset == snapshot_offset
    assert filter_._last_update == snapshot_last_update


def test_reset_returns_filter_to_init_state() -> None:
    """After reset, all internal state matches a fresh filter."""
    filter_ = SendspinTimeFilter()
    _feed_constant_offset(filter_, offset_us=50_000, count=10)

    # Sanity: state advanced.
    assert filter_.count > 0
    assert filter_._last_update != 0

    filter_.reset()

    assert filter_.count == 0
    assert filter_._last_update == 0
    assert filter_._offset == 0.0
    assert filter_._drift == 0.0
    assert math.isinf(filter_._offset_covariance)
    assert filter_._offset_drift_covariance == 0.0
    assert filter_._drift_covariance == 0.0
    assert filter_._current_time_element == TimeElement()


def test_converges_to_perfect_offset_zero_drift() -> None:
    """Perfect constant-offset stream converges within microsecond tolerance.

    After enough samples, ``compute_server_time(t)`` should match ``t + offset``.
    """
    filter_ = SendspinTimeFilter()
    offset = 250_000

    last_t = _feed_constant_offset(filter_, offset_us=offset, count=20)

    # After 20 perfect samples the filter should be well within 1 us.
    sample_client_time = last_t + 500_000
    assert abs(filter_.compute_server_time(sample_client_time) - (sample_client_time + offset)) <= 1


def test_compute_server_time_monotonic_after_convergence() -> None:
    """After convergence, increasing client_time produces non-decreasing server_time."""
    filter_ = SendspinTimeFilter()
    _feed_constant_offset(filter_, offset_us=10_000, count=20)

    base = 100_000_000
    prev = filter_.compute_server_time(base)
    for delta in (1, 10, 100, 1_000, 10_000, 100_000, 1_000_000):
        current = filter_.compute_server_time(base + delta)
        assert current >= prev
        prev = current


def test_compute_inverse_consistency_zero_drift() -> None:
    """compute_client_time(compute_server_time(t)) ~= t after convergence (zero drift)."""
    filter_ = SendspinTimeFilter()
    _feed_constant_offset(filter_, offset_us=42_000, count=20)

    for client_time in (500_000, 10_000_000, 999_999_999):
        server = filter_.compute_server_time(client_time)
        recovered = filter_.compute_client_time(server)
        assert abs(recovered - client_time) <= 1


def test_compute_inverse_consistency_with_drift() -> None:
    """Inverse consistency holds even when the SNR gate enables drift compensation."""
    filter_ = SendspinTimeFilter()

    # Server runs at 1.0e-4 us/us faster than client (i.e. ~100 us per second of skew).
    drift_rate = 1.0e-4
    base_offset = 100_000

    t = 1_000_000
    for _ in range(50):
        # Measured offset includes accumulated drift since baseline at t=0.
        measurement = round(base_offset + drift_rate * t)
        filter_.update(measurement, max_error=200, time_added=t)
        t += 1_000_000

    # If drift compensation is engaged the inverse must still round-trip.
    for client_time in (t + 0, t + 500_000, t + 5_000_000):
        server = filter_.compute_server_time(client_time)
        recovered = filter_.compute_client_time(server)
        assert abs(recovered - client_time) <= 1


def test_init_branches_do_not_set_use_drift() -> None:
    """The two warmup branches must not enable drift compensation."""
    filter_ = SendspinTimeFilter()

    filter_.update(measurement=1_000, max_error=500, time_added=1_000_000)
    assert filter_._current_time_element.use_drift is False

    filter_.update(measurement=1_000, max_error=500, time_added=2_000_000)
    assert filter_._current_time_element.use_drift is False
