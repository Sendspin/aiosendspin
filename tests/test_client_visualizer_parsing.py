"""Tests for strict client-side visualizer payload parsing (v1 wire)."""

from __future__ import annotations

import asyncio
import struct
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiosendspin.client.connection import SendspinConnection
from aiosendspin.clock import ManualClock
from aiosendspin.models.types import BinaryMessageType, Roles
from aiosendspin.models.visualizer import (
    ClientHelloVisualizerSpectrum,
    ClientHelloVisualizerSupport,
    StreamStartVisualizer,
    VisualizerFrame,
    VisualizerStatePayload,
)

from .conftest import make_sdk_client


def _basic_config(
    *,
    types: tuple[str, ...] = ("loudness",),
    n_disp_bins: int = 8,
) -> StreamStartVisualizer:
    spectrum: ClientHelloVisualizerSpectrum | None = None
    if "spectrum" in types:
        spectrum = ClientHelloVisualizerSpectrum(
            n_disp_bins=n_disp_bins, scale="lin", f_min=20, f_max=16_000
        )
    # The parser only consults `types` + spectrum metadata to validate frame
    # widths, so the rest of the config can stay at defaults.
    return StreamStartVisualizer(types=types, rate_max=30, spectrum=spectrum)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# loudness (msg 16)
# ---------------------------------------------------------------------------


def test_parse_loudness_frame() -> None:
    """Parse loudness frame."""
    payload = struct.pack(">q", 1_234_000) + struct.pack(">H", 12345)
    cfg = _basic_config()
    frame = SendspinConnection._parse_visualization_frame(  # noqa: SLF001
        BinaryMessageType.VISUALIZATION_LOUDNESS, payload, cfg
    )
    assert frame is not None
    assert frame.timestamp_us == 1_234_000
    assert frame.loudness == 12345


def test_parse_loudness_frame_rejects_wrong_length() -> None:
    """Parse loudness frame rejects wrong length."""
    cfg = _basic_config()
    bad = struct.pack(">q", 1) + b"\x00"  # only 1 byte of data instead of 2
    assert (
        SendspinConnection._parse_visualization_frame(  # noqa: SLF001
            BinaryMessageType.VISUALIZATION_LOUDNESS, bad, cfg
        )
        is None
    )


# ---------------------------------------------------------------------------
# f_peak (msg 18)
# ---------------------------------------------------------------------------


def test_parse_f_peak_frame() -> None:
    """Parse f peak frame."""
    payload = struct.pack(">q", 100) + struct.pack(">HH", 1024, 0x4000)
    cfg = _basic_config(types=("f_peak",))
    frame = SendspinConnection._parse_visualization_frame(  # noqa: SLF001
        BinaryMessageType.VISUALIZATION_F_PEAK, payload, cfg
    )
    assert frame is not None
    assert frame.f_peak_freq == 1024
    assert frame.f_peak_amp == 0x4000


def test_parse_f_peak_rejects_wrong_length() -> None:
    """Parse f peak rejects wrong length."""
    cfg = _basic_config(types=("f_peak",))
    bad = struct.pack(">q", 1) + struct.pack(">H", 100)  # missing amp
    assert (
        SendspinConnection._parse_visualization_frame(  # noqa: SLF001
            BinaryMessageType.VISUALIZATION_F_PEAK, bad, cfg
        )
        is None
    )


# ---------------------------------------------------------------------------
# spectrum (msg 19)
# ---------------------------------------------------------------------------


def test_parse_spectrum_frame() -> None:
    """Parse spectrum frame."""
    bins = list(range(8))
    payload = struct.pack(">q", 42) + struct.pack(">8H", *bins)
    cfg = _basic_config(types=("spectrum",), n_disp_bins=8)
    frame = SendspinConnection._parse_visualization_frame(  # noqa: SLF001
        BinaryMessageType.VISUALIZATION_SPECTRUM, payload, cfg
    )
    assert frame is not None
    assert frame.spectrum == bins


def test_parse_spectrum_rejects_wrong_bin_count() -> None:
    """Parse spectrum rejects wrong bin count."""
    payload = struct.pack(">q", 0) + struct.pack(">4H", 1, 2, 3, 4)
    cfg = _basic_config(types=("spectrum",), n_disp_bins=8)
    assert (
        SendspinConnection._parse_visualization_frame(  # noqa: SLF001
            BinaryMessageType.VISUALIZATION_SPECTRUM, payload, cfg
        )
        is None
    )


# ---------------------------------------------------------------------------
# peak (msg 20)
# ---------------------------------------------------------------------------


def test_parse_peak_frame() -> None:
    """Parse peak frame."""
    payload = struct.pack(">q", 99) + bytes([0xC8])
    cfg = _basic_config(types=("peak",))
    frame = SendspinConnection._parse_visualization_frame(  # noqa: SLF001
        BinaryMessageType.VISUALIZATION_PEAK, payload, cfg
    )
    assert frame is not None
    assert frame.peak_strength == 0xC8


# ---------------------------------------------------------------------------
# beat (msg 17) — delivered through the visualizer callback
# ---------------------------------------------------------------------------


_NOW_US = 10_000_000


def _connection_with_visualizer_callback(
    *, synced: bool = False
) -> tuple[SendspinConnection, list[VisualizerFrame]]:
    client = make_sdk_client(
        client_name="x",
        roles=[Roles.VISUALIZER],
        visualizer_support=ClientHelloVisualizerSupport(buffer_capacity=65536),
        visualizer_state=VisualizerStatePayload(types=["loudness", "beat"], rate_max=30),
        clock=ManualClock(now_us_value=_NOW_US),
    )
    received: list[VisualizerFrame] = []

    def _cb(frames: list[VisualizerFrame]) -> None:
        received.extend(frames)

    client.add_visualizer_listener(_cb)
    connection = SendspinConnection(client)
    connection._current_visualizer_config = _basic_config()  # noqa: SLF001
    if synced:
        # Server and client clocks agree.
        time_filter = MagicMock()
        time_filter.count = 2
        time_filter.compute_client_time.side_effect = lambda server_us: server_us
        connection._time_filter = time_filter  # noqa: SLF001
    return connection, received


def _loudness_body(timestamp_us: int) -> bytes:
    return struct.pack(">q", timestamp_us) + struct.pack(">H", 1)


def _beat_body(timestamp_us: int) -> bytes:
    return struct.pack(">q", timestamp_us) + bytes([0])


@pytest.mark.asyncio
async def test_handle_beat_dispatches_downbeat_frame() -> None:
    """Downbeat byte sets is_downbeat=True on the dispatched frame."""
    connection, received = _connection_with_visualizer_callback()
    body = struct.pack(">q", 100) + bytes([0b0000_0001])
    connection._handle_visualization_beat(body)  # noqa: SLF001
    assert len(received) == 1
    assert received[0].timestamp_us == 100
    assert received[0].is_downbeat is True


@pytest.mark.asyncio
async def test_handle_beat_dispatches_regular_frame() -> None:
    """Flags=0 dispatches is_downbeat=False."""
    connection, received = _connection_with_visualizer_callback()
    body = struct.pack(">q", 200) + bytes([0])
    connection._handle_visualization_beat(body)  # noqa: SLF001
    assert len(received) == 1
    assert received[0].timestamp_us == 200
    assert received[0].is_downbeat is False


@pytest.mark.asyncio
async def test_handle_beat_rejects_wrong_length() -> None:
    """Body without the trailing flag byte is dropped silently."""
    connection, received = _connection_with_visualizer_callback()
    connection._handle_visualization_beat(struct.pack(">q", 100))  # noqa: SLF001
    assert received == []


@pytest.mark.asyncio
async def test_handle_beat_rejects_empty_payload() -> None:
    """Empty body is dropped silently."""
    connection, received = _connection_with_visualizer_callback()
    connection._handle_visualization_beat(b"")  # noqa: SLF001
    assert received == []


# ---------------------------------------------------------------------------
# Reserved type 21 (formerly pitch)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reserved_type_21_is_ignored() -> None:
    """A type-21 binary reaches no visualizer callback and raises nothing."""
    connection, received = _connection_with_visualizer_callback()
    connection._visualizer_stream_active = True  # noqa: SLF001

    body = struct.pack(">q", _NOW_US) + struct.pack(">H", 0x4500) + bytes([200])
    connection._handle_binary_message(bytes([21]) + body)  # noqa: SLF001
    connection._handle_binary_message(  # noqa: SLF001
        bytes([BinaryMessageType.VISUALIZATION_LOUDNESS.value]) + _loudness_body(_NOW_US)
    )

    assert [frame.loudness for frame in received] == [1]


# ---------------------------------------------------------------------------
# Late data
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_past_frame_and_beat_are_dropped() -> None:
    """A frame or beat whose timestamp is already past on the local clock is not delivered."""
    connection, received = _connection_with_visualizer_callback(synced=True)

    connection._handle_visualization_frame(  # noqa: SLF001
        BinaryMessageType.VISUALIZATION_LOUDNESS, _loudness_body(_NOW_US - 1)
    )
    connection._handle_visualization_beat(_beat_body(_NOW_US - 1))  # noqa: SLF001

    assert received == []


@pytest.mark.asyncio
async def test_future_frame_and_beat_are_delivered() -> None:
    """A frame or beat still ahead of the local clock is delivered."""
    connection, received = _connection_with_visualizer_callback(synced=True)

    connection._handle_visualization_frame(  # noqa: SLF001
        BinaryMessageType.VISUALIZATION_LOUDNESS, _loudness_body(_NOW_US + 1)
    )
    connection._handle_visualization_beat(_beat_body(_NOW_US))  # noqa: SLF001

    assert [frame.timestamp_us for frame in received] == [_NOW_US + 1, _NOW_US]


@pytest.mark.asyncio
async def test_frames_are_delivered_before_time_sync() -> None:
    """Without a time-sync measurement, lateness is not judged and data is delivered."""
    connection, received = _connection_with_visualizer_callback()

    connection._handle_visualization_frame(  # noqa: SLF001
        BinaryMessageType.VISUALIZATION_LOUDNESS, _loudness_body(0)
    )
    connection._handle_visualization_beat(_beat_body(0))  # noqa: SLF001

    assert [frame.timestamp_us for frame in received] == [0, 0]


@pytest.mark.asyncio
async def test_unavailable_client_discards_frames_and_beats() -> None:
    """While unavailable, frames and beats are discarded without closing the connection."""
    connection, received = _connection_with_visualizer_callback(synced=True)
    connection.disconnect = AsyncMock()  # type: ignore[method-assign]
    connection._reported_available = False  # noqa: SLF001

    connection._handle_visualization_frame(  # noqa: SLF001
        BinaryMessageType.VISUALIZATION_LOUDNESS, _loudness_body(_NOW_US + 1)
    )
    connection._handle_visualization_beat(_beat_body(_NOW_US + 1))  # noqa: SLF001
    assert received == []

    connection._reported_available = True  # noqa: SLF001
    connection._handle_visualization_frame(  # noqa: SLF001
        BinaryMessageType.VISUALIZATION_LOUDNESS, _loudness_body(_NOW_US + 2)
    )
    connection._handle_visualization_beat(_beat_body(_NOW_US + 2))  # noqa: SLF001

    assert [frame.timestamp_us for frame in received] == [_NOW_US + 2, _NOW_US + 2]
    await asyncio.sleep(0)
    connection.disconnect.assert_not_awaited()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Truncated header
# ---------------------------------------------------------------------------


def test_parse_visualization_frame_rejects_truncated_header() -> None:
    """Parse visualization frame rejects truncated header."""
    cfg = _basic_config()
    assert (
        SendspinConnection._parse_visualization_frame(  # noqa: SLF001
            BinaryMessageType.VISUALIZATION_LOUDNESS, b"\x00\x01", cfg
        )
        is None
    )
