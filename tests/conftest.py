"""Shared test fixtures for aiosendspin tests."""

from __future__ import annotations

import math
import struct
from typing import Any
from unittest.mock import MagicMock

import pytest

from aiosendspin.client.client import SendspinClient
from aiosendspin.noise.keys import Identity
from aiosendspin.noise.trust_store import ClientPairingStore, InMemoryClientPairingStore


def make_sdk_client(
    *,
    identity: Identity | None = None,
    pairing_store: ClientPairingStore | None = None,
    **kwargs: Any,
) -> SendspinClient:
    """Build a client-SDK ``SendspinClient`` with the Noise dependencies defaulted.

    Existing client tests don't drive the Noise handshake, so this supplies a
    freshly generated identity and an empty in-memory pairing store, letting
    callers pass only their domain-relevant kwargs.
    """
    return SendspinClient(
        identity=identity or Identity.generate(),
        pairing_store=pairing_store or InMemoryClientPairingStore(),
        **kwargs,
    )


def sine_pcm_16bit(samples: int, *, rate: int = 48000, channels: int = 2, freq: int = 440) -> bytes:
    """Generate interleaved 16-bit sine PCM for tests."""
    out = bytearray()
    for n in range(samples):
        out += struct.pack("<h", int(30000 * math.sin(2 * math.pi * freq * n / rate))) * channels
    return bytes(out)


@pytest.fixture
def mock_loop() -> MagicMock:
    """Mock event loop with time() returning 0.0."""
    loop = MagicMock()
    loop.time.return_value = 0.0
    return loop


@pytest.fixture
def pcm_44100_stereo_16bit() -> bytes:
    """25ms of silence at 44100Hz stereo 16-bit (4408 bytes).

    Calculation: 44100 Hz * 0.025s * 2 channels * 2 bytes = 4410 bytes
    Rounded to frame alignment: 4408 bytes (1102 samples * 4 bytes/frame)
    """
    # 1102 samples at 44100Hz = ~24.99ms
    sample_count = 1102
    bytes_per_sample = 2  # 16-bit
    channels = 2
    return bytes(sample_count * bytes_per_sample * channels)


@pytest.fixture
def pcm_48000_stereo_16bit() -> bytes:
    """25ms of silence at 48000Hz stereo 16-bit (4800 bytes).

    Calculation: 48000 Hz * 0.025s * 2 channels * 2 bytes = 4800 bytes
    """
    sample_count = 1200  # 1200 samples at 48000Hz = 25ms
    bytes_per_sample = 2  # 16-bit
    channels = 2
    return bytes(sample_count * bytes_per_sample * channels)
