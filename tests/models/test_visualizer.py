"""Tests for the visualizer hello support and client/state models."""

from __future__ import annotations

import pytest
from mashumaro.exceptions import MissingField

from aiosendspin.models.core import ClientHelloPayload, ClientStatePayload
from aiosendspin.models.visualizer import (
    ClientHelloVisualizerSpectrum,
    ClientHelloVisualizerSupport,
    StreamStartVisualizer,
    VisualizerStatePayload,
)

_SPECTRUM = {"n_disp_bins": 16, "scale": "mel", "f_min": 20, "f_max": 16_000}


def test_client_state_visualizer_round_trips() -> None:
    """A client/state visualizer object survives a JSON round trip."""
    payload = ClientStatePayload(
        available=True,
        visualizer=VisualizerStatePayload(
            types=["spectrum", "beat"],
            rate_max=30,
            spectrum=ClientHelloVisualizerSpectrum.from_dict(_SPECTRUM),
        ),
    )

    assert ClientStatePayload.from_json(payload.to_json()) == payload


def test_client_state_visualizer_accepts_empty_types() -> None:
    """`types` may be empty to request no visualization data."""
    payload = ClientStatePayload.from_dict({"visualizer": {"types": [], "rate_max": 10}})

    assert payload.visualizer == VisualizerStatePayload(types=[], rate_max=10)
    assert payload.to_dict()["visualizer"] == {"types": [], "rate_max": 10}


def test_client_state_visualizer_parses_spec_deviations() -> None:
    """`spectrum` without its configuration and a bad rate_max parse, for the server to flag."""
    state = VisualizerStatePayload.from_dict({"types": ["spectrum"], "rate_max": 0})

    assert state.types == ["spectrum"]
    assert state.spectrum is None
    assert state.rate_max == 0


def test_client_state_visualizer_drops_duplicate_and_unknown_types() -> None:
    """Duplicate and unknown types are dropped, keeping order."""
    state = VisualizerStatePayload.from_dict(
        {"types": ["beat", "future_type", "loudness", "beat"], "rate_max": 10}
    )

    assert state.types == ["beat", "loudness"]


def test_client_state_visualizer_requires_rate_max() -> None:
    """rate_max is required."""
    with pytest.raises(MissingField):
        VisualizerStatePayload.from_dict({"types": []})


def test_hello_visualizer_support_with_only_buffer_capacity() -> None:
    """A hello support object with only buffer_capacity parses and serializes back unchanged."""
    hello = ClientHelloPayload.from_dict(
        {
            "name": "c",
            "supported_roles": ["visualizer@v1"],
            "visualizer@v1_support": {"buffer_capacity": 4096},
        }
    )

    support = hello.visualizer_support
    assert support is not None
    assert support == ClientHelloVisualizerSupport(buffer_capacity=4096)
    assert support.has_stream_config is False
    assert hello.to_dict()["visualizer@v1_support"] == {"buffer_capacity": 4096}


# DEPRECATED(spec-pr-195): remove in aiosendspin <version>
def test_hello_visualizer_support_keeps_legacy_stream_config() -> None:
    """A pre-#195 support object keeps its stream configuration for the server."""
    support = ClientHelloVisualizerSupport.from_dict(
        {"buffer_capacity": 4096, "types": ["loudness", "loudness", "spectrum"], "rate_max": 30}
    )

    assert support.has_stream_config is True
    assert support.types == ["loudness", "spectrum"]
    assert support.rate_max == 30
    assert support.spectrum is None


def test_stream_start_from_request_allows_empty_types() -> None:
    """A stream config can be derived from an empty request."""
    config = StreamStartVisualizer.from_request(VisualizerStatePayload(types=[], rate_max=10))

    assert config == StreamStartVisualizer(types=(), rate_max=10)


def test_stream_start_from_request_carries_spectrum_only_with_its_type() -> None:
    """The spectrum configuration is echoed only when `spectrum` is streamed."""
    spectrum = ClientHelloVisualizerSpectrum.from_dict(_SPECTRUM)
    request = VisualizerStatePayload(types=["loudness"], rate_max=10, spectrum=spectrum)

    assert StreamStartVisualizer.from_request(request).spectrum is None
