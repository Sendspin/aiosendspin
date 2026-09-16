"""Tests for source role models and their wiring into core payloads."""

from __future__ import annotations

import pytest

from aiosendspin.models.core import (
    ClientHelloPayload,
    ClientStatePayload,
    PairMethodDescriptor,
    ServerCommandPayload,
)
from aiosendspin.models.source import (
    ClientStreamEndMessage,
    ClientStreamStartMessage,
    ClientStreamStartSource,
)
from aiosendspin.models.types import (
    AudioCodec,
    ClientMessage,
    PairMethod,
    SignalState,
)


def _hello_dict() -> dict:
    return {
        "name": "Kitchen Line-In",
        "supported_roles": ["source@v1"],
        "source@v1_support": {
            "features": {"line_sense": True},
        },
    }


def test_hello_parses_source_support_and_features() -> None:
    """A source client/hello populates source_support with its features."""
    hello = ClientHelloPayload.from_dict(_hello_dict())
    assert hello.source_support is not None
    assert hello.source_support.features is not None
    assert hello.source_support.features.line_sense is True


def test_hello_serializes_support_under_versioned_alias() -> None:
    """source_support round-trips under the wire key ``source@v1_support``."""
    hello = ClientHelloPayload.from_dict(_hello_dict())
    assert "source@v1_support" in hello.to_dict()


def test_hello_source_role_requires_support_object() -> None:
    """Listing source@v1 requires its versioned support object."""
    with pytest.raises(ValueError, match="source@v1_support"):
        ClientHelloPayload(name="x", supported_roles=["source@v1"])


def test_hello_drops_source_support_without_role() -> None:
    """source_support is cleared when source@v1 is not in supported_roles."""
    hello_dict = _hello_dict() | {"supported_roles": ["controller@v1"]}
    assert ClientHelloPayload.from_dict(hello_dict).source_support is None


def test_hello_preserves_supported_pair_methods_positional_argument() -> None:
    """Source support is appended, so it does not displace supported_pair_methods."""
    pair_methods = [PairMethodDescriptor(method=PairMethod.PAIRING_PSK)]
    payload = ClientHelloPayload(
        "Client",
        [],
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        pair_methods,
    )

    assert payload.supported_pair_methods == pair_methods
    assert payload.source_support is None


def test_client_state_carries_source_signal() -> None:
    """client/state parses the source subobject, which now carries only signal."""
    payload = ClientStatePayload.from_dict({"source": {"signal": "present"}})
    assert payload.source is not None
    assert payload.source.signal is SignalState.PRESENT


def test_client_state_preserves_legacy_flag_positional_argument() -> None:
    """Source state does not displace existing client/state positional arguments."""
    payload = ClientStatePayload(True, None, True)  # noqa: FBT003

    assert payload.legacy_state_used is True
    assert payload.source is None
    assert payload.to_dict() == {"available": True, "legacy_state_used": True}


def test_server_command_source_carries_start_stop() -> None:
    """server/command carries a required 'start'/'stop' source command."""
    server_cmd = ServerCommandPayload.from_dict({"source": {"command": "start"}})
    assert server_cmd.source is not None
    assert server_cmd.source.command == "start"


def test_server_command_source_requires_command() -> None:
    """The source command field is required (no default) per the simplified spec."""
    from mashumaro.exceptions import InvalidFieldValue  # noqa: PLC0415

    with pytest.raises(InvalidFieldValue):
        ServerCommandPayload.from_dict({"source": {}})


def test_client_stream_messages_dispatch_by_discriminator() -> None:
    """client_stream messages resolve to their concrete classes via the type field."""
    start = ClientMessage.from_json(
        '{"type":"client_stream/start","payload":{"source":'
        '{"codec":"flac","channels":2,"sample_rate":48000,"bit_depth":16,"codec_header":"AAA="}}}'
    )
    assert isinstance(start, ClientStreamStartMessage)
    assert start.payload.source.codec is AudioCodec.FLAC

    end = ClientMessage.from_json('{"type":"client_stream/end"}')
    assert isinstance(end, ClientStreamEndMessage)


def test_client_stream_start_header_optional_for_all_codecs() -> None:
    """Message parsing leaves codec-specific header validation to the source role."""
    for codec in (AudioCodec.OPUS, AudioCodec.FLAC, AudioCodec.PCM):
        src = ClientStreamStartSource(codec=codec, channels=2, sample_rate=48000, bit_depth=16)
        assert src.codec_header is None
