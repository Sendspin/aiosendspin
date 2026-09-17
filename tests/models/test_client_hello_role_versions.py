"""Tests for client/hello role version handling and support object validation."""

from __future__ import annotations

from aiosendspin.models.core import ClientHelloPayload, DeviceInfo
from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.models.types import AudioCodec


def test_player_support_required_only_for_v1_role_id() -> None:
    """player@v2 without player@v1_support must not raise."""
    payload = ClientHelloPayload(
        client_id="c1",
        name="Client",
        version=1,
        supported_roles=["player@v2"],
    )
    assert payload.player_support is None


def test_player_v1_without_support_is_recorded_and_not_activatable() -> None:
    """player@v1 without player@v1_support is recorded as missing it, not rejected."""
    payload = ClientHelloPayload(
        client_id="c1",
        name="Client",
        version=1,
        supported_roles=["player@v1", "controller@v1"],
    )
    assert payload.missing_support_roles == ["player@v1"]
    assert payload.activatable_roles == ["controller@v1"]


def test_missing_support_record_cannot_be_spoofed() -> None:
    """missing_support_roles is taken from the parse, never from the wire."""
    payload = ClientHelloPayload.from_dict(
        {
            "name": "Client",
            "supported_roles": ["controller@v1"],
            "missing_support_roles": ["controller@v1"],
        }
    )
    assert payload.missing_support_roles is None
    assert payload.activatable_roles == ["controller@v1"]


def test_player_support_accepted_for_player_v1() -> None:
    """player@v1 accepts player@v1_support via player_support alias."""
    payload = ClientHelloPayload(
        client_id="c1",
        name="Client",
        version=1,
        supported_roles=["player@v1"],
        player_support=ClientHelloPlayerSupport(
            supported_formats=[
                SupportedAudioFormat(
                    codec=AudioCodec.PCM, sample_rate=48000, bit_depth=16, channels=2
                )
            ],
            buffer_capacity=100_000,
            supported_commands=[],
        ),
    )
    assert payload.player_support is not None


def test_player_support_serializes_with_spec_alias_name() -> None:
    """Serialized payload should use player@v1_support instead of legacy player_support."""
    payload = ClientHelloPayload(
        client_id="c1",
        name="Client",
        version=1,
        supported_roles=["player@v1"],
        player_support=ClientHelloPlayerSupport(
            supported_formats=[
                SupportedAudioFormat(
                    codec=AudioCodec.PCM, sample_rate=48000, bit_depth=16, channels=2
                )
            ],
            buffer_capacity=100_000,
            supported_commands=[],
        ),
    )
    serialized = payload.to_dict()
    assert "player@v1_support" in serialized
    assert "player_support" not in serialized


def test_device_info_serializes_mac_address_under_spec_key() -> None:
    """A set mac_address serializes under the spec wire key with its value."""
    serialized = DeviceInfo(mac_address="aa:bb:cc:dd:ee:ff").to_dict()
    assert serialized["mac_address"] == "aa:bb:cc:dd:ee:ff"


def test_device_info_omits_mac_address_when_unset() -> None:
    """An unset mac_address is omitted from the wire payload."""
    assert "mac_address" not in DeviceInfo().to_dict()
