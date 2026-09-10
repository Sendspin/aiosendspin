"""Tests for the pairing-related fields on client/hello and server/hello."""

from __future__ import annotations

import orjson

from aiosendspin.models.core import (
    ActivatePairing,
    ClientHelloPayload,
    DynamicPairMethodDescriptor,
    PairMethodDescriptor,
    ServerActivatePayload,
    ServerHelloPayload,
    SupportedPairMethods,
    UnpairedAccess,
)
from aiosendspin.models.types import Activity, PairMethod, TrustLevel


def test_client_hello_pairing_fields_round_trip() -> None:
    """client/hello carries trust, pairing methods, and unpaired-access flag."""
    payload = ClientHelloPayload(
        client_id="c1",
        name="Client",
        version=1,
        supported_roles=["controller@v1"],
        trust_level=TrustLevel.USER,
        supported_pair_methods=SupportedPairMethods(pairing_psk=PairMethodDescriptor()),
        unpaired_access=UnpairedAccess(enabled=True),
    )
    restored = ClientHelloPayload.from_json(payload.to_json())
    assert restored == payload
    assert restored.trust_level is TrustLevel.USER
    assert restored.supported_pair_methods == SupportedPairMethods(
        pairing_psk=PairMethodDescriptor()
    )
    assert restored.unpaired_access.enabled is True


def test_client_hello_defaults_when_pairing_fields_absent() -> None:
    """A hello without the new fields deserializes to spec-safe defaults (legacy clients)."""
    legacy = '{"client_id":"c1","name":"Client","version":1,"supported_roles":["controller@v1"]}'
    payload = ClientHelloPayload.from_json(legacy)
    assert payload.trust_level is TrustLevel.NONE
    assert payload.supported_pair_methods is None
    assert payload.unpaired_access == UnpairedAccess(enabled=False)


def test_server_hello_round_trips() -> None:
    """server/hello carries the name."""
    payload = ServerHelloPayload(name="Server")
    restored = ServerHelloPayload.from_json(payload.to_json())
    assert restored == payload
    assert restored.name == "Server"


def test_server_activate_pairing_object_round_trips() -> None:
    """server/activate carries the pairing object when present, else omits it."""
    payload = ServerActivatePayload(
        activities=[Activity.PAIRING],
        pairing=ActivatePairing(method=PairMethod.DYNAMIC_PAIRING_CODE, format="digits"),
    )
    restored = ServerActivatePayload.from_json(payload.to_json())
    assert restored.pairing == ActivatePairing(
        method=PairMethod.DYNAMIC_PAIRING_CODE, format="digits"
    )
    assert restored.activities == [Activity.PAIRING]
    assert ServerActivatePayload.from_json('{"activities":["playback"]}').pairing is None


def test_activate_pairing_omits_format_for_non_dynamic_methods() -> None:
    """The pairing object omits format when the method carries none."""
    payload = ServerActivatePayload(
        activities=[Activity.PAIRING],
        pairing=ActivatePairing(method=PairMethod.STATIC_PAIRING_CODE),
    )
    raw = orjson.loads(payload.to_json())
    assert raw["pairing"] == {"method": "static_pairing_code"}


def test_unrecognized_descriptor_format_is_dropped() -> None:
    """A format from a newer spec revision is ignored rather than selected or rejected."""
    raw = (
        '{"client_id":"c1","name":"Client","version":1,"supported_roles":["controller@v1"],'
        '"supported_pair_methods":{"dynamic_pairing_code":'
        '{"formats":["digits","holographic"],"out_channels":["display"]}}}'
    )
    payload = ClientHelloPayload.from_json(raw)
    assert payload.supported_pair_methods is not None
    dynamic = payload.supported_pair_methods.dynamic_pairing_code
    assert dynamic is not None
    assert dynamic.formats == ["digits"]


def test_unrecognized_activate_format_still_parses() -> None:
    """An activate with a newer-revision format parses; the client aborts, not errors."""
    raw = (
        '{"activities":["pairing"],'
        '"pairing":{"method":"dynamic_pairing_code","format":"holographic"}}'
    )
    payload = ServerActivatePayload.from_json(raw)
    assert payload.pairing is not None
    assert payload.pairing.format == "holographic"


def test_activate_pairing_languages_round_trip() -> None:
    """The dynamic pairing object carries the spoken-emission language hint, or omits it."""
    payload = ServerActivatePayload(
        activities=[Activity.PAIRING],
        pairing=ActivatePairing(
            method=PairMethod.DYNAMIC_PAIRING_CODE, format="digits", languages=["ca", "es", "en"]
        ),
    )
    restored = ServerActivatePayload.from_json(payload.to_json())
    assert restored.pairing is not None
    assert restored.pairing.languages == ["ca", "es", "en"]
    bare = ActivatePairing(method=PairMethod.DYNAMIC_PAIRING_CODE, format="digits")
    assert "languages" not in orjson.loads(bare.to_json())


def test_pair_method_descriptor_locations_round_trip() -> None:
    """A static-secret descriptor carries the locations hint, or omits it entirely."""
    descriptor = PairMethodDescriptor(locations=["device", "leaflet"])
    restored = PairMethodDescriptor.from_json(descriptor.to_json())
    assert restored.locations == ["device", "leaflet"]
    assert orjson.loads(PairMethodDescriptor().to_json()) == {}


def test_supported_pair_methods_serializes_keyed_by_method() -> None:
    """The advertisement goes on the wire as an object keyed by method identifier."""
    payload = ClientHelloPayload(
        client_id="c1",
        name="Client",
        version=1,
        supported_roles=["controller@v1"],
        supported_pair_methods=SupportedPairMethods(
            pairing_psk=PairMethodDescriptor(locations=["device"]),
            dynamic_pairing_code=DynamicPairMethodDescriptor(
                out_channels=["display"], formats=["digits"]
            ),
        ),
    )
    raw = orjson.loads(payload.to_json())
    assert raw["supported_pair_methods"] == {
        "pairing_psk": {"locations": ["device"]},
        "dynamic_pairing_code": {"out_channels": ["display"], "formats": ["digits"]},
    }


def test_superseded_pair_methods_list_is_accepted() -> None:
    """A client advertising the superseded list is understood, and the shape recorded."""
    raw = (
        '{"client_id":"c1","name":"Client","version":1,"supported_roles":["controller@v1"],'
        '"supported_pair_methods":[{"method":"pairing_psk","locations":["device"]},'
        '{"method":"dynamic_pairing_code","formats":["digits"],"out_channels":["speaker"]}]}'
    )
    payload = ClientHelloPayload.from_json(raw)
    assert payload.legacy_pair_methods_list_used is True
    methods = payload.supported_pair_methods
    assert methods is not None
    assert methods.pairing_psk == PairMethodDescriptor(locations=["device"])
    assert methods.dynamic_pairing_code == DynamicPairMethodDescriptor(
        out_channels=["speaker"], formats=["digits"]
    )


def test_current_pair_methods_object_is_not_flagged_as_legacy() -> None:
    """The keyed object is the current wire, so it sets no legacy record."""
    raw = (
        '{"client_id":"c1","name":"Client","version":1,"supported_roles":["controller@v1"],'
        '"supported_pair_methods":{"pairing_psk":{}}}'
    )
    payload = ClientHelloPayload.from_json(raw)
    assert payload.legacy_pair_methods_list_used is None
    assert payload.supported_pair_methods == SupportedPairMethods(
        pairing_psk=PairMethodDescriptor()
    )


def test_unrecognized_pair_method_is_ignored_not_rejected() -> None:
    """An identifier from a newer revision is dropped, its value never validated."""
    raw = (
        '{"client_id":"c1","name":"Client","version":1,"supported_roles":["controller@v1"],'
        '"supported_pair_methods":{"pairing_psk":{},"telepathy":{"anything":[1,2]}}}'
    )
    methods = ClientHelloPayload.from_json(raw).supported_pair_methods
    assert methods is not None
    assert methods.pairing_psk == PairMethodDescriptor()
    assert methods.ignored_methods == ["telepathy"]


def test_both_pairing_code_methods_degrade_to_dynamic() -> None:
    """Offering both code methods is a spec violation; the static one is disregarded."""
    raw = (
        '{"client_id":"c1","name":"Client","version":1,"supported_roles":["controller@v1"],'
        '"supported_pair_methods":{"static_pairing_code":{},'
        '"dynamic_pairing_code":{"formats":["digits"],"out_channels":["display"]}}}'
    )
    methods = ClientHelloPayload.from_json(raw).supported_pair_methods
    assert methods is not None
    assert methods.static_pairing_code is None
    assert methods.dynamic_pairing_code is not None
    assert methods.offered_both_pairing_code_methods is True


def test_dynamic_without_recognized_values_is_dropped() -> None:
    """A dynamic descriptor left with nothing usable is dropped, and recorded on its own.

    It is not an identifier the reader failed to recognize, so it is kept apart from those.
    """
    raw = (
        '{"client_id":"c1","name":"Client","version":1,"supported_roles":["controller@v1"],'
        '"supported_pair_methods":{"pairing_psk":{},'
        '"dynamic_pairing_code":{"formats":["holographic"],"out_channels":["display"]}}}'
    )
    methods = ClientHelloPayload.from_json(raw).supported_pair_methods
    assert methods is not None
    assert methods.dynamic_pairing_code is None
    assert methods.unusable_methods == ["dynamic_pairing_code"]
    assert methods.ignored_methods is None


def test_both_offered_disregards_static_even_when_dynamic_is_unusable() -> None:
    """Receiving both identifiers disregards the static descriptor, whatever the dynamic holds."""
    raw = (
        '{"client_id":"c1","name":"Client","version":1,"supported_roles":["controller@v1"],'
        '"supported_pair_methods":{"pairing_psk":{},"static_pairing_code":{},'
        '"dynamic_pairing_code":{"formats":["holographic"],"out_channels":["display"]}}}'
    )
    methods = ClientHelloPayload.from_json(raw).supported_pair_methods
    assert methods is not None
    assert methods.offered_both_pairing_code_methods is True
    assert methods.static_pairing_code is None
    assert methods.dynamic_pairing_code is None
    assert methods.unusable_methods == ["dynamic_pairing_code"]
    assert methods.pairing_psk is not None


def test_structured_descriptor_values_are_ignored_not_rejected() -> None:
    """A value that is not an identifier at all is ignored, like any unrecognized one.

    Objects and arrays cannot be looked up as identifiers, so a reader that tested
    membership first would fail the whole hello rather than skipping the value.
    """
    raw = (
        '{"client_id":"c1","name":"Client","version":1,"supported_roles":["controller@v1"],'
        '"supported_pair_methods":{"pairing_psk":{"locations":["device",{"a":1},["b"],7]}}}'
    )
    methods = ClientHelloPayload.from_json(raw).supported_pair_methods
    assert methods is not None
    assert methods.pairing_psk is not None
    assert methods.pairing_psk.locations == ["device"]


def test_dynamic_with_only_structured_values_is_dropped() -> None:
    """The same rule leaves a dynamic descriptor with nothing usable, so it is dropped."""
    raw = (
        '{"client_id":"c1","name":"Client","version":1,"supported_roles":["controller@v1"],'
        '"supported_pair_methods":{"pairing_psk":{},"dynamic_pairing_code":'
        '{"formats":[{"a":1}],"out_channels":["display"]}}}'
    )
    methods = ClientHelloPayload.from_json(raw).supported_pair_methods
    assert methods is not None
    assert methods.dynamic_pairing_code is None
    assert methods.unusable_methods == ["dynamic_pairing_code"]


def test_pair_method_records_cannot_be_spoofed_over_the_wire() -> None:
    """The parser overwrites its own records, so a client cannot plant them."""
    raw = (
        '{"client_id":"c1","name":"Client","version":1,"supported_roles":["controller@v1"],'
        '"supported_pair_methods":{"pairing_psk":{},"ignored_methods":["invented"],'
        '"unusable_methods":["invented"],"offered_both_pairing_code_methods":true}}'
    )
    methods = ClientHelloPayload.from_json(raw).supported_pair_methods
    assert methods is not None
    assert methods.ignored_methods is None
    assert methods.unusable_methods is None
    assert methods.offered_both_pairing_code_methods is None
