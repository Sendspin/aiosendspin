"""Tests that wire values of the wrong JSON type are recorded, converted or rejected."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import orjson
import pytest
from mashumaro.exceptions import InvalidFieldValue

from aiosendspin.models.base import SendspinModel, parse_noting_wire_deviations
from aiosendspin.models.core import ClientMessage
from aiosendspin.noise.models import PairingMessage


@dataclass
class _Fields(SendspinModel):
    number: int = 0
    flag: bool = False
    text: str = ""
    names: list[str] = field(default_factory=list)


def _parse(model: type[SendspinModel], data: object) -> tuple[Any, list[str]]:
    return parse_noting_wire_deviations(model.from_json, orjson.dumps(data))


@pytest.mark.parametrize(
    ("key", "value", "expected", "reason"),
    [
        ("number", 50.7, 50, "sent a number for 'number' instead of an integer"),
        ("number", True, 1, "sent a boolean for 'number' instead of an integer"),
        ("number", "42", 42, "sent a string for 'number' instead of an integer"),
        ("flag", "false", False, "sent a string for 'flag' instead of a boolean"),
        ("flag", 0, False, "sent an integer for 'flag' instead of a boolean"),
        ("text", 42, "42", "sent an integer for 'text' instead of a string"),
        ("names", [1, 2], ["1", "2"], "sent an integer for 'names' instead of a string"),
    ],
)
def test_tolerated_value_is_converted_and_recorded_once(
    key: str, value: object, expected: object, reason: str
) -> None:
    """A value with a clear reading converts, and each distinct deviation is recorded once."""
    parsed, reasons = _parse(_Fields, {key: value})

    assert getattr(parsed, key) == expected
    assert reasons == [reason]


def test_integral_float_is_an_integer() -> None:
    """JSON has one number type, so 50.0 is an integer and is not recorded."""
    parsed, reasons = _parse(_Fields, {"number": 50.0})

    assert parsed.number == 50
    assert type(parsed.number) is int
    assert reasons == []


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("flag", "yes"),
        ("flag", {}),
        ("flag", 2),
        ("text", None),
        ("text", True),
        ("names", "player@v1"),
        ("names", {"player@v1": 1}),
    ],
)
def test_value_without_a_reading_is_rejected(key: str, value: object) -> None:
    """A value whose conversion would change its meaning fails the parse."""
    with pytest.raises(InvalidFieldValue):
        _parse(_Fields, {key: value})


@pytest.mark.parametrize("literal", ["1e30", '"99999999999999999999"'])
def test_integer_beyond_64_bits_is_rejected(literal: str) -> None:
    """An integer the server could not echo back is rejected at parse."""
    text = '{"type": "client/time", "payload": {"client_transmitted": ' + literal + "}}"

    with pytest.raises(InvalidFieldValue):
        parse_noting_wire_deviations(ClientMessage.from_json, text)


def test_reason_names_the_nested_field_by_its_wire_key() -> None:
    """The recorded field path follows nested objects and uses aliased wire keys."""
    support = {
        "supported_formats": [
            {"codec": "pcm", "channels": "2", "sample_rate": 48000, "bit_depth": 16}
        ],
        "buffer_capacity": 100_000,
        "supported_commands": [],
    }
    hello = {
        "type": "client/hello",
        "payload": {
            "name": "Kitchen",
            "supported_roles": ["player@v1"],
            "player@v1_support": support,
        },
    }

    _, reasons = _parse(ClientMessage, hello)

    assert reasons == [
        "sent a string for 'player@v1_support.supported_formats.channels' instead of an integer"
    ]


@pytest.mark.parametrize(
    ("message", "reasons"),
    [
        ({"type": "client/pair-retry", "payload": None}, ["omitted the required payload object"]),
        ({"type": "client/pair-retry"}, ["omitted the required payload object"]),
        ({"type": "client/pair-retry", "payload": []}, ["sent a payload that is not an object"]),
        ({"type": "client/pair-retry", "payload": {}}, []),
    ],
)
def test_pair_retry_without_payload_is_recorded(message: object, reasons: list[str]) -> None:
    """A client/pair-retry whose payload is not an object still parses, and is recorded."""
    _, recorded = _parse(PairingMessage, message)

    assert recorded == reasons
