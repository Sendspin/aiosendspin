"""Tests that fields the spec types as `integer` always reach the wire as integers."""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import json
import pkgutil
from enum import IntEnum
from typing import Any

import pytest
from mashumaro.mixins.orjson import DataClassORJSONMixin

import aiosendspin.models as models_package
import aiosendspin.noise.models as noise_models
from aiosendspin.models.base import SendspinConfig, int_to_wire
from aiosendspin.models.controller import ControllerStatePayload
from aiosendspin.models.metadata import Progress, SessionUpdateMetadata
from aiosendspin.models.player import StreamStartPlayer
from aiosendspin.models.types import AudioCodec, RepeatMode


class _Level(IntEnum):
    LOUD = 11


class _Indexable:
    """Stands in for an exact-integer scalar such as `numpy.int64`."""

    def __index__(self) -> int:
        return 7


def _json_types(value: Any) -> list[type]:
    """Collect the Python type of every scalar in a parsed JSON structure."""
    if isinstance(value, dict):
        return [t for v in value.values() for t in _json_types(v)]
    if isinstance(value, list):
        return [t for v in value for t in _json_types(v)]
    return [type(value)]


# int_to_wire semantics


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (217000, 217000),
        (0, 0),
        (-5, -5),
        (_Level.LOUD, 11),
        (_Indexable(), 7),
        (217000.0, 217000),
        (-217000.0, -217000),
        # 2.9 * 1000 is 2899.9999999999995; truncating would lose a whole millisecond.
        (2.9 * 1000, 2900),
        (217000.4, 217000),
        (217000.6, 217001),
    ],
)
def test_int_to_wire_coerces(value: Any, expected: int) -> None:
    """Exact integers pass through and floats round to the nearest integer."""
    result = int_to_wire(value)
    assert result == expected
    assert type(result) is int


@pytest.mark.parametrize("value", ["217000", None, b"1", [1]])
def test_int_to_wire_rejects_non_numbers(value: Any) -> None:
    """A value that is not a number has no defensible integer form."""
    with pytest.raises(TypeError, match="expected an integer"):
        int_to_wire(value)


@pytest.mark.parametrize("value", [True, False])
def test_int_to_wire_rejects_bool(value: bool) -> None:  # noqa: FBT001
    """A bool must not become a plausible 1/0 on the wire.

    `bool` subclasses `int`, so type checkers accept it wherever an `int` is annotated
    and `operator.index` would happily convert it. Coercing it would turn a caller bug
    into a wire-valid value no client would reject.
    """
    with pytest.raises(TypeError, match="got bool"):
        int_to_wire(value)


def test_bool_typed_fields_are_unaffected() -> None:
    """Rejecting bool for int fields must not disturb genuinely bool-typed fields."""
    controller = ControllerStatePayload(
        supported_commands=[], volume=0, muted=True, repeat=RepeatMode.OFF, shuffle=True
    )
    payload = json.loads(controller.to_json())
    assert payload["muted"] is True
    assert payload["shuffle"] is True


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_int_to_wire_rejects_non_finite(value: float) -> None:
    """NaN and the infinities cannot be represented as integers."""
    with pytest.raises(ValueError, match="non-finite"):
        int_to_wire(value)


# The reported regression


def test_float_track_duration_serializes_as_integer() -> None:
    """A float track_duration must not reach the wire as `217000.0`.

    sendspin-cpp validates against the wire type and drops a mistyped field, so a float
    here means the duration silently goes missing on every strict client.
    """
    payload = json.loads(
        Progress(track_progress=1000, track_duration=217000.0, playback_speed=1000).to_json()
    )
    assert payload["track_duration"] == 217000
    assert type(payload["track_duration"]) is int


def test_metadata_role_path_emits_integers() -> None:
    """The whole metadata update, including the nested Progress, is integer-typed."""
    update = SessionUpdateMetadata(
        timestamp=1.5e6,
        year=2020.0,
        track=3.0,
        progress=Progress(track_progress=1000.0, track_duration=217000.0, playback_speed=1000.0),
    )
    payload = json.loads(update.to_json())

    assert payload["timestamp"] == 1_500_000
    assert payload["year"] == 2020
    assert payload["track"] == 3
    assert payload["progress"] == {
        "track_progress": 1000,
        "track_duration": 217000,
        "playback_speed": 1000,
    }
    assert float not in _json_types(payload)


def test_optional_union_members_still_work() -> None:
    """Coercion must not disturb optional int fields: None is omitted, a value is coerced."""
    assert "year" not in json.loads(SessionUpdateMetadata(timestamp=0, year=None).to_json())
    payload = json.loads(SessionUpdateMetadata(timestamp=0, year=2020.0).to_json())  # type: ignore[arg-type]
    assert payload["year"] == 2020
    assert type(payload["year"]) is int


def test_plain_int_fields_are_coerced() -> None:
    """Non-union int fields are covered too, not just the metadata ones."""
    payload = json.loads(
        StreamStartPlayer(
            codec=AudioCodec.PCM,
            codec_header=None,
            sample_rate=48000.0,
            channels=2.0,
            bit_depth=16.0,
        ).to_json()
    )
    assert payload["sample_rate"] == 48000
    assert float not in _json_types(payload)


# Structural guard against new models regressing


def _json_model_classes() -> list[type]:
    """Every model dataclass that is serialized to JSON."""
    found: dict[str, type] = {}
    modules = [noise_models]
    modules.extend(
        importlib.import_module(f"aiosendspin.models.{module_info.name}")
        for module_info in pkgutil.iter_modules(models_package.__path__)
    )
    for module in modules:
        for obj in vars(module).values():
            if (
                inspect.isclass(obj)
                and dataclasses.is_dataclass(obj)
                and issubclass(obj, DataClassORJSONMixin)
                and obj.__module__.startswith(("aiosendspin.models", "aiosendspin.noise.models"))
            ):
                found[f"{obj.__module__}.{obj.__qualname__}"] = obj
    return list(found.values())


def test_every_json_model_uses_sendspin_config() -> None:
    """Every JSON model must resolve to a SendspinConfig.

    mashumaro resolves `Config` by lookup, so a model declaring `Config(BaseConfig)`
    replaces the inherited config outright and silently loses the integer coercion.
    """
    classes = _json_model_classes()
    assert classes, "no model classes discovered; the discovery helper is broken"

    offenders = [
        f"{cls.__module__}.{cls.__qualname__}"
        for cls in classes
        if not issubclass(getattr(cls, "Config", object), SendspinConfig)
    ]
    assert not offenders, (
        "these models do not use SendspinConfig, so their int fields are not coerced: "
        f"{offenders}. Derive their Config from SendspinConfig rather than BaseConfig."
    )
