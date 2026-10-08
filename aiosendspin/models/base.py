"""Shared serialization behavior for Sendspin protocol models.

The protocol types `int`-annotated wire fields as integers, but Python does not enforce
annotations at runtime. This module keeps those fields integer-typed during serialization,
and checks the JSON type of integer, boolean, string and array fields during parsing.
It also provides the parse helpers that set aside unrecognized enum identifiers, and the
hooks that carry application-specific role objects between the wire and the models.
"""

from __future__ import annotations

import dataclasses
import math
import operator
import re
import types
from collections.abc import Callable
from contextvars import ContextVar
from enum import Enum
from functools import cache
from typing import Annotated, Any, Union, get_args, get_origin, get_type_hints

from mashumaro.config import BaseConfig
from mashumaro.exceptions import InvalidFieldValue
from mashumaro.mixins.orjson import DataClassORJSONMixin
from mashumaro.types import Alias


def int_to_wire(value: Any) -> int:
    """Coerce numeric values to wire integers.

    Indexable values preserve their integer value. Finite floats are rounded so arithmetic
    artifacts do not lose a unit, while booleans are rejected rather than becoming plausible
    1 or 0 values on the wire.

    Raises:
        TypeError: If the value is a boolean or is not numeric.
        ValueError: If the value is not finite.
    """
    if isinstance(value, bool):
        msg = f"expected an integer, got bool: {value!r}"
        raise TypeError(msg)

    try:
        return operator.index(value)
    except TypeError:
        pass

    if isinstance(value, float) or hasattr(value, "__float__"):
        as_float = float(value)
        if not math.isfinite(as_float):
            msg = f"cannot serialize non-finite value {value!r} as an integer"
            raise ValueError(msg)
        return round(as_float)

    msg = f"expected an integer, got {type(value).__name__}: {value!r}"
    raise TypeError(msg)


def split_enum_values(values: Any, enum_type: type[Enum]) -> tuple[Any, list[str]]:
    """Split a raw wire list into its entries and the identifiers ``enum_type`` lacks.

    Returns the list without unrecognized string entries, and those entries. Tolerance
    covers identifiers, not shape: a non-list value or a non-string entry is kept for the
    parse to reject.
    """
    if not isinstance(values, list):
        return values, []
    known = {member.value for member in enum_type}
    ignored = [v for v in values if isinstance(v, str) and v not in known]
    return [v for v in values if not isinstance(v, str) or v in known], ignored


def is_unknown_enum_value(value: Any, enum_type: type[Enum]) -> bool:
    """Return whether ``value`` is an identifier that names no member of ``enum_type``."""
    return isinstance(value, str) and value not in {member.value for member in enum_type}


APPLICATION_OBJECTS_FIELD = "application_objects"


def collect_application_objects(d: dict[str, Any]) -> dict[str, Any]:
    """Return ``d`` with its `_`-prefixed application-specific role objects nested.

    The objects move under ``application_objects``, which is always overwritten so the
    field cannot be set from the wire.
    """
    normalized = {k: v for k, v in d.items() if not k.startswith("_")}
    normalized[APPLICATION_OBJECTS_FIELD] = {k: v for k, v in d.items() if k.startswith("_")}
    return normalized


def expand_application_objects(d: dict[str, Any]) -> dict[str, Any]:
    """Return ``d`` with ``application_objects`` lifted back to top-level payload keys.

    Raises:
        ValueError: If an application object key does not start with `_`.
    """
    objects = d.pop(APPLICATION_OBJECTS_FIELD, None) or {}
    if invalid := sorted(key for key in objects if not key.startswith("_")):
        msg = f"application object keys must start with '_', got {invalid}"
        raise ValueError(msg)
    d.update(objects)
    return d


_JSON_TYPE_NAMES: dict[type, str] = {
    bool: "a boolean",
    int: "an integer",
    float: "a number",
    str: "a string",
    list: "an array",
    dict: "an object",
    type(None): "null",
}
_INT64_MIN = -(2**63)
_INT64_LIMIT = 2**63
_DECIMAL_INTEGER = re.compile(r"-?[0-9]+")
# Each located mismatch costs a reparse, so later ones are named only by their kind,
# or dropped when a located one had the same kind.
_MAX_LOCATED_MISMATCHES = 8


class _WireTypeMismatchError(Exception):
    """Raised by a parse hook while locating the field of a recorded type mismatch."""

    def __init__(self, got: str, expected: str) -> None:
        super().__init__(got, expected)
        self.got = got
        self.expected = expected


class _LocateMismatch:
    """Collector state that makes parse hooks raise at the type mismatch after ``skip`` others."""

    def __init__(self, skip: int) -> None:
        self.skip = skip


_wire_deviations: ContextVar[list[str | tuple[str, str]] | _LocateMismatch | None] = ContextVar(
    "wire_deviations", default=None
)


def _json_type(value: Any) -> str:
    return _JSON_TYPE_NAMES.get(type(value), type(value).__name__)


def _mismatch_reason(got: str, expected: str, path: str | None = None) -> str:
    where = f" for '{path}'" if path else ""
    return f"sent {got}{where} instead of {expected}"


def note_wire_deviation(reason: str) -> None:
    """Record a tolerated deviation for the enclosing ``parse_noting_wire_deviations``."""
    collector = _wire_deviations.get()
    if isinstance(collector, list):
        collector.append(reason)


def _note_type_mismatch(value: Any, expected: str) -> None:
    collector = _wire_deviations.get()
    if collector is None:
        return
    if isinstance(collector, _LocateMismatch):
        if collector.skip == 0:
            raise _WireTypeMismatchError(_json_type(value), expected)
        collector.skip -= 1
        return
    collector.append((_json_type(value), expected))


@cache
def _wire_field_name(holder_class: type, field_name: str) -> str:
    """Return the wire key of a model field, honoring an ``Alias`` annotation."""
    hint = get_type_hints(holder_class, include_extras=True).get(field_name)
    for meta in getattr(hint, "__metadata__", ()):
        if isinstance(meta, Alias):
            return meta.name
    return field_name


def _mismatch_path(exc: BaseException) -> tuple[_WireTypeMismatchError, str] | None:
    """Return the mismatch that ended a locating parse and the dotted path of its field."""
    fields: list[str] = []
    cause: BaseException | None = exc
    while isinstance(cause, InvalidFieldValue):
        fields.append(_wire_field_name(cause.holder_class, cause.field_name))
        cause = cause.__context__
    if not isinstance(cause, _WireTypeMismatchError):
        return None
    # Message envelopes wrap everything in ``payload``, which names nothing to a reader.
    if fields[:1] == ["payload"]:
        fields = fields[1:]
    return cause, ".".join(fields)


def parse_noting_wire_deviations[**P, T](
    parse: Callable[P, T], *args: P.args, **kwargs: P.kwargs
) -> tuple[T, list[str]]:
    """Run ``parse`` and return its result with each tolerated deviation it recorded.

    Each of the first recorded type mismatches is named by its field, found by parsing again.
    """
    collector: list[str | tuple[str, str]] = []
    token = _wire_deviations.set(collector)
    try:
        result = parse(*args, **kwargs)
    finally:
        _wire_deviations.reset(token)
    reasons: list[str] = []
    mismatches_before = 0
    located_kinds: set[tuple[str, str]] = set()
    for entry in collector:
        if isinstance(entry, str):
            reasons.append(entry)
            continue
        reason = _mismatch_reason(*entry)
        if mismatches_before < _MAX_LOCATED_MISMATCHES:
            token = _wire_deviations.set(_LocateMismatch(skip=mismatches_before))
            try:
                parse(*args, **kwargs)
            except InvalidFieldValue as exc:
                if located := _mismatch_path(exc):
                    mismatch, path = located
                    reason = _mismatch_reason(mismatch.got, mismatch.expected, path)
                    located_kinds.add(entry)
            finally:
                _wire_deviations.reset(token)
        elif entry in located_kinds:
            continue
        mismatches_before += 1
        reasons.append(reason)
    return result, list(dict.fromkeys(reasons))


def int_from_wire(value: Any) -> int:
    """Parse an integer field, recording and converting a number, boolean or decimal string.

    A float with no fractional part is a JSON integer and passes unrecorded.

    Raises:
        TypeError: If the value has no integer reading.
        ValueError: If a float or decimal string lies outside the signed 64-bit range.
    """
    if type(value) is int:
        return value
    if isinstance(value, bool):
        _note_type_mismatch(value, "an integer")
        return int(value)
    if isinstance(value, float) and math.isfinite(value):
        parsed = int(value)
        tolerated = not value.is_integer()
    elif isinstance(value, str) and _DECIMAL_INTEGER.fullmatch(value):
        parsed = int(value)
        tolerated = True
    else:
        msg = f"expected an integer, got {_json_type(value)}: {value!r}"
        raise TypeError(msg)
    if not _INT64_MIN <= parsed < _INT64_LIMIT:
        msg = f"integer out of the signed 64-bit range: {value!r}"
        raise ValueError(msg)
    if tolerated:
        _note_type_mismatch(value, "an integer")
    return parsed


def bool_from_wire(value: Any) -> bool:
    """Parse a boolean field, recording and converting ``"true"``/``"false"`` and ``1``/``0``.

    Raises:
        TypeError: If the value has no boolean reading.
    """
    if type(value) is bool:
        return value
    if value in ("true", "false") or (type(value) is int and value in (0, 1)):
        _note_type_mismatch(value, "a boolean")
        return value in ("true", 1)
    msg = f"expected a boolean, got {_json_type(value)}: {value!r}"
    raise TypeError(msg)


def str_from_wire(value: Any) -> str:
    """Parse a string field, recording and converting a number.

    Raises:
        TypeError: If the value is not a string or a number.
    """
    if type(value) is str:
        return value
    if isinstance(value, str):
        return str(value)
    if type(value) in (int, float):
        _note_type_mismatch(value, "a string")
        return str(value)
    msg = f"expected a string, got {_json_type(value)}: {value!r}"
    raise TypeError(msg)


@cache
def _array_fields(model: type) -> tuple[tuple[str, str, Any], ...]:
    """Return the name, wire key and type of each list and tuple field of a dataclass model."""
    hints = get_type_hints(model, include_extras=True)
    fields = []
    for field in dataclasses.fields(model):
        hint = hints[field.name]
        inner = get_args(hint)[0] if get_origin(hint) is Annotated else hint
        members = get_args(inner) if get_origin(inner) in (Union, types.UnionType) else (inner,)
        if any(get_origin(member) in (list, tuple) for member in members):
            fields.append((field.name, _wire_field_name(model, field.name), hint))
    return tuple(fields)


class SendspinConfig(BaseConfig):
    """Base mashumaro config for Sendspin models.

    Model configs must derive from this class to retain integer coercion and the wire type
    checks.
    """

    serialization_strategy = {  # noqa: RUF012
        int: {"serialize": int_to_wire, "deserialize": int_from_wire},
        bool: {"deserialize": bool_from_wire},
        str: {"deserialize": str_from_wire},
    }


class SendspinModel(DataClassORJSONMixin):
    """Base class for Sendspin protocol models. Applies `SendspinConfig` by default."""

    @classmethod
    def __pre_deserialize__(cls, d: dict[str, Any]) -> dict[str, Any]:
        """Reject a non-array value for a list or tuple field, which mashumaro would iterate.

        Subclass overrides return their result through this hook.

        Raises:
            InvalidFieldValue: If a list or tuple field holds a value other than an array or null.
        """
        for name, key, hint in _array_fields(cls):  # type: ignore[arg-type]
            value = d.get(key)
            if value is not None and not isinstance(value, list):
                raise InvalidFieldValue(name, hint, value, cls, msg="expected an array")
        return d

    class Config(SendspinConfig):
        """Config for parsing json messages."""
