"""Shared serialization behavior for Sendspin protocol models.

The protocol types `int`-annotated wire fields as integers, but Python does not enforce
annotations at runtime. This module keeps those fields integer-typed during serialization.
It also provides the parse helpers that set aside unrecognized enum identifiers, and the
hooks that carry application-specific role objects between the wire and the models.
"""

from __future__ import annotations

import math
import operator
from enum import Enum
from typing import Any

from mashumaro.config import BaseConfig
from mashumaro.mixins.orjson import DataClassORJSONMixin


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


class SendspinConfig(BaseConfig):
    """Base mashumaro config for Sendspin models.

    Model configs must derive from this class to retain integer coercion.
    """

    serialization_strategy = {int: {"serialize": int_to_wire}}  # noqa: RUF012


class SendspinModel(DataClassORJSONMixin):
    """Base class for Sendspin protocol models. Applies `SendspinConfig` by default."""

    class Config(SendspinConfig):
        """Config for parsing json messages."""
