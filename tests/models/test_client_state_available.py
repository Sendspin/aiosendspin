"""Client availability: the `available` boolean and legacy `state` wire compat.

New clients send the spec's `available` boolean. Legacy clients send a `state`
enum, which the server normalizes to `available` at deserialization time. The
payload itself has no `state` field.
"""

# DEPRECATED(spec-pr-115): remove in aiosendspin <version>

from __future__ import annotations

import pytest

from aiosendspin.models.core import ClientStatePayload


@pytest.mark.parametrize(
    ("state", "expected_available"),
    [
        ("synchronized", True),
        ("external_source", False),
        ("error", True),
    ],
)
def test_legacy_state_on_wire_normalizes_to_available(
    state: str,
    expected_available: bool,  # noqa: FBT001
) -> None:
    """A legacy client's `state` enum is translated to `available` on deserialization."""
    payload = ClientStatePayload.from_dict({"state": state})
    assert payload.available is expected_available


def test_new_available_wins_over_legacy_state_on_wire() -> None:
    """When both are present on the wire, `available` is authoritative."""
    payload = ClientStatePayload.from_dict({"available": True, "state": "external_source"})
    assert payload.available is True


def test_legacy_state_presence_is_recorded() -> None:
    """A top-level `state` field is recorded even when `available` is also present."""
    payload = ClientStatePayload.from_dict({"available": True, "state": "synchronized"})
    assert payload.legacy_state_used is True


def test_compliant_state_records_no_legacy_use() -> None:
    """A client/state without `state` records nothing, even if it spoofs the field."""
    payload = ClientStatePayload.from_dict({"available": True, "legacy_state_used": True})
    assert payload.legacy_state_used is None
