"""Tests for role negotiation: custom roles and server-defined ordering."""

from __future__ import annotations

from typing import Any

import pytest

from aiosendspin.server.roles.negotiation import negotiate_roles, sort_role_ids
from aiosendspin.server.roles.registry import ROLE_FACTORIES


def _factory(_client: Any) -> object:
    return object()


# --- Canonical family ordering ---


def test_sort_role_ids_orders_listed_families() -> None:
    """Listed families sort by server-defined order regardless of input order."""
    assert sort_role_ids(["controller@v1", "player@v1"]) == ["player@v1", "controller@v1"]


def test_sort_role_ids_stable_for_unlisted_families() -> None:
    """Unlisted families keep their input order and follow listed families."""
    assert sort_role_ids(["metadata@v1", "artwork@v1", "player@v1"]) == [
        "player@v1",
        "metadata@v1",
        "artwork@v1",
    ]


# --- Server-defined activation order ---


def test_negotiate_orders_player_before_controller() -> None:
    """Player must be activated before controller regardless of client order."""
    active = negotiate_roles(["controller@v1", "player@v1"])
    assert active == ["player@v1", "controller@v1"]


# DEPRECATED(spec-pr-86): remove in aiosendspin <version>
def test_negotiate_activates_draft_visualizer_by_default() -> None:
    """The legacy visualizer@_draft_r1 wire is negotiated in the default (lenient) mode."""
    assert negotiate_roles(["visualizer@_draft_r1"]) == ["visualizer@_draft_r1"]


# DEPRECATED(spec-pr-86): remove in aiosendspin <version>
def test_strict_excludes_draft_visualizer() -> None:
    """Under strict, the legacy draft wire is skipped and not activated."""
    assert negotiate_roles(["visualizer@_draft_r1"], strict=True) == []


# DEPRECATED(spec-pr-86): remove in aiosendspin <version>
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize(
    "roles",
    [
        ["visualizer@_draft_r1", "visualizer@v1"],
        ["visualizer@v1", "visualizer@_draft_r1"],
    ],
)
def test_v1_visualizer_wins_over_draft(roles: list[str], *, strict: bool) -> None:
    """A client offering both draft and v1 gets v1, whatever its order or the mode."""
    assert negotiate_roles(roles, strict=strict) == ["visualizer@v1"]


def test_negotiate_orders_player_before_controller_with_metadata() -> None:
    """Player before controller even when other families are present."""
    active = negotiate_roles(["metadata@v1", "controller@v1", "player@v1"])
    assert active == ["player@v1", "controller@v1", "metadata@v1"]


def test_negotiate_preserves_order_when_client_matches_server() -> None:
    """No reordering needed when client already sends player first."""
    active = negotiate_roles(["player@v1", "controller@v1"])
    assert active == ["player@v1", "controller@v1"]


def test_negotiate_unlisted_families_come_after_listed(monkeypatch: Any) -> None:
    """Families not in _FAMILY_ORDER sort after listed families."""
    monkeypatch.setitem(ROLE_FACTORIES, "customfoo@v1", _factory)
    active = negotiate_roles(["customfoo@v1", "controller@v1", "player@v1"])
    assert active[:2] == ["player@v1", "controller@v1"]
    assert "customfoo@v1" in active


# --- Custom role selection ---


def test_negotiate_accepts_registered_custom_role_in_known_family(
    monkeypatch: Any,
) -> None:
    """Registered custom role IDs should be negotiable without built-in map edits."""
    monkeypatch.setitem(ROLE_FACTORIES, "player@_custom_player_version", _factory)

    active_roles = negotiate_roles(["player@_custom_player_version"])

    assert active_roles == ["player@_custom_player_version"]


def test_negotiate_accepts_registered_custom_role_in_unknown_family(
    monkeypatch: Any,
) -> None:
    """Registered custom roles for unknown families should also be negotiable."""
    monkeypatch.setitem(ROLE_FACTORIES, "customaudio@v1", _factory)

    active_roles = negotiate_roles(["customaudio@v1"])

    assert active_roles == ["customaudio@v1"]


def test_negotiate_rejects_unregistered_custom_role_id() -> None:
    """Unregistered custom role IDs should remain non-negotiable."""
    active_roles = negotiate_roles(["player@_not_registered"])

    assert active_roles == []


def test_negotiate_prefers_first_client_role_when_custom_first(
    monkeypatch: Any,
) -> None:
    """Client order should determine same-family selection when custom role appears first."""
    monkeypatch.setitem(ROLE_FACTORIES, "player@_custom_player_version", _factory)

    active_roles = negotiate_roles(["player@_custom_player_version", "player@v1"])

    assert active_roles == ["player@_custom_player_version"]


def test_negotiate_prefers_first_client_role_when_standard_first(
    monkeypatch: Any,
) -> None:
    """Client order should determine same-family selection when standard role appears first."""
    monkeypatch.setitem(ROLE_FACTORIES, "player@_custom_player_version", _factory)
    monkeypatch.setitem(ROLE_FACTORIES, "player@v1", _factory)

    active_roles = negotiate_roles(["player@v1", "player@_custom_player_version"])

    assert active_roles == ["player@v1"]
