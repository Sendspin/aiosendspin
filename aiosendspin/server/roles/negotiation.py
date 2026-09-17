"""Server-side role negotiation helpers."""

from __future__ import annotations

from collections.abc import Iterable

from aiosendspin.models.types import role_family

from .registry import ROLE_FACTORIES

# Server-defined role family activation order. Families listed here are
# connected first (in the order shown); any unlisted families follow in
# client-provided order.
_FAMILY_ORDER = {
    family: i
    for i, family in enumerate(
        [
            # Player must come before controller so that PlayerGroupRole
            # already contains the player when ControllerGroupRole
            # reads group volume during on_member_join().
            "player",
            "controller",
        ]
    )
}

# DEPRECATED(spec-pr-86): remove in aiosendspin <version>
# Legacy backwards-compat wires. They are excluded from negotiation in strict mode,
# and in every mode lose to another registered role of the same family.
# `visualizer@_draft_r1` reuses binary type 16 with a different payload.
_LEGACY_ROLE_IDS = frozenset({"visualizer@_draft_r1"})


def sort_role_ids(role_ids: Iterable[str]) -> list[str]:
    """Order role IDs by server-defined family order; unlisted families keep input order."""
    return sorted(role_ids, key=lambda rid: _FAMILY_ORDER.get(role_family(rid), len(_FAMILY_ORDER)))


def negotiate_roles(client_supported_roles: list[str], *, strict: bool = False) -> list[str]:
    """Negotiate the mutually-supported role set from the client-supported role list.

    For each role family, pick the first role in client order that is registered
    in ROLE_FACTORIES. Legacy backwards-compat wires are skipped whenever the client
    also lists another registered role of the same family, whatever the client's
    order, and always when ``strict``. The result is sorted by the server-defined
    family activation order.
    """
    active: dict[str, str] = {}
    # DEPRECATED(spec-pr-86): remove in aiosendspin <version>
    spec_families = {
        role_family(role_id)
        for role_id in client_supported_roles
        if role_id in ROLE_FACTORIES and role_id not in _LEGACY_ROLE_IDS
    }

    for client_role_id in client_supported_roles:
        if client_role_id in _LEGACY_ROLE_IDS and (
            strict or role_family(client_role_id) in spec_families
        ):
            continue
        family = role_family(client_role_id)
        if family in active:
            continue

        if client_role_id in ROLE_FACTORIES:
            active[family] = client_role_id

    return sort_role_ids(active.values())
