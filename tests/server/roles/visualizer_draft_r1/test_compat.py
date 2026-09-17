"""Backwards-compat invariants for `VisualizerDraftR1Role`.

The shared `VisualizerGroupRole` (registered for the v1 package) filters
members by `VisualizerRoleProtocol` (runtime_checkable Protocol →
attribute presence). draft_r1 must NOT expose those attributes, or it
would accidentally start receiving beats it can't render.
"""

# DEPRECATED(spec-pr-86): remove in aiosendspin <version>

from __future__ import annotations

import pytest

from aiosendspin.server.roles.visualizer.types import VisualizerRoleProtocol
from aiosendspin.server.roles.visualizer_draft_r1.role import VisualizerDraftR1Role

_BEAT_PROTOCOL_ATTRS = (
    "wants_beats",
    "append_beats",
    "clear_beats",
    "set_beat_availability",
)


@pytest.mark.parametrize("attr", _BEAT_PROTOCOL_ATTRS)
def test_draft_r1_role_does_not_expose_beat_attr(attr: str) -> None:
    """draft_r1 must not advertise beat-protocol attributes."""
    assert not hasattr(VisualizerDraftR1Role, attr), (
        f"VisualizerDraftR1Role must not expose `{attr}`; otherwise the shared "
        f"VisualizerGroupRole would treat it as beat-capable and replay beats "
        f"onto a wire that cannot carry them."
    )


def test_draft_r1_role_does_not_satisfy_visualizer_role_protocol() -> None:
    """A draft_r1 instance must not pass the runtime Protocol isinstance check."""

    class _FakeClient:
        client_id = "x"

    role = VisualizerDraftR1Role(client=_FakeClient())  # type: ignore[arg-type]
    assert not isinstance(role, VisualizerRoleProtocol)
