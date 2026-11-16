"""The models package surfaces every model submodule."""

from __future__ import annotations

from aiosendspin import models


def test_all_model_submodules_are_exported() -> None:
    """The color and visualizer_draft_r1 modules are exported like their siblings."""
    for name in ("color", "visualizer_draft_r1"):
        assert name in models.__all__
        assert hasattr(models, name)
