"""Visualizer draft_r1 role.

Deprecated backwards-compat with clients on the `visualizer@_draft_r1` wire. The
shared `VisualizerGroupRole` (registered by the `visualizer@v1` package)
handles both wire versions across one role family. This package only
registers the role implementation.
"""

from aiosendspin.server.roles.registry import register_role
from aiosendspin.server.roles.visualizer_draft_r1.role import VisualizerDraftR1Role

# DEPRECATED(spec-pr-86): remove in aiosendspin <version>
register_role("visualizer@_draft_r1", lambda client: VisualizerDraftR1Role(client=client))

__all__ = ["VisualizerDraftR1Role"]
