"""Validators for `stream/clear` and `stream/end` `roles` lists.

Spec README:394,415 explicitly permit `_`-prefixed application role names.
"""

from __future__ import annotations

import pytest

from aiosendspin.models.core import StreamClearPayload, StreamEndPayload


def test_stream_clear_accepts_underscore_prefixed_role() -> None:
    """`_`-prefixed names are reserved for application-specific roles."""
    payload = StreamClearPayload(roles=["_custom_role"])
    assert payload.roles == ["_custom_role"]


def test_stream_clear_accepts_mixed_known_and_app_roles() -> None:
    """Known families and `_`-prefixed roles may appear together."""
    payload = StreamClearPayload(roles=["player", "_my_app"])
    assert payload.roles == ["player", "_my_app"]


def test_stream_clear_rejects_unknown_non_underscore_role() -> None:
    """Unknown families without `_` prefix are still rejected."""
    with pytest.raises(ValueError, match="invalid roles"):
        StreamClearPayload(roles=["controller"])


def test_stream_end_accepts_underscore_prefixed_role() -> None:
    """`_`-prefixed names are reserved for application-specific roles."""
    payload = StreamEndPayload(roles=["_custom_role"])
    assert payload.roles == ["_custom_role"]


def test_stream_end_accepts_mixed_known_and_app_roles() -> None:
    """Known families and `_`-prefixed roles may appear together."""
    payload = StreamEndPayload(roles=["player", "_my_app"])
    assert payload.roles == ["player", "_my_app"]


def test_stream_end_rejects_unknown_non_underscore_role() -> None:
    """Unknown families without `_` prefix are still rejected."""
    with pytest.raises(ValueError, match="invalid roles"):
        StreamEndPayload(roles=["controller"])


def test_stream_clear_from_timestamp_is_omitted_unless_set() -> None:
    """from_timestamp only appears on the wire for a timestamped clear."""
    assert "from_timestamp" not in StreamClearPayload(roles=["player"]).to_dict()
    payload = StreamClearPayload(roles=["player"], from_timestamp=1_500_000)
    assert payload.to_dict()["from_timestamp"] == 1_500_000
