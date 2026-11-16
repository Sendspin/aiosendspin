"""Tests for aiosendspin.util."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from aiosendspin.util import PrefixLoggerAdapter, create_task

if TYPE_CHECKING:
    import pytest


async def test_create_task_logs_fire_and_forget_exception(
    caplog: object,
) -> None:
    """A failing fire-and-forget task logs its exception instead of swallowing it."""

    async def boom() -> None:
        raise ValueError("boom")

    with caplog.at_level(logging.ERROR, logger="aiosendspin.util"):  # type: ignore[attr-defined]
        task = create_task(boom(), eager_start=False)
        await asyncio.sleep(0)  # run the task
        await asyncio.sleep(0)  # let the done callback fire

    assert task.done()
    logged = [r for r in caplog.records if r.name == "aiosendspin.util"]  # type: ignore[attr-defined]
    assert logged, "expected the failing task to be logged"
    assert logged[0].exc_info is not None


def test_prefix_logger_adapter_keeps_percent_in_prefix_literal(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A client id with ``%`` must not break formatting or hide the real call site."""
    adapter = PrefixLoggerAdapter(logging.getLogger(__name__), "id%s")
    with caplog.at_level(logging.INFO, logger=__name__):
        adapter.info("joined %s", "group")
        adapter.info("left 100%")

    assert caplog.messages == ["[id%s] joined group", "[id%s] left 100%"]
    assert {r.funcName for r in caplog.records} == {
        "test_prefix_logger_adapter_keeps_percent_in_prefix_literal"
    }
