"""Tests for aiosendspin.util."""

from __future__ import annotations

import asyncio
import logging

from aiosendspin.util import create_task


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
