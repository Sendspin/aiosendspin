"""Tests for aiosendspin.util."""

from __future__ import annotations

import asyncio
import inspect
import logging
import warnings
from typing import TYPE_CHECKING

from aiosendspin.util import create_task, warn_deprecated

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


def test_warn_deprecated_reports_once_per_api(caplog: pytest.LogCaptureFixture) -> None:
    """Each API's first use warns and logs once; later uses are silent."""
    with (
        caplog.at_level(logging.WARNING, logger="aiosendspin.util"),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        warn_deprecated("old_api", "gone")
        warn_deprecated("old_api", "gone")
        warn_deprecated("other_api", "gone")

    expected = ["old_api is deprecated: gone", "other_api is deprecated: gone"]
    assert [str(w.message) for w in caught] == expected
    assert all(w.category is DeprecationWarning for w in caught)
    assert [r.message for r in caplog.records] == expected


def test_warn_deprecated_attributes_the_warning_to_the_api_caller() -> None:
    """The warning points at the code that called the deprecated API."""

    def deprecated_api() -> None:
        warn_deprecated("deprecated_api", "gone")

    frame = inspect.currentframe()
    assert frame is not None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call_line = frame.f_lineno + 1
        deprecated_api()

    assert (caught[0].filename, caught[0].lineno) == (__file__, call_line)
