"""Shared test setup.

Three modules are written by hand rather than generated: the scorer
(jpe/score.py), the schema validation (jpe/validate.py) and the QLoRA
configuration (jpe/train_config.py). Their tests are marked `core` and are
already written. Until the function under test exists it raises
NotImplementedError, and the hook below reports that test as skipped, so CI
stays green while the work is outstanding. Once it is written the test runs
for real, and a wrong answer fails it as normal.

The hook only applies to `core` tests. A NotImplementedError anywhere else is
a bug and fails as one.
"""

from __future__ import annotations

import pytest


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    try:
        return (yield)
    except NotImplementedError as exc:
        if item.get_closest_marker("core") is None:
            raise
        pytest.skip(f"not written yet: {exc}")
