"""Test that deprecated functionality keeps working and warns.

The tests are grouped by the version in which the functionality was deprecated and the
version in which it will be removed. When a deprecation expires, its tests are deleted
together with its code in ``optimagic.deprecations``.

"""

import pytest

from optimagic.optimization.history import History
from optimagic.typing import Direction, EvalTask

# ======================================================================================
# Deprecated in 0.6.0, remove in 0.7.0
# ======================================================================================


@pytest.fixture
def history():
    return History(
        direction=Direction.MINIMIZE,
        params=[1.0, 2.0, 3.0],
        fun=[3.0, 2.0, 1.0],
        start_time=[3.0, 5.0, 8.0],
        stop_time=[4.0, 6.0, 9.0],
        batches=[0, 1, 2],
        task=[EvalTask.FUN, EvalTask.FUN, EvalTask.FUN],
    )


@pytest.mark.parametrize("attribute", ["time", "runtime"])
def test_history_time_attributes_are_deprecated(history, attribute):
    with pytest.warns(FutureWarning, match=f"`{attribute}` of History is deprecated"):
        got = getattr(history, attribute)
    start_time = history.start_time
    assert got == [t - start_time[0] for t in start_time]


def test_history_time_warning_points_to_the_caller(history):
    with pytest.warns(FutureWarning) as record:
        history.time  # noqa: B018
    assert record[0].filename == __file__
