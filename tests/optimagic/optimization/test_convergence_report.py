import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal as aaae

from optimagic.optimization.convergence_report import get_convergence_report
from optimagic.optimization.history import History
from optimagic.typing import Direction


def test_get_convergence_report_minimize():
    hist = History(
        direction=Direction.MINIMIZE,
        params=[{"a": 0}, {"a": 2.1}, {"a": 2.5}, {"a": 2.0}],
        fun=[5, 4.1, 4.4, 4.0],
        time=[0, 1, 2, 3],
    )

    calculated = pd.DataFrame.from_dict(get_convergence_report(hist))

    expected = np.array([[0.025, 0.25], [0.05, 1.0], [0.1, 1], [0.1, 2.0]])
    aaae(calculated.to_numpy(), expected)


def test_get_convergence_report_maximize():
    hist = History(
        direction=Direction.MAXIMIZE,
        params=[{"a": 0}, {"a": 2.1}, {"a": 2.5}, {"a": 2.0}],
        fun=[-5, -4.1, -4.4, -4.0],
        time=[0, 1, 2, 3],
    )

    calculated = pd.DataFrame.from_dict(get_convergence_report(hist))

    expected = np.array([[0.025, 0.25], [0.05, 1.0], [0.1, 1], [0.1, 2.0]])
    aaae(calculated.to_numpy(), expected)


def test_history_is_too_short():
    # first value is best, so history of accepted parameters has only one entry
    hist = History(
        direction=Direction.MAXIMIZE,
        params=[{"a": 0}, {"a": 2.1}, {"a": 2.5}, {"a": 2.0}],
        fun=[5, 4.1, 4.4, 4.0],
        time=[0, 1, 2, 3],
    )

    calculated = get_convergence_report(hist)
    assert calculated is None
