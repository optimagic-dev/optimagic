import itertools

import numpy as np
import pytest
from estimagic.optimization.optimize import minimize
from estimagic.visualization.history_plots import criterion_plot
from estimagic.visualization.history_plots import params_plot


def _minimize(kwargs):
    _res = minimize(
        criterion=lambda x: x @ x,
        params=np.arange(5),
        soft_lower_bounds=np.full(5, -1),
        soft_upper_bounds=np.full(5, 6),
        multistart_options={"n_samples": 1000, "convergence.max_discoveries": 5},
        log_options={"fast_logging": True},
        **kwargs,
    )
    return _res


@pytest.fixture()
def minimize_result():

    out = {}
    for multistart in [True, False]:
        res = []
        for algorithm in ["scipy_neldermead", "scipy_lbfgsb"]:
            _res = _minimize({"algorithm": algorithm, "multistart": multistart})
            res.append(_res)
        out[multistart] = res

    return out


TEST_CASES = list(
    itertools.product(
        [True, False],  # multistart
        [True, False],  # monotone
        [True, False],  # stack_multistart
        [True, False],  # exploration
    )
)


@pytest.mark.parametrize(
    "multistart, monotone, stack_multistart, exploration", TEST_CASES
)
def test_criterion_plot_list_input(
    minimize_result, multistart, monotone, stack_multistart, exploration
):

    res = minimize_result[multistart]

    if stack_multistart and monotone:
        with pytest.raises(ValueError):
            criterion_plot(
                res,
                monotone=monotone,
                stack_multistart=stack_multistart,
                exploration=exploration,
            )
    else:
        criterion_plot(
            res,
            monotone=monotone,
            stack_multistart=stack_multistart,
            exploration=exploration,
        )

    for _res in res:
        params_plot(_res)


def test_criterion_plot_different_input_types():

    _minimize({"algorithm": "scipy_lbfgsb", "logging": "test.db"}),
    res = ["test.db", _minimize({"algorithm": "scipy_neldermead"})]

    criterion_plot(res)
