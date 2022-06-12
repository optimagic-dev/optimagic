import itertools

import numpy as np
import pytest
from estimagic.optimization.optimize import minimize
from estimagic.visualization.history_plots import criterion_plot
from estimagic.visualization.history_plots import params_plot


@pytest.fixture()
def minimize_result():

    out = {}
    for multistart in [True, False]:
        res = []
        for algorithm in ["scipy_neldermead", "scipy_lbfgsb"]:
            _res = minimize(
                criterion=lambda x: x @ x,
                params=np.arange(5),
                algorithm=algorithm,
                soft_lower_bounds=np.full(5, -1),
                soft_upper_bounds=np.full(5, 6),
                multistart_options={
                    "n_samples": 1000,
                    "convergence.max_discoveries": 5,
                },
            )
            res.append(_res)
        out[multistart] = res
    return out


TEST_CASES = list(itertools.product([True, False], repeat=4))


@pytest.mark.parametrize(
    "multistart, monotone, stack_multistart, exploration", TEST_CASES
)
def test_criterion_plot_list_input(
    minimize_result, multistart, monotone, stack_multistart, exploration
):

    res = minimize_result[multistart]

    criterion_plot(
        res,
        monotone=monotone,
        stack_multistart=stack_multistart,
        show_exploration=exploration,
    )


@pytest.mark.parametrize("multistart", [True, False])
def test_params_plot_list_input(minimize_result, multistart):
    res = minimize_result[multistart]
    for _res in res:
        params_plot(_res)


def test_criterion_plot_different_input_types():

    # logged result
    minimize(
        criterion=lambda x: x @ x,
        params=np.arange(5),
        algorithm="scipy_lbfgsb",
        soft_lower_bounds=np.full(5, -1),
        soft_upper_bounds=np.full(5, 6),
        multistart_options={"n_samples": 1000, "convergence.max_discoveries": 5},
        log_options={"fast_logging": True},
        logging="test.db",
    )

    res = minimize(
        criterion=lambda x: x @ x,
        params=np.arange(5),
        algorithm="scipy_lbfgsb",
        soft_lower_bounds=np.full(5, -1),
        soft_upper_bounds=np.full(5, 6),
        multistart_options={"n_samples": 1000, "convergence.max_discoveries": 5},
    )

    results = ["test.db", res]

    criterion_plot(results)
    criterion_plot(results, monotone=True)
    criterion_plot(results, stack_multistart=True)
    criterion_plot(results, monotone=True, stack_multistart=True)
    criterion_plot(results, show_exploration=True)


def test_criterion_plot_wrong_inputs():

    with pytest.raises(ValueError):
        criterion_plot("bla", names=[1, 2])

    with pytest.raises(ValueError):
        criterion_plot(["bla", "bla"], names="blub")
