import itertools

import numpy as np
import pytest
from estimagic.optimization.optimize import minimize
from estimagic.visualization.history_plots import criterion_plot
from estimagic.visualization.history_plots import params_plot


CASES = list(itertools.product([True, False], repeat=2))


@pytest.mark.parametrize("multistart, monotone", CASES)
def test_history_plots_run(multistart, monotone):
    res = minimize(
        criterion=lambda x: x @ x,
        params=np.arange(5),
        algorithm="scipy_neldermead",
        multistart=multistart,
        soft_lower_bounds=np.full(5, -1),
        soft_upper_bounds=np.full(5, 6),
    )

    criterion_plot(res, monotone=monotone)
    params_plot(res)
