import numpy as np
import pytest
from estimagic.optimization.optimize import minimize
from estimagic.visualization.history_plots import criterion_plot
from estimagic.visualization.history_plots import params_plot


@pytest.mark.parametrize("multistart", [True, False])
def test_history_plots_run(multistart):
    res = minimize(
        criterion=lambda x: x @ x,
        params=np.arange(5),
        algorithm="scipy_neldermead",
        multistart=multistart,
        soft_lower_bounds=np.full(5, -1),
        soft_upper_bounds=np.full(5, 6),
    )

    criterion_plot(res)
    params_plot(res)
