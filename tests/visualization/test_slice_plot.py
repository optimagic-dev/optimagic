import itertools

import numpy as np
import pandas as pd
import pytest
from estimagic.visualization.plotting_utilities import combine_plots
from estimagic.visualization.slice_plot import slice_plots


@pytest.fixture
def problem():
    params = pd.DataFrame(
        np.zeros(4),
        columns=["value"],
    )

    params["lower_bound"] = -5
    params["upper_bound"] = 5

    out = {
        "criterion": lambda params: (params["value"] ** 2).sum(),  # sphere
        "params": params,
    }
    return out


@pytest.mark.parametrize(
    "n_gridpoints, n_random_values, plots_per_row, combine_plots_in_grid",
    itertools.product([21, 41], [2, 3], [1, 2], [True, False]),
)
def test_slice_plot(
    problem, n_gridpoints, n_random_values, plots_per_row, combine_plots_in_grid
):

    plots_dict = slice_plots(
        criterion=problem["criterion"],
        params=problem["params"],
        n_gridpoints=n_gridpoints,
        n_random_values=n_random_values,
    )
    if combine_plots_in_grid:
        combine_plots(
            plots=list(plots_dict.values()),
            plots_per_row=plots_per_row,
        )
