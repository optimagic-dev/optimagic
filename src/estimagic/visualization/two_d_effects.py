from itertools import combinations

import numpy as np
from plotly.subplots import make_subplots


def plot_2d_effects(
    criterion,
    params,
    n_gridpoints=10,
    plots_per_row=2,
):
    """Plot the surface of the criterion on 2D grids at given and random values.

    Args:
        criterion (callable): criterion function. Takes a DataFrame and
            returns a scalar value or dictionary with the entry "value".
        params (pandas.DataFrame): See :ref:`params`. Must contain finite
            lower and upper bounds for all parameters.
        n_gridpoints (int): Number of gridpoints in each direction on which the
            criterion function is evaluated.
        plots_per_row (int): How many plots are plotted per row.

    Returns:
        Fig: plotly.graph_objects.Figure

    """
    params = params.copy()

    _assert_all_params_are_bounded(params)

    points = _create_points_to_evaluate(params, n_gridpoints)

    # parallelize this
    points["criterion_value"] = [
        criterion(points.loc[row_id, params.index].to_frame(name="value"))
        for row_id in points.index
    ]

    # create the plot
    param_combinations = combinations(params.index, 2)
    n_cols = plots_per_row
    n_rows = int(np.ceil(len(param_combinations) / n_cols))
    fig = make_subplots(rows=n_rows, cols=n_cols)

    return fig


def _assert_all_params_are_bounded(params):
    """Raise a ValueError in case any parameter is missing a lower or upper bound.

    Args: params (pandas.DataFrame): params DataFrame.

    Raises:
        ValueError: When "lower_bound" or "upper_bound" are missing for any
            parameter or in general.

    """
    if (
        "lower_bound" not in params.columns
        or not np.isfinite(params["lower_bound"]).all()
    ):
        raise ValueError("All parameters need a finite lower bound.")
    if (
        "upper_bound" not in params.columns
        or not np.isfinite(params["upper_bound"]).all()
    ):
        raise ValueError("All parameters need a finite upper bound.")


def _create_points_to_evaluate(params, n_gridpoints):
    """Create a DataFrame with all points at which to evaluate the criterion.

    Args:
        params (pandas.DataFrame): See :ref:`params`. Must contain finite
            lower and upper bounds for all parameters.
        n_gridpoints (int): Number of gridpoints on which the criterion
            function is evaluated. This is the number per plotted line.

    Returns:
        points (pandas.DataFrame): Each point at which the criterion will be evaluated
            is a row. Columns are the index of params.

    """
    pass
