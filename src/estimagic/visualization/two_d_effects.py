from itertools import combinations
from itertools import product

import numpy as np
import plotly.graph_objects as go
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
        criterion(points.loc[row_id].to_frame(name="value")) for row_id in points.index
    ]

    param_combinations = combinations(params.index, 2)
    surfaces = _create_surfaces(points, params, param_combinations)

    n_cols = plots_per_row
    n_rows = int(np.ceil(len(surfaces) / n_cols))
    specs = [[{"is_3d": True}] * n_cols] * n_rows
    fig = make_subplots(rows=n_rows, cols=n_cols, specs=specs)

    # Plotly starts counting at 1
    row = 1
    column = 1
    for surface in surfaces:
        fig.add_trace(surface, row=row, col=column)
        if column < n_cols:
            column += 1
        else:
            column = 1
            row += 1

    return fig


def _assert_all_params_are_bounded(params):
    """Raise a ValueError in case any parameter is missing a lower or upper bound.

    Args:
        params (pandas.DataFrame): params DataFrame.

    Raises:
        ValueError: When "lower_bound" or "upper_bound" are missing for any
            parameter or in general.

    """
    for bound_type in ["lower", "upper"]:
        if (
            f"{bound_type}_bound" not in params.columns
            or not np.isfinite(params[f"{bound_type}_bound"]).all()
        ):
            raise ValueError(f"All parameters need a finite {bound_type} bound.")


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
    param_combis = combinations(params.index, 2)

    points = params[["value"]].T

    for x_name, y_name in param_combis:
        x_values = _create_linspace_with_start_value(params, x_name, n_gridpoints)
        y_values = _create_linspace_with_start_value(params, y_name, n_gridpoints)

        for x_val, y_val in product(x_values, y_values):
            new_point = params["value"].copy()
            new_point[x_name, y_name] = x_val, y_val
            points = points.append(new_point)

    unique_points = points.drop_duplicates()
    unique_points = unique_points.reset_index(drop=True)
    return unique_points


def _create_linspace_with_start_value(params, row_id, n_gridpoints):
    """Create a linspace that includes the start value of the parameter.

    Args:
        params (pandas.DataFrame): See :ref:`params`. Must contain finite
            lower and upper bounds for all parameters.
        row_id: index of the parameter for which to create the grid
        n_gridpoints (int): Number of gridpoints in each direction on which the
            criterion function is evaluated.

    Returns:
        np.array: length is n_gridpoints + 1.

    """
    x_space_without_start_value = np.linspace(
        params.loc[row_id, "lower_bound"],
        params.loc[row_id, "upper_bound"],
        n_gridpoints,
    )
    given_value = params.loc[row_id, "value"]
    x_space = np.append(x_space_without_start_value, [given_value])
    x_space.sort()
    return x_space


def _create_surfaces(points, params, param_combinations):
    surfaces = []
    for x_name, y_name in param_combinations:
        reduced_points = _reduce_to_points_with_fixed_other_dimensions(
            points, params, x_name, y_name
        )
        long_data = reduced_points.set_index([x_name, y_name])["criterion_value"]
        z = long_data.unstack()
        x = z.index
        y = z.columns

        surfaces.append(
            go.Surface(
                x=x,
                y=y,
                z=z,
                showscale=False,
                colorscale="rdbu_r",  # coolwarm
            )
        )
    return surfaces


def _reduce_to_points_with_fixed_other_dimensions(points, params, x_name, y_name):
    """Reduce points to those that are fixed in all other dimensions"""
    fixed_params = params.index.drop([x_name, y_name])
    fixed_values = params.loc[fixed_params, "value"]
    keep = (points[fixed_params] == fixed_values).all(axis=1)
    reduced = points[keep]
    return reduced
