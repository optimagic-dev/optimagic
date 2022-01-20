from itertools import combinations
from itertools import product

import numpy as np
import plotly.graph_objects as go
from estimagic.utilities import check_all_params_are_bounded
from estimagic.utilities import create_string_from_index_element
from plotly.subplots import make_subplots


def plot_2d_effects(
    criterion,
    params,
    n_gridpoints=10,
    plots_per_row=2,
):
    """Plot the surface of the criterion on 2D grids around a given value.

    Args:
        criterion (callable): criterion function. Takes a DataFrame and
            returns a scalar value or dictionary with the entry "value".
        params (pandas.DataFrame): See :ref:`params`. Must contain finite
            lower and upper bounds for all parameters. The plots will cover the whole
            distance between the bounds.
        n_gridpoints (int): Number of gridpoints in each direction on which the
            criterion function is evaluated.
        plots_per_row (int): How many plots are plotted per row.

    Returns:
        Fig: plotly.graph_objects.Figure

    """
    check_all_params_are_bounded(params)

    params = params.copy()
    if "name" not in params.columns:
        params["name"] = [create_string_from_index_element(tup) for tup in params.index]

    points = _create_points_to_evaluate(params, n_gridpoints)
    params_to_evaluate = _create_params_to_evaluate(points, params)
    points["criterion_value"] = [criterion(p) for p in params_to_evaluate]

    param_combinations = combinations(params.index, 2)
    surfaces = _create_surfaces(points, params, param_combinations)

    n_cols = plots_per_row
    n_rows = int(np.ceil(len(surfaces) / n_cols))
    specs = [[{"is_3d": True}] * n_cols] * n_rows
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=specs,
    )

    row = 1  # Plotly starts counting at 1
    column = 1
    for surface in surfaces:
        fig.add_trace(surface, row=row, col=column)

        if column < n_cols:
            column += 1
        else:
            column = 1
            row += 1

    # style plot
    fig.update_layout(width=n_cols * 400, height=n_rows * 400)

    return fig


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

    for x_loc, y_loc in param_combis:
        x_values = _create_linspace_with_start_value(params, x_loc, n_gridpoints)
        y_values = _create_linspace_with_start_value(params, y_loc, n_gridpoints)

        for x_val, y_val in product(x_values, y_values):
            new_point = params["value"].copy()
            new_point[x_loc, y_loc] = x_val, y_val
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
    for x_loc, y_loc in param_combinations:
        reduced_points = _reduce_to_points_with_fixed_other_dimensions(
            points, params, x_loc, y_loc
        )
        long_data = reduced_points.set_index([x_loc, y_loc])["criterion_value"]
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


def _reduce_to_points_with_fixed_other_dimensions(points, params, x_loc, y_loc):
    """Reduce points to those that are fixed in all other dimensions"""
    fixed_params = params.index.drop([x_loc, y_loc])
    fixed_values = params.loc[fixed_params, "value"]
    keep = (points[fixed_params] == fixed_values).all(axis=1)
    reduced = points[keep]
    return reduced


def _create_params_to_evaluate(points, params):
    """Create full params DataFrames with points to evaluate as "value" column.

    Args:
        points (pandas.DataFrame): Each point at which the criterion will be evaluated
            is a row. Columns are the index of params.
        params (pandas.DataFrame): See :ref:`params`.

    Returns:
        list: each entry is a params DataFrame with the "value" column replaced with
            one of the given points.

    """
    params_to_evaluate = []
    for loc in points.index:
        full_p = params.copy()
        full_p["value"] = points.loc[loc]
        params_to_evaluate.append(full_p)
    return params_to_evaluate
