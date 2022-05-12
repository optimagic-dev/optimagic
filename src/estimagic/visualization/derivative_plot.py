"""Visualize and compare derivative estimates."""
import itertools

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from estimagic.config import PLOTLY_PALETTE
from estimagic.config import PLOTLY_TEMPLATE
from estimagic.visualization.plotting_utilities import create_grid_plot
from estimagic.visualization.plotting_utilities import create_ind_dict


def derivative_plot(
    derivative_result,
    combine_plots_in_grid=True,
    template=PLOTLY_TEMPLATE,
    palette=PLOTLY_PALETTE,
):
    """Plot evaluations and derivative estimates.

    The resulting grid plot displays function evaluations and derivatives. The
    derivatives are visualized as a first-order Taylor approximation. Bands are drawn
    indicating the area in which forward and backward derivatives are located. This is
    done by filling the area between the derivative estimate with lowest and highest
    step size, respectively. Do not confuse these bands with statistical errors.

    This function does not require the params vector as plots are displayed relative to
    the point at which the derivative is calculated.

    Args:
        derivative_result (dict): The result dictionary of call to
            :func:`~estimagic.differentiation.derivatives.first_derivative` with
            return_info and return_func_value set to True.
        combine_plots_in_grid (bool): decide whether to return a one
            figure containing subplots for each factor pair or a dictionary
            of individual plots. Default True.
        template (str): The template for the figure. Default is "plotly_white".
        palette: The coloring palette for traces. Default is "qualitative.Plotly".

    Returns:
        plotly.Figure: The grid plot or dict of individual plots

    """

    func_value = derivative_result["func_value"]
    func_evals = derivative_result["func_evals"]
    derivative_candidates = derivative_result["derivative_candidates"]

    # remove index from main data for plotting
    df = func_evals.reset_index()
    df = df.assign(**{"step": df.step * df.sign})
    func_evals = df.set_index(["sign", "step_number", "dim_x", "dim_f"])

    # prepare derivative data
    df_der = _select_derivative_with_minimal_error(derivative_candidates)
    df_der_method = _select_derivative_with_minimal_error(
        derivative_candidates, given_method=True
    )

    # auxiliary
    grid_points = 2  # we do not need more than 2 grid points since all lines are affine
    func_value = np.atleast_1d(func_value)
    max_steps = df.groupby("dim_x")["step"].max()

    # dimensions of params vector (dim_x) span the vertical axis while dimensions of
    # output (dim_f) span the horizontal axis of produced figure
    dim_x = range(df["dim_x"].max() + 1)
    dim_f = range(df["dim_f"].max() + 1)

    # plotting

    # container for titles
    titles = []
    # container for x-axis titles
    x_axis = []
    # container for individual plots
    g_list = []

    # creating data traces for plotting faceted/individual plots
    for (row, col) in itertools.product(dim_x, dim_f):
        g_ind = []  # container for data for traces in individual plot

        # initial values and x grid
        y0 = func_value[col]
        x_grid = np.linspace(-max_steps[row], max_steps[row], grid_points)

        # initial values and x grid
        y0 = func_value[col]
        x_grid = np.linspace(-max_steps[row], max_steps[row], grid_points)

        # function evaluations scatter points
        _scatter_data = func_evals.query("dim_x == @row & dim_f == @col")

        trace_func_evals = go.Scatter(
            x=_scatter_data["step"],
            y=_scatter_data["eval"],
            mode="markers",
            name="Function Evaluation",
            legendgroup=1,
            marker={"color": "black"},
        )
        g_ind.append(trace_func_evals)

        # best derivative estimate given each method
        for i, method in enumerate(["forward", "central", "backward"]):
            _y = y0 + x_grid * df_der_method.loc[method, row, col]

            trace_method = go.Scatter(
                x=x_grid,
                y=_y,
                mode="lines",
                name=method,
                legendgroup=2 + i,
                line={"color": palette[i], "width": 5},
            )
            g_ind.append(trace_method)

        # fill area
        for sign, cmap_id in zip([1, -1], [0, 2]):  # cmap_id of ['forward', 'backward']
            _x_y = _select_eval_with_lowest_and_highest_step(func_evals, sign, row, col)
            diff = _x_y - np.array([0, y0])
            slope = diff[:, 1] / diff[:, 0]
            _y = y0 + x_grid * slope.reshape(-1, 1)

            trace_fill_lines = go.Scatter(
                x=x_grid,
                y=_y[0, :],
                mode="lines",
                line={"color": palette[cmap_id], "width": 1},
                showlegend=False,
            )
            g_ind.append(trace_fill_lines)

            trace_fill_area = go.Scatter(
                x=x_grid,
                y=_y[1, :],
                mode="lines",
                line={"color": palette[cmap_id], "width": 1},
                fill="tonexty",
            )
            g_ind.append(trace_fill_area)

        # overall best derivative estimate
        _y = y0 + x_grid * df_der.loc[row, col]
        trace_best_estimate = go.Scatter(
            x=x_grid,
            y=_y,
            mode="lines",
            name="Best Estimate",
            legendgroup=2,
            line={"color": "black", "width": 2},
        )
        g_ind.append(trace_best_estimate)

        # subplot x titles
        x_axis.append(rf"Value relative to x<sub>{0, row}</sub>")
        # subplot titles
        titles.append(f"dim_x, dim_f = {row, col}")
        # list of traces for individual plots
        g_list.append(g_ind)

    common_dependencies = {
        "ind_list": g_list,
        "names": titles,
        "clean_legend": True,
        "scientific_notation": True,
        "x_title": x_axis,
    }
    common_layout = {
        "template": template,
        "margin": {"l": 10, "r": 10, "t": 30, "b": 10},
    }

    # Plot with subplots
    if combine_plots_in_grid:
        g = create_grid_plot(
            rows=len(dim_x),
            cols=len(dim_f),
            **common_dependencies,
            kws={
                "height": 300 * len(dim_x),
                "width": 500 * len(dim_f),
                **common_layout,
            },
        )
        out = g

    # Dictionary for individual plots
    else:
        ind_dict = create_ind_dict(
            **common_dependencies,
            kws={"height": 300, "width": 500, "title_x": 0.5, **common_layout},
        )

        out = ind_dict

    return out


def _select_derivative_with_minimal_error(df_jac_cand, given_method=False):
    """Select derivatives with minimal error component wise.

    Args:
        df_jac_cand (pandas.DataFrame): Frame containing jacobian candidates.
        given_method (bool): Boolean indicating wether to condition on columns method
            in df_jac_cand. Default is False, which selects the overall best derivative
            estimate.

    Returns:
        df (pandas.DataFrame): The (best) derivative estimate.

    """
    given = ["method"] if given_method else []
    minimizer = df_jac_cand.groupby(given + ["dim_x", "dim_f"])["err"].idxmin()
    df = df_jac_cand.loc[minimizer]["der"]
    index_level_to_drop = list({"method", "num_term"} - set(given))
    df = df.droplevel(index_level_to_drop).copy()
    return df


def _select_eval_with_lowest_and_highest_step(df_evals, sign, dim_x, dim_f):
    """Select step and eval from data with highest and lowest step.

    Args:
        df_evals (pd.DataFrame): Frame containing func evaluations (long-format).
        sign (int): Direction of step.
        dim_x (int): Dimension of x to select.
        dim_f (int): Dimension of f to select.

    Returns:
        out (numpy.ndarray): Array of shape (2, 2). Columns correspond to step and eval,
            while rows correspond to lowest and highest step, respectively.

    """
    df = df_evals.loc[(sign, slice(None), dim_x, dim_f), ["step", "eval"]]
    df = df.dropna().sort_index()
    out = pd.concat([df.head(1), df.tail(1)]).to_numpy()

    return out
