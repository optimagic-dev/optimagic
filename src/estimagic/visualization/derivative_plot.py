"""Visualize and compare derivative estimates."""
import itertools

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def derivative_plot(
    derivative_result,
    combine_plots_in_grid=True,
    template="plotly_white",
    palette=px.colors.qualitative.Plotly,
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

    # dimensions of problem. dimensions of params vector span the vertical axis while
    # dimensions of output span the horizontal axis of produced figure
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

        trace_1 = go.Scatter(
            x=_scatter_data["step"],
            y=_scatter_data["eval"],
            mode="markers",
            name="Function Evaluation",
            legendgroup=1,
            marker={"color": palette[0]},
        )
        g_ind.append(trace_1)

        # overall best derivative estimate
        _y = y0 + x_grid * df_der.loc[row, col]
        trace_2 = go.Scatter(
            x=x_grid,
            y=_y,
            mode="lines",
            name="Best Estimate",
            legendgroup=2,
            line={"dash": "dash", "color": palette[1]},
        )
        g_ind.append(trace_2)

        # best derivative estimate given each method
        for i, method in enumerate(["forward", "central", "backward"]):
            _y = y0 + x_grid * df_der_method.loc[method, row, col]

            trace_3 = go.Scatter(
                x=x_grid,
                y=_y,
                mode="lines",
                name=method,
                legendgroup=2 + i,
                line={"color": palette[2 + i]},
            )
            g_ind.append(trace_3)

        # fill area
        for sign in [1, -1]:
            _x_y = _select_eval_with_lowest_and_highest_step(func_evals, sign, row, col)
            diff = _x_y - np.array([0, y0])
            slope = diff[:, 1] / diff[:, 0]
            _y = y0 + x_grid * slope.reshape(-1, 1)

            trace_4 = go.Scatter(
                x=x_grid,
                y=_y[0, :],
                mode="lines",
                legendgroup=6,
                line={"color": palette[6]},
                showlegend=False,
            )
            g_ind.append(trace_4)

            trace_5 = go.Scatter(
                x=x_grid,
                y=_y[1, :],
                mode="lines",
                name="",
                legendgroup=6,
                line={"color": palette[6]},
                fill="tonexty",
            )
            g_ind.append(trace_5)

        # subplot x titles
        x_axis.append(fr"Value relative to x<sub>{0, row}</sub>")
        # subplot titles
        titles.append(f"dim_x, dim_f = {row, col}")
        # list of traces for individual plots
        g_list.append(g_ind)

    # Plot with subplots
    if combine_plots_in_grid:
        g = make_subplots(rows=len(dim_x), cols=len(dim_f), subplot_titles=titles)
        for ind, (facet_row, facet_col) in enumerate(
            itertools.product(range(1, len(dim_x) + 1), range(1, len(dim_f) + 1))
        ):
            traces = g_list[ind]
            for trace in range(len(traces)):
                g.add_trace(traces[trace], row=facet_row, col=facet_col)

        # deleting duplicates in legend
        g = clean_legend_duplicates(g)

        # setting x-axis titles
        for i in range(1, len(g_list) + 1):
            g["layout"]["xaxis{}".format(i)]["title"] = x_axis[i - 1]

        # setting template theme and size
        g.update_layout(
            template=template, height=300 * len(dim_x), width=500 * len(dim_f)
        )

        # scientific notations for axis ticks
        g.update_yaxes(tickformat=".2e")
        g.update_xaxes(tickformat=".2e")

        out = g

    # Dictionary for individual plots
    if not combine_plots_in_grid:
        ind_dict = create_ind_dict(
            g_list,
            titles,
            clean_legend=True,
            sci_notation=True,
            x_title=x_axis,
            kws={"template": template, "height": 300, "width": 500, "title_x": 0.5},
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
    out = df.head(1).append(df.tail(1)).values.copy()
    return out


def create_ind_dict(
    ind_list,
    names,
    kws,
    x_title=None,
    y_title=None,
    clean_legend=False,
    sci_notation=False,
    share_xax=False,
    x_min=None,
    x_max=None,
):
    """Create a dictionary for individual plots from a list of traces.

    Args:
        ind_list (iterable): The list of traces for each individual plot.
        names (iterable): The list of titles for the each plot.
        kws (dict): The dictionary for the layout.update, unified for each
        individual plot.
        x_title (iterable or None): The list of x-axis labels for each plot. If None,
        then no labels are added.
        y_title (iterable or None): The list of y-axis labels for each plot. If None,
        then no labels are added.
        clean_legend (bool): If True, then cleans the legend from duplicates.
        Default False.
        sci_notation (bool): If True then updates the ticks on x- and y-axis to
        be displayed in a scientific notation. Default False.
        share_xax (bool): If True, then the x-axis domain is the same
        for each individual plot.
        x_min (int or None): The lower bound for share_xax.
        x_max (int or None): The upped bound for share_xax.

    Returns:
        dictionary of individual plots

    """
    fig_dict = {}
    if x_title is None:
        x_title = ["" for ind in range(len(ind_list))]
    if y_title is None:
        y_title = ["" for ind in range(len(ind_list))]

    for ind in range(len(ind_list)):
        fig = go.Figure()
        traces = ind_list[ind]
        for trace in range(len(traces)):
            fig.add_trace(traces[trace])
        # adding title and styling axes and theme
        fig.update_layout(
            title=names[ind], xaxis_title=x_title[ind], yaxis_title=y_title[ind], **kws
        )
        # scientific notations for axis ticks
        if sci_notation:
            fig.update_yaxes(tickformat=".2e")
            fig.update_xaxes(tickformat=".2e")
        # deleting duplicates in legend
        if clean_legend:
            fig = clean_legend_duplicates(fig)
        if share_xax:
            fig.update_xaxes(range=[x_min, x_max])
        # adding to dictionary
        key = names[ind].replace(" ", "_").lower()
        fig_dict[key] = fig

    return fig_dict


def clean_legend_duplicates(fig):
    names = set()
    fig.for_each_trace(
        lambda trace: trace.update(showlegend=False)
        if (trace.name in names)
        else names.add(trace.name)
    )
    return fig
