"""Visualize and compare derivative estimates."""
import itertools

import matplotlib.pyplot as plt
import numpy as np


def derivative_plot(
    df_evals,
    df_jac_cand,
    func_value,
    params,
    dim_x=None,
    dim_f=None,
    height=None,
    width=None,
):
    """Plot evaluations and derivative estimates.

    The resulting grid plot displays function evaluations and derivatives. The
    derivatives are visualized as a first-order Taylor approximation. Bands are drawn
    indicating the area in which forward and backward derivatives are located. This is
    done by filling the area between the derivative estimate with lowest and highest
    step size, respectively. Do not confuse these bands with statistical errors.

    Args:
        df_evals (pd.DataFrame): Frame containing func evaluations (long-format).
        df_jac_cand (pd.DataFrame): Frame containing jacobian candidates (long-format).
        func_value (np.ndarray): Func value at original params vector.
        params (np.ndarray): Initial params vector.
        dim_x (iterable): Input dimensions to consider. Default None, selects all.
        dim_f (iterable): Output dimensions to consider. Default None, selects all.
        height (float): Figure Height. Default None, which sets single fig-height to 11.
        width (float): Figure width. Default None, which sets single fig-width to 10.

    Returns:
        fig (matplotlib.pyplot.figure): The figure.

    """
    df = df_evals.reset_index()  # remove index from main data for plotting
    df = df.assign(**{"step": df.step * df.sign})
    df_evals = df.set_index(["sign", "step_number", "dim_x", "dim_f"])

    # subset data keeping inquired dimensions
    dim_x, dim_f = _get_dims_from_data_if_no_user_input_else_forward(df, dim_x, dim_f)
    df_evals = df_evals.query("dim_x in @dim_x & dim_f in @dim_f")

    # prepare derivative data
    df_der = _select_derivative_with_minimal_error(df_jac_cand)
    df_der_method = _select_derivative_with_minimal_error(
        df_jac_cand, given_method=True
    )

    # auxiliary
    grid_points = 2  # we do not need more grid points since all lines are affine
    func_value = np.atleast_1d(func_value)
    max_steps = df.groupby("dim_x")["step"].max()
    palette = {
        "forward": "tab:green",
        "central": "tab:blue",
        "backward": "tab:orange",
        1: "green",
        -1: "orange",
    }

    # plot
    width = 10 * len(dim_f) if width is None else width
    height = 11 * len(dim_x) if height is None else height

    fig, axes = plt.subplots(len(dim_x), len(dim_f), figsize=(width, height))
    axes = np.atleast_2d(axes)

    for ax, (row, col) in zip(axes.flatten(), itertools.product(dim_x, dim_f)):
        # labels and texts
        ax.set_xlabel(fr"Value relative to $x_{{0, {row}}}$", fontsize=14)
        ax.text(
            0.35,
            1.02,
            f"dim_x, dim_f = {row, col}",
            transform=ax.transAxes,
            color="grey",
            fontsize=14,
        )

        # initial values and x grid
        y0 = func_value[col]
        x_grid = np.linspace(-max_steps[row], max_steps[row], grid_points)

        # plot function evaluations scatter points
        _scatter_data = df_evals.loc[:, :, row, col]
        ax.scatter(
            _scatter_data["step"],
            _scatter_data["eval"],
            color="gray",
            label="Function Evaluation",
            edgecolor="black",
        )

        # draw overall best derivative estimate
        _y = y0 + x_grid * df_der.loc[row, col]
        ax.plot(
            x_grid,
            _y,
            color="black",
            label="Best Estimate",
            zorder=2,
            linewidth=1.5,
            linestyle="dashdot",
        )

        # draw best derivative estimate given each method
        for method in ["forward", "central", "backward"]:
            _y = y0 + x_grid * df_der_method.loc[method, row, col]
            ax.plot(
                x_grid, _y, color=palette[method], label=method, zorder=1, linewidth=2
            )

        # fill area
        for sign in [1, -1]:
            _x_y = _select_eval_with_lowest_and_highest_step(df_evals, sign, row, col)
            diff = _x_y - np.array([0, y0])
            slope = diff[:, 1] / diff[:, 0]
            _y = y0 + x_grid * slope.reshape(-1, 1)
            ax.plot(x_grid, _y.T, "--", color=palette[sign], linewidth=0.5)
            ax.fill_between(x_grid, _y[0, :], _y[1, :], alpha=0.15, color=palette[sign])

    # legend
    ncol = 5 if len(dim_f) > 1 else 3
    axes[0, 0].legend(
        loc="upper center",
        bbox_to_anchor=(len(dim_f) / 2 + 0.05 * len(dim_f), 1.15),
        ncol=ncol,
        fontsize=14,
    )
    return fig


def _get_dims_from_data_if_no_user_input_else_forward(df, dim_x, dim_f):
    """Select dimensions from data or pass user input through if not None.

    Args:
        df (pandas.DataFrame): Frame containing data on evaluations.
        dim_x (list-like or None): Dimensions of x to consider.
        dim_f (list-like or None): Dimensions of f to consider.

    Returns:
        dims (tuple[numpy.ndarray[1d]]): Dimensions to consider.

    """
    dimensions = df[["dim_x", "dim_f"]].max()
    dims = (
        np.atleast_1d(x) if x is not None else range(dimensions[n] + 1)
        for x, n in zip((dim_x, dim_f), ("dim_x", "dim_f"))
    )
    return dims


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
