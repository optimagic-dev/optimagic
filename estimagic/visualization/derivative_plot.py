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
    grid_points=50,
):
    """Plot evaluations and derivative estimates.

    The resulting grid plot displays function evaluations and derivatives. The
    derivatives are visualized as a first-rder Taylor approximation. Bands are drawn
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
        grid_points (int): Number of grid points used for plotting x-axis. Default 50.

    Returns:
        fig (matplotlib.pyplot.figure): The figure.

    """
    df = df_evals.reset_index()  # remove index from main data for plotting
    df = df.assign(**{"step": df.step * df.sign})
    df_evals = df.set_index(["sign", "step_number", "dim_x", "dim_f"])

    # subset data to keep inquired dimensions
    dimensions = df[["dim_x", "dim_f", "step_number"]].max()
    dim_x = (
        np.atleast_1d(dim_x) if dim_x is not None else range(dimensions["dim_x"] + 1)
    )
    dim_f = (
        np.atleast_1d(dim_f) if dim_f is not None else range(dimensions["dim_f"] + 1)
    )
    df = df.query("dim_x in @dim_x & dim_f in @dim_f")

    # prepare derivative data
    minimizer_method = df_jac_cand.groupby(["method", "dim_x", "dim_f"])["err"].idxmin()
    minimizer = df_jac_cand.groupby(["dim_x", "dim_f"])["err"].idxmin()

    der_method = df_jac_cand.loc[minimizer_method]["der"].droplevel("num_term")
    der = df_jac_cand.loc[minimizer]["der"].droplevel(["method", "num_term"])

    # auxiliary
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
        _x = np.linspace(-max_steps[row], max_steps[row], grid_points)

        # draw func evals scatter
        _scatter_data = df_evals.loc[:, :, row, col]
        ax.scatter(
            _scatter_data["step"],
            _scatter_data["eval"],
            color="gray",
            label="Function Evaluation",
            edgecolor="black",
        )

        # draw overall best_estimate of derivative
        _y = y0 + _x * der.loc[row, col]
        ax.plot(
            _x,
            _y,
            color="black",
            label="Best Estimate",
            zorder=2,
            linewidth=1.5,
            linestyle="dashdot",
        )

        # draw best derivative given each method
        for method in ["forward", "central", "backward"]:
            _der = der_method.loc[method, row, col]
            _y = y0 + _x * _der
            ax.plot(_x, _y, color=palette[method], label=method, zorder=1, linewidth=2)

        # fill area
        for sign in [1, -1]:
            y_list = []
            for step in ["start", "stop"]:
                if step == "start":
                    _x_y = (
                        df_evals.loc[(sign, slice(None), row, col), ["step", "eval"]]
                        .dropna()
                        .sort_index()
                        .head(1)
                        .values.flatten()
                    )  # selects lowest step for which eval is not nan
                elif step == "stop":
                    _x_y = (
                        df_evals.loc[(sign, slice(None), row, col), ["step", "eval"]]
                        .dropna()
                        .sort_index()
                        .tail(1)
                        .values.flatten()
                    )  # selects highest step for which eval is not nan
                diff = _x_y - np.array([0, y0])
                slope = diff[1] / diff[0]
                _y = y0 + _x * slope
                y_list.append(_y)
                ax.plot(_x, _y, "--", color=palette[sign], linewidth=0.5)
            ax.fill_between(_x, y_list[0], y_list[1], alpha=0.15, color=palette[sign])

    # legend
    ncol = 5 if len(dim_f) > 1 else 3
    axes[0, 0].legend(
        loc="upper center",
        bbox_to_anchor=(len(dim_f) / 2 + 0.05 * len(dim_f), 1.15),
        ncol=ncol,
        fontsize=14,
    )
    return fig
