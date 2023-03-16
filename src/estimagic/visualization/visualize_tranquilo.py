from copy import deepcopy

import numpy as np
import pandas as pd
import plotly.express as px
from numba import njit
from plotly import figure_factory as ff
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from estimagic.optimization.optimize_result import OptimizeResult
from estimagic.optimization.tranquilo.clustering import cluster
from estimagic.optimization.tranquilo.geometry import log_d_quality_calculator
from estimagic.optimization.tranquilo.volume import get_radius_after_volume_scaling


def visualize_tranquilo(results, iterations):
    """Plot diagnostic information of optimization result in given iteration(s).

    Generates plots with sample points (trustregion or heatmap), criterion evaluations,
    trustregion radii and other diagnostic information to compare different algorithms
    at an iteration or different iterations for a given algorithm.

    Currently works for the following algorithms: `tranquilo`, `tranquilo_ls`,
    `nag_pybobyqa` and `nag_dfols`.

    Args:
        results (dict or OptimizeResult): An estimagic optimization result or a
            dictionary with different estimagic optimization results.
        iterations (int, list, tuple or dict): The iterations to compare the results
            at. Can be an integer if we want to compare different results at the same
            iteration, a list or tuple if we want to compare different iterations of
            the same optimization result, or dictionary with the same keys as results
            and with integer values if we want to compare different iterations of
            different results.
    Returns:
        fig (plotly.Figure): Plotly figure that combines the following plots:
            - sample points: plot with model points at current iteration and the
                trust region, if number of parameters is not larger than 2, or
                a heatmap of (absolute) correlations of sample points for higher
                dimensional parameter spaces.
            - distance plot: L2 and infinity norm-distances of model points from
                the trustregion center.
            - criterion plot: function evaluations with sample points and current
                accepted point highlighted.
            - rho plots: the ratio of expected and actual improvement in function
                values at each iteration.
            - radius plots: trustregion radii at each iteration.
            - cluster plots: number of clusters relative to number of sample points
                at each iteration.
            - fekete criterion plots: the value of the fekete criterion at each
                iteration.

    """
    results = deepcopy(results)
    if isinstance(iterations, int):
        iterations = {case: iterations for case in results}
        results = {case: _process_results(results[case]) for case in results}
    elif isinstance(results, OptimizeResult):
        results = _process_results(results)
        results = {f"iteration {i}": results for i in iterations}
        iterations = {f"iteration {iteration}": iteration for iteration in iterations}

    cases = results.keys()
    nrows = 8
    ncols = len(cases)
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=list(cases),
        horizontal_spacing=1 / (ncols * 6),
        vertical_spacing=(1 / (nrows - 1)) / 4,
        shared_yaxes=True,
    )
    color_dict = {
        "existing": "rgb(0,0,255)",
        "new": "rgb(230,0,0)",
        "discarded": "rgb(0,0,0)",
    }
    xl = []
    xu = []
    for i, case in enumerate(cases):
        result = results[case]
        iteration = iterations[case]
        state = result.algorithm_output["states"][iteration]
        params_history = np.array(result.history["params"])
        criterion_history = np.array(result.history["criterion"])
        fig = _plot_sample_points(
            params_history, state, color_dict, fig, row=1, col=i + 1
        )
        fig = _plot_distances_from_center(
            params_history, state, fig, rows=[2, 3], col=i + 1
        )
        xl.append(fig.get_subplot(row=2, col=i + 1).xaxis.range[0])
        xu.append(fig.get_subplot(row=2, col=i + 1).xaxis.range[1])
        fig = _plot_criterion(
            criterion_history, state, color_dict, fig, row=4, col=i + 1
        )
        fig = _plot_rhos(result, fig, iteration=iteration, row=5, col=i + 1)
        fig = _plot_radii(result, fig, iteration=iteration, row=6, col=i + 1)
        fig = _plot_clusters_points_ratio(result, iteration, fig, row=7, col=i + 1)
        fig = _plot_fekete_criterion(result, fig, iteration=iteration, row=8, col=i + 1)
        fig.layout.annotations[i].update(y=1.015)
    for r in [2, 3]:
        for c in range(1, ncols + 1):
            fig.update_xaxes(range=[min(xl) - 0.25, max(xu) + 0.25], row=r, col=c)
    fig = _clean_legend_duplicates(fig)
    fig.update_layout(height=400 * nrows, width=460 * ncols, template="plotly_white")
    fig.update_yaxes(
        showgrid=False, showline=True, linewidth=1, linecolor="black", zeroline=False
    )
    fig.update_xaxes(
        showgrid=False, showline=True, linewidth=1, linecolor="black", zeroline=False
    )
    fig.update_layout(hovermode="x unified")

    return fig


def _plot_criterion(history, state, color_dict, fig, row, col):
    fig.add_trace(
        go.Scatter(
            y=history,
            x=np.arange(len(history)),
            showlegend=False,
            line_color="#C0C0C0",
            name="Criterion",
            mode="lines",
        ),
        row=row,
        col=col,
    )

    fig.add_trace(
        go.Scatter(
            y=history[state.old_indices_used],
            x=state.old_indices_used,
            mode="markers",
            marker_size=10,
            name="existing ",
            showlegend=False,
            marker_color=color_dict["existing"],
            opacity=0.6,
        ),
        col=col,
        row=row,
    )
    fig.add_trace(
        go.Scatter(
            y=history[state.new_indices],
            x=state.new_indices,
            mode="markers",
            marker_size=10,
            name="new ",
            showlegend=False,
            marker_color=color_dict["new"],
            opacity=0.6,
        ),
        col=col,
        row=row,
    )
    fig.add_trace(
        go.Scatter(
            y=history[
                getattr(state, "old_indices_discarded", np.array([], dtype="int"))
            ],
            x=getattr(state, "old_indices_discarded", np.array([], dtype="int")),
            mode="markers",
            marker_size=10,
            name="discarded ",
            showlegend=False,
            marker_color=color_dict["discarded"],
            opacity=0.6,
        ),
        col=col,
        row=row,
    )
    fig.add_trace(
        go.Scatter(
            y=[history[state.index]],
            x=[state.index],
            mode="markers",
            marker_size=12,
            name="current index",
            showlegend=False,
            marker_color="red",
            marker_symbol="star",
            marker_line_color="black",
            marker_line_width=1,
            opacity=0.6,
        ),
        col=col,
        row=row,
    )
    fig.update_xaxes(title_text="Function evaluations", row=row, col=col)
    if col == 1:
        fig.update_yaxes(title_text="Criterion value", row=row, col=col)
    return fig


def _plot_sample_points(history, state, color_dict, fig, row, col):
    sample_points = _get_sample_points(state, history)
    if state.x.shape[0] <= 2:
        trustregion = state.trustregion
        radius = trustregion.radius
        center = trustregion.center
        fig.add_shape(
            type="circle",
            xref="x",
            yref="y",
            x0=center[0] - radius,
            y0=center[1] - radius,
            x1=center[0] + radius,
            y1=center[1] + radius,
            line_width=0.5,
            col=col,
            row=row,
            line_color="grey",
        )

        fig.add_traces(
            px.scatter(
                sample_points,
                x=0,
                y=1,
                color="case",
                color_discrete_map=color_dict,
                opacity=0.7,
            ).data,
            cols=col,
            rows=row,
        )
        fig.update_traces(
            marker_size=10,
            marker_line_color="black",
            marker_line_width=2,
            col=col,
            row=row,
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1, col=col, row=row)
        fig.update_xaxes(scaleanchor="y", scaleratio=1, col=col, row=row)
    else:
        params = [col for col in sample_points.columns if col != "case"]
        corr = sample_points[params].corr().abs()
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.tril_indices_from(mask, k=-1)] = True
        corr = corr.where(mask)
        fig.add_trace(
            go.Heatmap(
                z=corr,
                x=corr.columns.values,
                y=corr.index.values,
                showscale=False,
                colorscale="Magenta",
                zmin=0,
                zmax=1,
                text=corr.to_numpy().round(2).tolist(),  # xxxx,
                texttemplate="%{text}",
            ),
            row=row,
            col=col,
        )
        fig.update_layout(yaxis_autorange="reversed")
        fig.update_xaxes(tickmode="array", tickvals=corr.index.values, col=col, row=row)
        fig.update_yaxes(
            tickmode="array", tickvals=corr.columns.values, col=col, row=row
        )
    return fig


def _plot_radii(res, fig, row, col, iteration):
    radii = [state.trustregion.radius for state in res.algorithm_output["states"]]
    traces = plot_line_with_lighlighted_point(
        x=np.arange(len(radii)), y=radii, highlighted_point=iteration, name="Radius"
    )
    fig.add_traces(
        traces,
        rows=row,
        cols=col,
    )
    fig.update_xaxes(title_text="Iteration", row=row, col=col)
    if col == 1:
        fig.update_yaxes(title_text="Radius", row=row, col=col)
    return fig


def _plot_rhos(res, fig, row, col, iteration):
    rhos = np.array([state.rho for state in res.algorithm_output["states"]])
    rhos[~pd.isna(rhos)] = np.clip(rhos[~pd.isna(rhos)], -1, 3)
    traces = plot_line_with_lighlighted_point(
        x=np.arange(len(rhos)), y=rhos, highlighted_point=iteration, name="Rho"
    )
    fig.add_traces(
        traces,
        rows=row,
        cols=col,
    )
    fig.update_xaxes(title_text="Iteration", row=row, col=col)
    if col == 1:
        fig.update_yaxes(title_text="Rho", row=row, col=col)
    return fig


def _plot_fekete_criterion(res, fig, row, col, iteration):
    fekete = _get_fekete_criterion(res)
    traces = plot_line_with_lighlighted_point(
        x=np.arange(len(fekete)), y=fekete, highlighted_point=iteration, name="Fekete"
    )
    fig.add_traces(
        traces,
        rows=row,
        cols=col,
    )
    fig.update_xaxes(title_text="Iteration", row=row, col=col)
    if col == 1:
        fig.update_yaxes(title_text="Fekete criterion", row=row, col=col)
    return fig


def _plot_clusters_points_ratio(res, iteration, fig, row, col):
    dim = res.params.shape[0]
    history = np.array(res.history["params"])
    states = res.algorithm_output["states"]
    colors = [
        "rgb(251,106,74)",
        "rgb(203,24,29)",
        "rgb(103,0,13)",
    ]
    for i, f in enumerate([1, 2, 10]):
        ratios = [np.nan]
        for state in states[1:]:
            n_points = state.model_indices.shape[0]
            points = history[state.model_indices]
            scaling = 1 / (f * n_points)
            radius = get_radius_after_volume_scaling(
                state.trustregion.radius, dim, scaling
            )
            _, centers = cluster(points, radius)
            n_clusters = centers.shape[0]
            ratios.append(n_clusters / n_points)
        fig.add_trace(
            go.Scatter(
                y=ratios,
                x=np.arange(len(ratios)),
                mode="lines",
                opacity=0.5,
                line_color=colors[i],
                line_width=1.5,
                name=f"s={f}*n",
            ),
            col=col,
            row=row,
        )
        fig.add_trace(
            go.Scatter(
                y=[ratios[iteration]],
                x=[iteration],
                mode="markers",
                marker_color=colors[i],
                opacity=1,
                marker_size=10,
                name=f"s={f}*n",
                showlegend=False,
            ),
            col=col,
            row=row,
        )
    fig.update_xaxes(title_text="Iteration", row=row, col=col)
    if col == 1:
        fig.update_yaxes(title_text="Cluster ratio", row=row, col=col)
    return fig


def _plot_distances_from_center(history, state, fig, col, rows):
    dist_sq = (
        np.linalg.norm(
            history[state.model_indices] - state.trustregion.center,
            axis=1,
        )
        / state.trustregion.radius
    )

    dist_inf = (
        np.linalg.norm(
            history[state.model_indices] - state.trustregion.center,
            axis=1,
            ord=np.inf,
        )
        / state.trustregion.radius
    )

    for r, inputs in enumerate([dist_sq, dist_inf]):
        data = ff.create_distplot(
            [inputs],
            show_curve=False,
            show_rug=True,
            group_labels=[""],
            show_hist=False,
        ).data

        data[0].update(
            {
                "yaxis": "y",
                "y": [0] * len(inputs),
                "showlegend": False,
                "marker_size": 20,
            }
        )
        fig.add_traces(data, cols=col, rows=rows[r])

    min_dist = min(dist_inf.min(), dist_sq.min())
    max_dist = max(dist_inf.max(), dist_sq.max())

    fig.update_xaxes(
        title_text="L2 norm", range=[min_dist, max_dist], row=rows[0], col=col
    )
    fig.update_xaxes(
        title_text="Inf norm", range=[min_dist, max_dist], row=rows[1], col=col
    )
    return fig


def _get_fekete_criterion(res):
    states = res.algorithm_output["states"][1:]
    history = np.array(res.history["params"])

    out = [np.nan] + [
        log_d_quality_calculator(
            sample=history[state.model_indices],
            trustregion=state.trustregion,
        )
        for state in states
    ]
    return out


def _get_sample_points(state, history):
    current_points = history[state.model_indices]
    discarded_points = history[
        getattr(state, "old_indices_discarded", np.array([], dtype="int"))
    ]
    df = pd.DataFrame(
        data=np.vstack([current_points, discarded_points]),
        index=np.hstack(
            [
                state.model_indices,
                getattr(state, "old_indices_discarded", np.array([], dtype="int")),
            ]
        ),
    )
    df["case"] = np.nan
    df.loc[state.new_indices, "case"] = "new"
    df.loc[state.old_indices_used, "case"] = "existing"
    df.loc[
        getattr(state, "old_indices_discarded", np.array([], dtype="int")), "case"
    ] = "discarded"
    return df


def plot_line_with_lighlighted_point(x, y, name, highlighted_point):
    """Plot line x,y, add markers to the line to highlight data points.
    args:
        x(np.ndarray or list): 1d array or list of data for x axis
        y(np.ndarray or list): 1d array or list of data for y axis
        highlight_points(np.ndarray or list): 1d array of indices of the to highlight.
        in case of
                - criterion: x is the array with numbers of function evaluations
                             y is the array with function values
                             highlight points is a nested list with lists of
                                    - existing points
                                    - new points
                                    - discarded points
                - other plots: x is the array with iteration numbers
                               y is the array with corresponding objective values.
                               highlight points is the index of the current iteration.

    returns:
        go.Figure

    """
    highlight_color = "#035096"
    highlight_size = 10
    line_color = "#C0C0C0"
    data = [
        go.Scatter(
            y=y, x=x, mode="lines", line_color=line_color, name=name, showlegend=False
        ),
        go.Scatter(
            x=[highlighted_point],
            y=[y[highlighted_point]],
            mode="markers",
            marker_color=highlight_color,
            marker_size=highlight_size,
            name="current val",
            showlegend=False,
        ),
    ]

    return data


def _clean_legend_duplicates(fig):
    trace_names = set()

    def disable_legend_if_duplicate(trace):
        if trace.name in trace_names:
            # in this case the legend is a duplicate
            trace.update(showlegend=False)
        else:
            trace_names.add(trace.name)

    fig.for_each_trace(disable_legend_if_duplicate)
    return fig


def _process_results(result):
    """Add model indices to states of optimization result."""
    result = deepcopy(result)
    xs = np.array(result.history["params"])
    if result.algorithm in ["nag_pybobyqa", "nag_dfols"]:
        for i in range(1, len(result.algorithm_output["states"])):
            state = result.algorithm_output["states"][i]
            result.algorithm_output["states"][i] = state._replace(
                model_indices=_get_model_indices(xs, state),
                new_indices=_get_model_indices(xs, state),
                index=_find_index(
                    xs,
                    state.x,
                )[0],
            )
    elif result.algorithm in ["tranquilo", "tranquilo_ls"]:
        pass
    else:
        NotImplementedError(
            f"Diagnostic plots are not implemented for {result.algorithm}"
        )
    return result


@njit
def _find_indices_in_trust_region(xs, center, radius):
    """Get the row indices of all parameter vectors in a trust region.

    This is for square trust regions, i.e. balls in term of an infinity norm.

    Args:
        xs (np.ndarray): 2d numpy array where each row is a parameter vector.
        center (np.ndarray): 1d numpy array that marks the center of the trust region.
        radius (float): Radius of the trust region.

    Returns:
        np.ndarray: The indices of parameters in the trust region.

    """
    n_obs, dim = xs.shape
    out = np.zeros(n_obs).astype(np.int64)
    success_counter = 0
    upper = center + radius
    lower = center - radius
    for i in range(n_obs):
        success = True
        for j in range(dim):
            value = xs[i, j]
            if not (lower[j] <= value <= upper[j]) or np.isnan(value):
                success = False
                continue
        if success:
            out[success_counter] = i
            success_counter += 1

    return out[:success_counter]


def _find_index(xs, point):
    radius = 1e-100
    out = np.array([])
    while len(out) == 0:
        out = _find_indices_in_trust_region(xs=xs, center=point, radius=radius)
        radius = np.sqrt(radius)
    if len(out) == 1:
        return out
    else:
        return out[0]


def _get_model_indices(xs, state):
    model_indices = np.array([])
    for point in state.model_points:
        model_indices = np.concatenate([model_indices, _find_index(xs, point)])
    return model_indices.astype(int)
