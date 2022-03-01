import itertools

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from estimagic.benchmarking.process_benchmark_results import (
    create_convergence_histories,
)
from estimagic.utilities import propose_alternatives
from plotly.subplots import make_subplots


def convergence_plot(
    problems=None,
    results=None,
    problem_subset=None,
    algorithm_subset=None,
    n_cols=2,
    distance_measure="criterion",
    monotone=True,
    normalize_distance=True,
    runtime_measure="n_evaluations",
    stopping_criterion="y",
    x_precision=1e-4,
    y_precision=1e-4,
    combine_plots_in_grid=True,
    template="plotly_white",
):
    """Plot convergence of optimizers for a set of problems.

    This creates a grid of plots, showing the convergence of the different
    algorithms on each problem. The faster a line falls, the faster the algorithm
    improved on the problem. The algorithm converged where its line reaches 0
    (if normalize_distance is True) or the horizontal blue line labeled "true solution".

    Each plot shows on the x axis the runtime_measure, which can be walltime or number
    of evaluations. Each algorithm's convergence is a line in the plot. Convergence can
    be measured by the criterion value of the particular time/evaluation. The
    convergence can be made monotone (i.e. always taking the bast value so far) or
    normalized such that the distance from the start to the true solution is one.

    Args:
        problems (dict): estimagic benchmarking problems dictionary. Keys are the
            problem names. Values contain information on the problem, including the
            solution value.
        results (dict): estimagic benchmarking results dictionary. Keys are
            tuples of the form (problem, algorithm), values are dictionaries of the
            collected information on the benchmark run, including 'criterion_history'
            and 'time_history'.
        problem_subset (list, optional): List of problem names. These must be a subset
            of the keys of the problems dictionary. If provided the convergence plot is
            only created for the problems specified in this list.
        algorithm_subset (list, optional): List of algorithm names. These must be a
            subset of the keys of the optimizer_options passed to run_benchmark. If
            provided only the convergence of the given algorithms are shown.
        n_cols (int): number of columns in the plot of grids. The number
            of rows is determined automatically.
        distance_measure (str): One of "criterion", "parameter_distance".
        monotone (bool): If True the best found criterion value so far is plotted.
            If False the particular criterion evaluation of that time is used.
        normalize_distance (bool): If True the progress is scaled by the total distance
            between the start value and the optimal value, i.e. 1 means the algorithm
            is as far from the solution as the start value and 0 means the algorithm
            has reached the solution value.
        runtime_measure (str): "n_evaluations" or "walltime".
        stopping_criterion (str): "x_and_y", "x_or_y", "x", "y" or None. If None, no
            clipping is done.
        x_precision (float or None): how close an algorithm must have gotten to the
            true parameter values (as percent of the Euclidean distance between start
            and solution parameters) before the criterion for clipping and convergence
            is fulfilled.
        y_precision (float or None): how close an algorithm must have gotten to the
            true criterion values (as percent of the distance between start
            and solution criterion value) before the criterion for clipping and
            convergence is fulfilled.
        combine_plots_in_grid (bool): decide whether to return a one
            figure containing subplots for each factor pair or a dictionary
            of individual plots. Default True.
        template (str): The template for the figure. Default is "plotly_white".

    Returns:
        plotly.Figure: The grid plot or dict of individual plots

    """
    # adding coloring palette
    palette = px.colors.qualitative.Plotly

    df, _ = create_convergence_histories(
        problems=problems,
        results=results,
        stopping_criterion=stopping_criterion,
        x_precision=x_precision,
        y_precision=y_precision,
    )

    # handle string provision for single problems / algorithms
    if isinstance(problem_subset, str):
        problem_subset = [problem_subset]
    if isinstance(algorithm_subset, str):
        algorithm_subset = [algorithm_subset]

    _check_only_allowed_subset_provided(problem_subset, df["problem"], "problem")
    _check_only_allowed_subset_provided(algorithm_subset, df["algorithm"], "algorithm")

    if problem_subset is not None:
        df = df[df["problem"].isin(problem_subset)]
    if algorithm_subset is not None:
        df = df[df["algorithm"].isin(algorithm_subset)]

    # plot configuration
    outcome = (
        f"{'monotone_' if monotone else ''}"
        + distance_measure
        + f"{'_normalized' if normalize_distance else ''}"
    )

    remaining_problems = df["problem"].unique()
    n_rows = int(np.ceil(len(remaining_problems) / n_cols))

    # pre - style plots labels
    y_labels = {
        "criterion": "Current Function Value",
        "monotone_criterion": "Best Function Value Found So Far",
        "criterion_normalized": "Share of Function Distance to Optimum<br>"
        + "Missing From Current Criterion Value",
        "monotone_criterion_normalized": "Share of Function Distance to Optimum<br>"
        + "Missing From Best So Far",
        "parameter_distance": "Distance Between Current and Optimal Parameters",
        "parameter_distance_normalized": "Share of Parameter Distance to Optimum<br>"
        + "Missing From Current Parameters",
        "monotone_parameter_distance_normalized": "Share of the Parameter Distance "
        + "to Optimum<br> Missing From the Best Parameters So Far",
        "monotone_parameter_distance": "Distance Between the Best Parameters So Far<br>"
        "and the Optimal Parameters",
    }

    x_labels = {
        "n_evaluations": "Number of Function Evaluations",
        "walltime": "Elapsed Time",
    }

    # container for individual plots
    g_list = []
    # container for titles
    titles = []

    # creating data traces for plotting faceted/individual plots
    # dropping usage of palette for algoritms, but use the built in pallete
    for prob_name in remaining_problems:

        g_ind = []  # container for data for traces in individual plot
        to_plot = df[df["problem"] == prob_name]

        for i, alg in enumerate(to_plot["algorithm"].unique()):

            temp = to_plot[to_plot["algorithm"] == alg]
            trace_1 = go.Scatter(
                x=temp[runtime_measure],
                y=temp[outcome],
                mode="lines",
                legendgroup=i,
                name=alg,
                line={"color": palette[i]},
            )
            g_ind.append(trace_1)

        if distance_measure == "criterion" and not normalize_distance:
            f_opt = problems[prob_name]["solution"]["value"]
            trace_2 = go.Scatter(
                y=[f_opt for i in to_plot[runtime_measure]],
                x=to_plot[runtime_measure],
                mode="lines",
                line={"color": palette[i + 1]},
                name="true solution",
                legendgroup=i + 1,
            )
            g_ind.append(trace_2)

        g_list.append(g_ind)
        titles.append(prob_name.replace("_", " ").title())

    # Plot with subplots
    if combine_plots_in_grid:
        g = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=titles,
            column_widths=[100] * n_cols,
            row_heights=[60] * n_rows,
        )
        for ind, (facet_row, facet_col) in enumerate(
            itertools.product(range(1, n_rows + 1), range(1, n_cols + 1))
        ):
            if ind + 1 > len(g_list):
                break  # if there are empty individual plots
            traces = g_list[ind]
            for trace in range(len(traces)):
                g.add_trace(traces[trace], row=facet_row, col=facet_col)
                # style axis labels
                g.update_yaxes(row=facet_row, col=facet_col, title=y_labels[outcome])
                g.update_xaxes(
                    row=facet_row, col=facet_col, title=x_labels[runtime_measure]
                )

        # deleting duplicates in legend
        g = clean_legend_duplicates(g)

        # setting template theme and size of the figure
        g.update_layout(
            template=template, height=300 * n_rows, width=500 * n_cols, title_x=0.5
        )
        out = g

    # Dictionary for individual plots
    if not combine_plots_in_grid:
        xaxis_title = [x_labels[runtime_measure] for ind in range(len(g_list))]
        yaxis_title = [y_labels[outcome] for ind in range(len(g_list))]

        ind_dict = create_ind_dict(
            g_list,
            titles,
            x_title=xaxis_title,
            y_title=yaxis_title,
            kws={"template": template, "height": 300, "width": 500, "title_x": 0.5},
        )

        out = ind_dict

    return out


def _check_only_allowed_subset_provided(subset, allowed, name):
    """Check if all entries of a proposed subset are in a Series.

    Args:
        subset (iterable or None): If None, no checks are performed. Else a ValueError
            is raised listing all entries that are not in the provided Series.
        allowed (iterable): allowed entries.
        name (str): name of the provided entries to use for the ValueError.

    Raises:
        ValueError

    """
    allowed = set(allowed)
    if subset is not None:
        missing = [entry for entry in subset if entry not in allowed]
        if missing:
            missing_msg = ""
            for entry in missing:
                proposed = propose_alternatives(entry, allowed)
                missing_msg += f"Invalid {name}: {entry}. Did you mean {proposed}?\n"
            raise ValueError(missing_msg)


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
