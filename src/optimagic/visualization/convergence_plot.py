from typing import Any, Literal

import numpy as np
import pandas as pd

from optimagic.benchmarking.process_benchmark_results import (
    process_benchmark_results,
)
from optimagic.config import DEFAULT_PALETTE
from optimagic.utilities import propose_alternatives
from optimagic.visualization.backends import grid_line_plot, line_plot
from optimagic.visualization.plotting_utilities import LineData, get_palette_cycle

BACKEND_TO_CONVERGENCE_PLOT_LEGEND_PROPERTIES: dict[str, dict[str, Any]] = {
    "plotly": {},
    "matplotlib": {"loc": "outside right upper", "fontsize": "x-small"},
    "bokeh": {
        "location": "top_right",
        "place": "right",
        "label_text_font_size": "8pt",
    },
    "altair": {"orient": "right"},
}

BACKEND_TO_CONVERGENCE_PLOT_MARGIN_PROPERTIES: dict[str, dict[str, int]] = {
    "plotly": {"l": 10, "r": 10, "t": 30, "b": 10},
    # "matplotlib": handles margins automatically via constrained layout
}

OUTCOME_TO_CONVERGENCE_PLOT_YLABEL: dict[str, str] = {
    "criterion": "Current Function Value",
    "monotone_criterion": "Best Function Value Found So Far",
    "criterion_normalized": (
        "Share of Function Distance to Optimum{linebreak}"
        "Missing From Current Criterion Value"
    ),
    "monotone_criterion_normalized": (
        "Share of Function Distance to Optimum{linebreak}Missing From Best So Far"
    ),
    "parameter_distance": "Distance Between Current and{linebreak}Optimal Parameters",
    "parameter_distance_normalized": (
        "Share of Parameter Distance to Optimum{linebreak}"
        "Missing From Current Parameters"
    ),
    "monotone_parameter_distance_normalized": (
        "Share of Parameter Distance to Optimum{linebreak}"
        "Missing From the Best Parameters So Far"
    ),
    "monotone_parameter_distance": (
        "Distance Between the Best Parameters{linebreak}"
        "So Far and the Optimal Parameters"
    ),
}

RUNTIME_MEASURE_TO_CONVERGENCE_PLOT_XLABEL: dict[str, str] = {
    "n_evaluations": "Number of Function Evaluations",
    "walltime": "Elapsed Time",
    "n_batches": "Number of Batches",
}


def convergence_plot(
    problems: dict[str, dict[str, Any]],
    results: dict[tuple[str, str], dict[str, Any]],
    *,
    problem_subset: list[str] | None = None,
    algorithm_subset: list[str] | None = None,
    n_cols: int = 2,
    distance_measure: Literal["criterion", "parameter_distance"] = "criterion",
    monotone: bool = True,
    normalize_distance: bool = True,
    runtime_measure: Literal[
        "n_evaluations", "walltime", "n_batches"
    ] = "n_evaluations",
    stopping_criterion: Literal["x", "y", "x_and_y", "x_or_y"] = "y",
    x_precision: float = 1e-4,
    y_precision: float = 1e-4,
    combine_plots_in_grid: bool = True,
    backend: Literal["plotly", "matplotlib", "bokeh", "altair"] = "plotly",
    template: str | None = None,
    palette: list[str] | str = DEFAULT_PALETTE,
) -> Any:
    """Plot convergence of optimizers for a set of problems.

    This creates a grid of plots, showing the convergence of the different
    algorithms on each problem. The faster a line falls, the faster the algorithm
    improved on the problem. The algorithm converged where its line reaches 0
    (if normalize_distance is True) or the horizontal line labeled "true solution".

    Each plot shows on the x axis the runtime_measure, which can be walltime, number
    of evaluations or number of batches. Each algorithm's convergence is a line in the
    plot. Convergence can be measured by the criterion value of the particular
    time/evaluation. The convergence can be made monotone (i.e. always taking the bast
    value so far) or normalized such that the distance from the start to the true
    solution is one.

    Args:
        problems: optimagic benchmarking problems dictionary. Keys are the problem
            names. Values contain information on the problem, including the solution
            value.
        results: optimagic benchmarking results dictionary. Keys are tuples of the form
            (problem, algorithm), values are dictionaries of the collected information
            on the benchmark run, including 'criterion_history' and 'time_history'.
        problem_subset: List of problem names. These must be a subset of the keys of the
            problems dictionary. If provided the convergence plot is only created for
            the problems specified in this list.
        algorithm_subset: List of algorithm names. These must be a subset of the keys of
            the optimizer_options passed to run_benchmark. If provided only the
            convergence of the given algorithms are shown.
        n_cols: number of columns in the plot of grids. The number of rows is determined
            automatically.
        distance_measure: One of "criterion", "parameter_distance".
        monotone: If True the best found criterion value so far is plotted.
            If False the particular criterion evaluation of that time is used.
        normalize_distance: If True the progress is scaled by the total distance between
            the start value and the optimal value, i.e. 1 means the algorithm is as far
            from the solution as the start value and 0 means the algorithm has reached
            the solution value.
        runtime_measure: This is the runtime until the desired convergence was reached
            by an algorithm.
        stopping_criterion: Determines how convergence is determined from the two
            precisions. To effectively disable convergence, set `x_precision` and/or
            `y_precision` to very small values (or 0).
        x_precision: how close an algorithm must have gotten to the true parameter
            values (as percent of the Euclidean distance between start and solution
            parameters) before the criterion for clipping and convergence is fulfilled.
        y_precision: how close an algorithm must have gotten to the true criterion
            values (as percent of the distance between start and solution criterion
            value) before the criterion for clipping and convergence is fulfilled.
        combine_plots_in_grid: Whether to return a single figure containing subplots
            for each factor pair or a dictionary of individual plots. Default is True.
        backend: The backend to use for plotting. Default is "plotly".
        template: The template for the figure. If not specified, the default template of
            the backend is used. For the 'bokeh' and 'altair' backends, this changes the
            global theme, which affects all plots from that backend in the session.
        palette: The coloring palette for traces. Default is the D3 qualitative palette.

    Returns:
        The figure object containing the convergence plot if `combine_plots_in_grid` is
            True. Otherwise, a dictionary mapping problem names to their respective
            figure objects is returned.

    """
    # ==================================================================================
    # Process inputs

    df, _ = process_benchmark_results(
        problems=problems,
        results=results,
        stopping_criterion=stopping_criterion,
        x_precision=x_precision,
        y_precision=y_precision,
    )

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

    # ==================================================================================
    # Extract backend-agnostic plotting data

    outcome = (
        f"{'monotone_' if monotone else ''}"
        + distance_measure
        + f"{'_normalized' if normalize_distance else ''}"
    )

    lines_list, titles = _extract_convergence_plot_lines(
        df=df,
        problems=problems,
        runtime_measure=runtime_measure,
        outcome=outcome,
        palette=palette,
        combine_plots_in_grid=combine_plots_in_grid,
        backend=backend,
    )

    n_rows = int(np.ceil(len(lines_list) / n_cols))

    # ==================================================================================
    # Generate the figure

    if combine_plots_in_grid:
        fig = grid_line_plot(
            lines_list,
            backend=backend,
            n_rows=n_rows,
            n_cols=n_cols,
            titles=titles,
            xlabels=(
                [RUNTIME_MEASURE_TO_CONVERGENCE_PLOT_XLABEL[runtime_measure]]
                * len(lines_list)
            ),
            ylabels=[OUTCOME_TO_CONVERGENCE_PLOT_YLABEL[outcome]] * len(lines_list),
            template=template,
            height=320 * n_rows,
            width=500 * n_cols,
            legend_properties=BACKEND_TO_CONVERGENCE_PLOT_LEGEND_PROPERTIES.get(
                backend, None
            ),
            margin_properties=BACKEND_TO_CONVERGENCE_PLOT_MARGIN_PROPERTIES.get(
                backend, None
            ),
        )

        return fig

    else:
        fig_dict = {}

        for i, subplot_lines in enumerate(lines_list):
            fig = line_plot(
                subplot_lines,
                backend=backend,
                title=titles[i],
                xlabel=RUNTIME_MEASURE_TO_CONVERGENCE_PLOT_XLABEL[runtime_measure],
                ylabel=OUTCOME_TO_CONVERGENCE_PLOT_YLABEL[outcome],
                template=template,
                height=320,
                width=500,
                legend_properties=BACKEND_TO_CONVERGENCE_PLOT_LEGEND_PROPERTIES.get(
                    backend, None
                ),
                margin_properties=BACKEND_TO_CONVERGENCE_PLOT_MARGIN_PROPERTIES.get(
                    backend, None
                ),
            )

            key = titles[i].replace(" ", "_").lower()
            fig_dict[key] = fig

        return fig_dict


def _extract_convergence_plot_lines(
    df: pd.DataFrame,
    problems: dict[str, dict[str, Any]],
    runtime_measure: str,
    outcome: str,
    palette: list[str] | str,
    combine_plots_in_grid: bool,
    backend: str,
) -> tuple[list[list[LineData]], list[str]]:
    lines_list = []  # container for all subplots
    titles = []

    for i, (_prob_name, _prob_data) in enumerate(df.groupby("problem", sort=False)):
        prob_name = str(_prob_name)
        subplot_lines = []  # container for data of traces in individual subplot
        palette_cycle = get_palette_cycle(palette)

        if runtime_measure == "n_batches":
            to_plot = (
                _prob_data.groupby(["algorithm", runtime_measure]).min().reset_index()
            )
        else:
            to_plot = _prob_data

        show_in_legend = True
        if combine_plots_in_grid:
            # If combining plots, only show in legend of first subplot
            # For 'bokeh' backend, show in legend for all subplots
            # as it does not support single legend on grid plots.
            # See: https://github.com/bokeh/bokeh/issues/7607
            show_in_legend = (i == 0) or (backend == "bokeh")

        for alg, group in to_plot.groupby("algorithm", sort=False):
            line_data = LineData(
                x=group[runtime_measure].to_numpy(),
                y=group[outcome].to_numpy(),
                name=str(alg),
                color=next(palette_cycle),
                # if combining plots, only show legend in first subplot
                show_in_legend=show_in_legend,
            )
            subplot_lines.append(line_data)

        if outcome in ("criterion", "monotone_criterion"):
            f_opt = problems[prob_name]["solution"]["value"]
            line_data = LineData(
                x=to_plot[runtime_measure].to_numpy(),
                y=np.full(to_plot[runtime_measure].shape, f_opt),
                name="true solution",
                color=next(palette_cycle),
                # if combining plots, only show legend in first subplot
                show_in_legend=show_in_legend,
            )
            subplot_lines.append(line_data)

        lines_list.append(subplot_lines)
        titles.append(prob_name.replace("_", " ").title())

    return lines_list, titles


def _check_only_allowed_subset_provided(
    subset: list[str] | None, allowed: pd.Series | list[str], name: str
) -> None:
    """Check if all entries of a proposed subset are in a Series.

    Args:
        subset: If None, no checks are performed. Else a ValueError is raised listing
            all entries that are not in the provided Series.
        allowed: allowed entries.
        name: name of the provided entries to use for the ValueError.

    Raises:
        ValueError

    """
    allowed_set = set(allowed)
    if subset is not None:
        missing = [entry for entry in subset if entry not in allowed_set]
        if missing:
            missing_msg = ""
            for entry in missing:
                proposed = propose_alternatives(entry, allowed_set)
                missing_msg += f"Invalid {name}: {entry}. Did you mean {proposed}?\n"
            raise ValueError(missing_msg)
