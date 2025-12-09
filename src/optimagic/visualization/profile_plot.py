import itertools
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from optimagic.benchmarking.process_benchmark_results import (
    process_benchmark_results,
)
from optimagic.config import DEFAULT_PALETTE
from optimagic.visualization.backends import line_plot
from optimagic.visualization.plotting_utilities import LineData, get_palette_cycle

BACKEND_TO_PROFILE_PLOT_LEGEND_PROPERTIES: dict[str, dict[str, Any]] = {
    "plotly": {"title": {"text": "algorithm"}},
    "matplotlib": {
        "loc": "outside right upper",
        "fontsize": "x-small",
        "title": "algorithm",
    },
    "bokeh": {
        "location": "top_right",
        "place": "right",
        "label_text_font_size": "8pt",
        "title": "algorithm",
    },
    "altair": {"orient": "right", "title": "algorithm"},
}

BACKEND_TO_PROFILE_PLOT_MARGIN_PROPERTIES: dict[str, dict[str, Any]] = {
    "plotly": {"l": 10, "r": 10, "t": 30, "b": 30},
    # "matplotlib": handles margins automatically via constrained layout
}


def profile_plot(
    problems: dict[str, dict[str, Any]],
    results: dict[tuple[str, str], dict[str, Any]],
    *,
    runtime_measure: Literal[
        "walltime", "n_evaluations", "n_batches"
    ] = "n_evaluations",
    normalize_runtime: bool = False,
    stopping_criterion: Literal["x", "y", "x_and_y", "x_or_y"] = "y",
    x_precision: float = 1e-4,
    y_precision: float = 1e-4,
    backend: Literal["plotly", "matplotlib", "bokeh", "altair"] = "plotly",
    template: str | None = None,
    palette: list[str] | str = DEFAULT_PALETTE,
) -> Any:
    """Compare optimizers over a problem set.

    This plot answers the question: What percentage of problems can each algorithm
    solve within a certain runtime budget?

    The runtime budget is plotted on the x axis and the share of problems each
    algorithm solved on the y axis.

    Thus, algorithms that are very specialized and perform well on some share of
    problems but are not able to solve more problems with a larger computational budget
    will have steep increases and then flat lines. Algorithms that are robust but slow,
    will have low shares in the beginning but reach very high.

    Note that failing to converge according to the given stopping_criterion and
    precisions is scored as needing an infinite computational budget.

    For details, see the description of performance and data profiles by
    Moré and Wild (2009).

    Args:
        problems: A dictionary where keys are the problem names. Values contain
            information on the problem, including the solution value.
        results: A dictionary where keys are tuples of the form (problem, algorithm),
            values are dictionaries of the collected information on the benchmark
            run, including 'criterion_history' and 'time_history'.
        runtime_measure: This is the runtime until the desired convergence was reached
            by an algorithm. This is called performance measure by Moré and Wild (2009).
        normalize_runtime: If True the runtime each algorithm needed for each problem is
            scaled by the time the fastest algorithm needed. If True, the resulting plot
            is what Moré and Wild (2009) called data profiles.
        stopping_criterion: Determines how convergence is determined from the two
            precisions.
        x_precision: How close an algorithm must have gotten to the true parameter
            values (as percent of the Euclidean distance between start and solution
            parameters) before the criterion for clipping and convergence is fulfilled.
        y_precision: How close an algorithm must have gotten to the true criterion
            values (as percent of the distance between start and solution criterion
            value) before the criterion for clipping and convergence is fulfilled.
        backend: The backend to use for plotting. Default is "plotly".
        template: The template for the figure. If not specified, the default template of
            the backend is used. For the 'bokeh' backend, this changes the global theme,
            which affects all Bokeh plots in the session.
        palette: The coloring palette for traces. Default is the D3 qualitative palette.

    Returns:
        The figure object containing the profile plot.

    """
    # ==================================================================================
    # Process inputs

    palette_cycle = get_palette_cycle(palette)

    if stopping_criterion is None:
        raise ValueError(
            "You must specify a stopping criterion for the performance plot. "
        )
    if runtime_measure not in ["walltime", "n_evaluations", "n_batches"]:
        raise ValueError(
            "Only 'walltime', 'n_evaluations' or 'n_batches' are allowed as "
            f"runtime_measure. You specified '{runtime_measure}'."
        )

    # ==================================================================================
    # Extract backend-agnostic plotting data from benchmark results

    df, converged_info = process_benchmark_results(
        problems=problems,
        results=results,
        stopping_criterion=stopping_criterion,
        x_precision=x_precision,
        y_precision=y_precision,
    )

    solution_times = create_solution_times(
        df,
        runtime_measure=runtime_measure,
        converged_info=converged_info,
    )

    lines = _extract_profile_plot_lines(
        solution_times=solution_times,
        normalize_runtime=normalize_runtime,
        converged_info=converged_info,
        palette_cycle=palette_cycle,
    )

    # ==================================================================================
    # Generate the figure

    fig = line_plot(
        lines,
        backend=backend,
        xlabel=_get_profile_plot_xlabel(runtime_measure, normalize_runtime),
        ylabel="Share of Problems Solved",
        template=template,
        height=300,
        width=500,
        legend_properties=BACKEND_TO_PROFILE_PLOT_LEGEND_PROPERTIES.get(backend, None),
        margin_properties=BACKEND_TO_PROFILE_PLOT_MARGIN_PROPERTIES.get(backend, None),
        horizontal_line=1.0,
    )

    return fig


def _extract_profile_plot_lines(
    solution_times: pd.DataFrame,
    normalize_runtime: bool,
    converged_info: pd.DataFrame,
    palette_cycle: "itertools.cycle[str]",
) -> list[LineData]:
    """Extract lines for profile plot from data.

    Args:
        solution_times: A DataFrame where columns are the names of the algorithms,
            indexes are the problems. Values are performance measures.
        normalize_runtime: If True the runtime each algorithm needed for each problem is
            scaled by the time the fastest algorithm needed.
        converged_info: A DataFrame where columns are the names of the algorithms,
            indexes are the problems. The values are boolean and True when the algorithm
            arrived at the solution with the desired precision.
        palette_cycle: Cycle of colors for plotting.

    Returns:
        A list of data objects containing data for each line of the profile plot.

    """
    if normalize_runtime:
        solution_times = solution_times.divide(solution_times.min(axis=1), axis=0)
        solution_times[~converged_info] = np.inf

    alphas = _determine_alpha_grid(solution_times)
    for_each_alpha = pd.concat(
        {alpha: solution_times <= alpha for alpha in alphas},
        names=["alpha"],
    )
    performance_profiles = for_each_alpha.groupby("alpha").mean().stack().reset_index()

    lines: list[LineData] = []

    for algorithm, data in performance_profiles.groupby("algorithm"):
        line_data = LineData(
            x=data["alpha"].to_numpy(),
            y=data[0].to_numpy(),
            name=str(algorithm),
            color=next(palette_cycle),
        )
        lines.append(line_data)

    return lines


def create_solution_times(
    df: pd.DataFrame,
    runtime_measure: Literal["walltime", "n_evaluations", "n_batches"],
    converged_info: pd.DataFrame,
    return_tidy: bool = True,
) -> pd.DataFrame:
    """Find the solution time for each algorithm and problem.

    Args:
        df: A DataFrame which contains 'problem', 'algorithm' and 'runtime_measure'
            as columns.
        runtime_measure: This is the runtime until the desired convergence was reached
            by an algorithm. This is called performance measure by Moré and Wild (2009).
        converged_info: A DataFrame where columns are the names of the algorithms,
            indexes are the problems. The values are boolean and True when the algorithm
            arrived at the solution with the desired precision.
        return_tidy: If True, the resulting DataFrame will be a tidy DataFrame
            with problem and algorithm as indexes and runtime_measure as column.
            If False, the resulting DataFrame will have problem, algorithm and
            runtime_measure as columns.

    Returns:
        A DataFrame. If return_tidy is True, indexes are the problems, columns are the
            algorithms. If return_tidy is False, columns are problem, algorithm and
            runtime_measure. The values are either the number of evaluations or the
            walltime each algorithm needed to achieve the desired precision. If the
            desired precision was not achieved the value is set to np.inf.

    """
    solution_times = (
        df.groupby(["problem", "algorithm"])[runtime_measure].max().unstack()
    )
    # We convert the dtype to float to support the use of np.inf
    solution_times = solution_times.astype(float).where(converged_info, other=np.inf)

    if not return_tidy:
        solution_times = solution_times.stack().reset_index()
        solution_times = solution_times.rename(
            columns={solution_times.columns[2]: runtime_measure}
        )

    return solution_times


def _determine_alpha_grid(solution_times: pd.DataFrame) -> list[np.float64]:
    switch_points = _find_switch_points(solution_times=solution_times)

    point_to_right = switch_points[-1] * 1.05
    extended_switch_points = np.append(switch_points, point_to_right)
    mid_points = (extended_switch_points[:-1] + extended_switch_points[1:]) / 2
    alphas = sorted(np.append(extended_switch_points, mid_points))
    return alphas


def _find_switch_points(solution_times: pd.DataFrame) -> NDArray[np.float64]:
    """Determine the switch points of the performance profiles.

    Args:
        solution_times: A DataFrame where columns are the names of the algorithms,
            indexes are the problems. Values are performance measures. They can be
            either float, when normalize_runtime was True or int when the
            runtime_measure are not normalized function evaluations or datetime when the
            not normalized walltime is used.

    Returns:
        A sorted array of switching points.

    """
    switch_points = np.unique(solution_times.values)
    if pd.api.types.is_float_dtype(switch_points):
        switch_points += 1e-10
    switch_points = switch_points[np.isfinite(switch_points)]
    return switch_points


def _get_profile_plot_xlabel(runtime_measure: str, normalize_runtime: bool) -> str:
    # The '{linebreak}' placeholder is replaced with the backend-specific line break
    # in the corresponding plotting function.

    if normalize_runtime:
        runtime_measure_to_xlabel = {
            "walltime": (
                "Multiple of Minimal Wall Time{linebreak}Needed to Solve the Problem"
            ),
            "n_evaluations": (
                "Multiple of Minimal Number of Function Evaluations"
                "{linebreak}Needed to Solve the Problem"
            ),
            "n_batches": (
                "Multiple of Minimal Number of Batches"
                "{linebreak}Needed to Solve the Problem"
            ),
        }
    else:
        runtime_measure_to_xlabel = {
            "walltime": "Wall Time Needed to Solve the Problem",
            "n_evaluations": "Number of Function Evaluations",
            "n_batches": "Number of Batches",
        }

    return runtime_measure_to_xlabel[runtime_measure]
