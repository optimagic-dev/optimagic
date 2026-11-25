from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, overload

import pandas as pd

from optimagic.benchmarking.benchmark_problem import BenchmarkProblem
from optimagic.benchmarking.benchmark_result import BenchmarkResult
from optimagic.benchmarking.process_benchmark_results import (
    StoppingCriterion,
    process_benchmark_results,
)
from optimagic.visualization.profile_plot import create_solution_times


def convergence_report(
    problems: Mapping[str, BenchmarkProblem],
    results: Mapping[tuple[str, str], BenchmarkResult],
    *,
    stopping_criterion: StoppingCriterion = "y",
    x_precision: float = 1e-4,
    y_precision: float = 1e-4,
) -> pd.DataFrame:
    """Create a DataFrame with convergence information for a set of problems.

    Args:
        problems: Dictionary that maps names to benchmark problems.
        results: Dictionary with benchmark results. Keys are tuples of the form
            (problem, algorithm).
        stopping_criterion: one of "x_and_y", "x_or_y", "x", "y". Determines
            how convergence is determined from the two precisions. Default is "y".
        x_precision: how close an algorithm must have gotten to the
            true parameter values (as percent of the Euclidean distance between start
            and solution parameters) before the criterion for clipping and convergence
            is fulfilled. Default is 1e-4.
        y_precision: how close an algorithm must have gotten to the
            true criterion values (as percent of the distance between start
            and solution criterion value) before the criterion for clipping and
            convergence is fulfilled. Default is 1e-4.

    Returns:
        DataFrame where indexes are the problems, columns are the algorithms and
        the dimensionality of the benchmark problems. For the algorithms column,
        the values are strings that are either "success", "failed", or "error".
        For the dimensionality column, the values denote the number of dimensions
        of the problem.

    """
    _, converged_info = process_benchmark_results(
        problems=problems,
        results=results,
        stopping_criterion=stopping_criterion,
        x_precision=x_precision,
        y_precision=y_precision,
    )

    report = _get_success_info(results, converged_info)
    report["dimensionality"] = report.index.map(_get_problem_dimensions(problems))

    return report


def rank_report(
    problems: Mapping[str, BenchmarkProblem],
    results: Mapping[tuple[str, str], BenchmarkResult],
    *,
    runtime_measure: Literal["n_evaluations", "n_batches", "walltime"] = (
        "n_evaluations"
    ),
    stopping_criterion: StoppingCriterion = "y",
    x_precision: float = 1e-4,
    y_precision: float = 1e-4,
) -> pd.DataFrame:
    """Create a DataFrame with rank information for a set of problems.

    Args:
        problems: Dictionary that maps names to benchmark problems.
        results: Dictionary with benchmark results. Keys are tuples of the form
            (problem, algorithm).
        runtime_measure: "n_evaluations", "n_batches" or "walltime".
            This is the runtime until the desired convergence was reached by an
            algorithm. This is called performance measure by Moré and Wild (2009).
            Default is "n_evaluations".
        stopping_criterion: one of "x_and_y", "x_or_y", "x", "y". Determines
            how convergence is determined from the two precisions.
        x_precision: how close an algorithm must have gotten to the
            true parameter values (as percent of the Euclidean distance between start
            and solution parameters) before the criterion for clipping and convergence
            is fulfilled. Default is 1e-4.
        y_precision: how close an algorithm must have gotten to the
            true criterion values (as percent of the distance between start
            and solution criterion value) before the criterion for clipping and
            convergence is fulfilled. Default is 1e-4.

    Returns:
        DataFrame where indexes are the problems, columns are the algorithms and the
        dimensionality of the problems. The values are the ranks of the algorithms
        for each problem, where 0 means the algorithm was the fastest, 1 means it
        was the second fastest and so on. If an algorithm did not converge on a
        problem, the value is "failed". If an algorithm did encounter an error
        during optimization, the value is "error".

    """
    histories, converged_info = process_benchmark_results(
        problems=problems,
        results=results,
        stopping_criterion=stopping_criterion,
        x_precision=x_precision,
        y_precision=y_precision,
    )

    solution_times = create_solution_times(
        histories, runtime_measure, converged_info, return_tidy=False
    )
    solution_times["rank"] = (
        solution_times.groupby("problem")[runtime_measure].rank(
            method="dense", ascending=True
        )
        - 1
    ).astype("Int64")

    success_info = _get_success_info(results, converged_info)

    df_wide = solution_times.pivot(index="problem", columns="algorithm", values="rank")
    report = df_wide.astype(str)
    report.columns.name = None

    report[~converged_info] = success_info
    report["dimensionality"] = report.index.map(_get_problem_dimensions(problems))

    return report


@overload
def traceback_report(
    problems: Mapping[str, BenchmarkProblem],
    results: Mapping[tuple[str, str], BenchmarkResult],
    return_type: Literal["dataframe"] = "dataframe",
) -> pd.DataFrame: ...


@overload
def traceback_report(
    problems: Mapping[str, BenchmarkProblem],
    results: Mapping[tuple[str, str], BenchmarkResult],
    return_type: Literal["text"],
) -> list[str]: ...


@overload
def traceback_report(
    problems: Mapping[str, BenchmarkProblem],
    results: Mapping[tuple[str, str], BenchmarkResult],
    return_type: Literal["markdown"],
) -> str: ...


@overload
def traceback_report(
    problems: Mapping[str, BenchmarkProblem],
    results: Mapping[tuple[str, str], BenchmarkResult],
    return_type: Literal["dict"],
) -> dict[tuple[str, str], str]: ...


def traceback_report(
    problems: Mapping[str, BenchmarkProblem],
    results: Mapping[tuple[str, str], BenchmarkResult],
    return_type: Literal["text", "markdown", "dict", "dataframe"] = "dataframe",
) -> list[str] | str | dict[tuple[str, str], str] | pd.DataFrame:
    """Create traceback report for all problems that have not been solved.

    Args:
        problems: Dictionary that maps names to benchmark problems.
        results: Dictionary with benchmark results. Keys are tuples of the form
            (problem, algorithm).
        return_type: either "text", "markdown", "dict" or "dataframe".
            If "text", the traceback report is returned as a string. If "markdown",
            it is a markdown string. If "dict", it is returned as a dictionary.
            If "dataframe", it is a tidy pandas DataFrame, where indexes are the
            algorithm and problem names, the columns are the tracebacks and the
            dimensionality of the problem. Default is "dataframe".

    Returns:
        The traceback report. If return_type is "text", the report is a list of
        strings. If "markdown", it is a formatted markdown string with algorithms and
        problem names as headers. If return_type is "dict", the report is a
        dictionary. If return_type is "dataframe", it is a tidy pandas DataFrame. In
        the latter case, indexes are the algorithm and problem names, the columns are
        the tracebacks and the dimensionality of the problems. The values are the
        tracebacks of the algorithms for problems where they stopped with an error.

    """
    tracebacks = {
        key: result.traceback
        for key, result in results.items()
        if result.traceback is not None
    }

    if return_type == "text":
        return list(tracebacks.values())

    if return_type == "markdown":
        markdown = "```python"
        for (problem_name, algorithm_name), traceback in tracebacks.items():
            if f"### {algorithm_name}" not in markdown:
                markdown += f"\n### {algorithm_name} \n"
            markdown += f"\n#### {problem_name} \n"
            markdown += f"\n{traceback} \n"
        markdown += "\n```"
        return markdown

    if return_type == "dict":
        return tracebacks

    if return_type == "dataframe":
        nested: dict[str, dict[str, str]] = {}
        for (problem_name, algorithm_name), traceback in tracebacks.items():
            nested.setdefault(algorithm_name, {})[problem_name] = traceback

        algorithms, problem_names, values = [], [], []
        for algorithm_name, problem_to_traceback in nested.items():
            for problem_name, traceback in problem_to_traceback.items():
                algorithms.append(algorithm_name)
                problem_names.append(problem_name)
                values.append(traceback)

        index = pd.MultiIndex.from_arrays(
            [algorithms, problem_names], names=["algorithm", "problem"]
        )
        report = pd.DataFrame({"traceback": values}, index=index)
        report["dimensionality"] = 0

        for problem_name, dim in _get_problem_dimensions(problems).items():
            if problem_name in report.index.get_level_values("problem"):
                report.loc[(slice(None), problem_name), "dimensionality"] = dim

        return report

    raise ValueError(
        f"return_type {return_type} is not supported. Must be one of "
        f"'text', 'markdown', 'dict' or 'dataframe'."
    )


def _get_success_info(
    results: Mapping[tuple[str, str], BenchmarkResult], converged_info: pd.DataFrame
) -> pd.DataFrame:
    """Create a DataFrame with information on whether an algorithm succeeded or not.

    Args:
        results: Dictionary with benchmark results. Keys are tuples of the form
            (problem, algorithm).
        converged_info: columns are the algorithms, indexes are the
            problems. The values are boolean and True when the algorithm arrived at
            the solution with the desired precision.

    Returns:
        DataFrame where indexes are the problems, columns are the algorithms. Values
        are strings that are either "success", "failed", or "error".

    """
    success_info = converged_info.replace({True: "success", False: "failed"})

    for key, value in results.items():
        if value.traceback is not None:
            success_info.at[key] = "error"

    return success_info


def _get_problem_dimensions(problems: Mapping[str, BenchmarkProblem]) -> dict[str, int]:
    """Get the number of parameters of each problem."""
    return {name: problem.n_params for name, problem in problems.items()}
