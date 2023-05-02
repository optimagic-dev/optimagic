import pandas as pd
from estimagic.benchmarking.process_benchmark_results import (
    process_benchmark_results,
)

from estimagic.visualization.profile_plot import create_solution_times


def convergence_report(
    problems, results, *, stopping_criterion="y", x_precision=1e-4, y_precision=1e-4
):
    """Create a DataFrame with convergence information for a set of problems.

    Args:
        problems (dict): estimagic benchmarking problems dictionary. Keys are the
            problem names. Values contain information on the problem, including the
            solution value.
        results (dict): estimagic benchmarking results dictionary. Keys are
            tuples of the form (problem, algorithm), values are dictionaries of the
            collected information on the benchmark run, including 'criterion_history'
            and 'time_history'.
        stopping_criterion (str): one of "x_and_y", "x_or_y", "x", "y". Determines
            how convergence is determined from the two precisions. Default is "y".
        x_precision (float or None): how close an algorithm must have gotten to the
            true parameter values (as percent of the Euclidean distance between start
            and solution parameters) before the criterion for clipping and convergence
            is fulfilled. Default is 1e-4.
        y_precision (float or None): how close an algorithm must have gotten to the
            true criterion values (as percent of the distance between start
            and solution criterion value) before the criterion for clipping and
            convergence is fulfilled. Default is 1e-4.

    Returns:
        pandas.DataFrame: columns are the algorithms and the dimensionality of the
            benchmark problems, indexes are the problems. For the algorithms columns,
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

    convergence_report = _get_success_info(results, converged_info)

    dim = {problem: len(problems[problem]["inputs"]["params"]) for problem in problems}
    convergence_report["dimensionality"] = convergence_report.index.map(dim)

    return convergence_report


def rank_report(
    problems,
    results,
    *,
    runtime_measure="n_evaluations",
    stopping_criterion="y",
    x_precision=1e-4,
    y_precision=1e-4,
):
    """Create a DataFrame with rank information for a set of problems.

    Args:
        problems (dict): estimagic benchmarking problems dictionary. Keys are the
            problem names. Values contain information on the problem, including the
            solution value.
        results (dict): estimagic benchmarking results dictionary. Keys are
            tuples of the form (problem, algorithm), values are dictionaries of the
            collected information on the benchmark run, including 'criterion_history'
            and 'time_history'.
        runtime_measure (str): "n_evaluations", "n_batches" or "walltime".
            This is the runtime until the desired convergence was reached by an
            algorithm. This is called performance measure by Moré and Wild (2009).
            Default is "n_evaluations".
        stopping_criterion (str): one of "x_and_y", "x_or_y", "x", "y". Determines
            how convergence is determined from the two precisions.
        x_precision (float or None): how close an algorithm must have gotten to the
            true parameter values (as percent of the Euclidean distance between start
            and solution parameters) before the criterion for clipping and convergence
            is fulfilled. Default is 1e-4.
        y_precision (float or None): how close an algorithm must have gotten to the
            true criterion values (as percent of the distance between start
            and solution criterion value) before the criterion for clipping and
            convergence is fulfilled. Default is 1e-4.

    Returns:
        pandas.DataFrame: columns are the algorithms, indexes are the problems.
            The values are the ranks of the algorithms for each problem,
            0 means the algorithm was the fastest, 1 means it was the second fastest
            and so on. If an algorithm did not converge on a problem, the value is
            "failed". If an algorithm did encounter an error during optimization,
            the value is "error".

    """
    histories, converged_info = process_benchmark_results(
        problems=problems,
        results=results,
        stopping_criterion=stopping_criterion,
        x_precision=x_precision,
        y_precision=y_precision,
    )

    solution_times = create_solution_times(histories, runtime_measure, converged_info)
    solution_times = solution_times.stack().reset_index()
    solution_times = solution_times.rename(
        columns={solution_times.columns[2]: runtime_measure}
    )

    success_info = _get_success_info(results, converged_info)

    solution_times["rank"] = (
        solution_times.groupby("problem")[runtime_measure].rank(
            method="dense", ascending=True
        )
        - 1
    ).astype("Int64")

    df_wide = solution_times.pivot(index="problem", columns="algorithm", values="rank")
    rank_report = df_wide.astype(str)
    rank_report[~converged_info] = success_info

    return rank_report


def traceback_report(results):
    """Create a DataFrame with tracebacks for all problems that have not been solved.

    Args:
        results (dict): estimagic benchmarking results dictionary. Keys are
            tuples of the form (problem, algorithm), values are dictionaries of the
            collected information on the benchmark run, including 'criterion_history'
            and 'time_history'.

    Returns:
        pandas.DataFrame: columns are the algorithms, indexes are the problems.
            The values are the tracebacks of the algorithms for problems where they
            stopped with an error.

    """
    algorithms = list({algo[1] for algo in results.keys()})

    tracebacks = {}
    for algo in algorithms:
        tracebacks[algo] = {}

    for key, value in results.items():
        if isinstance(value["solution"], str):
            tracebacks[key[1]][key[0]] = value["solution"]

    traceback_report = pd.DataFrame.from_dict(tracebacks, orient="columns")

    return traceback_report


def _get_success_info(results, converged_info):
    """Create a DataFrame with information on whether an algorithm succeeded or not.

    Args:
        results (dict): estimagic benchmarking results dictionary. Keys are
            tuples of the form (problem, algorithm), values are dictionaries of the
            collected information on the benchmark run, including 'criterion_history'
            and 'time_history'.
        converged_info (pandas.DataFrame): columns are the algorithms, indexes are the
            problems. The values are boolean and True when the algorithm arrived at
            the solution with the desired precision.

    Returns:
        pandas.DataFrame: columns are the algorithms, indexes are the problems.
           values are strings that are either "success", "failed", or "error".

    """
    success_info = converged_info.replace({True: "success", False: "failed"})

    for key, value in results.items():
        if isinstance(value["solution"], str):
            success_info.at[key] = "error"

    return success_info
