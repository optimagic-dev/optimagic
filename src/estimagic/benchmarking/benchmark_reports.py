import pandas as pd
from estimagic.benchmarking.process_benchmark_results import (
    process_benchmark_results,
)


def create_convergence_report(
    problems, results, *, stopping_criterion="y", x_precision=1e-4, y_precision=1e-4
):
    """Create a DataFrame with all information needed for the convergence report.

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
        pandas.DataFrame: columns are the optimizers and the dimensionality of the
            benchmark problems, index are the problems. For the optimizers columns,
            the values are strings that are either "success" "failed", or "error".
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

    convergence_report = converged_info.replace({True: "success", False: "failed"})

    for key, value in results.items():
        if isinstance(value["solution"], str):
            convergence_report.at[key] = "error"

    dim = {problem: len(problems[problem]["inputs"]["params"]) for problem in problems}
    convergence_report["dimensionality"] = convergence_report.index.map(dim)

    return convergence_report


def create_rank_report(
    problems,
    results,
    *,
    runtime_measure="n_evaluations",
    normalize_runtime=False,
    stopping_criterion="y",
    x_precision=1e-4,
    y_precision=1e-4,
):
    """Create a DataFrame with the ranks of the optimizers for a set of problems.

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
        normalize_runtime (bool): If True the runtime each algorithm needed for each
            problem is scaled by the time the fastest algorithm needed. If True, the
            resulting plot is what Moré and Wild (2009) called data profiles.
            Default is False.
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
        pandas.DataFrame: columns are the optimizers, index are the problems.
            The values are the ranks of the optimizers for each problem,
            0 means optimizers was the fastest, 1 means it was the second fastest
            and so on. If an optimizer did not converge on a problem, the value is
            "failed". If an optimizer did encounter an error during optimization,
            the value is "error".

    """
    histories, converged_info = process_benchmark_results(
        problems=problems,
        results=results,
        stopping_criterion=stopping_criterion,
        x_precision=x_precision,
        y_precision=y_precision,
    )
    scenarios = list({algo[1] for algo in results.keys()})

    success_info = converged_info.replace({True: "success", False: "failed"})
    for key, value in results.items():
        if isinstance(value["solution"], str):
            success_info.at[key] = "error"

    solution_times = histories.groupby(["problem", "algorithm"])[runtime_measure].max()

    if normalize_runtime:
        solution_times = solution_times.unstack()
        solution_times = solution_times.divide(solution_times.min(axis=1), axis=0)
        solution_times = solution_times.stack(dropna=False)
        solution_times.name = runtime_measure

    solution_times = solution_times.reset_index()
    solution_times["rank"] = (
        solution_times.groupby("problem")[runtime_measure].rank(
            method="dense", ascending=True
        )
        - 1
    ).astype("Int64")

    df_wide = solution_times.pivot(index="problem", columns="algorithm", values="rank")
    rank_report = df_wide.astype(str)[scenarios]
    rank_report[~converged_info] = success_info[scenarios]

    return rank_report


def create_traceback_report(results):
    """Create a DataFrame with the traceback for all problems that have not been solved.

    Args:
        results (dict): estimagic benchmarking results dictionary. Keys are
            tuples of the form (problem, algorithm), values are dictionaries of the
            collected information on the benchmark run, including 'criterion_history'
            and 'time_history'.

    Returns:
        pandas.DataFrame: columns are the optimizers, index are the problems.
            The values are the traceback of the optimizers for each problem
            the optimizers stopped with an error.

    """
    scenarios = list({algo[1] for algo in results.keys()})

    tracebacks = {}
    for scenario in scenarios:
        tracebacks[scenario] = {}

    for key, value in results.items():
        if isinstance(value["solution"], str):
            if key[1] in scenarios:
                tracebacks[key[1]][key[0]] = value["solution"]

    traceback_report = pd.DataFrame.from_dict(tracebacks, orient="columns")

    return traceback_report
