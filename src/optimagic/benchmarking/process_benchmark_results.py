"""Process benchmark results for plots and reports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from optimagic.benchmarking.benchmark_problem import (
    BenchmarkProblem,
    BenchmarkSolution,
)
from optimagic.benchmarking.benchmark_result import BenchmarkResult

StoppingCriterion = Literal["x_and_y", "x_or_y", "x", "y"]


def process_benchmark_results(
    problems: Mapping[str, BenchmarkProblem],
    results: Mapping[tuple[str, str], BenchmarkResult],
    stopping_criterion: StoppingCriterion | None,
    x_precision: float = 1e-4,
    y_precision: float = 1e-4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create tidy DataFrame with all information needed for the benchmarking plots.

    The results of each problem are compared to a reference solution, see
    ``get_reference_solutions``.

    Args:
        problems: Dictionary that maps names to benchmark problems.
        results: Dictionary with benchmark results. Keys are tuples of the form
            (problem, algorithm).
        stopping_criterion: one of "x_and_y", "x_or_y", "x", "y", or None.
            Determines how convergence is determined from the two precisions.
            If None, no convergence criterion is applied.
        x_precision: how close an algorithm must have gotten to the
            true parameter values (as percent of the Euclidean distance between start
            and solution parameters) before the criterion for clipping and convergence
            is fulfilled. Default is 1e-4.
        y_precision: how close an algorithm must have gotten to the
            true criterion values (as percent of the distance between start
            and solution criterion value) before the criterion for clipping and
            convergence is fulfilled. Default is 1e-4.

    Returns:
        Tuple of two DataFrames. The first is a tidy DataFrame with the columns

        - problem
        - algorithm
        - n_evaluations
        - n_batches
        - walltime
        - criterion
        - criterion_normalized
        - monotone_criterion
        - monotone_criterion_normalized
        - parameter_distance
        - parameter_distance_normalized
        - monotone_parameter_distance
        - monotone_parameter_distance_normalized

        The second has the problems as index and the algorithms as columns. Its values
        are True if the algorithm converged on the problem.

    """
    references = get_reference_solutions(problems, results)

    histories = []
    infos = []

    for (problem_name, algorithm_name), result in results.items():
        history, is_converged = _process_one_result(
            problem=problems[problem_name],
            result=result,
            reference=references[problem_name],
            stopping_criterion=stopping_criterion,
            x_precision=x_precision,
            y_precision=y_precision,
        )
        history["problem"] = problem_name
        history["algorithm"] = algorithm_name
        histories.append(history)

        info = {
            "problem": problem_name,
            "algorithm": algorithm_name,
            "is_converged": is_converged,
        }
        infos.append(info)

    tidy_histories = pd.concat(histories, ignore_index=True)
    converged_info = cast(
        pd.DataFrame, pd.DataFrame(infos).set_index(["problem", "algorithm"]).unstack()
    )
    converged_info.columns = pd.Index([tup[1] for tup in converged_info.columns])

    return tidy_histories, converged_info


def get_reference_solutions(
    problems: Mapping[str, BenchmarkProblem],
    results: Mapping[tuple[str, str], BenchmarkResult],
) -> dict[str, BenchmarkSolution]:
    """Get the solutions to which benchmark results are compared.

    The reference solution of a problem is

    - its stored solution if that solution is exact.
    - the best function value any optimizer in results found if the problem has no
      stored solution. The parameters of the reference solution are then None, i.e.
      convergence can only be measured in terms of function values.
    - the better one of the two if the stored solution is only the best known value.
      If an optimizer found a better value and the stored solution has parameters, the
      parameters at which the better value was found become the reference parameters.

    Args:
        problems: Dictionary that maps names to benchmark problems.
        results: Dictionary with benchmark results. Keys are tuples of the form
            (problem, algorithm).

    Returns:
        Dictionary that maps the names of all problems that occur in results to their
        reference solutions.

    """
    best_found: dict[str, tuple[float, NDArray[np.float64]]] = {}
    for (problem_name, _), result in results.items():
        fun_history = np.asarray(result.fun_history, dtype=np.float64)
        candidates = np.where(np.isnan(fun_history), np.inf, fun_history)
        idx = int(np.argmin(candidates))
        value = float(candidates[idx])
        if problem_name not in best_found or value < best_found[problem_name][0]:
            best_found[problem_name] = (value, np.asarray(result.params_history)[idx])

    references = {}
    for problem_name, (best_fun, best_params) in best_found.items():
        solution = problems[problem_name].solution
        if solution is None:
            references[problem_name] = BenchmarkSolution(fun=best_fun)
        elif solution.exact or solution.fun <= best_fun:
            references[problem_name] = solution
        else:
            params = None if solution.params is None else best_params
            references[problem_name] = BenchmarkSolution(fun=best_fun, params=params)

    return references


def _process_one_result(
    problem: BenchmarkProblem,
    result: BenchmarkResult,
    reference: BenchmarkSolution,
    stopping_criterion: StoppingCriterion | None,
    x_precision: float,
    y_precision: float,
) -> tuple[pd.DataFrame, bool]:
    # input processing
    assert isinstance(x_precision, float)
    assert isinstance(y_precision, float)

    # extract information
    _params_hist = np.asarray(result.params_history, dtype=np.float64)
    _solution_crit = reference.fun
    _start_crit = problem.start_fun
    _solution_x = reference.params
    _start_x = problem.start_params
    if _solution_x is not None and not np.isfinite(_solution_x).all():
        _solution_x = None

    # calculate the different transformations of criterion values
    crit_hist = np.array(result.fun_history, dtype=np.float64)
    if reference.exact:
        # values below an exact solution can only be caused by numerical imprecision
        crit_hist = np.clip(crit_hist, _solution_crit, np.inf)
    monotone_crit_hist = np.minimum.accumulate(crit_hist)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized_crit_hist = (crit_hist - _solution_crit) / (
            _start_crit - _solution_crit
        )
        normalized_monotone_crit_hist = (monotone_crit_hist - _solution_crit) / (
            _start_crit - _solution_crit
        )

    # calculate the different versions of params distance if we have a solution
    if _solution_x is not None:
        _needed_step = np.linalg.norm(_solution_x - _start_x)
        params_dist = np.linalg.norm(_params_hist - _solution_x, axis=1)
        monotone_params_dist = np.minimum.accumulate(params_dist)
        with np.errstate(divide="ignore", invalid="ignore"):
            params_dist_normalized = params_dist / _needed_step
            monotone_params_dist_normalized = monotone_params_dist / _needed_step
    else:
        params_dist = np.full(len(_params_hist), np.nan)
        monotone_params_dist = np.full(len(_params_hist), np.nan)
        params_dist_normalized = np.full(len(_params_hist), np.nan)
        monotone_params_dist_normalized = np.full(len(_params_hist), np.nan)

    # put everything together in a dict
    out_dict = {
        "n_evaluations": np.arange(len(crit_hist)),
        "n_batches": np.asarray(result.batches_history),
        "walltime": np.asarray(result.time_history),
        "criterion": crit_hist,
        "criterion_normalized": normalized_crit_hist,
        "monotone_criterion": monotone_crit_hist,
        "monotone_criterion_normalized": normalized_monotone_crit_hist,
        "parameter_distance": params_dist,
        "monotone_parameter_distance": monotone_params_dist,
        "parameter_distance_normalized": params_dist_normalized,
        "monotone_parameter_distance_normalized": monotone_params_dist_normalized,
    }

    # calculate at which iteration the problem has been solved
    is_converged = False
    if stopping_criterion is not None:
        is_converged_x, x_idx = _check_convergence(params_dist_normalized, x_precision)
        is_converged_y, y_idx = _check_convergence(normalized_crit_hist, y_precision)

        flag_aggregators = {
            "x": lambda x, y: x,
            "y": lambda x, y: y,
            "x_and_y": lambda x, y: x and y,
            "x_or_y": lambda x, y: x or y,
        }

        is_converged = flag_aggregators[stopping_criterion](
            x=is_converged_x, y=is_converged_y
        )

        if is_converged:
            idx_aggregators = {
                "x": lambda x, y: x,
                "y": lambda x, y: y,
                "x_and_y": _aggregate_idxs_with_and,
                "x_or_y": _aggregate_idxs_with_or,
            }
            solution_idx = idx_aggregators[stopping_criterion](x=x_idx, y=y_idx)
            if solution_idx is not None:
                out_dict = {k: v[: solution_idx + 1] for k, v in out_dict.items()}

    # create a DataFrame and add metadata
    out = pd.DataFrame(out_dict)

    return out, is_converged


def _check_convergence(
    values: NDArray[np.float64], threshold: float
) -> tuple[bool, int | None]:
    boo = values <= threshold
    if boo.any():
        return True, int(np.argmax(boo))
    return False, None


def _aggregate_idxs_with_and(x: int | None, y: int | None) -> int | None:
    if x is None or y is None:
        out = None
    else:
        out = max(x, y)
    return out


def _aggregate_idxs_with_or(x: int | None, y: int | None) -> int | None:
    if x is None and y is None:
        out = None
    elif x is None:
        out = y
    elif y is None:
        out = x
    else:
        out = min(x, y)
    return out
