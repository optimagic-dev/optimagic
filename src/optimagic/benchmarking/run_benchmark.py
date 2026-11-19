"""Run optimizers on sets of benchmark problems.

TO-DO:
- Add other benchmark sets:
    - finish medium scale problems from https://arxiv.org/pdf/1710.11005.pdf, Page 34.
    - add scalar problems from https://github.com/AxelThevenot
- Add option for deterministic noise or wiggle.

"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeVar, cast

import numpy as np

from optimagic.batch_evaluators import process_batch_evaluator
from optimagic.benchmarking.benchmark_problem import (
    BenchmarkProblem,
    aggregate_function_value,
)
from optimagic.benchmarking.benchmark_result import BenchmarkResult
from optimagic.optimization.algorithm import Algorithm
from optimagic.optimization.create_optimization_problem import (
    pre_process_user_algorithm,
)
from optimagic.optimization.optimize import minimize
from optimagic.optimization.optimize_options import OptimizeOptions
from optimagic.optimization.optimize_result import OptimizeResult
from optimagic.typing import BatchEvaluator, BatchEvaluatorLiteral

AlgorithmT = TypeVar("AlgorithmT", bound=Algorithm)

# Mapping and Sequence are used instead of dict and list such that, for example, a
# dictionary whose values are all algorithm classes is accepted by type checkers.
OptimizeOptionsType = (
    Mapping[
        str,
        str | Algorithm | type[Algorithm] | OptimizeOptions | dict[str, Any],
    ]
    | Sequence[str | Algorithm | type[Algorithm]]
)


def run_benchmark(
    problems: Mapping[str, BenchmarkProblem],
    optimize_options: OptimizeOptionsType,
    *,
    batch_evaluator: BatchEvaluatorLiteral | BatchEvaluator = "joblib",
    n_cores: int = 1,
    error_handling: Literal["raise", "continue"] = "continue",
    max_criterion_evaluations: int | None = 1_000,
    disable_convergence: bool = True,
) -> dict[tuple[str, str], BenchmarkResult]:
    """Run problems with different optimize options.

    Args:
        problems: Dictionary that maps names to benchmark problems, e.g. the result of
            ``get_benchmark_problems``.
        optimize_options: Dictionary that maps a name you choose (e.g.
            ``"lbfgsb_few_evaluations"``) to what is benchmarked under that name. The
            values can be the name of an algorithm (``"scipy_lbfgsb"``), an algorithm
            class (``om.algos.scipy_lbfgsb``), a configured algorithm
            (``om.algos.scipy_lbfgsb(stopping_maxfun=50)``) or ``om.OptimizeOptions``
            if you want to set more than the algorithm, e.g. options for numerical
            derivatives. Passing a list of algorithms or dictionaries with keyword
            arguments for ``minimize`` as values is deprecated.
        batch_evaluator: See :ref:`batch_evaluators`.
        n_cores: Number of optimizations that is run in parallel. Note that in
            addition to that an optimizer might parallelize.
        error_handling: One of "raise", "continue".
        max_criterion_evaluations: Shortcut to set the maximum number of function
            evaluations instead of passing them in via algo options. In case an
            optimizer does not support this stopping criterion, we also use this as
            max iterations. Options you set yourself take precedence.
        disable_convergence: If True, we set extremely strict convergence criteria by
            default, such that most optimizers will exploit their full computation
            budget set by max_criterion_evaluations. Options you set yourself take
            precedence.

    Returns:
        Dictionary with one benchmark result for each combination of problem and
        optimize options. The keys are tuples where the first entry is the name of the
        problem and the second the name of the optimize options.

    """
    evaluator = process_batch_evaluator(batch_evaluator)

    minimize_options = _process_optimize_options(
        optimize_options,
        max_evals=max_criterion_evaluations,
        disable_convergence=disable_convergence,
    )

    minimize_arguments, keys = _get_optimization_arguments_and_keys(
        problems, minimize_options
    )

    raw_results = evaluator(
        func=minimize,
        arguments=minimize_arguments,
        n_cores=n_cores,
        error_handling=error_handling,
        unpack_symbol="**",
    )

    processing_arguments = []
    for name, raw_result in zip(keys, raw_results, strict=True):
        processing_arguments.append(
            {"optimize_result": raw_result, "problem": problems[name[0]]}
        )

    results = evaluator(
        func=_process_one_result,
        arguments=processing_arguments,
        n_cores=n_cores,
        error_handling="raise",
        unpack_symbol="**",
    )

    return dict(zip(keys, results, strict=True))


def _process_optimize_options(
    raw_options: OptimizeOptionsType,
    max_evals: int | None,
    disable_convergence: bool,
) -> dict[str, dict[str, Any]]:
    """Convert optimize_options to keyword arguments for minimize.

    Args:
        raw_options: The optimize_options passed to run_benchmark.
        max_evals: Default for the maximum number of function evaluations and
            iterations of each algorithm.
        disable_convergence: Whether very strict convergence criteria are set as
            default.

    Returns:
        Dictionary that maps the names of the optimize options to keyword arguments
        for minimize. The keyword arguments do not contain the entries that depend on
        the benchmark problem.

    """
    dict_options: dict[str, Any]
    if isinstance(raw_options, Mapping):
        dict_options = dict(cast(Mapping[str, Any], raw_options))
    elif isinstance(raw_options, str):
        raise TypeError(
            "optimize_options must be a dictionary that maps names to algorithms or "
            f"OptimizeOptions, not the string {raw_options!r}."
        )
    else:
        _throw_list_of_algorithms_future_warning()
        dict_options = {}
        for option in raw_options:
            if isinstance(option, str):
                dict_options[option] = option
            elif isinstance(option, Algorithm):
                dict_options[option.name] = option
            else:
                dict_options[option.__name__] = option

    out = {}
    for name, option in dict_options.items():
        extra_kwargs: dict[str, Any] = {}
        if isinstance(option, dict):
            _throw_dict_options_future_warning()
            kwargs = cast(dict[str, Any], option)
            if "algorithm" not in kwargs:
                raise ValueError(
                    f"The optimize options {name} do not contain an algorithm."
                )
            extra_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in ("algorithm", "algo_options")
            }
            options = OptimizeOptions(
                algorithm=kwargs["algorithm"], algo_options=kwargs.get("algo_options")
            )
        elif isinstance(option, OptimizeOptions):
            options = option
        else:
            options = OptimizeOptions(algorithm=option)

        algorithm = _set_benchmark_defaults(
            pre_process_user_algorithm(options.algorithm),
            max_evals=max_evals,
            disable_convergence=disable_convergence,
        )

        out[name] = {
            "algorithm": algorithm,
            "algo_options": options.algo_options,
            "numdiff_options": options.numdiff_options,
            "scaling": options.scaling,
            "multistart": options.multistart,
            **extra_kwargs,
        }

    return out


def _set_benchmark_defaults(
    algorithm: AlgorithmT, max_evals: int | None, disable_convergence: bool
) -> AlgorithmT:
    """Set the stopping and convergence criteria that are used in benchmarks.

    Options the algorithm does not have are skipped. Options that were changed by the
    user are not overwritten. An option counts as changed if its value differs from the
    default value of the algorithm.

    """
    defaults: dict[str, Any] = {}
    if max_evals is not None:
        defaults["stopping_maxfun"] = max_evals
        defaults["stopping_maxiter"] = max_evals
    if disable_convergence:
        defaults["convergence_ftol_rel"] = 1e-14
        defaults["convergence_xtol_rel"] = 1e-14
        defaults["convergence_gtol_rel"] = 1e-14

    untouched = type(algorithm)()
    applicable = {}
    for name, value in defaults.items():
        if not hasattr(untouched, name):
            continue
        if getattr(algorithm, name) == getattr(untouched, name):
            applicable[name] = value

    return algorithm.with_option(**applicable)


def _get_optimization_arguments_and_keys(
    problems: Mapping[str, BenchmarkProblem],
    minimize_options: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    kwargs_list = []
    names = []

    for problem_name, problem in problems.items():
        for option_name, options in minimize_options.items():
            kwargs = {
                **options,
                # Each optimization gets a new function such that all optimizers see
                # the same noise if the problem has a seed.
                "fun": problem.fun,
                # The start parameters of a problem are read-only.
                "params": problem.start_params.copy(),
                "bounds": problem.bounds,
                "jac": problem.jac,
            }
            kwargs_list.append(kwargs)
            names.append((problem_name, option_name))

    return kwargs_list, names


def _process_one_result(
    optimize_result: OptimizeResult | str, problem: BenchmarkProblem
) -> BenchmarkResult:
    """Process the result of one optimization run.

    Args:
        optimize_result: Result of one optimization run or the traceback of the error
            that was raised during the optimization.
        problem: The problem that was optimized.

    Returns:
        The benchmark result.

    """
    problem_type = problem.function.problem_type
    noise_free_fun = problem.noise_free_fun

    # This will happen if the optimization raised an error
    if isinstance(optimize_result, str):
        return BenchmarkResult(
            params_history=problem.start_params.reshape(1, -1),
            fun_history=np.array([problem.start_fun]),
            time_history=np.array([np.inf]),
            batches_history=np.array([0], dtype=np.int64),
            optimize_result=None,
            traceback=optimize_result,
        )

    history = optimize_result.history
    if history is None or len(history.fun) == 0:
        raise ValueError(
            "Benchmark results can only be created for optimizations that collect a "
            "history."
        )

    # Only keep function evaluations. Evaluations of derivatives have no function
    # value.
    is_fun_evaluation = np.array([value is not None for value in history.fun])
    params_history = np.array(history.flat_params, dtype=np.float64)[is_fun_evaluation]

    if problem.noise is None:
        fun_history = np.array(history.fun, dtype=np.float64)[is_fun_evaluation]
    else:
        fun_history = np.array(
            [
                aggregate_function_value(noise_free_fun(p), problem_type=problem_type)
                for p in params_history
            ]
        )

    return BenchmarkResult(
        params_history=params_history,
        fun_history=fun_history,
        time_history=np.array(history.start_time, dtype=np.float64)[is_fun_evaluation],
        batches_history=np.array(history.batches, dtype=np.int64)[is_fun_evaluation],
        optimize_result=optimize_result,
    )


def _throw_list_of_algorithms_future_warning() -> None:
    msg = (
        "Passing a list of algorithms as `optimize_options` to `run_benchmark` is "
        "deprecated and will be removed in optimagic version 0.7.0. Pass a dictionary "
        "that maps names to algorithms instead, e.g. "
        '`{"lbfgsb": om.algos.scipy_lbfgsb, "nelder_mead": om.algos.scipy_neldermead}`.'
    )
    warnings.warn(msg, FutureWarning)


def _throw_dict_options_future_warning() -> None:
    msg = (
        "Specifying the entries of `optimize_options` in `run_benchmark` as "
        "dictionaries with keyword arguments for `minimize` is deprecated and will be "
        "removed in optimagic version 0.7.0. Use a configured algorithm instead, e.g. "
        "`om.algos.scipy_lbfgsb(stopping_maxfun=100)`, or `om.OptimizeOptions` if you "
        "need to set more than the algorithm."
    )
    warnings.warn(msg, FutureWarning)
