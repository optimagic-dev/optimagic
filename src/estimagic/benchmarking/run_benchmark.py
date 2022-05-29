"""Functions to create, run and visualize optimization benchmarks.

TO-DO:
- Add other benchmark sets:
    - finish medium scale problems from https://arxiv.org/pdf/1710.11005.pdf, Page 34.
    - add scalar problems from https://github.com/AxelThevenot
- Add option for deterministic noise or wiggle.
"""
import numpy as np
import pandas as pd
from estimagic import batch_evaluators
from estimagic.optimization.optimize import minimize
from estimagic.optimization.optimize_result import OptimizeResult
from estimagic.parameters.tree_registry import get_registry
from pybaum import tree_just_flatten


def run_benchmark(
    problems,
    optimize_options,
    *,
    batch_evaluator="joblib",
    n_cores=1,
    error_handling="continue",
    seed=None,
):
    """Run problems with different optimize options.

    Args:
        problems (dict): Nested dictionary with benchmark problems of the structure:
            {"name": {"inputs": {...}, "solution": {...}, "info": {...}}}
            where "inputs" are keyword arguments for ``minimize`` such as the criterion
            function and start parameters. "solution" contains the entries "params" and
            "value" and "info" might  contain information about the test problem.
        optimize_options (list or dict): Either a list of algorithms or a Nested
            dictionary that maps a name for optimizer settings
            (e.g. ``"lbfgsb_strict_criterion"``) to a dictionary of keyword arguments
            for arguments for ``minimize`` (e.g. ``{"algorithm": "scipy_lbfgsb",
            "algo_options": {"convergence.relative_criterion_tolerance": 1e-12}}``).
            Alternatively, the values can just be an algorithm which is then benchmarked
            at default settings.
        batch_evaluator (str or callable): See :ref:`batch_evaluators`.
        n_cores (int): Number of optimizations that is run in parallel. Note that in
            addition to that an optimizer might parallelize.
        error_handling (str): One of "raise", "continue".

    Returns:
        dict: Nested Dictionary with information on the benchmark run. The outer keys
            are tuples where the first entry is the name of the problem and the second
            the name of the optimize options. The values are dicts with the entries:
            "params_history", "criterion_history", "time_history" and "solution".
    """
    np.random.seed(seed)

    if isinstance(batch_evaluator, str):
        batch_evaluator = getattr(
            batch_evaluators, f"{batch_evaluator}_batch_evaluator"
        )
    opt_options = _process_optimize_options(optimize_options)

    kwargs_list, names = _get_kwargs_list_and_names(problems, opt_options)

    raw_results = batch_evaluator(
        func=minimize,
        arguments=kwargs_list,
        n_cores=n_cores,
        error_handling=error_handling,
        unpack_symbol="**",
    )

    results = _get_results(names, raw_results, kwargs_list)

    return results


def _process_optimize_options(raw_options):
    if not isinstance(raw_options, dict):
        dict_options = {}
        for option in raw_options:
            if isinstance(option, str):
                dict_options[option] = option
            else:
                dict_options[option.__name__] = option
    else:
        dict_options = raw_options

    out_options = {}
    for name, option in dict_options.items():
        if not isinstance(option, dict):
            option = {"algorithm": option}

        out_options[name] = option

    return out_options


def _get_kwargs_list_and_names(problems, opt_options):
    kwargs_list = []
    names = []

    for prob_name, problem in problems.items():
        for option_name, options in opt_options.items():
            kwargs = {**options, **problem["inputs"]}
            kwargs_list.append(kwargs)
            names.append((prob_name, option_name))

    return kwargs_list, names


def _get_results(names, raw_results, kwargs_list):
    registry = get_registry(extended=True)
    results = {}

    for name, result, inputs in zip(names, raw_results, kwargs_list):

        if isinstance(result, OptimizeResult):
            history = result.history
            params_history = pd.DataFrame(
                [tree_just_flatten(p, registry=registry) for p in history["params"]]
            )
            criterion_history = pd.Series(history["criterion"])
            time_history = pd.Series(history["runtime"])
        elif isinstance(result, str):
            _criterion = inputs["criterion"]

            params_history = pd.DataFrame(
                tree_just_flatten(inputs["params"], registry=registry)
            ).T
            criterion_history = pd.Series(_criterion(inputs["params"])["value"])

            time_history = pd.Series([np.inf])
        else:
            raise ValueError(
                "'result' object is expected to be of type 'dict' or 'str'."
            )

        results[name] = {
            "params_history": params_history,
            "criterion_history": criterion_history,
            "time_history": time_history,
            "solution": result,
        }

    return results
