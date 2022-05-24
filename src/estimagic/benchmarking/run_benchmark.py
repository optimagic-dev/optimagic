"""Functions to create, run and visualize optimization benchmarks.

TO-DO:
- Add other benchmark sets:
    - finish medium scale problems from https://arxiv.org/pdf/1710.11005.pdf, Page 34.
    - add scalar problems from https://github.com/AxelThevenot
- Add option for deterministic noise or wiggle.

"""
from pathlib import Path

import numpy as np
import pandas as pd
from estimagic import batch_evaluators
from estimagic.logging.read_log import read_optimization_histories
from estimagic.optimization.optimize import minimize


def run_benchmark(
    problems,
    optimize_options,
    *,
    logging_directory=None,
    batch_evaluator="joblib",
    n_cores=1,
    error_handling="continue",
    fast_logging=True,
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
        logging_directory (None or pathlib.Path): Directory in which the log databases
            are saved. By default, this is set to None, which means logging is
            switched off to save runtime.
        n_cores (int): Number of optimizations that is run in parallel. Note that in
            addition to that an optimizer might parallelize.
        error_handling (str): One of "raise", "continue".
        fast_logging (bool): Whether the slightly unsafe but much faster database
            configuration is chosen.

    Returns:
        dict: Nested Dictionary with information on the benchmark run. The outer keys
            are tuples where the first entry is the name of the problem and the second
            the name of the optimize options. The values are dicts with the entries:
            "runtime", "params_history", "criterion_history", "solution"

    """
    np.random.seed(seed)

    if isinstance(batch_evaluator, str):
        batch_evaluator = getattr(
            batch_evaluators, f"{batch_evaluator}_batch_evaluator"
        )
    opt_options = _process_optimize_options(optimize_options)

    if isinstance(logging_directory, Path):
        kwargs_list, names, log_paths = _get_kwargs_list_and_names_logging(
            problems, opt_options, logging_directory, fast_logging
        )
    else:
        kwargs_list, names = _get_kwargs_list_and_names_history(problems, opt_options)

    raw_results = batch_evaluator(
        func=minimize,
        arguments=kwargs_list,
        n_cores=n_cores,
        error_handling=error_handling,
        unpack_symbol="**",
    )

    if isinstance(logging_directory, Path):
        results = _get_results_logging(names, raw_results, log_paths)
    else:
        results = _get_results_history(names, raw_results)

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

        if "log_options" in option:
            raise ValueError(
                "Log options cannot be specified as part of optimize_options. Logging "
                "behavior is configured by the run_benchmark function."
            )
        out_options[name] = option

    return out_options


def _get_kwargs_list_and_names_history(problems, opt_options):
    kwargs_list = []
    names = []

    for prob_name, problem in problems.items():
        for option_name, options in opt_options.items():
            kwargs = {**options, **problem["inputs"]}
            kwargs_list.append(kwargs)
            names.append((prob_name, option_name))

    return kwargs_list, names


def _get_kwargs_list_and_names_logging(
    problems, opt_options, logging_directory, fast_logging
):
    logging_directory = Path(logging_directory)
    logging_directory.mkdir(parents=True, exist_ok=True)
    log_options = {"fast_logging": fast_logging, "if_table_exists": "replace"}

    kwargs_list = []
    names = []
    for prob_name, problem in problems.items():
        for option_name, options in opt_options.items():
            kwargs = {
                **options,
                **problem["inputs"],
                "logging": logging_directory / f"{prob_name}_{option_name}.db",
                "log_options": log_options,
            }
            kwargs_list.append(kwargs)
            names.append((prob_name, option_name))

    log_paths = [kwargs["logging"] for kwargs in kwargs_list]

    return kwargs_list, names, log_paths


def _get_results_history(names, raw_results):
    results = {}

    for name, result in zip(names, raw_results):

        if isinstance(result, dict):
            params_history = pd.concat(
                [hist["params"] for hist in result["history"]],
                axis=1,
                ignore_index=True,
            ).T
            criterion_history = pd.Series(
                [hist["scalar_criterion"] for hist in result["history"]]
            )

            timestamps = pd.Series([hist["timestamp"] for hist in result["history"]])
            stop = timestamps.max()
            start = timestamps.min()
            runtime = (stop - start).total_seconds()
            time_history = timestamps - start
        else:
            criterion_history = pd.Series(raw_results[0]["history"][0]["criterion"])
            params_history = raw_results[0]["history"][0]["params"]

            runtime = pd.Series([], dtype="datetime64[ns]")
            time_history = pd.Series([], dtype="datetime64[ns]")

        results[name] = {
            "params_history": params_history,
            "criterion_history": criterion_history,
            "time_history": time_history,
            "solution": result,
            "runtime": runtime,
        }

    return results


def _get_results_logging(names, raw_results, log_paths):
    results = {}

    for name, result, log_path in zip(names, raw_results, log_paths):
        histories = read_optimization_histories(log_path)
        stop = histories["metadata"]["timestamps"].max()
        start = histories["metadata"]["timestamps"].min()
        runtime = (stop - start).total_seconds()

        results[name] = {
            "params_history": histories["params"],
            "criterion_history": histories["values"],
            "time_history": histories["metadata"]["timestamps"] - start,
            "solution": result,
            "runtime": runtime,
        }

    return results
