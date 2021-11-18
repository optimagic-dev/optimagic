"""Functions to create, run and visualize optimization benchmarks.

TO-DO:
- Come up with a good specification for noise_options and implement adding noise.
- Add other benchmark sets:
    - medium scale problems from https://arxiv.org/pdf/1710.11005.pdf, Page 34.
    - scalar problems from https://github.com/AxelThevenot
- Think about a good way for handling seeds. Maybe this should be part of the noise
    options or only part of run_benchmark. Needs to be possible to have fixed noise
    and random noise. Maybe distinguish fixed noise by differentiable and
    non differentiable noise.
- Need to think about logging. We probably do not want to use databases for speed
    and disk cluttering reasons but we do want a full history. Maybe fast_logging?
- Instead of one plot_benchmark function we probably want one plotting function for each
    plot type. Inspiration:
    - https://arxiv.org/pdf/1710.11005.pdf
    - https://www.mcs.anl.gov/~more/dfo/

"""
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from estimagic import batch_evaluators
from estimagic.examples.more_wild import MORE_WILD_PROBLEMS
from estimagic.logging.read_log import read_optimization_histories
from estimagic.optimization.optimize import minimize


def get_problems(name, noise_options=None):
    """Get a dictionary of test problems for a benchmark.

    Args:
        name (str): The name of the set of test problems. Currently "more_wild"
            is the only supported one.
        noise_options (dict or None): Specficies the type of noise to add to the test
            problems. Has the entries:
            - type (str): "multiplicative" or "additive"
            - ...

    Returns:
        dict: Nested dictionary with benchmark problems of the structure:
            {"name": {"inputs": {...}, "solution": {...}, "info": {...}}}
            where "inputs" are keyword arguments for ``minimize`` such as the criterion
            function and start parameters. "solution" contains the entries "params" and
            "value" and "info" might  contain information about the test problem.

    """
    raw_problems = _get_raw_problems(name)

    noise_func = _get_noise_func(noise_options)

    problems = {}
    for name, specification in raw_problems.items():
        problems[name] = {
            "inputs": _create_problem_inputs(specification, noise_func),
            "solution": _create_problem_solution(specification),
            "info": specification.get("info", {}),
        }

    return problems


def run_benchmark(
    problems,
    optimize_options,
    logging_directory,
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
        optimize_options: Nested dictionary that maps a name to a set of keyword
            arguments for ``minimize``.
            batch_evaluator (str or callable): See :ref:`batch_evaluators`.
        logging_directory (pathlib.Path): Directory in which the log databases are
            saved.
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
    logging_directory = Path(logging_directory)
    logging_directory.mkdir(parents=True, exist_ok=True)

    if isinstance(batch_evaluator, str):
        batch_evaluator = getattr(
            batch_evaluators, f"{batch_evaluator}_batch_evaluator"
        )

    for options in optimize_options:
        if "log_options" in options:
            raise ValueError(
                "Log options cannot be specified as part of optimize_options. Logging "
                "behavior is configured by the run_benchmark function."
            )

    log_options = {"fast_logging": fast_logging, "if_table_exists": "replace"}

    kwargs_list = []
    names = []
    for prob_name, problem in problems.items():
        for option_name, options in optimize_options.items():
            kwargs = {
                **options,
                **problem["inputs"],
                "logging": logging_directory / f"{prob_name}_{option_name}.db",
                "log_options": log_options,
            }
            kwargs_list.append(kwargs)
            names.append((prob_name, option_name))

    log_paths = [kwargs["logging"] for kwargs in kwargs_list]

    raw_results = batch_evaluator(
        func=minimize,
        arguments=kwargs_list,
        n_cores=n_cores,
        error_handling=error_handling,
        unpack_symbol="**",
    )

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


def _get_raw_problems(name):
    if name == "more_wild":
        raw_problems = MORE_WILD_PROBLEMS
    else:
        raise NotImplementedError()
    return raw_problems


def _get_noise_func(noise_options):
    if noise_options is None:
        noise_func = lambda x: x  # noqa: E731
    else:
        raise NotImplementedError()
    return noise_func


def _create_problem_inputs(specification, noise_func):
    _criterion = partial(
        _internal_criterion_template,
        criterion=specification["criterion"],
        noise_func=noise_func,
    )
    _x = specification["start_x"]

    _params = pd.DataFrame(_x.reshape(-1, 1), columns=["value"])

    inputs = {"criterion": _criterion, "params": _params}
    return inputs


def _create_problem_solution(specification):
    _solution_x = specification.get("solution_x")
    if _solution_x is None:
        _solution_x = specification["start_x"] * np.nan

    _params = pd.DataFrame(_solution_x.reshape(-1, 1), columns=["value"])
    _value = specification["solution_criterion"]

    solution = {
        "params": _params,
        "value": _value,
    }
    return solution


def _internal_criterion_template(params, criterion, noise_func):
    x = params["value"].to_numpy()
    clean_value = criterion(x)
    noisy_value = noise_func(clean_value)
    if isinstance(noisy_value, np.ndarray):
        out = {"contributions": noisy_value, "value": noisy_value @ noisy_value}
    else:
        out = noisy_value
    return out
