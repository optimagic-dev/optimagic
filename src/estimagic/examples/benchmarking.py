"""Functions to create, run and visualize optimization benchmarks.

TO-DO:
- Add other benchmark sets:
    - finish medium scale problems from https://arxiv.org/pdf/1710.11005.pdf, Page 34.
    - add scalar problems from https://github.com/AxelThevenot
- Add option for deterministic noise or wiggle.

"""
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from estimagic import batch_evaluators
from estimagic.examples.cartis_roberts import CARTIS_ROBERTS_PROBLEMS
from estimagic.examples.more_wild import MORE_WILD_PROBLEMS
from estimagic.examples.noise_distributions import NOISE_DISTRIBUTIONS
from estimagic.logging.read_log import read_optimization_histories
from estimagic.optimization.optimize import minimize


def get_problems(
    name,
    additive_noise=False,
    additive_noise_options=None,
    multiplicative_noise=False,
    multiplicative_noise_options=None,
):
    """Get a dictionary of test problems for a benchmark.

    Args:
        name (str): The name of the set of test problems. Currently "more_wild"
            is the only supported one.
        additive_noise (bool): Whether to add additive noise to the problem.
            Default False.
        additive_noise_options (dict or None): Specifies the amount and distribution
            of the addititve noise added to the problem. Has the entries:
            - distribition (str): One of "normal", "gumbel", "uniform", "logistic".
            Default "normal".
            - std (float): The standard deviation of the noise. This works for all
            distributions, even if those distributions are normally not specified
            via a standard deviation (e.g. uniform).
            - correlation (float): Number between 0 and 1 that specifies the auto
            correlation of the noise.
        multiplicative_noise (bool): Whether to add multiplicative noise to the problem.
            Default False.
        multiplicative_noise_options (dict or None): Specifies the amount and
            distribition of the multiplicative noise added to the problem. Has entries:
            - distribition (str): One of "normal", "gumbel", "uniform", "logistic".
            Default "normal".
            - std (float): The standard deviation of the noise. This works for all
            distributions, even if those distributions are normally not specified
            via a standard deviation (e.g. uniform).
            - correlation (float): Number between 0 and 1 that specifies the auto
            correlation of the noise.
            - clipping_value (float): A non-negative float. Multiplicative noise
            becomes zero if the function value is zero. To avoid this, we do not
            implement multiplicative noise as `f_noisy = f * epsilon` but by
            `f_noisy` = f + (epsilon - 1) * f_clipped` where f_clipped is bounded
            away from zero from both sides by the clipping value.

    Returns:
        dict: Nested dictionary with benchmark problems of the structure:
            {"name": {"inputs": {...}, "solution": {...}, "info": {...}}}
            where "inputs" are keyword arguments for ``minimize`` such as the criterion
            function and start parameters. "solution" contains the entries "params" and
            "value" and "info" might  contain information about the test problem.

    """
    raw_problems = _get_raw_problems(name)

    if additive_noise:
        additive_options = _process_noise_options(additive_noise_options, False)
    else:
        additive_options = None

    if multiplicative_noise:
        multiplicative_options = _process_noise_options(
            multiplicative_noise_options, True
        )
    else:
        multiplicative_options = None

    problems = {}
    for name, specification in raw_problems.items():
        inputs = _create_problem_inputs(
            specification,
            additive_options=additive_options,
            multiplicative_options=multiplicative_options,
        )

        problems[name] = {
            "inputs": inputs,
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
        optimize_options (list or dict): Either a list of algorithms or a Nested
            dictionary that maps a name for optimizer settings
            (e.g. ``"lbfgsb_strict_criterion"``) to a dictionary of keyword arguments
            for arguments for ``minimize`` (e.g. ``{"algorithm": "scipy_lbfgsb",
            "algo_options": {"convergence.relative_criterion_tolerance": 1e-12}}``).
            Alternatively, the values can just be an algorithm which is then benchmarked
            at default settings.
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

    opt_options = _process_optimize_options(optimize_options)

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
    elif name == "cartis_roberts":
        raw_problems = CARTIS_ROBERTS_PROBLEMS
    else:
        raise NotImplementedError()
    return raw_problems


def _create_problem_inputs(specification, additive_options, multiplicative_options):
    _criterion = partial(
        _internal_criterion_template,
        criterion=specification["criterion"],
        additive_options=additive_options,
        multiplicative_options=multiplicative_options,
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


def _internal_criterion_template(
    params, criterion, additive_options, multiplicative_options
):
    x = params["value"].to_numpy()
    critval = criterion(x)

    noise = _get_combined_noise(
        critval,
        additive_options=additive_options,
        multiplicative_options=multiplicative_options,
    )

    noisy_critval = critval + noise

    if isinstance(noisy_critval, np.ndarray):
        out = {
            "root_contributions": noisy_critval,
            "value": noisy_critval @ noisy_critval,
        }
    else:
        out = noisy_critval
    return out


def _get_combined_noise(fval, additive_options, multiplicative_options):
    size = len(np.atleast_1d(fval))
    if multiplicative_options is not None:
        options = multiplicative_options.copy()
        std = options.pop("std")
        clipval = options.pop("clipping_value")
        scaled_std = std * _clip_away_from_zero(fval, clipval)
        multiplicative_noise = _sample_from_distribution(
            **options, std=scaled_std, size=size
        )
    else:
        multiplicative_noise = 0

    if additive_options is not None:
        additive_noise = _sample_from_distribution(**additive_options, size=size)
    else:
        additive_noise = 0

    return multiplicative_noise + additive_noise


def _sample_from_distribution(distribution, mean, std, size, correlation=0):
    sample = NOISE_DISTRIBUTIONS[distribution](size=size)
    dim = size if isinstance(size, int) else size[1]
    if correlation != 0 and dim > 1:
        chol = np.linalg.cholesky(np.diag(np.ones(dim) - correlation) + correlation)
        sample = (chol @ sample.T).T
        sample = sample / sample.std()
    sample *= std
    sample += mean
    return sample


def _process_noise_options(options, is_multiplicative):
    options = {} if options is None else options

    defaults = {"std": 0.01, "distribution": "normal", "correlation": 0, "mean": 0}
    if is_multiplicative:
        defaults["clipping_value"] = 1

    processed = {
        **defaults,
        **options,
    }

    distribution = processed["distribution"]
    if distribution not in NOISE_DISTRIBUTIONS:
        raise ValueError(
            f"Invalid distribution: {distribution}. "
            "Allowed are {list(NOISE_DISTRIBUTIONS)}"
        )

    std = processed["std"]
    if std < 0:
        raise ValueError(f"std must be non-negative. Not: {std}")

    corr = processed["correlation"]
    if corr < 0:
        raise ValueError(f"corr must be non-negative. Not: {corr}")

    if is_multiplicative:
        clipping_value = processed["clipping_value"]
        if clipping_value < 0:
            raise ValueError(
                f"clipping_value must be non-negative. Not: {clipping_value}"
            )

    return processed


def _clip_away_from_zero(a, clipval):
    is_scalar = np.isscalar(a)
    a = np.atleast_1d(a)

    is_positive = a >= 0

    clipped = np.where(is_positive, np.clip(a, clipval, np.inf), a)
    clipped = np.where(~is_positive, np.clip(clipped, -np.inf, -clipval), clipped)

    if is_scalar:
        clipped = clipped[0]
    return clipped


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
