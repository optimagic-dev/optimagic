import warnings
from functools import partial

import numpy as np
import pandas as pd
from estimagic.benchmarking.cartis_roberts import CARTIS_ROBERTS_PROBLEMS
from estimagic.benchmarking.more_wild import MORE_WILD_PROBLEMS
from estimagic.benchmarking.noise_distributions import NOISE_DISTRIBUTIONS


def get_benchmark_problems(
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
            - distribition (str): One of "normal", "gumbel", "uniform", "logistic",
            "laplace". Default "normal".
            - std (float): The standard deviation of the noise. This works for all
            distributions, even if those distributions are normally not specified
            via a standard deviation (e.g. uniform).
            - correlation (float): Number between 0 and 1 that specifies the auto
            correlation of the noise.
        multiplicative_noise (bool): Whether to add multiplicative noise to the problem.
            Default False.
        multiplicative_noise_options (dict or None): Specifies the amount and
            distribition of the multiplicative noise added to the problem. Has entries:
            - distribition (str): One of "normal", "gumbel", "uniform", "logistic",
            "laplace". Default "normal".
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


def _get_raw_problems(name):
    if name == "more_wild":
        raw_problems = MORE_WILD_PROBLEMS
    elif name == "cartis_roberts":
        warnings.warn(
            "Only a subset of the cartis_roberts benchmark suite is currently "
            "implemented. Do not use this for any published work."
        )
        raw_problems = CARTIS_ROBERTS_PROBLEMS
    elif name == "example":
        subset = {
            "linear_full_rank_good_start",
            "rosenbrock_good_start",
            "helical_valley_good_start",
            "powell_singular_good_start",
            "freudenstein_roth_good_start",
            "bard_good_start",
            "box_3d",
            "jennrich_sampson",
            "brown_dennis_good_start",
            "chebyquad_6",
            "bdqrtic_8",
            "mancino_5_good_start",
        }
        raw_problems = {k: v for k, v in MORE_WILD_PROBLEMS.items() if k in subset}
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
