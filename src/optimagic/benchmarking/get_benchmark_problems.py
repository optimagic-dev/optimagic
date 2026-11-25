"""Get the sets of benchmark problems that are shipped with optimagic."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, get_args

import numpy as np
from numpy.typing import NDArray

from optimagic.benchmarking.benchmark_problem import (
    BenchmarkFunction,
    BenchmarkProblem,
)
from optimagic.benchmarking.noise import (
    AdditiveBenchmarkNoise,
    BenchmarkNoise,
    MultiplicativeBenchmarkNoise,
    pre_process_noise,
)
from optimagic.benchmarking.scaling import (
    BenchmarkScaler,
    LinspaceBenchmarkScaler,
    pre_process_scaling,
)
from optimagic.typing import AggregationLevel
from optimagic.utilities import get_rng

BenchmarkSetName = Literal["more_wild", "cartis_roberts", "example", "estimagic"]


def get_benchmark_problems(
    name: BenchmarkSetName,
    *,
    noise: bool | BenchmarkNoise = False,
    scaling: bool | BenchmarkScaler = False,
    seed: int | np.random.Generator | None = None,
    exclude: str | Sequence[str] | None = None,
    # deprecated
    additive_noise: bool | None = None,
    additive_noise_options: dict[str, Any] | None = None,
    multiplicative_noise: bool | None = None,
    multiplicative_noise_options: dict[str, Any] | None = None,
    scaling_options: dict[str, Any] | None = None,
) -> dict[str, BenchmarkProblem]:
    """Get a dictionary of test problems for a benchmark.

    Args:
        name: The name of the set of test problems. "more_wild" are the least-squares
            problems of Moré and Wild (2009) with 2 to 12 parameters and one additional
            problem with 100 parameters. "cartis_roberts" are the least-squares
            problems of Cartis and Roberts (2019) with 25 to 120 parameters. "example"
            is a small subset of "more_wild" and "estimagic" a mix of problems from
            both sets.
        noise: Whether and how noise is added to the function values. If False, no
            noise is added. If True, normally distributed noise with a standard
            deviation of 0.01 is added. Pass an ``AdditiveBenchmarkNoise`` or a
            ``MultiplicativeBenchmarkNoise`` to control the noise. Noise objects can be
            combined with ``+``.
        scaling: Whether and how the parameter space of the problems is rescaled. If
            False, the problems are not rescaled. If True, the parameters are multiplied
            by ``np.linspace(0.1, 10, n_params)``. Pass a ``LinspaceBenchmarkScaler`` or
            another ``BenchmarkScaler`` to control the scaling.
        seed: Seed for the noise. With an int or a numpy.random.Generator, the noise
            is reproducible and all optimizers see the same noise on a problem. If
            None, the noise differs between optimizations.
        exclude: Name or names of problems that are left out.
        additive_noise: Deprecated. Use ``noise`` instead.
        additive_noise_options: Deprecated. Use ``noise`` instead.
        multiplicative_noise: Deprecated. Use ``noise`` instead.
        multiplicative_noise_options: Deprecated. Use ``noise`` instead.
        scaling_options: Deprecated. Use ``scaling`` instead.

    Returns:
        Dictionary that maps the names of the problems to benchmark problems. The
        dictionary can be filtered or merged with other problem dictionaries before it
        is passed to ``run_benchmark``.

    """
    # ==================================================================================
    # handle deprecations
    # ==================================================================================
    noise = _replace_deprecated_noise_arguments(
        noise=noise,
        additive_noise=additive_noise,
        additive_noise_options=additive_noise_options,
        multiplicative_noise=multiplicative_noise,
        multiplicative_noise_options=multiplicative_noise_options,
    )
    scaling = _replace_deprecated_scaling_options(
        scaling=scaling, scaling_options=scaling_options
    )

    # ==================================================================================
    # process arguments
    # ==================================================================================
    processed_noise = pre_process_noise(noise)
    scaler = pre_process_scaling(scaling)

    if exclude is None:
        excluded: set[str] = set()
    elif isinstance(exclude, str):
        excluded = {exclude}
    else:
        excluded = set(exclude)

    rng = get_rng(seed)

    # ==================================================================================
    # create the problems
    # ==================================================================================
    problems = {}
    for problem_name, raw_problem in _get_problem_set(name).items():
        if problem_name in excluded:
            continue
        problem = raw_problem
        if scaler is not None:
            problem = problem.with_scaling(scaler)
        if processed_noise is not None:
            # Each problem gets its own seed such that the noise of a problem does not
            # depend on which other problems are optimized before it.
            problem_seed = None if seed is None else int(rng.integers(2**32))
            problem = problem.with_noise(processed_noise, seed=problem_seed)
        problems[problem_name] = problem

    return problems


@dataclass(frozen=True)
class BenchmarkFunctionWithSteps(BenchmarkFunction):
    """Benchmark function that is evaluated at rounded parameters.

    Rounding makes the function piecewise constant. This is a problem for optimizers
    that use numerical derivatives with small step sizes.

    """

    function: BenchmarkFunction
    """The function that is evaluated at the rounded parameters."""

    decimals: int = 3
    """The number of decimals to which the parameters are rounded."""

    differentiable = False

    def fun(self, x: NDArray[np.float64]) -> float | NDArray[np.float64]:
        return self.function.fun(x.round(self.decimals))

    @property
    def problem_type(self) -> AggregationLevel:
        return self.function.problem_type

    def supports_n_params(self, n_params: int) -> bool:
        return self.function.supports_n_params(n_params)


def _get_problem_set(name: str) -> dict[str, BenchmarkProblem]:
    # The problem sets are imported here to keep `import optimagic` fast. Importing
    # the Cartis-Roberts problems imports numba.
    from optimagic.benchmarking.more_wild import MORE_WILD_PROBLEMS

    if name == "more_wild":
        problems = dict(MORE_WILD_PROBLEMS)
    elif name == "cartis_roberts":
        from optimagic.benchmarking.cartis_roberts import CARTIS_ROBERTS_PROBLEMS

        problems = dict(CARTIS_ROBERTS_PROBLEMS)
    elif name == "example":
        subset = {
            "rosenbrock_good_start",
            "helical_valley_good_start",
            "powell_singular_good_start",
            "freudenstein_roth_good_start",
            "bard_good_start",
            "box_3d",
            "brown_dennis_good_start",
            "chebyquad_6",
            "bdqrtic_8",
            "mancino_5_good_start",
        }
        problems = {k: v for k, v in MORE_WILD_PROBLEMS.items() if k in subset}
    elif name == "estimagic":
        from optimagic.benchmarking.cartis_roberts import CARTIS_ROBERTS_PROBLEMS

        subset_mw = {
            "cube_8",
            "chebyquad_6",
            "bdqrtic_8",
            "linear_full_rank_bad_start",
            "chebyquad_7",
            "osborne_two_bad_start",
            "bdqrtic_10",
            "bdqrtic_11",
            "heart_eight_bad_start",
            "mancino_5_bad_start",
            "chebyquad_8",
            "cube_6",
            "cube_5",
            "bdqrtic_12",
            "chebyquad_10",
            "chebyquad_9",
            "chebyquad_11",
            "mancino_8",
            "mancino_10",
            "mancino_12_bad_start",
        }
        subset_cr = {
            "hatfldg",
            "bratu_3d",
            "cbratu_2d",
            "chnrsbne",
            "bratu_2d",
            "vardimne",
            "penalty_1",
            "arglale",
            "arglble",
        }
        subset_add_steps = {
            "rosenbrock_good_start",
            "cube_5",
            "chebyquad_10",
        }
        problems = {}
        for k, v in MORE_WILD_PROBLEMS.items():
            if k in subset_mw:
                problems[k] = v
            if k in subset_add_steps:
                problems[f"{k}_with_steps"] = replace(
                    v, function=BenchmarkFunctionWithSteps(v.function)
                )

        for k, v in CARTIS_ROBERTS_PROBLEMS.items():
            if k in subset_cr:
                problems[k] = v

    else:
        raise ValueError(
            f"Invalid name of a benchmark set: {name}. Valid names are "
            f"{get_args(BenchmarkSetName)}."
        )
    return problems


def _replace_deprecated_noise_arguments(
    noise: bool | BenchmarkNoise,
    additive_noise: bool | None,
    additive_noise_options: dict[str, Any] | None,
    multiplicative_noise: bool | None,
    multiplicative_noise_options: dict[str, Any] | None,
) -> bool | BenchmarkNoise:
    """Convert the deprecated noise arguments to a value for the noise argument."""
    deprecated = {
        "additive_noise": additive_noise,
        "additive_noise_options": additive_noise_options,
        "multiplicative_noise": multiplicative_noise,
        "multiplicative_noise_options": multiplicative_noise_options,
    }
    used = [name for name, value in deprecated.items() if value is not None]
    if not used:
        return noise

    msg = (
        f"The arguments {used} of `get_benchmark_problems` are deprecated and will be "
        "removed in optimagic version 0.7.0. Use the `noise` argument instead, e.g. "
        "`noise=om.AdditiveBenchmarkNoise(std=0.1)` or "
        "`noise=om.AdditiveBenchmarkNoise() + om.MultiplicativeBenchmarkNoise()`."
    )
    warnings.warn(msg, FutureWarning)

    if noise is not False:
        raise ValueError(
            f"You cannot use the `noise` argument and the deprecated arguments {used} "
            "at the same time."
        )

    components: list[BenchmarkNoise] = []
    if multiplicative_noise:
        options = multiplicative_noise_options or {}
        components.append(MultiplicativeBenchmarkNoise(**options))
    if additive_noise:
        options = additive_noise_options or {}
        components.append(AdditiveBenchmarkNoise(**options))

    if not components:
        return False
    out = components[0]
    for component in components[1:]:
        out = out + component
    return out


def _replace_deprecated_scaling_options(
    scaling: bool | BenchmarkScaler, scaling_options: dict[str, Any] | None
) -> bool | BenchmarkScaler:
    """Convert the deprecated scaling_options to a value for the scaling argument."""
    if scaling_options is None:
        return scaling

    msg = (
        "The argument `scaling_options` of `get_benchmark_problems` is deprecated and "
        "will be removed in optimagic version 0.7.0. Pass a scaler as `scaling` "
        "instead, e.g. `scaling=om.LinspaceBenchmarkScaler(min_scale=0.1, "
        "max_scale=10)`."
    )
    warnings.warn(msg, FutureWarning)

    if isinstance(scaling, BenchmarkScaler):
        raise ValueError(
            "You cannot pass a BenchmarkScaler as `scaling` and use the deprecated "
            "argument `scaling_options` at the same time."
        )
    if scaling:
        return LinspaceBenchmarkScaler(**scaling_options)
    return False
