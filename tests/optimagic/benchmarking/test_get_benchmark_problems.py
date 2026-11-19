import warnings
from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from optimagic import minimize
from optimagic.benchmarking.benchmark_problem import BenchmarkProblem
from optimagic.benchmarking.get_benchmark_problems import (
    LeastSquaresFunctionWithSteps,
    get_benchmark_problems,
)
from optimagic.benchmarking.more_wild import Rosenbrock
from optimagic.benchmarking.noise import (
    AdditiveBenchmarkNoise,
    CombinedBenchmarkNoise,
    MultiplicativeBenchmarkNoise,
)
from optimagic.benchmarking.scaling import LinspaceBenchmarkScaler

NAMES = ["more_wild", "cartis_roberts", "example", "estimagic"]

PARMETRIZATION = []
for name in NAMES:
    for noise, scaling in product([False, True], repeat=2):
        PARMETRIZATION.append((name, noise, scaling))


@pytest.mark.parametrize("name, noise, scaling", PARMETRIZATION)
def test_get_problems(name, noise, scaling):
    problems = get_benchmark_problems(name=name, noise=noise, scaling=scaling)
    first = problems[list(problems)[0]]
    func = first.fun
    params = first.start_params

    first_eval = func(params)
    second_eval = func(params)

    if noise:
        assert not np.allclose(first_eval, second_eval)
    else:
        assert np.allclose(first_eval, second_eval)

    for problem in problems.values():
        assert isinstance(problem, BenchmarkProblem)
        assert isinstance(problem.start_params, np.ndarray)
        assert (problem.noise is not None) == noise
        assert (problem.scaler is not None) == scaling


@pytest.mark.parametrize("name", NAMES)
def test_number_of_problems(name):
    expected = {"more_wild": 54, "cartis_roberts": 60, "example": 10, "estimagic": 32}
    assert len(get_benchmark_problems(name)) == expected[name]


def test_invalid_name_raises():
    with pytest.raises(ValueError, match="Invalid name of a benchmark set: bla"):
        get_benchmark_problems("bla")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("exclude", ["box_3d", ["box_3d"], ("box_3d", "chebyquad_6")])
def test_exclude(exclude):
    problems = get_benchmark_problems("example", exclude=exclude)
    excluded = {exclude} if isinstance(exclude, str) else set(exclude)
    assert set(problems) == set(get_benchmark_problems("example")) - excluded


def test_problems_can_be_filtered_and_merged_like_dicts():
    example = get_benchmark_problems("example")
    cartis_roberts = get_benchmark_problems("cartis_roberts")

    small = {name: p for name, p in example.items() if p.n_params <= 3}
    assert set(small) == {
        "rosenbrock_good_start",
        "helical_valley_good_start",
        "freudenstein_roth_good_start",
        "bard_good_start",
        "box_3d",
    }

    merged = small | {"arglale": cartis_roberts["arglale"]}
    assert len(merged) == 6


def test_shipped_problem_sets_are_not_modified():
    get_benchmark_problems("example", noise=True, scaling=True)
    problem = get_benchmark_problems("example")["rosenbrock_good_start"]
    assert problem.noise is None
    assert problem.scaler is None
    aaae(problem.start_params, np.array([-1.2, 1.0]))


# ======================================================================================
# noise and scaling
# ======================================================================================


def test_noise_true_is_additive_noise_with_defaults():
    problems = get_benchmark_problems("example", noise=True)
    for problem in problems.values():
        assert problem.noise == AdditiveBenchmarkNoise()


def test_custom_noise():
    noise = MultiplicativeBenchmarkNoise(std=0.3) + AdditiveBenchmarkNoise(std=0.1)
    problems = get_benchmark_problems("example", noise=noise)
    for problem in problems.values():
        assert problem.noise is noise


def test_noise_with_invalid_type_raises():
    with pytest.raises(TypeError, match="noise must be a bool or an instance"):
        get_benchmark_problems(
            "example",
            noise={"std": 0.1},  # ty: ignore[invalid-argument-type]
        )


def test_scaling_true_is_linspace_scaler_with_defaults():
    problems = get_benchmark_problems("example", scaling=True)
    for problem in problems.values():
        assert problem.scaler == LinspaceBenchmarkScaler()


def test_custom_scaling():
    scaler = LinspaceBenchmarkScaler(min_scale=1, max_scale=3)
    problem = get_benchmark_problems("example", scaling=scaler)["rosenbrock_good_start"]
    assert problem.scaler is scaler
    aaae(problem.start_params, np.array([-1.2, 3.0]))
    assert problem.solution is not None
    aaae(np.asarray(problem.solution.params), np.array([1.0, 3.0]))


def test_scaling_with_invalid_type_raises():
    with pytest.raises(TypeError, match="scaling must be a bool or an instance"):
        get_benchmark_problems(
            "example",
            scaling="linspace",  # ty: ignore[invalid-argument-type]
        )


@pytest.mark.parametrize("name", ["more_wild", "cartis_roberts"])
def test_noise_free_fun_is_scaled_like_start_params(name):
    """Ported from #723, where the noise free function ignored the scaling."""
    unscaled = get_benchmark_problems(name)
    problems = get_benchmark_problems(name, noise=True, scaling=True)
    for problem_name, problem in problems.items():
        assert np.allclose(problem.start_fun, unscaled[problem_name].start_fun)


def test_seed_makes_noise_reproducible():
    first = get_benchmark_problems("example", noise=True, seed=0)
    second = get_benchmark_problems("example", noise=True, seed=0)
    other = get_benchmark_problems("example", noise=True, seed=1)

    for name, problem in first.items():
        x = problem.start_params
        assert np.array_equal(problem.fun(x), second[name].fun(x))
        assert not np.array_equal(problem.fun(x), other[name].fun(x))


def test_each_problem_has_its_own_seed():
    problems = get_benchmark_problems("example", noise=True, seed=0)
    seeds = [problem.seed for problem in problems.values()]
    assert len(set(seeds)) == len(problems)


def test_seed_can_be_a_generator():
    first = get_benchmark_problems("example", noise=True, seed=np.random.default_rng(3))
    second = get_benchmark_problems(
        "example", noise=True, seed=np.random.default_rng(3)
    )
    assert [p.seed for p in first.values()] == [p.seed for p in second.values()]


def test_without_seed_problems_have_no_seed():
    problems = get_benchmark_problems("example", noise=True)
    assert all(problem.seed is None for problem in problems.values())


# ======================================================================================
# problems with steps
# ======================================================================================


def test_function_with_steps():
    function = LeastSquaresFunctionWithSteps(Rosenbrock())
    x = np.array([0.0001, 0.0002])
    aaae(function.fun(x), Rosenbrock().fun(np.zeros(2)))
    assert not np.allclose(Rosenbrock().fun(x), Rosenbrock().fun(np.zeros(2)))

    assert function.differentiable is False
    assert function.supports_n_params(2)
    assert not function.supports_n_params(3)


def test_estimagic_set_contains_problems_with_steps():
    problems = get_benchmark_problems("estimagic")
    with_steps = {name for name in problems if name.endswith("_with_steps")}
    assert with_steps == {
        "rosenbrock_good_start_with_steps",
        "cube_5_with_steps",
        "chebyquad_10_with_steps",
    }
    problem = problems["rosenbrock_good_start_with_steps"]
    assert problem.function == LeastSquaresFunctionWithSteps(Rosenbrock())
    assert problem.solution is not None
    assert problem.solution.fun == 0


# ======================================================================================
# problems can be solved
# ======================================================================================


def test_minimize_problem_with_partialled_criterion():
    """Ported from #723, where this problem was treated as scalar problem."""
    problem = get_benchmark_problems("cartis_roberts")["bratu_2d"]
    res = minimize(
        fun=problem.fun, params=problem.start_params, algorithm="scipy_ls_trf"
    )
    assert problem.solution is not None
    assert np.allclose(res.fun, problem.solution.fun, atol=1e-8)


# ======================================================================================
# deprecations
# ======================================================================================


def test_additive_noise_is_deprecated():
    with pytest.warns(FutureWarning, match="'additive_noise'"):
        problems = get_benchmark_problems("example", additive_noise=True)
    assert problems["box_3d"].noise == AdditiveBenchmarkNoise()


def test_additive_noise_options_are_deprecated():
    options = {"distribution": "uniform", "std": 0.3, "correlation": 0.2}
    with pytest.warns(FutureWarning, match="will be removed in optimagic version 0.7"):
        problems = get_benchmark_problems(
            "example", additive_noise=True, additive_noise_options=options
        )
    assert problems["box_3d"].noise == AdditiveBenchmarkNoise(
        distribution="uniform", std=0.3, correlation=0.2
    )


def test_multiplicative_noise_is_deprecated():
    options = {"std": 0.3, "clipping_value": 2.0}
    with pytest.warns(FutureWarning, match="'multiplicative_noise'"):
        problems = get_benchmark_problems(
            "example", multiplicative_noise=True, multiplicative_noise_options=options
        )
    assert problems["box_3d"].noise == MultiplicativeBenchmarkNoise(
        std=0.3, clipping_value=2.0
    )


def test_deprecated_additive_and_multiplicative_noise_are_combined():
    with pytest.warns(FutureWarning, match="Use the `noise` argument instead"):
        problems = get_benchmark_problems(
            "example", additive_noise=True, multiplicative_noise=True
        )
    noise = problems["box_3d"].noise
    assert isinstance(noise, CombinedBenchmarkNoise)
    assert noise.components == (
        MultiplicativeBenchmarkNoise(),
        AdditiveBenchmarkNoise(),
    )


def test_deprecated_noise_options_without_noise_are_ignored():
    with pytest.warns(FutureWarning, match="'additive_noise_options'"):
        problems = get_benchmark_problems(
            "example", additive_noise=False, additive_noise_options={"std": 0.3}
        )
    assert problems["box_3d"].noise is None


def test_deprecated_noise_arguments_cannot_be_combined_with_noise():
    with pytest.warns(FutureWarning):
        with pytest.raises(ValueError, match="at the same time"):
            get_benchmark_problems("example", noise=True, additive_noise=True)


def test_scaling_options_are_deprecated():
    options = {"min_scale": 1, "max_scale": 3}
    with pytest.warns(FutureWarning, match="`scaling_options`"):
        problems = get_benchmark_problems(
            "example", scaling=True, scaling_options=options
        )
    assert problems["box_3d"].scaler == LinspaceBenchmarkScaler(
        min_scale=1, max_scale=3
    )


def test_deprecated_scaling_options_without_scaling_are_ignored():
    with pytest.warns(FutureWarning, match="`scaling_options`"):
        problems = get_benchmark_problems(
            "example", scaling_options={"min_scale": 1, "max_scale": 3}
        )
    assert problems["box_3d"].scaler is None


def test_deprecated_scaling_options_cannot_be_combined_with_scaler():
    with pytest.warns(FutureWarning):
        with pytest.raises(ValueError, match="at the same time"):
            get_benchmark_problems(
                "example",
                scaling=LinspaceBenchmarkScaler(),
                scaling_options={"min_scale": 1},
            )


def test_new_arguments_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        get_benchmark_problems(
            "example",
            noise=AdditiveBenchmarkNoise(),
            scaling=LinspaceBenchmarkScaler(),
            seed=0,
        )
