from itertools import product

import numpy as np
import pytest

from optimagic import minimize
from optimagic.benchmarking.get_benchmark_problems import (
    _step_func,
    get_benchmark_problems,
)

PARMETRIZATION = []
for name in ["more_wild", "cartis_roberts", "example", "estimagic"]:
    for additive, multiplicative, scaling in product([False, True], repeat=3):
        PARMETRIZATION.append((name, additive, multiplicative, scaling))


@pytest.mark.parametrize(
    "name, additive_noise, multiplicative_noise, scaling", PARMETRIZATION
)
def test_get_problems(name, additive_noise, multiplicative_noise, scaling):
    is_noisy = any((additive_noise, multiplicative_noise))
    problems = get_benchmark_problems(
        name=name,
        additive_noise=additive_noise,
        multiplicative_noise=multiplicative_noise,
        scaling=scaling,
    )
    first_name = list(problems)[0]
    first = problems[first_name]
    func = first["inputs"]["fun"]
    params = first["inputs"]["params"]

    first_eval = func(params)
    second_eval = func(params)

    if is_noisy:
        assert not np.allclose(first_eval, second_eval)
    else:
        assert np.allclose(first_eval, second_eval)

    for problem in problems.values():
        assert isinstance(problem["inputs"]["params"], np.ndarray)
        assert isinstance(problem["solution"]["params"], np.ndarray)


def test_step_func():
    p = np.array([0.0001, 0.0002])
    got = _step_func(p, lambda x: x @ x)
    assert np.allclose(got, 0)
    assert not np.allclose(p @ p, 0)


@pytest.mark.parametrize("name", ["more_wild", "cartis_roberts"])
def test_noise_free_fun_is_scaled_like_inputs(name):
    problems = get_benchmark_problems(name, additive_noise=True, scaling=True)
    for problem in problems.values():
        start_x = problem["inputs"]["params"]
        residuals = problem["noise_free_fun"](start_x)
        assert np.allclose(residuals @ residuals, problem["start_criterion"])


def test_minimize_problem_with_partialled_criterion():
    problem = get_benchmark_problems("cartis_roberts")["bratu_2d"]
    res = minimize(**problem["inputs"], algorithm="scipy_ls_trf")
    assert np.allclose(res.fun, problem["solution"]["value"], atol=1e-8)
