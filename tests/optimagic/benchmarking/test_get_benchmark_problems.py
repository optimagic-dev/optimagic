from itertools import product

import numpy as np
import pytest

import optimagic as om
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


LIKELIHOOD_SET_NAMES = [
    "likelihood_small",
    "likelihood_large",
    "likelihood_misspecified_small",
    "likelihood_misspecified_large",
]


@pytest.mark.parametrize("name", LIKELIHOOD_SET_NAMES)
def test_get_likelihood_problems(name):
    problems = get_benchmark_problems(name)
    for problem in problems.values():
        inputs = problem["inputs"]
        assert set(inputs) <= {"fun", "params", "jac", "bounds"}
        assert len(inputs["jac"]) == 2
        assert problem["info"]["has_bounds"] == ("bounds" in inputs)
        start_value = inputs["fun"](inputs["params"]).sum()
        np.testing.assert_allclose(start_value, problem["start_criterion"])
        assert problem["solution"]["value"] < problem["start_criterion"]


@pytest.mark.parametrize("problem_name", ["logit_5_good_start", "garch_3_bad_start"])
def test_likelihood_derivatives_and_bounds_with_scaling(problem_name):
    unscaled = get_benchmark_problems("likelihood_small")[problem_name]
    problem = get_benchmark_problems("likelihood_small", scaling=True)[problem_name]
    inputs = problem["inputs"]
    params = inputs["params"]
    # default scaling options
    scaling_factor = np.linspace(0.1, 10, len(params))
    np.testing.assert_allclose(params, unscaled["inputs"]["params"] * scaling_factor)

    np.testing.assert_allclose(
        inputs["fun"](params), unscaled["inputs"]["fun"](unscaled["inputs"]["params"])
    )

    gradient, jac = inputs["jac"]
    bounds = inputs.get("bounds", om.Bounds())
    numerical = om.first_derivative(inputs["fun"], params, bounds=bounds).derivative
    np.testing.assert_allclose(jac(params), numerical, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(gradient(params), jac(params).sum(axis=0))

    if "bounds" in inputs:
        unscaled_bounds = unscaled["inputs"]["bounds"]
        np.testing.assert_allclose(
            inputs["bounds"].lower, unscaled_bounds.lower * scaling_factor
        )
        np.testing.assert_allclose(
            inputs["bounds"].upper, unscaled_bounds.upper * scaling_factor
        )


def test_noisy_likelihood_problems_use_numerical_derivatives():
    problems = get_benchmark_problems("likelihood_small", additive_noise=True)
    assert all("jac" not in problem["inputs"] for problem in problems.values())


@pytest.mark.parametrize("problem_name", ["logit_5_bad_start", "garch_3_good_start"])
def test_minimize_likelihood_problem(problem_name):
    problem = get_benchmark_problems("likelihood_small")[problem_name]
    res = om.minimize(**problem["inputs"], algorithm="scipy_lbfgsb")
    np.testing.assert_allclose(res.fun, problem["solution"]["value"], rtol=1e-6)
