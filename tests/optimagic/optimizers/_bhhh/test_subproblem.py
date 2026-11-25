import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import minimize

from optimagic.optimizers._bhhh.linalg import factorize_opg, unconstrained_step
from optimagic.optimizers._bhhh.subproblem import compute_step, max_feasible_step_length
from optimagic.utilities import get_rng


def _random_problem(seed, n_params=5):
    rng = get_rng(seed)
    jac = rng.normal(size=(50, n_params)) * rng.uniform(0.01, 100, size=n_params)
    jac += 0.3 * rng.normal(size=n_params)
    step, _ = unconstrained_step(factorize_opg(jac))
    # bounds that cut off a random subset of the unconstrained step
    lower = -np.abs(step) * rng.uniform(0, 1.5, size=n_params)
    upper = np.abs(step) * rng.uniform(0, 1.5, size=n_params)
    lower[rng.uniform(size=n_params) < 0.3] = -np.inf
    upper[rng.uniform(size=n_params) < 0.3] = np.inf
    return jac, lower, upper


def test_unconstrained_step_is_returned_if_feasible():
    jac, _, _ = _random_problem(0)
    fact = factorize_opg(jac)
    inf = np.full(5, np.inf)
    assert_allclose(compute_step(fact, -inf, inf)[0], unconstrained_step(fact)[0])
    assert compute_step(fact, -inf, inf)[1] == unconstrained_step(fact)[1]


@pytest.mark.parametrize("seed", range(10))
def test_box_constrained_step_satisfies_kkt_conditions(seed):
    jac, lower, upper = _random_problem(seed)
    fact = factorize_opg(jac)
    step, decrement = compute_step(fact, lower, upper)
    gradient = jac.sum(axis=0)
    opg = jac.T @ jac
    model_gradient = (gradient + opg @ step) / fact.scale

    tol = 1e-6 * np.max(np.abs(gradient / fact.scale))
    at_lower = step <= lower
    at_upper = step >= upper
    free = ~at_lower & ~at_upper
    assert np.all((step >= lower) & (step <= upper))
    assert_allclose(model_gradient[free], 0, atol=tol)
    assert np.all(model_gradient[at_lower] >= -tol)
    assert np.all(model_gradient[at_upper] <= tol)
    model_decrease = -(gradient @ step + step @ opg @ step / 2)
    assert_allclose(decrement, 2 * model_decrease, rtol=1e-6)


@pytest.mark.parametrize("seed", range(5))
def test_box_constrained_step_matches_lbfgsb(seed):
    jac, lower, upper = _random_problem(seed)
    fact = factorize_opg(jac)
    step, _ = compute_step(fact, lower, upper)

    # solve the same quadratic in scaled variables
    gradient = jac.sum(axis=0) / fact.scale
    hessian = jac.T @ jac / np.outer(fact.scale, fact.scale)
    res = minimize(
        lambda q: (gradient @ q + q @ hessian @ q / 2, gradient + hessian @ q),
        x0=np.zeros(5),
        jac=True,
        method="L-BFGS-B",
        bounds=list(zip(lower * fact.scale, upper * fact.scale, strict=True)),
        options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 10_000},
    )
    assert_allclose(step * fact.scale, res.x, atol=1e-5)


def test_step_is_zero_at_bound_constrained_optimum():
    # the gradient pushes the first parameter against its lower bound
    jac = np.array([[1.0, 0.5], [2.0, -0.5], [0.5, 0.0]])
    fact = factorize_opg(jac)
    lower = np.array([0.0, -np.inf])
    upper = np.array([np.inf, np.inf])
    step, decrement = compute_step(fact, lower, upper)
    assert step[0] == 0
    assert_allclose(decrement, 0, atol=1e-12)


def test_max_feasible_step_length():
    x = np.array([0.0, 1.0, 2.0])
    direction = np.array([1.0, -2.0, 0.0])
    lower = np.array([-1.0, 0.0, -np.inf])
    upper = np.array([4.0, 3.0, np.inf])
    assert max_feasible_step_length(x, direction, lower, upper) == 0.5


def test_max_feasible_step_length_without_bounds():
    x = np.zeros(2)
    inf = np.full(2, np.inf)
    assert max_feasible_step_length(x, np.ones(2), -inf, inf) == np.inf
