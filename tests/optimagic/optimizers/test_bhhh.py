"""Test the BHHH algorithm end to end."""

import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from scipy.special import expit

import optimagic as om
from optimagic import mark
from optimagic.benchmarking.likelihood.problems import (
    LIKELIHOOD_SETS,
    get_likelihood_problems,
)
from optimagic.benchmarking.likelihood.solutions import LIKELIHOOD_SOLUTIONS
from optimagic.exceptions import (
    InvalidAlgoOptionError,
    InvalidFunctionError,
    OptimagicError,
)
from optimagic.optimizers.bhhh import BHHH, LineSearch
from optimagic.utilities import get_rng

# ======================================================================================
# Helpers
# ======================================================================================


@dataclass(frozen=True)
class BenchmarkProblem:
    fun: Callable[..., Any]
    jac: Callable[..., Any]
    params: np.ndarray
    bounds: om.Bounds | None
    solution_x: np.ndarray
    solution_fun: float


@dataclass(frozen=True)
class Logit:
    fun: Callable[..., Any]
    jac: Callable[..., Any]
    gradient: Callable[..., Any]
    solution: np.ndarray


def _get_benchmark_problem(set_name, instance_name, start):
    instance = next(i for i in LIKELIHOOD_SETS[set_name] if i.name == instance_name)
    likelihood = instance.build()
    solution = LIKELIHOOD_SOLUTIONS[instance_name]
    bounds = None
    if likelihood.lower_bounds is not None or likelihood.upper_bounds is not None:
        bounds = om.Bounds(lower=likelihood.lower_bounds, upper=likelihood.upper_bounds)
    return BenchmarkProblem(
        fun=mark.likelihood(likelihood.fun),
        jac=mark.likelihood(likelihood.jac),
        params=likelihood.good_start if start == "good" else likelihood.bad_start,
        bounds=bounds,
        solution_x=np.array(solution["params"]),
        solution_fun=solution["value"],
    )


def _info(res: om.OptimizeResult) -> dict[str, Any]:
    assert res.algorithm_output is not None
    return res.algorithm_output


def _n_iterations(res: om.OptimizeResult) -> int:
    assert res.n_iterations is not None
    return res.n_iterations


def _logit_data(n_obs=500, seed=0):
    rng = get_rng(seed)
    x = np.column_stack([np.ones(n_obs), rng.normal(size=(n_obs, 2))])
    y = rng.uniform(size=n_obs) < expit(x @ np.array([0.5, 1.0, -1.0]))
    return x, y


def _get_logit(x, y):
    """Return log-likelihood contributions of a logit and their Jacobian."""

    @mark.likelihood
    def loglike(params):
        index = x @ params
        return y * index - np.logaddexp(0, index)

    @mark.likelihood
    def loglike_jac(params):
        return (y - expit(x @ params))[:, None] * x

    return loglike, loglike_jac


@pytest.fixture
def logit():
    x, y = _logit_data()
    loglike, loglike_jac = _get_logit(x, y)

    def gradient(params):
        return loglike_jac(params).sum(axis=0)

    reference = om.maximize(
        fun=loglike,
        jac=gradient,
        params=np.zeros(3),
        algorithm="scipy_lbfgsb",
        algo_options={"convergence_ftol_rel": 1e-15, "convergence_gtol_abs": 1e-12},
    )
    return Logit(
        fun=loglike, jac=loglike_jac, gradient=gradient, solution=reference.params
    )


# ======================================================================================
# Benchmark problems
# ======================================================================================

SMALL_GOOD_START = [
    "logit_5",
    "logit_badly_scaled_8",
    "probit_badly_scaled_6",
    "poisson_5",
    "negative_binomial_5",
    "tobit_6",
    "ordered_probit_8",
    "multinomial_logit_11",
    "weibull_5",
    "gamma_regression_6",
    "garch_3",
    "garch_t_4",
    "random_effects_probit_6",
    "mixed_logit_zero_sd_8",
    "stochastic_frontier_6",
    "heckman_10",
    "zero_inflated_poisson_8",
    "fixed_effects_poisson_18",
]

SMALL_BAD_START = [
    "logit_5",
    "probit_badly_scaled_6",
    "negative_binomial_5",
    "multinomial_logit_11",
    "garch_3",
    "tobit_6",
]

MISSPECIFIED = [
    "poisson_overdispersed_severe_5",
    "poisson_underdispersed_severe_5",
    "regression_t_errors_severe_5",
    "logit_misclassified_severe_6",
    "logit_sign_restricted_moderate_8",
    "garch_t_innovations_severe_3",
]

CASES = (
    [("likelihood_small", name, "good") for name in SMALL_GOOD_START]
    + [("likelihood_small", name, "bad") for name in SMALL_BAD_START]
    + [("likelihood_misspecified_small", name, "good") for name in MISSPECIFIED]
    + [("likelihood_misspecified_small", name, "bad") for name in MISSPECIFIED]
)


@pytest.mark.parametrize("set_name, instance_name, start", CASES)
def test_bhhh_solves_benchmark_problem(set_name, instance_name, start):
    problem = _get_benchmark_problem(set_name, instance_name, start)
    res = om.minimize(
        fun=problem.fun,
        jac=problem.jac,
        params=problem.params,
        bounds=problem.bounds,
        algorithm="bhhh",
    )
    assert res.success
    assert _info(res)["decrement"] <= 1e-8
    assert abs(res.fun - problem.solution_fun) < 1e-6
    aaae(res.params, problem.solution_x, decimal=3)


def test_bhhh_with_active_bounds_at_solution():
    problem = _get_benchmark_problem(
        "likelihood_misspecified_small", "logit_sign_restricted_moderate_8", "good"
    )
    res = om.minimize(
        fun=problem.fun,
        jac=problem.jac,
        params=problem.params,
        bounds=problem.bounds,
        algorithm="bhhh",
    )
    active = _info(res)["active_lower_bounds"] + _info(res)["active_upper_bounds"]
    assert res.success
    assert len(active) > 0


# ======================================================================================
# Interfaces and problem formulations
# ======================================================================================


def test_maximize_log_likelihood(logit):
    res = om.maximize(
        fun=logit.fun, jac=logit.jac, params=np.zeros(3), algorithm="bhhh"
    )
    assert res.success
    aaae(res.params, logit.solution, decimal=5)


def test_numerical_derivatives(logit):
    res = om.maximize(fun=logit.fun, params=np.zeros(3), algorithm="bhhh")
    assert res.success
    aaae(res.params, logit.solution, decimal=5)


def test_pytree_params_and_algorithm_instance(logit):
    def fun(params):
        return logit.fun(np.array([params["a"], *params["b"]]))

    res = om.maximize(
        fun=mark.likelihood(fun),
        params={"a": 0.0, "b": np.zeros(2)},
        algorithm=BHHH(globalization=LineSearch(expansion="none")),
    )
    assert res.success
    aaae(res.params["b"], logit.solution[1:], decimal=5)


def test_bound_active_at_solution_matches_lbfgsb(logit):
    bounds = om.Bounds(upper=np.array([np.inf, 0.7, np.inf]))
    res = om.maximize(
        fun=logit.fun,
        jac=logit.jac,
        params=np.zeros(3),
        bounds=bounds,
        algorithm="bhhh",
    )
    expected = om.maximize(
        fun=logit.fun,
        jac=logit.gradient,
        params=np.zeros(3),
        bounds=bounds,
        algorithm="scipy_lbfgsb",
        algo_options={"convergence_ftol_rel": 1e-15, "convergence_gtol_abs": 1e-12},
    )
    assert res.success
    assert res.params[1] == 0.7
    assert _info(res)["active_upper_bounds"] == [1]
    aaae(res.params, expected.params, decimal=5)


def test_start_on_bound_that_is_active_at_solution(logit):
    bounds = om.Bounds(lower=np.array([-np.inf, -np.inf, -0.5]))
    res = om.maximize(
        fun=logit.fun,
        jac=logit.jac,
        params=np.array([0.0, 0.0, -0.5]),
        bounds=bounds,
        algorithm="bhhh",
    )
    assert res.success
    assert res.params[2] == -0.5


def test_start_on_bound_that_is_inactive_at_solution(logit):
    bounds = om.Bounds(lower=np.array([0.0, 0.0, -10.0]))
    res = om.maximize(
        fun=logit.fun,
        jac=logit.jac,
        params=np.array([0.0, 0.0, 0.0]),
        bounds=bounds,
        algorithm="bhhh",
    )
    assert res.success
    aaae(res.params, logit.solution, decimal=5)


def test_non_finite_objective_at_trial_points_is_rejected():
    y = get_rng(0).normal(loc=1.0, size=200)

    @mark.likelihood
    def fun(params):
        out = -((y - params[0]) ** 2) / 2
        return out if params[0] < 1.2 else np.full_like(y, np.nan)

    res = om.maximize(fun=fun, params=np.zeros(1), algorithm="bhhh")
    assert res.success
    aaae(res.params, [y.mean()], decimal=5)


@pytest.mark.parametrize("error_handling", ["raise", "continue"])
def test_objective_that_raises_at_trial_points_is_rejected(error_handling):
    y = get_rng(0).normal(loc=1.0, size=200)

    @mark.likelihood
    def fun(params):
        if params[0] >= 1.2:
            raise ValueError("Objective failed.")
        return -((y - params[0]) ** 2) / 2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = om.maximize(
            fun=fun,
            params=np.zeros(1),
            algorithm="bhhh",
            error_handling=error_handling,
        )
    assert res.success
    aaae(res.params, [y.mean()], decimal=5)
    if error_handling == "raise":
        assert _info(res)["n_failed_fun_evals"] > 0
        assert "ValueError: Objective failed." in str(res.message)


def test_non_finite_derivative_at_start_raises(logit):
    @mark.likelihood
    def jac(params):
        return np.full_like(logit.jac(params), np.nan)

    with pytest.raises(OptimagicError, match="not finite|infinite or NaN"):
        om.maximize(fun=logit.fun, jac=jac, params=np.zeros(3), algorithm="bhhh")


def test_failed_derivative_at_accepted_point_rejects_step(logit):
    start = np.zeros(3)
    failures = []

    @mark.likelihood
    def jac(params):
        if not failures and not np.array_equal(params, start):
            failures.append(params)
            raise ValueError("Derivative failed.")
        return logit.jac(params)

    res = om.maximize(fun=logit.fun, jac=jac, params=start, algorithm="bhhh")
    assert res.success
    assert _info(res)["n_failed_jac_evals"] == 1
    aaae(res.params, logit.solution, decimal=5)


def test_derivative_that_always_fails_after_start_is_reported(logit):
    start = np.zeros(3)

    @mark.likelihood
    def jac(params):
        if not np.array_equal(params, start):
            raise ValueError("Derivative failed.")
        return logit.jac(params)

    res = om.maximize(fun=logit.fun, jac=jac, params=start, algorithm="bhhh")
    assert not res.success
    assert _info(res)["stop_reason"] == "line_search_failure"
    assert "ValueError: Derivative failed." in str(res.message)


def test_wrong_derivative_is_reported_as_failure(logit):
    @mark.likelihood
    def wrong_jac(params):
        return -logit.jac(params)

    res = om.maximize(
        fun=logit.fun, jac=wrong_jac, params=np.zeros(3), algorithm="bhhh"
    )
    assert not res.success
    assert _info(res)["stop_reason"] == "line_search_failure"
    assert "line search" in str(res.message)


def test_least_squares_problem_does_not_report_success():
    # For zero-residual least squares, the decrement does not converge to zero.
    @mark.least_squares
    def fun(params):
        return params

    res = om.minimize(
        fun=fun,
        params=np.arange(1, 4, dtype=float),
        algorithm=om.algos.bhhh(stopping_maxiter=30),
    )
    assert not res.success
    assert _info(res)["decrement"] > 1


def test_scalar_objective_raises():
    with pytest.raises(InvalidFunctionError):
        om.minimize(fun=lambda x: x @ x, params=np.ones(2), algorithm="bhhh")


# ======================================================================================
# Stopping criteria and results
# ======================================================================================


def test_result_fields(logit):
    res = om.maximize(
        fun=logit.fun, jac=logit.jac, params=np.zeros(3), algorithm="bhhh"
    )
    info = _info(res)
    assert np.asarray(res.jac).shape == (3,)
    assert res.hess_inv is None
    assert info["stop_reason"] == "convergence_decrement_tol"
    assert info["rank"] == 3
    assert 1 <= info["condition_number"] < 10
    assert info["active_lower_bounds"] == info["active_upper_bounds"] == []
    n_iterations = _n_iterations(res)
    assert n_iterations > 0
    assert res.n_jac_evals == n_iterations + 1
    assert res.n_fun_evals is not None
    assert res.n_fun_evals >= n_iterations + 1
    assert "robust standard errors" in str(res.message)


@pytest.mark.parametrize(
    "options, stop_reason, n_iterations, success",
    [
        ({"stopping_maxiter": 1}, "stopping_maxiter", 1, False),
        ({"stopping_maxfun": 1}, "stopping_maxfun", 0, False),
        ({"convergence_ftol_abs": 1e10}, "convergence_ftol_abs", 1, True),
        ({"convergence_xtol_rel": 1e10}, "convergence_xtol_rel", 0, True),
    ],
)
def test_stopping_criteria(logit, options, stop_reason, n_iterations, success):
    res = om.maximize(
        fun=logit.fun,
        jac=logit.jac,
        params=np.zeros(3),
        algorithm=om.algos.bhhh(**options),
    )
    assert res.success is success
    assert _info(res)["stop_reason"] == stop_reason
    assert res.n_iterations == n_iterations


def test_large_decrement_tolerance_stops_at_start(logit):
    res = om.maximize(
        fun=logit.fun,
        jac=logit.jac,
        params=np.zeros(3),
        algorithm=om.algos.bhhh().with_convergence(decrement_tol=1e10),
    )
    assert res.success
    assert res.n_iterations == 0
    aaae(res.params, np.zeros(3))


def test_tighter_decrement_tolerance_gets_closer(logit):
    loose, tight = (
        om.maximize(
            fun=logit.fun,
            jac=logit.jac,
            params=np.zeros(3),
            algorithm=om.algos.bhhh(convergence_decrement_tol=tol),
        )
        for tol in (1e-2, 1e-12)
    )
    assert _info(loose)["decrement"] > 1e-12
    assert _info(tight)["decrement"] <= 1e-12
    assert tight.fun >= loose.fun


# ======================================================================================
# Warnings about the problem formulation
# ======================================================================================


def test_warns_about_parameter_without_effect(logit):
    @mark.likelihood
    def fun(params):
        return logit.fun(params[:3])

    with pytest.warns(UserWarning) as record:
        res = om.maximize(fun=fun, params=np.zeros(4), algorithm="bhhh")
    messages = [str(w.message) for w in record]
    assert any("positions [3] are zero at the start" in m for m in messages)
    assert any("nearly singular directions: [3]" in m for m in messages)
    assert res.params[3] == 0
    aaae(res.params[:3], logit.solution, decimal=5)


def test_warns_about_collinear_parameters():
    x, y = _logit_data()
    x = np.column_stack([x, x[:, 1]])
    loglike, loglike_jac = _get_logit(x, y)
    with pytest.warns(UserWarning, match=r"nearly singular directions: \[[13], [13]\]"):
        res = om.maximize(
            fun=loglike, jac=loglike_jac, params=np.zeros(4), algorithm="bhhh"
        )
    assert _info(res)["rank"] == 3


def test_warns_about_fewer_observations_than_parameters():
    x, y = _logit_data(n_obs=2)
    loglike, loglike_jac = _get_logit(x, y)
    with pytest.warns(UserWarning, match="fewer likelihood contributions"):
        om.maximize(
            fun=loglike,
            jac=loglike_jac,
            params=np.zeros(3),
            algorithm=om.algos.bhhh(stopping_maxiter=5),
        )


def test_no_warnings_for_well_posed_problem(logit):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        om.maximize(fun=logit.fun, jac=logit.jac, params=np.zeros(3), algorithm="bhhh")


# ======================================================================================
# Options
# ======================================================================================


@pytest.mark.parametrize(
    "options",
    [
        {"globalization": "trust_region"},
        {"convergence_decrement_tol": -1.0},
        {"stopping_maxiter": 0},
        {"converence_gtol_abs": 1e-8},
    ],
)
def test_invalid_options_raise(options):
    with pytest.raises(InvalidAlgoOptionError):
        BHHH(**options)


def test_globalization_string_and_instance_are_equivalent(logit):
    from_string, from_instance = (
        om.maximize(
            fun=logit.fun,
            jac=logit.jac,
            params=np.zeros(3),
            algorithm=BHHH(globalization=globalization),
        )
        for globalization in ("line_search", LineSearch())
    )
    aaae(from_string.params, from_instance.params, decimal=12)
    assert from_string.n_fun_evals == from_instance.n_fun_evals


# ======================================================================================
# Regression guard on the likelihood benchmark
# ======================================================================================


@pytest.mark.slow
@pytest.mark.parametrize(
    "set_name, min_solved",
    [("likelihood_small", 60), ("likelihood_misspecified_small", 69)],
)
def test_bhhh_on_likelihood_benchmark_set(set_name, min_solved):
    # The failures are bad starts in regions where the OPG is a poor model (e.g. gamma
    # and Weibull regressions). From the bad starts of the normal mixtures, BHHH
    # converges to stationary points that are not the global optimum.
    other_stationary_points = {
        "normal_mixture_5_bad_start",
        "normal_mixture_three_components_severe_5_bad_start",
    }
    problems = get_likelihood_problems(set_name)
    n_solved = 0
    for name, problem in problems.items():
        bounds = om.Bounds(lower=problem["lower_bounds"], upper=problem["upper_bounds"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = om.minimize(
                fun=problem["fun"],
                jac=problem["jac"],
                params=problem["start_x"],
                bounds=bounds,
                algorithm=om.algos.bhhh(stopping_maxfun=20_000),
            )
        target = problem["solution_criterion"]
        solved = abs(res.fun - target) <= 1e-5 * max(1, abs(target))
        if res.success and not solved:
            assert name in other_stationary_points
        n_solved += int(bool(res.success) and solved)
    assert n_solved >= min_solved
