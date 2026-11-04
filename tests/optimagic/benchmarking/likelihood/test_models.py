from functools import partial

import numpy as np
import pytest
from scipy import special

import optimagic as om
from optimagic.benchmarking.likelihood import models
from optimagic.benchmarking.likelihood.models import _log_normal_interval

# Small versions of all models, covering all options of the data generating processes.
# The second entry says whether the model is correctly specified.
MODELS = {
    "logit": (partial(models.binary_choice, n_obs=300, n_params=5), True),
    "logit_scaled": (
        partial(models.binary_choice, n_obs=300, n_params=6, badly_scaled=True),
        True,
    ),
    "logit_correlated": (
        partial(models.binary_choice, n_obs=300, n_params=4, correlation=0.5),
        True,
    ),
    "probit": (
        partial(models.binary_choice, n_obs=300, n_params=5, model="probit"),
        True,
    ),
    "logit_probit_data": (
        partial(models.binary_choice, n_obs=300, n_params=5, dgp_errors="normal"),
        False,
    ),
    "logit_misclassified": (
        partial(models.binary_choice, n_obs=300, n_params=5, dgp_misclassification=0.1),
        False,
    ),
    "probit_heteroskedastic": (
        partial(
            models.binary_choice,
            n_obs=300,
            n_params=5,
            model="probit",
            dgp_heteroskedasticity=0.5,
        ),
        False,
    ),
    "logit_sign_restricted": (
        partial(models.binary_choice, n_obs=300, n_params=6, n_sign_restricted=3),
        False,
    ),
    "poisson": (partial(models.poisson_regression, n_obs=300, n_params=4), True),
    "poisson_overdispersed": (
        partial(models.poisson_regression, n_obs=300, n_params=4, dgp_dispersion=3.0),
        False,
    ),
    "poisson_underdispersed": (
        partial(models.poisson_regression, n_obs=300, n_params=4, dgp_dispersion=0.4),
        False,
    ),
    "fixed_effects_poisson": (
        partial(models.fixed_effects_poisson, n_groups=5, n_periods=10, n_slopes=3),
        True,
    ),
    "fixed_effects_poisson_overdispersed": (
        partial(
            models.fixed_effects_poisson,
            n_groups=5,
            n_periods=10,
            n_slopes=3,
            dgp_dispersion=2.0,
        ),
        False,
    ),
    "negative_binomial": (
        partial(models.negative_binomial, n_obs=300, n_params=5),
        True,
    ),
    "zero_inflated_poisson": (
        partial(
            models.zero_inflated_poisson,
            n_obs=300,
            n_inflation_params=3,
            n_count_params=4,
        ),
        True,
    ),
    "zero_inflated_poisson_negbin_data": (
        partial(
            models.zero_inflated_poisson,
            n_obs=300,
            n_inflation_params=4,
            n_count_params=3,
            dgp_dispersion=3.0,
        ),
        False,
    ),
    "heteroskedastic_regression": (
        partial(
            models.linear_regression, n_obs=300, n_mean_params=4, n_variance_params=3
        ),
        True,
    ),
    "regression_t_errors": (
        partial(
            models.linear_regression,
            n_obs=300,
            n_mean_params=4,
            n_variance_params=1,
            dgp_error_df=4.0,
        ),
        False,
    ),
    "regression_ignored_heteroskedasticity": (
        partial(
            models.linear_regression,
            n_obs=300,
            n_mean_params=4,
            n_variance_params=1,
            dgp_heteroskedasticity=0.5,
            correlation=0.3,
        ),
        False,
    ),
    "tobit": (partial(models.tobit, n_obs=300, n_params=5), True),
    "tobit_t_errors": (
        partial(models.tobit, n_obs=300, n_params=5, dgp_error_df=4.0),
        False,
    ),
    "tobit_heteroskedastic": (
        partial(models.tobit, n_obs=300, n_params=5, dgp_heteroskedasticity=0.5),
        False,
    ),
    "gamma_regression": (partial(models.gamma_regression, n_obs=300, n_params=5), True),
    "gamma_regression_lognormal_data": (
        partial(models.gamma_regression, n_obs=300, n_params=5, dgp_lognormal=True),
        False,
    ),
    "stochastic_frontier": (
        partial(models.stochastic_frontier, n_obs=300, n_params=5),
        True,
    ),
    "stochastic_frontier_wrong_skew": (
        partial(models.stochastic_frontier, n_obs=300, n_params=5, dgp_wrong_skew=True),
        False,
    ),
    "ordered_probit": (
        partial(models.ordered_probit, n_obs=300, n_slopes=3, n_categories=5),
        True,
    ),
    "ordered_probit_logistic_errors": (
        partial(
            models.ordered_probit,
            n_obs=300,
            n_slopes=3,
            n_categories=4,
            dgp_errors="logistic",
        ),
        False,
    ),
    "ordered_probit_gumbel_errors": (
        partial(
            models.ordered_probit,
            n_obs=300,
            n_slopes=3,
            n_categories=4,
            dgp_errors="gumbel",
        ),
        False,
    ),
    "multinomial_logit": (
        partial(
            models.multinomial_logit,
            n_obs=300,
            n_alternatives=4,
            n_attributes=2,
            n_characteristics=2,
        ),
        True,
    ),
    "multinomial_logit_nested_data": (
        partial(
            models.multinomial_logit,
            n_obs=300,
            n_alternatives=5,
            n_attributes=2,
            n_characteristics=1,
            dgp_nest_dissimilarity=0.5,
        ),
        False,
    ),
    "weibull": (partial(models.weibull_duration, n_obs=300, n_params=4), True),
    "weibull_frailty": (
        partial(
            models.weibull_duration, n_obs=300, n_params=4, dgp_frailty_variance=0.5
        ),
        False,
    ),
    "multivariate_normal": (
        partial(models.multivariate_normal, n_obs=300, dim=4),
        True,
    ),
    "multivariate_normal_t_data": (
        partial(models.multivariate_normal, n_obs=300, dim=3, dgp_df=5.0),
        False,
    ),
    "garch": (partial(models.garch, n_obs=500), True),
    "garch_t_innovations": (
        partial(models.garch, n_obs=500, dgp_innovation_df=5.0),
        False,
    ),
    "garch_t_likelihood": (
        partial(models.garch, n_obs=500, t_likelihood=True, dgp_innovation_df=6.0),
        True,
    ),
    "random_effects_probit": (
        partial(
            models.random_effects_probit, n_individuals=100, n_periods=4, n_params=4
        ),
        True,
    ),
    "random_effects_probit_bimodal_effects": (
        partial(
            models.random_effects_probit,
            n_individuals=100,
            n_periods=4,
            n_params=4,
            dgp_bimodal_effects=True,
        ),
        False,
    ),
    "mixed_logit": (
        partial(
            models.mixed_logit,
            n_individuals=50,
            n_situations=4,
            n_alternatives=3,
            n_fixed=2,
            n_random=2,
            n_draws=20,
        ),
        True,
    ),
    "mixed_logit_correlated_zero_sd": (
        partial(
            models.mixed_logit,
            n_individuals=50,
            n_situations=4,
            n_alternatives=3,
            n_fixed=1,
            n_random=3,
            n_draws=20,
            correlated=True,
            zero_sd=True,
        ),
        True,
    ),
    "mixed_logit_lognormal_coefficients": (
        partial(
            models.mixed_logit,
            n_individuals=50,
            n_situations=4,
            n_alternatives=3,
            n_fixed=1,
            n_random=2,
            n_draws=20,
            correlated=True,
            dgp_lognormal=True,
        ),
        False,
    ),
    "normal_mixture": (partial(models.normal_mixture, n_obs=300), True),
    "normal_mixture_three_components": (
        partial(models.normal_mixture, n_obs=300, dgp_three_components=True),
        False,
    ),
    "heckman_selection": (
        partial(
            models.heckman_selection,
            n_obs=300,
            n_selection_params=4,
            n_outcome_params=3,
        ),
        True,
    ),
    "heckman_selection_t_errors": (
        partial(
            models.heckman_selection,
            n_obs=300,
            n_selection_params=4,
            n_outcome_params=2,
            dgp_error_df=4.0,
        ),
        False,
    ),
}


def _bounds(likelihood):
    k = likelihood.n_params
    lower = likelihood.lower_bounds
    upper = likelihood.upper_bounds
    lower = np.full(k, -np.inf) if lower is None else lower
    upper = np.full(k, np.inf) if upper is None else upper
    return lower, upper


def _interior_point(likelihood, seed):
    """Perturb the true params or the good start, staying inside the bounds."""
    lower, upper = _bounds(likelihood)
    base = likelihood.true_params
    if base is None or not np.isfinite(base).all():
        base = likelihood.good_start
    point = base + 0.1 * np.random.default_rng(seed).normal(size=len(base))
    return np.clip(point, lower + 1e-3, upper - 1e-3)


@pytest.fixture(params=list(MODELS), scope="module")
def model_case(request):
    builder, is_correct = MODELS[request.param]
    return builder(seed=1), is_correct


def test_shapes_and_finite_values(model_case):
    likelihood, _ = model_case
    k = likelihood.n_params
    points = [
        likelihood.good_start,
        likelihood.bad_start,
        _interior_point(likelihood, 0),
    ]
    for x in points:
        contributions = likelihood.fun(x)
        jac = likelihood.jac(x)
        assert contributions.ndim == 1
        assert jac.shape == (len(contributions), k)
        assert np.isfinite(contributions).all()
        assert np.isfinite(jac).all()


@pytest.mark.parametrize("point", ["good_start", "bad_start", "interior"])
def test_jac_matches_numerical_derivative(model_case, point):
    likelihood, _ = model_case
    if point == "interior":
        x = _interior_point(likelihood, 1)
    else:
        x = getattr(likelihood, point)
    lower, upper = _bounds(likelihood)
    numerical = om.first_derivative(
        likelihood.fun, x, bounds=om.Bounds(lower=lower, upper=upper)
    ).derivative
    analytical = likelihood.jac(x)
    scale = max(1.0, np.abs(analytical).max())
    np.testing.assert_allclose(analytical / scale, numerical / scale, atol=1e-6)


def test_start_values_satisfy_bounds(model_case):
    likelihood, _ = model_case
    lower, upper = _bounds(likelihood)
    for x in [likelihood.good_start, likelihood.bad_start]:
        assert x.shape == (likelihood.n_params,)
        assert (x >= lower).all()
        assert (x <= upper).all()


def test_good_start_differs_from_bad_start(model_case):
    likelihood, _ = model_case
    assert not np.allclose(likelihood.good_start, likelihood.bad_start)


def test_true_params_iff_correctly_specified(model_case):
    likelihood, is_correct = model_case
    if is_correct:
        assert likelihood.true_params is not None
        assert likelihood.true_params.shape == (likelihood.n_params,)
    else:
        assert likelihood.true_params is None


@pytest.mark.parametrize("name", ["logit", "mixed_logit", "garch", "ordered_probit"])
def test_simulation_is_deterministic(name):
    builder, _ = MODELS[name]
    first = builder(seed=3)
    second = builder(seed=3)
    other_seed = builder(seed=4)
    x = first.good_start
    np.testing.assert_array_equal(first.fun(x), second.fun(x))
    assert not np.array_equal(first.fun(x), other_seed.fun(x))


def test_sign_restrictions_are_lower_bounds():
    likelihood = models.binary_choice(
        n_obs=300, n_params=6, seed=1, n_sign_restricted=2
    )
    expected = np.array([-np.inf] * 4 + [0.0] * 2)
    np.testing.assert_array_equal(likelihood.lower_bounds, expected)
    assert likelihood.upper_bounds is None


def test_mixed_logit_bounds_only_restrict_cholesky_diagonal():
    likelihood = models.mixed_logit(
        n_individuals=20,
        n_situations=2,
        n_alternatives=3,
        n_fixed=1,
        n_random=2,
        seed=1,
        correlated=True,
    )
    # params: 1 fixed, 2 means, chol entries (0, 0), (1, 0), (1, 1)
    expected = np.array([-np.inf, -np.inf, -np.inf, 0.0, -np.inf, 0.0])
    np.testing.assert_array_equal(likelihood.lower_bounds, expected)


def test_multivariate_normal_true_params_reproduce_covariance():
    likelihood = models.multivariate_normal(n_obs=20_000, dim=3, seed=1)
    true = likelihood.true_params
    assert true is not None
    rows, cols = np.tril_indices(3)
    chol = np.zeros((3, 3))
    chol[rows, cols] = true[3:]
    chol[np.diag_indices(3)] = np.exp(np.diag(chol))
    assert chol[1, 0] < 0
    # the true params are close to the maximum likelihood estimates
    score = -likelihood.jac(true).sum(axis=0) / 20_000
    assert np.abs(score).max() < 0.05


def test_normal_mixture_solution_is_not_unique():
    likelihood = models.normal_mixture(n_obs=100, seed=1)
    assert not likelihood.unique_solution
    swapped = likelihood.good_start[[1, 0, 3, 2, 4]] * np.array([1, 1, 1, 1, -1])
    np.testing.assert_allclose(
        likelihood.fun(likelihood.good_start).sum(), likelihood.fun(swapped).sum()
    )


def test_log_normal_interval_is_accurate_in_tails():
    lower = np.array([-np.inf, -40.0, 39.0, -1.0, 2.0])
    upper = np.array([-38.0, -39.0, np.inf, 1.0, 2.0 + 1e-9])
    got = _log_normal_interval(lower, upper)
    expected = np.array(
        [
            special.log_ndtr(-38.0),
            special.log_ndtr(-39.0)
            + np.log1p(-np.exp(special.log_ndtr(-40.0) - special.log_ndtr(-39.0))),
            special.log_ndtr(-39.0),
            np.log(special.ndtr(1.0) - special.ndtr(-1.0)),
            np.log(1e-9) - 0.5 * 2.0**2 - 0.5 * np.log(2 * np.pi),
        ]
    )
    np.testing.assert_allclose(got, expected, rtol=1e-6)
