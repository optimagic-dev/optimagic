"""Simulated likelihood models for the likelihood benchmark sets.

Each public function simulates a dataset from a data generating process and returns a
:class:`SimulatedLikelihood` for a (possibly misspecified) model of that data. All
criterion functions return the negative log-likelihood contributions, i.e. the sum of
the contributions is minimized.

Data are simulated with a seeded random number generator (see ``sampling``), so every
problem is deterministic. Arguments that start with ``dgp_`` change only the data
generating process; when they deviate from their default, the model is misspecified.

"""

from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, signal, special

from optimagic.benchmarking.likelihood import sampling

_LOG_SQRT_2PI = 0.5 * np.log(2 * np.pi)

ParamsFunction = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass(frozen=True)
class SimulatedLikelihood:
    fun: ParamsFunction
    """Negative log-likelihood contributions; maps params (k,) to an array (n_obs,)."""

    jac: ParamsFunction
    """Jacobian of fun; maps params (k,) to an array of shape (n_obs, k)."""

    good_start: NDArray[np.float64]
    """Start values a practitioner would choose without knowing the solution."""

    bad_start: NDArray[np.float64]
    """Start values far from the solution with finite criterion and derivatives."""

    lower_bounds: NDArray[np.float64] | None = None
    """Lower bounds on the params or None if the problem is unbounded from below."""

    upper_bounds: NDArray[np.float64] | None = None
    """Upper bounds on the params or None if the problem is unbounded from above."""

    true_params: NDArray[np.float64] | None = None
    """Params of the data generating process; None if the model is misspecified."""

    unique_solution: bool = True
    """False if the solution is only unique up to relabeling (e.g. mixture models)."""

    @property
    def n_params(self) -> int:
        return len(self.good_start)


# ======================================================================================
# Shared helpers
# ======================================================================================


def _covariates(
    rng: np.random.Generator,
    n_obs: int,
    n_covariates: int,
    correlation: float = 0.0,
) -> NDArray[np.float64]:
    """Draw standard normal covariates with equal pairwise correlation.

    Returns:
        Array of shape (n_obs, n_covariates).

    """
    x = sampling.standard_normal(rng, (n_obs, n_covariates))
    if correlation != 0:
        common = sampling.standard_normal(rng, (n_obs, 1))
        x = np.sqrt(1 - correlation) * x + np.sqrt(correlation) * common
    return x


def _add_intercept(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.column_stack([np.ones(len(x)), x])


def _slopes(n_slopes: int, strength: float) -> NDArray[np.float64]:
    """Create a deterministic slope pattern with alternating signs.

    For uncorrelated standard normal covariates, the linear index has standard
    deviation ``strength``.

    """
    j = np.arange(n_slopes)
    raw = (-1.0) ** j * (0.5 + 0.5 * ((0.618034 * (j + 1)) % 1.0))
    if n_slopes == 0:
        return raw
    return strength * raw / np.linalg.norm(raw)


def _regressor_scales(n_covariates: int, badly_scaled: bool) -> NDArray[np.float64]:
    """Return column scales between 0.01 and 100 if badly_scaled, otherwise ones."""
    if badly_scaled:
        return np.logspace(-2, 2, n_covariates)
    return np.ones(n_covariates)


def _overdispersed_counts(
    rng: np.random.Generator, mean: NDArray[np.float64], dispersion: float
) -> NDArray[np.float64]:
    """Draw counts with the given mean and variance dispersion * mean.

    A dispersion of one gives Poisson counts, a dispersion above one gamma-mixed
    Poisson (negative binomial) counts and a dispersion below one binomial counts.

    """
    if dispersion == 1:
        return sampling.poisson(rng, mean)
    if dispersion > 1:
        excess = dispersion - 1
        rate = sampling.gamma(rng, mean / excess, excess, mean.shape)
        return sampling.poisson(rng, rate)
    probability = 1 - dispersion
    n_trials = np.maximum(np.round(mean / probability), 1)
    return sampling.binomial(rng, n_trials, probability)


def _logsumexp(a: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
    """Compute log(sum(exp(a))) along axis; faster than scipy for finite inputs."""
    a_max = a.max(axis=axis, keepdims=True)
    out = np.log(np.exp(a - a_max).sum(axis=axis, keepdims=True)) + a_max
    return np.squeeze(out, axis=axis)


def _log_normal_pdf(z: NDArray[np.float64]) -> NDArray[np.float64]:
    return -0.5 * z**2 - _LOG_SQRT_2PI


def _inverse_mills(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute phi(z) / Phi(z), accurate also for very negative z."""
    return np.exp(_log_normal_pdf(z) - special.log_ndtr(z))


def _log_normal_interval(
    lower: NDArray[np.float64], upper: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Compute log(Phi(upper) - Phi(lower)) for lower < upper, accurate in the tails."""
    flip = lower > 0
    lo = np.where(flip, -upper, lower)
    hi = np.where(flip, -lower, upper)
    log_hi = special.log_ndtr(hi)
    log_lo = special.log_ndtr(lo)
    with np.errstate(divide="ignore"):
        return log_hi + np.log(-np.expm1(log_lo - log_hi))


def _clip(
    x: NDArray[np.float64],
    lower: NDArray[np.float64] | None,
    upper: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    lower = np.full_like(x, -np.inf) if lower is None else lower
    upper = np.full_like(x, np.inf) if upper is None else upper
    return np.clip(x, lower, upper)


def _simulated_likelihood(
    *,
    fun: ParamsFunction,
    jac: ParamsFunction,
    good_start: NDArray[np.float64],
    bad_start: NDArray[np.float64],
    true_params: NDArray[np.float64] | None,
    lower_bounds: NDArray[np.float64] | None = None,
    upper_bounds: NDArray[np.float64] | None = None,
    unique_solution: bool = True,
) -> SimulatedLikelihood:
    """Create a SimulatedLikelihood with start values clipped to the bounds."""
    return SimulatedLikelihood(
        fun=fun,
        jac=jac,
        good_start=_clip(
            np.asarray(good_start, dtype=float), lower_bounds, upper_bounds
        ),
        bad_start=_clip(np.asarray(bad_start, dtype=float), lower_bounds, upper_bounds),
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        true_params=true_params,
        unique_solution=unique_solution,
    )


# ======================================================================================
# Binary choice: logit and probit
# ======================================================================================


def binary_choice(
    *,
    n_obs: int,
    n_params: int,
    seed: int,
    model: Literal["logit", "probit"] = "logit",
    strength: float = 1.0,
    correlation: float = 0.0,
    badly_scaled: bool = False,
    n_sign_restricted: int = 0,
    dgp_errors: Literal["logistic", "normal"] | None = None,
    dgp_misclassification: float = 0.0,
    dgp_heteroskedasticity: float = 0.0,
) -> SimulatedLikelihood:
    """Simulate a binary logit or probit model with an intercept.

    Args:
        n_obs: Number of observations.
        n_params: Number of params (intercept plus slopes).
        seed: Seed of the random number generator.
        model: The estimated model.
        strength: Standard deviation of the linear index without intercept.
        correlation: Pairwise correlation of the covariates.
        badly_scaled: If True, the covariates have scales between 0.01 and 100.
        n_sign_restricted: The last n_sign_restricted slopes get a lower bound of
            zero. Restrictions on slopes whose true value is negative are binding.
        dgp_errors: Distribution of the latent errors. Defaults to the one implied by
            model.
        dgp_misclassification: Probability that an outcome is flipped.
        dgp_heteroskedasticity: Coefficient of the first covariate in the log standard
            deviation of the latent errors.

    """
    rng = np.random.default_rng(seed)
    n_covariates = n_params - 1
    scales = _regressor_scales(n_covariates, badly_scaled)
    x = _add_intercept(_covariates(rng, n_obs, n_covariates, correlation) * scales)
    true_params = np.r_[0.3, _slopes(n_covariates, strength) / scales]

    default_errors = "logistic" if model == "logit" else "normal"
    errors = default_errors if dgp_errors is None else dgp_errors
    if errors == "logistic":
        epsilon = sampling.standard_logistic(rng, n_obs)
    else:
        epsilon = sampling.standard_normal(rng, n_obs)
    epsilon = epsilon * np.exp(dgp_heteroskedasticity * x[:, 1] / scales[0])
    y = (x @ true_params + epsilon > 0).astype(float)
    if dgp_misclassification > 0:
        flip = sampling.uniform(rng, n_obs) < dgp_misclassification
        y = np.where(flip, 1 - y, y)

    is_correct = (
        errors == default_errors
        and dgp_misclassification == 0
        and dgp_heteroskedasticity == 0
        and n_sign_restricted == 0
    )

    if n_sign_restricted > 0:
        lower = np.full(n_params, -np.inf)
        lower[n_params - n_sign_restricted :] = 0.0
    else:
        lower = None

    share = y.mean()
    if model == "logit":
        fun, jac = _logit_fun, _logit_jac
        intercept = special.logit(share)
    else:
        fun, jac = _probit_fun, _probit_jac
        intercept = special.ndtri(share)

    return _simulated_likelihood(
        fun=partial(fun, x=x, y=y),
        jac=partial(jac, x=x, y=y),
        good_start=np.r_[intercept, np.zeros(n_covariates)],
        bad_start=np.r_[-2.0, -4 * true_params[1:]],
        true_params=true_params if is_correct else None,
        lower_bounds=lower,
    )


def _logit_fun(
    params: NDArray[np.float64], *, x: NDArray[np.float64], y: NDArray[np.float64]
) -> NDArray[np.float64]:
    return -special.log_expit((2 * y - 1) * (x @ params))


def _logit_jac(
    params: NDArray[np.float64], *, x: NDArray[np.float64], y: NDArray[np.float64]
) -> NDArray[np.float64]:
    return -(y - special.expit(x @ params))[:, None] * x


def _probit_fun(
    params: NDArray[np.float64], *, x: NDArray[np.float64], y: NDArray[np.float64]
) -> NDArray[np.float64]:
    return -special.log_ndtr((2 * y - 1) * (x @ params))


def _probit_jac(
    params: NDArray[np.float64], *, x: NDArray[np.float64], y: NDArray[np.float64]
) -> NDArray[np.float64]:
    sign = 2 * y - 1
    return -(sign * _inverse_mills(sign * (x @ params)))[:, None] * x


# ======================================================================================
# Count data: Poisson, fixed effects Poisson, negative binomial, zero inflated Poisson
# ======================================================================================


def poisson_regression(
    *,
    n_obs: int,
    n_params: int,
    seed: int,
    strength: float = 0.5,
    correlation: float = 0.0,
    dgp_dispersion: float = 1.0,
) -> SimulatedLikelihood:
    """Simulate a Poisson regression with an intercept.

    Args:
        n_obs: Number of observations.
        n_params: Number of params (intercept plus slopes).
        seed: Seed of the random number generator.
        strength: Standard deviation of the linear index without intercept.
        correlation: Pairwise correlation of the covariates.
        dgp_dispersion: Ratio of the variance to the mean of the counts. Values other
            than one make the Poisson likelihood a quasi-likelihood.

    """
    rng = np.random.default_rng(seed)
    x = _add_intercept(_covariates(rng, n_obs, n_params - 1, correlation))
    true_params = np.r_[1.0, _slopes(n_params - 1, strength)]
    y = _overdispersed_counts(rng, np.exp(x @ true_params), dgp_dispersion)

    log_mean = np.log(y.mean())
    return _simulated_likelihood(
        fun=partial(_poisson_fun, x=x, y=y),
        jac=partial(_poisson_jac, x=x, y=y),
        good_start=np.r_[log_mean, np.zeros(n_params - 1)],
        bad_start=np.r_[log_mean + 3, -3 * true_params[1:]],
        true_params=true_params if dgp_dispersion == 1 else None,
    )


def _poisson_fun(
    params: NDArray[np.float64], *, x: NDArray[np.float64], y: NDArray[np.float64]
) -> NDArray[np.float64]:
    index = x @ params
    return np.exp(index) - y * index + special.gammaln(y + 1)


def _poisson_jac(
    params: NDArray[np.float64], *, x: NDArray[np.float64], y: NDArray[np.float64]
) -> NDArray[np.float64]:
    return (np.exp(x @ params) - y)[:, None] * x


def fixed_effects_poisson(
    *,
    n_groups: int,
    n_periods: int,
    n_slopes: int,
    seed: int,
    strength: float = 0.5,
    dgp_dispersion: float = 1.0,
) -> SimulatedLikelihood:
    """Simulate a panel Poisson regression with group fixed effects.

    The params are the slopes followed by one fixed effect per group. The covariates
    are correlated with the fixed effects.

    Args:
        n_groups: Number of groups.
        n_periods: Number of observations per group.
        n_slopes: Number of slopes.
        seed: Seed of the random number generator.
        strength: Standard deviation of the linear index without fixed effects.
        dgp_dispersion: Ratio of the variance to the mean of the counts.

    """
    rng = np.random.default_rng(seed)
    n_obs = n_groups * n_periods
    group = np.repeat(np.arange(n_groups), n_periods)
    effects = 1.5 + 0.3 * sampling.standard_normal(rng, n_groups)
    x = _covariates(rng, n_obs, n_slopes) + (effects[group] - 1.5)[:, None]
    slopes = _slopes(n_slopes, strength)
    true_params = np.r_[slopes, effects]
    y = _overdispersed_counts(rng, np.exp(x @ slopes + effects[group]), dgp_dispersion)

    return _simulated_likelihood(
        fun=partial(_fixed_effects_poisson_fun, x=x, y=y, group=group),
        jac=partial(_fixed_effects_poisson_jac, x=x, y=y, group=group),
        good_start=np.r_[np.zeros(n_slopes), np.full(n_groups, np.log(y.mean()))],
        bad_start=np.r_[-3 * slopes, np.full(n_groups, np.log(y.mean()) - 2)],
        true_params=true_params if dgp_dispersion == 1 else None,
    )


def _fixed_effects_poisson_fun(
    params: NDArray[np.float64],
    *,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    group: NDArray[np.int64],
) -> NDArray[np.float64]:
    n_slopes = x.shape[1]
    index = x @ params[:n_slopes] + params[n_slopes:][group]
    return np.exp(index) - y * index + special.gammaln(y + 1)


def _fixed_effects_poisson_jac(
    params: NDArray[np.float64],
    *,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    group: NDArray[np.int64],
) -> NDArray[np.float64]:
    n_slopes = x.shape[1]
    residual = np.exp(x @ params[:n_slopes] + params[n_slopes:][group]) - y
    out = np.zeros((len(y), len(params)))
    out[:, :n_slopes] = residual[:, None] * x
    out[np.arange(len(y)), n_slopes + group] = residual
    return out


def negative_binomial(
    *,
    n_obs: int,
    n_params: int,
    seed: int,
    strength: float = 0.5,
    dispersion: float = 0.7,
) -> SimulatedLikelihood:
    """Simulate a negative binomial (NB2) regression.

    The params are the regression coefficients (with intercept) followed by the log of
    the dispersion parameter alpha, where Var(y) = mu + alpha * mu**2.

    Args:
        n_obs: Number of observations.
        n_params: Number of params.
        seed: Seed of the random number generator.
        strength: Standard deviation of the linear index without intercept.
        dispersion: The true alpha.

    """
    rng = np.random.default_rng(seed)
    n_slopes = n_params - 2
    x = _add_intercept(_covariates(rng, n_obs, n_slopes))
    beta = np.r_[1.0, _slopes(n_slopes, strength)]
    mean = np.exp(x @ beta)
    rate = sampling.gamma(rng, 1 / dispersion, dispersion * mean, n_obs)
    y = sampling.poisson(rng, rate)

    log_mean = np.log(y.mean())
    return _simulated_likelihood(
        fun=partial(_negative_binomial_fun, x=x, y=y),
        jac=partial(_negative_binomial_jac, x=x, y=y),
        good_start=np.r_[log_mean, np.zeros(n_slopes), 0.0],
        bad_start=np.r_[log_mean + 2, -3 * beta[1:], 3.0],
        true_params=np.r_[beta, np.log(dispersion)],
    )


def _negative_binomial_fun(
    params: NDArray[np.float64], *, x: NDArray[np.float64], y: NDArray[np.float64]
) -> NDArray[np.float64]:
    index = x @ params[:-1]
    r = np.exp(-params[-1])
    log_r_plus_mean = np.logaddexp(np.log(r), index)
    loglike = (
        special.gammaln(y + r)
        - special.gammaln(r)
        - special.gammaln(y + 1)
        + r * (np.log(r) - log_r_plus_mean)
        + y * (index - log_r_plus_mean)
    )
    return -loglike


def _negative_binomial_jac(
    params: NDArray[np.float64], *, x: NDArray[np.float64], y: NDArray[np.float64]
) -> NDArray[np.float64]:
    mean = np.exp(x @ params[:-1])
    r = np.exp(-params[-1])
    d_index = r * (y - mean) / (r + mean)
    d_r = (
        special.digamma(y + r)
        - special.digamma(r)
        + np.log(r / (r + mean))
        + (mean - y) / (r + mean)
    )
    return -np.column_stack([d_index[:, None] * x, -r * d_r])


def zero_inflated_poisson(
    *,
    n_obs: int,
    n_inflation_params: int,
    n_count_params: int,
    seed: int,
    dgp_dispersion: float | None = None,
) -> SimulatedLikelihood:
    """Simulate a zero inflated Poisson regression.

    The params are the coefficients of the logit model for structural zeros followed by
    the coefficients of the Poisson model. Both include an intercept and share the
    leading covariates.

    Args:
        n_obs: Number of observations.
        n_inflation_params: Number of params of the inflation model.
        n_count_params: Number of params of the count model.
        seed: Seed of the random number generator.
        dgp_dispersion: If not None, the data are negative binomial counts with this
            dispersion (variance over mean) and without zero inflation.

    """
    rng = np.random.default_rng(seed)
    n_covariates = max(n_inflation_params, n_count_params) - 1
    covariates = _covariates(rng, n_obs, n_covariates)
    z = _add_intercept(covariates[:, : n_inflation_params - 1])
    x = _add_intercept(covariates[:, : n_count_params - 1])
    gamma = np.r_[-1.0, _slopes(n_inflation_params - 1, 0.5)]
    beta = np.r_[1.0, _slopes(n_count_params - 1, 0.4)[::-1]]

    if dgp_dispersion is None:
        counts = sampling.poisson(rng, np.exp(x @ beta))
        structural_zero = sampling.uniform(rng, n_obs) < special.expit(z @ gamma)
        y = np.where(structural_zero, 0.0, counts)
    else:
        y = _overdispersed_counts(rng, np.exp(x @ beta), dgp_dispersion)

    log_mean = np.log(y.mean())
    return _simulated_likelihood(
        fun=partial(_zero_inflated_poisson_fun, z=z, x=x, y=y),
        jac=partial(_zero_inflated_poisson_jac, z=z, x=x, y=y),
        good_start=np.r_[
            np.zeros(n_inflation_params), log_mean, np.zeros(len(beta) - 1)
        ],
        bad_start=np.r_[3.0, -3 * gamma[1:], log_mean + 2, -3 * beta[1:]],
        true_params=np.r_[gamma, beta] if dgp_dispersion is None else None,
    )


def _zero_inflated_poisson_loglike(
    params: NDArray[np.float64],
    z: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> tuple[NDArray[np.float64], ...]:
    inflation_index = z @ params[: z.shape[1]]
    count_index = x @ params[z.shape[1] :]
    mean = np.exp(count_index)
    log_inflation = special.log_expit(inflation_index)
    log_no_inflation = special.log_expit(-inflation_index)
    loglike = np.where(
        y == 0,
        np.logaddexp(log_inflation, log_no_inflation - mean),
        log_no_inflation + y * count_index - mean - special.gammaln(y + 1),
    )
    return loglike, log_inflation, mean


def _zero_inflated_poisson_fun(
    params: NDArray[np.float64],
    *,
    z: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> NDArray[np.float64]:
    return -_zero_inflated_poisson_loglike(params, z, x, y)[0]


def _zero_inflated_poisson_jac(
    params: NDArray[np.float64],
    *,
    z: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> NDArray[np.float64]:
    loglike, log_inflation, mean = _zero_inflated_poisson_loglike(params, z, x, y)
    inflation = np.exp(log_inflation)
    is_zero = y == 0
    # posterior probability that a zero is structural
    structural = np.zeros_like(y)
    structural[is_zero] = np.exp(log_inflation[is_zero] - loglike[is_zero])
    d_inflation = np.where(is_zero, structural - inflation, -inflation)
    d_count = np.where(is_zero, -(1 - structural) * mean, y - mean)
    return -np.column_stack([d_inflation[:, None] * z, d_count[:, None] * x])


# ======================================================================================
# Continuous outcomes: linear regression, Tobit, gamma regression, stochastic frontier
# ======================================================================================


def linear_regression(
    *,
    n_obs: int,
    n_mean_params: int,
    n_variance_params: int,
    seed: int,
    correlation: float = 0.0,
    dgp_error_df: float | None = None,
    dgp_heteroskedasticity: float = 0.0,
) -> SimulatedLikelihood:
    """Simulate a normal linear regression with multiplicative heteroskedasticity.

    The params are the mean coefficients beta followed by the variance coefficients
    gamma, with log sd(y_i) = z_i @ gamma. Both x and z include an intercept and z uses
    the leading covariates of x.

    Args:
        n_obs: Number of observations.
        n_mean_params: Number of mean params (intercept plus slopes).
        n_variance_params: Number of variance params (at most n_mean_params).
        seed: Seed of the random number generator.
        correlation: Pairwise correlation of the covariates.
        dgp_error_df: If not None, the errors are t distributed with this many degrees
            of freedom instead of normal.
        dgp_heteroskedasticity: Additional coefficient of the first covariate in the
            log standard deviation of the errors that the model ignores.

    """
    rng = np.random.default_rng(seed)
    x = _add_intercept(_covariates(rng, n_obs, n_mean_params - 1, correlation))
    z = x[:, :n_variance_params]
    beta = np.r_[2.0, _slopes(n_mean_params - 1, 1.0)]
    gamma = np.r_[0.0, _slopes(n_variance_params - 1, 0.5)[::-1]]

    if dgp_error_df is None:
        errors = sampling.standard_normal(rng, n_obs)
    else:
        errors = sampling.standardized_t(rng, dgp_error_df, n_obs)
    log_sd = z @ gamma + dgp_heteroskedasticity * x[:, 1]
    y = x @ beta + np.exp(log_sd) * errors

    is_correct = dgp_error_df is None and dgp_heteroskedasticity == 0
    log_sd_y = np.log(y.std())
    return _simulated_likelihood(
        fun=partial(_linear_regression_fun, x=x, z=z, y=y),
        jac=partial(_linear_regression_jac, x=x, z=z, y=y),
        good_start=np.r_[
            y.mean(), np.zeros(n_mean_params - 1), log_sd_y, np.zeros(len(gamma) - 1)
        ],
        bad_start=np.r_[0.0, -3 * beta[1:], log_sd_y - 3, -3 * gamma[1:]],
        true_params=np.r_[beta, gamma] if is_correct else None,
    )


def _linear_regression_fun(
    params: NDArray[np.float64],
    *,
    x: NDArray[np.float64],
    z: NDArray[np.float64],
    y: NDArray[np.float64],
) -> NDArray[np.float64]:
    n_mean = x.shape[1]
    log_sd = z @ params[n_mean:]
    residual = y - x @ params[:n_mean]
    return _LOG_SQRT_2PI + log_sd + 0.5 * (residual * np.exp(-log_sd)) ** 2


def _linear_regression_jac(
    params: NDArray[np.float64],
    *,
    x: NDArray[np.float64],
    z: NDArray[np.float64],
    y: NDArray[np.float64],
) -> NDArray[np.float64]:
    n_mean = x.shape[1]
    precision = np.exp(-2 * z @ params[n_mean:])
    residual = y - x @ params[:n_mean]
    d_mean = -(residual * precision)[:, None] * x
    d_variance = (1 - residual**2 * precision)[:, None] * z
    return np.column_stack([d_mean, d_variance])


def tobit(
    *,
    n_obs: int,
    n_params: int,
    seed: int,
    dgp_error_df: float | None = None,
    dgp_heteroskedasticity: float = 0.0,
) -> SimulatedLikelihood:
    """Simulate a Tobit model, left censored at zero.

    The params are the coefficients (with intercept) followed by the log of the error
    standard deviation.

    Args:
        n_obs: Number of observations.
        n_params: Number of params.
        seed: Seed of the random number generator.
        dgp_error_df: If not None, the errors are t distributed with this many degrees
            of freedom instead of normal.
        dgp_heteroskedasticity: Coefficient of the first covariate in the log standard
            deviation of the errors that the model ignores.

    """
    rng = np.random.default_rng(seed)
    n_slopes = n_params - 2
    x = _add_intercept(_covariates(rng, n_obs, n_slopes))
    beta = np.r_[0.5, _slopes(n_slopes, 1.0)]
    sd = 1.5
    if dgp_error_df is None:
        errors = sampling.standard_normal(rng, n_obs)
    else:
        errors = sampling.standardized_t(rng, dgp_error_df, n_obs)
    errors = errors * sd * np.exp(dgp_heteroskedasticity * x[:, 1])
    y = np.maximum(x @ beta + errors, 0.0)

    is_correct = dgp_error_df is None and dgp_heteroskedasticity == 0
    log_sd_y = np.log(y.std())
    return _simulated_likelihood(
        fun=partial(_tobit_fun, x=x, y=y),
        jac=partial(_tobit_jac, x=x, y=y),
        good_start=np.r_[y.mean(), np.zeros(n_slopes), log_sd_y],
        bad_start=np.r_[-3.0, -3 * beta[1:], log_sd_y - 3],
        true_params=np.r_[beta, np.log(sd)] if is_correct else None,
    )


def _tobit_fun(
    params: NDArray[np.float64], *, x: NDArray[np.float64], y: NDArray[np.float64]
) -> NDArray[np.float64]:
    index = x @ params[:-1]
    log_sd = params[-1]
    standardized = (y - index) * np.exp(-log_sd)
    loglike = np.where(
        y > 0,
        _log_normal_pdf(standardized) - log_sd,
        special.log_ndtr(standardized),
    )
    return -loglike


def _tobit_jac(
    params: NDArray[np.float64], *, x: NDArray[np.float64], y: NDArray[np.float64]
) -> NDArray[np.float64]:
    index = x @ params[:-1]
    sd = np.exp(params[-1])
    standardized = (y - index) / sd
    # at censored observations, standardized is -index / sd
    mills = _inverse_mills(standardized)
    d_index = np.where(y > 0, standardized / sd, -mills / sd)
    d_log_sd = np.where(y > 0, standardized**2 - 1, -mills * standardized)
    return -np.column_stack([d_index[:, None] * x, d_log_sd])


def gamma_regression(
    *,
    n_obs: int,
    n_params: int,
    seed: int,
    dgp_lognormal: bool = False,
) -> SimulatedLikelihood:
    """Simulate a gamma regression with log link.

    The params are the coefficients of the log mean (with intercept) followed by the
    log of the shape parameter.

    Args:
        n_obs: Number of observations.
        n_params: Number of params.
        seed: Seed of the random number generator.
        dgp_lognormal: If True, the outcomes are lognormal with the same mean and
            variance as the gamma distribution.

    """
    rng = np.random.default_rng(seed)
    n_slopes = n_params - 2
    x = _add_intercept(_covariates(rng, n_obs, n_slopes))
    beta = np.r_[1.0, _slopes(n_slopes, 0.5)]
    shape = 2.0
    mean = np.exp(x @ beta)
    if dgp_lognormal:
        sd = np.sqrt(np.log(1 + 1 / shape))
        y = mean * np.exp(sd * sampling.standard_normal(rng, n_obs) - 0.5 * sd**2)
    else:
        y = sampling.gamma(rng, shape, mean / shape, n_obs)

    log_mean = np.log(y.mean())
    return _simulated_likelihood(
        fun=partial(_gamma_regression_fun, x=x, y=y),
        jac=partial(_gamma_regression_jac, x=x, y=y),
        good_start=np.r_[log_mean, np.zeros(n_slopes), 0.0],
        bad_start=np.r_[log_mean - 3, -3 * beta[1:], 4.0],
        true_params=None if dgp_lognormal else np.r_[beta, np.log(shape)],
    )


def _gamma_regression_fun(
    params: NDArray[np.float64], *, x: NDArray[np.float64], y: NDArray[np.float64]
) -> NDArray[np.float64]:
    log_shape = params[-1]
    shape = np.exp(log_shape)
    log_ratio = np.log(y) - x @ params[:-1]
    loglike = (
        shape * (log_shape + log_ratio)
        - shape * np.exp(log_ratio)
        - special.gammaln(shape)
        - np.log(y)
    )
    return -loglike


def _gamma_regression_jac(
    params: NDArray[np.float64], *, x: NDArray[np.float64], y: NDArray[np.float64]
) -> NDArray[np.float64]:
    log_shape = params[-1]
    shape = np.exp(log_shape)
    log_ratio = np.log(y) - x @ params[:-1]
    ratio = np.exp(log_ratio)
    d_index = shape * (ratio - 1)
    d_log_shape = shape * (log_shape + log_ratio + 1 - ratio - special.digamma(shape))
    return -np.column_stack([d_index[:, None] * x, d_log_shape])


def stochastic_frontier(
    *,
    n_obs: int,
    n_params: int,
    seed: int,
    dgp_wrong_skew: bool = False,
) -> SimulatedLikelihood:
    """Simulate a normal-half normal stochastic production frontier.

    The params are the frontier coefficients (with intercept), the log standard
    deviation of the noise and the standard deviation of the inefficiency term, which
    is bounded below by zero.

    Args:
        n_obs: Number of observations.
        n_params: Number of params.
        seed: Seed of the random number generator.
        dgp_wrong_skew: If True, the inefficiency term enters with the wrong sign, so
            the residuals are skewed to the right and the solution has no inefficiency,
            i.e. it lies on the bound.

    """
    rng = np.random.default_rng(seed)
    n_slopes = n_params - 3
    x = _add_intercept(_covariates(rng, n_obs, n_slopes))
    beta = np.r_[1.0, _slopes(n_slopes, 0.5)]
    sd_noise, sd_inefficiency = 0.5, 1.0
    noise = sd_noise * sampling.standard_normal(rng, n_obs)
    inefficiency = sd_inefficiency * np.abs(sampling.standard_normal(rng, n_obs))
    sign = 1.0 if dgp_wrong_skew else -1.0
    y = x @ beta + noise + sign * inefficiency

    lower = np.r_[np.full(n_params - 1, -np.inf), 0.0]
    sd_y = y.std()
    return _simulated_likelihood(
        fun=partial(_stochastic_frontier_fun, x=x, y=y),
        jac=partial(_stochastic_frontier_jac, x=x, y=y),
        good_start=np.r_[y.mean(), np.zeros(n_slopes), np.log(sd_y), 0.5 * sd_y],
        bad_start=np.r_[y.mean() + 5, -3 * beta[1:], np.log(sd_y) - 3, 5 * sd_y],
        true_params=(
            None if dgp_wrong_skew else np.r_[beta, np.log(sd_noise), sd_inefficiency]
        ),
        lower_bounds=lower,
    )


def _stochastic_frontier_parts(
    params: NDArray[np.float64], x: NDArray[np.float64], y: NDArray[np.float64]
) -> tuple[NDArray[np.float64], ...]:
    residual = y - x @ params[:-2]
    sd_noise = np.exp(params[-2])
    sd_inefficiency = params[-1]
    sd = np.sqrt(sd_noise**2 + sd_inefficiency**2)
    skew_arg = -residual * sd_inefficiency / (sd_noise * sd)
    return residual, sd_noise, sd_inefficiency, sd, skew_arg


def _stochastic_frontier_fun(
    params: NDArray[np.float64], *, x: NDArray[np.float64], y: NDArray[np.float64]
) -> NDArray[np.float64]:
    residual, _, _, sd, skew_arg = _stochastic_frontier_parts(params, x, y)
    loglike = (
        np.log(2)
        - np.log(sd)
        + _log_normal_pdf(residual / sd)
        + special.log_ndtr(skew_arg)
    )
    return -loglike


def _stochastic_frontier_jac(
    params: NDArray[np.float64], *, x: NDArray[np.float64], y: NDArray[np.float64]
) -> NDArray[np.float64]:
    residual, sd_noise, sd_ineff, sd, skew_arg = _stochastic_frontier_parts(
        params, x, y
    )
    mills = _inverse_mills(skew_arg)
    d_residual = -residual / sd**2 - mills * sd_ineff / (sd_noise * sd)
    d_sd = -1 / sd + residual**2 / sd**3 - mills * skew_arg / sd
    d_sd_ineff = d_sd * sd_ineff / sd - mills * residual / (sd_noise * sd)
    d_sd_noise = d_sd * sd_noise / sd + mills * residual * sd_ineff / (sd_noise**2 * sd)
    return -np.column_stack(
        [-d_residual[:, None] * x, d_sd_noise * sd_noise, d_sd_ineff]
    )


# ======================================================================================
# Ordered and multinomial choice
# ======================================================================================


def ordered_probit(
    *,
    n_obs: int,
    n_slopes: int,
    n_categories: int,
    seed: int,
    dgp_errors: Literal["normal", "logistic", "gumbel"] = "normal",
) -> SimulatedLikelihood:
    """Simulate an ordered probit model.

    The params are the slopes (without intercept) followed by the cutpoints, which are
    parametrized as the first cutpoint and the logs of the increments.

    Args:
        n_obs: Number of observations.
        n_slopes: Number of slopes.
        n_categories: Number of outcome categories.
        seed: Seed of the random number generator.
        dgp_errors: Distribution of the latent errors, standardized to mean zero and
            unit variance.

    """
    rng = np.random.default_rng(seed)
    x = _covariates(rng, n_obs, n_slopes)
    beta = _slopes(n_slopes, 1.0)
    if dgp_errors == "normal":
        errors = sampling.standard_normal(rng, n_obs)
    elif dgp_errors == "logistic":
        errors = sampling.standard_logistic(rng, n_obs) * np.sqrt(3) / np.pi
    else:
        gumbel = sampling.standard_gumbel(rng, n_obs)
        errors = (gumbel - np.euler_gamma) * np.sqrt(6) / np.pi
    probabilities = np.linspace(0, 1, n_categories + 1)[1:-1]
    cutpoints = special.ndtri(probabilities) * np.sqrt(2)
    y = np.searchsorted(cutpoints, x @ beta + errors)

    shares = np.cumsum(np.bincount(y, minlength=n_categories))[:-1] / n_obs
    good_cutpoints = special.ndtri(shares)
    bad_cutpoints = 3.0 + 0.1 * np.arange(n_categories - 1)
    return _simulated_likelihood(
        fun=partial(_ordered_probit_fun, x=x, y=y),
        jac=partial(_ordered_probit_jac, x=x, y=y),
        good_start=np.r_[np.zeros(n_slopes), _cutpoints_to_params(good_cutpoints)],
        bad_start=np.r_[-3 * beta, _cutpoints_to_params(bad_cutpoints)],
        true_params=(
            np.r_[beta, _cutpoints_to_params(cutpoints)]
            if dgp_errors == "normal"
            else None
        ),
    )


def _cutpoints_to_params(cutpoints: NDArray[np.floating]) -> NDArray[np.float64]:
    return np.r_[cutpoints[0], np.log(np.diff(cutpoints))]


def _ordered_probit_parts(
    params: NDArray[np.float64], x: NDArray[np.float64], y: NDArray[np.int64]
) -> tuple[NDArray[np.float64], ...]:
    n_slopes = x.shape[1]
    cut_params = params[n_slopes:]
    cutpoints = np.cumsum(np.r_[cut_params[0], np.exp(cut_params[1:])])
    padded = np.r_[-np.inf, cutpoints, np.inf]
    index = x @ params[:n_slopes]
    upper = padded[y + 1] - index
    lower = padded[y] - index
    return upper, lower, _log_normal_interval(lower, upper)


def _ordered_probit_fun(
    params: NDArray[np.float64], *, x: NDArray[np.float64], y: NDArray[np.int64]
) -> NDArray[np.float64]:
    return -_ordered_probit_parts(params, x, y)[2]


def _ordered_probit_jac(
    params: NDArray[np.float64], *, x: NDArray[np.float64], y: NDArray[np.int64]
) -> NDArray[np.float64]:
    n_obs, n_slopes = x.shape
    n_cutpoints = len(params) - n_slopes
    upper, lower, loglike = _ordered_probit_parts(params, x, y)
    d_upper = np.exp(_log_normal_pdf(upper) - loglike)
    d_lower = -np.exp(_log_normal_pdf(lower) - loglike)

    # derivative with respect to the cutpoints, shape (n_obs, n_cutpoints)
    d_cutpoints = np.zeros((n_obs, n_cutpoints + 2))
    rows = np.arange(n_obs)
    d_cutpoints[rows, y + 1] += d_upper
    d_cutpoints[rows, y] += d_lower
    d_cutpoints = d_cutpoints[:, 1:-1]

    # cutpoint j depends on cut param m <= j; with increments on the log scale
    d_cut_params = np.cumsum(d_cutpoints[:, ::-1], axis=1)[:, ::-1]
    d_cut_params[:, 1:] *= np.exp(params[n_slopes + 1 :])

    d_slopes = -(d_upper + d_lower)[:, None] * x
    return -np.column_stack([d_slopes, d_cut_params])


def multinomial_logit(
    *,
    n_obs: int,
    n_alternatives: int,
    n_attributes: int,
    n_characteristics: int,
    seed: int,
    dgp_nest_dissimilarity: float = 1.0,
) -> SimulatedLikelihood:
    """Simulate a multinomial logit model with attributes and characteristics.

    Utilities are V_ij = w_ij @ beta + x_i @ gamma_j with gamma_0 = 0, where w are
    alternative specific attributes and x are individual characteristics including an
    intercept. The params are beta followed by the rows of gamma for the alternatives
    1, ..., n_alternatives - 1, so there are n_attributes + (n_alternatives - 1) *
    (n_characteristics + 1) params.

    Args:
        n_obs: Number of observations.
        n_alternatives: Number of alternatives.
        n_attributes: Number of alternative specific attributes.
        n_characteristics: Number of individual characteristics (without intercept).
        seed: Seed of the random number generator.
        dgp_nest_dissimilarity: If below one, the choices are generated from a nested
            logit model in which alternative 0 is alone and the other alternatives
            form two nests with this dissimilarity parameter. This violates the
            independence of irrelevant alternatives assumed by the model.

    """
    rng = np.random.default_rng(seed)
    w = sampling.standard_normal(rng, (n_obs, n_alternatives, n_attributes))
    x = _add_intercept(_covariates(rng, n_obs, n_characteristics))
    beta = _slopes(n_attributes, 1.0)
    alt = np.arange(1, n_alternatives)[:, None]
    col = np.arange(n_characteristics + 1)[None, :]
    gamma = 0.6 * np.cos(1.7 * alt + 2.3 * col + 0.4) / np.sqrt(n_characteristics + 1)
    true_params = np.r_[beta, gamma.flatten()]

    utility = _multinomial_logit_utility(true_params, w, x)
    if dgp_nest_dissimilarity == 1:
        y = np.argmax(utility + sampling.standard_gumbel(rng, utility.shape), axis=1)
    else:
        log_probs = _nested_logit_log_probabilities(utility, dgp_nest_dissimilarity)
        y = sampling.categorical(rng, np.exp(log_probs))

    return _simulated_likelihood(
        fun=partial(_multinomial_logit_fun, w=w, x=x, y=y),
        jac=partial(_multinomial_logit_jac, w=w, x=x, y=y),
        good_start=np.zeros_like(true_params),
        bad_start=-3 * true_params,
        true_params=true_params if dgp_nest_dissimilarity == 1 else None,
    )


def _nested_logit_log_probabilities(
    utility: NDArray[np.float64], dissimilarity: float
) -> NDArray[np.float64]:
    """Compute nested logit choice probabilities, with alternative 0 in its own nest."""
    n_alternatives = utility.shape[1]
    split = 1 + (n_alternatives - 1) // 2
    nests = [np.arange(1), np.arange(1, split), np.arange(split, n_alternatives)]
    dissimilarities = [1.0, dissimilarity, dissimilarity]
    log_probs = np.empty_like(utility)
    inclusive = []
    for nest, lam in zip(nests, dissimilarities, strict=True):
        scaled = utility[:, nest] / lam
        nest_inclusive = special.logsumexp(scaled, axis=1)
        log_probs[:, nest] = scaled - nest_inclusive[:, None]
        inclusive.append(lam * nest_inclusive)
    inclusive_arr = np.column_stack(inclusive)
    log_nest_probs = inclusive_arr - special.logsumexp(inclusive_arr, axis=1)[:, None]
    for i, nest in enumerate(nests):
        log_probs[:, nest] += log_nest_probs[:, [i]]
    return log_probs


def _multinomial_logit_utility(
    params: NDArray[np.float64], w: NDArray[np.float64], x: NDArray[np.float64]
) -> NDArray[np.float64]:
    n_obs, n_alternatives, n_attributes = w.shape
    gamma = params[n_attributes:].reshape(n_alternatives - 1, x.shape[1])
    utility = w @ params[:n_attributes]
    utility[:, 1:] += x @ gamma.T
    return utility


def _multinomial_logit_fun(
    params: NDArray[np.float64],
    *,
    w: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.int64],
) -> NDArray[np.float64]:
    utility = _multinomial_logit_utility(params, w, x)
    chosen = utility[np.arange(len(y)), y]
    return _logsumexp(utility, axis=1) - chosen


def _multinomial_logit_jac(
    params: NDArray[np.float64],
    *,
    w: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.int64],
) -> NDArray[np.float64]:
    n_obs, n_alternatives, _ = w.shape
    utility = _multinomial_logit_utility(params, w, x)
    probs = special.softmax(utility, axis=1)
    rows = np.arange(n_obs)
    d_beta = w[rows, y] - np.einsum("ij,ijk->ik", probs, w)
    chosen = np.zeros_like(probs)
    chosen[rows, y] = 1.0
    d_gamma = (chosen - probs)[:, 1:, None] * x[:, None, :]
    return -np.column_stack([d_beta, d_gamma.reshape(n_obs, -1)])


# ======================================================================================
# Duration: Weibull proportional hazards with censoring
# ======================================================================================


def weibull_duration(
    *,
    n_obs: int,
    n_params: int,
    seed: int,
    dgp_frailty_variance: float = 0.0,
) -> SimulatedLikelihood:
    """Simulate a Weibull proportional hazards model with random right censoring.

    The params are the coefficients of the log hazard scale (with intercept) followed
    by the log of the Weibull shape.

    Args:
        n_obs: Number of observations.
        n_params: Number of params.
        seed: Seed of the random number generator.
        dgp_frailty_variance: Variance of a gamma distributed multiplicative frailty
            with mean one that the model ignores.

    """
    rng = np.random.default_rng(seed)
    n_slopes = n_params - 2
    x = _add_intercept(_covariates(rng, n_obs, n_slopes))
    beta = np.r_[-0.5, _slopes(n_slopes, 0.5)]
    shape = 1.5
    hazard_scale = np.exp(x @ beta)
    if dgp_frailty_variance > 0:
        hazard_scale = hazard_scale * sampling.gamma(
            rng, 1 / dgp_frailty_variance, dgp_frailty_variance, n_obs
        )
    duration = (sampling.exponential(rng, n_obs) / hazard_scale) ** (1 / shape)
    censoring = 2 * np.median(duration) * sampling.exponential(rng, n_obs)
    t = np.minimum(duration, censoring)
    event = (duration <= censoring).astype(float)

    return _simulated_likelihood(
        fun=partial(_weibull_fun, x=x, log_t=np.log(t), event=event),
        jac=partial(_weibull_jac, x=x, log_t=np.log(t), event=event),
        good_start=np.r_[np.log(event.sum() / t.sum()), np.zeros(n_slopes), 0.0],
        bad_start=np.r_[2.0, -3 * beta[1:], np.log(4.0)],
        true_params=(np.r_[beta, np.log(shape)] if dgp_frailty_variance == 0 else None),
    )


def _weibull_fun(
    params: NDArray[np.float64],
    *,
    x: NDArray[np.float64],
    log_t: NDArray[np.float64],
    event: NDArray[np.float64],
) -> NDArray[np.float64]:
    index = x @ params[:-1]
    log_shape = params[-1]
    shape = np.exp(log_shape)
    log_cumulative_hazard = index + shape * log_t
    loglike = event * (index + log_shape + (shape - 1) * log_t) - np.exp(
        log_cumulative_hazard
    )
    return -loglike


def _weibull_jac(
    params: NDArray[np.float64],
    *,
    x: NDArray[np.float64],
    log_t: NDArray[np.float64],
    event: NDArray[np.float64],
) -> NDArray[np.float64]:
    index = x @ params[:-1]
    shape = np.exp(params[-1])
    cumulative_hazard = np.exp(index + shape * log_t)
    d_index = event - cumulative_hazard
    d_log_shape = event * (1 + shape * log_t) - cumulative_hazard * shape * log_t
    return -np.column_stack([d_index[:, None] * x, d_log_shape])


# ======================================================================================
# Multivariate normal with Cholesky parametrized covariance
# ======================================================================================


def multivariate_normal(
    *,
    n_obs: int,
    dim: int,
    seed: int,
    dgp_df: float | None = None,
) -> SimulatedLikelihood:
    """Simulate a multivariate normal distribution with unknown mean and covariance.

    The params are the mean followed by the lower triangle of the Cholesky factor of
    the covariance matrix in row major order, with the diagonal on the log scale. There
    are dim + dim * (dim + 1) / 2 params.

    Args:
        n_obs: Number of observations.
        dim: Dimension of the outcomes.
        seed: Seed of the random number generator.
        dgp_df: If not None, the data come from a multivariate t distribution with
            this many degrees of freedom and the same covariance matrix.

    """
    rng = np.random.default_rng(seed)
    mean = np.linspace(-2, 2, dim)
    sds = np.linspace(0.5, 3, dim)
    lags = np.abs(np.subtract.outer(np.arange(dim), np.arange(dim)))
    cov = (-0.4) ** lags * np.outer(sds, sds)
    chol = np.linalg.cholesky(cov)
    z = sampling.standard_normal(rng, (n_obs, dim))
    if dgp_df is not None:
        chi2 = sampling.gamma(rng, dgp_df / 2, 2.0, (n_obs, 1))
        z = z * np.sqrt((dgp_df - 2) / chi2)
    y = mean + z @ chol.T

    rows, cols = np.tril_indices(dim)
    is_diag = rows == cols
    chol_params = chol[rows, cols]
    chol_params[is_diag] = np.log(chol_params[is_diag])

    bad_chol = np.where(is_diag, np.log(0.1), 0.0)
    return _simulated_likelihood(
        fun=partial(_multivariate_normal_fun, y=y),
        jac=partial(_multivariate_normal_jac, y=y),
        good_start=np.zeros(dim + len(rows)),
        bad_start=np.r_[np.full(dim, 10.0), bad_chol],
        true_params=np.r_[mean, chol_params] if dgp_df is None else None,
    )


def _multivariate_normal_parts(
    params: NDArray[np.float64], y: NDArray[np.float64]
) -> tuple[NDArray[np.float64], ...]:
    dim = y.shape[1]
    rows, cols = np.tril_indices(dim)
    is_diag = rows == cols
    chol_params = params[dim:]
    chol = np.zeros((dim, dim))
    chol[rows, cols] = np.where(is_diag, np.exp(chol_params), chol_params)
    standardized = linalg.solve_triangular(chol, (y - params[:dim]).T, lower=True).T
    return chol, standardized, chol_params[is_diag]


def _multivariate_normal_fun(
    params: NDArray[np.float64], *, y: NDArray[np.float64]
) -> NDArray[np.float64]:
    dim = y.shape[1]
    _, standardized, log_diag = _multivariate_normal_parts(params, y)
    return dim * _LOG_SQRT_2PI + log_diag.sum() + 0.5 * (standardized**2).sum(axis=1)


def _multivariate_normal_jac(
    params: NDArray[np.float64], *, y: NDArray[np.float64]
) -> NDArray[np.float64]:
    dim = y.shape[1]
    rows, cols = np.tril_indices(dim)
    is_diag = rows == cols
    chol, standardized, _ = _multivariate_normal_parts(params, y)
    # rows of d_mean are inv(chol).T @ standardized_i
    d_mean = linalg.solve_triangular(chol, standardized.T, lower=True, trans="T").T
    d_chol = d_mean[:, rows] * standardized[:, cols]
    d_chol[:, is_diag] = d_chol[:, is_diag] * np.diag(chol) - 1
    return -np.column_stack([d_mean, d_chol])


# ======================================================================================
# Time series: GARCH(1, 1)
# ======================================================================================


def garch(
    *,
    n_obs: int,
    seed: int,
    t_likelihood: bool = False,
    dgp_innovation_df: float | None = None,
) -> SimulatedLikelihood:
    """Simulate a GARCH(1, 1) model of returns with mean zero.

    The conditional variance is s2_t = omega + alpha * r_{t-1}**2 + beta * s2_{t-1},
    started at the sample variance. The params are (omega, alpha, beta) on their
    natural scale with bounds, followed by the degrees of freedom of the standardized t
    distribution if t_likelihood is True.

    Args:
        n_obs: Number of observations.
        seed: Seed of the random number generator.
        t_likelihood: If True, the innovations are modeled as standardized t, else as
            standard normal.
        dgp_innovation_df: If not None, the innovations are standardized t with this
            many degrees of freedom, else standard normal.

    """
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 0.05, 0.08, 0.9
    n_burn_in = 500
    n_total = n_obs + n_burn_in
    if dgp_innovation_df is None:
        innovations = sampling.standard_normal(rng, n_total)
    else:
        innovations = sampling.standardized_t(rng, dgp_innovation_df, n_total)
    returns = np.empty(n_total)
    variance = omega / (1 - alpha - beta)
    for t in range(n_total):
        returns[t] = np.sqrt(variance) * innovations[t]
        variance = omega + alpha * returns[t] ** 2 + beta * variance
    squared = returns[n_burn_in:] ** 2
    initial = squared.mean()

    good_start = np.array([0.05 * initial, 0.05, 0.9])
    bad_start = np.array([0.001 * initial, 0.6, 0.01])
    lower = np.array([1e-6, 0.0, 0.0])
    upper = np.array([np.inf, 1.0, 1.0])
    true_params = np.array([omega, alpha, beta])
    if t_likelihood:
        good_start = np.r_[good_start, 8.0]
        bad_start = np.r_[bad_start, 2.2]
        lower = np.r_[lower, 2.1]
        upper = np.r_[upper, 500.0]
        true_params = np.r_[true_params, dgp_innovation_df]

    is_correct = (dgp_innovation_df is not None) == t_likelihood
    return _simulated_likelihood(
        fun=partial(_garch_fun, squared=squared, initial=initial),
        jac=partial(_garch_jac, squared=squared, initial=initial),
        good_start=good_start,
        bad_start=bad_start,
        true_params=true_params if is_correct else None,
        lower_bounds=lower,
        upper_bounds=upper,
    )


def _garch_variance(
    params: NDArray[np.float64], squared: NDArray[np.float64], initial: float
) -> NDArray[np.float64]:
    omega, alpha, beta = params[:3]
    shocks = omega + alpha * squared[:-1]
    recursion = signal.lfilter([1.0], [1.0, -beta], shocks, zi=[beta * initial])[0]
    return np.r_[initial, recursion]


def _garch_variance_jac(
    params: NDArray[np.float64],
    squared: NDArray[np.float64],
    variance: NDArray[np.float64],
) -> NDArray[np.float64]:
    beta = params[2]
    inputs = np.column_stack([np.ones(len(squared) - 1), squared[:-1], variance[:-1]])
    recursion = signal.lfilter([1.0], [1.0, -beta], inputs, axis=0)
    return np.vstack([np.zeros((1, 3)), recursion])


def _garch_loglike_and_derivative(
    params: NDArray[np.float64],
    squared: NDArray[np.float64],
    variance: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
    """Return log-likelihood contributions and their derivatives wrt variance and df."""
    if len(params) == 3:
        loglike = -0.5 * (np.log(2 * np.pi) + np.log(variance) + squared / variance)
        d_variance = 0.5 * (squared / variance - 1) / variance
        return loglike, d_variance, None
    df = params[3]
    ratio = squared / (variance * (df - 2))
    loglike = (
        special.gammaln((df + 1) / 2)
        - special.gammaln(df / 2)
        - 0.5 * np.log(np.pi * (df - 2))
        - 0.5 * np.log(variance)
        - 0.5 * (df + 1) * np.log1p(ratio)
    )
    d_variance = -0.5 / variance + 0.5 * (df + 1) * ratio / (variance * (1 + ratio))
    d_df = (
        0.5 * special.digamma((df + 1) / 2)
        - 0.5 * special.digamma(df / 2)
        - 0.5 / (df - 2)
        - 0.5 * np.log1p(ratio)
        + 0.5 * (df + 1) * ratio / ((df - 2) * (1 + ratio))
    )
    return loglike, d_variance, d_df


def _garch_fun(
    params: NDArray[np.float64], *, squared: NDArray[np.float64], initial: float
) -> NDArray[np.float64]:
    variance = _garch_variance(params, squared, initial)
    return -_garch_loglike_and_derivative(params, squared, variance)[0]


def _garch_jac(
    params: NDArray[np.float64], *, squared: NDArray[np.float64], initial: float
) -> NDArray[np.float64]:
    variance = _garch_variance(params, squared, initial)
    _, d_variance, d_df = _garch_loglike_and_derivative(params, squared, variance)
    out = d_variance[:, None] * _garch_variance_jac(params, squared, variance)
    if d_df is not None:
        out = np.column_stack([out, d_df])
    return -out


# ======================================================================================
# Panel models with random effects: random effects probit, mixed logit
# ======================================================================================


def random_effects_probit(
    *,
    n_individuals: int,
    n_periods: int,
    n_params: int,
    seed: int,
    n_nodes: int = 12,
    dgp_bimodal_effects: bool = False,
) -> SimulatedLikelihood:
    """Simulate a random effects panel probit model.

    The likelihood integrates over the normal random effect with Gauss-Hermite
    quadrature. The params are the coefficients (with intercept) followed by the log
    standard deviation of the random effect.

    Args:
        n_individuals: Number of individuals.
        n_periods: Number of periods per individual.
        n_params: Number of params.
        seed: Seed of the random number generator.
        n_nodes: Number of quadrature nodes.
        dgp_bimodal_effects: If True, the random effects follow a bimodal mixture of
            normals with unit variance instead of a standard normal distribution.

    """
    rng = np.random.default_rng(seed)
    n_slopes = n_params - 2
    x = sampling.standard_normal(rng, (n_individuals, n_periods, n_slopes))
    x = np.concatenate([np.ones((n_individuals, n_periods, 1)), x], axis=2)
    beta = np.r_[-0.3, _slopes(n_slopes, 1.0)]
    if dgp_bimodal_effects:
        sign = np.where(sampling.uniform(rng, n_individuals) < 0.5, -1.0, 1.0)
        effects = 0.9 * sign + np.sqrt(1 - 0.81) * sampling.standard_normal(
            rng, n_individuals
        )
    else:
        effects = sampling.standard_normal(rng, n_individuals)
    errors = sampling.standard_normal(rng, (n_individuals, n_periods))
    y = (x @ beta + effects[:, None] + errors > 0).astype(float)

    nodes, weights = np.polynomial.hermite_e.hermegauss(n_nodes)
    log_weights = np.log(weights / weights.sum())
    kwargs = {"x": x, "sign": 2 * y - 1, "nodes": nodes, "log_weights": log_weights}
    return _simulated_likelihood(
        fun=partial(_random_effects_probit_fun, **kwargs),
        jac=partial(_random_effects_probit_jac, **kwargs),
        good_start=np.r_[special.ndtri(y.mean()), np.zeros(n_slopes), np.log(0.3)],
        bad_start=np.r_[1.0, -3 * beta[1:], np.log(5.0)],
        true_params=None if dgp_bimodal_effects else np.r_[beta, 0.0],
    )


def _random_effects_probit_parts(
    params: NDArray[np.float64],
    x: NDArray[np.float64],
    sign: NDArray[np.float64],
    nodes: NDArray[np.float64],
    log_weights: NDArray[np.float64],
) -> tuple[NDArray[np.float64], ...]:
    sd = np.exp(params[-1])
    index = x @ params[:-1]
    # shape (n_individuals, n_periods, n_nodes)
    arg = sign[:, :, None] * (index[:, :, None] + sd * nodes)
    log_joint = special.log_ndtr(arg).sum(axis=1) + log_weights
    loglike = _logsumexp(log_joint, axis=1)
    return arg, log_joint, loglike, sd


def _random_effects_probit_fun(
    params: NDArray[np.float64],
    *,
    x: NDArray[np.float64],
    sign: NDArray[np.float64],
    nodes: NDArray[np.float64],
    log_weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    return -_random_effects_probit_parts(params, x, sign, nodes, log_weights)[2]


def _random_effects_probit_jac(
    params: NDArray[np.float64],
    *,
    x: NDArray[np.float64],
    sign: NDArray[np.float64],
    nodes: NDArray[np.float64],
    log_weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    arg, log_joint, loglike, sd = _random_effects_probit_parts(
        params, x, sign, nodes, log_weights
    )
    posterior = np.exp(log_joint - loglike[:, None])
    d_arg = sign[:, :, None] * _inverse_mills(arg)
    d_beta = np.einsum("iq,itq,itk->ik", posterior, d_arg, x)
    d_log_sd = np.einsum("iq,itq,q->i", posterior, d_arg, sd * nodes)
    return -np.column_stack([d_beta, d_log_sd])


def mixed_logit(
    *,
    n_individuals: int,
    n_situations: int,
    n_alternatives: int,
    n_fixed: int,
    n_random: int,
    seed: int,
    correlated: bool = False,
    n_draws: int = 50,
    zero_sd: bool = False,
    dgp_lognormal: bool = False,
) -> SimulatedLikelihood:
    """Simulate a panel mixed logit model, estimated by maximum simulated likelihood.

    Each individual makes n_situations choices among n_alternatives alternatives with
    random attributes. The first n_fixed attributes have fixed coefficients, the others
    normally distributed individual specific coefficients. The params are the fixed
    coefficients, the means of the random coefficients and the Cholesky factor of their
    covariance: its diagonal if correlated is False, else its lower triangle in row
    major order. The diagonal of the Cholesky factor is bounded below by zero.

    With few draws relative to the number of individuals, the simulated likelihood is
    noticeably biased, so the estimates can differ from the true params by more than
    their standard errors suggest, and the information matrix equality fails even
    though the model is correctly specified.

    Args:
        n_individuals: Number of individuals.
        n_situations: Number of choice situations per individual.
        n_alternatives: Number of alternatives.
        n_fixed: Number of attributes with fixed coefficients.
        n_random: Number of attributes with random coefficients.
        seed: Seed of the random number generator.
        correlated: Whether the random coefficients are correlated.
        n_draws: Number of simulation draws per individual.
        zero_sd: If True, the last random coefficient has a standard deviation of
            zero, so its bound can be active at the solution.
        dgp_lognormal: If True, the random coefficients follow a lognormal distribution
            with the same mean and covariance as the normal distribution.

    """
    rng = np.random.default_rng(seed)
    n_attributes = n_fixed + n_random
    shape = (n_individuals, n_situations, n_alternatives, n_attributes)
    w = sampling.standard_normal(rng, shape)
    fixed = _slopes(n_fixed, 1.0)
    means = np.linspace(1.0, 2.0, n_random)
    sds = np.linspace(0.8, 0.4, n_random)
    if zero_sd:
        sds[-1] = 0.0
    lags = np.abs(np.subtract.outer(np.arange(n_random), np.arange(n_random)))
    corr = 0.3**lags if correlated else np.eye(n_random)
    cov = corr * np.outer(sds, sds)
    chol = np.zeros((n_random, n_random))
    n_nonzero = n_random - 1 if zero_sd else n_random
    chol[:n_nonzero, :n_nonzero] = np.linalg.cholesky(cov[:n_nonzero, :n_nonzero])

    standard = sampling.standard_normal(rng, (n_individuals, n_random))
    if dgp_lognormal:
        # lognormal with the given means and sds, using the correlation of the normals
        log_sd = np.sqrt(np.log1p((sds / means) ** 2))
        log_mean = np.log(means) - 0.5 * log_sd**2
        correlated_standard = standard @ np.linalg.cholesky(corr).T
        coefficients = np.exp(log_mean + log_sd * correlated_standard)
    else:
        coefficients = means + standard @ chol.T
    utility = w[..., :n_fixed] @ fixed + np.einsum(
        "ntjk,nk->ntj", w[..., n_fixed:], coefficients
    )
    y = np.argmax(utility + sampling.standard_gumbel(rng, utility.shape), axis=2)
    draws = sampling.standard_normal(rng, (n_individuals, n_draws, n_random))

    if correlated:
        rows, cols = np.tril_indices(n_random)
    else:
        rows = cols = np.arange(n_random)
    is_diag = rows == cols
    chol_params = chol[rows, cols]
    lower = np.r_[np.full(n_attributes, -np.inf), np.where(is_diag, 0.0, -np.inf)]

    kwargs = {
        "w": w,
        "y": y,
        "draws": draws,
        "n_fixed": n_fixed,
        "correlated": correlated,
    }
    return _simulated_likelihood(
        fun=partial(_mixed_logit_fun, **kwargs),
        jac=partial(_mixed_logit_jac, **kwargs),
        good_start=np.r_[np.zeros(n_attributes), np.where(is_diag, 0.5, 0.0)],
        bad_start=np.r_[-3 * fixed, -3 * means, np.where(is_diag, 5.0, 0.0)],
        true_params=None if dgp_lognormal else np.r_[fixed, means, chol_params],
        lower_bounds=lower,
    )


def _mixed_logit_parts(
    params: NDArray[np.float64],
    w: NDArray[np.float64],
    y: NDArray[np.int64],
    draws: NDArray[np.float64],
    n_fixed: int,
    correlated: bool,
) -> tuple[NDArray[np.float64], ...]:
    n_individuals, n_situations, n_alternatives, n_attributes = w.shape
    n_random = n_attributes - n_fixed
    chol = np.zeros((n_random, n_random))
    if correlated:
        chol[np.tril_indices(n_random)] = params[n_attributes:]
    else:
        chol[np.diag_indices(n_random)] = params[n_attributes:]
    # coefficients per individual and draw, shape (n_individuals, n_draws, n_attr)
    random = params[n_fixed:n_attributes] + draws @ chol.T
    fixed = np.broadcast_to(params[:n_fixed], (*random.shape[:2], n_fixed))
    coefficients = np.concatenate([fixed, random], axis=2)
    # utility, shape (n_individuals, n_situations, n_alternatives, n_draws)
    flat_w = w.reshape(n_individuals, n_situations * n_alternatives, n_attributes)
    utility = (flat_w @ coefficients.transpose(0, 2, 1)).reshape(
        n_individuals, n_situations, n_alternatives, -1
    )
    log_denominator = _logsumexp(utility, axis=2)
    chosen = np.take_along_axis(utility, y[:, :, None, None], axis=2)[:, :, 0]
    log_kernel = (chosen - log_denominator).sum(axis=1)
    loglike = _logsumexp(log_kernel, axis=1) - np.log(draws.shape[1])
    return utility, log_denominator, log_kernel, loglike


def _mixed_logit_fun(
    params: NDArray[np.float64],
    *,
    w: NDArray[np.float64],
    y: NDArray[np.int64],
    draws: NDArray[np.float64],
    n_fixed: int,
    correlated: bool,
) -> NDArray[np.float64]:
    return -_mixed_logit_parts(params, w, y, draws, n_fixed, correlated)[3]


def _mixed_logit_jac(
    params: NDArray[np.float64],
    *,
    w: NDArray[np.float64],
    y: NDArray[np.int64],
    draws: NDArray[np.float64],
    n_fixed: int,
    correlated: bool,
) -> NDArray[np.float64]:
    n_individuals, n_situations, n_alternatives, n_attributes = w.shape
    n_draws, n_random = draws.shape[1:]
    utility, log_denominator, log_kernel, loglike = _mixed_logit_parts(
        params, w, y, draws, n_fixed, correlated
    )
    probs = np.exp(utility - log_denominator[:, :, None])
    posterior = np.exp(log_kernel - loglike[:, None] - np.log(n_draws))
    chosen_attributes = np.take_along_axis(w, y[:, :, None, None], axis=2)[:, :, 0]
    # derivative of the log kernel wrt the coefficients, shape (n_ind, n_draws, n_attr)
    flat_probs = probs.reshape(n_individuals, n_situations * n_alternatives, n_draws)
    flat_w = w.reshape(n_individuals, n_situations * n_alternatives, n_attributes)
    expected_attributes = flat_probs.transpose(0, 2, 1) @ flat_w
    d_coefficients = chosen_attributes.sum(axis=1)[:, None, :] - expected_attributes
    weighted = posterior[:, :, None] * d_coefficients
    d_means = weighted.sum(axis=1)
    if correlated:
        rows, cols = np.tril_indices(n_random)
    else:
        rows = cols = np.arange(n_random)
    d_chol = (weighted[:, :, n_fixed:].transpose(0, 2, 1) @ draws)[:, rows, cols]
    return -np.column_stack([d_means, d_chol])


# ======================================================================================
# Mixture and selection models
# ======================================================================================


def normal_mixture(
    *,
    n_obs: int,
    seed: int,
    dgp_three_components: bool = False,
) -> SimulatedLikelihood:
    """Simulate a two component univariate normal mixture.

    The params are (mean_1, mean_2, log_sd_1, log_sd_2, logit_weight_1). The solution is
    only unique up to swapping the component labels.

    Args:
        n_obs: Number of observations.
        seed: Seed of the random number generator.
        dgp_three_components: If True, the data come from a mixture of three normals.

    """
    rng = np.random.default_rng(seed)
    if dgp_three_components:
        weights = np.array([0.3, 0.4, 0.3])
        means = np.array([-2.0, 0.0, 2.5])
        sds = np.array([0.5, 1.0, 0.7])
    else:
        weights = np.array([0.35, 0.65])
        means = np.array([-1.0, 1.5])
        sds = np.array([0.7, 1.0])
    component = sampling.categorical(rng, np.tile(weights, (n_obs, 1)))
    y = means[component] + sds[component] * sampling.standard_normal(rng, n_obs)

    true_params = np.r_[means, np.log(sds), special.logit(weights[0])]
    center, spread = y.mean(), y.std()
    return _simulated_likelihood(
        fun=partial(_normal_mixture_fun, y=y),
        jac=partial(_normal_mixture_jac, y=y),
        good_start=np.r_[
            center - spread, center + spread, np.log(spread), np.log(spread), 0.0
        ],
        bad_start=np.r_[
            center + 3 * spread,
            center + 4 * spread,
            np.log(spread) + 2,
            np.log(spread) + 1,
            0.0,
        ],
        true_params=None if dgp_three_components else true_params,
        unique_solution=False,
    )


def _normal_mixture_parts(
    params: NDArray[np.float64], y: NDArray[np.float64]
) -> tuple[NDArray[np.float64], ...]:
    means, log_sds, logit_weight = params[:2], params[2:4], params[4]
    log_weights = np.array(
        [special.log_expit(logit_weight), special.log_expit(-logit_weight)]
    )
    standardized = (y[:, None] - means) * np.exp(-log_sds)
    log_joint = log_weights + _log_normal_pdf(standardized) - log_sds
    loglike = special.logsumexp(log_joint, axis=1)
    return standardized, log_joint, loglike


def _normal_mixture_fun(
    params: NDArray[np.float64], *, y: NDArray[np.float64]
) -> NDArray[np.float64]:
    return -_normal_mixture_parts(params, y)[2]


def _normal_mixture_jac(
    params: NDArray[np.float64], *, y: NDArray[np.float64]
) -> NDArray[np.float64]:
    standardized, log_joint, loglike = _normal_mixture_parts(params, y)
    posterior = np.exp(log_joint - loglike[:, None])
    d_means = posterior * standardized * np.exp(-params[2:4])
    d_log_sds = posterior * (standardized**2 - 1)
    d_logit_weight = posterior[:, 0] - special.expit(params[4])
    return -np.column_stack([d_means, d_log_sds, d_logit_weight])


def heckman_selection(
    *,
    n_obs: int,
    n_selection_params: int,
    n_outcome_params: int,
    seed: int,
    dgp_error_df: float | None = None,
) -> SimulatedLikelihood:
    """Simulate a Heckman sample selection model, estimated by full maximum likelihood.

    The outcome y = x @ beta + e is observed if z @ gamma + u > 0, where (u, e) are
    bivariate normal with unit variance of u, standard deviation sigma of e and
    correlation rho. z contains the covariates of x plus excluded covariates. The params
    are gamma, beta, log(sigma) and arctanh(rho).

    Args:
        n_obs: Number of observations.
        n_selection_params: Number of params of the selection equation.
        n_outcome_params: Number of params of the outcome equation (at most
            n_selection_params).
        seed: Seed of the random number generator.
        dgp_error_df: If not None, (u, e) follow a bivariate t distribution with this
            many degrees of freedom and the same covariance matrix.

    """
    rng = np.random.default_rng(seed)
    z = _add_intercept(_covariates(rng, n_obs, n_selection_params - 1))
    x = z[:, :n_outcome_params]
    gamma = np.r_[0.3, _slopes(n_selection_params - 1, 1.0)[::-1]]
    beta = np.r_[1.0, _slopes(n_outcome_params - 1, 1.0)]
    sd, rho = 1.2, 0.5
    standard = sampling.standard_normal(rng, (n_obs, 2))
    if dgp_error_df is not None:
        chi2 = sampling.gamma(rng, dgp_error_df / 2, 2.0, (n_obs, 1))
        standard = standard * np.sqrt((dgp_error_df - 2) / chi2)
    u = standard[:, 0]
    e = sd * (rho * standard[:, 0] + np.sqrt(1 - rho**2) * standard[:, 1])
    selected = (z @ gamma + u > 0).astype(float)
    y = np.where(selected == 1, x @ beta + e, 0.0)

    observed = y[selected == 1]
    log_sd_y = np.log(observed.std())
    return _simulated_likelihood(
        fun=partial(_heckman_fun, z=z, x=x, y=y, selected=selected),
        jac=partial(_heckman_jac, z=z, x=x, y=y, selected=selected),
        good_start=np.r_[
            special.ndtri(selected.mean()),
            np.zeros(n_selection_params - 1),
            observed.mean(),
            np.zeros(n_outcome_params - 1),
            log_sd_y,
            0.0,
        ],
        bad_start=np.r_[-3 * gamma, -3 * beta, log_sd_y - 3, np.arctanh(-0.9)],
        true_params=(
            np.r_[gamma, beta, np.log(sd), np.arctanh(rho)]
            if dgp_error_df is None
            else None
        ),
    )


def _heckman_parts(
    params: NDArray[np.float64],
    z: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    selected: NDArray[np.float64],
) -> tuple[NDArray[np.float64], ...]:
    """Return the selection index, standardized residual and the normal cdf argument.

    The argument of the normal cdf is -selection_index for unselected observations.

    """
    n_selection, n_outcome = z.shape[1], x.shape[1]
    selection_index = z @ params[:n_selection]
    outcome_index = x @ params[n_selection : n_selection + n_outcome]
    rho = np.tanh(params[-1])
    standardized = (y - outcome_index) * np.exp(-params[-2])
    arg = np.where(
        selected == 1,
        (selection_index + rho * standardized) / np.sqrt(1 - rho**2),
        -selection_index,
    )
    return selection_index, standardized, arg


def _heckman_fun(
    params: NDArray[np.float64],
    *,
    z: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    selected: NDArray[np.float64],
) -> NDArray[np.float64]:
    _, standardized, arg = _heckman_parts(params, z, x, y, selected)
    loglike = special.log_ndtr(arg) + selected * (
        _log_normal_pdf(standardized) - params[-2]
    )
    return -loglike


def _heckman_jac(
    params: NDArray[np.float64],
    *,
    z: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    selected: NDArray[np.float64],
) -> NDArray[np.float64]:
    selection_index, standardized, arg = _heckman_parts(params, z, x, y, selected)
    sd = np.exp(params[-2])
    rho = np.tanh(params[-1])
    root = np.sqrt(1 - rho**2)
    mills = _inverse_mills(arg)
    is_selected = selected == 1
    d_selection = np.where(is_selected, mills / root, -mills)
    d_outcome = np.where(
        is_selected, -mills * rho / (sd * root) + standardized / sd, 0.0
    )
    d_log_sd = np.where(
        is_selected, -mills * rho * standardized / root + standardized**2 - 1, 0.0
    )
    d_atanh_rho = np.where(
        is_selected, mills * (standardized + rho * selection_index) / root, 0.0
    )
    return -np.column_stack(
        [
            d_selection[:, None] * z,
            d_outcome[:, None] * x,
            d_log_sd,
            d_atanh_rho,
        ]
    )
