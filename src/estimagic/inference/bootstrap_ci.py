import numpy as np
from estimagic.inference.bootstrap_helpers import check_inputs
from scipy.stats import norm


def compute_ci(
    base_outcome_flat,
    estimates,
    ci_method="percentile",
    alpha=0.05,
):
    """Compute confidence interval of bootstrap estimates.

    Parts of the code of the
    subfunctions of this function are taken from Daniel Saxton's resample library, as
    found on https://github.com/dsaxton/resample/ .


    Args:
        base_outcome_flat (list): List of flat base outcomes, i.e. the outcome
            statistics evaluated on the original data set.
        estimates (np.ndarray): Array of estimates in the bootstrap samples.
        ci_method (str): Method of choice for confidence interval computation.
        alpha (float): Significance level of choice.

    Returns:
        np.ndarray: 1d array of the lower confidence interval, where the k'th entry
            contains the lower confidence interval for the k'th parameter.
        np.ndarray: 1d array of the upper confidence interval, where the k'th entry
            contains the upper confidence interval for the k'th parameter.
    """
    check_inputs(alpha=alpha, ci_method=ci_method)

    funcname = "_ci_" + ci_method
    func = globals()[funcname]

    if ci_method == "percentile":
        cis = func(estimates, alpha)
    else:
        cis = func(estimates, base_outcome_flat, alpha)

    return cis[:, 0], cis[:, 1]


def _ci_percentile(estimates, alpha):
    """Compute percentile type confidence interval of bootstrap estimates.

    Args:
        estimates (np.ndarray): Array of estimates in the bootstrap samples.
        alpha (float): significance level of choice.

    Returns:
        cis (np.ndarray): Array where k'th row contains CI for k'th parameter.
    """
    num_params = estimates.shape[1]
    cis = np.zeros((num_params, 2))

    for k in range(num_params):

        q = _eqf(estimates[:, k])
        cis[k, :] = np.array([q(alpha / 2), q(1 - alpha / 2)])

    return cis


def _ci_bc(estimates, theta, alpha):
    """Compute bc type confidence interval of bootstrap estimates.

    Args:
        estimates (np.ndarray): Array of estimates in the bootstrap samples.
        theta (pytree): Pytree of base outcomes.
        alpha (float): Significance level of choice.

    Returns:
        cis (np.ndarray): Array where k'th row contains CI for k'th parameter.
    """
    num_params = estimates.shape[1]
    cis = np.zeros((num_params, 2))

    for k in range(num_params):

        q = _eqf(estimates[:, k])
        params = estimates[:, k]

        # Bias correction
        z_naught = norm.ppf(np.mean(params <= theta[k]))
        z_low = norm.ppf(alpha)
        z_high = norm.ppf(1 - alpha)

        p1 = norm.cdf(z_naught + (z_naught + z_low))
        p2 = norm.cdf(z_naught + (z_naught + z_high))

        cis[k, :] = np.array([q(p1), q(p2)])

    return cis


def _ci_t(estimates, theta, alpha):
    """Compute studentized confidence interval of bootstrap estimates.

    Args:
        estimates (np.ndarray): Array of estimates in the bootstrap samples.
        theta (pytree): Pytree of base outcomes.
        alpha (float): Significance level of choice.

    Returns:
        cis (np.ndarray): Array where k'th row contains CI for k'th parameter.
    """
    num_params = estimates.shape[1]
    cis = np.zeros((num_params, 2))

    for k in range(num_params):

        params = estimates[:, k]

        theta_std = np.std(params)

        tq = _eqf((params - theta[k]) / theta_std)
        t1 = tq(1 - alpha / 2)
        t2 = tq(alpha / 2)

        cis[k, :] = np.array([theta[k] - theta_std * t1, theta[k] - theta_std * t2])

    return cis


def _ci_normal(estimates, theta, alpha):
    """Compute approximate normal confidence interval of bootstrap estimates.

    Args:
        estimates (np.ndarray): Array of estimates in the bootstrap samples.
        theta (pytree): Pytree of base outcomes.
        alpha (float): Significance level of choice.

    Returns:
        cis (np.ndarray): Array where k'th row contains CI for k'th parameter.
    """
    num_params = estimates.shape[1]
    cis = np.zeros((num_params, 2))

    for k in range(num_params):

        params = estimates[:, k]
        theta_std = np.std(params)
        t = norm.ppf(alpha / 2)

        cis[k, :] = np.array([theta[k] + theta_std * t, theta[k] - theta_std * t])

    return cis


def _ci_basic(estimates, theta, alpha):
    """Compute basic bootstrap confidence interval of bootstrap estimates.

    Args:
        estimates (np.ndarray): Array of estimates in the bootstrap samples.
        theta (pytree): Pytree of base outcomes.
        alpha (float): Significance level of choice.

    Returns:
        cis (np.ndarray): Array where k'th row contains CI for k'th parameter.
    """
    num_params = estimates.shape[1]
    cis = np.zeros((num_params, 2))

    for k in range(num_params):

        q = _eqf(estimates[:, k])

        cis[k, :] = np.array(
            [2 * theta[k] - q(1 - alpha / 2), 2 * theta[k] - q(alpha / 2)]
        )

    return cis


def _eqf(sample):
    """Return empirical quantile function of the given sample.

    Args:
        sample (np.ndarray): Sample to base quantile function on.

    Returns:
        f (callable): Quantile function for given sample.
    """

    def f(x):
        return np.quantile(sample, x)

    return f
