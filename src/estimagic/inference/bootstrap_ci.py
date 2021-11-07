import numpy as np
import pandas as pd
from estimagic.inference.bootstrap_helpers import check_inputs
from joblib import delayed
from joblib import Parallel
from scipy.stats import norm


def compute_ci(data, outcome, estimates, ci_method="percentile", alpha=0.05, n_cores=1):
    """Compute confidence interval of bootstrap estimates. Parts of the code of the
    subfunctions of this function are taken from Daniel Saxton's resample library, as
    found on https://github.com/dsaxton/resample/ .


    Args:
        data (pandas.DataFrame): original dataset.
        outcome (callable): function of the data calculating statistic of interest.
            Needs to return a pandas Series.
        estimates (pandas.DataFrame): DataFrame of estimates in the bootstrap samples.
        ci_method (str): method of choice for confidence interval computation.
        alpha (float): significance level of choice.
        n_cores (int): number of jobs for parallelization.

    Returns:
        cis (pandas.DataFrame): DataFrame where k'th row contains CI for k'th parameter.

    """

    check_inputs(data=data, alpha=alpha, ci_method=ci_method)

    funcname = "_ci_" + ci_method

    cis = globals()[funcname](data, outcome, estimates, alpha, n_cores)

    return pd.DataFrame(
        cis, index=estimates.columns.tolist(), columns=["lower_ci", "upper_ci"]
    )


def _ci_percentile(data, outcome, estimates, alpha, n_cores):
    """Compute percentile type confidence interval of bootstrap estimates.

    Args:
        data (pd.DataFrame): original dataset.
        outcome (callable): function of the data calculating statistic of interest.
        estimates (data.Frame): DataFrame of estimates in the bootstrap samples.
        alpha (float): significance level of choice.
        n_cores (int): number of jobs for parallelization.

    Returns:
        cis (np.array): array where k'th row contains CI for k'th parameter.

    """

    num_params = estimates.shape[1]
    boot_est = estimates.values
    cis = np.zeros((num_params, 2))

    for k in range(num_params):

        q = _eqf(boot_est[:, k])
        cis[k, :] = np.array([q(alpha / 2), q(1 - alpha / 2)])

    return cis


def _ci_bca(data, outcome, estimates, alpha, n_cores):
    """Compute bca type confidence interval of bootstrap estimates.

    Args:
        data (pd.DataFrame): original dataset.
        outcome (callable): function of the data calculating statistic of interest.
        estimates (data.Frame): DataFrame of estimates in the bootstrap samples.
        alpha (float): significance level of choice.
        n_cores (int): number of jobs for parallelization.

    Returns:
        cis (np.array): array where k'th row contains CI for k'th parameter.

    """

    num_params = estimates.shape[1]
    boot_est = estimates.values
    cis = np.zeros((num_params, 2))

    theta = outcome(data)

    jack_est = _jackknife(data, outcome, n_cores)
    jack_mean = np.mean(jack_est, axis=0)

    for k in range(num_params):

        q = _eqf(boot_est[:, k])
        params = boot_est[:, k]

        # bias correction
        z_naught = norm.ppf(np.mean(params <= theta[k]))
        z_low = norm.ppf(alpha)
        z_high = norm.ppf(1 - alpha)

        # accelaration
        acc = np.sum((jack_mean[k] - jack_est[k]) ** 3) / (
            6 * np.sum((jack_mean[k] - jack_est[k]) ** 2) ** (3 / 2)
        )

        p1 = norm.cdf(z_naught + (z_naught + z_low) / (1 - acc * (z_naught + z_low)))
        p2 = norm.cdf(z_naught + (z_naught + z_high) / (1 - acc * (z_naught + z_high)))

        cis[k, :] = np.array([q(p1), q(p2)])

    return cis


def _ci_bc(data, outcome, estimates, alpha, n_cores):
    """Compute bc type confidence interval of bootstrap estimates.

    Args:
        data (pd.DataFrame): original dataset.
        outcome (callable): function of the data calculating statistic of interest.
        estimates (data.Frame): DataFrame of estimates in the bootstrap samples.
        alpha (float): significance level of choice.
        n_cores (int): number of jobs for parallelization.

    Returns:
        cis (np.array): array where k'th row contains CI for k'th parameter.

    """

    num_params = estimates.shape[1]
    boot_est = estimates.values
    cis = np.zeros((num_params, 2))

    theta = outcome(data)

    for k in range(num_params):

        q = _eqf(boot_est[:, k])
        params = boot_est[:, k]

        # bias correction
        z_naught = norm.ppf(np.mean(params <= theta[k]))
        z_low = norm.ppf(alpha)
        z_high = norm.ppf(1 - alpha)

        p1 = norm.cdf(z_naught + (z_naught + z_low))
        p2 = norm.cdf(z_naught + (z_naught + z_high))

        cis[k, :] = np.array([q(p1), q(p2)])

    return cis


def _ci_t(data, outcome, estimates, alpha, n_cores):
    """Compute studentized confidence interval of bootstrap estimates.

    Args:
        data (pd.DataFrame): original dataset.
        outcome (callable): function of the data calculating statistic of interest.
        estimates (data.Frame): DataFrame of estimates in the bootstrap samples.
        alpha (float): significance level of choice.
        n_cores (int): number of jobs for parallelization.

    Returns:
        cis (np.array): array where k'th row contains CI for k'th parameter.

    """

    num_params = estimates.shape[1]
    boot_est = estimates.values
    cis = np.zeros((num_params, 2))
    theta = outcome(data)

    for k in range(num_params):

        params = boot_est[:, k]

        theta_std = np.std(params)

        tq = _eqf((params - theta[k]) / theta_std)
        t1 = tq(1 - alpha / 2)
        t2 = tq(alpha / 2)

        cis[k, :] = np.array([theta[k] - theta_std * t1, theta[k] - theta_std * t2])

    return cis


def _ci_normal(data, outcome, estimates, alpha, n_cores):
    """Compute approximate normal confidence interval of bootstrap estimates.

    Args:
        data (pd.DataFrame): original dataset.
        outcome (callable): function of the data calculating statistic of interest.
        estimates (data.Frame): DataFrame of estimates in the bootstrap samples.
        alpha (float): significance level of choice.
        n_cores (int): number of jobs for parallelization.

    Returns:
        cis (np.array): array where k'th row contains CI for k'th parameter.

    """

    num_params = estimates.shape[1]
    boot_est = estimates.values
    cis = np.zeros((num_params, 2))
    theta = outcome(data)

    for k in range(num_params):

        params = boot_est[:, k]
        theta_std = np.std(params)
        t = norm.ppf(alpha / 2)

        cis[k, :] = np.array([theta[k] + theta_std * t, theta[k] - theta_std * t])

    return cis


def _ci_basic(data, outcome, estimates, alpha, n_cores):
    """Compute basic bootstrap confidence interval of bootstrap estimates.

    Args:
        data (pd.DataFrame): original dataset.
        outcome (callable): function of the data calculating statistic of interest.
        estimates (data.Frame): DataFrame of estimates in the bootstrap samples.
        alpha (float): significance level of choice.
        n_cores (int): number of jobs for parallelization.

    Returns:
        cis (np.array): array where k'th row contains CI for k'th parameter.

    """

    num_params = estimates.shape[1]
    boot_est = estimates.values
    cis = np.zeros((num_params, 2))
    theta = outcome(data)

    for k in range(num_params):

        q = _eqf(boot_est[:, k])

        cis[k, :] = np.array(
            [2 * theta[k] - q(1 - alpha / 2), 2 * theta[k] - q(alpha / 2)]
        )

    return cis


def _jackknife(data, outcome, n_cores=1):
    """Calculate leave-one-out estimator.

    Args:
        data (pd.DataFrame): original dataset.
        n_cores (int): number of jobs for parallelization.

    Returns:
        jk_estimates (pd.DataFrame): DataFrame of estimated parameters.

    """

    n = len(data)

    def loop(i):

        df = data.drop(index=i)
        return outcome(df)

    jk_estimates = Parallel(n_jobs=n_cores)(delayed(loop)(i) for i in range(n))

    return np.array(jk_estimates)


def _eqf(sample):
    """Return empirical quantile function of the given sample.

    Args:
        sample (pd.DataFrame): sample to base quantile function on.

    Returns:
        f (callable): quantile function for given sample.

    """

    def f(x):
        return np.quantile(sample, x)

    return f
