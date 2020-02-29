import numpy as np
import pandas as pd
from joblib import delayed
from joblib import Parallel
from scipy.stats import norm


def compute_ci(data, f, estimates, ci_method="percentile", alpha=0.05, num_threads=1):
    """Compute confidence interval of bootstrap estimates.

    Args:
        data (pd.DataFrame): original dataset.
        f (callable): function of the data calculating statistic of interest.
        estimates (data.Frame): DataFrame of estimates in the bootstrap samples.
        ci_method (str): method of choice for confidence interval computation.
        alpha (float): significance level of choice.
        num_threads (int): number of jobs for parallelization.

    Returns:
        cis (pd.DataFrame): DataFrame where k'th row contains CI for k'th parameter.

    """

    _check_inputs(data=data, alpha=alpha, ci_method=ci_method)

    funcname = "_ci_" + ci_method

    cis = globals()[funcname](data, f, estimates, alpha, num_threads)

    return pd.DataFrame(
        cis, index=estimates.columns.tolist(), columns=["lower_ci", "upper_ci"]
    )


def _ci_percentile(data, f, estimates, alpha, num_threads):
    """Compute percentile type confidence interval of bootstrap estimates.

    Args:
        data (pd.DataFrame): original dataset.
        f (callable): function of the data calculating statistic of interest.
        estimates (data.Frame): DataFrame of estimates in the bootstrap samples.
        alpha (float): significance level of choice.
        num_threads (int): number of jobs for parallelization.

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


def _ci_bca(data, f, estimates, alpha, num_threads):
    """Compute bca type confidence interval of bootstrap estimates.

    Args:
        data (pd.DataFrame): original dataset.
        f (callable): function of the data calculating statistic of interest.
        estimates (data.Frame): DataFrame of estimates in the bootstrap samples.
        alpha (float): significance level of choice.
        num_threads (int): number of jobs for parallelization.

    Returns:
        cis (np.array): array where k'th row contains CI for k'th parameter.

    """

    num_params = estimates.shape[1]
    boot_est = estimates.values
    cis = np.zeros((num_params, 2))

    theta = f(data)

    jack_est = _jackknife(data, f, num_threads)
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


def _ci_bc(data, f, estimates, alpha, num_threads):
    """Compute bc type confidence interval of bootstrap estimates.

    Args:
        data (pd.DataFrame): original dataset.
        f (callable): function of the data calculating statistic of interest.
        estimates (data.Frame): DataFrame of estimates in the bootstrap samples.
        alpha (float): significance level of choice.
        num_threads (int): number of jobs for parallelization.

    Returns:
        cis (np.array): array where k'th row contains CI for k'th parameter.

    """

    num_params = estimates.shape[1]
    boot_est = estimates.values
    cis = np.zeros((num_params, 2))

    theta = f(data)

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


def _ci_t(data, f, estimates, alpha, num_threads):
    """Compute studentized confidence interval of bootstrap estimates.

    Args:
        data (pd.DataFrame): original dataset.
        f (callable): function of the data calculating statistic of interest.
        estimates (data.Frame): DataFrame of estimates in the bootstrap samples.
        alpha (float): significance level of choice.
        num_threads (int): number of jobs for parallelization.

    Returns:
        cis (np.array): array where k'th row contains CI for k'th parameter.

    """

    num_params = estimates.shape[1]
    boot_est = estimates.values
    cis = np.zeros((num_params, 2))
    theta = f(data)

    for k in range(num_params):

        params = boot_est[:, k]

        theta_std = np.std(params)

        tq = _eqf((params - theta[k]) / theta_std)
        t1 = tq(1 - alpha / 2)
        t2 = tq(alpha / 2)

        cis[k, :] = np.array([theta[k] - theta_std * t1, theta[k] - theta_std * t2])

    return cis


def _ci_normal(data, f, estimates, alpha, num_threads):
    """Compute approximate normal confidence interval of bootstrap estimates.

    Args:
        data (pd.DataFrame): original dataset.
        f (callable): function of the data calculating statistic of interest.
        estimates (data.Frame): DataFrame of estimates in the bootstrap samples.
        alpha (float): significance level of choice.
        num_threads (int): number of jobs for parallelization.

    Returns:
        cis (np.array): array where k'th row contains CI for k'th parameter.

    """

    num_params = estimates.shape[1]
    boot_est = estimates.values
    cis = np.zeros((num_params, 2))
    theta = f(data)

    for k in range(num_params):

        params = boot_est[:, k]
        theta_std = np.std(params)
        t = norm.ppf(alpha / 2)

        cis[k, :] = np.array([theta[k] + theta_std * t, theta[k] - theta_std * t])

    return cis


def _ci_basic(data, f, estimates, alpha, num_threads):
    """Compute basic bootstrap confidence interval of bootstrap estimates.

     Args:
         data (pd.DataFrame): original dataset.
         f (callable): function of the data calculating statistic of interest.
         estimates (data.Frame): DataFrame of estimates in the bootstrap samples.
         alpha (float): significance level of choice.
         num_threads (int): number of jobs for parallelization.

     Returns:
         cis (np.array): array where k'th row contains CI for k'th parameter.

     """

    num_params = estimates.shape[1]
    boot_est = estimates.values
    cis = np.zeros((num_params, 2))
    theta = f(data)

    for k in range(num_params):

        q = _eqf(boot_est[:, k])

        cis[k, :] = np.array(
            [2 * theta[k] - q(1 - alpha / 2), 2 * theta[k] - q(alpha / 2)]
        )

    return cis


def _jackknife(data, f, num_threads=1):
    """Calculate leave-one-out estimator.

    Args:
        data (pd.DataFrame): original dataset.
        num_threads (int): number of jobs for parallelization.

    Returns:
        jk_estimates (pd.DataFrame): DataFrame of estimated parameters.

    """

    n = len(data)

    def loop(i):

        df = data.drop(index=i)
        return f(df)

    jk_estimates = Parallel(n_jobs=num_threads)(delayed(loop)(i) for i in range(n))

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


def _check_inputs(data, cluster_by=None, ci_method="percentile", alpha=0.05):
    """ Check validity of inputs.
    Args:
        data (pd.DataFrame): original dataset.
        cluster_by (str): column name of variable to cluster by.
        ci_method (str): method of choice for confidence interval computation.
        alpha (float): significance level of choice.

    """

    ci_method_list = ["percentile", "bca", "bc", "t", "normal", "basic"]

    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input 'data' must be DataFrame.")

    elif (cluster_by is not None) and (cluster_by not in data.columns.tolist()):
        raise ValueError(
            "Input 'cluster_by' must be None or a column name of DataFrame."
        )

    elif ci_method not in ci_method_list:
        raise ValueError(
            "ci_method must be 'percentile', 'bc',"
            " 'bca', 't', 'basic' or 'normal', '{method}'"
            " was supplied".format(method=ci_method)
        )

    elif alpha > 1 or alpha < 0:
        raise ValueError("Input 'alpha' must be in [0,1].")
