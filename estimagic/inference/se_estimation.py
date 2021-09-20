"""Variance estimators for maximum likelihood."""
import numpy as np


def observed_information_matrix(hess):
    """Observed information matrix or BHHH estimator.

    Args:
        hess (np.array): "hessian" - a k + 1 x k + 1-dimensional array of
            second derivatives of the pseudo-log-likelihood function w.r.t.
            the parameters
    Returns:
        oim_se (np.array): a 1d array of k + 1 standard errors
        oim_var (np.array): 2d variance-covariance matrix

    """
    hess = -hess.copy()
    oim_var = np.linalg.inv(hess)
    oim_se = np.sqrt(np.diag(oim_var))
    return oim_se, oim_var


def outer_product_of_gradients(jac):
    """Outer product of gradients estimator.

    Args:
        jac (np.array): "jacobian" - an n x k + 1-dimensional array of first
            derivatives of the pseudo-log-likelihood function w.r.t. the parameters

    Returns:
        opg_se (np.array): a 1d array of k + 1 standard errors
        opg_var (np.array): 2d variance-covariance matrix

    """
    opg_var = np.linalg.inv(np.dot(jac.T, jac))
    opg_se = np.sqrt(np.diag(opg_var))
    return opg_se, opg_var


def sandwich_step(hess, meat):
    """The sandwich estimator for variance estimation.

    Args:
        hess (np.array): "hessian" - a k + 1 x k + 1-dimensional array of
            second derivatives of the pseudo-log-likelihood function w.r.t.
            the parameters
        meat (np.array): the variance of the total scores

    Returns:
        se (np.array): a 1d array of k + 1 standard errors
        var (np.array): 2d variance-covariance matrix

    """
    invhessian = np.linalg.inv(hess)
    var = np.dot(np.dot(invhessian, meat), invhessian)
    se = np.sqrt(np.diag(var))
    return se, var


def clustering(design_options, jac):
    """Variance estimation for each cluster.

    The function takes the sum of the jacobian observations for each cluster.
    The result is the meat of the sandwich estimator.

    Args:
        design_options (pd.DataFrame): dataframe containing psu, stratum,
            population/design weight and/or a finite population corrector (fpc)
        jac (np.array): "jacobian" - an n x k + 1-dimensional array of first
            derivatives of the pseudo-log-likelihood function w.r.t. the parameters

    Returns:
        cluster_meat (np.array): 2d square array of length k + 1. Variance of
            the likelihood equation (Pg.557, 14-10, Greene 7th edition)

    """

    list_of_clusters = design_options["psu"].unique()
    meat = np.zeros([len(jac[0, :]), len(jac[0, :])])
    for psu in list_of_clusters:
        psu_scores = jac[design_options["psu"] == psu]
        psu_scores_sum = psu_scores.sum(axis=0)
        meat += np.dot(psu_scores_sum[:, None], psu_scores_sum[:, None].T)
    cluster_meat = len(list_of_clusters) / (len(list_of_clusters) - 1) * meat
    return cluster_meat


def stratification(design_options, jac):
    """Variance estimatio for each strata stratum.

    The function takes the sum of the jacobian observations for each cluster
    within strata. The result is the meat of the sandwich estimator.

    Args:
        design_options (pd.DataFrame): dataframe containing psu, stratum,
            population/design weight and/or a finite population corrector (fpc)
        jac (np.array): "jacobian" - an n x k + 1-dimensional array of first
            derivatives of the pseudo-log-likelihood function w.r.t. the parameters

    Returns:
        strata_meat (np.array): 2d square array of length k + 1. Variance of
        the likelihood equation

    """
    n_params = len(jac[0, :])
    stratum_col = design_options["strata"]
    # Stratification does not require clusters
    if "psu" not in design_options:
        design_options["psu"] = design_options.index
    else:
        pass
    psu_col = design_options["psu"]
    strata_meat = np.zeros([n_params, n_params])
    # Variance estimation per stratum
    for stratum in stratum_col.unique():
        psu_in_strata = psu_col[stratum_col == stratum].unique()
        psu_jac = np.zeros([n_params])
        if "fpc" in design_options:
            fpc = design_options["fpc"][stratum_col == stratum].unique()
        else:
            fpc = 1
        # psu_jac stacks the sum of the observations for each cluster.
        for psu in psu_in_strata:
            psu_jac = np.vstack([psu_jac, np.sum(jac[psu_col == psu], axis=0)])
        psu_jac_mean = np.sum(psu_jac, axis=0) / len(psu_in_strata)
        if len(psu_in_strata) > 1:
            mid_step = np.dot(
                (psu_jac[1:] - psu_jac_mean).T, (psu_jac[1:] - psu_jac_mean)
            )
            strata_meat += (
                fpc * (len(psu_in_strata) / (len(psu_in_strata) - 1)) * mid_step
            )
        # Apply "grand-mean" method for single unit stratum
        elif len(psu_in_strata) == 1:
            strata_meat += fpc * np.dot(psu_jac[1:].T, psu_jac[1:])

    return strata_meat


def robust_se(jac, hess):
    """Robust standard errors.

    Args:
        jac (np.array): "jacobian" - an n x k + 1-dimensional array of first
            derivatives of the pseudo-log-likelihood function w.r.t. the parameters
        hess (np.array): "hessian" - a k + 1 x k + 1-dimensional array of second
            derivatives of the pseudo-log-likelihood function w.r.t. the parameters

    Returns:
        se (np.array): a 1d array of k + 1 standard errors
        var (np.array): 2d variance-covariance matrix

    """
    sum_scores = np.dot((jac).T, jac)
    meat = (len(jac) / (len(jac) - 1)) * sum_scores
    se, var = sandwich_step(hess, meat)
    return se, var


def cluster_robust_se(jac, hess, design_options):
    """Cluster robust standard errors.

    A cluster is a group of observations that correlate amongst each other,
    but not between groups. Each cluster is seen as independent. As the number
    of clusters increase, the standard errors approach robust standard errors.

    Args:
        jac (np.array): "jacobian" - an n x k + 1-dimensional array of first
            derivatives of the pseudo-log-likelihood function w.r.t. the parameters
        hess (np.array): "hessian" - a k + 1 x k + 1-dimensional array of
            second derivatives of the pseudo-log-likelihood function w.r.t.
            the parameters
    Returns:
        cluster_robust_se (np.array): a 1d array of k + 1 standard errors
        cluster_robust_var (np.array): 2d variance-covariance matrix

    """
    cluster_meat = clustering(design_options, jac)
    cluster_robust_se, cluster_robust_var = sandwich_step(hess, cluster_meat)
    return cluster_robust_se, cluster_robust_var


def strata_robust_se(jac, hess, design_options):
    """Cluster robust standard errors.

    A stratum is a group of observations that share common information. Each
    stratum can be constructed based on age, gender, education, region, etc.
    The function runs the same formulation for cluster_robust_se for each
    stratum and returns the sum. Each stratum contain primary sampling units
    (psu) or clusters. If observations are independent, but wish to have to
    strata, make the psu column take the values of the index.

    Args:
        jac (np.array): "jacobian" - an n x k + 1-dimensional array of first
            derivatives of the pseudo-log-likelihood function w.r.t. the parameters
        hess (np.array): "hessian" - a k + 1 x k + 1-dimensional array of
            second derivatives of the pseudo-log-likelihood function w.r.t.
            the parameters
        design_options (pd.DataFrame): dataframe containing psu, stratum,
            population/design weight and/or a finite population corrector (fpc)

    Returns:
        strata_robust_se (np.array): a 1d array of k + 1 standard errors
        strata_robust_var (np.array): 2d variance-covariance matrix

    """
    strata_meat = stratification(design_options, jac)
    strata_robust_se, strata_robust_var = sandwich_step(hess, strata_meat)
    return strata_robust_se, strata_robust_var
