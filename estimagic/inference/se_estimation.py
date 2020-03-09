"""Variance estimators for maximum likelihood."""
import numpy as np
import pandas as pd

from estimagic.differentiation.differentiation import hessian
from estimagic.differentiation.differentiation import jacobian


def np_jac(log_like_obs, params, method, log_like_kwargs):
    """We wrote this function because we did not want to touch the
    differentiation files.

    """
    numpy_jacobian = jacobian(log_like_obs, params, method, func_kwargs=log_like_kwargs)
    return numpy_jacobian.to_numpy()


def np_hess(log_like, params, method, log_like_kwargs):
    """We wrote this function because we did not want to touch the
    differentiation files.

    """
    numpy_hessian = hessian(log_like, params, method, func_kwargs=log_like_kwargs)
    return numpy_hessian.to_numpy()


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


def variance_estimator(jac=None, hess=None, design_options=None, cov_type=None):
    """Chooses the appropriate variance estimator.

    Args:
        jac (np.array): "jacobian" - an n x k + 1-dimensional array of first
            derivatives of the pseudo-log-likelihood function w.r.t. the parameters
        hess (np.array): "hessian" - a k + 1 x k + 1-dimensional array of
            second derivatives of the pseudo-log-likelihood function w.r.t.
            the parameters
        design_options (pd.DataFrame): dataframe containing psu, stratum,
            population/design weight and/or a finite population corrector (fpc)
        cov_type (str): One of ["opg", "oim", "sandwich"]. opg and oim only
            work when *design_options* is None. This takes the cov_type specified
            in the likelihood_inference function.

    Returns:
        se (np.array): a 1d array of k + 1 standard errors
        var (np.array): 2d variance-covariance matrix

    Examples:

        >>> ve = variance_estimator
        >>> small_jac = np.array([[0.267383, 1.33691], [0.306403, 1.83842]])
        >>> small_hess = np.array([[-4053.07, -21604.3], [-21604.3, -137843]])
        >>> d_opt = pd.DataFrame()
        >>> j = "jacobian"
        >>> h = "hessian"
        >>> s = "sandwich"

        >>> se_jac, var_jac = ve(jac=small_jac, design_options=d_opt, cov_type=j)
        >>> se_jac, var_jac
        (array([27.74510442,  4.9636265 ]), array([[ 769.79081933, -137.17437899],
               [-137.17437899,   24.637588  ]]))

        >>> se_hess, var_hess = ve(hess=small_hess, design_options=d_opt, cov_type=h)
        >>> se_hess, var_hess
        (array([0.0387201 , 0.00663951]), array([[ 1.49924600e-03, -2.34978637e-04],
               [-2.34978637e-04,  4.40831161e-05]]))

        >>> se_s, var_s = ve(small_jac, small_hess, d_opt, cov_type=s)
        >>> se_s
        array([1.28620084e-04, 1.39268467e-05])

        >>> se, var = ve(hess=small_hess, design_options=d_opt, cov_type="turtles")
        Traceback (most recent call last):
            ...
        Exception: Unsupported or incorrect cov_type specified.

    """
    if design_options.empty or (
        "weight" in design_options and len(design_options) == 1
    ):
        if cov_type == "jacobian":
            opg_se, opg_var = outer_product_of_gradients(jac)
            return opg_se, opg_var
        elif cov_type == "hessian":
            oim_se, oim_var = observed_information_matrix(hess)
            return oim_se, oim_var
        elif cov_type == "sandwich":
            sandwich_se, sandwich_var = robust_se(jac, hess)
            return sandwich_se, sandwich_var
        else:
            raise Exception("Unsupported or incorrect cov_type specified.")

    elif ("psu" in design_options) and ("strata" not in design_options):
        cluster_se, cluster_var = cluster_robust_se(jac, hess, design_options)
        return cluster_se, cluster_var

    elif ("strata") and ("psu" not in design_options):
        strata_se, strata_var = strata_robust_se(jac, hess, design_options)
        return strata_se, strata_var

    elif "psu" and "strata" in design_options:
        strata_se, strata_var = strata_robust_se(jac, hess, design_options)
        return strata_se, strata_var

    else:
        raise Exception("Check design options specified.")


def choose_case(log_like_obs, params, log_like_kwargs, design_options, cov_type):
    """Creates necessary objects for the variance estimator.

    Args:
        log_like_obs (func): The pseudo-log-likelihood function. It is the
            log-likelihood contribution per individual.
        params (pd.DataFrame): The index consists of the paramater names specified
            by the user, the "value" column is the parameter values.
        log_like_kwargs (dict): In addition to the params argument directly
            taken by likelihood_inference function, additional keyword arguments for the
            likelihood function may include dependent variable, independent variables
            and design options.
            Example of simple logit model arguments:
                log_like_kwargs = {
                    "y": y,
                    "x": x,
                    "design_options": design_options
                }
        design_options (pd.DataFrame): dataframe containing psu, stratum,
            population/design weight and/or a finite population corrector (fpc)
        cov_type (str): One of ["opg", "oim", "sandwich"]. opg and oim only
            work when *design_options* is empty. This takes the cov_type specified
            in the likelihood_inference function.

    Returns:
        se (np.array): a 1d array of k + 1 standard errors
        var (np.array): 2d variance-covariance matrix

    Examples:

        >>> from estimagic.inference.sample_models import logit
        >>> from estimagic.inference.sample_models import probit
        >>> cc = choose_case
        >>> params = pd.DataFrame(data=[0.5, 0.5], columns=["value"])
        >>> x = np.array([[1., 5.], [1., 6.]])
        >>> y = np.array([[1., 1]])
        >>> d_opt = pd.DataFrame()
        >>> logit_kwargs = {"y": y, "x": x, "design_options": d_opt}

        >>> cc(logit, params, logit_kwargs, d_opt, cov_type="jacobian")
        (array([212.37277788,  40.10565957]), array([[45102.19678307, -8486.9195158 ],
               [-8486.9195158 ,  1608.46392969]]))
        >>> cc(logit, params, logit_kwargs, d_opt, cov_type="hessian")
        (array([40.93302927,  7.56841945]), array([[1675.51288498, -308.54018839],
               [-308.54018839,   57.28097291]]))
        >>> cc(logit, params, logit_kwargs, d_opt, cov_type="sandwich")
        (array([11.50709079,  2.08007668]), array([[132.41313852, -23.8377008 ],
               [-23.8377008 ,   4.32671901]]))
        >>> cc(logit, params, logit_kwargs, d_opt, cov_type="turtles")
        Traceback (most recent call last):
            ...
        Exception: Incorrect or unsupported cov_type specified.
        >>> d_opt = pd.DataFrame(data=[1, 2], columns=["psu"])
        >>> cc(logit, params, logit_kwargs, d_opt, cov_type="opg")
        Traceback (most recent call last):
            ...
        Exception: Specifying psu or strata does not allow for oim and opg estimation.
    """

    def log_like(params, **log_like_kwargs):
        return log_like_obs(params, **log_like_kwargs).sum()

    if cov_type != "sandwich" and ("psu" or "strata") in design_options:
        raise Exception(
            "Specifying psu or strata does not allow for oim and opg estimation."
        )
    if cov_type == "jacobian":
        jac = np_jac(
            log_like_obs, params, method="central", log_like_kwargs=log_like_kwargs
        )
        se, var = variance_estimator(
            jac=jac, design_options=design_options, cov_type=cov_type
        )
    elif cov_type == "hessian":
        hess = np_hess(
            log_like, params, method="central", log_like_kwargs=log_like_kwargs
        )
        se, var = variance_estimator(
            hess=hess, design_options=design_options, cov_type=cov_type
        )
    elif cov_type == "sandwich":
        jac = np_jac(
            log_like_obs, params, method="central", log_like_kwargs=log_like_kwargs
        )
        hess = np_hess(
            log_like, params, method="central", log_like_kwargs=log_like_kwargs
        )
        se, var = variance_estimator(
            jac=jac, hess=hess, design_options=design_options, cov_type=cov_type
        )
    else:
        raise Exception("Incorrect or unsupported cov_type specified.")
    return se, var


def likelihood_inference(
    log_like_obs, params, log_like_kwargs, design_options, cov_type="jacobian"
):
    """Pseudolikelihood estimation and inference.

    Args:
        log_like_obs (func): The pseudo-log-likelihood function. It is the
            log-likelihood contribution per individual.
        params (pd.DataFrame): The index consists of the paramater names specified
            by the user, the "value" column is the parameter values.
        log_like_kwargs (dict): In addition to the params argument directly
            taken by likelihood_inference function, additional keyword arguments for the
            likelihood function may include dependent variable, independent variables
            and design options.
            Example of simple logit model arguments:
                log_like_kwargs = {
                    "y": y,
                    "x": x,
                    "design_options": design_options
                }
        design_options (pd.DataFrame): dataframe containing psu, stratum,
            population/design weight and/or a finite population corrector (fpc)
        cov_type (str): One of ["opg", "oim", "sandwich"]. opg and oim only
            work when *design_options* is empty. opg is default.

    Returns:
        model_inference_table (pd.DataFrame):
            - "value": params that maximize likelihood
            - "standard_error": standard errors of the params
            - "ci_lower": using the 95% critical value of a normal distribution * -1
            - "ci_upper": using the 95% critical value of a normal distribution
        params_cov (pd.DataFrame): Covariance matrix of estimated parameters.
            Index and columns are the same as params.index.

    Examples:

        >>> from estimagic.inference.sample_models import logit
        >>> cc = choose_case
        >>> params = pd.DataFrame(data=[0.5, 0.5], columns=["value"])
        >>> x = np.array([[1., 5.], [1., 6.]])
        >>> y = np.array([[1., 1]])
        >>> d_opt = pd.DataFrame()
        >>> logit_kwargs = {"y": y, "x": x, "design_options": d_opt}
        >>> j = "jacobian"
        >>> se, var = cc(logit, params, logit_kwargs, d_opt, cov_type=j)
        >>> se, var
        (array([212.37277788,  40.10565957]), array([[45102.19678307, -8486.9195158 ],
               [-8486.9195158 ,  1608.46392969]]))

        >>> inf_table, cov = inference_table(params, se, var, cov_type="jacobian")
        >>> li = likelihood_inference
        >>> li(logit, params, logit_kwargs, d_opt, j) #doctest: +NORMALIZE_WHITESPACE
        (   value  jacobian_standard_errors    ci_lower    ci_upper
         0    0.5                212.372778 -415.750645  416.750645
         1    0.5                 40.105660  -78.107093   79.107093,
                       0            1
         0  45102.196783 -8486.919516
         1  -8486.919516  1608.463930)

    """
    log_like_se, log_like_var = choose_case(
        log_like_obs, params, log_like_kwargs, design_options, cov_type
    )
    model_inference_table, params_cov = inference_table(
        params, log_like_se, log_like_var, cov_type
    )
    return model_inference_table, params_cov


def inference_table(params, se, var, cov_type):
    """Creates table of parameters, standard errors, and confidence intervals.

    Args:
        params (pd.DataFrame): The index consists of the parmater names,
            the "value" column are the parameter values.
        se (np.array): a 1d array of k + 1 standard errors
        var (np.array): 2d variance-covariance matrix
        cov_type (str): variance estimator.

    Returns:
        params_df (pd.DataFrame): columns of the parameter values, standard
            errors, and a 95% confidence interval of the parameter.
        cov (pd.DataFrame): the variance-covariance matrix with parameter names
            for the index and columns

    """
    params_df = pd.DataFrame()
    params_df["value"] = params["value"]
    params_df["{}_standard_errors".format(cov_type)] = se
    params_df["ci_lower"] = (
        params_df["value"] - 1.96 * params_df["{}_standard_errors".format(cov_type)]
    )
    params_df["ci_upper"] = (
        params_df["value"] + 1.96 * params_df["{}_standard_errors".format(cov_type)]
    )
    cov = pd.DataFrame(data=var, columns=params.index, index=params.index)
    return params_df, cov


if __name__ == "__main__":
    import doctest

    doctest.testmod()
