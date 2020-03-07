"""Variance estimators for maximum likelihood."""
import numpy as np
import pandas as pd
from estimagic.differentiation.differentiation import jacobian
from estimagic.differentiation.differentiation import hessian


def design_options_preprocessing(data, design_dict=None):
    """Construct design options dataframe for parameter and variance estimation.



    Args:
        data (pd.DataFrame): a pandas dataset
        design_dict (dict): a dictionary where the keys are the survey properties
            and the values are the respective column names

        Example:
            design_dict = {"psu": "school", "strata: "stratum", "weight": "dweight"}

        Key-value descriptions:
            psu (string): name of column by which you wish to cluster
            strata (string): name of column that defines the strata
            weight (string): name of column that defines the weight.
                If you have both probability weights and design weights,
                multiply the columns to create a new weight column.
            fpc (string): name of column that defines the finite population
                correction. Value is unique to each strata and is less than
                1. If unspecified, value equals 1.

    Returns:
        design_options (pd.DataFrame): dataframe containing psu, stratum,
            population/design weight and/or a finite population corrector (fpc)

    """
    design_options = pd.DataFrame()
    if design_dict is None:
        pass
    else:
        for option in design_dict.keys():
            design_options[option] = data[design_dict[option]]
    return design_options


def observed_information_matrix(hess):
    """Observed information matrix or BHHH estimator.

    Args:
        hessian (np.array): a k + 1 x k + 1-dimensional array of second derivatives
            of the pseudo-log-likelihood function w.r.t. the parameters
        model (string): 'logit' or 'probit'

    """
    hess = -hess.copy()
    oim_var = np.linalg.inv(hess)
    oim_se = np.sqrt(np.diag(oim_var))
    return oim_se, oim_var


def outer_product_of_gradients(jac):
    """Outer product of gradients estimator.

    Args:
        jacobian (np.array): an n x k + 1-dimensional array of first
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
        hessian (np.array): 2d numpy array with the hessian of the
            pseudo-log-likelihood function evaluated at `params`
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
        jacobian (np.array): an n x k + 1-dimensional array of first
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
        jacobian (np.array): an n x k + 1-dimensional array of first
            derivatives of the pseudo-log-likelihood function w.r.t. the parameters

    Returns:
        strata_meat (np.array): 2d square array of length k + 1. Variance of
        the likelihood equation

    """
    n_var = len(jac[0, :])
    stratum_col = design_options["strata"]
    if "psu" not in design_options.columns:
        design_options["psu"] = design_options.index
    else:
        pass
    psu_col = design_options["psu"]
    strata_meat = np.zeros([n_var, n_var])
    for stratum in stratum_col.unique():
        n_psu = psu_col[stratum_col == stratum].unique()
        psu_jac = np.zeros([n_var])
        if "fpc" in design_options.columns:
            fpc = design_options["fpc"][stratum_col == stratum].unique()
        else:
            fpc = 1
        for psu in n_psu:
            psu_jac = np.vstack([psu_jac, np.sum(jac[psu_col == psu], axis=0)])
        psu_jac_mean = np.sum(psu_jac, axis=0) / len(n_psu)
        if len(n_psu) > 1:
            mid_step = np.dot(
                (psu_jac[1:] - psu_jac_mean).T, (psu_jac[1:] - psu_jac_mean)
            )
            strata_meat += fpc * (len(n_psu) / (len(n_psu) - 1)) * mid_step
        elif len(n_psu) == 1:
            strata_meat += fpc * np.dot(psu_jac[1:].T, psu_jac[1:])
        else:
            break

    return strata_meat


def robust_se(jac, hess):
    """Robust standard errors for binary estimation.

    Args:
        jacobian (np.array): an n x k + 1-dimensional array of first
            derivatives of the pseudo-log-likelihood function w.r.t. the parameters
        hessian (np.array): a k + 1 x k + 1-dimensional array of second derivatives
            of the pseudo-log-likelihood function w.r.t. the parameters

    Returns:
        robust_se (np.array): a 1d array of k + 1 standard errors
        robust_var (np.array): 2d variance-covariance matrix

    """
    sum_scores = np.dot((jac).T, jac)
    meat = (len(jac) / (len(jac) - 1)) * sum_scores
    se, var = sandwich_step(hess, meat)
    return se, var


def cluster_robust_se(jac, hess, design_options):
    """Cluster robust standard errors for logit estimation.

    A cluster is a group of observations that correlate amongst each other,
    but not between groups. Each cluster is seen as independent. As the number
    of clusters increase, the standard errors approach robust standard errors.

    Args:
        jacobian (np.array): an n x k + 1-dimensional array of first
            derivatives of the pseudo-log-likelihood function w.r.t. the parameters
        hessian (np.array): a k + 1 x k + 1-dimensional array of second derivatives
            of the pseudo-log-likelihood function w.r.t. the parameters

    Returns:
        cluster_robust_se (np.array): a 1d array of k + 1 standard errors
        cluster_robust_var (np.array): 2d variance-covariance matrix

    """
    cluster_meat = clustering(design_options, jac)
    cluster_robust_se, cluster_robust_var = sandwich_step(hess, cluster_meat)
    return cluster_robust_se, cluster_robust_var


def strata_robust_se(jac, hess, design_options):
    """Cluster robust standard errors for logit estimation.

    A stratum is a group of observations that share common information. Each
    stratum can be constructed based on age, gender, education, region, etc.
    The function runs the same formulation for cluster_robust_se for each
    stratum and returns the sum. Each stratum contain primary sampling units
    (psu) or clusters. If observations are independent, but wish to have to
    strata, make the psu column take the values of the index.

    Args:
        jacobian (np.array): an n x k + 1-dimensional array of first
            derivatives of the pseudo-log-likelihood function w.r.t. the parameters
        hessian (np.array): a k + 1 x k + 1-dimensional array of second derivatives
            of the pseudo-log-likelihood function w.r.t. the parameters
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
        jacobian (np.array): an n x k + 1-dimensional array of first
            derivatives of the pseudo-log-likelihood function w.r.t. the parameters
        hessian (np.array): a k + 1 x k + 1-dimensional array of second derivatives
            of the pseudo-log-likelihood function w.r.t. the parameters
        design_options (pd.DataFrame): dataframe containing psu, stratum,
            population/design weight and/or a finite population corrector (fpc)

    Returns:
        se (np.array): a 1d array of k + 1 standard errors
        var (np.array): 2d variance-covariance matrix

    """
    if design_options.empty or (
        "weight" in design_options.columns and len(design_options.columns) == 1
    ):
        if cov_type == "opg":
            opg_se, opg_var = outer_product_of_gradients(jac)
            return opg_se, opg_var
        elif cov_type == "oim":
            oim_se, oim_var = observed_information_matrix(hess)
            return oim_se, oim_var
        elif cov_type == "sandwich":
            sandwich_se, sandwich_var = robust_se(jac, hess)
            return sandwich_se, sandwich_var
        else:
            raise Exception("Unsupported or incorrect cov_type specified.")

    elif ("psu" in design_options.columns) and ("strata" not in design_options.columns):
        cluster_se, cluster_var = cluster_robust_se(
            jac, hess, design_options
        )
        return cluster_se, cluster_var

    elif ("strata") and ("psu" not in design_options.columns):
        strata_se, strata_var = strata_robust_se(
            jac, hess, design_options
        )
        return strata_se, strata_var

    elif "psu" and "strata" in design_options.columns:
        strata_se, strata_var = strata_robust_se(
            jac, hess, design_options
        )
        return strata_se, strata_var

    else:
        raise Exception("Check design options specified.")


def choose_case(log_like_obs, params, log_like_kwargs, design_options, cov_type):
    """Chooses variance estimation.

    Args:
        params (pd.DataFrame): The index consists of the parmater names,
            the "value" column are the parameter values.
        formulas (string or list of strings): a list of strings to be used by
        patsy to extract dataframes of the dependent and independent variables
        data (pd.DataFrame): a pandas dataset
        model (str): "logit" or "probit"
        design_dict (dict): dicitonary containing specified design options
        design_options (pd.DataFrame): dataframe containing psu, stratum,
            population/design weight and/or a finite population corrector (fpc)
        cov_type (str): variance estimator.

    Returns:
        se (np.array): a 1d array of k + 1 standard errors
        var (np.array): 2d variance-covariance matrix

    """
    def log_like(params, **log_like_kwargs):
        return log_like_obs(params, **log_like_kwargs).sum()
        
    if cov_type != "sandwich" and "weight" not in design_options.columns):
        raise Exception("Specifying psu or strata does not allow for oim and opg estimation.")
    if cov_type == "opg":
        jac = jacobian(log_like_obs, params, method="central", func_kwargs=log_like_kwargs).to_numpy()
        se, var = variance_estimator(
            jac=jac, design_options=design_options, cov_type=cov_type
        )
    elif cov_type == "oim":
        hess = hessian(log_like, params, method="central", func_kwargs=log_like_kwargs).to_numpy()
        se, var = variance_estimator(
            hess=hess, design_options=design_options, cov_type=cov_type
        )
    elif cov_type == "sandwich":
        jac = jacobian(log_like_obs, params, method="central", func_kwargs=log_like_kwargs).to_numpy()
        hess = hessian(log_like, params, method="central", func_kwargs=log_like_kwargs).to_numpy()
        se, var = variance_estimator(
            jac=jac, hess=hess, design_options=design_options, cov_type=cov_type
        )
    else:
        raise Exception("Incorrect or unsupported cov_type specified.")
    return se, var


def likelihood_inference(log_like_obs, params, log_like_kwargs, design_options, cov_type="opg"):
    """Pseudolikelihood estimation and inference.

    Args:
        log_like_kwargs (dict): Additional keyword arguments for the
            likelihood function.
            Example:
                log_like_kwargs = {
                    "formulas": equation,
                    "data": orig_data,
                    "model": "probit"
                }
        design_dict (dict): dicitonary containing specified design options
        cov_type (str): One of ["opg", "oim", "sandwich"]. opg and oim only
            work when *design_dict* is None. opg is default.

    Returns:
        params (pd.DataFrame): params that maximize likelihood
            - "standard_error"
            - "ci_lower"
            - "ci_upper"
        cov (pd.DataFrame): Covariance matrix of estimated parameters. Index and columns
            are the same as params.index.

    """
    log_like_se, log_like_var = choose_case(log_like_obs, params, log_like_kwargs, design_options, cov_type)
    params_df, cov = inference_table(params, log_like_se, log_like_var, cov_type=cov_type)
    return params_df, cov


def inference_table(params, se, var, cov_type):
    """Creates table parametera, standard errors, and confidence intervals.

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
