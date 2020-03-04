"""Likelihood functions and derivatives of an unconstrained model."""
import numpy as np
import pandas as pd
from patsy import dmatrices
from scipy import stats

from estimagic.optimization.optimize import maximize


def mle_processing(formulas, data):
    """Process user input for an unconstrained model for MLE.

    Args:
        formulas (str or list of strings): a list of strings to be used by
            patsy to extract dataframes of the dep. and ind. variables
        data (pd.DataFrame): A pandas DataFrame

    Returns:
        start_params (pd.DataFrame): a vector of zeroes of length k + 1
        y (np.array): an 1d array with the dependent variable
        x (np.array): a 2d array with the independent variables

    """
    # extract data arrays
    y, x = dmatrices(formulas[0], data, return_type="dataframe")
    y = y[y.columns[0]]

    # extract dimensions
    beta_names = list(x.columns)

    # set-up index for params_df
    index = beta_names

    # make params_df
    start_params = pd.DataFrame(index=index)
    start_params["value"] = np.zeros(x.shape[1])

    return start_params, y.to_numpy().astype(int), x.to_numpy()


def estimate_parameters(log_like, design_options, log_like_kwargs, dashboard=False):
    """Estimate parameters that maximize log-likelihood function.

    Args:
        log_like (function): log-likelihood function
        design_options (pd.DataFrame): dataframe containing psu, stratum,
            population/design weight and/or a finite population corrector (fpc)
        log_like_kwargs (dict): contains model equation, data, and type of
            binary-choice model
            Example:
                log_like_kwargs = {
                   "formulas": formulas,
                   "data": orig_data,
                   "model": "logit"
                   }
        dashboard (bool): Switch on the dashboard

    Returns:
        res: parameter vector and other optimization results.

    """
    params, y, x = mle_processing(log_like_kwargs["formulas"], log_like_kwargs["data"])
    res = maximize(
        criterion=log_like,
        params=params,
        algorithm="scipy_L-BFGS-B",
        criterion_kwargs={
            "y": y,
            "x": x,
            "model": log_like_kwargs["model"],
            "design_options": design_options,
        },
        dashboard=dashboard,
    )
    return res


def estimate_likelihood(params, y, x, model, design_options):
    """Pseudo-log-likelihood.

    Args:
        params (pd.DataFrame): The index consists of the parameter names,
            the "value" column are the parameter values.
        y (np.array): 1d numpy array with the dependent variable
        x (np.array): 2d numpy array with the independent variables
        model (string): "logit" or "probit"
        design_options (pd.DataFrame): dataframe containing psu, stratum,
            population/design weight and/or a finite population corrector (fpc)

    Returns:
        loglike (np.array): 1d numpy array with sum of likelihood contributions

    """
    return estimate_likelihood_obs(params, y, x, model, design_options).sum()


def estimate_likelihood_obs(params, y, x, model, design_options):
    """Pseudo-log-likelihood contribution per individual.

    Args:
        params (pd.DataFrame): The index consists of the parmater names,
            the "value" column are the parameter values.
        y (np.array): 1d numpy array with the dependent variable
        x (np.array): 2d numpy array with the independent variables
        model (string): "logit" or "probit"
        design_options (pd.DataFrame): dataframe containing psu, stratum,
            population/design weight and/or a finite population corrector (fpc)

    Returns:
        loglike (np.array): 1d numpy array with likelihood contribution per individual

    """
    q = 2 * y - 1
    if model == "logit":
        c = np.log(1 / (1 + np.exp(-(q * np.dot(x, params["value"])))))
        if "weight" in design_options.columns:
            return c * design_options["weight"].to_numpy()
        else:
            return c
    elif model == "probit":
        c = np.log(stats.norm._cdf(np.dot(q[:, None] * x, params["value"])))
        if "weight" in design_options.columns:
            return c * design_options["weight"].to_numpy()
        else:
            return c
    else:
        print("Criterion function is misspecified or not supported.")


def mle_jacobian(params, formulas, data, model, design_options):
    """Jacobian of the pseudo-log-likelihood function for each observation.

    Args:
        params (pd.DataFrame): The index consists of the parmater names,
            the "value" column are the parameter values.
        formulas (list): a list of strings to be used by patsy to extract
            dataframes of the dependent and independent variables
        data (pd.DataFrame): a pandas dataset
        model (string): "logit" or "probit"
        design_options (pd.DataFrame): dataframe containing psu, stratum,
            population/design weight and/or a finite population corrector (fpc)

    Returns:
        jacobian (np.array): an n x k + 1-dimensional array of first
            derivatives of the pseudo-log-likelihood function w.r.t. the parameters

    """
    y, x = dmatrices(formulas[0], data, return_type="dataframe")
    y = y[y.columns[0]]

    if model == "logit":
        c = 1 / (1 + np.exp(-(np.dot(x, params["value"]))))
        jacobian = (y - c)[:, None] * x
        if "weight" in design_options.columns:
            weight = design_options["weight"].to_numpy()[:, None]
            weighted_jacobian = weight / weight.mean() * jacobian.to_numpy()
            return weighted_jacobian
        else:
            return jacobian.to_numpy()

    elif model == "probit":
        pdf = stats.norm._pdf(np.dot(x, params["value"]))
        cdf = stats.norm._cdf(np.dot(x, params["value"]))
        jacobian = ((pdf / (cdf * (1 - cdf))) * (y - cdf))[:, None] * x
        if "weight" in design_options.columns:
            weight = design_options["weight"].to_numpy()[:, None]
            weighted_jacobian = weight / weight.mean() * jacobian.to_numpy()
            return weighted_jacobian
        else:
            return jacobian.to_numpy()
    else:
        print("Criterion function is misspecified or not supported.")


def mle_hessian(params, formulas, data, model, design_options):
    """Hessian matrix of the pseudo-log-likelihood function.

    Args:
        params (pd.DataFrame): The index consists of the parmater names,
            the "value" column are the parameter values.
        formulas (list): a list of strings to be used by patsy to extract
            dataframes of the dependent and independent variables
        data (pd.DataFrame): a pandas dataset
        model (string): "logit" or "probit"
        design_options (pd.DataFrame): dataframe containing psu, stratum,
            population/design weight and/or a finite population corrector (fpc)
    Returns:
        hessian (np.array): a k + 1 x k + 1-dimensional array of second derivatives
            of the pseudo-log-likelihood function w.r.t. the parameters

    """
    y, x = dmatrices(formulas[0], data, return_type="dataframe")
    y = y[y.columns[0]]

    if model == "logit":
        if "weight" in design_options.columns:
            weight = design_options["weight"].to_numpy()
            c = 1 / (1 + np.exp(-(np.dot(x, params["value"]))))
            return -np.dot(weight / weight.mean() * c * (1 - c) * x.T, x)
        else:
            c = 1 / (1 + np.exp(-(np.dot(x, params["value"]))))
            return -np.dot(c * (1 - c) * x.T, x)
    elif model == "probit":
        q = 2 * y - 1
        pdf = stats.norm._pdf(np.dot(q[:, None] * x, params["value"]))
        cdf = stats.norm._cdf(np.dot(q[:, None] * x, params["value"]))
        delt = (pdf * q) / cdf
        mid = np.dot(x, params["value"]) + delt
        if "weight" in design_options.columns:
            weight = design_options["weight"].to_numpy()
            tranpose = (
                (delt * mid)[:, None] * weight[:, None] / weight[:, None].mean() * x
            )
            hessian = np.dot(tranpose.T, x)
        else:
            tranpose = (delt * mid)[:, None] * x
            hessian = np.dot(tranpose.T, x)

        return hessian
    else:
        print("Criterion function is misspecified or not supported.")
