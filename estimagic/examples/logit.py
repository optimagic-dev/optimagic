"""Likelihood functions and derivatives of a logit model."""
import numpy as np


def logit_loglike_and_derivative(params, y, x):
    return logit_loglike(params, y, x), logit_derivative(params, y, x)


def logit_loglike(params, y, x):
    """Log-likelihood function of a logit model.

    Args:
        params (pd.DataFrame): The index consists of the parameter names,
            the "value" column are the parameter values.
        y (np.array): 1d numpy array with the dependent variable
        x (np.array): 2d numpy array with the independent variables

    Returns:
        loglike (np.array): 1d numpy array with likelihood contribution  per individual

    """
    q = 2 * y - 1
    contribs = np.log(1 / (1 + np.exp(-(q * np.dot(x, params["value"])))))

    out = {"value": contribs.sum(), "contributions": contribs}

    return out


def logit_derivative(params, y, x):
    """Derivative of the log-likelihood for each observation of a logit model.

    Args:
        params (pd.DataFrame): The index consists of the parmater names,
            the "value" column are the parameter values.
        y (np.array): 1d numpy array with the dependent variable
        x (np.array): 2d numpy array with the independent variables

    Returns:
        jac : array-like
            The derivative of the loglikelihood for each observation evaluated
            at `params`.
    """
    c = 1 / (1 + np.exp(-(np.dot(x, params["value"]))))
    jac = (y - c)[:, None] * x
    grad = jac.sum(axis=0)
    out = {"value": grad, "contributions": jac}
    return out


def logit_hessian(params, y, x):
    """Hessian matrix of the log-likelihood.

    Args:
        params (pd.DataFrame): The index consists of the parmater names,
            the "value" column are the parameter values.
        y (np.array): 1d numpy array with the dependent variable
        x (np.array): 2d numpy array with the independent variables

    Returns:
        hessian (np.array) : 2d numpy array with the hessian of the
            logl-ikelihood function evaluated at `params`

    """
    c = 1 / (1 + np.exp(-(np.dot(x, params["value"]))))
    return -np.dot(c * (1 - c) * x.T, x)
