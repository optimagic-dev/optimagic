"""Likelihood functions and derivatives of a logit model."""
import numpy as np


def logit_loglike(params, y, x):
    """Log-likelihood function of a logit model.

    Args:
        params (Series): The index consists of the parmater names,
            the values are the parameters.
        y (np.array): 1d numpy array with the dependent variable
        x (np.array): 2d numpy array with the independent variables

    Returns:
        loglike (np.array): 1d numpy array with likelihood contribution  per individual

    """
    return logit_loglikeobs(params, y, x).sum()


def logit_loglikeobs(params, y, x):
    """Log-likelihood contribution per individual of a logit model.

    Args:
        params (Series): The index consists of the parmater names,
            the values are the parameters.
        y (np.array): 1d numpy array with the dependent variable
        x (np.array): 2d numpy array with the independent variables

    Returns:
        loglike (np.array): 1d numpy array with likelihood contribution per individual

    """
    q = 2 * y - 1
    return np.log(1 / (1 + np.exp(-(q * np.dot(x, params)))))


def logit_gradient(params, y, x):
    """Gradient of the log-likelihood for each observation of a logit model.

    Args:
        parmas (Series): The index consists of the parmater names,
            the values are the parameters.
        y (np.array): 1d numpy array with the dependent variable
        x (np.array): 2d numpy array with the independent variables

    Returns:
        gradient (np.array)

    """
    y = y
    c = 1 / (1 + np.exp(-(np.dot(x, params))))
    return np.dot(y - c, x)


def logit_jacobian(params, y, x):
    """Jacobian of the log-likelihood for each observation of a logit model.

    Args:
        params (Series): The index consists of the parmater names,
            the values are the parameters.
        y (np.array): 1d numpy array with the dependent variable
        x (np.array): 2d numpy array with the independent variables

    Returns:
        jac : array-like
            The derivative of the loglikelihood for each observation evaluated
            at `params`.
    """
    y = y
    c = 1 / (1 + np.exp(-(np.dot(x, params))))
    return (y - c)[:, None] * x


def logit_hessian(params, y, x):
    """Hessian matrix of the log-likelihood.

    Args:
        params (Series): The index consists of the parmater names,
            the values are the parameters.
        y (np.array): 1d numpy array with the dependent variable
        x (np.array): 2d numpy array with the independent variables

    Returns:
        hessian (np.array) : 2d numpy array with the hessian of the
            logl-ikelihood function evaluated at `params`

    """
    c = 1 / (1 + np.exp(-(np.dot(x, params))))
    return -np.dot(c * (1 - c) * x.T, x)
