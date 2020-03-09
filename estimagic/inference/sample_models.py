"""Likelihood functions."""
import numpy as np
from scipy import stats


def logit(params, y, x, design_options):
    """Logit model. Pseudo-log-likelihood contribution per individual.

    Args:
        params (pd.DataFrame): The index consists of the parmater names,
            the "value" column are the parameter values.
        y (np.array): 1d numpy array with the dependent variable
        x (np.array): 2d numpy array with the independent variables
        design_options (pd.DataFrame): dataframe containing psu, stratum,
            population/design weight and/or a finite population corrector (fpc)

    Returns:
        loglike (np.array): 1d numpy array with likelihood contribution per individual

    Examples:

        >>> import pandas as pd
        >>> params = pd.DataFrame(data=[0.5, 0.5], columns=["value"])
        >>> y = np.array([[1., 1]])
        >>> x = np.array([[1., 5.], [1., 6.]])
        >>> d_opt = pd.DataFrame()
        >>> logit(params, y, x, d_opt)
        array([[-0.04858735, -0.02975042]])
        >>> d_opt = pd.DataFrame(data=[0.8, 0.2], columns=["weight"])
        >>> logit(params, y, x, d_opt)
        array([[-0.03886988, -0.00595008]])

    """
    q = 2 * y - 1
    c = np.log(1 / (1 + np.exp(-(q * np.dot(x, params["value"])))))
    if "weight" in design_options.columns:
        return c * design_options["weight"].to_numpy()
    else:
        return c


def probit(params, y, x, design_options):
    """Probit model. Pseudo-log-likelihood contribution per individual.

    Args:
        params (pd.DataFrame): The index consists of the parmater names,
            the "value" column are the parameter values.
        y (np.array): 1d numpy array with the dependent variable
        x (np.array): 2d numpy array with the independent variables
        design_options (pd.DataFrame): dataframe containing psu, stratum,
            population/design weight and/or a finite population corrector (fpc)

    Returns:
        loglike (np.array): 1d numpy array with likelihood contribution per individual

    Examples:

        >>> import pandas as pd
        >>> params = pd.DataFrame(data=[0.5, 0.5], columns=["value"])
        >>> y = np.array([[1., 1]])
        >>> x = np.array([[1., 5.], [1., 6.]])
        >>> d_opt = pd.DataFrame()
        >>> probit(params, y, x, d_opt)
        array([[-0.00135081, -0.00023266]])
        >>> d_opt = pd.DataFrame(data=[0.8, 0.2], columns=["weight"])
        >>> probit(params, y, x, d_opt)
        array([[-1.08064797e-03, -4.65312283e-05]])

    """
    q = 2 * y - 1
    c = np.log(stats.norm._cdf(np.dot(q[:, None] * x, params["value"])))
    if "weight" in design_options.columns:
        return c * design_options["weight"].to_numpy()
    else:
        return c


if __name__ == "__main__":
    import doctest

    doctest.testmod()
