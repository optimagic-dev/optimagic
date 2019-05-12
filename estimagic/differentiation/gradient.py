import pandas as pd
import numpy as np
from estimagic.differentiation.diff_auxiliary import forward
from estimagic.differentiation.diff_auxiliary import backward
from estimagic.differentiation.diff_auxiliary import central


def gradient(
    func, params_sr, eps_max=1e-12, method="central", extrapolant=None,
    func_args=None, func_kwargs=None
):
    """Calculate the gradient of *func*.

    Args:
        func (function): A function that maps params_sr into a float.
        params_sr (Series): see :ref:`parmas_df`
        func_args (list): additional positional arguments for func.
        func_kwargs (dict): additional positional arguments for func.
        eps_max (float): The maximum tolerance for the convergence
        method (string): The method for the computation of the derivative
    Returns:
        gradient (Series): The index is the index of params_sr.

    """
    # set default arguments
    func_args = [] if func_args is None else func_args
    func_kwargs = {} if func_kwargs is None else func_kwargs
    eps = np.finfo(float).eps
    # Calculate the value of the function for the observations
    f_x0 = func(params_sr, *func_args, **func_kwargs)
    grad = pd.Series(index=params_sr.index)
    if method == "forward":
        f = forward
    elif method == "backward":
        f = backward
    elif method == 'central':
        f = central
    else:
        raise ValueError('The given method was not found.')
    h = 2 * np.sqrt(eps)
    for var in grad.index:
        if extrapolant == 'richardson':
            f_h = f(func, f_x0, params_sr, var, h * 4, *func_args, **func_kwargs)
            f_half = f(func, f_x0, params_sr, var, h * 2, *func_args, **func_kwargs)
            f_qua = f(func, f_x0, params_sr, var, h, *func_args, **func_kwargs)
            if method == 'central':
                f_diff = (f_h - 20 * f_half + 64 * f_qua) / 45
            else:
                f_diff = (f_h - 6 * f_half + 8 * f_qua) / 3
        else:
            f_diff = f(func, f_x0, params_sr, var, h, *func_args, **func_kwargs)
        grad[var] = f_diff
    return grad
