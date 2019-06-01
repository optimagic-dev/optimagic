import pandas as pd
import numpy as np
from estimagic.differentiation.first_order_auxiliary import forward
from estimagic.differentiation.first_order_auxiliary import backward
from estimagic.differentiation.first_order_auxiliary import central
from estimagic.differentiation.first_order_auxiliary import richardson


def gradient(
    func,
    params_sr,
    method="central",
    extrapolant=None,
    func_args=None,
    func_kwargs=None,
):
    """Calculate the gradient of *func*.

    Args:
        func (function): A function that maps params_sr into a float.
        params_sr (Series): see :ref:`parmas_df`
        func_args (list): additional positional arguments for func.
        func_kwargs (dict): additional positional arguments for func.
        extrapolant (string): A variable indicating the use of a extrapolation method.
        method (string): The method for the computation of the derivative. Default is
                         central as it gives the highest accuracy.
    Returns:
        Series: The index is the index of params_sr.

    """
    # set default arguments
    func_args = [] if func_args is None else func_args
    func_kwargs = {} if func_kwargs is None else func_kwargs
    # Each directional derivative is indexed by variable name
    grad = pd.Series(index=params_sr.index)
    # Calculate the value of the function for the observations
    f_x0 = func(params_sr, *func_args, **func_kwargs)
    # Check which method is given and if it is supported
    if method == "forward":
        f = forward
    elif method == "backward":
        f = backward
    elif method == "central":
        f = central
    else:
        raise ValueError("The given method was not found.")
    for var in grad.index:
        # The rule of thumb for the stepsize is implemented
        h = (1 + abs(params_sr[var])) * np.sqrt(np.finfo(float).eps)
        if extrapolant == "richardson":
            f_diff = richardson(
                f, func, f_x0, params_sr, var, h, method, *func_args, **func_kwargs
            )
        else:
            f_diff = f(func, f_x0, params_sr, var, h, *func_args, **func_kwargs)
        grad[var] = f_diff / h
    return grad
