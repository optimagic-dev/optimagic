import pandas as pd
import numpy as np
from estimagic.differentiation.first_order_auxiliary import forward
from estimagic.differentiation.first_order_auxiliary import backward
from estimagic.differentiation.first_order_auxiliary import central
from estimagic.differentiation.first_order_auxiliary import richardson


def jacobian(
    func,
    params_sr,
    method="central",
    extrapolant=None,
    func_args=None,
    func_kwargs=None,
):
    """Calculate the jacobian of *func*.

    Args:
        func (function): A function that maps params_sr into a numpy array
            or pandas Series.
        params_sr (Series): see :ref:`parmas_df`
        method (string): The method for the computation of the derivative. Default is
                         central as it gives the highest accuracy.
        extrapolant (string): A variable indicating the use of a extrapolation method.
        func_args (list): additional positional arguments for func.
        func_kwargs (dict): additional positional arguments for func.

    Returns:
        DataFrame: If func returns a Series, the index is the index of this Series or
        the index is 0,1,2... if func returns a numpy array. The columns are the
        index of params_sr.

    """
    # set default arguments
    func_args = [] if func_args is None else func_args
    func_kwargs = {} if func_kwargs is None else func_kwargs
    # Calculate the value of the function for the observations
    f_x0 = func(params_sr, *func_args, **func_kwargs)
    if isinstance(f_x0, np.ndarray):
        jacobi = pd.DataFrame(index=range(len(f_x0)), columns=params_sr.index)
    elif isinstance(f_x0, pd.Series):
        jacobi = pd.DataFrame(index=f_x0.index, columns=params_sr.index)
    else:
        raise ValueError("Unsupported output of function with type" + str(f_x0.dtype))
    if method == "forward":
        f = forward
    elif method == "backward":
        f = backward
    elif method == "central":
        f = central
    else:
        raise ValueError("The given method was not found.")
    for var in jacobi.columns:
        # The rule of thumb for the stepsize is implemented
        h = (1 + abs(params_sr[var])) * np.sqrt(np.finfo(float).eps)
        if extrapolant == "richardson":
            # For the richardson extrapolation a seperate function is called.
            f_diff = richardson(
                f, func, f_x0, params_sr, var, h, method, *func_args, **func_kwargs
            )
        else:
            f_diff = f(func, f_x0, params_sr, var, h, *func_args, **func_kwargs)
        jacobi[var] = f_diff / h
    return jacobi
