import pandas as pd
import numpy as np
from estimagic.differentiation.gradient import gradient
from estimagic.differentiation.diff_auxiliary import forward
from estimagic.differentiation.diff_auxiliary import backward
from estimagic.differentiation.diff_auxiliary import central

def jacobian(func, params_sr, method="central", extrapolant=None, func_args=None,
        func_kwargs=None):
    """Calculate the jacobian of *func*.

    Args:
        func (function): A function that maps params_sr into a numpy array
            or pandas Series.
        params_sr (Series): see :ref:`parmas_df`
        func_args (list): additional positional arguments for func.
        func_kwargs (dict): additional positional arguments for func.

    Returns:
        jacobian (pd.DataFrame): The index is the index of the output of func
            or 0, 1, ... if func returns a numpy array. The columns are the
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
        raise ValueError('Unsupported output of function with type' + str(f_x0.dtype))
    if method == "forward":
        f = forward
    elif method == "backward":
        f = backward
    elif method == 'central':
        f = central
    else:
        raise ValueError('The given method was not found.')
    h = 2 * np.sqrt(np.finfo(float).eps)
    for var in jacobi.columns:
        if params_sr[var] + h == params_sr[var]:
            h = abs(params_sr[var]) * np.finfo(float).eps
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
        jacobi[var] = f_diff
    return jacobi
