import pandas as pd
import numpy as np
from estimagic.differentiation.diff_auxiliary import forward_hess
from estimagic.differentiation.diff_auxiliary import backward_hess
from estimagic.differentiation.diff_auxiliary import central_hess
from estimagic.differentiation.diff_auxiliary import hess_richardson


def hessian(func, params_sr, method="central", extrapolant=None, func_args=None,
        func_kwargs=None):
    """Calculate the hessian of *func*.

    Args:
        func (function): A function that maps params_sr into a float.

        params_sr (Series): see :ref:`parmas_df` for details.

        func_args (list): additional positional arguments for func.
        func_kwargs (dict): additional positional arguments for func.

    Returns:
        gradient (Series): The index and columns are the index of params_sr.

    """
    # set default arguments
    func_args = [] if func_args is None else func_args
    func_kwargs = {} if func_kwargs is None else func_kwargs
    hess = pd.DataFrame(index=params_sr.index, columns=params_sr.index, dtype=float)
    f_x0 = func(params_sr, *func_args, **func_kwargs)
    if not extrapolant:
        if method == "forward":
            f = forward_hess
        elif method == "backward":
            f = backward_hess
        elif method == 'central':
            f = central_hess
        else:
            raise ValueError('The given method was not found.')
    elif extrapolant == 'richardson':
        f = hess_richardson
    else:
        raise ValueError('The given extrapolation method is not supported')
    for i, var_1 in enumerate(hess.index):
        h_1 = (1. + abs(params_sr[var_1])) * np.cbrt(np.finfo(float).eps)
        for var_2 in hess.index[i:]:
            h_2 = (1. + abs(params_sr[var_2])) * np.cbrt(np.finfo(float).eps)
            f_diff = f(func, f_x0, params_sr, var_1, var_2, h_1, h_2, method,
                       *func_args, **func_kwargs) / (h_1 * h_2)
            hess.loc[var_1, var_2] = f_diff
            hess.loc[var_2, var_1] = f_diff
    return hess




