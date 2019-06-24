import numdifftools as nd
import numpy as np
import pandas as pd

from estimagic.differentiation.first_order_auxiliary import backward
from estimagic.differentiation.first_order_auxiliary import central
from estimagic.differentiation.first_order_auxiliary import forward


def jacobian(
    func,
    params_sr,
    method="central",
    extrapolation="richardson",
    func_args=None,
    func_kwargs=None,
):
    """
    Calculate the jacobian of *func*.

    Args:
        func (function): A function that maps params_sr into a numpy array
                        or pandas Series.

        params_sr (Series): see :ref:`parmas_df`
        method (string): The method for the computation of the derivative. Default is
                         central as it gives the highest accuracy.
        extrapolation (string): This variable allows to specify the use of the
                                richardson extrapolation.
        func_args (list): additional positional arguments for func.
        func_kwargs (dict): additional positional arguments for func.

    Returns:
        DataFrame: If func returns a Series, the index is the index of this Series or
        the index is 0,1,2... if func returns a numpy array. The columns are the
        index of params_sr.

    """
    if method not in ["central", "forward", "backward"]:
        raise ValueError("The given method is not supported.")
    # set default arguments
    func_args = [] if func_args is None else func_args
    func_kwargs = {} if func_kwargs is None else func_kwargs
    f_x0 = func(params_sr, *func_args, **func_kwargs)
    if extrapolation == "richardson":
        # For the richardson extrapolation we use, the numdifftools library.
        jac_np = nd.Jacobian(func, method=method)(params_sr, *func_args, **func_kwargs)
        if isinstance(f_x0, pd.Series):
            jac = pd.DataFrame(index=f_x0.index, columns=params_sr.index, data=jac_np)
        else:
            jac = pd.DataFrame(columns=params_sr.index, data=jac_np)
        return jac
    else:
        if method == "forward":
            f = forward
        elif method == "backward":
            f = backward
        else:
            f = central
        if isinstance(f_x0, pd.Series):
            jac = pd.DataFrame(index=f_x0.index, columns=params_sr.index)
        else:
            jac = pd.DataFrame(columns=params_sr.index)
        for var in jac.columns:
            # The rule of thumb for the stepsize is implemented
            h = (1 + abs(params_sr[var])) * np.sqrt(np.finfo(float).eps)
            f_diff = f(func, f_x0, params_sr, var, h, *func_args, **func_kwargs)
            jac[var] = f_diff / h
        return jac
