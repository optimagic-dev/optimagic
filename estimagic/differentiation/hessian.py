import numdifftools as nd
import numpy as np
import pandas as pd

from estimagic.differentiation.first_order_auxiliary import central


def hessian(
    func,
    params_sr,
    method="central",
    extrapolation=True,
    func_args=None,
    func_kwargs=None,
):
    """
    Calculate the hessian of *func*.

    Args:
        func (function): A function that maps params_sr into a float.
        params_sr (Series): see :ref:`parmas_df`
        method (string): The method for the computation of the derivative. Default is
                         central as it gives the highest accuracy.
        extrapolation (bool): This variable allows to specify the use of the
                                richardson extrapolation.
        func_args (list): additional positional arguments for func.
        func_kwargs (dict): additional positional arguments for func.

    Returns:
        DataFrame: The index and columns are the index of params_sr.

    """
    if method != "central":
        raise ValueError("The given method is not supported.")
    # set default arguments
    func_args = [] if func_args is None else func_args
    func_kwargs = {} if func_kwargs is None else func_kwargs
    if extrapolation:
        # For the richardson extrapolation we use, the numdifftools library.
        hess_np = nd.Hessian(func, method=method)(params_sr, *func_args, **func_kwargs)
        return pd.DataFrame(
            data=hess_np, index=params_sr.index, columns=params_sr.index
        )
    else:
        hess = pd.DataFrame(index=params_sr.index, columns=params_sr.index, dtype=float)
        for var_1 in params_sr.index:
            h_1 = (1.0 + abs(params_sr[var_1])) * np.cbrt(np.finfo(float).eps)
            for var_2 in params_sr.index:
                h_2 = (1.0 + abs(params_sr[var_2])) * np.cbrt(np.finfo(float).eps)
                params_r = params_sr.copy()
                params_r[var_2] += h_2
                # Calculate the first derivative w.r.t. var_1 at (params_sr + h_2) with
                # the central method. This is not the right f_x0, but the real one
                # isn't needed for
                # the central method.
                f_plus = central(
                    func, None, params_r, var_1, h_1, *func_args, **func_kwargs
                )
                params_l = params_sr.copy()
                params_l[var_2] -= h_2
                # Calculate the first derivative w.r.t. var_1 at (params_sr - h_2) with
                # the central method. This is not the right f_x0, but the real one
                # isn't needed for the central method.
                f_minus = central(
                    func, None, params_l, var_1, h_1, *func_args, **func_kwargs
                )
                f_diff = (f_plus - f_minus) / (2.0 * h_1 * h_2)
                hess.loc[var_1, var_2] = f_diff
                hess.loc[var_2, var_1] = f_diff
        return hess
