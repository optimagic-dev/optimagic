import pandas as pd
import numpy as np
from estimagic.differentiation.gradient import gradient


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
    if method == "forward":
        f = forward_hess
    elif method == "backward":
        f = backward_hess
    elif method == 'central':
        f = central_hess
    else:
        raise ValueError('The given method was not found.')
    for var_1 in hess.index:
        h_1 = (1 + abs(params_sr[var_1])) * np.cbrt(np.finfo(float).eps)
        for var_2 in hess.columns:
            h_2 = (1 + abs(params_sr[var_2])) * np.cbrt(np.finfo(float).eps)
            if extrapolant == 'richardson':
                pol = []
                for i in [1, 2, 4]:
                    pol += [f(func, f_x0, params_sr, var_1, var_2, h_1 * i, h_2 * i,
                              *func_args,
                              **func_kwargs)]
                if method == 'central':
                    f_diff = (pol[2] - 20 * pol[1] + 64 * pol[0]) / 45
                else:
                    f_diff = (pol - 6 * pol + 8 * pol) / 3
            else:
                f_diff = f(func, f_x0, params_sr, var_1, var_2, h_1, h_2, *func_args,
                           **func_kwargs)
            hess.loc[var_1, var_2] = f_diff
    return hess


def central_hess(f, f_x0, params_sr, var1, var2, h_1, h_2, *func_args, **func_kwargs):
    params_r_r = params_sr.copy()
    params_r_r[var1] = params_r_r[var1] + h_1
    params_r_r[var2] = params_r_r[var2] + h_2
    params_r_l = params_sr.copy()
    params_r_l[var1] = params_r_l[var1] + h_1
    params_r_l[var2] = params_r_l[var2] - h_2
    params_l_r = params_sr.copy()
    params_l_r[var1] = params_l_r[var1] - h_1
    params_l_r[var2] = params_l_r[var2] + h_2
    params_l_l = params_sr.copy()
    params_l_l[var1] = params_l_l[var1] - h_1
    params_l_l[var2] = params_l_l[var2] - h_2
    cent_r_r = f(params_r_r, *func_args, **func_kwargs)
    cent_r_l = f(params_r_l, *func_args, **func_kwargs)
    cent_l_r = f(params_l_r, *func_args, **func_kwargs)
    cent_l_l = f(params_l_l, *func_args, **func_kwargs)
    return (cent_r_r - cent_r_l - cent_l_r + cent_l_l) / (4.0 * (h_1 * h_2))


def forward_hess(f, f_x0, params_sr, var1, var2, h_1, h_2, *func_args, **func_kwargs):
    params_r_r = params_sr.copy()
    params_r_r[var1] += h_1
    params_r_r[var2] += h_2
    params_r_1 = params_sr.copy()
    params_r_1[var1] += h_1
    params_r_2 = params_sr.copy()
    params_r_2[var2] += h_2
    forward_r_r = f(params_r_r, *func_args, **func_kwargs)
    forward_r_1 = f(params_r_1, *func_args, **func_kwargs)
    forward_r_2 = f(params_r_2, *func_args, **func_kwargs)
    return (forward_r_r - forward_r_1 - forward_r_2 + f_x0) / (h_1 *h_2)


def backward_hess(f, f_x0, params_sr, var, h, method, extrapolant, *func_args, **func_kwargs):
    params = params_sr.copy()
    params[var] = params_sr[var] - h
    return (f_x0 - gradient(f, params, method, extrapolant, *func_args, **func_kwargs)) / h
