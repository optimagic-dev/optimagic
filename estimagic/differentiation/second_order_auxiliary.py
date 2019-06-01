"""
This module contains functions for the calculation of a hessian matrix. The functions
call two times the one step differentiation functions.
"""

import numpy as np

from estimagic.differentiation.first_order_auxiliary import forward
from estimagic.differentiation.first_order_auxiliary import backward
from estimagic.differentiation.first_order_auxiliary import central
from estimagic.differentiation.first_order_auxiliary import richardson


def central_hess(
    f, f_x0, params_sr, var_1, var_2, h_1, h_2, method, *func_args, **func_kwargs
):
    params_r = params_sr.copy()
    params_r[var_2] += h_2
    # Calculate the first derivative w.r.t. var_1 at (params_sr + h_2) with the
    # central method. This is not the right f_x0, but the real one isn't needed for
    # the central method.
    f_plus = central(f, f_x0, params_r, var_1, h_1, *func_args, **func_kwargs)
    params_l = params_sr.copy()
    params_l[var_2] -= h_2
    # Calculate the first derivative w.r.t. var_1 at (params_sr - h_2) with the
    # central method. This is not the right f_x0, but the real one isn't needed for
    # the central method.
    f_minus = central(f, f_x0, params_l, var_1, h_1, *func_args, **func_kwargs)
    return (f_plus - f_minus) / 2.0


def forward_hess(
    f, f_x0, params_sr, var_1, var_2, h_1, h_2, method, *func_args, **func_kwargs
):
    f_x0_var_1 = forward(f, f_x0, params_sr, var_1, h_1, *func_args, **func_kwargs)
    params = params_sr.copy()
    params[var_2] += h_2
    # Calculate the first derivative w.r.t. var_1 at (params_sr + h_2) with the
    # forward method. Therefore we need f(x_0 + var_2).
    f_x0_var_2 = f(params, *func_args, **func_kwargs)
    f_x0_var_1_var_2 = forward(
        f, f_x0_var_2, params, var_1, h_1, *func_args, **func_kwargs
    )
    return f_x0_var_1_var_2 - f_x0_var_1


def backward_hess(
    f, f_x0, params_sr, var_1, var_2, h_1, h_2, method, *func_args, **func_kwargs
):
    f_x0_var_1 = backward(f, f_x0, params_sr, var_1, h_1, *func_args, **func_kwargs)
    params = params_sr.copy()
    params[var_2] -= h_2
    # Calculate the first derivative w.r.t. var_1 at (x_0 - h_2) with the
    # forward method. Therefore we need f(x_0 - var_2).
    f_x0_var_2 = f(params, *func_args, **func_kwargs)
    f_x0_var_1_var_2 = backward(
        f, f_x0_var_2, params, var_1, h_1, *func_args, **func_kwargs
    )
    return f_x0_var_1 - f_x0_var_1_var_2


def hess_richardson(
    func, f_x0, params_sr, var_1, var_2, h_1, h_2, method, *func_args, **func_kwargs
):
    if method == "forward":
        f = forward
    elif method == "backward":
        f = backward
    elif method == "central":
        f = central
    else:
        raise ValueError("The given method was not found.")
    f_i_x0 = richardson(
        f, func, f_x0, params_sr, var_1, h_1, method, *func_args, **func_kwargs
    )
    order = 3
    eval_points = [1, 2, 4]
    pol_plus = np.zeros(order)
    pol_minus = np.zeros(order)
    for index, i in enumerate(eval_points):
        params_plus = params_sr.copy()
        params_plus[var_2] += i * h_2
        f_x0_plus = func(params_plus, *func_args, **func_kwargs)
        params_minus = params_sr.copy()
        params_minus[var_2] -= i * h_2
        f_x0_minus = func(params_minus, *func_args, **func_kwargs)
        pol_plus[index] = richardson(
            f,
            func,
            f_x0_plus,
            params_plus,
            var_1,
            h_1,
            method,
            *func_args,
            **func_kwargs
        )
        pol_minus[index] = richardson(
            f,
            func,
            f_x0_minus,
            params_minus,
            var_1,
            h_1,
            method,
            *func_args,
            **func_kwargs
        )
    if method == "central":
        pol_factors = (pol_plus - pol_minus) / 2
        f_diff = (pol_factors[2] / 4 - 10 * pol_factors[1] + 64 * pol_factors[0]) / 45
    elif method == "forward":
        pol_factors = pol_plus - f_i_x0
        f_diff = (pol_factors[2] / 4 - 3 * pol_factors[1] + 8 * pol_factors[0]) / 3
    else:
        pol_factors = f_i_x0 - pol_minus
        f_diff = (pol_factors[2] / 4 - 3 * pol_factors[1] + 8 * pol_factors[0]) / 3
    return f_diff
