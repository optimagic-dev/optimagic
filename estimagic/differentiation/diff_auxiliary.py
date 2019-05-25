import numpy as np


def central(func, f_x0, params_sr, var, h, *func_args, **func_kwargs):
    params_r = params_sr.copy()
    params_r[var] += h
    params_l = params_sr.copy()
    params_l[var] -= h
    central_diff = func(params_r, *func_args, **func_kwargs) - func(
        params_l, *func_args, **func_kwargs
    )
    return central_diff / 2.0


def forward(func, f_x0, params_sr, var, h, *func_args, **func_kwargs):
    params = params_sr.copy()
    params[var] += h
    return func(params, *func_args, **func_kwargs) - f_x0


def backward(func, f_x0, params_sr, var, h, *func_args, **func_kwargs):
    params = params_sr.copy()
    params[var] -= h
    return f_x0 - func(params, *func_args, **func_kwargs)


def richardson(f, func, f_x0, params_sr, var, h, method, *func_args, **func_kwargs):
    pol = []
    for i in [1, 2, 4]:
        pol += [f(func, f_x0, params_sr, var, h * i, *func_args, **func_kwargs)]
    if method == "central":
        f_diff = (pol[2] / 4 - 10 * pol[1] + 64 * pol[0]) / 45
    else:
        f_diff = (pol[2] / 4 - 3 * pol[1] + 8 * pol[0]) / 3
    return f_diff


def central_hess(f, f_x0, params_sr, var_1, var_2, h_1, h_2, method, *func_args,
                 **func_kwargs):
    params_r = params_sr.copy()
    params_r[var_2] += h_2
    f_plus = central(f, f_x0, params_r, var_1, h_1, *func_args, **func_kwargs)
    params_l = params_sr.copy()
    params_l[var_2] -= h_2
    f_minus = central(f, f_x0, params_l, var_1, h_1, *func_args, **func_kwargs)
    return (f_plus - f_minus) / 2.0


def forward_hess(f, f_x0, params_sr, var_1, var_2, h_1, h_2, method, *func_args,
                 **func_kwargs):
    f_x0_i = forward(f, f_x0, params_sr, var_1, h_1, *func_args, **func_kwargs)
    params = params_sr.copy()
    params[var_2] += h_2
    f_x0_j = f(params, *func_args, **func_kwargs)
    f_i_j = forward(f, f_x0_j, params, var_1, h_1, *func_args, **func_kwargs)
    return f_i_j - f_x0_i


def backward_hess(f, f_x0, params_sr, var_1, var_2, h_1, h_2, method, *func_args,
                  **func_kwargs):
    f_x0_i = backward(f, f_x0, params_sr, var_1, h_1, *func_args, **func_kwargs)
    params = params_sr.copy()
    params[var_2] -= h_2
    f_x0_j = f(params, *func_args, **func_kwargs)
    f_i_j = backward(f, f_x0_j, params, var_1, h_1, *func_args, **func_kwargs)
    return f_x0_i - f_i_j


def hess_richardson(
    func, f_x0, params_sr, var_1, var_2, h_1, h_2, method, *func_args, **func_kwargs
):
    if method == "forward":
        f = forward
    elif method == "backward":
        f = backward
    elif method == 'central':
        f = central
    f_i_x0 = richardson(f, func, f_x0, params_sr, var_1, h_1, method, *func_args,
                        **func_kwargs)
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
        pol_plus[index] = richardson(f, func, f_x0_plus, params_plus, var_1, h_1,
                                     method, *func_args, **func_kwargs)
        pol_minus[index] = richardson(f, func, f_x0_minus, params_minus, var_1, h_1,
                                      method, *func_args, **func_kwargs)
    if method == "central":
        pol_factors = (pol_plus - pol_minus) / 2
        f_diff = (pol_factors[2] / 4 - 10 * pol_factors[1] + 64 * pol_factors[0]) / 45
    elif method == 'forward':
        pol_factors = (pol_plus - f_i_x0)
        f_diff = (pol_factors[2] / 4 - 3 * pol_factors[1] + 8 * pol_factors[0]) / 3
    elif method == 'backward':
        pol_factors = (f_i_x0 - pol_minus)
        f_diff = (pol_factors[2] / 4 - 3 * pol_factors[1] + 8 * pol_factors[0]) / 3
    return f_diff
