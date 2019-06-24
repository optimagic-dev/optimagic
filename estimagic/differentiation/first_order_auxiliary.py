"""
This module contains auxiliary functions for a one step differentiation.
"""


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
