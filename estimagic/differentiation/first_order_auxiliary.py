"""
This module contains auxiliary functions for a one step differentiation.
"""


def central(func, f_x0, params_sr, var, h, *func_args, **func_kwargs):
    """
    This function calculates the central difference.
    Args:
        func (fucntion): A function that maps params_sr into a float.
        f_x0 (func output): The function value at params_sr, func_args and func_kwargs.
        params_sr (Series): see :ref:`parmas_df`
        var (string): The variable of which the finite difference is calculated.
        h (float): The infinitesimal step.
        func_args (list): additional positional arguments for func.
        func_kwargs (dict): additional positional arguments for func.

    Returns:
        func output: A float or a series, depending on the output type of func
    """
    params_r = params_sr.copy()
    params_r[var] += h
    params_l = params_sr.copy()
    params_l[var] -= h
    central_diff = func(params_r, *func_args, **func_kwargs) - func(
        params_l, *func_args, **func_kwargs
    )
    return central_diff / 2.0


def forward(func, f_x0, params_sr, var, h, *func_args, **func_kwargs):
    """
    This function calculates the forward difference.
    Args:
        func (fucntion): A function that maps params_sr into a float.
        f_x0 (func output): The function value at params_sr, func_args and func_kwargs.
        params_sr (Series): see :ref:`parmas_df`
        var (string): The variable of which the finite difference is calculated.
        h (float): The infinitesimal step.
        func_args (list): additional positional arguments for func.
        func_kwargs (dict): additional positional arguments for func.

    Returns:
        func output: A float or a series, depending on the output type of func
    """
    params = params_sr.copy()
    params[var] += h
    return func(params, *func_args, **func_kwargs) - f_x0


def backward(func, f_x0, params_sr, var, h, *func_args, **func_kwargs):
    """
    This function calculates the backward difference.
    Args:
        func (fucntion): A function that maps params_sr into a float.
        f_x0 (func output): The function value at params_sr, func_args and func_kwargs.
        params_sr (Series): see :ref:`parmas_df`
        var (string): The variable of which the finite difference is calculated.
        h (float): The infinitesimal step.
        func_args (list): additional positional arguments for func.
        func_kwargs (dict): additional positional arguments for func.

    Returns:
        func output: A float or a series, depending on the output type of func
    """
    params = params_sr.copy()
    params[var] -= h
    return f_x0 - func(params, *func_args, **func_kwargs)
