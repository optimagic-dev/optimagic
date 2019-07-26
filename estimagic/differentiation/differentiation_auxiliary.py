"""Auxiliary functions for a one step differentiation."""


def central(func, f_x0, params, var, h, *func_args, **func_kwargs):
    """Calculate the central difference.

    Args:
        func (func): A function that maps params_sr into a float.
        f_x0 (func output): The function value at params_sr, func_args and func_kwargs.
        params (DataFrame): see :ref:`parmas_df`
        var (string): The variable of which the finite difference is calculated.
        h (float): The infinitesimal step.
        func_args (list): additional positional arguments for func.
        func_kwargs (dict): additional positional arguments for func.

    Returns:
        func output: A float or a series, depending on the output type of func

    """
    params_r = params.copy()
    params_r.loc[var, "value"] += h
    params_l = params.copy()
    params_l.loc[var, "value"] -= h
    central_diff = func(params_r, *func_args, **func_kwargs) - func(
        params_l, *func_args, **func_kwargs
    )
    return central_diff / 2.0


def forward(func, f_x0, params, var, h, *func_args, **func_kwargs):
    """Calculate the forward difference.

    Args:
        func (func): A function that maps params_sr into a float.
        f_x0 (func output): The function value at params_sr, func_args and func_kwargs.
        params (DataFrame): see :ref:`parmas_df`
        var (string): The variable of which the finite difference is calculated.
        h (float): The infinitesimal step.
        func_args (list): additional positional arguments for func.
        func_kwargs (dict): additional positional arguments for func.

    Returns:
        func output: A float or a series, depending on the output type of func

    """
    params = params.copy()
    params.loc[var, "value"] += h
    return func(params, *func_args, **func_kwargs) - f_x0


def backward(func, f_x0, params, var, h, *func_args, **func_kwargs):
    """Calculate the backward difference.

    Args:
        func (func): A function that maps params_sr into a float.
        f_x0 (func output): The function value at params, func_args and func_kwargs.
        params_sr (Series): see :ref:`parmas_df`
        var (string): The variable of which the finite difference is calculated.
        h (float): The infinitesimal step.
        func_args (list): additional positional arguments for func.
        func_kwargs (dict): additional positional arguments for func.

    Returns:
        func output: A float or a series, depending on the output type of func
    """
    params = params.copy()
    params.loc[var, "value"] -= h
    return f_x0 - func(params, *func_args, **func_kwargs)
