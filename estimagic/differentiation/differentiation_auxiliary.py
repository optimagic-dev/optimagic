"""Auxiliary functions for a one step differentiation."""


def central(internal_func, f_x0, params, i, h):
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
    params_r[i] += h
    params_l = params.copy()
    params_l[i] -= h
    central_diff = internal_func(params_r) - internal_func(params_l)
    return central_diff / 2.0


def forward(internal_func, f_x0, params, i, h):
    """Calculate the forward difference.

    Args:
        internal_func (func): A function wrapping the original function, s.t. it is
        only dependent on the variable to differentiate
        f_x0 (func output): The function value at params_sr, func_args and func_kwargs.
        params (DataFrame): see :ref:`parmas_df`
        i (int): The variable of which the finite difference is calculated.
        h (float): The infinitesimal step.

    Returns:
        func output: A float or a series, depending on the output type of func

    """
    params = params.copy()
    params[i] += h
    return internal_func(params) - f_x0


def backward(internal_func, f_x0, params, i, h):
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
    params[i] -= h
    return f_x0 - internal_func(params)
