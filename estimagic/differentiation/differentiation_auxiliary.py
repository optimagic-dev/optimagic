"""Auxiliary functions for a one step differentiation."""


def central(internal_func, f_x0, params_value, i, h):
    """Calculate the central difference.

    Args:
        internal_func (func): A function with a 1d numpy array as it's only argument.
        f_x0 (numpy array): The function value at params_value
        params_value (numpy array): The value column of param_df as numpy array.
        i (int): The variable's index of which the finite difference is calculated.
        h (float): The infinitesimal step.

    Returns:
        func output: A float or a series, depending on the output type of func

    """
    params_value_r = params_value.copy()
    params_value_r[i] += h
    params_value_l = params_value.copy()
    params_value_l[i] -= h
    central_diff = internal_func(params_value_r) - internal_func(params_value_l)
    return central_diff / 2.0


def forward(internal_func, f_x0, params_value, i, h):
    """Calculate the forward difference.

    Args:
        internal_func (func): A function with a 1d numpy array as it's only argument.
        f_x0 (numpy array): The function value at params_value.
        params_value (numpy array): The value column of param_df as numpy array.
        i (int): The variable's index of which the finite difference is calculated.
        h (float): The infinitesimal step.

    Returns:
        func output: A float or a series, depending on the output type of func

    """
    params_value = params_value.copy()
    params_value[i] += h
    return internal_func(params_value) - f_x0


def backward(internal_func, f_x0, params_value, i, h):
    """Calculate the backward difference.

    Args:
        internal_func (func): A function with a 1d numpy array as it's only argument.
        f_x0 (numpy array): The function value at params_value.
        params_value (numpy array): The value column of param_df as numpy array.
        i (int): The variable's index of which the finite difference is calculated.
        h (float): The infinitesimal step.

    Returns:
        func output: A float or a series, depending on the output type of func
    """
    params_value = params_value.copy()
    params_value[i] -= h
    return f_x0 - internal_func(params_value)
