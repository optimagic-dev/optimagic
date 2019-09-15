import numdifftools as nd
import numpy as np
import pandas as pd

from estimagic.differentiation import differentiation_auxiliary as aux


def gradient(
    func, params, method="central", extrapolation=True, func_args=None, func_kwargs=None
):
    """
    Calculate the gradient of *func*.

    Args:
        func (function): A function that maps params_sr into a float.
        params (DataFrame): see :ref:`params`
        method (str): The method for the computation of the derivative. Default is
            central as it gives the highest accuracy.
        extrapolation (bool): This variable allows to specify the use of the
            richardson extrapolation.
        func_args (list): additional positional arguments for func.
        func_kwargs (dict): additional positional arguments for func.

    Returns:
        Series: The index is the index of params_sr.

    """
    if method not in ["central", "forward", "backward"]:
        raise ValueError("Method has to be in ['central', 'forward', 'backward']")

    func_args = [] if func_args is None else func_args
    func_kwargs = {} if func_kwargs is None else func_kwargs

    internal_func = _create_internal_func(func, params, func_args, func_kwargs)
    params_value = params["value"].to_numpy()

    if extrapolation:
        grad_np = nd.Gradient(internal_func, method=method)(params_value)
    else:
        grad_np = _no_extrapolation_gradient(internal_func, params_value, method)
    return pd.Series(data=grad_np, index=params.index, name="gradient")


def _no_extrapolation_gradient(internal_func, params_value, method):
    grad = np.empty_like(params_value)
    f_x0 = internal_func(params_value)
    finite_diff = getattr(aux, method)
    for i, val in enumerate(params_value):
        h = (1 + abs(val)) * np.sqrt(np.finfo(float).eps)
        grad[i] = finite_diff(internal_func, f_x0, params_value, i, h) / h
    return grad


def jacobian(
    func, params, method="central", extrapolation=True, func_args=None, func_kwargs=None
):
    """
    Calculate the jacobian of *func*.

    Args:
        func (function): A function that maps params_sr into a numpy array
                        or pandas Series.

        params (DataFrame): see :ref:`params`
        method (string): The method for the computation of the derivative. Default is
                         central as it gives the highest accuracy.
        extrapolation (bool): This variable allows to specify the use of the
                                richardson extrapolation.
        func_args (list): additional positional arguments for func.
        func_kwargs (dict): additional positional arguments for func.

    Returns:
        DataFrame: If func returns a Series, the index is the index of this Series or
        the index is 0,1,2... if func returns a numpy array. The columns are the
        index of params_sr.

    """
    if method not in ["central", "forward", "backward"]:
        raise ValueError("Method has to be in ['central', 'forward', 'backward']")

    func_args = [] if func_args is None else func_args
    func_kwargs = {} if func_kwargs is None else func_kwargs

    f_x0 = func(params, *func_args, **func_kwargs)

    internal_func = _create_internal_func(func, params, func_args, func_kwargs)
    params_value = params["value"].to_numpy()

    if extrapolation:
        jac_np = nd.Jacobian(internal_func, method=method)(params_value)
    else:
        jac_np = _no_extrapolation_jacobian(internal_func, params_value, method)

    if isinstance(f_x0, pd.Series):
        return pd.DataFrame(index=f_x0.index, columns=params.index, data=jac_np)
    else:
        return pd.DataFrame(columns=params.index, data=jac_np)


def _no_extrapolation_jacobian(internal_func, params_value, method):
    f_x0_np = internal_func(params_value)
    jac = np.empty((len(f_x0_np), len(params_value)))
    finite_diff = getattr(aux, method)
    for i, val in enumerate(params_value):
        # The rule of thumb for the stepsize is implemented
        h = (1 + abs(val)) * np.sqrt(np.finfo(float).eps)
        f_diff = finite_diff(internal_func, f_x0_np, params_value, i, h)
        jac[:, i] = f_diff / h
    return jac


def hessian(
    func, params, method="central", extrapolation=True, func_args=None, func_kwargs=None
):
    """
    Calculate the hessian of *func*.

    Args:
        func (function): A function that maps params_sr into a float.
        params (DataFrame): see :ref:`params`
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
        raise ValueError("Only the method 'central' is supported.")

    func_args = [] if func_args is None else func_args
    func_kwargs = {} if func_kwargs is None else func_kwargs

    internal_func = _create_internal_func(func, params, func_args, func_kwargs)
    params_value = params["value"].to_numpy()

    if extrapolation:
        hess_np = nd.Hessian(internal_func, method=method)(params_value)
    else:
        hess_np = _no_extrapolation_hessian(internal_func, params_value, method)
    return pd.DataFrame(data=hess_np, index=params.index, columns=params.index)


def _no_extrapolation_hessian(internal_func, params_value, method):
    finite_diff = getattr(aux, method)
    hess = np.empty((len(params_value), len(params_value)))
    for i, val_1 in enumerate(params_value):
        h_1 = (1.0 + abs(val_1)) * np.cbrt(np.finfo(float).eps)
        for j, val_2 in enumerate(params_value):
            h_2 = (1.0 + abs(val_2)) * np.cbrt(np.finfo(float).eps)
            params_r = params_value.copy()
            params_r[j] += h_2
            # Calculate the first derivative w.r.t. var_1 at (params_sr + h_2) with
            # the central method. This is not the right f_x0, but the real one
            # isn't needed for the central method.
            f_plus = finite_diff(internal_func, None, params_r, i, h_1)
            params_l = params_value.copy()
            params_l[j] -= h_2
            # Calculate the first derivative w.r.t. var_1 at (params_sr - h_2) with
            # the central method. This is not the right f_x0, but the real one
            # isn't needed for the central method.
            f_minus = finite_diff(internal_func, None, params_l, i, h_1)
            f_diff = (f_plus - f_minus) / (2.0 * h_1 * h_2)
            hess[i, j] = f_diff
            hess[i, j] = f_diff
    return hess


def _create_internal_func(func, params, func_args, func_kwargs):
    def internal_func(x):
        p = params.copy(deep=True)
        p["value"] = x
        # ERROR_0: The gradient function wraps the internal_criterion here. Note that, p
        # is a params DataFrame whereas internal criterion expects a numpy vector. Go to
        # create_internal_criterion.
        func_value = func(p, *func_args, **func_kwargs)
        if isinstance(func_value, (pd.DataFrame, pd.Series)):
            func_value = func_value.to_numpy()
        return func_value

    return internal_func
