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
        params (DataFrame): see :ref:`parmas_df`
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

    if extrapolation:
        internal_func = _create_internal_func(func, params, func_args, func_kwargs)
        grad_np = nd.Gradient(internal_func, method=method)(params["value"].to_numpy())
        grad = pd.Series(data=grad_np, index=params.index, name="gradient")
    else:
        grad = pd.Series(index=params.index, name="gradient")
        f_x0 = func(params, *func_args, **func_kwargs)
        finite_diff = getattr(aux, method)
        for var in params.index:
            h = (1 + abs(params.loc[var, "value"])) * np.sqrt(np.finfo(float).eps)
            grad[var] = (
                finite_diff(func, f_x0, params, var, h, *func_args, **func_kwargs) / h
            )
    return grad


def jacobian(
    func, params, method="central", extrapolation=True, func_args=None, func_kwargs=None
):
    """
    Calculate the jacobian of *func*.

    Args:
        func (function): A function that maps params_sr into a numpy array
                        or pandas Series.

        params (DataFrame): see :ref:`parmas_df`
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

    if extrapolation:
        internal_func = _create_internal_func(func, params, func_args, func_kwargs)
        jac_np = nd.Jacobian(internal_func, method=method)(params["value"].to_numpy())
        if isinstance(f_x0, pd.Series):
            jac = pd.DataFrame(index=f_x0.index, columns=params.index, data=jac_np)
        else:
            jac = pd.DataFrame(columns=params.index, data=jac_np)
    else:
        finite_diff = getattr(aux, method)
        if isinstance(f_x0, pd.Series):
            jac = pd.DataFrame(index=f_x0.index, columns=params.index)
        else:
            jac = pd.DataFrame(columns=params.index)
        for var in jac.columns:
            # The rule of thumb for the stepsize is implemented
            h = (1 + abs(params.loc[var, "value"])) * np.sqrt(np.finfo(float).eps)
            f_diff = finite_diff(func, f_x0, params, var, h, *func_args, **func_kwargs)
            jac[var] = f_diff / h
    return jac


def hessian(
    func, params, method="central", extrapolation=True, func_args=None, func_kwargs=None
):
    """
    Calculate the hessian of *func*.

    Args:
        func (function): A function that maps params_sr into a float.
        params (DataFrame): see :ref:`parmas_df`
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

    if extrapolation:
        internal_func = _create_internal_func(func, params, func_args, func_kwargs)
        hess_np = nd.Hessian(internal_func, method=method)(params["value"].to_numpy())
        hess = pd.DataFrame(data=hess_np, index=params.index, columns=params.index)
    else:
        finite_diff = getattr(aux, method)
        hess = pd.DataFrame(index=params.index, columns=params.index, dtype=float)
        for var_1 in params.index:
            h_1 = (1.0 + abs(params.loc[var_1, "value"])) * np.cbrt(np.finfo(float).eps)
            for var_2 in params.index:
                h_2 = (1.0 + abs(params.loc[var_2, "value"])) * np.cbrt(
                    np.finfo(float).eps
                )
                params_r = params.copy()
                params_r.loc[var_2, "value"] += h_2
                # Calculate the first derivative w.r.t. var_1 at (params_sr + h_2) with
                # the central method. This is not the right f_x0, but the real one
                # isn't needed for the central method.
                f_plus = finite_diff(
                    func, None, params_r, var_1, h_1, *func_args, **func_kwargs
                )
                params_l = params.copy()
                params_l.loc[var_2, "value"] -= h_2
                # Calculate the first derivative w.r.t. var_1 at (params_sr - h_2) with
                # the central method. This is not the right f_x0, but the real one
                # isn't needed for the central method.
                f_minus = finite_diff(
                    func, None, params_l, var_1, h_1, *func_args, **func_kwargs
                )
                f_diff = (f_plus - f_minus) / (2.0 * h_1 * h_2)
                hess.loc[var_1, var_2] = f_diff
                hess.loc[var_2, var_1] = f_diff
    return hess


def _create_internal_func(func, params, func_args, func_kwargs):
    def internal_func(x):
        p = params.copy(deep=True)
        p["value"] = x
        func_value = func(p, *func_args, **func_kwargs)
        if isinstance(func_value, (pd.DataFrame, pd.Series)):
            func_value = func_value.to_numpy()
        return func_value

    return internal_func
