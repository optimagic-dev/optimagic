import numdifftools as nd
import pandas as pd


def hessian(func, params_sr, method="central", func_args=None, func_kwargs=None):
    """
    Calculate the hessian of *func*.

    Args:
        func (function): A function that maps params_sr into a float.
        params_sr (Series): see :ref:`parmas_df`
        func_args (list): additional positional arguments for func.
        func_kwargs (dict): additional positional arguments for func.
        extrapolant (string): A variable indicating the use of a extrapolation method.
        method (string): The method for the computation of the derivative. Default is
                         central as it gives the highest accuracy.

    Returns:
        DataFrame: The index and columns are the index of params_sr.

    """
    if method != "central":
        raise ValueError("The given method is not supported")
    # set default arguments
    func_args = [] if func_args is None else func_args
    func_kwargs = {} if func_kwargs is None else func_kwargs
    if method == "central":
        order = 2
    else:
        order = 1
    hess_np = nd.Hessian(func, method=method, order=order)(
        params_sr, *func_args, **func_kwargs
    )
    return pd.DataFrame(data=hess_np, index=params_sr.index, columns=params_sr.index)
