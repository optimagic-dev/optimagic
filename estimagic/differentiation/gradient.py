import numdifftools as nd
import pandas as pd


def gradient(func, params_sr, method="central", func_args=None, func_kwargs=None):
    """
    Calculate the gradient of *func*.

    Args:
        func (function): A function that maps params_sr into a float.
        params_sr (Series): see :ref:`parmas_df`
        func_args (list): additional positional arguments for func.
        func_kwargs (dict): additional positional arguments for func.
        extrapolant (string): A variable indicating the use of a extrapolation method.
        method (string): The method for the computation of the derivative. Default is
                         central as it gives the highest accuracy.

    Returns:
        Series: The index is the index of params_sr.

    """
    # set default arguments
    func_args = [] if func_args is None else func_args
    func_kwargs = {} if func_kwargs is None else func_kwargs
    grad_np = nd.Gradient(func, method=method)(params_sr, *func_args, **func_kwargs)
    return pd.Series(data=grad_np, index=params_sr.index)
