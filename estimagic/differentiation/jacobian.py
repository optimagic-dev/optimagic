import numdifftools as nd
import pandas as pd


def jacobian(func, params_sr, method="central", func_args=None, func_kwargs=None):
    """
    Calculate the jacobian of *func*.

    Args:
        func (function): A function that maps params_sr into a numpy array
                        or pandas Series.

        params_sr (Series): see :ref:`parmas_df`
        func_args (list): additional positional arguments for func.
        func_kwargs (dict): additional positional arguments for func.
        extrapolant (string): A variable indicating the use of a extrapolation method.
        method (string): The method for the computation of the derivative. Default is
                         central as it gives the highest accuracy.

    Returns:
        DataFrame: If func returns a Series, the index is the index of this Series or
        the index is 0,1,2... if func returns a numpy array. The columns are the
        index of params_sr.

    """
    # set default arguments
    func_args = [] if func_args is None else func_args
    func_kwargs = {} if func_kwargs is None else func_kwargs
    f_x0 = func(params_sr, *func_args, **func_kwargs)
    jac_np = nd.Jacobian(func, method=method)(params_sr, *func_args, **func_kwargs)
    if isinstance(f_x0, pd.Series):
        jac = pd.DataFrame(index=f_x0.index, columns=params_sr.index, data=jac_np)
    else:
        jac = pd.DataFrame(columns=params_sr.index, data=jac_np)
    return jac
