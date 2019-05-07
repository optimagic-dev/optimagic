def jacobian(func, params_sr, func_args=None, func_kwargs=None):
    """Calculate the jacobian of *func*.

    Args:
        func (function): A function that maps params_sr into a numpy array
            or pandas Series.
        params_sr (Series): see :ref:`parmas_df`
        func_args (list): additional positional arguments for func.
        func_kwargs (dict): additional positional arguments for func.

    Returns:
        jacobian (pd.DataFrame): The index is the index of the output of func
            or 0, 1, ... if func returns a numpy array. The columns are the
            index of params_sr.

    """
