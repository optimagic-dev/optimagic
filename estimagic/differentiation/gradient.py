def gradient(func, params_sr, func_args=None, func_kwargs=None):
    """Calculate the gradient of *func*.

    Args:
        func (function): A function that maps params_sr into a float.
        params_sr (Series): see :ref:`parmas_df`
        func_args (list): additional positional arguments for func.
        func_kwargs (dict): additional positional arguments for func.

    Returns:
        gradient (Series): The index is the index of params_sr.

    """
