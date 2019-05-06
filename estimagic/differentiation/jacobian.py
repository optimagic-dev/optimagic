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


    Notes:
        - Use numdifftools to do all the hard work. Check if it already works with a
            params_sr, otherwise write a wrapper.
        - The documentation on numdifftools is a good introduction to numerical
            derivatives:
            https://numdifftools.readthedocs.io/en/latest/src/numerical/main.html
            you should read that completely!
        - we will need additional arguments to configure the differentiation but I did
            not add them to the interface yet. For example we want to specify if we
            have forward, central or backward differences and if we want to use
            richardson extrapolations. I'm not sure if richardson extrapolations
            can only be used with central differences or also with the other ones.
            If it can be used for all we should have one argument called method (which
            differences to use) and one called richardson_extrapolation which can be
            True or False. Otherwise we should just have one called method that takes
            values [richardson-extrapolation, 'central-differences', ...]
        - The default of all options should be what gives us most precision.
        - We don't want arguments for things nobody ever wants to change. For example,
            if there is a clear rule for choosing the optimal step-size for the finite
            differences, we don't need to make this configurable.
        - We don't want to support anything that has a complex step in it because it is
            sometimes not clear what happens when we call numba or Fortran functions
            with complex numbers

    """
