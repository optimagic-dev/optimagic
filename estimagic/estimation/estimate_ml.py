def estimate_ml(
    loglike,
    params,
    maximize_options,
    *,
    constraints=None,
    logging=False,
    log_options=None,
    loglike_kwargs=None,
    derivative=None,
    derivative_kwargs=None,
    loglike_and_derivative=None,
    loglike_and_derivative_kwargs=None,
    numdiff_options=None,
    jacobian=None,
    jacobian_kwargs=None,
    hessian=None,
    hessian_kwargs=None,
    ci_level=0.95,
    n_samples=10_000,
    bounds_handling="raise",
    design_info=None,
):
    """Do a maximum likelihood (ml) estimation.

    This is a high level interface for our lower level functions for maximization,
    numerical differentiation and inference. It does the full workflow for maximum
    likelihood estimation with just one function call.

    While we have good defaults, you can still configure each aspect of each steps
    vial the optional arguments of this functions. If you find it easier to do the
    "difficult" steps (mainly maximization and calculating numerical derivatives
    of a potentially noisy function) separately, you can do so and just provide those
    results as ``params``, ``jacobian`` and ``hessian``.

    The docstring is aspirational and not all options are supported yet.

    Args:
        loglike (callable): Likelihood function that takes a params DataFrame (and
            potentially other keyword arguments) and returns a dictionary that has at
            least the entries "value" (a scalar float) and "contributions" (a 1d numpy
            array or pandas Series) with the log likelihood contribution per individual.
        params (pd.DataFrame): DataFrame where the "value" column contains estimated
            parameters of a likelihood model. See :ref:`params` for details.
        maximize_options (dict or False): Keyword arguments that govern the numerical
            optimization. Valid entries are all arguments of
            :func:`~estimagic.optimization.optimize.minimize` except for criterion,
            derivative, criterion_and_derivative and params. If you pass False as
            maximize_options you signal that ``params`` are already the optimal
            parameters and no numerical optimization is needed.
        constraints (list): List with constraint dictionaries.
            See .. _link: ../../docs/source/how_to_guides/how_to_use_constraints.ipynb
        logging (pathlib.Path, str or False): Path to sqlite3 file (which typically has
            the file extension ``.db``. If the file does not exist, it will be created.
            The dashboard can only be used when logging is used.
        log_options (dict): Additional keyword arguments to configure the logging.
            - "fast_logging": A boolean that determines if "unsafe" settings are used
            to speed up write processes to the database. This should only be used for
            very short running criterion functions where the main purpose of the log
            is a real-time dashboard and it would not be catastrophic to get a
            corrupted database in case of a sudden system shutdown. If one evaluation
            of the criterion function (and gradient if applicable) takes more than
            100 ms, the logging overhead is negligible.
            - "if_table_exists": (str) One of "extend", "replace", "raise". What to
            do if the tables we want to write to already exist. Default "extend".
            - "if_database_exists": (str): One of "extend", "replace", "raise". What to
            do if the database we want to write to already exists. Default "extend".
        derivative (callable): Function takes params and potentially other keyword
            arguments and calculates the first derivative of loglike. It can either
            return a numpy array or pandas Series/DataFrame with the derivative or
            a dictionary with derivatives of each output of loglike. If loglike
            returns a dict but derivative does not, it is your responsibility to
            make sure that the correct derivative for the numerical optimizers you are
            using is returned.
        derivative_kwargs (dict): Additional keyword arguments for loglike.
        loglike_and_derivative (callable): Return a tuple consisting of the result
            of loglike and the result of derivative. Only use this if you can exploit
            synergies in the calculation of loglike and derivative.
        loglike_and_derivative_kwargs (dict): Additional keyword argumenst for
            loglike_and_derivative.
        numdiff_options (dict): Keyword arguments for the calculation of numerical
            derivatives for the calculation of standard errors. See
            :ref:`first_derivative` for details.
        jacobian (callable or pandas.DataFrame or False): A function that take
            ``params`` and potentially other keyword arguments and returns the jacobian
            of loglike["contributions"] with respect to the params. Alternatively you
            can pass a pandas.DataFrame with the jacobian at the optimal parameters.
            This is only possible if you pass ``maximize_options=False``. Note that you
            only need to pass a jacobian function if you have a closed form jacobian but
            decided not to return it as part of ``derivative`` (e.g. because you use
            a scalar optimizer and can calculate a gradient in a way that is faster
            than calculating and summing te jacobian). If you pass None, a numerical
            jacobian will be calculated. If you pass ``False``, you signal that no
            jacobian should be calculated. Thus no result that requires the jacobian
            will be calculated.
        jacobian_kwargs (dict): Additional keyword arguments for jacobian.
        hessian (callable or pd.DataFrame): A function that takes
            ``params`` and potentially other keyword arguments and returns the hessian
            of loglike["value"] with respect to the params. Alternatively you
            can pass a pandas.DataFrame with the hessian at the optimal parameters.
            This is only possible if you pass ``maximize_options=False``. If you pass
            None, a numerical hessian will be calculated. If you pass ``False``, you
            signal that no jacobian should be calculated. Thus no result that requires
            the jacobian will be calculated.
        hessian_kwargs (dict): Additional keyword arguments for hessian.
        ci_level (float): Confidence level for the calculation of confidence intervals.
        n_samples (int): Number of samples used to transform the covariance matrix of
            the internal parameter vector into the covariance matrix of the external
            parameters. For background information about internal and external params
            see :ref:`implementation_of_constraints`. This is only used if you have
            constraints in the ``minimize_options``
        bounds_handling (str): One of "clip", "raise", "ignore". Determines how bounds
            are handled. If "clip", confidence intervals are clipped at the bounds.
            Standard errors are only adjusted if a sampling step is necessary due to
            additional constraints. If "raise" and any lower or upper bound is binding,
            we raise an error. If "ignore", boundary problems are simply ignored.
        design_info (pandas.DataFrame): DataFrame with one row per observation that
            contains some or all of the variables "psu" (primary sampling unit),
            "stratum" and "fpc" (finite population corrector). See
            :ref:`robust_likelihood_inference` for details.

    Returns:
        dict: The estimated parameters, standard errors and covariance matrix of the
            parameters.

    """
