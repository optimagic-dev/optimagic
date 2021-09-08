"""Do a method of simlated moments estimation."""


def estimate_msm(
    simulate_moments,
    empirical_moments,
    moments_cov,
    params,
    *,
    minimize_options=None,
    simulate_moments_kwargs=None,
    weights="diagonal",
    numdiff_options=None,
    jacobian=None,
    jacobian_kwargs=None,
    simulate_moments_and_jacobian=None,
    simulate_moments_and_jacobian_kwargs=None,
    ci_level=0.95,
    n_samples=10_000,
):
    """Do a method of simulated moments or indirect inference estimation.

    This is a high level interface for our lower level functions for minimization,
    numerical differentiation, inference and sensitivity analysis. It does the full
    workflow for MSM or indirect inference estimation with just one function call.

    While we have good defaults, you can still configure each aspect of each steps
    vial the optional arguments of this functions. If you find it easier to do the
    "difficult" steps (mainly minimization and calculating numerical derivatives
    of a potentially noisy function) separately, you can do so and just provide those
    results as ``params`` and ``jacobian``.

    The docstring is aspirational and not all options are supported yet.

    Args:
        simulate_moments (callable): Function that takes params and potentially other
            keyworrd arguments and returns simulated moments as a pandas Series.
            Alternatively, the function can return a dict with any number of entries
            as long as one of those entries is "simulated_moments".
        empirical_moments (pandas.Series): A pandas series with the empirical
            equivalents of the simulated moments.
        moments_cov (pandas.DataFrame): A quadratic pandas DataFrame with the covariance
            matrix of the empirical moments. This is typically calculated with
            our ``get_moments_cov`` function. The index and columns need to be the same
            as the index of ``empirical_moments``.
        params (pandas.DataFrame): Start params for the optimization. See :ref:`params`
            for details.
        simulate_moments_kwargs (dict): Additional keyword arguments for
            ``simulate_moments``.
        weights (str or pandas.DataFrame): Either a DataFrame with a positive
            semi-definite weighting matrix or a string that specifies one of the
            pre-implemented weighting matrices: "diagonal" (default), "identity" or
            "optimal". Note that "optimal" refers to the asymptotically optimal
            weighting matrix and is often not a good choice due to large finite sample
            bias.
        minimize_options (dict or False): Keyword arguments that govern the numerical
            optimization. Valid entries are all arguments of
            :func:`~estimagic.optimization.optimize.minimize` except for criterion,
            derivative, criterion_and_derivative and params. If you pass False as
            minimize_options you signal that ``params`` are already the optimal
            parameters and no numerical optimization is needed. In contrast to
            ``minimize`` we have a default algorithm that depends on which libraries
            you have installed. The algorithms we try are ``tao_pounders``,
            ``nag_dfols``, ``nag_pybobyqa``, ``nlopt_bobyqa``, ``scipy_neldermead``.
            We strongly recommend that you either install ``petsc4py`` or
            ``nag_dfols``.
        numdiff_options (dict): Keyword arguments for the calculation of numerical
            derivatives for the calculation of standard errors. See
            :ref:`first_derivative` for details.
        jacobian (callable or pandas.DataFrame): A function that take ``params`` and
            potentially other keyword arguments and returns the jacobian of
            simulate_moments with respect to the params. Alternatively you can pass
            a pandas.DataFrame with the jacobian at the optimal parameters. This is
            only possible if you pass ``minimize_options=False``.
        jacobian_kwargs (dict): Additional keyword arguments for jacobian.
        simulate_moments_and_jacobian (callable): A function that takes params and
            potentially other keyword arguments and returns a tuple with simulated
            moments and the jacobian of simulated moments with respect to params.
        simulate_moments_and_jacobian_kwargs (dict): Additional keyword arguments for
            simulate_moments_and_jacobian.
        ci_level (float): Confidence level for the calculation of confidence intervals.
        n_samples (int): Number of samples used to transform the covariance matrix of
            the internal parameter vector into the covariance matrix of the external
            parameters. For background information about internal and external params
            see :ref:`implementation_of_constraints`. This is only used if you have
            constraints in the ``minimize_options``

        Returns:
            dict: The estimated parameters, standard errors and sensitivity measures.

    """
    pass


def get_msm_optimization_functions(
    simulate_moments,
    weights,
    simulate_moments_kwargs=None,
    jacobian=None,
    jacobian_kwargs=None,
    simulate_moments_and_jacobian=None,
    simulate_moments_and_jacobian_kwargs=None,
):
    """Construct criterion functions and their derivatives for msm estimation.

    Args:


    Returns:
        tuple: A tuple of length three containing criterion, derivative and
            criterion_and_derivative. The first entry is always a function, the other
            values are None if not enough inputs were provided. All resulting functions
            take params as only argument.

    """
    pass
