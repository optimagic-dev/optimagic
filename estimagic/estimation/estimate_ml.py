def estimate_ml(
    loglike,
    params,
    minimize_options,
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

    """
