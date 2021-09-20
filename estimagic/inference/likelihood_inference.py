import functools

from estimagic.decorators import numpy_interface
from estimagic.differentiation.derivatives import first_derivative
from estimagic.inference.likelihood_covs import cov_jacobian
from estimagic.inference.shared import calculate_inference_quantities
from estimagic.inference.shared import transform_covariance
from estimagic.parameters.process_constraints import process_constraints
from estimagic.parameters.reparametrize import reparametrize_to_internal


def do_likelihood_inference(
    loglike,
    params,
    *,
    cov_type="jacobian",
    loglike_kwargs=None,
    constraints=None,
    numdiff_options=None,
    bounds_handling="raise",
    n_samples=10_000,
    jacobian=None,
    jacobian_kwargs=None,
    hessian=None,
    hessian_kwargs=None,
    design_info=None,
    significance_level=0.05,
):
    """Return standard error, confidence interval and p-value for likelihood estimates.

    Args:
        loglike (callable): Likelihood function that takes a params DataFrame (and
            potentially other keyword arguments) and returns a dictionary that has at
            least the entries "value" (a scalar float) and "contributions" (a 1d numpy
            array or pandas Series) with the log likelihood contribution per individual.
        params (pd.DataFrame): DataFrame where the "value" column contains estimated
            parameters of a likelihood model. See :ref:`params` for details.
        cov_type (str): One of "jacobian", "hessian", "sandwich". "jacobian" is
            sometimes also called outer product of gradients and is the default because
            it is numerically more robust than the others. "hessian" is sometimes also
            called observed fisher information.
        loglike_kwargs (dict): Additional keyword arguments for loglike.
        constraints (list): List with constraint dictionaries.
            See .. _link: ../../docs/source/how_to_guides/how_to_use_constraints.ipynb
        numdiff_options (dict): Keyword arguments for the calculation of numerical
            derivatives. See :ref:`first_derivative` for details.
        bounds_handling (str): One of "clip", "raise", "ignore". Determines how bounds
            are handled. If "clip", confidence intervals are clipped at the bounds.
            Standard errors are only adjusted if a sampling step is necessary due to
            additional constraints. If "raise" and any lower or upper bound is binding,
            we raise an error. If "ignore", boundary problems are simply ignored.
        n_samples (int): Number of samples used to transform the covariance matrix of
            the internal parameter vector into the covariance matrix of the external
            parameters. For background information about internal and external params
            see :ref:`implementation_of_constraints`.
        jacobian (numpy.ndarray, pandas.DataFrame or callable): Pre-calculated jacobian
            of loglike at params or function to calculate it. The function is expected
            to take ``params`` as the first argument but can have additional keyword
            arguments.
        jacobian_kwargs (dict): Additional keyword arguments for jacobian.
        hessian (numpy.ndarray, pandas.DataFrame or callable): Pre-calculated hessian
            of loglike at params or function to calculate it. The function is expected
            to take ``params`` as the first argument.
        hessian_kwargs (dict): Additional keyword arguments for hessian.
        design_info (pd.DataFrame): dataframe with one row per individual in the
            dataset, i.e. the same length as the "contributions" entry of ``loglike``.
            It has some or all of the following columns:
            - "psu": Indicates the primary sampling unit of the individual.
            - "stratum": Indicates the stratum of the individual.
            - "weight": Sampling weight of the individual.
            - "fpc": Finite population corrector.

    Returns:
        pd.DataFrame: DataFrame with same index as params, containing the columns
            "value", "standard_error", "pvalue", "ci_lower" and "ci_upper".
            Parameters that do not have a standard error (e.g. because they were fixed
            during estimation) contain NaNs in all but the "value" column. The value
            column is only reproduced for convenience.
        pd.DataFrame: Quadratic DataFrame containing the covariance matrix of the free
            parameters. If parameters were fixed (explicitly or by other constraints),
            the index is a subset of params.index. The columns are the same as the
            index.

    """
    numdiff_options = {} if numdiff_options is None else numdiff_options
    loglike_kwargs = {} if loglike_kwargs is None else loglike_kwargs
    constraints = [] if constraints is None else constraints

    if design_info is not None:
        raise NotImplementedError(
            "Cluster robust standard errors are not yet implemented."
        )

    if jacobian is not None:
        raise NotImplementedError()

    if hessian is not None:
        raise NotImplementedError()

    # calculate internal covariance matrix
    loglike = functools.partial(loglike, **loglike_kwargs)
    internal_loglike = numpy_interface(loglike, params=params, constraints=constraints)
    processed_constraints, processed_params = process_constraints(constraints, params)

    internal_params = reparametrize_to_internal(
        external=params["value"].to_numpy(),
        internal_free=processed_params["_internal_free"],
        processed_constraints=processed_constraints,
    )

    if cov_type == "jacobian":
        numdiff_options = numdiff_options.copy()
        numdiff_options["key"] = "contributions"

        internal_jac = first_derivative(
            internal_loglike,
            internal_params,
            **numdiff_options,
        )
        internal_cov = cov_jacobian(internal_jac["derivative"])
    else:
        raise NotImplementedError("Hessian calculation is not yet implemented.")

    # transform internal covariance matrix
    free_cov = transform_covariance(
        params=params,
        internal_cov=internal_cov,
        constraints=constraints,
        n_samples=n_samples,
        bounds_handling=bounds_handling,
    )

    # calculate quantities
    inference_quantities = calculate_inference_quantities(
        params=params,
        free_cov=free_cov,
        ci_level=1 - significance_level,
    )

    out = {"summary": inference_quantities, "cov": free_cov}

    return out
