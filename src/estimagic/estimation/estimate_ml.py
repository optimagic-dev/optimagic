from estimagic.differentiation.derivatives import first_derivative
from estimagic.differentiation.derivatives import second_derivative
from estimagic.exceptions import InvalidFunctionError
from estimagic.inference.ml_covs import cov_cluster_robust
from estimagic.inference.ml_covs import cov_hessian
from estimagic.inference.ml_covs import cov_jacobian
from estimagic.inference.ml_covs import cov_robust
from estimagic.inference.ml_covs import cov_strata_robust
from estimagic.inference.shared import calculate_inference_quantities
from estimagic.inference.shared import check_is_optimized_and_derivative_case
from estimagic.inference.shared import get_derivative_case
from estimagic.inference.shared import transform_covariance
from estimagic.optimization.optimize import maximize
from estimagic.parameters.conversion import get_converter
from estimagic.parameters.parameter_bounds import get_bounds
from estimagic.shared.check_option_dicts import check_numdiff_options
from estimagic.shared.check_option_dicts import check_optimization_options


def estimate_ml(
    loglike,
    params,
    optimize_options,
    *,
    lower_bounds=None,
    upper_bounds=None,
    constraints=None,
    logging=False,
    log_options=None,
    loglike_kwargs=None,
    numdiff_options=None,
    jacobian=None,
    jacobian_kwargs=None,
    hessian=False,
    hessian_kwargs=None,
    ci_level=0.95,
    n_samples=10_000,
    bounds_handling="raise",
    design_info=None,
):
    """Do a maximum likelihood (ml) estimation.

    This is a high level interface of our lower level functions for maximization,
    numerical differentiation and inference. It does the full workflow for maximum
    likelihood estimation with just one function call.

    While we have good defaults, you can still configure each aspect of each step
    via the optional arguments of this function. If you find it easier to do the
    "difficult" steps (mainly maximization and calculating numerical derivatives
    of a potentially noisy function) separately, you can do so and just provide those
    results as ``params``, ``jacobian`` and ``hessian``.

    The docstring is aspirational and not all options are supported yet.

    Args:
        loglike (callable): Likelihood function that takes a params (and potentially
            other keyword arguments) and returns a dictionary that has at least the
            entries "value" (a scalar float) and "contributions" (a 1d numpy array or
            pandas Series) with the log likelihood contribution per individual.
        params (pytree): A pytree containing the estimated or start parameters of the
            likelihood model. If the supplied parameters are estimated parameters, set
            optimize_options to False. Pytrees can be a numpy array, a pandas Series, a
            DataFrame with "value" column, a float and any kind of (nested) dictionary
            or list containing these elements. See :ref:`params` for examples.
        optimize_options (dict, str or False): Keyword arguments that govern the
            numerical optimization. Valid entries are all arguments of
            :func:`~estimagic.optimization.optimize.minimize` except for criterion. If
            you pass False as optimize_options you signal that ``params`` are already
            the optimal parameters and no numerical optimization is needed. If you pass
            a str as optimize_options it is used as the ``algorithm`` option.
        lower_bounds (pytree): A pytree with the same structure as params with lower
            bounds for the parameters. Can be ``-np.inf`` for parameters with no lower
            bound.
        upper_bounds (pytree): As lower_bounds. Can be ``np.inf`` for parameters with
            no upper bound.
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
        loglike_kwargs (dict): Additional keyword arguments for loglike.
        numdiff_options (dict): Keyword arguments for the calculation of numerical
            derivatives for the calculation of standard errors. See
            :ref:`first_derivative` for details.
        jacobian (callable or None): A function that takes ``params`` and potentially
            other keyword arguments and returns the jacobian of loglike["contributions"]
            with respect to the params. Note that you only need to pass a Jacobian
            function if you have a closed form Jacobian but decided not to return it as
            part of ``derivative`` (e.g.  because you use a scalar optimizer and can
            calculate a gradient in a way that is faster than calculating and summing
            the Jacobian). If you pass None, a numerical Jacobian will be calculated.
        jacobian_kwargs (dict): Additional keyword arguments for the Jacobian function.
        hessian (callable or None or False): A function that takes ``params`` and
            potentially other keyword arguments and returns the Hessian of
            loglike["value"] with respect to the params.  If you pass None, a numerical
            Hessian will be calculated. If you pass ``False``, you signal that no
            Hessian should be calculated. Thus, no result that requires the Hessian will
            be calculated.
        hessian_kwargs (dict): Additional keyword arguments for the Hessian function.
        ci_level (float): Confidence level for the calculation of confidence intervals.
            The default is 0.95.
        n_samples (int): Number of samples used to transform the covariance matrix of
            the internal parameter vector into the covariance matrix of the external
            parameters. For background information about internal and external params
            see :ref:`implementation_of_constraints`. This is only used if you have
            specified constraints.
        bounds_handling (str): One of "clip", "raise", "ignore". Determines how bounds
            are handled. If "clip", confidence intervals are clipped at the bounds.
            Standard errors are only adjusted if a sampling step is necessary due to
            additional constraints. If "raise" and any lower or upper bound is binding,
            we raise an Error. If "ignore", boundary problems are simply ignored.
        design_info (pandas.DataFrame): DataFrame with one row per observation that
            contains some or all of the variables "psu" (primary sampling unit),
            "stratum" and "fpc" (finite population corrector). See
            :ref:`robust_likelihood_inference` for details.

    Returns:
        dict: The estimated parameters, standard errors and covariance matrix of the
            parameters.

    """
    # ==================================================================================
    # Check and process inputs
    # ==================================================================================

    is_optimized = optimize_options is False

    if not is_optimized:
        if isinstance(optimize_options, str):
            optimize_options = {"algorithm": optimize_options}

        check_optimization_options(
            optimize_options,
            usage="estimate_ml",
            algorithm_mandatory=True,
        )

    jac_case = get_derivative_case(jacobian)
    hess_case = get_derivative_case(hessian)

    check_is_optimized_and_derivative_case(is_optimized, jac_case)
    check_is_optimized_and_derivative_case(is_optimized, hess_case)

    cov_cases = _get_cov_cases(jac_case, hess_case, design_info)

    check_numdiff_options(numdiff_options, "estimate_ml")
    numdiff_options = {} if numdiff_options in (None, False) else numdiff_options
    loglike_kwargs = {} if loglike_kwargs is None else loglike_kwargs
    constraints = [] if constraints is None else constraints
    jacobian_kwargs = {} if jacobian_kwargs is None else jacobian_kwargs
    hessian_kwargs = {} if hessian_kwargs is None else hessian_kwargs

    # ==================================================================================
    # Calculate estimates via maximization (if necessary)
    # ==================================================================================

    if is_optimized:
        estimates = params
    else:
        opt_res = maximize(
            criterion=loglike,
            criterion_kwargs=loglike_kwargs,
            params=params,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            constraints=constraints,
            logging=logging,
            log_options=log_options,
            **optimize_options,
        )
        estimates = opt_res.params

    # ==================================================================================
    # Do first function evaluations at estimated parameters
    # ==================================================================================

    try:
        loglike_eval = loglike(estimates, **loglike_kwargs)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        msg = "Error while evaluating loglike at estimated params."
        raise InvalidFunctionError(msg) from e

    if callable(jacobian):
        try:
            jacobian_eval = jacobian(estimates, **jacobian_kwargs)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            msg = "Error while evaluating derivative at estimated params."
            raise InvalidFunctionError(msg) from e
    else:
        jacobian_eval = None

    # ==================================================================================
    # Get the converter for params and function outputs
    # ==================================================================================

    lower_bounds, upper_bounds = get_bounds(
        params,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )

    converter, flat_estimates = get_converter(
        func=loglike,
        params=estimates,
        constraints=constraints,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        func_eval=loglike_eval,
        primary_key="contributions",
        scaling=False,
        scaling_options=None,
        derivative_eval=jacobian_eval,
    )

    # ==================================================================================
    # Calculate internal jacobian
    # ==================================================================================

    if jac_case == "closed-form":
        x = converter.params_to_internal(estimates)
        int_jac = converter.derivative_to_internal(jacobian_eval, x)
    # switch to "numerical" even if jac_case == "skip" because jac is required for ml.
    elif jac_case == "numerical":

        def func(x):
            p = converter.params_from_internal(x)
            loglike_eval = loglike(p, **loglike_kwargs)["contributions"]
            out = converter.func_to_internal(loglike_eval)
            return out

        jac_res = first_derivative(
            func=func,
            params=flat_estimates.values,
            lower_bounds=flat_estimates.lower_bounds,
            upper_bounds=flat_estimates.upper_bounds,
            **numdiff_options,
        )

        int_jac = jac_res["derivative"]
        jac_numdiff_info = {k: v for k, v in jac_res.items() if k != "derivative"}
    else:
        int_jac = None

    # ==================================================================================
    # Calculate internal Hessian
    # ==================================================================================

    if hess_case == "skip":
        int_hess = None
    elif hess_case == "numerical":

        def func(x):
            p = converter.params_from_internal(x)
            loglike_eval = loglike(p, **loglike_kwargs)["value"]
            out = converter.func_to_internal(loglike_eval)
            return out

        hess_res = second_derivative(
            func=func,
            params=flat_estimates.values,
            lower_bounds=flat_estimates.lower_bounds,
            upper_bounds=flat_estimates.upper_bounds,
            **numdiff_options,
        )
        int_hess = hess_res["derivative"]
        hess_numdiff_info = {k: v for k, v in hess_res.items() if k != "derivative"}
    elif hess_case == "closed-form" and constraints:
        raise NotImplementedError(
            "Closed-form Hessians are not yet compatible with constraints."
        )
    else:
        int_hess = hessian(estimates, **hessian_kwargs)

    # ==================================================================================
    # Calculate all available internal cov types
    # ==================================================================================

    int_covs = {}
    if "jacobian" in cov_cases:
        int_covs["cov_jacobian"] = cov_jacobian(int_jac)
    if "hessian" in cov_cases:
        int_covs["cov_hessian"] = cov_hessian(int_hess)
    if "robust" in cov_cases:
        int_covs["cov_robust"] = cov_robust(jac=int_jac, hess=int_hess)
    if "cluster_robust" in cov_cases:
        int_covs["cov_cluster_robust"] = cov_cluster_robust(
            jac=int_jac, hess=int_hess, design_info=design_info
        )
    if "strata_robust" in cov_cases:
        int_covs["cov_strata_robust"] = cov_strata_robust(
            jac=int_jac, hess=int_hess, design_info=design_info
        )

    # ==================================================================================
    # Calculate all available external covs and summaries
    # ==================================================================================

    covs = {}
    summaries = {}
    for case in cov_cases:
        cov = transform_covariance(
            flat_params=flat_estimates,
            internal_cov=int_covs[f"cov_{case}"],
            converter=converter,
            n_samples=n_samples,
            bounds_handling=bounds_handling,
        )
        summary = calculate_inference_quantities(
            estimates=estimates,
            flat_estimates=flat_estimates,
            free_cov=cov,
            ci_level=ci_level,
        )

        covs[f"cov_{case}"] = cov
        summaries[f"summary_{case}"] = summary

    # ==================================================================================
    # Calculate external jac and hess (if no transforming constraints)
    # ==================================================================================

    if not converter.has_transforming_constraints:
        ext_jac = int_jac
        ext_hess = int_hess
    else:
        ext_jac = "No external Jacobian defined due to constraints."
        ext_hess = "No external Hessian defined due to constraints."

    # ==================================================================================
    # Construct output
    # ==================================================================================

    out = {
        **summaries,
        **covs,
        "jacobian": ext_jac,
        "hessian": ext_hess,
    }

    if not is_optimized:
        out["optimize_res"] = opt_res

    if jac_case == "numerical":
        out["jacobian_numdiff_info"] = jac_numdiff_info

    if hess_case == "numerical":
        out["hessian_numdiff_info"] = hess_numdiff_info

    return out


def _get_cov_cases(jac_case, hess_case, design_info):
    if jac_case == "skip" and hess_case == "skip":
        raise ValueError("Jacobian and Hessian cannot both be False.")
    elif jac_case == "skip" and hess_case != "skip":
        cases = ["hessian"]
    elif hess_case == "skip" and jac_case != "skip":
        cases = ["jacobian"]
    else:
        cases = ["jacobian", "hessian", "robust"]
        if design_info is not None:
            if "psu" in design_info:
                cases.append("cluster_robust")
            if {"strata", "psu", "fpc"}.issubset(design_info):
                cases.append("strata_robust")

    return cases
