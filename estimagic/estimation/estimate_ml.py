from estimagic.inference.ml_covs import cov_cluster_robust
from estimagic.inference.ml_covs import cov_hessian
from estimagic.inference.ml_covs import cov_jacobian
from estimagic.inference.ml_covs import cov_robust
from estimagic.inference.ml_covs import cov_strata_robust
from estimagic.inference.shared import calculate_inference_quantities
from estimagic.inference.shared import check_is_optimized_and_derivative_case
from estimagic.inference.shared import get_derivative_case
from estimagic.inference.shared import get_internal_first_derivative
from estimagic.inference.shared import transform_covariance
from estimagic.optimization.optimize import maximize
from estimagic.parameters.parameter_conversion import get_derivative_conversion_function
from estimagic.parameters.process_constraints import process_constraints
from estimagic.shared.check_option_dicts import check_numdiff_options
from estimagic.shared.check_option_dicts import check_optimization_options


def estimate_ml(
    loglike,
    params,
    optimize_options,
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
        loglike (callable): Likelihood function that takes a params DataFrame (and
            potentially other keyword arguments) and returns a dictionary that has at
            least the entries "value" (a scalar float) and "contributions" (a 1d numpy
            array or pandas Series) with the log likelihood contribution per individual.
        params (pd.DataFrame): DataFrame where the "value" column contains the
            estimated or start parameters of a likelihood model. See :ref:`params` for
            details. If the supplied parameters are estimated parameters, set
            optimize_options to False.
        optimize_options (dict or False): Keyword arguments that govern the numerical
            optimization. Valid entries are all arguments of
            :func:`~estimagic.optimization.optimize.minimize` except for criterion,
            derivative, criterion_and_derivative and params. If you pass False as
            optimize_options you signal that ``params`` are already the optimal
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
        loglike_kwargs (dict): Additional keyword arguments for loglike.
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
        loglike_and_derivative_kwargs (dict): Additional keyword arguments for
            loglike_and_derivative.
        numdiff_options (dict): Keyword arguments for the calculation of numerical
            derivatives for the calculation of standard errors. See
            :ref:`first_derivative` for details.
        jacobian (callable or pandas.DataFrame or False): A function that takes
            ``params`` and potentially other keyword arguments and returns the jacobian
            of loglike["contributions"] with respect to the params. Alternatively, you
            can pass a pandas.DataFrame with the Jacobian at the optimal parameters.
            This is only possible if you pass ``optimize_options=False``. Note that you
            only need to pass a Jacobian function if you have a closed form Jacobian but
            decided not to return it as part of ``derivative`` (e.g. because you use
            a scalar optimizer and can calculate a gradient in a way that is faster
            than calculating and summing the Jacobian). If you pass None, a numerical
            Jacobian will be calculated. If you pass ``False``, you signal that no
            Jacobian should be calculated. Thus, no result that requires the Jacobian
            will be calculated.
        jacobian_kwargs (dict): Additional keyword arguments for the Jacobian function.
        hessian (callable or pd.DataFrame): A function that takes
            ``params`` and potentially other keyword arguments and returns the Hessian
            of loglike["value"] with respect to the params. Alternatively, you
            can pass a pandas.DataFrame with the Hessian at the optimal parameters.
            This is only possible if you pass ``optimize_options=False``. If you pass
            None, a numerical Hessian will be calculated. If you pass ``False``, you
            signal that no Hessian should be calculated. Thus, no result that requires
            the Hessian will be calculated.
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

    constraints = [] if constraints is None else constraints

    processed_constraints, _ = process_constraints(constraints, params)

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
            constraints=constraints,
            derivative=derivative,
            derivative_kwargs=derivative_kwargs,
            criterion_and_derivative=loglike_and_derivative,
            criterion_and_derivative_kwargs=loglike_and_derivative_kwargs,
            logging=logging,
            log_options=log_options,
            **optimize_options,
        )
        estimates = opt_res["solution_params"]

    # ==================================================================================
    # Calculate internal jacobian
    # ==================================================================================

    deriv_to_internal = get_derivative_conversion_function(
        params=params, constraints=constraints
    )

    if jac_case == "pre-calculated":
        int_jac = deriv_to_internal(jacobian)
    elif jac_case == "closed-form":
        jacobian_kwargs = {} if jacobian_kwargs is None else jacobian_kwargs
        _jac = jacobian(estimates, **jacobian_kwargs)
        int_jac = deriv_to_internal(_jac)
    # switch to "numerical" even if jac_case == "skip" because jac is required for ml.
    elif jac_case == "numerical":
        options = numdiff_options.copy()
        options["key"] = "contributions"
        deriv_res = get_internal_first_derivative(
            func=loglike,
            params=estimates,
            constraints=constraints,
            func_kwargs=loglike_kwargs,
            numdiff_options=options,
        )
        int_jac = deriv_res["derivative"]
        jac_numdiff_info = {k: v for k, v in deriv_res.items() if k != "derivative"}
    else:
        int_jac = None

    # ==================================================================================
    # Calculate internal Hessian (most of this is not yet implemented)
    # ==================================================================================

    if hess_case == "skip":
        int_hess = None
    elif hess_case == "numerical":
        raise NotImplementedError("Numerical Hessian calculation is not yet supported.")
        hess_numdiff_info = {}
    elif hess_case in ("closed-form", "pre-calculated") and constraints:
        raise NotImplementedError(
            "Closed-form or pre-calculated Hessians are not yet compatible with "
            "constraints."
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
            params=estimates,
            internal_cov=int_covs[f"cov_{case}"],
            constraints=constraints,
            n_samples=n_samples,
            bounds_handling=bounds_handling,
        )
        summary = calculate_inference_quantities(
            params=estimates,
            free_cov=cov,
            ci_level=ci_level,
        )

        covs[f"cov_{case}"] = cov
        summaries[f"summary_{case}"] = summary

    # ==================================================================================
    # Calculate external jac and hess (if no transforming constraints)
    # ==================================================================================

    if not processed_constraints:
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
