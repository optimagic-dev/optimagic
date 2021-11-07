"""Do a method of simlated moments estimation."""
import functools
from collections.abc import Callable

import numpy as np
import pandas as pd
from estimagic.estimation.msm_weighting import get_weighting_matrix
from estimagic.inference.msm_covs import cov_optimal
from estimagic.inference.msm_covs import cov_robust
from estimagic.inference.shared import calculate_inference_quantities
from estimagic.inference.shared import check_is_optimized_and_derivative_case
from estimagic.inference.shared import get_derivative_case
from estimagic.inference.shared import get_internal_first_derivative
from estimagic.inference.shared import transform_covariance
from estimagic.optimization.optimize import minimize
from estimagic.parameters.parameter_conversion import get_derivative_conversion_function
from estimagic.parameters.process_constraints import process_constraints
from estimagic.sensitivity.msm_sensitivity import calculate_sensitivity_measures
from estimagic.shared.check_option_dicts import check_numdiff_options
from estimagic.shared.check_option_dicts import check_optimization_options


def estimate_msm(
    simulate_moments,
    empirical_moments,
    moments_cov,
    params,
    optimize_options,
    *,
    constraints=None,
    logging=False,
    log_options=None,
    simulate_moments_kwargs=None,
    weights="diagonal",
    numdiff_options=None,
    jacobian=None,
    jacobian_kwargs=None,
    simulate_moments_and_jacobian=None,
    simulate_moments_and_jacobian_kwargs=None,
    ci_level=0.95,
    n_samples=10_000,
    bounds_handling="raise",
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
        optimize_options (dict or False): Keyword arguments that govern the numerical
            optimization. Valid entries are all arguments of
            :func:`~estimagic.optimization.optimize.minimize` except for criterion,
            derivative, criterion_and_derivative and params. If you pass False as
            optimize_options you signal that ``params`` are already the optimal
            parameters and no numerical optimization is needed.
        numdiff_options (dict): Keyword arguments for the calculation of numerical
            derivatives for the calculation of standard errors. See
            :ref:`first_derivative` for details. Note that by default we increase the
            step_size by a factor of 2 compared to the rule of thumb for optimal
            step sizes. This is because many msm criterion functions are slightly
            noisy.
        jacobian (callable or pandas.DataFrame): A function that take ``params`` and
            potentially other keyword arguments and returns the Jacobian of
            simulate_moments with respect to the params. Alternatively, you can pass
            a pandas.DataFrame with the Jacobian at the optimal parameters. This is
            only possible if you pass ``optimize_options=False``.
        jacobian_kwargs (dict): Additional keyword arguments for the Jacobian function.
        simulate_moments_and_jacobian (callable): A function that takes params and
            potentially other keyword arguments and returns a tuple with simulated
            moments and the jacobian of simulated moments with respect to params.
        simulate_moments_and_jacobian_kwargs (dict): Additional keyword arguments for
            simulate_moments_and_jacobian.
        ci_level (float): Confidence level for the calculation of confidence intervals.
            The default is 0.95
        n_samples (int): Number of samples used to transform the covariance matrix of
            the internal parameter vector into the covariance matrix of the external
            parameters. For background information about internal and external params
            see :ref:`implementation_of_constraints`. This is only used if you have
            constraints in the ``optimize_options``
        bounds_handling (str): One of "clip", "raise", "ignore". Determines how bounds
            are handled. If "clip", confidence intervals are clipped at the bounds.
            Standard errors are only adjusted if a sampling step is necessary due to
            additional constraints. If "raise" and any lower or upper bound is binding,
            we raise an error. If "ignore", boundary problems are simply ignored.

        Returns:
            dict: The estimated parameters, standard errors and sensitivity measures
                and covariance matrix of the parameters.

    """
    # ==================================================================================
    # Check and process inputs
    # ==================================================================================
    is_optimized = optimize_options is False

    check_optimization_options(
        optimize_options,
        usage="estimate_msm",
        algorithm_mandatory=True,
    )

    jac_case = get_derivative_case(jacobian)

    cov_case = _get_cov_case(weights)

    check_is_optimized_and_derivative_case(is_optimized, jac_case)

    check_numdiff_options(numdiff_options, "estimate_msm")

    numdiff_options = {} if numdiff_options in (None, False) else numdiff_options.copy()
    numdiff_options["key"] = "simulated_moments"
    if "scaling_factor" not in numdiff_options:
        numdiff_options["scaling_factor"] = 2

    if not isinstance(weights, (np.ndarray, pd.DataFrame)):
        weights = get_weighting_matrix(moments_cov, weights)

    constraints = [] if constraints is None else constraints

    processed_constraints, _ = process_constraints(constraints, params)

    # ==================================================================================
    # Calculate estimates via minimization (if necessary)
    # ==================================================================================

    if is_optimized:
        opt_res = {"solution_params": params}
    else:
        funcs = get_msm_optimization_functions(
            simulate_moments=simulate_moments,
            empirical_moments=empirical_moments,
            weights=weights,
            simulate_moments_kwargs=simulate_moments_kwargs,
            jacobian=jacobian,
            jacobian_kwargs=jacobian_kwargs,
            simulate_moments_and_jacobian=simulate_moments_and_jacobian,
            simulate_moments_and_jacobian_kwargs=simulate_moments_and_jacobian_kwargs,
        )

        opt_res = minimize(
            constraints=constraints,
            logging=logging,
            log_options=log_options,
            params=params,
            **funcs,
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
    # switch to "numerical" even if jac_case == "skip" because jac is required for msm.
    else:
        deriv_res = get_internal_first_derivative(
            func=simulate_moments,
            params=estimates,
            constraints=constraints,
            func_kwargs=simulate_moments_kwargs,
            numdiff_options=numdiff_options,
        )
        int_jac = deriv_res["derivative"]
        numdiff_info = {k: v for k, v in deriv_res.items() if k != "derivative"}

    # ==================================================================================
    # Calculate internal cov
    # ==================================================================================

    if cov_case == "optimal":
        cov = cov_optimal(int_jac, weights)
    else:
        cov = cov_robust(int_jac, weights, moments_cov)

    # ==================================================================================
    # Calculate external cov and summary
    # ==================================================================================

    cov = transform_covariance(
        params=estimates,
        internal_cov=cov,
        constraints=constraints,
        n_samples=n_samples,
        bounds_handling=bounds_handling,
    )

    summary = calculate_inference_quantities(
        params=estimates,
        free_cov=cov,
        ci_level=ci_level,
    )
    # ==================================================================================
    # Calculate external jac (if no transforming constraints)
    # ==================================================================================

    if not processed_constraints:
        if isinstance(moments_cov, pd.DataFrame):
            moments_names = moments_cov.index
        else:
            moments_names = None
        ext_jac = pd.DataFrame(int_jac, columns=cov.index, index=moments_names)
    else:
        ext_jac = "No external jacobian defined due to constraints."

    # ==================================================================================
    # Calculate sensitivity measures (if no transforming constraints)
    # ==================================================================================

    if not processed_constraints:
        sensitivities = calculate_sensitivity_measures(
            jac=int_jac,
            weights=weights,
            moments_cov=moments_cov,
            params_cov=cov,
        )
    else:
        sensitivities = "No sensitivity measures can be calculated due to constraints."

    # ==================================================================================
    # Construct output
    # ==================================================================================
    out = {
        "summary": summary,
        "cov": cov,
        "sensitivity": sensitivities,
        "jacobian": ext_jac,
    }

    if not is_optimized:
        out["optimize_res"] = opt_res

    if jac_case == "numerical":
        out["jacobian_numdiff_info"] = numdiff_info

    return out


def get_msm_optimization_functions(
    simulate_moments,
    empirical_moments,
    weights,
    simulate_moments_kwargs=None,
    jacobian=None,
    jacobian_kwargs=None,
    simulate_moments_and_jacobian=None,
    simulate_moments_and_jacobian_kwargs=None,
):
    """Construct criterion functions and their derivatives for msm estimation.

    Args:
        simulate_moments (callable): Function that takes params and potentially other
            keyworrd arguments and returns simulated moments as a pandas Series.
            Alternatively, the function can return a dict with any number of entries
            as long as one of those entries is "simulated_moments".
        empirical_moments (pandas.Series): A pandas series with the empirical
            equivalents of the simulated moments.
        weights (pandas.DataFrame): DataFrame with a positive
            semi-definite weighting matrix.
        simulate_moments_kwargs (dict): Additional keyword arguments for
            ``simulate_moments``.
        jacobian (callable or pandas.DataFrame): A function that take ``params`` and
            potentially other keyword arguments and returns the jacobian of
            simulate_moments with respect to the params. Alternatively you can pass
            a pandas.DataFrame with the jacobian at the optimal parameters. This is
            only possible if you pass ``optimize_options=False``.
        jacobian_kwargs (dict): Additional keyword arguments for jacobian.
        simulate_moments_and_jacobian (callable): A function that takes params and
            potentially other keyword arguments and returns a tuple with simulated
            moments and the jacobian of simulated moments with respect to params.
        simulate_moments_and_jacobian_kwargs (dict): Additional keyword arguments for
            simulate_moments_and_jacobian.

    Returns:
        dict: Dictionary containing at least the entry "criterion". If enough inputs
            are provided it also contains the entries "derivative" and
            "criterion_and_derivative". All values are functions that take params
            as only argument.

    """
    _simulate_moments = _partial_kwargs(simulate_moments, simulate_moments_kwargs)
    _jacobian = _partial_kwargs(jacobian, jacobian_kwargs)
    _simulate_moments_and_jacobian = _partial_kwargs(
        simulate_moments_and_jacobian, simulate_moments_and_jacobian_kwargs
    )

    criterion = functools.partial(
        _msm_criterion,
        simulate_moments=_simulate_moments,
        empirical_moments=empirical_moments,
        weights=weights,
    )

    out = {"criterion": criterion}

    if _jacobian is not None:
        raise NotImplementedError(
            "Closed form jacobians are not yet supported in estimate_msm"
        )

    if _simulate_moments_and_jacobian is not None:
        raise NotImplementedError(
            "Closed form jacobians are not yet supported in estimate_msm"
        )

    return out


def _msm_criterion(params, simulate_moments, empirical_moments, weights):
    """Calculate msm criterion given parameters and building blocks."""
    simulated = simulate_moments(params)
    if isinstance(simulated, dict):
        simulated = simulated["simulated_moments"]
    deviations = simulated - empirical_moments
    out = deviations @ weights @ deviations
    return out


def _partial_kwargs(func, kwargs):
    """Partial keyword arguments into a function.

    In contrast to normal partial this works if kwargs in None. If func is not a
    callable it simply returns None.

    """
    if isinstance(func, Callable):
        if kwargs not in (None, {}):
            out = functools.partial(func, **kwargs)
        else:
            out = func
    else:
        out = None

    return out


def _get_cov_case(weights):
    return weights == "optimal"
