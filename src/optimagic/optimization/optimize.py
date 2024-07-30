import functools
import warnings
from pathlib import Path

from optimagic.batch_evaluators import process_batch_evaluator
from optimagic.exceptions import (
    InvalidFunctionError,
    InvalidKwargsError,
    MissingInputError,
    AliasError,
)
from optimagic.logging.create_tables import (
    make_optimization_iteration_table,
    make_optimization_problem_table,
    make_steps_table,
)
from optimagic.logging.load_database import load_database
from optimagic.logging.write_to_database import append_row
from optimagic.optimization.check_arguments import check_optimize_kwargs
from optimagic.optimization.error_penalty import get_error_penalty_function
from optimagic.optimization.get_algorithm import (
    get_final_algorithm,
    process_user_algorithm,
)
from optimagic.optimization.internal_criterion_template import (
    internal_criterion_and_derivative_template,
)
from optimagic.optimization.optimization_logging import log_scheduled_steps_and_get_ids
from optimagic.optimization.process_multistart_sample import process_multistart_sample
from optimagic.optimization.process_results import process_internal_optimizer_result
from optimagic.optimization.tiktak import WEIGHT_FUNCTIONS, run_multistart_optimization
from optimagic.parameters.conversion import (
    aggregate_func_output_to_value,
    get_converter,
)
from optimagic.parameters.nonlinear_constraints import process_nonlinear_constraints
from optimagic.shared.process_user_function import (
    process_func_of_params,
    get_kwargs_from_args,
)
from optimagic.optimization.scipy_aliases import (
    map_method_to_algorithm,
    split_fun_and_jac,
)
from optimagic import deprecations
from optimagic.deprecations import (
    replace_and_warn_about_deprecated_algo_options,
    replace_and_warn_about_deprecated_bounds,
)
from optimagic.parameters.bounds import Bounds, pre_process_bounds


def maximize(
    fun=None,
    params=None,
    algorithm=None,
    *,
    bounds=None,
    constraints=None,
    fun_kwargs=None,
    algo_options=None,
    jac=None,
    jac_kwargs=None,
    fun_and_jac=None,
    fun_and_jac_kwargs=None,
    numdiff_options=None,
    logging=False,
    log_options=None,
    error_handling="raise",
    error_penalty=None,
    scaling=False,
    scaling_options=None,
    multistart=False,
    multistart_options=None,
    collect_history=True,
    skip_checks=False,
    # scipy aliases
    x0=None,
    method=None,
    args=None,
    # scipy arguments that are not yet supported
    hess=None,
    hessp=None,
    callback=None,
    # scipy arguments that will never be supported
    options=None,
    tol=None,
    # deprecated arguments
    criterion=None,
    criterion_kwargs=None,
    derivative=None,
    derivative_kwargs=None,
    criterion_and_derivative=None,
    criterion_and_derivative_kwargs=None,
    lower_bounds=None,
    upper_bounds=None,
    soft_lower_bounds=None,
    soft_upper_bounds=None,
):
    """Maximize criterion using algorithm subject to constraints."""
    return _optimize(
        direction="maximize",
        fun=fun,
        params=params,
        bounds=bounds,
        algorithm=algorithm,
        fun_kwargs=fun_kwargs,
        constraints=constraints,
        algo_options=algo_options,
        jac=jac,
        jac_kwargs=jac_kwargs,
        fun_and_jac=fun_and_jac,
        fun_and_jac_kwargs=fun_and_jac_kwargs,
        numdiff_options=numdiff_options,
        logging=logging,
        log_options=log_options,
        error_handling=error_handling,
        error_penalty=error_penalty,
        scaling=scaling,
        scaling_options=scaling_options,
        multistart=multistart,
        multistart_options=multistart_options,
        collect_history=collect_history,
        skip_checks=skip_checks,
        # scipy aliases
        x0=x0,
        method=method,
        args=args,
        # scipy arguments that are not yet supported
        hess=hess,
        hessp=hessp,
        callback=callback,
        # scipy arguments that will never be supported
        options=options,
        tol=tol,
        # deprecated arguments
        criterion=criterion,
        criterion_kwargs=criterion_kwargs,
        derivative=derivative,
        derivative_kwargs=derivative_kwargs,
        criterion_and_derivative=criterion_and_derivative,
        criterion_and_derivative_kwargs=criterion_and_derivative_kwargs,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        soft_lower_bounds=soft_lower_bounds,
        soft_upper_bounds=soft_upper_bounds,
    )


def minimize(
    fun=None,
    params=None,
    algorithm=None,
    *,
    bounds=None,
    constraints=None,
    algo_options=None,
    jac=None,
    jac_kwargs=None,
    fun_and_jac=None,
    fun_and_jac_kwargs=None,
    numdiff_options=None,
    logging=False,
    log_options=None,
    error_handling="raise",
    error_penalty=None,
    scaling=False,
    scaling_options=None,
    multistart=False,
    multistart_options=None,
    collect_history=True,
    skip_checks=False,
    # scipy aliases
    x0=None,
    method=None,
    args=None,
    # scipy arguments that are not yet supported
    hess=None,
    hessp=None,
    callback=None,
    # scipy arguments that will never be supported
    options=None,
    tol=None,
    # deprecated arguments
    criterion=None,
    criterion_kwargs=None,
    derivative=None,
    derivative_kwargs=None,
    criterion_and_derivative=None,
    criterion_and_derivative_kwargs=None,
    lower_bounds=None,
    upper_bounds=None,
    soft_lower_bounds=None,
    soft_upper_bounds=None,
    fun_kwargs=None,
):
    """Minimize criterion using algorithm subject to constraints."""

    return _optimize(
        direction="minimize",
        fun=fun,
        params=params,
        algorithm=algorithm,
        bounds=bounds,
        fun_kwargs=fun_kwargs,
        constraints=constraints,
        algo_options=algo_options,
        jac=jac,
        jac_kwargs=jac_kwargs,
        fun_and_jac=fun_and_jac,
        fun_and_jac_kwargs=fun_and_jac_kwargs,
        numdiff_options=numdiff_options,
        logging=logging,
        log_options=log_options,
        error_handling=error_handling,
        error_penalty=error_penalty,
        scaling=scaling,
        scaling_options=scaling_options,
        multistart=multistart,
        multistart_options=multistart_options,
        collect_history=collect_history,
        skip_checks=skip_checks,
        # scipy aliases
        x0=x0,
        method=method,
        args=args,
        # scipy arguments that are not yet supported
        hess=hess,
        hessp=hessp,
        callback=callback,
        # scipy arguments that will never be supported
        options=options,
        tol=tol,
        # deprecated arguments
        criterion=criterion,
        criterion_kwargs=criterion_kwargs,
        derivative=derivative,
        derivative_kwargs=derivative_kwargs,
        criterion_and_derivative=criterion_and_derivative,
        criterion_and_derivative_kwargs=criterion_and_derivative_kwargs,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        soft_lower_bounds=soft_lower_bounds,
        soft_upper_bounds=soft_upper_bounds,
    )


def _optimize(
    direction,
    fun,
    params,
    algorithm,
    *,
    bounds,
    fun_kwargs,
    constraints,
    algo_options,
    jac,
    jac_kwargs,
    fun_and_jac,
    fun_and_jac_kwargs,
    numdiff_options,
    logging,
    log_options,
    error_handling,
    error_penalty,
    scaling,
    scaling_options,
    multistart,
    multistart_options,
    collect_history,
    skip_checks,
    # scipy aliases
    x0,
    method,
    args,
    # scipy arguments that are not yet supported
    hess,
    hessp,
    callback,
    # scipy arguments that will never be supported
    options,
    tol,
    # deprecated arguments
    criterion,
    criterion_kwargs,
    derivative,
    derivative_kwargs,
    criterion_and_derivative,
    criterion_and_derivative_kwargs,
    lower_bounds,
    upper_bounds,
    soft_lower_bounds,
    soft_upper_bounds,
):
    """Minimize or maximize criterion using algorithm subject to constraints.

    Arguments are the same as in maximize and minimize, with an additional direction
    argument. Direction is a string that can take the values "maximize" and "minimize".

    Returns are the same as in maximize and minimize.

    """
    # ==================================================================================
    # error handling needed as long as fun is an optional argument (i.e. until
    # criterion is fully removed).
    # ==================================================================================

    if fun is None and criterion is None:
        msg = (
            "Missing objective function. Please provide an objective function as the "
            "first positional argument or as the keyword argument `fun`."
        )
        raise MissingInputError(msg)

    if params is None and x0 is None:
        msg = (
            "Missing start parameters. Please provide start parameters as the second "
            "positional argument or as the keyword argument `params`."
        )
        raise MissingInputError(msg)

    if algorithm is None and method is None:
        msg = (
            "Missing algorithm. Please provide an algorithm as the third positional "
            "argument or as the keyword argument `algorithm`."
        )
        raise MissingInputError(msg)

    # ==================================================================================
    # deprecations
    # ==================================================================================

    if criterion is not None:
        deprecations.throw_criterion_future_warning()
        fun = criterion if fun is None else fun

    if criterion_kwargs is not None:
        deprecations.throw_criterion_kwargs_future_warning()
        fun_kwargs = criterion_kwargs if fun_kwargs is None else fun_kwargs

    if derivative is not None:
        deprecations.throw_derivative_future_warning()
        jac = derivative if jac is None else jac

    if derivative_kwargs is not None:
        deprecations.throw_derivative_kwargs_future_warning()
        jac_kwargs = derivative_kwargs if jac_kwargs is None else jac_kwargs

    if criterion_and_derivative is not None:
        deprecations.throw_criterion_and_derivative_future_warning()
        fun_and_jac = criterion_and_derivative if fun_and_jac is None else fun_and_jac

    if criterion_and_derivative_kwargs is not None:
        deprecations.throw_criterion_and_derivative_kwargs_future_warning()
        fun_and_jac_kwargs = (
            criterion_and_derivative_kwargs
            if fun_and_jac_kwargs is None
            else fun_and_jac_kwargs
        )

    algo_options = replace_and_warn_about_deprecated_algo_options(algo_options)

    bounds = replace_and_warn_about_deprecated_bounds(
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        bounds=bounds,
        soft_lower_bounds=soft_lower_bounds,
        soft_upper_bounds=soft_upper_bounds,
    )

    # ==================================================================================
    # handle scipy aliases
    # ==================================================================================

    if x0 is not None:
        if params is not None:
            msg = (
                "x0 is an alias for params (for better compatibility with scipy). "
                "Do not use both x0 and params."
            )
            raise AliasError(msg)
        else:
            params = x0

    if method is not None:
        if algorithm is not None:
            msg = (
                "method is an alias for algorithm to select the scipy optimizers under "
                "their original name. Do not use both method and algorithm."
            )
            raise AliasError(msg)
        else:
            algorithm = map_method_to_algorithm(method)

    if args is not None:
        if (
            fun_kwargs is not None
            or jac_kwargs is not None
            or fun_and_jac_kwargs is not None
        ):
            msg = (
                "args is an alternative to fun_kwargs, jac_kwargs and "
                "fun_and_jac_kwargs that optimagic supports for compatibility "
                "with scipy. Do not use args in conjunction with any of the other "
                "arguments."
            )
            raise AliasError(msg)
        else:
            kwargs = get_kwargs_from_args(args, fun, offset=1)
            fun_kwargs, jac_kwargs, fun_and_jac_kwargs = kwargs, kwargs, kwargs

    # jac is not an alias but we need to handle the case where `jac=True`, i.e. fun is
    # actually fun_and_jac. This is not recommended in optimagic because then optimizers
    # cannot evaluate fun in isolation but we can easily support it for compatibility.
    if jac is True:
        jac = None
        if fun_and_jac is None:
            fun_and_jac = fun
            fun = split_fun_and_jac(fun_and_jac, target="fun")

    bounds = pre_process_bounds(bounds)

    # ==================================================================================
    # Handle scipy arguments that are not yet implemented
    # ==================================================================================

    if hess is not None:
        msg = (
            "The hess argument is not yet supported in optimagic. Creat an issue on "
            "https://github.com/OpenSourceEconomics/optimagic/ if you have urgent need "
            "for this feature."
        )
        raise NotImplementedError(msg)

    if hessp is not None:
        msg = (
            "The hessp argument is not yet supported in optimagic. Creat an issue on "
            "https://github.com/OpenSourceEconomics/optimagic/ if you have urgent need "
            "for this feature."
        )
        raise NotImplementedError(msg)

    if callback is not None:
        msg = (
            "The callback argument is not yet supported in optimagic. Creat an issue "
            "on https://github.com/OpenSourceEconomics/optimagic/ if you have urgent "
            "need for this feature."
        )
        raise NotImplementedError(msg)

    # ==================================================================================
    # Handle scipy arguments that will never be supported
    # ==================================================================================

    if options is not None:
        # TODO: Add link to a how-to guide or tutorial for this
        msg = (
            "The options argument is not supported in optimagic. Please use the "
            "algo_options argument instead."
        )
        raise NotImplementedError(msg)

    if tol is not None:
        # TODO: Add link to a how-to guide or tutorial for this
        msg = (
            "The tol argument is not supported in optimagic. Please use "
            "algo_options or configured algorithms instead to set convergence criteria "
            "for your optimizer."
        )
        raise NotImplementedError(msg)

    # ==================================================================================
    # Set default values and check options
    # ==================================================================================
    fun_kwargs = _setdefault(fun_kwargs, {})
    constraints = _setdefault(constraints, [])
    algo_options = _setdefault(algo_options, {})
    jac_kwargs = _setdefault(jac_kwargs, {})
    fun_and_jac_kwargs = _setdefault(fun_and_jac_kwargs, {})
    numdiff_options = _setdefault(numdiff_options, {})
    log_options = _setdefault(log_options, {})
    scaling_options = _setdefault(scaling_options, {})
    error_penalty = _setdefault(error_penalty, {})
    multistart_options = _setdefault(multistart_options, {})
    if logging:
        logging = Path(logging)

    if not skip_checks:
        check_optimize_kwargs(
            direction=direction,
            criterion=fun,
            criterion_kwargs=fun_kwargs,
            params=params,
            algorithm=algorithm,
            constraints=constraints,
            algo_options=algo_options,
            derivative=jac,
            derivative_kwargs=jac_kwargs,
            criterion_and_derivative=fun_and_jac,
            criterion_and_derivative_kwargs=fun_and_jac_kwargs,
            numdiff_options=numdiff_options,
            logging=logging,
            log_options=log_options,
            error_handling=error_handling,
            error_penalty=error_penalty,
            scaling=scaling,
            scaling_options=scaling_options,
            multistart=multistart,
            multistart_options=multistart_options,
        )
    # ==================================================================================
    # Get the algorithm info
    # ==================================================================================
    raw_algo, algo_info = process_user_algorithm(algorithm)

    algo_kwargs = set(algo_info.arguments)

    if algo_info.primary_criterion_entry == "root_contributions":
        if direction == "maximize":
            msg = (
                "Optimizers that exploit a least squares structure like {} can only be "
                "used for minimization."
            )
            raise ValueError(msg.format(algo_info.name))

    # ==================================================================================
    # Split constraints into nonlinear and reparametrization parts
    # ==================================================================================
    if isinstance(constraints, dict):
        constraints = [constraints]

    nonlinear_constraints = [c for c in constraints if c["type"] == "nonlinear"]

    if nonlinear_constraints and "nonlinear_constraints" not in algo_kwargs:
        raise ValueError(
            f"Algorithm {algo_info.name} does not support nonlinear constraints."
        )

    # the following constraints will be handled via reparametrization
    constraints = [c for c in constraints if c["type"] != "nonlinear"]

    # ==================================================================================
    # prepare logging
    # ==================================================================================
    if logging:
        problem_data = {
            "direction": direction,
            # "criterion"-criterion,
            "criterion_kwargs": fun_kwargs,
            "algorithm": algorithm,
            "constraints": constraints,
            "algo_options": algo_options,
            # "derivative"-derivative,
            "derivative_kwargs": jac_kwargs,
            # "criterion_and_derivative"-criterion_and_derivative,
            "criterion_and_derivative_kwargs": fun_and_jac_kwargs,
            "numdiff_options": numdiff_options,
            "log_options": log_options,
            "error_handling": error_handling,
            "error_penalty": error_penalty,
            "params": params,
        }

    # ==================================================================================
    # partial the kwargs into corresponding functions
    # ==================================================================================
    fun = process_func_of_params(
        func=fun,
        kwargs=fun_kwargs,
        name="criterion",
        skip_checks=skip_checks,
    )
    if isinstance(jac, dict):
        jac = jac.get(algo_info.primary_criterion_entry)
    if jac is not None:
        jac = process_func_of_params(
            func=jac,
            kwargs=jac_kwargs,
            name="derivative",
            skip_checks=skip_checks,
        )
    if isinstance(fun_and_jac, dict):
        fun_and_jac = fun_and_jac.get(algo_info.primary_criterion_entry)

    if fun_and_jac is not None:
        fun_and_jac = process_func_of_params(
            func=fun_and_jac,
            kwargs=fun_and_jac_kwargs,
            name="criterion_and_derivative",
            skip_checks=skip_checks,
        )

    # ==================================================================================
    # Do first evaluation of user provided functions
    # ==================================================================================
    try:
        first_crit_eval = fun(params)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        msg = "Error while evaluating criterion at start params."
        raise InvalidFunctionError(msg) from e

    # do first derivative evaluation (if given)
    if jac is not None:
        try:
            first_deriv_eval = jac(params)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            msg = "Error while evaluating derivative at start params."
            raise InvalidFunctionError(msg) from e

    if fun_and_jac is not None:
        try:
            first_crit_and_deriv_eval = fun_and_jac(params)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            msg = "Error while evaluating criterion_and_derivative at start params."
            raise InvalidFunctionError(msg) from e

    if jac is not None:
        used_deriv = first_deriv_eval
    elif fun_and_jac is not None:
        used_deriv = first_crit_and_deriv_eval[1]
    else:
        used_deriv = None

    # ==================================================================================
    # Get the converter (for tree flattening, constraints and scaling)
    # ==================================================================================
    converter, internal_params = get_converter(
        params=params,
        constraints=constraints,
        bounds=bounds,
        func_eval=first_crit_eval,
        primary_key=algo_info.primary_criterion_entry,
        scaling=scaling,
        scaling_options=scaling_options,
        derivative_eval=used_deriv,
        add_soft_bounds=multistart,
    )

    # ==================================================================================
    # initialize the log database
    # ==================================================================================
    if logging:
        problem_data["free_mask"] = internal_params.free_mask
        database = _create_and_initialize_database(logging, log_options, problem_data)
    else:
        database = None

    # ==================================================================================
    # Do some things that require internal parameters or bounds
    # ==================================================================================

    if converter.has_transforming_constraints and multistart:
        raise NotImplementedError(
            "multistart optimizations are not yet compatible with transforming "
            "constraints."
        )

    numdiff_options = _fill_numdiff_options_with_defaults(
        numdiff_options=numdiff_options,
        lower_bounds=internal_params.lower_bounds,
        upper_bounds=internal_params.upper_bounds,
    )

    # get error penalty function
    error_penalty_func = get_error_penalty_function(
        error_handling=error_handling,
        start_x=internal_params.values,
        start_criterion=converter.func_to_internal(first_crit_eval),
        error_penalty=error_penalty,
        primary_key=algo_info.primary_criterion_entry,
        direction=direction,
    )

    # process nonlinear constraints:
    internal_constraints = process_nonlinear_constraints(
        nonlinear_constraints=nonlinear_constraints,
        params=params,
        converter=converter,
        numdiff_options=numdiff_options,
        skip_checks=skip_checks,
    )

    x = internal_params.values
    # ==================================================================================
    # get the internal algorithm
    # ==================================================================================
    internal_algorithm = get_final_algorithm(
        raw_algorithm=raw_algo,
        algo_info=algo_info,
        valid_kwargs=algo_kwargs,
        lower_bounds=internal_params.lower_bounds,
        upper_bounds=internal_params.upper_bounds,
        nonlinear_constraints=internal_constraints,
        algo_options=algo_options,
        logging=logging,
        database=database,
        collect_history=collect_history,
    )
    # ==================================================================================
    # partial arguments into the internal_criterion_and_derivative_template
    # ==================================================================================
    to_partial = {
        "direction": direction,
        "criterion": fun,
        "converter": converter,
        "derivative": jac,
        "criterion_and_derivative": fun_and_jac,
        "numdiff_options": numdiff_options,
        "logging": logging,
        "database": database,
        "algo_info": algo_info,
        "error_handling": error_handling,
        "error_penalty_func": error_penalty_func,
    }

    internal_criterion_and_derivative = functools.partial(
        internal_criterion_and_derivative_template,
        **to_partial,
    )

    problem_functions = {}
    for task in ["criterion", "derivative", "criterion_and_derivative"]:
        if task in algo_kwargs:
            problem_functions[task] = functools.partial(
                internal_criterion_and_derivative,
                task=task,
            )

    # ==================================================================================
    # Do actual optimization
    # ==================================================================================
    if not multistart:
        steps = [{"type": "optimization", "name": "optimization"}]

        step_ids = log_scheduled_steps_and_get_ids(
            steps=steps,
            logging=logging,
            database=database,
        )

        raw_res = internal_algorithm(**problem_functions, x=x, step_id=step_ids[0])
    else:
        multistart_options = _fill_multistart_options_with_defaults(
            options=multistart_options,
            params=params,
            x=x,
            params_to_internal=converter.params_to_internal,
        )

        raw_res = run_multistart_optimization(
            local_algorithm=internal_algorithm,
            primary_key=algo_info.primary_criterion_entry,
            problem_functions=problem_functions,
            x=x,
            lower_sampling_bounds=internal_params.soft_lower_bounds,
            upper_sampling_bounds=internal_params.soft_upper_bounds,
            options=multistart_options,
            logging=logging,
            database=database,
            error_handling=error_handling,
        )

    # ==================================================================================
    # Process the result
    # ==================================================================================

    _scalar_start_criterion = aggregate_func_output_to_value(
        converter.func_to_internal(first_crit_eval),
        algo_info.primary_criterion_entry,
    )

    fixed_result_kwargs = {
        "start_fun": _scalar_start_criterion,
        "start_params": params,
        "algorithm": algo_info.name,
        "direction": direction,
        "n_free": internal_params.free_mask.sum(),
    }

    res = process_internal_optimizer_result(
        raw_res,
        converter=converter,
        primary_key=algo_info.primary_criterion_entry,
        fixed_kwargs=fixed_result_kwargs,
        skip_checks=skip_checks,
    )

    return res


def _create_and_initialize_database(logging, log_options, problem_data):
    """Create and initialize to sqlite database for logging."""
    path = Path(logging)
    fast_logging = log_options.get("fast_logging", False)
    if_table_exists = log_options.get("if_table_exists", "extend")
    if_database_exists = log_options.get("if_database_exists", "extend")

    if "if_exists" in log_options and "if_table_exists" not in log_options:
        warnings.warn("The log_option 'if_exists' was renamed to 'if_table_exists'.")

    if logging.exists():
        if if_database_exists == "raise":
            raise FileExistsError(
                f"The database {logging} already exists and the log_option "
                "'if_database_exists' is set to 'raise'"
            )
        elif if_database_exists == "replace":
            logging.unlink()

    database = load_database(path_or_database=path, fast_logging=fast_logging)

    # create the optimization_iterations table
    make_optimization_iteration_table(
        database=database,
        if_exists=if_table_exists,
    )

    # create and initialize the steps table; This is alway extended if it exists.
    make_steps_table(database, if_exists=if_table_exists)

    # create_and_initialize the optimization_problem table
    make_optimization_problem_table(database, if_exists=if_table_exists)

    not_saved = [
        "criterion",
        "criterion_kwargs",
        "constraints",
        "derivative",
        "derivative_kwargs",
        "criterion_and_derivative",
        "criterion_and_derivative_kwargs",
    ]
    problem_data = {
        key: val for key, val in problem_data.items() if key not in not_saved
    }

    append_row(problem_data, "optimization_problem", database=database)

    return database


def _fill_numdiff_options_with_defaults(numdiff_options, lower_bounds, upper_bounds):
    """Fill options for numerical derivatives during optimization with defaults."""
    method = numdiff_options.get("method", "forward")
    default_error_handling = "raise" if method == "central" else "raise_strict"

    relevant = {
        "method",
        "n_steps",
        "base_steps",
        "scaling_factor",
        "lower_bounds",
        "upper_bounds",
        "step_ratio",
        "min_steps",
        "n_cores",
        "error_handling",
        "batch_evaluator",
    }

    ignored = [option for option in numdiff_options if option not in relevant]

    if ignored:
        raise InvalidKwargsError(
            f"The following numdiff_options are not allowed:\n\n{ignored}"
        )

    numdiff_options = {
        key: val for key, val in numdiff_options.items() if key in relevant
    }

    # only define the ones that deviate from the normal defaults
    default_numdiff_options = {
        "method": "forward",
        "bounds": Bounds(lower=lower_bounds, upper=upper_bounds),
        "error_handling": default_error_handling,
        "return_info": False,
    }

    numdiff_options = {**default_numdiff_options, **numdiff_options}
    return numdiff_options


def _setdefault(candidate, default):
    out = default if candidate is None else candidate
    return out


def _fill_multistart_options_with_defaults(options, params, x, params_to_internal):
    """Fill options for multistart optimization with defaults."""
    defaults = {
        "sample": None,
        "n_samples": 10 * len(x),
        "share_optimizations": 0.1,
        "sampling_distribution": "uniform",
        "sampling_method": "sobol" if len(x) <= 200 else "random",
        "mixing_weight_method": "tiktak",
        "mixing_weight_bounds": (0.1, 0.995),
        "convergence_relative_params_tolerance": 0.01,
        "convergence_max_discoveries": 2,
        "n_cores": 1,
        "batch_evaluator": "joblib",
        "seed": None,
        "exploration_error_handling": "continue",
        "optimization_error_handling": "continue",
    }

    options = {k.replace(".", "_"): v for k, v in options.items()}
    out = {**defaults, **options}

    if "batch_size" not in out:
        out["batch_size"] = out["n_cores"]
    else:
        if out["batch_size"] < out["n_cores"]:
            raise ValueError("batch_size must be at least as large as n_cores.")

    out["batch_evaluator"] = process_batch_evaluator(out["batch_evaluator"])

    if isinstance(out["mixing_weight_method"], str):
        out["mixing_weight_method"] = WEIGHT_FUNCTIONS[out["mixing_weight_method"]]

    if out["sample"] is not None:
        out["sample"] = process_multistart_sample(
            out["sample"], params, params_to_internal
        )
        out["n_samples"] = len(out["sample"])

    out["n_optimizations"] = max(1, int(out["n_samples"] * out["share_optimizations"]))
    del out["share_optimizations"]

    return out
