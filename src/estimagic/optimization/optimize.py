import functools
import warnings
from pathlib import Path

from estimagic.batch_evaluators import process_batch_evaluator
from estimagic.exceptions import InvalidFunctionError
from estimagic.exceptions import InvalidKwargsError
from estimagic.logging.database_utilities import append_row
from estimagic.logging.database_utilities import load_database
from estimagic.logging.database_utilities import make_optimization_iteration_table
from estimagic.logging.database_utilities import make_optimization_problem_table
from estimagic.logging.database_utilities import make_steps_table
from estimagic.optimization.check_arguments import check_optimize_kwargs
from estimagic.optimization.error_penalty import get_error_penalty_function
from estimagic.optimization.get_algorithm import get_final_algorithm
from estimagic.optimization.get_algorithm import process_user_algorithm
from estimagic.optimization.internal_criterion_template import (
    internal_criterion_and_derivative_template,
)
from estimagic.optimization.optimization_logging import log_scheduled_steps_and_get_ids
from estimagic.optimization.process_multistart_sample import process_multistart_sample
from estimagic.optimization.process_results import process_internal_optimizer_result
from estimagic.optimization.tiktak import run_multistart_optimization
from estimagic.optimization.tiktak import WEIGHT_FUNCTIONS
from estimagic.parameters.conversion import aggregate_func_output_to_value
from estimagic.parameters.conversion import get_converter
from estimagic.parameters.nonlinear_constraints import process_nonlinear_constraints
from estimagic.process_user_function import process_func_of_params


def maximize(
    criterion,
    params,
    algorithm,
    *,
    lower_bounds=None,
    upper_bounds=None,
    soft_lower_bounds=None,
    soft_upper_bounds=None,
    criterion_kwargs=None,
    constraints=None,
    algo_options=None,
    derivative=None,
    derivative_kwargs=None,
    criterion_and_derivative=None,
    criterion_and_derivative_kwargs=None,
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
):
    """Maximize criterion using algorithm subject to constraints.

    Args:
        criterion (callable): A function that takes a params as first argument and
            returns a scalar (if only scalar algorithms will be used) or a dictionary
            that contains at the entries "value" (a scalar float), "contributions" (a
            pytree containing the summands that make up the criterion value) or
            "root_contributions" (a pytree containing the residuals of a least-squares
            problem) and any number of additional entries. The additional dict entries
            will be stored in a database if logging is used.
        params (pandas): A pytree containing the parameters with respect to which the
            criterion is optimized. Examples are a numpy array, a pandas Series,
            a DataFrame with "value" column, a float and any kind of (nested) dictionary
            or list containing these elements. See :ref:`params` for examples.
        algorithm (str or callable): Specifies the optimization algorithm. For built-in
            algorithms this is a string with the name of the algorithm. Otherwise it can
            be a callable with the estimagic algorithm interface. See :ref:`algorithms`.
        lower_bounds (pytree): A pytree with the same structure as params with lower
            bounds for the parameters. Can be ``-np.inf`` for parameters with no lower
            bound.
        upper_bounds (pytree): As lower_bounds. Can be ``np.inf`` for parameters with
            no upper bound.
        soft_lower_bounds (pytree): As lower bounds but the bounds are not imposed
            during optimization and just used to sample start values if multistart
            optimization is performed.
        soft_upper_bounds (pytree): As soft_lower_bounds.
        criterion_kwargs (dict): Additional keyword arguments for criterion
        constraints (list, dict): List with constraint dictionaries or single dict.
            See :ref:`constraints`.
        algo_options (dict): Algorithm specific configuration of the optimization. See
            :ref:`list_of_algorithms` for supported options of each algorithm.
        derivative (callable): Function that calculates the first derivative
            of criterion. For most algorithm, this is the gradient of the scalar
            output (or "value" entry of the dict). However some algorithms (e.g. bhhh)
            require the jacobian of the "contributions" entry of the dict. You will get
            an error if you provide the wrong type of derivative.
        derivative_kwargs (dict): Additional keyword arguments for derivative.
        criterion_and_derivative (callable): Function that returns criterion
            and derivative as a tuple. This can be used to exploit synergies in the
            evaluation of both functions. The fist element of the tuple has to be
            exactly the same as the output of criterion. The second has to be exactly
            the same as the output of derivative.
        criterion_and_derivative_kwargs (dict): Additional keyword arguments for
            criterion and derivative.
        numdiff_options (dict): Keyword arguments for the calculation of numerical
            derivatives. See :ref:`first_derivative` for details. Note that the default
            method is changed to "forward" for speed reasons.
        logging (pathlib.Path, str or False): Path to sqlite3 file (which typically has
            the file extension ``.db``. If the file does not exist, it will be created.
            When doing parallel optimizations and logging is provided, you have to
            provide a different path for each optimization you are running. You can
            disable logging completely by setting it to False, but we highly recommend
            not to do so. The dashboard can only be used when logging is used.
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
        error_handling (str): Either "raise" or "continue". Note that "continue" does
            not absolutely guarantee that no error is raised but we try to handle as
            many errors as possible in that case without aborting the optimization.
        error_penalty (dict): Dict with the entries "constant" (float) and "slope"
            (float). If the criterion or gradient raise an error and error_handling is
            "continue", return ``constant + slope * norm(params - start_params)`` where
            ``norm`` is the euclidean distance as criterion value and adjust the
            derivative accordingly. This is meant to guide the optimizer back into a
            valid region of parameter space (in direction of the start parameters).
            Note that the constant has to be high enough to ensure that the penalty is
            actually a bad function value. The default constant is f0 + abs(f0) + 100
            for minimizations and f0 - abs(f0) - 100 for maximizations, where
            f0 is the criterion value at start parameters. The default slope is 0.1.
        scaling (bool): If True, the parameter vector is rescaled internally for
            better performance with scale sensitive optimizers.
        scaling_options (dict or None): Options to configure the internal scaling ot
            the parameter vector. See :ref:`scaling` for details and recommendations.
        multistart (bool): Whether to do the optimization from multiple starting points.
            Requires the params to have the columns ``"soft_lower_bound"`` and
            ``"soft_upper_bounds"`` with finite values for all parameters, unless
            the standard bounds are already finite for all parameters.
        multistart_options (dict): Options to configure the optimization from multiple
            starting values. The dictionary has the following entries
            (all of which are optional):
            - n_samples (int): Number of sampled points on which to do one function
            evaluation. Default is 10 * n_params.
            - sample (pandas.DataFrame or numpy.ndarray) A user definde sample.
            If this is provided, n_samples, sampling_method and sampling_distribution
            are not used.
            - share_optimizations (float): Share of sampled points that is used to
            construct a starting point for a local optimization. Default 0.1.
            - sampling_distribution (str): One rof "uniform", "triangle". Default is
            "uniform" as in the original tiktak algorithm.
            - sampling_method (str): One of "random", "sobol", "halton", "hammersley",
            "korobov", "latin_hypercube" or a numpy array or DataFrame with custom
            points. Default is sobol for problems with up to 30 parameters and random
            for problems with more than 30 parameters.
            - mixing_weight_method (str or callable): Specifies how much weight is put
            on the currently best point when calculating a new starting point for a
            local optimization out of the currently best point and the next random
            starting point. Either "tiktak" or "linear" or a callable that takes the
            arguments ``iteration``, ``n_iterations``, ``min_weight``, ``max_weight``.
            Default "tiktak".
            - mixing_weight_bounds (tuple): A tuple consisting of a lower and upper
            bound on mixing weights. Default (0.1, 0.995).
            - convergence_max_discoveries (int): The multistart optimization converges
            if the currently best local optimum has been discovered independently in
            ``convergence_max_discoveries`` many local optimizations. Default 2.
            - convergence.relative_params_tolerance (float): Determines the maximum
            relative distance two parameter vectors can have to be considered equal
            for convergence purposes.
            - n_cores (int): Number cores used to evaluate the criterion function in
            parallel during exploration stages and number of parallel local
            optimization in optimization stages. Default 1.
            - batch_evaluator (str or callable): See :ref:`batch_evaluators` for
            details. Default "joblib".
            - batch_size (int): If n_cores is larger than one, several starting points
            for local optimizations are created with the same weight and from the same
            currently best point. The ``batch_size`` argument is a way to reproduce
            this behavior on a small machine where less cores are available. By
            default the batch_size is equal to ``n_cores``. It can never be smaller
            than ``n_cores``.
            - seed (int): Random seed for the creation of starting values. Default None.
            - exploration_error_handling (str): One of "raise" or "continue". Default
            is continue, which means that failed function evaluations are simply
            discarded from the sample.
            - optimization_error_handling (str): One of "raise" or "continue". Default
            is continue, which means that failed optimizations are simply discarded.
        collect_history (bool): Whether the history of parameters and criterion values
            should be collected and returned as part of the result. Default True.
        skip_checks (bool): Whether checks on the inputs are skipped. This makes the
            optimization faster, especially for very fast criterion functions. Default
            False.

    """
    return _optimize(
        direction="maximize",
        criterion=criterion,
        params=params,
        algorithm=algorithm,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        soft_lower_bounds=soft_lower_bounds,
        soft_upper_bounds=soft_upper_bounds,
        criterion_kwargs=criterion_kwargs,
        constraints=constraints,
        algo_options=algo_options,
        derivative=derivative,
        derivative_kwargs=derivative_kwargs,
        criterion_and_derivative=criterion_and_derivative,
        criterion_and_derivative_kwargs=criterion_and_derivative_kwargs,
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
    )


def minimize(
    criterion,
    params,
    algorithm,
    *,
    lower_bounds=None,
    upper_bounds=None,
    soft_lower_bounds=None,
    soft_upper_bounds=None,
    criterion_kwargs=None,
    constraints=None,
    algo_options=None,
    derivative=None,
    derivative_kwargs=None,
    criterion_and_derivative=None,
    criterion_and_derivative_kwargs=None,
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
):
    """Minimize criterion using algorithm subject to constraints.

    Args:
        criterion (callable): A function that takes a params as first argument and
            returns a scalar (if only scalar algorithms will be used) or a dictionary
            that contains at the entries "value" (a scalar float), "contributions" (a
            pytree containing the summands that make up the criterion value) or
            "root_contributions" (a pytree containing the residuals of a least-squares
            problem) and any number of additional entries. The additional dict entries
            will be stored in a database if logging is used.
        params (pandas): A pytree containing the parameters with respect to which the
            criterion is optimized. Examples are a numpy array, a pandas Series,
            a DataFrame with "value" column, a float and any kind of (nested) dictionary
            or list containing these elements. See :ref:`params` for examples.
        algorithm (str or callable): Specifies the optimization algorithm. For built-in
            algorithms this is a string with the name of the algorithm. Otherwise it can
            be a callable with the estimagic algorithm interface. See :ref:`algorithms`.
        lower_bounds (pytree): A pytree with the same structure as params with lower
            bounds for the parameters. Can be ``-np.inf`` for parameters with no lower
            bound.
        upper_bounds (pytree): As lower_bounds. Can be ``np.inf`` for parameters with
            no upper bound.
        soft_lower_bounds (pytree): As lower bounds but the bounds are not imposed
            during optimization and just used to sample start values if multistart
            optimization is performed.
        soft_upper_bounds (pytree): As soft_lower_bounds.
        criterion_kwargs (dict): Additional keyword arguments for criterion
        constraints (list, dict): List with constraint dictionaries or single dict.
            See :ref:`constraints`.
        algo_options (dict): Algorithm specific configuration of the optimization. See
            :ref:`list_of_algorithms` for supported options of each algorithm.
        derivative (callable): Function that calculates the first derivative
            of criterion. For most algorithm, this is the gradient of the scalar
            output (or "value" entry of the dict). However some algorithms (e.g. bhhh)
            require the jacobian of the "contributions" entry of the dict. You will get
            an error if you provide the wrong type of derivative.
        derivative_kwargs (dict): Additional keyword arguments for derivative.
        criterion_and_derivative (callable): Function that returns criterion
            and derivative as a tuple. This can be used to exploit synergies in the
            evaluation of both functions. The fist element of the tuple has to be
            exactly the same as the output of criterion. The second has to be exactly
            the same as the output of derivative.
        criterion_and_derivative_kwargs (dict): Additional keyword arguments for
            criterion and derivative.
        numdiff_options (dict): Keyword arguments for the calculation of numerical
            derivatives. See :ref:`first_derivative` for details. Note that the default
            method is changed to "forward" for speed reasons.
        logging (pathlib.Path, str or False): Path to sqlite3 file (which typically has
            the file extension ``.db``. If the file does not exist, it will be created.
            When doing parallel optimizations and logging is provided, you have to
            provide a different path for each optimization you are running. You can
            disable logging completely by setting it to False, but we highly recommend
            not to do so. The dashboard can only be used when logging is used.
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
        error_handling (str): Either "raise" or "continue". Note that "continue" does
            not absolutely guarantee that no error is raised but we try to handle as
            many errors as possible in that case without aborting the optimization.
        error_penalty (dict): Dict with the entries "constant" (float) and "slope"
            (float). If the criterion or gradient raise an error and error_handling is
            "continue", return ``constant + slope * norm(params - start_params)`` where
            ``norm`` is the euclidean distance as criterion value and adjust the
            derivative accordingly. This is meant to guide the optimizer back into a
            valid region of parameter space (in direction of the start parameters).
            Note that the constant has to be high enough to ensure that the penalty is
            actually a bad function value. The default constant is f0 + abs(f0) + 100
            for minimizations and f0 - abs(f0) - 100 for maximizations, where
            f0 is the criterion value at start parameters. The default slope is 0.1.
        scaling (bool): If True, the parameter vector is rescaled internally for
            better performance with scale sensitive optimizers.
        scaling_options (dict or None): Options to configure the internal scaling ot
            the parameter vector. See :ref:`scaling` for details and recommendations.
        multistart (bool): Whether to do the optimization from multiple starting points.
            Requires the params to have the columns ``"soft_lower_bound"`` and
            ``"soft_upper_bounds"`` with finite values for all parameters, unless
            the standard bounds are already finite for all parameters.
        multistart_options (dict): Options to configure the optimization from multiple
            starting values. The dictionary has the following entries
            (all of which are optional):
            - n_samples (int): Number of sampled points on which to do one function
            evaluation. Default is 10 * n_params.
            - sample (pandas.DataFrame or numpy.ndarray) A user definde sample.
            If this is provided, n_samples, sampling_method and sampling_distribution
            are not used.
            - share_optimizations (float): Share of sampled points that is used to
            construct a starting point for a local optimization. Default 0.1.
            - sampling_distribution (str): One rof "uniform", "triangle". Default is
            "uniform" as in the original tiktak algorithm.
            - sampling_method (str): One of "random", "sobol", "halton", "hammersley",
            "korobov", "latin_hypercube" or a numpy array or DataFrame with custom
            points. Default is sobol for problems with up to 30 parameters and random
            for problems with more than 30 parameters.
            - mixing_weight_method (str or callable): Specifies how much weight is put
            on the currently best point when calculating a new starting point for a
            local optimization out of the currently best point and the next random
            starting point. Either "tiktak" or "linear" or a callable that takes the
            arguments ``iteration``, ``n_iterations``, ``min_weight``, ``max_weight``.
            Default "tiktak".
            - mixing_weight_bounds (tuple): A tuple consisting of a lower and upper
            bound on mixing weights. Default (0.1, 0.995).
            - convergence_max_discoveries (int): The multistart optimization converges
            if the currently best local optimum has been discovered independently in
            ``convergence_max_discoveries`` many local optimizations. Default 2.
            - convergence.relative_params_tolerance (float): Determines the maximum
            relative distance two parameter vectors can have to be considered equal
            for convergence purposes.
            - n_cores (int): Number cores used to evaluate the criterion function in
            parallel during exploration stages and number of parallel local
            optimization in optimization stages. Default 1.
            - batch_evaluator (str or callable): See :ref:`batch_evaluators` for
            details. Default "joblib".
            - batch_size (int): If n_cores is larger than one, several starting points
            for local optimizations are created with the same weight and from the same
            currently best point. The ``batch_size`` argument is a way to reproduce
            this behavior on a small machine where less cores are available. By
            default the batch_size is equal to ``n_cores``. It can never be smaller
            than ``n_cores``.
            - seed (int): Random seed for the creation of starting values. Default None.
            - exploration_error_handling (str): One of "raise" or "continue". Default
            is continue, which means that failed function evaluations are simply
            discarded from the sample.
            - optimization_error_handling (str): One of "raise" or "continue". Default
            is continue, which means that failed optimizations are simply discarded.
        collect_history (bool): Whether the history of parameters and criterion values
            should be collected and returned as part of the result. Default True.
        skip_checks (bool): Whether checks on the inputs are skipped. This makes the
            optimization faster, especially for very fast criterion functions. Default
            False.

    """
    return _optimize(
        direction="minimize",
        criterion=criterion,
        params=params,
        algorithm=algorithm,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        soft_lower_bounds=soft_lower_bounds,
        soft_upper_bounds=soft_upper_bounds,
        criterion_kwargs=criterion_kwargs,
        constraints=constraints,
        algo_options=algo_options,
        derivative=derivative,
        derivative_kwargs=derivative_kwargs,
        criterion_and_derivative=criterion_and_derivative,
        criterion_and_derivative_kwargs=criterion_and_derivative_kwargs,
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
    )


def _optimize(
    direction,
    criterion,
    params,
    algorithm,
    *,
    lower_bounds=None,
    upper_bounds=None,
    soft_lower_bounds=None,
    soft_upper_bounds=None,
    criterion_kwargs,
    constraints,
    algo_options,
    derivative,
    derivative_kwargs,
    criterion_and_derivative,
    criterion_and_derivative_kwargs,
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
):
    """Minimize or maximize criterion using algorithm subject to constraints.

    Arguments are the same as in maximize and minimize, with an additional direction
    argument. Direction is a string that can take the values "maximize" and "minimize".

    Returns are the same as in maximize and minimize.

    """
    # ==================================================================================
    # Set default values and check options
    # ==================================================================================
    criterion_kwargs = _setdefault(criterion_kwargs, {})
    constraints = _setdefault(constraints, [])
    algo_options = _setdefault(algo_options, {})
    derivative_kwargs = _setdefault(derivative_kwargs, {})
    criterion_and_derivative_kwargs = _setdefault(criterion_and_derivative_kwargs, {})
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
            criterion=criterion,
            criterion_kwargs=criterion_kwargs,
            params=params,
            algorithm=algorithm,
            constraints=constraints,
            algo_options=algo_options,
            derivative=derivative,
            derivative_kwargs=derivative_kwargs,
            criterion_and_derivative=criterion_and_derivative,
            criterion_and_derivative_kwargs=criterion_and_derivative_kwargs,
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
            "criterion_kwargs": criterion_kwargs,
            "algorithm": algorithm,
            "constraints": constraints,
            "algo_options": algo_options,
            # "derivative"-derivative,
            "derivative_kwargs": derivative_kwargs,
            # "criterion_and_derivative"-criterion_and_derivative,
            "criterion_and_derivative_kwargs": criterion_and_derivative_kwargs,
            "numdiff_options": numdiff_options,
            "log_options": log_options,
            "error_handling": error_handling,
            "error_penalty": error_penalty,
            "params": params,
        }

    # ==================================================================================
    # partial the kwargs into corresponding functions
    # ==================================================================================
    criterion = process_func_of_params(
        func=criterion,
        kwargs=criterion_kwargs,
        name="criterion",
        skip_checks=skip_checks,
    )
    if isinstance(derivative, dict):
        derivative = derivative.get(algo_info.primary_criterion_entry)
    if derivative is not None:
        derivative = process_func_of_params(
            func=derivative,
            kwargs=derivative_kwargs,
            name="derivative",
            skip_checks=skip_checks,
        )
    if isinstance(criterion_and_derivative, dict):
        criterion_and_derivative = criterion_and_derivative.get(
            algo_info.primary_criterion_entry
        )

    if criterion_and_derivative is not None:
        criterion_and_derivative = process_func_of_params(
            func=criterion_and_derivative,
            kwargs=criterion_and_derivative_kwargs,
            name="criterion_and_derivative",
            skip_checks=skip_checks,
        )

    # ==================================================================================
    # Do first evaluation of user provided functions
    # ==================================================================================
    try:
        first_crit_eval = criterion(params)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        msg = "Error while evaluating criterion at start params."
        raise InvalidFunctionError(msg) from e

    # do first derivative evaluation (if given)
    if derivative is not None:
        try:
            first_deriv_eval = derivative(params)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            msg = "Error while evaluating derivative at start params."
            raise InvalidFunctionError(msg) from e

    if criterion_and_derivative is not None:
        try:
            first_crit_and_deriv_eval = criterion_and_derivative(params)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            msg = "Error while evaluating criterion_and_derivative at start params."
            raise InvalidFunctionError(msg) from e

    if derivative is not None:
        used_deriv = first_deriv_eval
    elif criterion_and_derivative is not None:
        used_deriv = first_crit_and_deriv_eval[1]
    else:
        used_deriv = None

    # ==================================================================================
    # Get the converter (for tree flattening, constraints and scaling)
    # ==================================================================================
    converter, internal_params = get_converter(
        func=criterion,
        params=params,
        constraints=constraints,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        func_eval=first_crit_eval,
        primary_key=algo_info.primary_criterion_entry,
        scaling=scaling,
        scaling_options=scaling_options,
        derivative_eval=used_deriv,
        soft_lower_bounds=soft_lower_bounds,
        soft_upper_bounds=soft_upper_bounds,
        add_soft_bounds=multistart,
    )

    # ==================================================================================
    # initialize the log database
    # ==================================================================================
    if logging:
        problem_data["free_mask"] = internal_params.free_mask
        database = _create_and_initialize_database(logging, log_options, problem_data)
        db_kwargs = {
            "database": database,
            "path": logging,
            "fast_logging": log_options.get("fast_logging", False),
        }
    else:
        db_kwargs = {"database": None, "path": None, "fast_logging": False}

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
        db_kwargs=db_kwargs,
        collect_history=collect_history,
    )
    # ==================================================================================
    # partial arguments into the internal_criterion_and_derivative_template
    # ==================================================================================
    to_partial = {
        "direction": direction,
        "criterion": criterion,
        "converter": converter,
        "derivative": derivative,
        "criterion_and_derivative": criterion_and_derivative,
        "numdiff_options": numdiff_options,
        "logging": logging,
        "db_kwargs": db_kwargs,
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
            db_kwargs=db_kwargs,
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
            db_kwargs=db_kwargs,
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
        "start_criterion": _scalar_start_criterion,
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

    database = load_database(path=path, fast_logging=fast_logging)

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

    append_row(problem_data, "optimization_problem", database, path, fast_logging)

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
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
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
