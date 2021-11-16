import functools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from estimagic import batch_evaluators as be
from estimagic.config import CRITERION_PENALTY_CONSTANT
from estimagic.config import CRITERION_PENALTY_SLOPE
from estimagic.logging.database_utilities import append_row
from estimagic.logging.database_utilities import load_database
from estimagic.logging.database_utilities import make_optimization_iteration_table
from estimagic.logging.database_utilities import make_optimization_problem_table
from estimagic.logging.database_utilities import make_steps_table
from estimagic.optimization.check_arguments import check_optimize_kwargs
from estimagic.optimization.get_algorithm import get_algorithm
from estimagic.optimization.internal_criterion_template import (
    internal_criterion_and_derivative_template,
)
from estimagic.optimization.optimization_logging import log_scheduled_steps_and_get_ids
from estimagic.optimization.process_results import process_internal_optimizer_result
from estimagic.optimization.scaling import calculate_scaling_factor_and_offset
from estimagic.optimization.tiktak import get_internal_sampling_bounds
from estimagic.optimization.tiktak import run_multistart_optimization
from estimagic.optimization.tiktak import WEIGHT_FUNCTIONS
from estimagic.parameters.parameter_conversion import get_derivative_conversion_function
from estimagic.parameters.parameter_conversion import get_internal_bounds
from estimagic.parameters.parameter_conversion import get_reparametrize_functions
from estimagic.parameters.parameter_preprocessing import add_default_bounds_to_params
from estimagic.parameters.parameter_preprocessing import check_params_are_valid
from estimagic.parameters.process_constraints import process_constraints
from estimagic.utilities import hash_array


def maximize(
    criterion,
    params,
    algorithm,
    *,
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
    cache_size=100,
    scaling=False,
    scaling_options=None,
    multistart=False,
    multistart_options=None,
):
    """Maximize criterion using algorithm subject to constraints.

    Args:
        criterion (callable): A function that takes a pandas DataFrame (see
            :ref:`params`) as first argument and returns one of the following:

            - scalar floating point or a :class:`numpy.ndarray` (depending on the
              algorithm)
            - a dictionary that contains at the entries "value" (a scalar float),
              "contributions" or "root_contributions" (depending on the algortihm) and
              any number of additional entries. The additional dict entries will be
              logged and (if supported) displayed in the dashboard. Check the
              documentation of your algorithm to see which entries or output type are
              required.
        params (pandas.DataFrame): A DataFrame with a column called "value" and optional
            additional columns. See :ref:`params` for detail.
        algorithm (str or callable): Specifies the optimization algorithm. For supported
            algorithms this is a string with the name of the algorithm. Otherwise it can
            be a callable with the estimagic algorithm interface. See :ref:`algorithms`.
        criterion_kwargs (dict): Additional keyword arguments for criterion
        constraints (list): List with constraint dictionaries.
            See .. _link: ../../docs/source/how_to_guides/how_to_use_constraints.ipynb
        algo_options (dict): Algorithm specific configuration of the optimization. See
            :ref:`list_of_algorithms` for supported options of each algorithm.
        derivative (callable, optional): Function that calculates the first derivative
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
        cache_size (int): Number of criterion and derivative evaluations that are cached
            in memory in case they are needed.
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
            - sampling_distribution (str): One of "uniform", "triangle". Default is
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

    """
    return _optimize(
        direction="maximize",
        criterion=criterion,
        params=params,
        algorithm=algorithm,
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
        cache_size=cache_size,
        scaling=scaling,
        scaling_options=scaling_options,
        multistart=multistart,
        multistart_options=multistart_options,
    )


def minimize(
    criterion,
    params,
    algorithm,
    *,
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
    cache_size=100,
    scaling=False,
    scaling_options=None,
    multistart=False,
    multistart_options=None,
):
    """Minimize criterion using algorithm subject to constraints.

    Args:
        criterion (Callable): A function that takes a pandas DataFrame (see
            :ref:`params`) as first argument and returns one of the following:
            - scalar floating point or a numpy array (depending on the algorithm)
            - a dictionary that contains at the entries "value" (a scalar float),
            "contributions" or "root_contributions" (depending on the algortihm) and
            any number of additional entries. The additional dict entries will be
            logged and (if supported) displayed in the dashboard. Check the
            documentation of your algorithm to see which entries or output type
            are required.
        params (pandas.DataFrame): A DataFrame with a column called "value" and optional
            additional columns. See :ref:`params` for detail.
        algorithm (str or callable): Specifies the optimization algorithm. For supported
            algorithms this is a string with the name of the algorithm. Otherwise it can
            be a callable with the estimagic algorithm interface. See :ref:`algorithms`.
        criterion_kwargs (dict): Additional keyword arguments for criterion
        constraints (list): List with constraint dictionaries.
            See .. _link: ../../docs/source/how_to_guides/how_to_use_constranits.ipynb
        algo_options (dict): Algorithm specific configuration of the optimization. See
            :ref:`list_of_algorithms` for supported options of each algorithm.
        derivative (callable, optional): Function that calculates the first derivative
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
        cache_size (int): Number of criterion and derivative evaluations that are cached
            in memory in case they are needed.
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
            - sampling_distribution (str): One of "uniform", "triangle". Default is
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
            - batch_evaluator (str or callaber): See :ref:`batch_evaluators` for
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

    """
    return _optimize(
        direction="minimize",
        criterion=criterion,
        params=params,
        algorithm=algorithm,
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
        cache_size=cache_size,
        scaling=scaling,
        scaling_options=scaling_options,
        multistart=multistart,
        multistart_options=multistart_options,
    )


def _optimize(
    direction,
    criterion,
    params,
    algorithm,
    *,
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
    cache_size,
    scaling,
    scaling_options,
    multistart,
    multistart_options,
):
    """Minimize or maximize criterion using algorithm subject to constraints.

    Arguments are the same as in maximize and minimize, with an additional direction
    argument. Direction is a string that can take the values "maximize" and "minimize".

    Returns are the same as in maximize and minimize.

    """
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
        cache_size=cache_size,
        scaling=scaling,
        scaling_options=scaling_options,
        multistart=multistart,
        multistart_options=multistart_options,
    )

    # store some arguments in a dictionary to save them in the database later
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
            "cache_size": int(cache_size),
        }

    # partial the kwargs into corresponding functions
    criterion = functools.partial(criterion, **criterion_kwargs)
    if derivative is not None:
        derivative = functools.partial(derivative, **derivative_kwargs)
    if criterion_and_derivative is not None:
        criterion_and_derivative = functools.partial(
            criterion_and_derivative, **criterion_and_derivative_kwargs
        )

    # process params and constraints
    params = add_default_bounds_to_params(params)
    for col in ["value", "lower_bound", "upper_bound"]:
        params[col] = params[col].astype(float)
    check_params_are_valid(params)

    # get processed params and constraints
    if constraints:
        pc, pp = process_constraints(constraints, params)
    else:
        pc, pp = None, None

    if pc and multistart:
        types = {constr["type"] for constr in pc}
        raise NotImplementedError(
            "multistart optimizations are not yet compatible with transforming "
            f"constraints. Your transforming constraints are of type {types}."
        )

    # calculate scaling factor and offset and redo params and constraint processing
    if scaling:
        scaling_factor, scaling_offset = calculate_scaling_factor_and_offset(
            params=params,
            constraints=constraints,
            criterion=criterion,
            **scaling_options,
            processed_params=pp,
            processed_constraints=pc,
        )
        pc, pp = process_constraints(
            constraints=constraints,
            params=params,
            scaling_factor=scaling_factor,
            scaling_offset=scaling_offset,
        )
    else:
        scaling_factor, scaling_offset = None, None

    if logging:
        # name and group column are needed in the dashboard but could lead to problems
        # if present anywhere else
        params_with_name_and_group = _add_name_and_group_columns_to_params(params)
        problem_data["params"] = params_with_name_and_group

    params_to_internal, params_from_internal = get_reparametrize_functions(
        params=params,
        constraints=constraints,
        scaling_factor=scaling_factor,
        scaling_offset=scaling_offset,
        processed_params=pp,
        processed_constraints=pc,
    )
    # get internal parameters and bounds
    x = params_to_internal(params["value"].to_numpy())

    # this if condition reduces overhead in the no-constraints case
    if constraints in [None, []]:
        lower_bounds = params["lower_bound"].to_numpy()
        upper_bounds = params["upper_bound"].to_numpy()
    else:
        lower_bounds, upper_bounds = get_internal_bounds(
            params=params,
            constraints=constraints,
            scaling_factor=scaling_factor,
            scaling_offset=scaling_offset,
            processed_params=pp,
        )

    # get convert derivative
    convert_derivative = get_derivative_conversion_function(
        params=params,
        constraints=constraints,
        scaling_factor=scaling_factor,
        scaling_offset=scaling_offset,
        processed_params=pp,
        processed_constraints=pc,
    )

    # do first function evaluation
    first_eval = {
        "internal_params": x,
        "external_params": params,
        "output": criterion(params),
    }

    # fill numdiff_options with defaults
    numdiff_options = _fill_numdiff_options_with_defaults(
        numdiff_options, lower_bounds, upper_bounds
    )

    # create and initialize the database
    if logging:
        database = _create_and_initialize_database(
            logging, log_options, first_eval, problem_data
        )
        db_kwargs = {
            "database": database,
            "path": logging,
            "fast_logging": log_options.get("fast_logging", False),
        }
    else:
        db_kwargs = {"database": None, "path": None, "fast_logging": False}

    # get the algorithm
    internal_algorithm = get_algorithm(
        algorithm=algorithm,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        algo_options=algo_options,
        logging=logging,
        db_kwargs=db_kwargs,
    )

    # set default error penalty
    error_penalty = _fill_error_penalty_with_defaults(
        error_penalty, first_eval, direction
    )

    # create cache
    x_hash = hash_array(x)
    cache = {x_hash: {"criterion": first_eval["output"]}}

    # partial the internal_criterion_and_derivative_template
    always_partialled = {
        "direction": direction,
        "criterion": criterion,
        "params": params,
        "reparametrize_from_internal": params_from_internal,
        "convert_derivative": convert_derivative,
        "derivative": derivative,
        "criterion_and_derivative": criterion_and_derivative,
        "numdiff_options": numdiff_options,
        "logging": logging,
        "db_kwargs": db_kwargs,
        "first_criterion_evaluation": first_eval,
        "cache": cache,
        "cache_size": cache_size,
    }

    internal_criterion_and_derivative = functools.partial(
        internal_criterion_and_derivative_template,
        **always_partialled,
    )

    # do actual optimizations
    if not multistart:

        steps = [{"type": "optimization", "name": "optimization"}]

        step_ids = log_scheduled_steps_and_get_ids(
            steps=steps,
            logging=logging,
            db_kwargs=db_kwargs,
        )
        internal_criterion_and_derivative = functools.partial(
            internal_criterion_and_derivative,
            error_handling=error_handling,
            error_penalty=error_penalty,
        )
        raw_res = internal_algorithm(internal_criterion_and_derivative, x, step_ids[0])
    else:

        lower, upper = get_internal_sampling_bounds(params, constraints)

        multistart_options = _fill_multistart_options_with_defaults(
            options=multistart_options,
            params=params,
            x=x,
            params_to_internal=params_to_internal,
        )

        raw_res = run_multistart_optimization(
            local_algorithm=internal_algorithm,
            criterion_and_derivative=internal_criterion_and_derivative,
            x=x,
            lower_bounds=lower,
            upper_bounds=upper,
            options=multistart_options,
            logging=logging,
            db_kwargs=db_kwargs,
            error_handling=error_handling,
            error_penalty=error_penalty,
        )

    res = process_internal_optimizer_result(
        raw_res,
        direction=direction,
        params_from_internal=params_from_internal,
    )

    return res


def _fill_error_penalty_with_defaults(error_penalty, first_eval, direction):
    error_penalty = error_penalty.copy()
    first_value = first_eval["output"]
    first_value = first_value if np.isscalar(first_value) else first_value["value"]

    if direction == "minimize":
        default_constant = (
            first_value + np.abs(first_value) + CRITERION_PENALTY_CONSTANT
        )
        default_slope = CRITERION_PENALTY_SLOPE
    else:
        default_constant = (
            first_value - np.abs(first_value) - CRITERION_PENALTY_CONSTANT
        )
        default_slope = -CRITERION_PENALTY_SLOPE

    error_penalty["constant"] = error_penalty.get("constant", default_constant)
    error_penalty["slope"] = error_penalty.get("slope", default_slope)

    return error_penalty


def _create_and_initialize_database(logging, log_options, first_eval, problem_data):
    # extract information
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
        first_eval=first_eval,
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
        warnings.warn(
            "The following numdiff options were ignored because they will be set "
            f"internally during the optimization:\n\n{ignored}"
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


def _add_name_and_group_columns_to_params(params):
    """Add a group and name column to the params.

    Args:
        params (pd.DataFrame): See :ref:`params`.

    Returns:
        params (pd.DataFrame): With defaults expanded params DataFrame.

    """
    params = params.copy()

    if "group" not in params.columns:
        params["group"] = "All Parameters"

    if "name" not in params.columns:
        names = [_index_element_to_string(tup) for tup in params.index]
        params["name"] = names
    else:
        params["name"] = params["name"].str.replace("-", "_")

    return params


def _index_element_to_string(element, separator="_"):
    if isinstance(element, (tuple, list)):
        as_strings = [str(entry).replace("-", "_") for entry in element]
        res_string = separator.join(as_strings)
    else:
        res_string = str(element)
    return res_string


def _setdefault(candidate, default):
    out = default if candidate is None else candidate
    return out


def _fill_multistart_options_with_defaults(options, params, x, params_to_internal):
    defaults = {
        "sample": None,
        "n_samples": 10 * len(x),
        "share_optimizations": 0.1,
        "sampling_distribution": "uniform",
        "sampling_method": "sobol" if len(x) <= 30 else "random",
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

    if isinstance(out["batch_evaluator"], str):
        out["batch_evaluator"] = getattr(
            be, f"{out['batch_evaluator']}_batch_evaluator"
        )

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


def process_multistart_sample(raw_sample, params, params_to_internal):
    if isinstance(raw_sample, pd.DataFrame):
        if not raw_sample.columns.equals(params.index):
            raise ValueError(
                "If you provide a custom sample as DataFrame the columns of that "
                "DataFrame and the index of params must be equal."
            )
        sample = raw_sample[params.index].to_numpy()
    elif isinstance(raw_sample, np.ndarray):
        _, n_params = raw_sample.shape
        if n_params != len(params):
            raise ValueError(
                "If you provide a custom sample as a numpy array it must have as many "
                "columns as parameters."
            )
        sample = raw_sample

    sample = np.array([params_to_internal(x) for x in sample])

    return sample
