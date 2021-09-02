import functools
import inspect
import warnings

import numpy as np

import estimagic.batch_evaluators as be
from estimagic.config import CRITERION_PENALTY_CONSTANT
from estimagic.config import CRITERION_PENALTY_SLOPE
from estimagic.logging.database_utilities import append_row
from estimagic.logging.database_utilities import load_database
from estimagic.logging.database_utilities import make_optimization_iteration_table
from estimagic.logging.database_utilities import make_optimization_problem_table
from estimagic.logging.database_utilities import make_optimization_status_table
from estimagic.optimization import AVAILABLE_ALGORITHMS
from estimagic.optimization.broadcast_arguments import broadcast_arguments
from estimagic.optimization.check_arguments import check_argument
from estimagic.optimization.internal_criterion_template import (
    internal_criterion_and_derivative_template,
)
from estimagic.optimization.scaling import calculate_scaling_factor_and_offset
from estimagic.parameters.parameter_conversion import get_derivative_conversion_function
from estimagic.parameters.parameter_conversion import get_internal_bounds
from estimagic.parameters.parameter_conversion import get_reparametrize_functions
from estimagic.parameters.parameter_preprocessing import add_default_bounds_to_params
from estimagic.parameters.parameter_preprocessing import check_params_are_valid
from estimagic.utilities import hash_array
from estimagic.utilities import propose_algorithms


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
    batch_evaluator="joblib",
    batch_evaluator_options=None,
    cache_size=100,
    scaling_options=None,
):
    """Maximize criterion using algorithm subject to constraints.

    Each argument except for batch_evaluator and batch_evaluator_options can also be
    replaced by a list of arguments in which case several optimizations are run in
    parallel. For this, either all arguments must be lists of the same length, or some
    arguments can be provided as single arguments in which case they are automatically
    broadcasted.

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
            - "suffix": A string that is appended to the default table names, separated
            by an underscore. You can use this if you want to write the log into an
            existing database where the default names "optimization_iterations",
            "optimization_status" and "optimization_problem" are already in use.
            - "fast_logging": A boolean that determines if "unsafe" settings are used
            to speed up write processes to the database. This should only be used for
            very short running criterion functions where the main purpose of the log
            is a real-time dashboard and it would not be catastrophic to get a
            corrupted database in case of a sudden system shutdown. If one evaluation
            of the criterion function (and gradient if applicable) takes more than
            100 ms, the logging overhead is negligible.
            - "if_exists": (str) One of "extend", "replace", "raise"
            - "save_all_arguments": (bool). If True, all arguments to maximize
              that can be pickled are saved in the log file. Otherwise, only the
              information needed by the dashboard is saved. Default False.
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
        batch_evaluator (str or Callable): Name of a pre-implemented batch evaluator
            (currently 'joblib' and 'pathos_mp') or Callable with the same interface
            as the estimagic batch_evaluators. See :ref:`batch_evaluators`.
        batch_evaluator_options (dict): Additional configurations for the batch
            batch evaluator. See :ref:`batch_evaluators`.
        cache_size (int): Number of criterion and derivative evaluations that are cached
            in memory in case they are needed.
        scaling_options (dict or None): Options to configure the internal scaling ot
            the parameter vector. By default no rescaling is done. See :ref:`scaling`
            for details and recommendations.

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
        batch_evaluator=batch_evaluator,
        batch_evaluator_options=batch_evaluator_options,
        cache_size=cache_size,
        scaling_options=scaling_options,
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
    batch_evaluator="joblib",
    batch_evaluator_options=None,
    cache_size=100,
    scaling_options=None,
):
    """Minimize criterion using algorithm subject to constraints.

    Each argument except for batch_evaluator and batch_evaluator_options can also be
    replaced by a list of arguments in which case several optimizations are run in
    parallel. For this, either all arguments must be lists of the same length, or some
    arguments can be provided as single arguments in which case they are automatically
    broadcasted.

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
            - "suffix": A string that is appended to the default table names, separated
            by an underscore. You can use this if you want to write the log into an
            existing database where the default names "optimization_iterations",
            "optimization_status" and "optimization_problem" are already in use.
            - "fast_logging": A boolean that determines if "unsafe" settings are used
            to speed up write processes to the database. This should only be used for
            very short running criterion functions where the main purpose of the log
            is a real-time dashboard and it would not be catastrophic to get a
            corrupted database in case of a sudden system shutdown. If one evaluation
            of the criterion function (and gradient if applicable) takes more than
            100 ms, the logging overhead is negligible.
            - "if_exists": (str) One of "extend", "replace", "raise"
            - "save_all_arguments": (bool). If True, all arguments to minimize
              that can be pickled are saved in the log file. Otherwise, only the
              information needed by the dashboard is saved. Default False.
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
        batch_evaluator (str or Callable): Name of a pre-implemented batch evaluator
            (currently 'joblib' and 'pathos_mp') or Callable with the same interface
            as the estimagic batch_evaluators. See :ref:`batch_evaluators`.
        batch_evaluator_options (dict): Additional configurations for the batch
            batch evaluator. See :ref:`batch_evaluators`.
        cache_size (int): Number of criterion and derivative evaluations that are cached
            in memory in case they are needed.
        scaling_options (dict or None): Options to configure the internal scaling ot
            the parameter vector. By default no rescaling is done. See :ref:`scaling`
            for details and recommendations.

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
        batch_evaluator=batch_evaluator,
        batch_evaluator_options=batch_evaluator_options,
        cache_size=cache_size,
        scaling_options=scaling_options,
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
    batch_evaluator,
    batch_evaluator_options,
    cache_size,
    scaling_options,
):
    """Minimize or maximize criterion using algorithm subject to constraints.

    Each argument except for batch_evaluator and batch_evaluator_options can also be
    replaced by a list of arguments in which case several optimizations are run in
    parallel. For this, either all arguments must be lists of the same length, or some
    arguments can be provided as single arguments in which case they are automatically
    broadcasted.

    Args:
        direction (str): One of "maximize" or "minimize".
        criterion (Callable): A function that takes a pandas DataFrame (see
            :ref:`params`) as first argument and returns one of the following:
            - scalar floating point or a numpy array (depending on the algorithm)
            - a dictionary that contains at the entries "value" (a scalar float),
            "contributions" or "root_contributions" (depending on the algortihm) and
            any number of additional entries. The additional dict entries will be
            logged and (if supported) displayed in the dashboard. Check the
            documentation of your algorithm to see which entries or output type
            are required.
        params (pd.DataFrame): A DataFrame with a column called "value" and optional
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
            - "suffix": A string that is appended to the default table names, separated
            by an underscore. You can use this if you want to write the log into an
            existing database where the default names "optimization_iterations",
            "optimization_status" and "optimization_problem" are already in use.
            - "fast_logging": A boolean that determines if "unsafe" settings are used
            to speed up write processes to the database. This should only be used for
            very short running criterion functions where the main purpose of the log
            is a real-time dashboard and it would not be catastrophic to get a
            corrupted database in case of a sudden system shutdown. If one evaluation
            of the criterion function (and gradient if applicable) takes more than
            100 ms, the logging overhead is negligible.
            - "if_exists": (str) One of "extend", "replace", "raise"
            - "save_all_arguments": (bool). If True, all arguments to
              optimize that can be pickled are saved in the log file. Otherwise, only
              the information needed by the dashboard is saved. Default False.
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
        batch_evaluator (str or Callable): Name of a pre-implemented batch evaluator
            (currently 'joblib' and 'pathos_mp') or Callable with the same interface
            as the estimagic batch_evaluators. See :ref:`batch_evaluators`.
        batch_evaluator_options (dict): Additional configurations for the batch
            batch evaluator. See :ref:`batch_evaluators`.
        cache_size (int): Number of criterion and derivative evaluations that are cached
            in memory in case they are needed.
        scaling_options (dict or None): Options to configure the internal scaling ot
            the parameter vector. By default no rescaling is done. See :ref:`scaling`
            for details and recommendations.

    """
    arguments = broadcast_arguments(
        direction=direction,
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
        scaling_options=scaling_options,
    )

    # do rough sanity checks before actual optimization for quicker feedback
    for arg in arguments:
        check_argument(arg)

    if isinstance(batch_evaluator, str):
        batch_evaluator = getattr(be, f"{batch_evaluator}_batch_evaluator")

    if batch_evaluator_options is None:
        batch_evaluator_options = {}

    batch_evaluator_options["unpack_symbol"] = "**"
    default_batch_error_handling = "raise" if len(arguments) == 1 else "continue"
    batch_evaluator_options["error_handling"] = batch_evaluator_options.get(
        "error_handling", default_batch_error_handling
    )

    res = batch_evaluator(_single_optimize, arguments, **batch_evaluator_options)

    res = [_dummy_result_from_traceback(r) for (r) in res]

    res = res[0] if len(res) == 1 else res

    return res


def _single_optimize(
    direction,
    criterion,
    criterion_kwargs,
    params,
    algorithm,
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
    scaling_options,
):
    """Minimize or maximize *criterion* using *algorithm* subject to *constraints*.

    See the docstring of ``_optimize`` for an explanation of all arguments.

    Returns:
        dict: The optimization result.

    """
    # store all arguments in a dictionary to save them in the database later
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

    # calculate scaling factor and offset
    if scaling_options not in (None, {}):
        scaling_factor, scaling_offset = calculate_scaling_factor_and_offset(
            params=params,
            constraints=constraints,
            criterion=criterion,
            **scaling_options,
        )
    else:
        scaling_factor, scaling_offset = None, None

    # name and group column are needed in the dashboard but could lead to problems
    # if present anywhere else
    params_with_name_and_group = _add_name_and_group_columns_to_params(params)
    problem_data["params"] = params_with_name_and_group

    params_to_internal, params_from_internal = get_reparametrize_functions(
        params=params,
        constraints=constraints,
        scaling_factor=scaling_factor,
        scaling_offset=scaling_offset,
    )

    # get internal parameters and bounds
    x = params_to_internal(params["value"].to_numpy())
    lower_bounds, upper_bounds = get_internal_bounds(
        params=params,
        constraints=constraints,
        scaling_factor=scaling_factor,
        scaling_offset=scaling_offset,
    )

    # process algorithm and algo_options
    if isinstance(algorithm, str):
        algo_name = algorithm
    else:
        algo_name = getattr(algorithm, "name", "your algorithm")

    if isinstance(algorithm, str):
        try:
            algorithm = AVAILABLE_ALGORITHMS[algorithm]
        except KeyError:
            proposed = propose_algorithms(algorithm, list(AVAILABLE_ALGORITHMS))
            raise ValueError(
                f"Invalid algorithm: {algorithm}. Did you mean {proposed}?"
            ) from None

    algo_options = _adjust_options_to_algorithms(
        algo_options, lower_bounds, upper_bounds, algorithm, algo_name
    )

    # get convert derivative
    convert_derivative = get_derivative_conversion_function(
        params=params,
        constraints=constraints,
        scaling_factor=scaling_factor,
        scaling_offset=scaling_offset,
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
    if not logging:
        database = False
    else:
        database = _create_and_initialize_database(
            logging, log_options, first_eval, problem_data
        )

    # set default error penalty
    error_penalty = _fill_error_penalty_with_defaults(
        error_penalty, first_eval, direction
    )

    # create cache
    x_hash = hash_array(x)
    cache = {x_hash: {"criterion": first_eval["output"]}}

    # partial the internal_criterion_and_derivative_template
    internal_criterion_and_derivative = functools.partial(
        internal_criterion_and_derivative_template,
        direction=direction,
        criterion=criterion,
        params=params,
        reparametrize_from_internal=params_from_internal,
        convert_derivative=convert_derivative,
        derivative=derivative,
        criterion_and_derivative=criterion_and_derivative,
        numdiff_options=numdiff_options,
        database=database,
        database_path=logging,
        log_options=log_options,
        error_handling=error_handling,
        error_penalty=error_penalty,
        first_criterion_evaluation=first_eval,
        cache=cache,
        cache_size=cache_size,
    )

    res = algorithm(internal_criterion_and_derivative, x, **algo_options)

    p = params.copy()
    p["value"] = params_from_internal(res["solution_x"])
    res["solution_params"] = p

    if "solution_criterion" not in res:
        res["solution_criterion"] = criterion(p)

    # in the long run we can get some of those from the database if logging was used.
    optional_entries = [
        "solution_derivative",
        "solution_hessian",
        "n_criterion_evaluations",
        "n_derivative_evaluations",
        "n_iterations",
        "success",
        "reached_convergence_criterion",
        "message",
    ]

    for entry in optional_entries:
        res[entry] = res.get(entry, f"Not reported by {algo_name}")

    if logging:
        _log_final_status(res, database, logging, log_options)

    return res


def _log_final_status(res, database, path, log_options):
    if isinstance(res["success"], str) and res["success"].startswith("Not reported"):
        status = "unknown"
    elif res["success"]:
        status = "finished successfully"
    else:
        status = "failed"

    fast_logging = log_options.get("fast_logging", False)
    append_row({"status": status}, "optimization_status", database, path, fast_logging)


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
    path = logging
    fast_logging = log_options.get("fast_logging", False)
    if_exists = log_options.get("if_exists", "extend")
    save_all_arguments = log_options.get("save_all_arguments", False)
    database = load_database(path=path, fast_logging=fast_logging)

    # create the optimization_iterations table
    make_optimization_iteration_table(
        database=database,
        first_eval=first_eval,
        if_exists=if_exists,
    )

    # create and initialize the optimization_status table
    make_optimization_status_table(database, if_exists)
    append_row(
        {"status": "running"}, "optimization_status", database, path, fast_logging
    )

    # create_and_initialize the optimization_problem table
    make_optimization_problem_table(database, if_exists, save_all_arguments)
    if not save_all_arguments:
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


def _adjust_options_to_algorithms(
    algo_options, lower_bounds, upper_bounds, algorithm, algo_name
):
    """Reduce the algo_options and check if bounds are compatible with algorithm."""

    # convert algo option keys to valid Python arguments
    algo_options = {key.replace(".", "_"): val for key, val in algo_options.items()}

    valid = set(inspect.signature(algorithm).parameters)

    if isinstance(algorithm, functools.partial):
        partialed_in = set(algorithm.args).union(set(algorithm.keywords))
        valid = valid.difference(partialed_in)

    reduced = {key: val for key, val in algo_options.items() if key in valid}

    ignored = {key: val for key, val in algo_options.items() if key not in valid}

    if ignored:
        warnings.warn(
            "The following algo_options were ignored because they are not compatible "
            f"with {algo_name}:\n\n {ignored}"
        )

    if "lower_bounds" not in valid and not (lower_bounds == -np.inf).all():
        raise ValueError(
            f"{algo_name} does not support lower bounds but your optimization "
            "problem has lower bounds (either because you specified them explicitly "
            "or because they were implied by other constraints)."
        )

    if "upper_bounds" not in valid and not (upper_bounds == np.inf).all():
        raise ValueError(
            f"{algo_name} does not support upper bounds but your optimization "
            "problem has upper bounds (either because you specified them explicitly "
            "or because they were implied by other constraints)."
        )

    if "lower_bounds" in valid:
        reduced["lower_bounds"] = lower_bounds

    if "upper_bounds" in valid:
        reduced["upper_bounds"] = upper_bounds

    return reduced


def _dummy_result_from_traceback(candidate):
    if isinstance(candidate, str):
        out = {
            "solution_params": None,
            "solution_criterion": None,
            "solution_derivative": None,
            "solution_hessian": None,
            "n_criterion_evaluations": None,
            "n_derivative_evaluations": None,
            "n_iterations": None,
            "success": False,
            "reached_convergence_criterion": None,
            "message": candidate,
        }
    else:
        out = candidate
    return out
