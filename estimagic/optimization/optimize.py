"""Functional wrapper around the pygmo, nlopt and scipy libraries."""
import functools
import json
from collections import namedtuple
from multiprocessing import Event
from multiprocessing import Process
from multiprocessing import Queue
from pathlib import Path
from warnings import simplefilter

import numpy as np
import pandas as pd
from joblib import delayed
from joblib import Parallel
from scipy.optimize._numdiff import approx_derivative

from estimagic.config import DEFAULT_DATABASE_NAME
from estimagic.dashboard.server_functions import run_server
from estimagic.decorators import expand_criterion_output
from estimagic.decorators import handle_exceptions
from estimagic.decorators import log_evaluation
from estimagic.decorators import log_gradient
from estimagic.decorators import log_gradient_status
from estimagic.decorators import negative_criterion
from estimagic.decorators import numpy_interface
from estimagic.logging.create_database import prepare_database
from estimagic.logging.update_database import update_scalar_field
from estimagic.optimization.broadcast_arguments import broadcast_arguments
from estimagic.optimization.check_arguments import check_arguments
from estimagic.optimization.pounders import minimize_pounders_np
from estimagic.optimization.process_constraints import process_constraints
from estimagic.optimization.pygmo import minimize_pygmo_np
from estimagic.optimization.reparametrize import reparametrize_from_internal
from estimagic.optimization.reparametrize import reparametrize_to_internal
from estimagic.optimization.scipy import minimize_scipy_np
from estimagic.optimization.utilities import index_element_to_string
from estimagic.optimization.utilities import propose_algorithms


QueueEntry = namedtuple("QueueEntry", ["iteration", "params", "fitness"])


def maximize(
    criterion,
    params,
    algorithm,
    criterion_kwargs=None,
    constraints=None,
    general_options=None,
    algo_options=None,
    gradient_options=None,
    logging=DEFAULT_DATABASE_NAME,
    log_options=None,
    dashboard=False,
    db_options=None,
):
    """Maximize *criterion* using *algorithm* subject to *constraints* and bounds.

    Each argument except for ``general_options`` can also be replaced by a list of
    arguments in which case several optimizations are run in parallel. For this, either
    all arguments must be lists of the same length, or some arguments can be provided
    as single arguments in which case they are automatically broadcasted.

    Args:
        criterion (callable or list of callables):
            Python function that takes a pandas DataFrame with parameters as the first
            argument and returns a scalar floating point value.

        params (pd.DataFrame or list of pd.DataFrames):
            See :ref:`params`.

        algorithm (str or list of strings):
            specifies the optimization algorithm. See :ref:`list_of_algorithms`.

        criterion_kwargs (dict or list of dicts):
            additional keyword arguments for criterion

        constraints (list or list of lists):
            list with constraint dictionaries. See for details.

        general_options (dict):
            additional configurations for the optimization

        algo_options (dict or list of dicts):
            algorithm specific configurations for the optimization

        gradient_options (dict):
            Options for the gradient function.

        logging (str or pathlib.Path): Path to an sqlite3 file which typically has the
            file extension ``.db``. If the file does not exist, it will be created. See
            :ref:`logging` for details.

        log_options (dict): Keyword arguments to influence the logging. See
            :ref:`logging` for details.

        dashboard (bool):
            whether to create and show a dashboard. See :ref:`dashboard` for details.

        db_options (dict):
            dictionary with kwargs to be supplied to the run_server function. See
                :ref:`dashboard` for details.

    """
    # Set a flag for a maximization problem.
    general_options = {} if general_options is None else general_options
    general_options["_is_maximization"] = True

    results = minimize(
        criterion=criterion,
        params=params,
        algorithm=algorithm,
        criterion_kwargs=criterion_kwargs,
        constraints=constraints,
        general_options=general_options,
        algo_options=algo_options,
        gradient_options=gradient_options,
        logging=logging,
        log_options=log_options,
        dashboard=dashboard,
        db_options=db_options,
    )

    # Change the fitness value. ``results`` is either a tuple of results and params or a
    # list of tuples.
    if isinstance(results, list):
        for result in results:
            result[0]["fitness"] = -result[0]["fitness"]
    else:
        results[0]["fitness"] = -results[0]["fitness"]

    return results


def minimize(
    criterion,
    params,
    algorithm,
    criterion_kwargs=None,
    constraints=None,
    general_options=None,
    algo_options=None,
    gradient_options=None,
    logging=DEFAULT_DATABASE_NAME,
    log_options=None,
    dashboard=False,
    db_options=None,
):
    """Minimize *criterion* using *algorithm* subject to *constraints* and bounds.

    Each argument except for ``general_options`` can also be replaced by a list of
    arguments in which case several optimizations are run in parallel. For this, either
    all arguments must be lists of the same length, or some arguments can be provided
    as single arguments in which case they are automatically broadcasted.

    Args:
        criterion (function or list of functions):
            Python function that takes a pandas DataFrame with parameters as the first
            argument and returns a scalar floating point value.

        params (pd.DataFrame or list of pd.DataFrames):
            See :ref:`params`.

        algorithm (str or list of strings):
            specifies the optimization algorithm. See :ref:`list_of_algorithms`.

        criterion_kwargs (dict or list of dicts):
            additional keyword arguments for criterion

        constraints (list or list of lists):
            list with constraint dictionaries. See for details.

        general_options (dict):
            additional configurations for the optimization

        algo_options (dict or list of dicts):
            algorithm specific configurations for the optimization

        gradient_options (dict):
            Options for the gradient function.

        logging (str or pathlib.Path): Path to an sqlite3 file which typically has the
            file extension ``.db``. If the file does not exist, it will be created. See
            :ref:`logging` for details.

        log_options (dict): Keyword arguments to influence the logging. See
            :ref:`logging` for details.

        dashboard (bool):
            whether to create and show a dashboard. See :ref:`dashboard` for details.

        db_options (dict):
            dictionary with kwargs to be supplied to the run_server function. See
                :ref:`dashboard` for details.

    """
    criterion_kwargs = {} if criterion_kwargs is None else criterion_kwargs
    constraints = [] if constraints is None else constraints
    algo_options = {} if algo_options is None else algo_options
    log_options = {} if log_options is None else log_options
    db_options = {} if db_options is None else db_options
    general_options = {} if general_options is None else general_options

    # Gradients are currently not allowed to be passed to minimize.
    gradient = None

    arguments = broadcast_arguments(
        criterion=criterion,
        params=params,
        algorithm=algorithm,
        criterion_kwargs=criterion_kwargs,
        constraints=constraints,
        general_options=general_options,
        algo_options=algo_options,
        gradient=gradient,
        gradient_options=gradient_options,
        logging=logging,
        log_options=log_options,
        dashboard=dashboard,
        db_options=db_options,
    )
    check_arguments(arguments)

    if len(arguments) == 1:
        # Run only one optimization
        arguments = arguments[0]
        results = _single_minimize(**arguments)
    else:
        # Run multiple optimizations
        if dashboard:
            raise NotImplementedError(
                "Dashboard cannot be used for multiple optimizations, yet."
            )

        # set up multiprocessing
        if "n_cores" not in arguments[0]["general_options"]:
            raise ValueError(
                "n_cores need to be specified in general_options"
                + " if multiple optimizations should be run."
            )
        n_cores = arguments[0]["general_options"]["n_cores"]

        results = Parallel(n_jobs=n_cores)(
            delayed(_one_argument_single_minimize)(argument) for argument in arguments
        )

    return results


def _single_minimize(
    criterion,
    params,
    algorithm,
    criterion_kwargs,
    constraints,
    general_options,
    algo_options,
    gradient,
    gradient_options,
    logging,
    log_options,
    dashboard,
    db_options,
):
    """Minimize * criterion * using * algorithm * subject to * constraints * and bounds.
    Only one minimization.

    Args:
        criterion (function):
            Python function that takes a pandas DataFrame with parameters as the first
            argument and returns a scalar floating point value.

        params (pd.DataFrame):
            See :ref:`params`.

        algorithm (str):
            specifies the optimization algorithm. See :ref:`list_of_algorithms`.

        criterion_kwargs (dict):
            additional keyword arguments for criterion

        constraints (list):
            list with constraint dictionaries. See for details.

        general_options (dict):
            additional configurations for the optimization

        algo_options (dict):
            algorithm specific configurations for the optimization

        gradient (callable or None):
            Gradient function.

        gradient_options (dict):
            Options for the gradient function.

        logging (str or pathlib.Path): Path to an sqlite3 file which typically has the
            file extension ``.db``. If the file does not exist, it will be created. See
            :ref:`logging` for details.

        log_options (dict): Keyword arguments to influence the logging. See
            :ref:`logging` for details.

        dashboard (bool):
            whether to create and show a dashboard

        db_options (dict):
            dictionary with kwargs to be supplied to the run_server function.

    """
    simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
    params = _process_params(params)

    # Apply decorator two handle criterion functions with one or two returns.
    criterion = expand_criterion_output(criterion)

    is_maximization = general_options.get("_is_maximization", False)
    criterion = negative_criterion(criterion) if is_maximization else criterion
    fitness_factor = -1 if is_maximization else 1

    criterion_out, comparison_plot_data = criterion(params, **criterion_kwargs)
    if np.isscalar(criterion_out):
        fitness_eval = fitness_factor * criterion_out
    else:
        fitness_eval = fitness_factor * np.mean(np.square(criterion_out))

    if np.any(np.isnan(fitness_eval)):
        raise ValueError(
            "The criterion function evaluated at the start parameters returns NaNs."
        )

    if logging:
        database = prepare_database(
            logging, params, comparison_plot_data, log_options, constraints
        )
    else:
        database = False

    general_options["_start_criterion_value"] = criterion_out

    constraints, params = process_constraints(constraints, params)
    internal_params = reparametrize_to_internal(params, constraints)

    queue = Queue() if dashboard else None
    if dashboard:
        stop_signal = Event()
        outer_server_process = Process(
            target=run_server,
            kwargs={
                "queue": queue,
                "db_options": db_options,
                "start_param_df": params,
                "start_fitness": fitness_eval,
                "stop_signal": stop_signal,
            },
            daemon=False,
        )
        outer_server_process.start()

    result, params = _internal_minimize(
        criterion=criterion,
        criterion_kwargs=criterion_kwargs,
        params=params,
        internal_params=internal_params,
        constraints=constraints,
        algorithm=algorithm,
        algo_options=algo_options,
        gradient=gradient,
        gradient_options=gradient_options,
        general_options=general_options,
        database=database,
        queue=queue,
        fitness_factor=fitness_factor,
    )

    if dashboard:
        stop_signal.set()
        outer_server_process.terminate()

    return result, params


def _one_argument_single_minimize(kwargs):
    """Wrapper for single_minimize to use kwargs with multiprocessing."""
    return _single_minimize(**kwargs)


def _internal_minimize(
    criterion,
    criterion_kwargs,
    params,
    internal_params,
    constraints,
    algorithm,
    algo_options,
    gradient,
    gradient_options,
    general_options,
    database,
    queue,
    fitness_factor,
):
    """Create the internal criterion function and minimize it.

    Args:
        criterion (function):
            Python function that takes a pandas DataFrame with parameters as the first
            argument and returns a scalar floating point value.

        criterion_kwargs (dict):
            additional keyword arguments for criterion

        params (pd.DataFrame):
            See :ref:`params`.

        internal_params (DataFrame):
            See :ref:`params`.

        constraints (list):
            list with constraint dictionaries. See for details.

        algorithm (str):
            specifies the optimization algorithm. See :ref:`list_of_algorithms`.

        algo_options (dict):
            algorithm specific configurations for the optimization

        gradient (callable or None):
            Gradient function.

        gradient_options (dict):
            Options for the gradient function.

        general_options (dict):
            additional configurations for the optimization

        database (sqlalchemy.MetaData). The engine that connects to the
            database can be accessed via ``database.bind``.

        queue (Queue):
            queue to which the fitness evaluations and params DataFrames are supplied.

        fitness_factor (float):
            multiplicative factor for the fitness displayed in the dashboard.
            Set to -1 for maximizations to plot the fitness that is being maximized.

    """
    logging_decorator = functools.partial(
        log_evaluation,
        database=database,
        tables=["params_history", "criterion_history", "comparison_plot"],
    )

    internal_criterion = create_internal_criterion(
        criterion=criterion,
        params=params,
        constraints=constraints,
        criterion_kwargs=criterion_kwargs,
        logging_decorator=logging_decorator,
        general_options=general_options,
        database=database,
        queue=queue,
        fitness_factor=fitness_factor,
    )

    internal_gradient = create_internal_gradient(
        gradient=gradient,
        gradient_options=gradient_options,
        criterion=criterion,
        params=params,
        internal_params=internal_params,
        constraints=constraints,
        criterion_kwargs=criterion_kwargs,
        general_options=general_options,
        database=database,
        fitness_factor=fitness_factor,
        algorithm=algorithm,
    )

    current_dir_path = Path(__file__).resolve().parent
    with open(current_dir_path / "algo_dict.json") as j:
        algos = json.load(j)
    origin, algo_name = algorithm.split("_", 1)

    try:
        assert algo_name in algos[origin], "Invalid algorithm requested: {}".format(
            algorithm
        )
    except (AssertionError, KeyError):
        proposals = propose_algorithms(algorithm, algos)
        raise NotImplementedError(
            f"{algorithm} is not a valid choice. Did you mean one of {proposals}?"
        )

    bounds = _internal_bounds_from_params(params)

    if database:
        update_scalar_field(database, "optimization_status", "running")

    if origin in ["nlopt", "pygmo"]:
        results = minimize_pygmo_np(
            internal_criterion,
            internal_params,
            bounds,
            origin,
            algo_name,
            algo_options,
            internal_gradient,
        )

    elif origin == "scipy":
        results = minimize_scipy_np(
            internal_criterion,
            internal_params,
            bounds=bounds,
            algo_name=algo_name,
            algo_options=algo_options,
            gradient=internal_gradient,
        )
    elif origin == "tao":
        crit_val = general_options["_start_criterion_value"]
        len_criterion_value = 1 if np.isscalar(crit_val) else len(crit_val)
        results = minimize_pounders_np(
            internal_criterion,
            internal_params,
            bounds,
            n_errors=len_criterion_value,
            **algo_options,
        )
    else:
        raise NotImplementedError("Invalid algorithm requested.")

    if database:
        update_scalar_field(database, "optimization_status", results["status"])

    params = reparametrize_from_internal(
        internal=results["x"],
        fixed_values=params["_internal_fixed_value"].to_numpy(),
        pre_replacements=params["_pre_replacements"].to_numpy().astype(int),
        processed_constraints=constraints,
        post_replacements=params["_post_replacements"].to_numpy().astype(int),
        processed_params=params,
    )

    return results, params


def create_internal_criterion(
    criterion,
    params,
    constraints,
    criterion_kwargs,
    logging_decorator,
    general_options,
    database,
    queue,
    fitness_factor,
):
    """Create the internal criterion function.

    Args:
        criterion (function):
            Python function that takes a pandas DataFrame with parameters as the first
            argument and returns a scalar floating point value.

        params (pd.DataFrame):
            See :ref:`params`.

        constraints (list):
            list with constraint dictionaries. See for details.

        criterion_kwargs (dict):
            additional keyword arguments for criterion

        logging_decorator (callable):
            Decorator used for logging information. Either log parameters and fitness
            values during the optimization or log the gradient status.

        general_options (dict):
            additional configurations for the optimization

        database (sqlalchemy.MetaData). The engine that connects to the
            database can be accessed via ``database.bind``.

        queue (Queue):
            queue to which the fitness evaluations and params DataFrames are supplied.

        fitness_factor (float):
            multiplicative factor for the fitness displayed in the dashboard.
            Set to -1 for maximizations to plot the fitness that is being maximized.

    Returns:
        internal_criterion (function):
            function that takes an internal_params DataFrame as only argument.
            It calls the original criterion function after the necessary
            reparametrizations and passes the results to the dashboard queue if given
            before returning the fitness evaluation.

    """
    c = np.zeros(1)

    @handle_exceptions(database, params, constraints, params, general_options)
    @numpy_interface(params, constraints)
    @logging_decorator
    def internal_criterion(p, counter=c):
        criterion_out, comparison_plot_data = criterion(p, **criterion_kwargs)
        if np.isscalar(criterion_out):
            fitness_eval = criterion_out
        else:
            # Todo: This is a temporary fix for POUNDERs which returns an array.
            fitness_eval = np.mean(np.square(criterion_out))

        if queue is not None:
            queue.put(
                QueueEntry(
                    iteration=counter[0],
                    params=p,
                    fitness=fitness_factor * fitness_eval,
                )
            )
        counter += 1

        return criterion_out, comparison_plot_data

    return internal_criterion


def _process_params(params):
    assert (
        not params.index.duplicated().any()
    ), "No duplicates allowed in the index of params."
    params = params.copy()
    if "lower" not in params.columns:
        params["lower"] = -np.inf
    else:
        params["lower"].fillna(-np.inf)

    if "upper" not in params.columns:
        params["upper"] = np.inf
    else:
        params["upper"].fillna(np.inf)

    if "group" not in params.columns:
        params["group"] = "All Parameters"

    if "name" not in params.columns:
        names = [index_element_to_string(tup) for tup in params.index]
        params["name"] = names

    assert "_fixed" not in params.columns, "Invalid column name _fixed in params_df."

    invalid_names = ["_fixed_value", "_is_fixed_to_value", "_is_fixed_to_other"]
    invalid_present_columns = []
    for col in params.columns:
        if col in invalid_names or col.startswith("_internal"):
            invalid_present_columns.append(col)

    if len(invalid_present_columns) > 0:
        msg = (
            "Column names starting with '_internal' and as well as any other of the "
            f"following columns are not allowed in params:\n{invalid_names}."
            f"This is violated for:\n{invalid_present_columns}."
        )
        raise ValueError(msg)
    return params


def create_internal_gradient(
    gradient,
    gradient_options,
    criterion,
    params,
    internal_params,
    constraints,
    criterion_kwargs,
    general_options,
    database,
    fitness_factor,
    algorithm,
):
    n_internal_params = params["_internal_free"].sum()
    gradient_options = {} if gradient_options is None else gradient_options

    if gradient is None:
        gradient = approx_derivative
        default_options = {
            "method": "2-point",
            "rel_step": None,
            "f0": None,
            "sparsity": None,
            "as_linear_operator": False,
        }
        gradient_options = {**default_options, **gradient_options}

        if gradient_options["method"] == "2-point":
            n_gradient_evaluations = 2 * n_internal_params
        elif gradient_options["method"] == "3-point":
            n_gradient_evaluations = 3 * n_internal_params
        else:
            raise ValueError(
                f"Gradient method '{gradient_options['method']} not supported."
            )

    else:
        n_gradient_evaluations = gradient_options.pop("n_gradient_evaluations", None)

    logging_decorator = functools.partial(
        log_gradient_status,
        database=database,
        n_gradient_evaluations=n_gradient_evaluations,
    )

    internal_criterion = create_internal_criterion(
        criterion=criterion,
        params=params,
        constraints=constraints,
        criterion_kwargs=criterion_kwargs,
        logging_decorator=logging_decorator,
        general_options=general_options,
        database=database,
        queue=None,
        fitness_factor=fitness_factor,
    )
    bounds = _internal_bounds_from_params(params)
    names = params.query("_internal_free")["name"].tolist()

    @log_gradient(database, names)
    def internal_gradient(x):
        return gradient(internal_criterion, x, bounds=bounds, **gradient_options)

    return internal_gradient


def _internal_bounds_from_params(params):
    bounds = tuple(
        params.query("_internal_free")[["_internal_lower", "_internal_upper"]]
        .to_numpy()
        .T
    )
    return bounds
