"""Functional wrapper around the pygmo, nlopt and scipy libraries."""
import json
from collections import namedtuple
from multiprocessing import Event
from multiprocessing import Pool
from multiprocessing import Process
from multiprocessing import Queue
from pathlib import Path
from warnings import simplefilter

import numpy as np
import pandas as pd
import pygmo as pg
from scipy.optimize import minimize as scipy_minimize
from scipy.optimize._numdiff import approx_derivative

from estimagic.config import DEFAULT_DATABASE_NAME
from estimagic.config import OPTIMIZER_SAVE_GRADIENTS
from estimagic.dashboard.server_functions import run_server
from estimagic.decorators import logging
from estimagic.decorators import x_to_params
from estimagic.logging.create_database import prepare_database
from estimagic.optimization.pounders import minimize_pounders
from estimagic.optimization.process_arguments import process_optimization_arguments
from estimagic.optimization.process_constraints import process_constraints
from estimagic.optimization.reparametrize import reparametrize_from_internal
from estimagic.optimization.reparametrize import reparametrize_to_internal
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

    def neg_criterion(params, **criterion_kwargs):
        return -criterion(params, **criterion_kwargs)

    # identify the criterion function as belongig to a maximization problem
    if general_options is None:
        general_options = {"_maximization": True}
    else:
        general_options["_maximization"] = True

    res_dict, params = minimize(
        neg_criterion,
        params=params,
        algorithm=algorithm,
        criterion_kwargs=criterion_kwargs,
        constraints=constraints,
        general_options=general_options,
        algo_options=algo_options,
        logging=logging,
        log_options=log_options,
        dashboard=dashboard,
        db_options=db_options,
    )
    res_dict["fun"] = -res_dict["fun"]

    return res_dict, params


def minimize(
    criterion,
    params,
    algorithm,
    criterion_kwargs=None,
    constraints=None,
    general_options=None,
    algo_options=None,
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
    arguments = process_optimization_arguments(
        criterion=criterion,
        params=params,
        algorithm=algorithm,
        criterion_kwargs=criterion_kwargs,
        constraints=constraints,
        general_options=general_options,
        algo_options=algo_options,
        logging=logging,
        log_options=log_options,
        dashboard=dashboard,
        db_options=db_options,
    )

    if len(arguments) == 1:
        # Run only one optimization
        arguments = arguments[0]
        result = _single_minimize(**arguments)
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
        pool = Pool(processes=n_cores)
        result = pool.map(_one_argument_single_minimize, arguments)

    return result


def _single_minimize(
    criterion,
    params,
    algorithm,
    criterion_kwargs,
    constraints,
    general_options,
    algo_options,
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

    database = prepare_database(logging, params, log_options) if logging else False

    fitness_factor = -1 if general_options.get("_maximization", False) else 1
    fitness_eval = fitness_factor * criterion(params, **criterion_kwargs)
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

    result = _internal_minimize(
        criterion=criterion,
        criterion_kwargs=criterion_kwargs,
        params=params,
        internal_params=internal_params,
        constraints=constraints,
        algorithm=algorithm,
        algo_options=algo_options,
        general_options=general_options,
        database=database,
        queue=queue,
        fitness_factor=fitness_factor,
    )

    if dashboard:
        stop_signal.set()
        outer_server_process.terminate()
    return result


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

        general_options (dict):
            additional configurations for the optimization

        database (sqlalchemy.sql.schema.MetaData). The engine that connects to the
            database can be accessed via ``database.bind``.

        queue (Queue):
            queue to which the fitness evaluations and params DataFrames are supplied.

        fitness_factor (float):
            multiplicative factor for the fitness displayed in the dashboard.
            Set to -1 for maximizations to plot the fitness that is being maximized.

    """
    internal_criterion = create_internal_criterion(
        criterion=criterion,
        params=params,
        constraints=constraints,
        criterion_kwargs=criterion_kwargs,
        database=database,
        tables=["params_history", "criterion_history"],
        queue=queue,
        fitness_factor=fitness_factor,
    )

    internal_jac = create_internal_jac(
        criterion=criterion,
        params=params,
        constraints=constraints,
        criterion_kwargs=criterion_kwargs,
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

    if origin == "pygmo" and algorithm != "pygmo_simulated_annealing":
        assert (
            "popsize" in algo_options
        ), f"For genetic optimizers like {algo_name}, popsize is mandatory."
        assert (
            "gen" in algo_options
        ), f"For genetic optimizers like {algo_name}, gen is mandatory."

    if origin in ["nlopt", "pygmo"]:
        prob = _create_problem(internal_criterion, params)
        algo = _create_algorithm(algo_name, algo_options, origin)
        pop = _create_population(prob, algo_options, internal_params)
        evolved = algo.evolve(pop)
        result = _process_results(evolved, params, internal_params, constraints, origin)
    elif origin == "scipy":
        bounds = _get_scipy_bounds(params)
        minimized = scipy_minimize(
            internal_criterion,
            internal_params,
            jac=internal_jac,
            method=algo_name,
            bounds=bounds,
            options=algo_options,
        )
        result = _process_results(
            minimized, params, internal_params, constraints, origin
        )
    elif origin == "tao":
        len_output = algo_options.pop("len_output", None)
        if len_output is None:
            len_output = len(criterion(params))

        bounds = (
            params.query("_internal_free")[["lower", "upper"]].to_numpy().T.tolist()
        )
        minimized = minimize_pounders(
            internal_criterion, internal_params, len_output, bounds, **algo_options
        )
        result = _process_results(
            minimized, params, internal_params, constraints, origin
        )
    else:
        raise ValueError("Invalid algorithm requested.")

    return result


def create_internal_criterion(
    criterion,
    params,
    constraints,
    criterion_kwargs,
    database,
    tables,
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

        database (sqlalchemy.sql.schema.MetaData). The engine that connects to the
            database can be accessed via ``database.bind``.

        tables (List[str]):
            A list of tables to log the information.

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
    c = np.zeros(1, dtype=int)

    @x_to_params(params, constraints)
    @logging(database, tables)
    def internal_criterion(p, counter=c):
        fitness_eval = criterion(p, **criterion_kwargs)

        # For Pounders, return the sum of squared errors.
        _fitness_eval = (
            fitness_eval.sum() if isinstance(fitness_eval, np.ndarray) else fitness_eval
        )

        if queue is not None:
            queue.put(
                QueueEntry(
                    iteration=counter[0],
                    params=p,
                    fitness=fitness_factor * _fitness_eval,
                )
            )
        counter += 1
        return fitness_eval

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


def _get_scipy_bounds(params):
    bounds = (
        params.query("_internal_free")[["lower", "upper"]]
        .replace([-np.inf, np.inf], [None, None])
        .to_numpy()
    )

    return bounds


def _create_problem(internal_criterion, params):
    params = params.query("_internal_free")

    class Problem:
        def fitness(self, x):
            return [internal_criterion(x)]

        def get_bounds(self):
            lb = params["_internal_lower"].to_numpy()
            ub = params["_internal_upper"].to_numpy()
            return lb, ub

    return Problem()


def _create_algorithm(algo_name, algo_options, origin):
    """Create a pygmo algorithm."""
    if origin == "nlopt":
        algo = pg.algorithm(pg.nlopt(solver=algo_name))
        for option, val in algo_options.items():
            setattr(algo.extract(pg.nlopt), option, val)
    elif origin == "pygmo":
        pygmo_uda = getattr(pg, algo_name)
        algo_options = algo_options.copy()
        if "popsize" in algo_options:
            del algo_options["popsize"]
        algo = pg.algorithm(pygmo_uda(**algo_options))

    return algo


def _create_population(problem, algo_options, internal_params):
    """Create a pygmo population object.

    Todo:
        - constrain random initial values to be in some bounds
        - remove hardcoded seed

    """
    popsize = algo_options.copy().pop("popsize", 1)
    pop = pg.population(problem, size=popsize - 1, seed=5471)
    pop.push_back(internal_params)
    return pop


def _process_results(res, params, internal_params, constraints, origin):
    """Convert optimization results into json serializable dictionary.
    Args:
        res: Result from numerical optimizer.
        params (DataFrame): See :ref:`params`.
        internal_params (DataFrame): See :ref:`params`.
        constraints (list): constraints for the optimization
        origin (str): takes the values "pygmo", "nlopt", "scipy"
    """
    if origin == "scipy":
        res_dict = {**res}
        for key, value in res_dict.items():
            if isinstance(value, np.ndarray):
                res_dict[key] = value.tolist()
        x = res.x
    elif origin in ["pygmo", "nlopt"]:
        x = res.champion_x
        res_dict = {"fun": res.champion_f[0]}
    elif origin in ["tao"]:
        x = res["x"]
        res_dict = {**res}
    params = reparametrize_from_internal(
        internal=x,
        fixed_values=params["_internal_fixed_value"].to_numpy(),
        pre_replacements=params["_pre_replacements"].to_numpy().astype(int),
        processed_constraints=constraints,
        post_replacements=params["_post_replacements"].to_numpy().astype(int),
        processed_params=params,
    )

    return res_dict, params


def create_internal_jac(
    criterion,
    params,
    constraints,
    criterion_kwargs,
    database,
    fitness_factor,
    algorithm,
    method="3-point",
    rel_step=None,
    f0=None,
    sparsity=None,
    as_linear_operator=False,
):

    database = database if algorithm in OPTIMIZER_SAVE_GRADIENTS else None

    internal_criterion = create_internal_criterion(
        criterion=criterion,
        params=params,
        constraints=constraints,
        criterion_kwargs=criterion_kwargs,
        database=database,
        tables=["gradient_params_history", "gradient_history"],
        queue=None,
        fitness_factor=fitness_factor,
    )
    bounds = tuple(params.query("_internal_free")[["lower", "upper"]].to_numpy().T)

    def internal_jac(x):
        return approx_derivative(
            internal_criterion,
            x,
            method,
            rel_step,
            f0,
            bounds,
            sparsity,
            as_linear_operator,
        )

    return internal_jac
