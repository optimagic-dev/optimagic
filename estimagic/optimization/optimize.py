"""Functional wrapper around the pygmo, nlopt and scipy libraries."""
import json
from collections import namedtuple
from multiprocessing import Event
from multiprocessing import Process
from multiprocessing import Queue
from pathlib import Path

import numpy as np
import pygmo as pg
from scipy.optimize import minimize as scipy_minimize

from estimagic.dashboard.server_functions import run_server
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
    criterion_args=None,
    criterion_kwargs=None,
    constraints=None,
    general_options=None,
    algo_options=None,
    dashboard=False,
    db_options=None,
):
    """
    Maximize *criterion* using *algorithm* subject to *constraints* and bounds.

    Args:
        criterion (function):
            Python function that takes a pandas Series with parameters as the first
            argument and returns a scalar floating point value.

        params (pd.DataFrame):
            See :ref:`params`.

        algorithm (str):
            specifies the optimization algorithm. See :ref:`list_of_algorithms`.

        criterion_args (list or tuple):
            additional positional arguments for criterion

        criterion_kwargs (dict):
            additional keyword arguments for criterion

        constraints (list):
            list with constraint dictionaries. See for details.

        general_options (dict):
            additional configurations for the optimization.

        algo_options (dict):
            algorithm specific configurations for the optimization.

        dashboard (bool):
            whether to create and show a dashboard.

        db_options (dict):
            dictionary with kwargs to be supplied to the run_server function.

    """

    def neg_criterion(*criterion_args, **criterion_kwargs):
        return -criterion(*criterion_args, **criterion_kwargs)

    res_dict, params = minimize(
        neg_criterion,
        params=params,
        algorithm=algorithm,
        criterion_args=criterion_args,
        criterion_kwargs=criterion_kwargs,
        constraints=constraints,
        general_options=general_options,
        algo_options=algo_options,
        dashboard=dashboard,
        db_options=db_options,
    )
    res_dict["fun"] = -res_dict["fun"]

    return res_dict, params


def minimize(
    criterion,
    params,
    algorithm,
    criterion_args=None,
    criterion_kwargs=None,
    constraints=None,
    general_options=None,
    algo_options=None,
    dashboard=False,
    db_options=None,
):
    """Minimize *criterion* using *algorithm* subject to *constraints* and bounds.

    Args:
        criterion (function):
            Python function that takes a pandas Series with parameters as the first
            argument and returns a scalar floating point value.

        params (pd.DataFrame):
            See :ref:`params`.

        algorithm (str):
            specifies the optimization algorithm. See :ref:`list_of_algorithms`.

        criterion_args (list or tuple):
            additional positional arguments for criterion

        criterion_kwargs (dict):
            additional keyword arguments for criterion

        constraints (list):
            list with constraint dictionaries. See for details.

        general_options (dict):
            additional configurations for the optimization

        algo_options (dict):
            algorithm specific configurations for the optimization

        dashboard (bool):
            whether to create and show a dashboard

        db_options (dict):
            dictionary with kwargs to be supplied to the run_server function.

    """
    # set default arguments
    criterion_args = [] if criterion_args is None else criterion_args
    criterion_kwargs = {} if criterion_kwargs is None else criterion_kwargs
    constraints = [] if constraints is None else constraints
    general_options = {} if general_options is None else general_options
    algo_options = {} if algo_options is None else algo_options
    db_options = {} if db_options is None else db_options

    params = _process_params(params)
    fitness_eval = criterion(params, *criterion_args, **criterion_kwargs)
    constraints = process_constraints(constraints, params)
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

    result, timing_info = _minimize(
        criterion=criterion,
        criterion_args=criterion_args,
        criterion_kwargs=criterion_kwargs,
        params=params,
        internal_params=internal_params,
        constraints=constraints,
        algorithm=algorithm,
        algo_options=algo_options,
        general_options=general_options,
        queue=queue,
    )

    if dashboard:
        stop_signal.set()
        outer_server_process.terminate()
    return result, timing_info


def _minimize(
    criterion,
    criterion_args,
    criterion_kwargs,
    params,
    internal_params,
    constraints,
    algorithm,
    algo_options,
    general_options,
    queue,
):
    """
    Create the internal criterion function and minimize it.

    Args:
        criterion (function):
            Python function that takes a pandas Series with parameters as the first
            argument and returns a scalar floating point value.

        criterion_args (list or tuple):
            additional positional arguments for criterion

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

        queue (Queue):
            queue to which originally the parameters DataFrame is supplied and to which
            the updated parameter Series will be supplied later.

    """
    internal_criterion = _create_internal_criterion(
        criterion=criterion,
        params=params,
        internal_params=internal_params,
        constraints=constraints,
        criterion_args=criterion_args,
        criterion_kwargs=criterion_kwargs,
        queue=queue,
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

    if origin in ["nlopt", "pygmo"]:
        prob = _create_problem(internal_criterion, internal_params)
        algo = _create_algorithm(algo_name, algo_options, origin)
        pop = _create_population(prob, algo_options, internal_params)
        evolved = algo.evolve(pop)
        result = _process_results(evolved, params, internal_params, constraints, origin)
    elif origin == "scipy":
        bounds = _get_scipy_bounds(internal_params)
        x0 = _x_from_params(params, constraints)
        minimized = scipy_minimize(
            internal_criterion,
            x0,
            method=algo_name,
            bounds=bounds,
            options=algo_options,
        )
        result = _process_results(
            minimized, params, internal_params, constraints, origin
        )
    else:
        raise ValueError("Invalid algorithm requested.")

    return result


def _create_internal_criterion(
    criterion,
    params,
    internal_params,
    constraints,
    criterion_args,
    criterion_kwargs,
    queue,
):
    c = np.ones(1, dtype=int)

    def internal_criterion(x, counter=c):
        p = _params_from_x(x, internal_params, constraints, params)
        fitness_eval = criterion(p, *criterion_args, **criterion_kwargs)
        if queue is not None:
            queue.put(QueueEntry(iteration=counter[0], params=p, fitness=fitness_eval))
        counter += 1
        return fitness_eval

    return internal_criterion


def _params_from_x(x, internal_params, constraints, params):
    internal_params = internal_params.copy(deep=True)
    internal_params["value"] = x
    updated_params = reparametrize_from_internal(internal_params, constraints, params)
    return updated_params


def _x_from_params(params, constraints):
    return reparametrize_to_internal(params, constraints)["value"].to_numpy()


def _process_params(params):
    assert (
        not params.index.duplicated().any()
    ), "No duplicates allowed in the index of params."
    params = params.copy()
    if "lower" not in params.columns:
        params["lower"] = -np.inf
    if "upper" not in params.columns:
        params["upper"] = np.inf
    if "group" not in params.columns:
        params["group"] = "All Parameters"

    if "name" not in params.columns:
        names = [index_element_to_string(tup) for tup in params.index]
        params["name"] = names

    assert "_fixed" not in params.columns, "Invalid column name _fixed in params_df."
    return params


def _get_scipy_bounds(params):
    unprocessed_bounds = params[["lower", "upper"]].to_numpy().tolist()
    bounds = []
    for lower, upper in unprocessed_bounds:
        bounds.append((_convert_bound(lower), _convert_bound(upper)))
    return bounds


def _convert_bound(x):
    if np.isfinite(x):
        return x
    else:
        return None


def _create_problem(internal_criterion, internal_params):
    class Problem:
        def fitness(self, x):
            return [internal_criterion(x)]

        def get_bounds(self):
            return (internal_params["lower"], internal_params["upper"])

    return Problem()


def _create_algorithm(algo_name, algo_options, origin):
    """Create a pygmo algorithm.

    Todo:
        - Pass the algo options through

    """
    if origin == "nlopt":
        algo = pg.algorithm(pg.nlopt(solver=algo_name))
        for option, val in algo_options.items():
            setattr(algo.extract(pg.nlopt), option, val)
    elif origin == "pygmo":
        pygmo_uda = getattr(pg, algo_name)
        algo = pg.algorithm(pygmo_uda(**algo_options))

    return algo


def _create_population(problem, algo_options, internal_params):
    """Create a pygmo population object.

    Todo:
        - constrain random initial values to be in some bounds
        - remove hardcoded seed

    """
    popsize = algo_options.copy().pop("popsize", 1)
    x0 = internal_params["value"].to_numpy()
    pop = pg.population(problem, size=popsize - 1, seed=5471)
    pop.push_back(x0)
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
        res_dict = {}
        res_dict.update(res)
        for key, value in res_dict.items():
            if isinstance(value, np.ndarray):
                res_dict[key] = value.tolist()
        x = res.x
    elif origin in ["pygmo", "nlopt"]:
        x = res.champion_x
        res_dict = {"fun": res.champion_f[0]}
    params = _params_from_x(x, internal_params, constraints, params)

    res_dict["internal_x"] = x.tolist()
    return res_dict, params
