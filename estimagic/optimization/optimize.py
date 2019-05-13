"""Functional wrapper around the object oriented pygmo library."""
import json
import os
from collections import namedtuple
from queue import Queue
from threading import Thread

import numpy as np
import pygmo as pg

from estimagic.dashboard.server_functions import run_server
from estimagic.optimization.process_constraints import process_constraints
from estimagic.optimization.reparametrize import reparametrize_from_internal
from estimagic.optimization.reparametrize import reparametrize_to_internal
from estimagic.optimization.utilities import index_tuple_to_string

QueueEntry = namedtuple("QueueEntry", ["params", "fitness", "still_running"])


def maximize(
    criterion,
    params,
    algorithm,
    criterion_args=None,
    criterion_kwargs=None,
    constraints=None,
    general_options=None,
    algo_options=None,
    dashboard=True,
    db_options=None,
):
    """
    Maximize *criterion* using *algorithm* subject to *constraints* and bounds.

    Args:
        criterion (function):
            Python function that takes a pandas Series with parameters as the first
            argument and returns a scalar floating point value.

        params (pd.DataFrame):
            See :ref:`params_df`.

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
    res_dict["f"] = -res_dict["f"]

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
    dashboard=True,
    db_options=None,
):
    """Minimize *criterion* using *algorithm* subject to *constraints* and bounds.

    Args:
        criterion (function):
            Python function that takes a pandas Series with parameters as the first
            argument and returns a scalar floating point value.

        params (pd.DataFrame):
            See :ref:`params_df`.

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

    params = _process_params_df(params)
    fitness_eval = criterion(params["value"], *criterion_args, **criterion_kwargs)
    constraints = process_constraints(constraints, params)
    internal_params = reparametrize_to_internal(params, constraints)

    queue = Queue() if dashboard is True else None
    start_signal = Queue() if dashboard is True else None
    if dashboard is True:
        # later only the parameter series can be supplied
        # but for the setup of the dashboard we want the whole DataFrame
        queue.put(QueueEntry(params=params, fitness=fitness_eval, still_running=True))

        # To-Do: Don't hard code the port
        server_thread = Thread(
            target=run_server,
            kwargs={
                "queue": queue,
                "port": 5039,
                "db_options": db_options,
                "start_signal": start_signal,
            },
            daemon=True,
        )
        server_thread.start()

    # wait for server_thread to give start signal
    while start_signal.qsize() == 0:
        pass

    result = _minimize(
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

    queue.put(QueueEntry(params=result[1], fitness=result[0]["f"], still_running=False))
    return result


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
            See :ref:`params_df`.

        internal_params (DataFrame):
            See :ref:`params_df`.

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

    prob = _create_problem(internal_criterion, internal_params)
    algo = _create_algorithm(algorithm, algo_options)
    pop = _create_population(prob, algo_options, internal_params)
    evolved = algo.evolve(pop)
    result = _process_pygmo_results(evolved, params, internal_params, constraints)
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
    def internal_criterion(x):
        params_sr = _params_sr_from_x(x, internal_params, constraints, params)
        fitness_eval = criterion(params_sr, *criterion_args, **criterion_kwargs)
        if queue is not None:
            queue.put(
                QueueEntry(params=params_sr, fitness=fitness_eval, still_running=True)
            )
        return [fitness_eval]

    return internal_criterion


def _params_sr_from_x(x, internal_params, constraints, params):
    internal_params = internal_params.copy(deep=True)
    internal_params["value"] = x
    params = reparametrize_from_internal(internal_params, constraints, params)
    return params


def _process_params_df(params):
    params = params.copy()
    if "lower" not in params.columns:
        params["lower"] = -np.inf
    if "upper" not in params.columns:
        params["upper"] = np.inf
    if "fixed" not in params.columns:
        # todo: does this have to be removed after we move fixed to constraints?
        params["fixed"] = False
    if "group" not in params.columns:
        params["group"] = "All Parameters"

    if "name" not in params.columns:
        names = [index_tuple_to_string(tup) for tup in params.index]
        params["name"] = names
    return params


def _create_problem(internal_criterion, internal_params):
    class Problem:
        def fitness(self, x):
            return internal_criterion(x)

        def get_bounds(self):
            return (internal_params["lower"], internal_params["upper"])

    return Problem()


def _create_algorithm(algorithm, algo_options):
    """Create a pygmo algorithm.

    Todo:
        - Pass the algo options through

    """
    with open(os.path.join(os.path.dirname(__file__), "algo_dict.json")) as j:
        algos = json.load(j)
    prefix, alg = algorithm.split("_", 1)

    assert alg in algos[prefix], "Invalid algorithm requested: {}".format(algorithm)

    if prefix == "nlopt":
        algo = pg.algorithm(pg.nlopt(solver=alg))
    elif prefix == "pygmo":
        pygmo_uda = getattr(pg, alg)
        algo = pg.algorithm(pygmo_uda())

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


def _process_pygmo_results(pygmo_res, params, internal_params, constraints):
    """Convert evolved population into json serializable dictionary.

    Todo:
        - implement this function.

    """
    x = pygmo_res.champion_x
    params = _params_sr_from_x(x, internal_params, constraints, params)

    res_dict = {
        "x": params.to_numpy().tolist(),
        "internal_x": x.tolist(),
        "f": pygmo_res.champion_f,
    }
    return res_dict, params
