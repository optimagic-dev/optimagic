"""Functional wrapper around the pygmo, nlopt and scipy libraries."""
import json
from collections import namedtuple
from multiprocessing import Event
from multiprocessing import Process
from multiprocessing import Queue
from pathlib import Path

import numpy as np
import pandas as pd
import pygmo as pg
from scipy.optimize import minimize as scipy_minimize

from multiprocessing import Pool

from pathos.multiprocessing import ProcessingPool
from itertools import repeat

import dill

from estimagic.dashboard.server_functions import run_server
from estimagic.differentiation.differentiation import gradient
from estimagic.optimization.process_constraints import process_constraints
from estimagic.optimization.process_arguments import process_optimization_arguments
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
    Run several optimizations if called by lists of inputs.

    Args:
        criterion (function or list of functions):
            Python function that takes a pandas Series with parameters as the first
            argument and returns a scalar floating point value.

        params (pd.DataFrame or list of pd.DataFrames):
            See :ref:`params`.

        algorithm (str or list of strings):
            specifies the optimization algorithm. See :ref:`list_of_algorithms`.

        criterion_args (list)::
            additional positional arguments for criterion

        criterion_kwargs (dict or list of dicts):
            additional keyword arguments for criterion

        constraints (list or list of lists):
            list with constraint dictionaries. See for details.

        general_options (dict):
            additional configurations for the optimization

        algo_options (dict or list of dicts):
            algorithm specific configurations for the optimization

        dashboard (bool):
            whether to create and show a dashboard

        db_options (dict):
            dictionary with kwargs to be supplied to the run_server function.

    """

    arguments = process_optimization_arguments(
        criterion=criterion,
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
    n_opts = (
        1 if isinstance(arguments["params"], pd.DataFrame) else len(arguments["params"])
    )

    if n_opts == 1:
        result = _single_minimize(**arguments)
    else:
        # set up pool
        if not "n_cores" in arguments["general_options"][0]:
            raise ValueError(
                "n_cores need to be specified if multiple optimizations should be run."
            )
        n_cores = arguments["general_options"][0]["n_cores"]
        pool = Pool(processes=n_cores)

        result = pool.starmap(_single_minimize, map(list, zip(*arguments.values())))

    return result


def _single_minimize(
    criterion,
    params,
    algorithm,
    criterion_args,
    criterion_kwargs,
    constraints,
    general_options,
    algo_options,
    dashboard,
    db_options,
):
    """Minimize * criterion * using * algorithm * subject to * constraints * and bounds. Only one minimization.

    Args:
        criterion(function):
            Python function that takes a pandas Series with parameters as the first
            argument and returns a scalar floating point value.

        params(pd.DataFrame):
            See: ref: `params`.

        algorithm(str):
            specifies the optimization algorithm. See: ref: `list_of_algorithms`.

        criterion_args(list or tuple):
            additional positional arguments for criterion

        criterion_kwargs(dict):
            additional keyword arguments for criterion

        constraints(list):
            list with constraint dictionaries. See for details.

        general_options(dict):
            additional configurations for the optimization

        algo_options(dict):
            algorithm specific configurations for the optimization

        dashboard(bool):
            whether to create and show a dashboard

        db_options(dict):
            dictionary with kwargs to be supplied to the run_server function.

    """
    params = _process_params(params)
    fitness_eval = criterion(params, *criterion_args, **criterion_kwargs)
    constraints = process_constraints(constraints, params)
    internal_params = reparametrize_to_internal(params, constraints, None)

    scaling_factor = calculate_scaling_factor(
        criterion,
        params,
        internal_params,
        constraints,
        general_options,
        criterion_args,
        criterion_kwargs,
    )

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
        criterion_args=criterion_args,
        criterion_kwargs=criterion_kwargs,
        params=params,
        internal_params=internal_params,
        constraints=constraints,
        scaling_factor=scaling_factor,
        algorithm=algorithm,
        algo_options=algo_options,
        general_options=general_options,
        queue=queue,
    )

    if dashboard:
        stop_signal.set()
        outer_server_process.terminate()
    return result


def _internal_minimize(
    criterion,
    criterion_args,
    criterion_kwargs,
    params,
    internal_params,
    constraints,
    scaling_factor,
    algorithm,
    algo_options,
    general_options,
    queue,
):
    """
    Create the internal criterion function and minimize it.

    Args:
        criterion(function):
            Python function that takes a pandas Series with parameters as the first
            argument and returns a scalar floating point value.

        criterion_args(list or tuple):
            additional positional arguments for criterion

        criterion_kwargs(dict):
            additional keyword arguments for criterion

        params(pd.DataFrame):
            See: ref: `params`.

        internal_params(DataFrame):
            See: ref: `params`.

        constraints(list):
            list with constraint dictionaries. See for details.

        algorithm(str):
            specifies the optimization algorithm. See: ref: `list_of_algorithms`.

        algo_options(dict):
            algorithm specific configurations for the optimization

        general_options(dict):
            additional configurations for the optimization

        queue(Queue):
            queue to which originally the parameters DataFrame is supplied and to which
            the updated parameter Series will be supplied later.

    """
    internal_criterion = create_internal_criterion(
        criterion=criterion,
        params=params,
        internal_params=internal_params,
        constraints=constraints,
        scaling_factor=scaling_factor,
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
        result = _process_results(
            evolved, params, internal_params, constraints, origin, scaling_factor
        )
    elif origin == "scipy":
        bounds = _get_scipy_bounds(internal_params)
        x0 = _x_from_params(params, constraints, scaling_factor)
        minimized = scipy_minimize(
            internal_criterion,
            x0,
            method=algo_name,
            bounds=bounds,
            options=algo_options,
        )
        result = _process_results(
            minimized, params, internal_params, constraints, origin, scaling_factor
        )
    else:
        raise ValueError("Invalid algorithm requested.")

    return result


def create_internal_criterion(
    criterion,
    params,
    internal_params,
    constraints,
    scaling_factor,
    criterion_args,
    criterion_kwargs,
    queue,
):
    c = np.ones(1, dtype=int)

    def internal_criterion(x, counter=c):
        p = _params_from_x(x, internal_params, constraints, params, scaling_factor)
        fitness_eval = criterion(p, *criterion_args, **criterion_kwargs)
        if queue is not None:
            queue.put(QueueEntry(iteration=counter[0], params=p, fitness=fitness_eval))
        counter += 1
        return fitness_eval

    return internal_criterion


def _params_from_x(x, internal_params, constraints, params, scaling_factor):
    internal_params = internal_params.copy(deep=True)
    # :func:`internal_criterion` always assumes that `x` is a NumPy array, but if we
    # pass the internal criterion function to :func:`gradient`, x is a DataFrame.
    # Setting a series to a DataFrame will convert the column "value" to object type
    # which causes trouble in following NumPy routines assuming a numeric type.
    internal_params["value"] = x["value"] if isinstance(x, pd.DataFrame) else x
    updated_params = reparametrize_from_internal(
        internal_params, constraints, params, scaling_factor
    )
    return updated_params


def _x_from_params(params, constraints, scaling_factor):
    return reparametrize_to_internal(params, constraints, scaling_factor)[
        "value"
    ].to_numpy()


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


def _process_results(res, params, internal_params, constraints, origin, scaling_factor):
    """Convert optimization results into json serializable dictionary.
    Args:
        res: Result from numerical optimizer.
        params(DataFrame): See: ref: `params`.
        internal_params(DataFrame): See: ref: `params`.
        constraints(list): constraints for the optimization
        origin(str): takes the values "pygmo", "nlopt", "scipy"
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
    params = _params_from_x(x, internal_params, constraints, params, scaling_factor)

    res_dict["internal_x"] = x.tolist()
    return res_dict, params


def calculate_scaling_factor(
    criterion,
    params,
    internal,
    constraints,
    general_options,
    criterion_args,
    criterion_kwargs,
):
    """Calculate the scaling factor for the internal parameters.

    There are multiple ways the user is able to rescale the parameter vector which is
    specified in ``general_options["scaling"]``.

    - ``None`` (default): No scaling happens.
    - ``"start_values"``: Divide parameters which are not in ``[-1, 1]`` by their
      starting values.
    - ``"gradient"``: Divide parameters which are not in ``[-1e-2, 1e-2]`` by the
      inverse of the gradient. By default, the computation method is ``"central"`` with
      extrapolation set to ``False``. The user can change the defaults by passing
      ``scaling_gradient_method`` or ``scaling_gradient_extrapolation``. See
      : func: `~estimagic.differentiation.differentiation.gradient` for more details.

    Note that the scaling factor should be defined such that unscaling is done by
    multiplying the scaling factor. This simplifies: func: `_rescale_from_internal`.

    Args:
        criterion(func): Criterion function.
        params(DataFrame): See: ref: `params`.
        internal(DataFrame): See: ref: `params`.
        constraints(dict): Dictionary containing constraints.
        general_options(dict): General options. See: ref: `estimation_general_options`.
        criterion_args(list): List of arguments of the criterion function.
        crtierion_kwargs(dict): Dictionary of keyword arguments of the criterion
            function.

    Returns:
        scaling_factor(Series): Scaling Factor.

    """
    scaling = general_options.get("scaling", None)

    if scaling is None:
        scaling_factor = np.ones(len(internal))
    elif scaling == "start_values":
        scaling_factor = internal["value"].abs().clip(1)
    elif scaling == "gradient":
        method = general_options.get("scaling_gradient_method", "central")
        extrapolation = general_options.get("scaling_gradient_extrapolation", False)

        internal_criterion = create_internal_criterion(
            criterion,
            params,
            internal,
            constraints,
            None,
            criterion_args,
            criterion_kwargs,
            queue=None,
        )

        gradients = gradient(
            internal_criterion,
            internal,
            method,
            extrapolation,
            criterion_args,
            criterion_kwargs,
        )
        scaling_factor = np.clip(1 / gradients.abs(), 1e-2, None)
    else:
        raise NotImplementedError(f"Scaling method {scaling} is not implemented.")

    scaling_factor = pd.Series(scaling_factor, index=internal.index)

    return scaling_factor
