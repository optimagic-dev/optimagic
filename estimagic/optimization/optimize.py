"""Functional wrapper around the object oriented pygmo library."""
import json
import os

import numpy as np
import pygmo as pg

from estimagic.optimization.process_constraints import process_constraints
from estimagic.optimization.reparametrize import reparametrize_from_internal
from estimagic.optimization.reparametrize import reparametrize_to_internal


def maximize(
    func,
    params,
    algorithm,
    func_args=None,
    func_kwargs=None,
    constraints=None,
    general_options=None,
    algo_options=None,
):
    """Maximize *func* using *algorithm* subject to *constraints* and bounds.

    Args:
        func (function):
            Python function that takes a pandas Series with parameters as the first
            argument and returns a scalar floating point value.

        params (pd.DataFrame):
            See :ref:`params_df`.

        algorithm (str):
            specifies the optimization algorithm. See :ref:`list_of_algorithms`.

        func_args (list or tuple):
            additional positional arguments for func

        func_kwargs (dict):
            additional keyword arguments for func

        constraints (list):
            list with constraint dictionaries. See for details.

    """

    def neg_func(*func_args, **func_kwargs):
        return -func(*func_args, **func_kwargs)

    res_dict, params = minimize(
        neg_func,
        params,
        algorithm,
        func_args,
        func_kwargs,
        constraints,
        general_options,
        algo_options,
    )
    res_dict["f"] = -res_dict["f"]

    return res_dict, params


def minimize(
    func,
    params,
    algorithm,
    func_args=None,
    func_kwargs=None,
    constraints=None,
    general_options=None,
    algo_options=None,
):
    """Minimize *func* using *algorithm* subject to *constraints* and bounds.

    Args:
        func (function):
            Python function that takes a pandas Series with parameters as the first
            argument and returns a scalar floating point value.

        params (pd.DataFrame):
            See :ref:`params_df`.

        algorithm (str):
            specifies the optimization algorithm. See :ref:`list_of_algorithms`.

        func_args (list or tuple):
            additional positional arguments for func

        func_kwargs (dict):
            additional keyword arguments for func

        constraints (list):
            list with constraint dictionaries. See for details.

    """
    params = _process_params_df(params)
    func_args = [] if func_args is None else func_args
    func_kwargs = {} if func_kwargs is None else func_kwargs
    constraints = [] if constraints is None else constraints
    general_options = {} if general_options is None else {}
    algo_options = {} if algo_options is None else {}

    prob = _create_problem(func, params, func_args, func_kwargs, constraints)
    algo = _create_algorithm(algorithm, algo_options)
    pop = _create_population(prob, algo_options)
    evolved = algo.evolve(pop)
    result = _process_pygmo_results(evolved, prob)
    return result


def _process_params_df(params):
    params = params.copy()
    if "lower" not in params.columns:
        params["lower"] = -np.inf
    if "upper" not in params.columns:
        params["upper"] = np.inf
    if "fixed" not in params.columns:
        # todo: does this have to be removed after we move fixed to constraints?
        params["fixed"] = False
    return params


def _create_problem(func, params, func_args, func_kwargs, constraints):
    class Problem:
        def __init__(self, func, params, func_args, func_kwargs, constraints):
            self.func = func
            self.params = params
            self.func_args = func_args
            self.func_kwargs = func_kwargs
            self.constraints = process_constraints(constraints, params)
            self.index = params.index

            internal_params = reparametrize_to_internal(params, self.constraints)
            self.internal_params = internal_params
            self.internal_index = internal_params.index

        def _params_sr_from_x(self, x):
            internal_params = self.internal_params.copy(deep=True)
            internal_params["value"] = x
            params = reparametrize_from_internal(
                internal_params, self.constraints, self.params
            )
            return params

        def fitness(self, x):
            params = self._params_sr_from_x(x)
            return [func(params, *self.func_args, **self.func_kwargs)]

        def get_bounds(self):
            return (self.internal_params["lower"], self.internal_params["upper"])

    return Problem(func, params, func_args, func_kwargs, constraints)


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


def _create_population(problem, algo_options):
    """Create a pygmo population object.

    Todo:
        - constrain random initial values to be in some bounds
        - remove hardcoded seed

    """
    popsize = algo_options.copy().pop("popsize", 1)
    x0 = problem.internal_params["value"].to_numpy()
    pop = pg.population(problem, size=popsize - 1, seed=5471)
    pop.push_back(x0)
    return pop


def _process_pygmo_results(pygmo_res, prob):
    """Convert evolved population into json serializable dictionary.

    Todo:
        - implement this function.

    """
    x = pygmo_res.champion_x
    params = prob._params_sr_from_x(x)

    res_dict = {
        "x": params.to_numpy().tolist(),
        "internal_x": x.tolist(),
        "f": pygmo_res.champion_f,
    }
    return res_dict, params
