"""Functional wrapper around the object oriented pygmo library."""


import pygmo as pg
import pandas as pd
import os
import json
from estimagic.optimization.process_constraints import process_constraints
from estimagic.optimization.reparametrize import reparametrize_to_internal, reparametrize_from_internal


def minimize(
    func, params, algorithm, func_args=[], func_kwargs={}, constraints=[], general_options={}, algo_options={}
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
    prob = _create_problem(func, params, func_args, func_kwargs, constraints)
    algo = _create_algorithm(algorithm, algo_options)
    pop = _create_population(prob, algo_options)
    evolved = algo.evolve(pop)
    result = _process_pygmo_results(evolved)
    return result


def _create_problem(func, params, func_args, func_kwargs, constraints):
    class Problem:
        def __init__(self, func, params, func_args, func_kwargs, constraints):
            self.func = func
            self.func_args = func_args
            self.func_kwargs = func_kwargs
            self.constraints = process_constraints(constraints, params)

            self.params = params
            self.index = params.index

            internal_params = reparametrize_to_internal(params, self.constraints)
            self.internal_params = internal_params
            self.internal_index = internal_params.index

        def fitness(self, x):
            internal_params = pd.Series(data=x, index=self.internal_index)
            params = reparametrize_from_internal(
                internal_params, self.constraints, self.params)
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
    x0 = problem.internal_params['value'].to_numpy()
    pop = pg.population(problem, size=popsize - 1, seed=5471)
    pop.push_back(x0)
    return pop


def _process_pygmo_results(pygmo_res):
    """Convert evolved population into json serializable dictionary.

    Todo:
        - implement this function.

    """
    return pygmo_res
