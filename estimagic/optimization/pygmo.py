import pygmo as pg

from estimagic.config import DEFAULT_SEED


def minimize_pygmo(
    internal_criterion, params, internal_params, origin, algo_name, algo_options,
):
    if origin == "pygmo" and algo_name != "simulated_annealing":
        assert (
            "popsize" in algo_options
        ), f"For genetic optimizers like {algo_name}, popsize is mandatory."
        assert (
            "gen" in algo_options
        ), f"For genetic optimizers like {algo_name}, gen is mandatory."

    prob = _create_problem(internal_criterion, params)
    algo = _create_algorithm(algo_name, algo_options, origin)
    pop = _create_population(prob, algo_options, internal_params)
    evolved = algo.evolve(pop)
    result = _process_pygmo_results(evolved)

    return result


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
    """Create a pygmo algorithm.

    Todo: There should be a simplification which works for both, pygmo and nlopt.

    """
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

    """
    popsize = algo_options.copy().pop("popsize", 1) - 1
    pop = pg.population(
        problem, size=popsize, seed=algo_options.get("seed", DEFAULT_SEED)
    )
    pop.push_back(internal_params)
    return pop


def _process_pygmo_results(evolved):
    results = {"fun": evolved.champion_f[0], "x": evolved.champion_x}

    return results
