import pygmo as pg

from estimagic.config import DEFAULT_SEED


def minimize_pygmo_np(func, x0, bounds, origin, algo_name, algo_options):
    """Minimize a function with pygmo.

    Args:
        func (callable): Objective function.
        x0 (np.ndarray): Starting values of the parameters.
        bounds (Tuple[np.ndarray]): A tuple containing two NumPy arrays where the first
            corresponds to the lower and the second to the upper bound. Unbounded
            parameters are represented by infinite values. The arrays have the same
            length as the parameter vector.
        origin ({"nlopt", "pygmo"}): Either an optimizer from NLOPT or pygmo.
        algo_name (str): One of the optimizers of the pygmo package.
        algo_options (dict): Options for the optimizer.

    Returns:
        results (dict): Dictionary with processed optimization results.

    """
    if origin == "pygmo" and algo_name != "simulated_annealing":
        assert (
            "popsize" in algo_options
        ), f"For genetic optimizers like {algo_name}, popsize is mandatory."
        assert (
            "gen" in algo_options
        ), f"For genetic optimizers like {algo_name}, gen is mandatory."

    prob = _create_problem(func, bounds)
    algo = _create_algorithm(algo_name, algo_options, origin)
    pop = _create_population(prob, algo_options, x0)
    evolved = algo.evolve(pop)
    result = _process_pygmo_results(evolved)

    return result


def _create_problem(func, bounds):
    class Problem:
        def fitness(self, x):
            return [func(x)]

        def get_bounds(self):
            return bounds

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


def _create_population(problem, algo_options, x0):
    """Create a pygmo population object.

    Args:
        problem (pygmo.Problem)
        algo_options (dict)
        x0 (np.ndarray)

    Todo:
        - constrain random initial values to be in some bounds

    """
    popsize = algo_options.copy().pop("popsize", 1) - 1
    pop = pg.population(
        problem, size=popsize, seed=algo_options.get("seed", DEFAULT_SEED)
    )
    pop.push_back(x0)
    return pop


def _process_pygmo_results(evolved):
    results = {
        # Harmonized results.
        "criterion": evolved.champion_f[0],
        "x": evolved.champion_x,
        "n_evaluations": evolved.problem.get_fevals(),
        # Other results.
        "n_evaluations_jacobian": evolved.problem.get_gevals(),
        "n_evaluations_hessian": evolved.problem.get_hevals(),
        "n_constraints": evolved.problem.get_nc(),
        "n_constraints_equality": evolved.problem.get_nec(),
        "n_constraints_inequality": evolved.problem.get_nic(),
    }

    return results
