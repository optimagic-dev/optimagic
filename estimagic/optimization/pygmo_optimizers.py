import functools
import warnings

import numpy as np
import pygmo as pg

from estimagic import batch_evaluators
from estimagic.optimization.algo_options import CONVERGENCE_RELATIVE_PARAMS_TOLERANCE
from estimagic.optimization.algo_options import STOPPING_MAX_CRITERION_EVALUATIONS

STOPPING_MAX_ITERATIONS_GENETIC = 1000


def pygmo_gaco(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    population_size=None,
    batch_evaluator=None,
    n_cores=1,
    seed=None,
    discard_start_params=False,
    #
    stopping_max_iterations=STOPPING_MAX_ITERATIONS_GENETIC,
    kernel_size=63,
    convergence_speed=1.0,
    oracle=0.0,
    accuracy=0.01,
    threshold=1,
    std_convergence_speed=7,
    stopping_max_n_without_improvements=100000,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS,
    focus=0.0,
    activate_memory_for_multiple_calls=False,
):
    """Minimize a scalar function using the extended ant colony algorithm.

    The version available through pygmo is an extended version of the original ant
    colony algorithm proposed by :cite:`Schlueter2009`.

    This algorithm can be applied to box-bounded problems.

    Ant colony optimization is a class of optimization algorithms modeled on the actions
    of an ant colony. Artificial ‘ants’ (e.g. simulation agents) locate optimal
    solutions by moving through a parameter space representing all possible solutions.
    Real ants lay down pheromones directing each other to resources while exploring
    their environment. The simulated ‘ants’ similarly record their positions and the
    quality of their solutions, so that in later simulation iterations more ants locate
    better solutions.

    The extended ant colony algorithm generates future generations of ants by using the
    a multi-kernel gaussian distribution based on three parameters (i.e., pheromone
    values) which are computed depending on the quality of each previous solution. The
    solutions are ranked through an oracle penalty method.

    - population_size (int): Size of the population. If None, it's twice the number of
      parameters but at least 64.
    - batch_evaluator (str or Callable): Name of a pre-implemented batch evaluator
      (currently 'joblib' and 'pathos_mp') or Callable with the same interface as the
      estimagic batch_evaluators. See :ref:`batch_evaluators`.
    - n_cores (int): Number of cores to use.
    - seed (int): seed used by the internal random number generator.
    - discard_start_params (bool): If True, the start params are not guaranteed to be
      part of the initial population. This saves one criterion function evaluation that
      cannot be done in parallel with other evaluations. Default False.

    - stopping_max_iterations (int): Number of generations to evolve.
    - kernel_size (int): Number of solutions stored in the solution archive.
    - convergence_speed (float): This parameter is useful for managing the convergence
      speed towards the found minima (the smaller the faster).
    - oracle (float): oracle parameter used in the penalty method.
    - accuracy (float): accuracy parameter for maintaining a minimum penalty function's
      values distances.
    - threshold (int): when the generations reach the threshold then the convergence
      speed is set to 0.01 automatically.
    - std_convergence_speed (int): parameter that determines the convergence speed of
      the standard deviations.
    - stopping.max_n_without_improvements (int): if a positive integer is assigned here,
      the algorithm will count the runs without improvements, if this number exceeds the
      given value, the algorithm will be stopped.
    - stopping.max_criterion_evaluations (int): If the maximum number of function
      evaluation is reached, the optimization stops but we do not count this as
      successful convergence.
    - focus (float): this parameter makes the search for the optimum greedier and more
      focused on local improvements (the higher the greedier). If the value is very
      high, the search is more focused around the current best solutions.
    - activate_memory_for_multiple_calls (bool): if true, memory is activated in the
      algorithm for multiple calls.

    """
    _check_that_every_param_is_bounded(lower_bounds, upper_bounds)

    population_size = _determine_population_size(population_size=population_size, x=x)

    algo_specific_options = {
        "gen": stopping_max_iterations,
        "ker": kernel_size,
        "q": convergence_speed,
        "oracle": oracle,
        "acc": accuracy,
        "threshold": threshold,
        "n_gen_mark": std_convergence_speed,
        "impstop": stopping_max_n_without_improvements,
        "evalstop": stopping_max_criterion_evaluations,
        "focus": focus,
        "memory": activate_memory_for_multiple_calls,
    }
    algo_options = _create_algo_options(
        population_size=population_size,
        n_cores=n_cores,
        seed=seed,
        discard_start_params=discard_start_params,
        batch_evaluator=batch_evaluator,
        algo_specific_options=algo_specific_options,
    )

    res = _minimize_pygmo(
        criterion_and_derivative=criterion_and_derivative,
        x=x,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        method="gaco",
        algo_options=algo_options,
    )
    return res


def pygmo_bee_colony(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    population_size=None,
    batch_evaluator=None,
    n_cores=1,
    seed=None,
    discard_start_params=False,
    max_n_trials=1,
):
    """Minimize a scalar function using the artifical bee colony algorithm.

    The Artificial Bee Colony Algorithm was originally proposed by :cite:`Karaboga2007`.

    The implemented version of the algorithm is proposed in :cite:`Mernik2015`.

    - population_size (int): Size of the population. If None, it's twice the number of
      parameters but at least 64.
    - batch_evaluator (str or Callable): Name of a pre-implemented batch evaluator
      (currently 'joblib' and 'pathos_mp') or Callable with the same interface as the
      estimagic batch_evaluators. See :ref:`batch_evaluators`.
    - n_cores (int): Number of cores to use.
    - seed (int): seed used by the internal random number generator.
    - discard_start_params (bool): If True, the start params are not guaranteed to be
      part of the initial population. This saves one criterion function evaluation that
      cannot be done in parallel with other evaluations. Default False.
    - max_n_trials (int): Maximum number of trials for abandoning a source.

    """
    population_size = _determine_population_size(population_size=population_size, x=x)

    algo_options = _create_algo_options(
        population_size=population_size,
        n_cores=n_cores,
        seed=seed,
        discard_start_params=discard_start_params,
        batch_evaluator=batch_evaluator,
        algo_specific_options={"limit": max_n_trials},
    )

    res = _minimize_pygmo(
        criterion_and_derivative=criterion_and_derivative,
        x=x,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        method="bee_colony",
        algo_options=algo_options,
    )
    return res


def pygmo_de(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    population_size=None,
    batch_evaluator=None,
    n_cores=1,
    seed=None,
    discard_start_params=False,
    stopping_max_iterations=STOPPING_MAX_ITERATIONS_GENETIC,
    weight_coefficient=0.8,
    crossover_probability=0.9,
    mutation_variant=2,
    criterion_tolerance=1e-6,
    convergence_relative_params_tolerance=CONVERGENCE_RELATIVE_PARAMS_TOLERANCE,
):
    """Minimize a scalar function using the differential evolution algorithm.

    Differential Evolution is an heuristic optimizer originally presented in
    :cite:`Storn1997`.

    - population_size (int): Size of the population. If None, it's twice the number of
      parameters but at least 64.
    - batch_evaluator (str or Callable): Name of a pre-implemented batch evaluator
      (currently 'joblib' and 'pathos_mp') or Callable with the same interface as the
      estimagic batch_evaluators. See :ref:`batch_evaluators`.
    - n_cores (int): Number of cores to use.
    - seed (int): seed used by the internal random number generator.
    - discard_start_params (bool): If True, the start params are not guaranteed to be
      part of the initial population. This saves one criterion function evaluation that
      cannot be done in parallel with other evaluations. Default False.
    - stopping_max_iterations (int): Number of generations to evolve.
    - weight_coefficient (float): Weight coefficient. It is denoted by $F$ in the main
      paper and must lie in [0, 2]. It controls the amplification of the differential
      variation $(x_{r_2, G} - x_{r_3, G})$.
    - crossover_probability (float): Crossover probability.
    - mutation_variant (int): code for the mutation variant to create a new candidate
      individual. The default is 2. The following are available:

        - 1:   best/1/exp
        - 2:   rand/1/exp
        - 3:   rand-to-best/1/exp
        - 4:   best/2/exp
        - 5:   rand/2/exp
        - 6:   best/1/bin
        - 7:   rand/1/bin
        - 8:   rand-to-best/1/bin
        - 9:   best/2/bin
        - 10:  rand/2/bin

    - criterion_tolerance: stopping criteria on the f tolerance. Default is 1e-6. It is
      not clear whether this is the absolute or relative criterion tolerance.
    - convergence_relative_params_tolerance: stopping criteria on the x tolerance. In
      pygmo the default is 1e-6 but we use our default value of 1e-5.

    """
    population_size = _determine_population_size(population_size=population_size, x=x)

    algo_specific_options = {
        "gen": stopping_max_iterations,
        "F": weight_coefficient,
        "CR": crossover_probability,
        "variant": mutation_variant,
        "ftol": criterion_tolerance,
        "xtol": convergence_relative_params_tolerance,
    }
    algo_options = _create_algo_options(
        population_size=population_size,
        n_cores=n_cores,
        seed=seed,
        discard_start_params=discard_start_params,
        batch_evaluator=batch_evaluator,
        algo_specific_options=algo_specific_options,
    )

    res = _minimize_pygmo(
        criterion_and_derivative=criterion_and_derivative,
        x=x,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        method="de",
        algo_options=algo_options,
    )
    return res


def pygmo_sea(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    population_size=None,
    batch_evaluator=None,
    n_cores=1,
    seed=None,
    discard_start_params=False,
    stopping_max_iterations=STOPPING_MAX_ITERATIONS_GENETIC,
):
    """Minimize a scalar function using the (N+1)-ES simple evolutionary algorithm.

    This algorithm represents the simplest evolutionary strategy, where a population of
    $\lambda$ individuals at each generation produces one offspring by mutating its best
    individual uniformly at random within the bounds. Should the offspring be better
    than the worst individual in the population it will substitute it.

    See :cite:`Oliveto2007`.

    The algorithm is only suited for bounded parameter spaces.

    - population_size (int): Size of the population. If None, it's twice the number of
      parameters but at least 64.
    - batch_evaluator (str or Callable): Name of a pre-implemented batch evaluator
      (currently 'joblib' and 'pathos_mp') or Callable with the same interface as the
      estimagic batch_evaluators. See :ref:`batch_evaluators`.
    - n_cores (int): Number of cores to use.
    - seed (int): seed used by the internal random number generator.
    - discard_start_params (bool): If True, the start params are not guaranteed to be
      part of the initial population. This saves one criterion function evaluation that
      cannot be done in parallel with other evaluations. Default False.
    - stopping_max_iterations (int): number of generations to consider. Each generation
      will compute the objective function once.

    """
    _check_that_every_param_is_bounded(lower_bounds, upper_bounds)
    population_size = _determine_population_size(population_size=population_size, x=x)

    algo_options = _create_algo_options(
        population_size=population_size,
        n_cores=n_cores,
        seed=seed,
        discard_start_params=discard_start_params,
        batch_evaluator=batch_evaluator,
        algo_specific_options={"gen": stopping_max_iterations},
    )

    res = _minimize_pygmo(
        criterion_and_derivative=criterion_and_derivative,
        x=x,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        method="sea",
        algo_options=algo_options,
    )
    return res


# ====================================================================================


def _minimize_pygmo(
    criterion_and_derivative, x, lower_bounds, upper_bounds, method, algo_options=None
):
    """Minimize a function with pygmo.

    Args:
        criterion_and_derivative (callable):
        x (np.ndarray): Starting values of the parameters.
        lower_bounds (np.ndarray):
        upper_bounds (np.ndarray):
        method (str): One of the optimizers of the pygmo package.
        algo_options (dict): Options for the optimizer. In addition to
            the algo options that will be passed directly to the pygmo
            algorithms we have the following entries:
            - population_size (int): Population size for genetic algorithms.
            - batch_evaluator (str or callable): An estimagic batch evaluator,
                default joblib batch evaluator.
            - n_cores (int): Number of cores used for parallel evaluation of
                the criterion function. Default 1.
            - seed (int or None): Random seed for drawing the initial
                population.
            - discard_start_params (bool): If True, the start params are not
                guaranteed to be part of the initial population. This saves one
                criterion function evaluation that cannot be done in parallel
                with other evaluations. Default False.

    Returns:
        results (dict): Dictionary with optimization results.

    """
    algo_options = {} if algo_options is None else algo_options.copy()
    population_size = algo_options.pop("population_size", 1)

    batch_evaluator = algo_options.pop("batch_evaluator", "joblib_batch_evaluator")
    if isinstance(batch_evaluator, str):
        batch_evaluator = getattr(batch_evaluators, batch_evaluator)
    n_cores = algo_options.pop("n_cores", 1)
    seed = algo_options.pop("seed", None)
    discard_start_params = algo_options.pop("discard_start_params", False)

    algo_info = {
        "parallelizes": n_cores != 1,
        "name": f"pygmo_{method}",
        "needs_scaling": True,
        "primary_criterion_entry": "value",
    }
    func = functools.partial(
        criterion_and_derivative, task="criterion", algorithm_info=algo_info
    )
    gradient = functools.partial(
        criterion_and_derivative, task="derivative", algorithm_info=algo_info
    )

    bounds = (lower_bounds, upper_bounds)
    prob = _create_problem(
        func=func,
        bounds=bounds,
        gradient_=gradient,
        dim=len(x),
        batch_evaluator=batch_evaluator,
        n_cores=n_cores,
    )
    algo = _create_algorithm(method, algo_options, n_cores)
    pop = _create_population(
        prob, population_size, x, seed=seed, discard_start_params=discard_start_params
    )
    evolved = algo.evolve(pop)
    result = _process_pygmo_results(evolved)

    return result


def _create_problem(func, bounds, gradient_, dim, batch_evaluator, n_cores):
    class Problem:
        def fitness(self, x):
            return [func(x)]

        def get_bounds(self):
            return bounds

        def gradient(self, dv):
            return gradient_(dv)

        def batch_fitness(self, dvs):
            dv_list = dvs.reshape(-1, dim)
            eval_list = batch_evaluator(
                func=func,
                arguments=dv_list,
                n_cores=n_cores,
                # Error handling is done on a higher level
                error_handling="raise",
            )
            evals = np.array(eval_list)
            return evals

    problem = pg.problem(Problem())
    return problem


def _create_algorithm(method, algo_options, n_cores):
    """Create a pygmo algorithm."""
    pygmo_uda = getattr(pg, method)
    algo = pygmo_uda(**algo_options)
    try:
        algo.set_bfe(pg.bfe())
    except AttributeError:
        if n_cores >= 2:
            warnings.warn(
                f"Your specified algorithm {method} does not support parallelization. "
                "Choose another algorithm such as pygmo_gaco to parallelize."
            )
    out = pg.algorithm(algo)
    return out


def _create_population(problem, population_size, x, seed, discard_start_params):
    """Create a pygmo population object.
    Args:
        problem (pygmo.Problem)
        algo_options (dict)
        x (np.ndarray)
    Todo:
        - constrain random initial values to be in some bounds
    """
    if not discard_start_params:
        population_size = population_size - 1

    pop = pg.population(
        problem,
        size=population_size,
        seed=seed,
        b=pg.bfe(),
    )
    if not discard_start_params:
        pop.push_back(x)
    return pop


def _process_pygmo_results(evolved):
    results = {
        # Harmonized results.
        "solution_x": evolved.champion_x,
        "solution_criterion": evolved.champion_f[0],
        "solution_derivative": None,
        "solution_hessian": None,
        "n_criterion_evaluations": evolved.problem.get_fevals(),
        "n_derivative_evaluations": evolved.problem.get_gevals(),
        "n_iterations": None,
        "success": True,
        "reached_convergence_criterion": "Number of generations reached.",
        "message": None,
    }

    return results


def _create_algo_options(
    population_size,
    n_cores,
    seed,
    discard_start_params,
    batch_evaluator,
    algo_specific_options,
):
    algo_options = {
        "population_size": population_size,
        "n_cores": n_cores,
        "seed": seed,
        "discard_start_params": discard_start_params,
    }
    if batch_evaluator is not None:
        algo_options["batch_evaluator"] = batch_evaluator

    algo_options.update(**algo_specific_options)
    return algo_options


def _check_that_every_param_is_bounded(lower_bounds, upper_bounds):
    assert np.isfinite(lower_bounds).all(), "The lower bounds must all be finite."
    assert np.isfinite(upper_bounds).all(), "The upper bounds must all be finite."


def _determine_population_size(population_size, x):
    if population_size is None:
        population_size = int(np.clip(2 * len(x), 64, np.inf))
    return population_size
