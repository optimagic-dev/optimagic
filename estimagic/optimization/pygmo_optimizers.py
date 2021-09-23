"""Implement pygmo optimizers."""
import functools
import warnings

import numpy as np

from estimagic import batch_evaluators
from estimagic.config import IS_PYGMO_INSTALLED
from estimagic.optimization.algo_options import CONVERGENCE_RELATIVE_PARAMS_TOLERANCE
from estimagic.optimization.algo_options import STOPPING_MAX_CRITERION_EVALUATIONS

STOPPING_MAX_ITERATIONS_GENETIC = 1000

try:
    import pygmo as pg
except ImportError:
    pass


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
    speed_parameter_q=1.0,
    oracle=0.0,
    accuracy=0.01,
    threshold=1,
    speed_of_std_values_convergence=7,
    stopping_max_n_without_improvements=100000,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS,
    focus=0.0,
    cache=False,
):
    """Minimize a scalar function using the generalized ant colony algorithm.

    The version available through pygmo is an generalized version of the
    original ant colony algorithm proposed by :cite:`Schlueter2009`.

    This algorithm can be applied to box-bounded problems.

    Ant colony optimization is a class of optimization algorithms modeled on the
    actions of an ant colony. Artificial "ants" (e.g. simulation agents) locate
    optimal solutions by moving through a parameter space representing all
    possible solutions. Real ants lay down pheromones directing each other to
    resources while exploring their environment. The simulated "ants" similarly
    record their positions and the quality of their solutions, so that in later
    simulation iterations more ants locate better solutions.

    The generalized ant colony algorithm generates future generations of ants by
    using a multi-kernel gaussian distribution based on three parameters (i.e.,
    pheromone values) which are computed depending on the quality of each
    previous solution. The solutions are ranked through an oracle penalty
    method.

    - population_size (int): Size of the population. If None, it's twice the
      number of parameters but at least 64.
    - batch_evaluator (str or Callable): Name of a pre-implemented batch
      evaluator (currently 'joblib' and 'pathos_mp') or Callable with the same
      interface as the estimagic batch_evaluators. See :ref:`batch_evaluators`.
    - n_cores (int): Number of cores to use.
    - seed (int): seed used by the internal random number generator.
    - discard_start_params (bool): If True, the start params are not guaranteed
      to be part of the initial population. This saves one criterion function
      evaluation that cannot be done in parallel with other evaluations. Default
      False.

    - stopping.max_iterations (int): Number of generations to evolve.
    - kernel_size (int): Number of solutions stored in the solution archive.
    - speed_parameter_q (float): This parameter manages the convergence speed
      towards the found minima (the smaller the faster). In the pygmo
      documentation it is referred to as $q$. It must be positive and can be
      larger than 1. The default is 1.0 until **threshold** is reached. Then it
      is set to 0.01.
    - oracle (float): oracle parameter used in the penalty method.
    - accuracy (float): accuracy parameter for maintaining a minimum penalty
      function's values distances.
    - threshold (int): when the iteration counter reaches the threshold the
      convergence speed is set to 0.01 automatically. To deactivate this effect
      set the threshold to stopping.max_iterations which is the largest allowed
      value.
    - speed_of_std_values_convergence (int): parameter that determines the
      convergence speed of the standard deviations. This must be an integer
      (`n_gen_mark` in pygmo and pagmo).
    - stopping.max_n_without_improvements (int): if a positive integer is
      assigned here, the algorithm will count the runs without improvements, if
      this number exceeds the given value, the algorithm will be stopped.
    - stopping.max_criterion_evaluations (int): maximum number of function
      evaluations.
    - focus (float): this parameter makes the search for the optimum greedier
      and more focused on local improvements (the higher the greedier). If the
      value is very high, the search is more focused around the current best
      solutions. Values larger than 1 are allowed.
    - cache (bool): if True, memory is activated in the algorithm for multiple calls.

    """
    _check_that_every_param_is_bounded(lower_bounds, upper_bounds)

    population_size = _determine_population_size(
        population_size=population_size, x=x, lower_bound=64
    )

    if isinstance(speed_of_std_values_convergence, float):
        if not speed_of_std_values_convergence.is_integer():
            raise ValueError(
                "The speed_of_std_values_convergence parameter must be an integer. "
                f"You specified {speed_of_std_values_convergence}."
            )

    algo_specific_options = {
        "gen": int(stopping_max_iterations),
        "ker": kernel_size,
        "q": speed_parameter_q,
        "oracle": oracle,
        "acc": accuracy,
        "threshold": threshold,
        "n_gen_mark": int(speed_of_std_values_convergence),
        "impstop": stopping_max_n_without_improvements,
        "evalstop": stopping_max_criterion_evaluations,
        "focus": focus,
        "memory": cache,
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
    stopping_max_iterations=STOPPING_MAX_ITERATIONS_GENETIC,
    batch_evaluator=None,
    n_cores=1,
    seed=None,
    discard_start_params=False,
    max_n_trials=1,
    population_size=None,
):
    """Minimize a scalar function using the artifical bee colony algorithm.

    The Artificial Bee Colony Algorithm was originally proposed by
    :cite:`Karaboga2007`. The implemented version of the algorithm is proposed
    in :cite:`Mernik2015`. The algorithm is only suited for bounded parameter
    spaces.

    - stopping.max_iterations (int): Number of generations to evolve.
    - batch_evaluator (str or Callable): Name of a pre-implemented batch
      evaluator (currently 'joblib' and 'pathos_mp') or Callable with the same
      interface as the estimagic batch_evaluators. See :ref:`batch_evaluators`.
    - n_cores (int): Number of cores to use.
    - seed (int): seed used by the internal random number generator.
    - discard_start_params (bool): If True, the start params are not guaranteed
      to be part of the initial population. This saves one criterion function
      evaluation that cannot be done in parallel with other evaluations. Default
      False.
    - max_n_trials (int): Maximum number of trials for abandoning a source.
      Default is 1.
    - population_size (int): Size of the population. If None, it's twice the
      number of parameters but at least 20.

    """
    population_size = _determine_population_size(
        population_size=population_size, x=x, lower_bound=20
    )
    algo_options = _create_algo_options(
        population_size=population_size,
        n_cores=n_cores,
        seed=seed,
        discard_start_params=discard_start_params,
        batch_evaluator=batch_evaluator,
        algo_specific_options={
            "limit": max_n_trials,
            "gen": int(stopping_max_iterations),
        },
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
    mutation_variant="rand/1/exp",
    convergence_criterion_tolerance=1e-6,
    convergence_relative_params_tolerance=CONVERGENCE_RELATIVE_PARAMS_TOLERANCE,
):
    """Minimize a scalar function using the differential evolution algorithm.

    Differential Evolution is a heuristic optimizer originally presented in
    :cite:`Storn1997`. The algorithm is only suited for bounded parameter
    spaces.

    - population_size (int): Size of the population. If None, it's twice the
      number of parameters but at least 10.
    - batch_evaluator (str or Callable): Name of a pre-implemented batch
      evaluator (currently 'joblib' and 'pathos_mp') or Callable with the same
      interface as the estimagic batch_evaluators. See :ref:`batch_evaluators`.
    - n_cores (int): Number of cores to use.
    - seed (int): seed used by the internal random number generator.
    - discard_start_params (bool): If True, the start params are not guaranteed
      to be part of the initial population. This saves one criterion function
      evaluation that cannot be done in parallel with other evaluations. Default
      False.
    - stopping.max_iterations (int): Number of generations to evolve.
    - weight_coefficient (float): Weight coefficient. It is denoted by $F$ in
      the main paper and must lie in [0, 2]. It controls the amplification of
      the differential variation $(x_{r_2, G} - x_{r_3, G})$.
    - crossover_probability (float): Crossover probability.
    - mutation_variant (str or int): code for the mutation variant to create a
      new candidate individual. The default is . The following are available:
        - "best/1/exp" (1, when specified as int)
        - "rand/1/exp" (2, when specified as int)
        - "rand-to-best/1/exp" (3, when specified as int)
        - "best/2/exp" (4, when specified as int)
        - "rand/2/exp" (5, when specified as int)
        - "best/1/bin" (6, when specified as int)
        - "rand/1/bin" (7, when specified as int)
        - "rand-to-best/1/bin" (8, when specified as int)
        - "best/2/bin" (9, when specified as int)
        - "rand/2/bin" (10, when specified as int)
    - convergence.criterion_tolerance: stopping criteria on the criterion
      tolerance. Default is 1e-6. It is not clear whether this is the absolute
      or relative criterion tolerance.
    - convergence.relative_params_tolerance: stopping criteria on the x
      tolerance. In pygmo the default is 1e-6 but we use our default value of
      1e-5.

    """
    population_size = _determine_population_size(
        population_size=population_size, x=x, lower_bound=10
    )

    # support both integer and string specification of the mutation variant
    mutation_variant_str_to_int = {
        "best/1/exp": 1,
        "rand/1/exp": 2,
        "rand-to-best/1/exp": 3,
        "best/2/exp": 4,
        "rand/2/exp": 5,
        "best/1/bin": 6,
        "rand/1/bin": 7,
        "rand-to-best/1/bin": 8,
        "best/2/bin": 9,
        "rand/2/bin": 10,
    }
    mutation_variant = _convert_str_to_int(
        str_to_int=mutation_variant_str_to_int, value=mutation_variant
    )

    algo_specific_options = {
        "gen": int(stopping_max_iterations),
        "F": weight_coefficient,
        "CR": crossover_probability,
        "variant": mutation_variant,
        "ftol": convergence_criterion_tolerance,
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
    stopping_max_iterations=10_000,  # Each generation will compute the objective once
):
    r"""Minimize a scalar function using the (N+1)-ES simple evolutionary algorithm.

    This algorithm represents the simplest evolutionary strategy, where a population of
    $\lambda$ individuals at each generation produces one offspring by mutating its best
    individual uniformly at random within the bounds. Should the offspring be better
    than the worst individual in the population it will substitute it.

    See :cite:`Oliveto2007`.

    The algorithm is only suited for bounded parameter spaces.

    - population_size (int): Size of the population. If None, it's twice the number of
      parameters but at least 10.
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
    population_size = _determine_population_size(
        population_size=population_size, x=x, lower_bound=10
    )

    algo_options = _create_algo_options(
        population_size=population_size,
        n_cores=n_cores,
        seed=seed,
        discard_start_params=discard_start_params,
        batch_evaluator=batch_evaluator,
        algo_specific_options={"gen": int(stopping_max_iterations)},
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


def pygmo_sga(
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
    crossover_probability=0.9,
    crossover_strategy="exponential",
    eta_c=None,
    mutation_probability=0.02,
    mutation_strategy="polynomial",
    mutation_polynomial_distribution_index=None,
    mutation_gaussian_width=None,
    selection_strategy="tournament",
    selection_truncated_n_best=None,
    selection_tournament_size=None,
):
    """Minimize a scalar function using a simple genetic algorithm.

    A detailed description of the algorithm can be found `in the pagmo2 documentation
    <https://esa.github.io/pagmo2/docs/cpp/algorithms/sga.html>`_.

    See also: :cite:`Oliveto2007`.

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
    - stopping.max_iterations (int): Number of generations to evolve.
    - crossover_probability (float): Crossover probability.
    - crossover_strategy (str): the crossover strategy. One of “exponential”,“binomial”,
      “single” or “sbx”. Default is "exponential".
    - eta_c (float): distribution index for “sbx” crossover. This is an inactive
      parameter if other types of crossovers are selected. Can be in [1, 100].
    - mutation_probability (float): Mutation probability.
    - mutation_strategy (str): Mutation strategy. Must be "gaussian", "polynomial" or
      "uniform". Default is "polynomial".
    - mutation_polynomial_distribution_index (float): Must be in [0, 1]. Default is 1.
    - mutation_gaussian_width (float): Must be in [0, 1]. Default is 1.
    - selection_strategy (str): Selection strategy. Must be "tournament" or "truncated".
    - selection_truncated_n_best (int): number of best individuals to use in the
      "truncated" selection mechanism.
    - selection_tournament_size (int): size of the tournament in the "tournament"
      selection mechanism. Default is 1.

    """
    _check_that_every_param_is_bounded(lower_bounds, upper_bounds)
    population_size = _determine_population_size(
        population_size=population_size, x=x, lower_bound=64
    )

    if eta_c is not None and crossover_strategy != "sbx":
        warnings.warn(
            f"You specified crossover strategy {crossover_strategy} and eta_c. "
            "However, eta_c is ignored because it is only used when the "
            "crossover_strategy is set to sbx."
        )
    eta_c = 1.0 if eta_c is None else eta_c

    if (
        mutation_polynomial_distribution_index is not None
    ) and mutation_strategy != "polynomial":
        warnings.warn(
            "You specified a mutation_polynomial_distribution_index but did not choose "
            "polynomial as your mutation_strategy. Thus, "
            "mutation_polynomial_distribution_index will be ignored."
        )
    if mutation_gaussian_width is not None and mutation_strategy != "gaussian":
        warnings.warn(
            "You specified a mutation_gaussian_width but did not choose gaussion as "
            "your mutation_strategy. Thus, mutation_gaussian_width will be ignored."
        )
    if selection_strategy != "truncated" and selection_truncated_n_best is not None:
        warnings.warn(
            "You specified selection_truncated_n_best but did not specify truncated as "
            "your selection strategy. Therefore, selection_truncated_n_best is ignored."
        )
    if selection_strategy != "tournament" and selection_tournament_size is not None:
        warnings.warn(
            "You specified selection_tournament_size but did not specify tournament as "
            "your selection strategy. Therefore, selection_tournament_size is ignored."
        )

    if mutation_strategy == "gaussian" and mutation_gaussian_width is not None:
        param_m = mutation_gaussian_width
    elif (
        mutation_strategy == "polynomial"
        and mutation_polynomial_distribution_index is not None
    ):
        param_m = mutation_polynomial_distribution_index
    else:
        param_m = 1.0

    if selection_strategy == "truncated" and selection_truncated_n_best is not None:
        param_s = selection_truncated_n_best
    elif selection_strategy == "tournament" and selection_tournament_size is not None:
        param_s = selection_tournament_size
    else:
        param_s = 2

    algo_specific_options = {
        "gen": int(stopping_max_iterations),
        "cr": crossover_probability,
        "eta_c": eta_c,
        "m": mutation_probability,
        "param_m": param_m,
        "crossover": crossover_strategy,
        "mutation": mutation_strategy,
        "selection": selection_strategy,
        "param_s": param_s,
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
        method="sga",
        algo_options=algo_options,
    )
    return res


def pygmo_sade(
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
    jde=True,
    stopping_max_iterations=STOPPING_MAX_ITERATIONS_GENETIC,
    mutation_variant="rand/1/exp",
    keep_adapted_params=False,
    ftol=1e-6,
    xtol=1e-6,
):
    """Minimize a scalar function using Self-adaptive Differential Evolution.

    The original Differential Evolution algorithm (pygmo_de) can be significantly
    improved introducing the idea of parameter self-adaptation.

    Many different proposals have been made to self-adapt both the crossover and the
    F parameters of the original differential evolution algorithm. pygmo's
    implementation supports two different mechanisms. The first one, proposed by
    :cite:`Brest2006`, does not make use of the differential evolution operators to
    produce new values for the weight coefficient $F$ and the crossover probability
    $CR$ and, strictly speaking, is thus not self-adaptation, rather parameter control.
    The resulting differential evolution variant is often referred to as jDE.
    The second variant is inspired by the ideas introduced by :cite:`Elsayed2011` and
    uses a variaton of the selected DE operator to produce new $CR$ anf $F$ parameters
    for each individual. This variant is referred to iDE.

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
    - jde (bool): Whether to use the jDE self-adaptation variant to control the $F$ and
      $CR$ parameter. If True jDE is used, else iDE.
    - stopping.max_iterations (int): Number of generations to evolve.
    - mutation_variant (int or str): code for the mutation variant to create a new
      candidate individual. The default is "rand/1/exp". The first ten are the
      classical mutation variants introduced in the orginal DE algorithm, the remaining
      ones are, instead, considered in the work by :cite:`Elsayed2011`.
      The following are available:

        - "best/1/exp" or 1
        - "rand/1/exp" or 2
        - "rand-to-best/1/exp" or 3
        - "best/2/exp" or 4
        - "rand/2/exp" or 5
        - "best/1/bin" or 6
        - "rand/1/bin" or 7
        - "rand-to-best/1/bin" or 8
        - "best/2/bin" or 9
        - "rand/2/bin" or 10
        - "rand/3/exp" or 11
        - "rand/3/bin" or 12
        - "best/3/exp" or 13
        - "best/3/bin" or 14
        - "rand-to-current/2/exp" or 15
        - "rand-to-current/2/bin" or 16
        - "rand-to-best-and-current/2/exp" or 17
        - "rand-to-best-and-current/2/bin" or 18

    - keep_adapted_params (bool):  when true the adapted parameters $CR$ anf $F$ are
      not reset between successive calls to the evolve method. Default is False.
    - ftol (float): stopping criteria on the x tolerance.
    - xtol (float): stopping criteria on the f tolerance.

    """
    _check_that_every_param_is_bounded(lower_bounds, upper_bounds)

    population_size = _determine_population_size(
        population_size=population_size, x=x, lower_bound=64
    )
    mutation_variant_str_to_int = {
        "best/1/exp": 1,
        "rand/1/exp": 2,
        "rand-to-best/1/exp": 3,
        "best/2/exp": 4,
        "rand/2/exp": 5,
        "best/1/bin": 6,
        "rand/1/bin": 7,
        "rand-to-best/1/bin": 8,
        "best/2/bin": 9,
        "rand/2/bin": 10,
        "rand/3/exp": 11,
        "rand/3/bin": 12,
        "best/3/exp": 13,
        "best/3/bin": 14,
        "rand-to-current/2/exp": 15,
        "rand-to-current/2/bin": 16,
        "rand-to-best-and-current/2/exp": 17,
        "rand-to-best-and-current/2/bin": 18,
    }
    mutation_variant = _convert_str_to_int(
        str_to_int=mutation_variant_str_to_int, value=mutation_variant
    )

    algo_specific_options = {
        "gen": int(stopping_max_iterations),
        "variant": mutation_variant,
        "variant_adptv": 1 if jde else 2,
        "ftol": ftol,
        "xtol": xtol,
        "memory": keep_adapted_params,
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
        method="sade",
        algo_options=algo_options,
    )
    return res


def pygmo_cmaes(
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
    backward_horizon=None,
    variance_loss_compensation=None,
    learning_rate_rank_one_update=None,
    learning_rate_rank_mu_update=None,
    initial_step_size=0.5,
    ftol=1e-6,
    xtol=1e-6,
    keep_adapted_params=False,
):
    r"""Minimize a scalar function using the Covariance Matrix Evolutionary Strategy.

    CMA-ES is one of the most successful algorithm, classified as an Evolutionary
    Strategy, for derivative-free global optimization. The version supported by
    estimagic is the version described in :cite:`Hansen2006`.

    In contrast to the pygmo version, estimagic always sets force_bounds to True. This
    avoids that ill defined parameter values are evaluated.

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

    - stopping.max_iterations (int): Number of generations to evolve.
    - backward_horizon (float): backward time horizon for the evolution path. It must
      lie betwen 0 and 1.
    - variance_loss_compensation (float): makes partly up for the small variance loss in
      case the indicator is zero. `cs` in the MATLAB Code of :cite:`Hansen2006`. It must
      lie between 0 and 1.
    - learning_rate_rank_one_update (float): learning rate for the rank-one update of
      the covariance matrix. `c1` in the pygmo and pagmo documentation. It must lie
      between 0 and 1.
    - learning_rate_rank_mu_update (float): learning rate for the rank-mu update of the
      covariance matrix. `cmu` in the pygmo and pagmo documentation. It must lie between
      0 and 1.
    - initial_step_size (float): initial step size, :math:`\sigma^0` in the original
      paper.
    - ftol (float): stopping criteria on the x tolerance.
    - xtol (float): stopping criteria on the f tolerance.
    - keep_adapted_params (bool):  when true the adapted parameters are not reset
      between successive calls to the evolve method. Default is False.

    """
    _check_that_every_param_is_bounded(lower_bounds, upper_bounds)

    population_size = _determine_population_size(
        population_size=population_size, x=x, lower_bound=64
    )

    algo_specific_options = {
        "gen": int(stopping_max_iterations),
        "cc": _replace_none(var=backward_horizon, none_value=-1.0),
        "cs": _replace_none(var=variance_loss_compensation, none_value=-1.0),
        "c1": _replace_none(var=learning_rate_rank_one_update, none_value=-1.0),
        "cmu": _replace_none(var=learning_rate_rank_mu_update, none_value=-1.0),
        "sigma0": initial_step_size,
        "ftol": ftol,
        "xtol": xtol,
        "memory": keep_adapted_params,
        "force_bounds": True,
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
        method="cmaes",
        algo_options=algo_options,
    )
    return res


def pygmo_simulated_annealing(
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
    start_temperature=10.0,
    end_temperature=0.01,
    n_temp_adjustments=10,
    n_range_adjustments=10,
    bin_size=10,
    start_range=1.0,
):
    """Minimize a function with the simulated annealing algorithm.

    This version of the simulated annealing algorithm is, essentially, an iterative
    random search procedure with adaptive moves along the coordinate directions. It
    permits uphill moves under the control of metropolis criterion, in the hope to avoid
    the first local minima encountered. This version is the one proposed in
    :cite:`Corana1987`.

    .. note: When selecting the starting and final temperature values it helps to think
        about the tempertaure as the deterioration in the objective function value that
        still has a 37% chance of being accepted.

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
    - start_temperature (float): starting temperature. Must be > 0.
    - end_temperature (float): final temperature. Our default (0.01) is lower than in
      pygmo and pagmo. The final temperature must be positive.
    - n_temp_adjustments (int): number of temperature adjustments in the annealing
      schedule.
    - n_range_adjustments (int): number of adjustments of the search range performed at
      a constant temperature.
    - bin_size (int): number of mutations that are used to compute the acceptance rate.
    - start_range (float): starting range for mutating the decision vector. It must lie
      between 0 and 1.

    """
    _check_that_every_param_is_bounded(lower_bounds, upper_bounds)

    population_size = _determine_population_size(
        population_size=population_size, x=x, lower_bound=64
    )

    algo_specific_options = {
        "Ts": start_temperature,
        "Tf": end_temperature,
        "n_T_adj": int(n_temp_adjustments),
        "n_range_adj": int(n_range_adjustments),
        "bin_size": bin_size,
        "start_range": start_range,
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
        method="simulated_annealing",
        algo_options=algo_options,
    )
    return res


def pygmo_pso(
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
    omega=0.7298,
    force_of_previous_best=2.05,
    force_of_best_in_neighborhood=2.05,
    max_velocity=0.5,
    algo_variant=5,
    neighbor_definition="lbest",
    neighbor_param=None,
    keep_velocities=False,
):
    r"""Minimize a scalar function using Particle Swarm Optimization.

    Particle swarm optimization (PSO) is a population based algorithm inspired by the
    foraging behaviour of swarms. In PSO each point has memory of the position where it
    achieved the best performance xli (local memory) and of the best decision vector
    :math:`x^g` in a certain neighbourhood, and uses this information to update its
    position.

    For a survey on particle swarm optimization algorithms, see :cite:`Poli2007`.

    Each particle determines its future position :math:`x_{i+1} = x_i + v_i` where

    .. math:: v_{i+1} = \omega (v_i + \eta_1 \cdot \mathbf{r}_1 \cdot (x_i - x^{l}_i) +
        \eta_2 \cdot \mathbf{r}_2 \cdot (x_i - x^g))

    - population_size (int): Size of the population. If None, it's twice the number of
      parameters but at least 10.
    - batch_evaluator (str or Callable): Name of a pre-implemented batch evaluator
      (currently 'joblib' and 'pathos_mp') or Callable with the same interface as the
      estimagic batch_evaluators. See :ref:`batch_evaluators`.
    - n_cores (int): Number of cores to use.
    - seed (int): seed used by the internal random number generator.
    - discard_start_params (bool): If True, the start params are not guaranteed to be
      part of the initial population. This saves one criterion function evaluation that
      cannot be done in parallel with other evaluations. Default False.
    - stopping.max_iterations (int): Number of generations to evolve.

    - omega (float): depending on the variant chosen, :math:`\omega` is the particles'
      inertia weight or the construction coefficient. It must lie between 0 and 1.
    - force_of_previous_best (float): :math:`\eta_1` in the equation above. It's the
      magnitude of the force, applied to the particle’s velocity, in the direction of
      its previous best position. It must lie between 0 and 4.
    - force_of_best_in_neighborhood (float): :math:`\eta_2` in the equation above. It's
      the magnitude of the force, applied to the particle’s velocity, in the direction
      of the best position in its neighborhood. It must lie between 0 and 4.
    - max_velocity (float): maximum allowed particle velocity as fraction of the box
      bounds. It must lie between 0 and 1.
    - algo_variant (int or str): algorithm variant to be used:
        - 1 or "canonical_inertia": Canonical (with inertia weight)
        - 2 or "social_and_cog_rand": Same social and cognitive rand.
        - 3 or "all_components_rand": Same rand. for all components
        - 4 or "one_rand": Only one rand.
        - 5 or "canonical_constriction": Canonical (with constriction fact.)
        - 6 or "fips": Fully Informed (FIPS)

    - neighbor_definition (int or str): swarm topology that defines each particle's
      neighbors that is to be used:
        - 1 or "gbest"
        - 2 or "lbest"
        - 3 or "Von Neumann"
        - 4 or "Adaptive random"

    - neighbor_param (int): the neighbourhood parameter. If the lbest topology is
      selected (neighbor_definition=2), it represents each particle's indegree (also
      outdegree) in the swarm topology. Particles have neighbours up to a radius of k =
      neighbor_param / 2 in the ring. If the Randomly-varying neighbourhood topology is
      selected (neighbor_definition=4), it represents each particle’s maximum outdegree
      in the swarm topology. The minimum outdegree is 1 (the particle always connects
      back to itself). If neighbor_definition is 1 or 3 this parameter is ignored.
    - keep_velocities (bool): when true the particle velocities are not reset between
      successive calls to `evolve`.

    """
    _check_that_every_param_is_bounded(lower_bounds, upper_bounds)

    if neighbor_definition in [1, 3] and neighbor_param is not None:
        warnings.warn(
            "You gave a neighbor parameter but selected a neighbor_definition "
            "that ignores this parameter."
        )

    neighbor_param = _replace_none(neighbor_param, 4)

    population_size = _determine_population_size(
        population_size=population_size, x=x, lower_bound=10
    )

    algo_specific_options = {
        "gen": int(stopping_max_iterations),
        "omega": omega,
        "eta1": force_of_previous_best,
        "eta2": force_of_best_in_neighborhood,
        "max_vel": max_velocity,
        "variant": algo_variant,
        "neighb_type": neighbor_definition,
        "neighb_param": neighbor_param,
        "memory": keep_velocities,
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
        method="pso",
        algo_options=algo_options,
    )
    return res


def pygmo_pso_gen(
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
    omega=0.7298,
    force_of_previous_best=2.05,
    force_of_best_in_neighborhood=2.05,
    max_velocity=0.5,
    algo_variant="canonical_constriction",
    neighbor_definition=2,
    neighbor_param=None,
    keep_velocities=False,
):
    r"""Minimize a scalar function with generational Particle Swarm Optimization.

    Particle Swarm Optimization (generational) is identical to pso, but does update the
    velocities of each particle before new particle positions are computed (taking into
    consideration all updated particle velocities). Each particle is thus evaluated on
    the same seed within a generation as opposed to the standard PSO which evaluates
    single particle at a time. Consequently, the generational PSO algorithm is suited
    for stochastic optimization problems.

    For a survey on particle swarm optimization algorithms, see :cite:`Poli2007`.

    Each particle determines its future position :math:`x_{i+1} = x_i + v_i` where

    .. math:: v_{i+1} = \omega (v_i + \eta_1 \cdot \mathbf{r}_1 \cdot (x_i - x^{l}_i) +
        \eta_2 \cdot \mathbf{r}_2 \cdot (x_i - x^g))

    - population_size (int): Size of the population. If None, it's twice the number of
      parameters but at least 10.
    - batch_evaluator (str or Callable): Name of a pre-implemented batch evaluator
      (currently 'joblib' and 'pathos_mp') or Callable with the same interface as the
      estimagic batch_evaluators. See :ref:`batch_evaluators`.
    - n_cores (int): Number of cores to use.
    - seed (int): seed used by the internal random number generator.
    - discard_start_params (bool): If True, the start params are not guaranteed to be
      part of the initial population. This saves one criterion function evaluation that
      cannot be done in parallel with other evaluations. Default False.
    - stopping.max_iterations (int): Number of generations to evolve.

    - omega (float): depending on the variant chosen, :math:`\omega` is the particles'
      inertia weight or the constructuion coefficient. It must lie between 0 and 1.
    - force_of_previous_best (float): :math:`\eta_1` in the equation above. It's the
      magnitude of the force, applied to the particle’s velocity, in the direction of
      its previous best position. It must lie between 0 and 4.
    - force_of_best_in_neighborhood (float): :math:`\eta_2` in the equation above. It's
      the magnitude of the force, applied to the particle’s velocity, in the direction
      of the best position in its neighborhood. It must lie between 0 and 4.
    - max_velocity (float): maximum allowed particle velocity as fraction of the box
      bounds. It must lie between 0 and 1.
    - algo_variant (int): code of the algorithm's variant to be used:

        - 1: Canonical (with inertia weight)
        - 2: Same social and cognitive rand.
        - 3: Same rand. for all components
        - 4: Only one rand.
        - 5: Canonical (with constriction fact.)
        - 6: Fully Informed (FIPS)

    - neighbor_definition (int): code for the swarm topology that defines each
      particle's neighbors that is to be used:

        - 1: gbest
        - 2: lbest
        - 3: Von Neumann
        - 4: Adaptive random

    - neighbor_param (int): the neighbourhood parameter. If the lbest topology is
      selected (neighbor_definition=2), it represents each particle's indegree (also
      outdegree) in the swarm topology. Particles have neighbours up to a radius of k =
      neighbor_param / 2 in the ring. If the Randomly-varying neighbourhood topology is
      selected (neighbor_definition=4), it represents each particle’s maximum outdegree
      in the swarm topology. The minimum outdegree is 1 (the particle always connects
      back to itself). If neighbor_definition is 1 or 3 this parameter is ignored.
    - keep_velocities (bool): when true the particle velocities are not reset between
      successive calls to `evolve`.

    """
    _check_that_every_param_is_bounded(lower_bounds, upper_bounds)

    if neighbor_definition in [1, 3] and neighbor_param is not None:
        warnings.warn(
            "You gave a neighbor parameter but selected a neighbor_definition "
            "that ignores this parameter."
        )
    neighbor_param = _replace_none(neighbor_param, 4)
    neighbor_str_to_int = {
        "gbest": 1,
        "lbest": 2,
        "Von Neumann": 3,
        "Adaptive random": 4,
    }
    neighbor_param = _convert_str_to_int(neighbor_str_to_int, neighbor_param)

    algo_variant_str_to_int = {
        "canonical_inertia": 1,
        "social_and_cog_rand": 2,
        "all_components_rand": 3,
        "one_rand": 4,
        "canonical_constriction": 5,
        "fips": 6,
    }
    algo_variant = _convert_str_to_int(algo_variant_str_to_int, algo_variant)

    population_size = _determine_population_size(
        population_size=population_size, x=x, lower_bound=10
    )

    algo_specific_options = {
        "gen": int(stopping_max_iterations),
        "omega": omega,
        "eta1": force_of_previous_best,
        "eta2": force_of_best_in_neighborhood,
        "max_vel": max_velocity,
        "variant": algo_variant,
        "neighb_type": neighbor_definition,
        "neighb_param": neighbor_param,
        "memory": keep_velocities,
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
        method="pso_gen",
        algo_options=algo_options,
    )
    return res


def pygmo_mbh(
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
    inner_algorithm=None,
    # this is 30 instead of 5 in pygmo for our sum of squares test to pass
    stopping_max_inner_runs_without_improvement=30,
    perturbation=0.01,
):
    """Minimize a scalar function using generalized Monotonic Basin Hopping.

    Monotonic basin hopping, or simply, basin hopping, is an algorithm rooted in the
    idea of mapping the objective function $f(x_0)$ into the local minima found starting
    from $x_0$. This simple idea allows a substantial increase of efficiency in solving
    problems, such as the Lennard-Jones cluster or the MGA-1DSM interplanetary
    trajectory problem that are conjectured to have a so-called funnel structure.

    See :cite:`Wales1997` for the paper introducing the basin hopping idea for a
    Lennard-Jones cluster optimization.

    pygmo provides an original generalization of this concept resulting in a
    meta-algorithm that operates on a population. When a population containing a single
    individual is used the original method is recovered.

    - population_size (int): Size of the population. If None, it's twice the number of
      parameters but at least 250.
    - batch_evaluator (str or Callable): Name of a pre-implemented batch evaluator
      (currently 'joblib' and 'pathos_mp') or Callable with the same interface as the
      estimagic batch_evaluators. See :ref:`batch_evaluators`.
    - n_cores (int): Number of cores to use.
    - seed (int): seed used by the internal random number generator.
    - discard_start_params (bool): If True, the start params are not guaranteed to be
      part of the initial population. This saves one criterion function evaluation that
      cannot be done in parallel with other evaluations. Default False.
    - inner_algorithm (pygmo.algorithm): an pygmo algorithm or a user-defined algorithm,
      either C++ or Python. If None the `pygmo.compass_search` algorithm will be used.
    - stopping.max_inner_runs_without_improvement (int): consecutive runs of the inner
      algorithm that need to result in no improvement for mbh to stop.
    - perturbation (float): the perturbation to be applied to each component.

    """
    _check_that_every_param_is_bounded(lower_bounds, upper_bounds)

    # the min default population size is this large to pass our sum of squares tests.
    population_size = _determine_population_size(
        population_size=population_size, x=x, lower_bound=250
    )

    algo_specific_options = {
        "algo": inner_algorithm,
        "stop": stopping_max_inner_runs_without_improvement,
        "perturb": perturbation,
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
        method="mbh",
        algo_options=algo_options,
    )
    return res


def pygmo_xnes(
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
    learning_rate_mean_update=1.0,
    learning_rate_step_size_update=None,
    learning_rate_cov_matrix_update=None,
    initial_search_share=1.0,
    ftol=1e-6,
    xtol=1e-6,
    keep_adapted_params=False,
):
    r"""Minimize a scalar function using Exponential Evolution Strategies.

    Exponential Natural Evolution Strategies is an algorithm closely related to CMAES
    and based on the adaptation of a gaussian sampling distribution via the so-called
    natural gradient. Like CMAES it is based on the idea of sampling new trial vectors
    from a multivariate distribution and using the new sampled points to update the
    distribution parameters. Naively this could be done following the gradient of the
    expected fitness as approximated by a finite number of sampled points. While this
    idea offers a powerful lead on algorithmic construction it has some major drawbacks
    that are solved in the so-called Natural Evolution Strategies class of algorithms by
    adopting, instead, the natural gradient. xNES is one of the most performing variants
    in this class.

    See :cite:`Glasmachers2010` and the `pagmo documentation on xNES
    <https://esa.github.io/pagmo2/docs/cpp/algorithms/xnes.html#_CPPv4N5pagmo4xnesE>`_
    for details.

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
    - stopping.max_iterations (int): Number of generations to evolve.

    - learning_rate_mean_update (float): learning rate for the mean update
      (:math:`\eta_\mu`). It must be between 0 and 1 or None.
    - learning_rate_step_size_update (float): learning rate for the step-size update. It
      must be between 0 and 1 or None.
    - learning_rate_cov_matrix_update (float): learning rate for the covariance matrix
      update. It must be between 0 and 1 or None.
    - initial_search_share (float): share of the given search space that will be
      initally searched. It must be between 0 and 1. Default is 1.
    - ftol (float): stopping criteria on the x tolerance.
    - xtol (float): stopping criteria on the f tolerance.
    - keep_adapted_params (bool): when true the adapted parameters are not reset between
      successive calls to the evolve method. Default is False.

    """
    _check_that_every_param_is_bounded(lower_bounds, upper_bounds)

    population_size = _determine_population_size(
        population_size=population_size, x=x, lower_bound=64
    )

    algo_specific_options = {
        "gen": int(stopping_max_iterations),
        "eta_mu": _replace_none(learning_rate_mean_update, -1),
        "eta_sigma": _replace_none(learning_rate_step_size_update, -1),
        "eta_b": _replace_none(learning_rate_cov_matrix_update, -1),
        "sigma0": initial_search_share,
        "ftol": ftol,
        "xtol": xtol,
        "memory": keep_adapted_params,
        "force_bounds": True,
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
        method="xnes",
        algo_options=algo_options,
    )
    return res


def pygmo_gwo(
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
    """Minimize a scalar function usinng the Grey Wolf Optimizer.

    The grey wolf optimizer was proposed by :cite:`Mirjalili2014`. The pygmo
    implementation that is wrapped by estimagic is pased on the pseudo code provided in
    that paper.

    This algorithm is a classic example of a highly criticizable line of search that led
    in the first decades of our millenia to the development of an entire zoo of
    metaphors inspiring optimzation heuristics. In our opinion they, as is the case for
    the grey wolf optimizer, are often but small variations of already existing
    heuristics rebranded with unnecessray and convoluted biological metaphors. In the
    case of GWO this is particularly evident as the position update rule is shokingly
    trivial and can also be easily seen as a product of an evolutionary metaphor or a
    particle swarm one. Such an update rule is also not particulary effective and
    results in a rather poor performance most of times.

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
    - stopping.max_iterations (int): Number of generations to evolve.

    """
    _check_that_every_param_is_bounded(lower_bounds, upper_bounds)

    population_size = _determine_population_size(
        population_size=population_size, x=x, lower_bound=64
    )
    algo_options = _create_algo_options(
        population_size=population_size,
        n_cores=n_cores,
        seed=seed,
        discard_start_params=discard_start_params,
        batch_evaluator=batch_evaluator,
        algo_specific_options={"gen": int(stopping_max_iterations)},
    )

    res = _minimize_pygmo(
        criterion_and_derivative=criterion_and_derivative,
        x=x,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        method="gwo",
        algo_options=algo_options,
    )
    return res


def pygmo_compass_search(
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
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS,
    start_range=0.1,
    stop_range=0.01,
    reduction_coeff=0.5,
):
    """Minimize a scalar function using compass search.

    The algorithm is described in :cite:`Kolda2003`.

    It is considered slow but reliable. It should not be used for stochastic problems.

    - population_size (int): Size of the population. Even though the algorithm is not
      population based the population size does affect the results of the algorithm.
    - batch_evaluator (str or Callable): Name of a pre-implemented batch evaluator
      (currently 'joblib' and 'pathos_mp') or Callable with the same interface as the
      estimagic batch_evaluators. See :ref:`batch_evaluators`.
    - n_cores (int): Number of cores to use.
    - seed (int): seed used by the internal random number generator.
    - discard_start_params (bool): If True, the start params are not guaranteed to be
      part of the initial population. This saves one criterion function evaluation that
      cannot be done in parallel with other evaluations. Default False.
    - stopping.max_criterion_evaluations (int): maximum number of function evaluations.
    - start_range (float): the start range. Must be in (0, 1].
    - stop_range (float): the stop range. Must be in (0, start_range].
    - reduction_coeff (float): the range reduction coefficient. Must be in (0, 1).

    """
    _check_that_every_param_is_bounded(lower_bounds, upper_bounds)

    algo_specific_options = {
        "max_fevals": stopping_max_criterion_evaluations,
        "start_range": start_range,
        "stop_range": stop_range,
        "reduction_coeff": reduction_coeff,
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
        method="compass_search",
        algo_options=algo_options,
    )
    return res


def pygmo_ihs(
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
    choose_from_memory_probability=0.85,
    min_pitch_adjustment_rate=0.35,
    max_pitch_adjustment_rate=0.99,
    min_distance_bandwidth=1e-5,
    max_distance_bandwidth=1.0,
):
    """Minimize a scalar function using the improved harmony search algorithm.

    Improved harmony search (IHS) was introduced by :cite:`Mahdavi2007`.
    IHS supports stochastic problems.

    - population_size (int): Size of the population. If None, it's twice the number of
      parameters.
    - batch_evaluator (str or Callable): Name of a pre-implemented batch evaluator
      (currently 'joblib' and 'pathos_mp') or Callable with the same interface as the
      estimagic batch_evaluators. See :ref:`batch_evaluators`.
    - n_cores (int): Number of cores to use.
    - seed (int): seed used by the internal random number generator.
    - discard_start_params (bool): If True, the start params are not guaranteed to be
      part of the initial population. This saves one criterion function evaluation that
      cannot be done in parallel with other evaluations. Default False.
    - stopping.max_iterations (int): Number of generations to evolve.
    - choose_from_memory_probability (float): probability of choosing from memory
      (similar to a crossover probability).
    - min_pitch_adjustment_rate (float): minimum pitch adjustment rate. (similar to a
      mutation rate). It must be between 0 and 1.
    - max_pitch_adjustment_rate (float): maximum pitch adjustment rate. (similar to a
      mutation rate). It must be between 0 and 1.
    - min_distance_bandwidth (float): minimum distance bandwidth. (similar to a mutation
      width). It must be positive.
    - max_distance_bandwidth (float): maximum distance bandwidth. (similar to a mutation
      width).

    """
    _check_that_every_param_is_bounded(lower_bounds, upper_bounds)

    if population_size is not None:
        warnings.warn("The population size has no effect on IHS' performance.")

    population_size = _determine_population_size(
        population_size=population_size, x=x, lower_bound=1
    )

    algo_specific_options = {
        "gen": int(stopping_max_iterations),
        "phmcr": choose_from_memory_probability,
        "ppar_min": min_pitch_adjustment_rate,
        "ppar_max": max_pitch_adjustment_rate,
        "bw_min": min_distance_bandwidth,
        "bw_max": max_distance_bandwidth,
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
        method="ihs",
        algo_options=algo_options,
    )
    return res


def pygmo_de1220(
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
    jde=True,
    stopping_max_iterations=STOPPING_MAX_ITERATIONS_GENETIC,
    allowed_variants=None,
    keep_adapted_params=False,
    ftol=1e-6,
    xtol=1e-6,
):
    """Minimize a scalar function using Self-adaptive Differential Evolution, pygmo flavor.

    See `the PAGMO documentation for details
    <https://esa.github.io/pagmo2/docs/cpp/algorithms/de1220.html>`_.

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
    - jde (bool): Whether to use the jDE self-adaptation variant to control the $F$ and
      $CR$ parameter. If True jDE is used, else iDE.
    - stopping.max_iterations (int): Number of generations to evolve.
    - allowed_variants (array-like object): allowed mutation variants (can be codes
      or strings). Each code refers to one mutation variant to create a new candidate
      individual. The first ten refer to the classical mutation variants introduced in
      the original DE algorithm, the remaining ones are, instead, considered in the work
      by :cite:`Elsayed2011`. The default is ["rand/1/exp", "rand-to-best/1/exp",
      "rand/1/bin", "rand/2/bin", "best/3/exp", "best/3/bin", "rand-to-current/2/exp",
      "rand-to-current/2/bin"]. The following are available:
        - 1 or "best/1/exp"
        - 2 or "rand/1/exp"
        - 3 or "rand-to-best/1/exp"
        - 4 or "best/2/exp"
        - 5 or "rand/2/exp"
        - 6 or "best/1/bin"
        - 7 or "rand/1/bin"
        - 8 or "rand-to-best/1/bin"
        - 9 or "best/2/bin"
        - 10 or "rand/2/bin"
        - 11 or "rand/3/exp"
        - 12 or "rand/3/bin"
        - 13 or "best/3/exp"
        - 14 or "best/3/bin"
        - 15 or "rand-to-current/2/exp"
        - 16 or "rand-to-current/2/bin"
        - 17 or "rand-to-best-and-current/2/exp"
        - 18 or "rand-to-best-and-current/2/bin"

    - keep_adapted_params (bool):  when true the adapted parameters $CR$ anf $F$ are not
      reset between successive calls to the evolve method. Default is False.
    - ftol (float): stopping criteria on the x tolerance.
    - xtol (float): stopping criteria on the f tolerance.

    """
    _check_that_every_param_is_bounded(lower_bounds, upper_bounds)
    variant_str_to_int = {
        "best/1/exp": 1,
        "rand/1/exp": 2,
        "rand-to-best/1/exp": 3,
        "best/2/exp": 4,
        "rand/2/exp": 5,
        "best/1/bin": 6,
        "rand/1/bin": 7,
        "rand-to-best/1/bin": 8,
        "best/2/bin": 9,
        "rand/2/bin": 10,
        "rand/3/exp": 11,
        "rand/3/bin": 12,
        "best/3/exp": 13,
        "best/3/bin": 14,
        "rand-to-current/2/exp": 15,
        "rand-to-current/2/bin": 16,
        "rand-to-best-and-current/2/exp": 17,
        "rand-to-best-and-current/2/bin": 18,
    }
    if allowed_variants is None:
        allowed_variant_codes = [2, 3, 7, 10, 13, 14, 15, 16]
    else:
        allowed_variant_codes = []
        for variant in allowed_variants:
            allowed_variant_codes.append(
                _convert_str_to_int(variant_str_to_int, variant)
            )

    population_size = _determine_population_size(
        population_size=population_size, x=x, lower_bound=64
    )

    algo_specific_options = {
        "gen": int(stopping_max_iterations),
        "variant_adptv": 1 if jde else 2,
        "ftol": ftol,
        "xtol": xtol,
        "memory": keep_adapted_params,
        "allowed_variants": allowed_variant_codes,
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
        method="de1220",
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
    if not IS_PYGMO_INSTALLED:
        raise NotImplementedError(
            f"The pygmo package is not installed and required for '{method}'."
        )

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
        "population_size": _replace_none(population_size, -1),
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


def _determine_population_size(population_size, x, lower_bound):
    if population_size is None:
        population_size = int(np.clip(2 * len(x), lower_bound, np.inf))
    else:
        population_size = int(population_size)
    return population_size


def _convert_str_to_int(str_to_int, value):
    if value in str_to_int.keys():
        out = str_to_int[value]
    elif value not in str_to_int.values():
        raise ValueError(
            f"You specified {value} as value. "
            f"It must be one of {', '.join(str_to_int.keys())}"
        )
    else:
        out = value
    return out


def _replace_none(var, none_value):
    out = var if var is not None else none_value
    return out
