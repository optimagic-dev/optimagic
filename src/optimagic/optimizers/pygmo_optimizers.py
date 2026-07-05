"""Implement pygmo optimizers.

Notes for converting to the new algorithm interface:

- `create_algo_options` is not needed anymore because the only thing it did was mixing
  options that are supported by all optimizers (e.g. population_size, seed) with
  specific options. Then later they had to be taken appart again. Instead you need
  to pass  a few more arguments directly to `_minimize_pygmo`.
- Calling `_check_that_every_param_is_bounded` is not needed anymore. I do that check
  once in `_minimize_pygmo`.
- The documentation often just says float where I suspect PositiveFloats; Leave it at
  float for now and add todos where needed. Don't spend a lot of time on it.
- There are some specific type checks and type conversions that should not be needed
  anymore after switching to the new interface.
- Whenever we had batch_evaluator as `algo_option` we don't need it anymore but we
  should have `n_cores` in those algorithms.

"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Literal

import numpy as np
from numpy.typing import NDArray

from optimagic import mark
from optimagic.config import IS_PYGMO_INSTALLED
from optimagic.exceptions import NotInstalledError
from optimagic.optimization.algo_options import (
    CONVERGENCE_XTOL_REL,
    STOPPING_MAXFUN_GLOBAL,
    get_population_size,
)
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.internal_optimization_problem import (
    InternalOptimizationProblem,
)
from optimagic.typing import (
    AggregationLevel,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
)

STOPPING_MAX_ITERATIONS_GENETIC = 250

if TYPE_CHECKING:
    import pygmo as pg


@mark.minimizer(
    name="pygmo_gaco",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_PYGMO_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=True,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class PygmoGaco(Algorithm):
    """Minimize a scalar function using the extended ant colony algorithm (GACO).

    The version available through pygmo is a generalized version of the original
    ant colony algorithm proposed by :cite:`Schlueter2009`.

    Ant colony optimization is a class of optimization algorithms modeled on the
    actions of an ant colony. Artificial "ants" (e.g. simulation agents) locate
    optimal solutions by moving through a parameter space representing all
    possible solutions. Real ants lay down pheromones directing each other to
    resources while exploring their environment. The simulated "ants" similarly
    record their positions and the quality of their solutions, so that in later
    simulation iterations more ants locate better solutions.

    The extended ant colony algorithm generates future generations of ants by
    using a multi-kernel gaussian distribution based on three parameters (i.e.,
    pheromone values) which are computed depending on the quality of each
    previous solution. The solutions are ranked through an oracle penalty
    method.

    Like all pygmo optimizers, this is a global, derivative-free, population
    based algorithm. It is a good choice for multimodal problems where no
    derivatives are available and a large budget of function evaluations is
    affordable. In contrast to most other pygmo algorithms, it supports parallel
    function evaluation via ``n_cores``. For smooth problems with only one
    optimum, local optimizers are much more efficient.

    The algorithm requires finite bounds on all parameters.

    """

    population_size: int | None = None
    """Size of the population. If None, it is set to ``10 * (n_params + 1)``, but
    at least 64."""

    n_cores: int = 1
    """Number of cores used for parallel function evaluation."""

    seed: int | None = None
    """Seed used by the internal random number generator."""

    discard_start_params: bool = False
    """If True, the start parameters are not guaranteed to be part of the initial
    population.

    This saves one criterion function evaluation that cannot be done in parallel
    with other evaluations. Default False.

    """

    stopping_maxiter: PositiveInt = STOPPING_MAX_ITERATIONS_GENETIC
    """Number of generations to evolve (the ``gen`` parameter in pygmo)."""

    kernel_size: PositiveInt = 63
    """Number of solutions stored in the solution archive (the ``ker`` parameter
    in pygmo).

    The default value is taken from pygmo.

    """

    speed_parameter_q: PositiveFloat = 1.0
    """This parameter manages the convergence speed towards the found minima (the
    smaller the faster).

    In the pygmo documentation it is referred to as :math:`q`. It must be
    positive and can be larger than 1. The default is 1.0 until ``threshold`` is
    reached. Then it is set to 0.01 automatically.

    """

    oracle: float = 0.0
    """Oracle parameter used in the penalty method."""

    accuracy: PositiveFloat = 0.01
    """Accuracy parameter for maintaining a minimum penalty function's values
    distances."""

    threshold: PositiveInt = 1
    """When the generation counter reaches the threshold, the convergence speed
    ``speed_parameter_q`` is set to 0.01 automatically.

    It must lie between 1 and ``stopping_maxiter``. To deactivate this effect,
    set the threshold to ``stopping_maxiter``, which is the largest allowed
    value.

    """

    speed_of_std_values_convergence: int = 7
    """Parameter that determines the convergence speed of the standard
    deviations.

    This must be an integer (the ``n_gen_mark`` parameter in pygmo and pagmo).

    """

    stopping_max_n_without_improvements: PositiveInt = 100000
    """If a positive integer is assigned here, the algorithm counts the runs
    without improvements; if this number exceeds the given value, the algorithm
    is stopped (the ``impstop`` parameter in pygmo)."""

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL
    """Maximum number of function evaluations (the ``evalstop`` parameter in
    pygmo)."""

    focus: NonNegativeFloat = 0.0
    """This parameter makes the search for the optimum greedier and more focused
    on local improvements (the higher the greedier).

    If the value is very high, the search is more focused around the current
    best solutions. Values larger than 1 are allowed.

    """

    cache: bool = False
    """If True, memory is activated in the algorithm for multiple calls (the
    ``memory`` parameter in pygmo)."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=64
        )

        algo_specific_options = {
            "gen": self.stopping_maxiter,
            "ker": self.kernel_size,
            "q": self.speed_parameter_q,
            "oracle": self.oracle,
            "acc": self.accuracy,
            "threshold": self.threshold,
            "n_gen_mark": self.speed_of_std_values_convergence,
            "impstop": self.stopping_max_n_without_improvements,
            "evalstop": self.stopping_maxfun,
            "focus": self.focus,
            "memory": self.cache,
        }

        res = _minimize_pygmo(
            problem=problem,
            x0=x0,
            method="gaco",
            specific_options=algo_specific_options,
            population_size=population_size,
            n_cores=self.n_cores,
            seed=self.seed,
            discard_start_params=self.discard_start_params,
        )

        return res


@mark.minimizer(
    name="pygmo_bee_colony",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_PYGMO_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class PygmoBeeColony(Algorithm):
    """Minimize a scalar function using the artificial bee colony algorithm.

    The artificial bee colony algorithm (ABC) was originally proposed by
    :cite:`Karaboga2007`. The implemented version of the algorithm is the one
    proposed in :cite:`Mernik2015`.

    The algorithm mimics the foraging behavior of a honey bee swarm: employed
    bees exploit known food sources (candidate solutions), onlooker bees pick
    promising sources based on their quality, and food sources that could not be
    improved for ``max_n_trials`` trials are abandoned and replaced by random new
    ones.

    It is a global, derivative-free, population based algorithm that is suited
    for multimodal problems where no derivatives are available and many function
    evaluations are affordable. For smooth problems with only one optimum, local
    optimizers are much more efficient.

    The algorithm requires finite bounds on all parameters.

    """

    stopping_maxiter: PositiveInt = STOPPING_MAX_ITERATIONS_GENETIC
    """Number of generations to evolve (the ``gen`` parameter in pygmo)."""

    seed: int | None = None
    """Seed used by the internal random number generator."""

    discard_start_params: bool = False
    """If True, the start parameters are not guaranteed to be part of the initial
    population.

    This saves one criterion function evaluation that cannot be done in parallel
    with other evaluations. Default False.

    """

    max_n_trials: PositiveInt = 1
    """Maximum number of trials for abandoning a food source (the ``limit``
    parameter in pygmo).

    optimagic uses a default of 1, whereas pygmo's default is 20.

    """

    population_size: int | None = None
    """Size of the population. If None, it is set to ``10 * (n_params + 1)``, but
    at least 20."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=20
        )

        algo_specific_options = {
            "limit": self.max_n_trials,
            "gen": self.stopping_maxiter,
        }

        res = _minimize_pygmo(
            problem=problem,
            x0=x0,
            method="bee_colony",
            specific_options=algo_specific_options,
            population_size=population_size,
            n_cores=1,
            seed=self.seed,
            discard_start_params=self.discard_start_params,
        )
        return res


@mark.minimizer(
    name="pygmo_de",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_PYGMO_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class PygmoDe(Algorithm):
    """Minimize a scalar function using the differential evolution algorithm.

    Differential evolution (DE) is a heuristic global optimizer originally
    presented in :cite:`Storn1997`. It maintains a population of candidate
    solutions and creates new candidates by combining existing ones according to
    a simple mutation and crossover recipe, keeping a new candidate whenever it
    improves upon the population member it replaces.

    DE is derivative-free and copes well with multimodal and non-smooth scalar
    problems. Like all population based algorithms, it needs many function
    evaluations, so local optimizers should be preferred for smooth problems
    with only one optimum.

    The algorithm requires finite bounds on all parameters.

    """

    population_size: int | None = None
    """Size of the population. If None, it is set to ``10 * (n_params + 1)``, but
    at least 10."""

    seed: int | None = None
    """Seed used by the internal random number generator."""

    discard_start_params: bool = False
    """If True, the start parameters are not guaranteed to be part of the initial
    population.

    This saves one criterion function evaluation that cannot be done in parallel
    with other evaluations. Default False.

    """

    stopping_maxiter: PositiveInt = STOPPING_MAX_ITERATIONS_GENETIC
    """Number of generations to evolve (the ``gen`` parameter in pygmo)."""

    # TODO: Refine type to fix range [0,2]
    weight_coefficient: NonNegativeFloat = 0.8
    r"""Weight coefficient, denoted by :math:`F` in the differential evolution
    literature.

    It controls the amplification of the differential variation
    :math:`(x_{r_2, G} - x_{r_3, G})`. The original paper allows values in
    [0, 2], but the pygmo implementation requires the value to lie in [0, 1].
    The default value is taken from pygmo.

    """

    # TODO: Probably refine type to fix range [0,1]
    crossover_probability: NonNegativeFloat = 0.9
    """Crossover probability. It must lie in [0, 1]. The default value is taken
    from pygmo."""

    mutation_variant: Literal[
        "best/1/exp",
        "rand/1/exp",
        "rand-to-best/1/exp",
        "best/2/exp",
        "rand/2/exp",
        "best/1/bin",
        "rand/1/bin",
        "rand-to-best/1/bin",
        "best/2/bin",
        "rand/2/bin",
    ] = "rand/1/exp"
    """Mutation variant used to create a new candidate individual.

    The default is "rand/1/exp". The following variants are available (the pygmo
    integer code given in parentheses is accepted as well):

    - "best/1/exp" (1)
    - "rand/1/exp" (2)
    - "rand-to-best/1/exp" (3)
    - "best/2/exp" (4)
    - "rand/2/exp" (5)
    - "best/1/bin" (6)
    - "rand/1/bin" (7)
    - "rand-to-best/1/bin" (8)
    - "best/2/bin" (9)
    - "rand/2/bin" (10)

    """

    convergence_criterion_tolerance: NonNegativeFloat = 1e-6
    """Stop when the absolute difference between the best and the worst objective
    value in the population is smaller than this value (the ``ftol`` parameter in
    pygmo).

    The default value is taken from pygmo.

    """

    convergence_relative_params_tolerance: NonNegativeFloat = CONVERGENCE_XTOL_REL
    """Stop when the sum of absolute differences between the best and the worst
    parameter vector in the population is smaller than this value (the ``xtol``
    parameter in pygmo).

    In pygmo the default is 1e-6, but optimagic uses its default value of 1e-5.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=10
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
            str_to_int=mutation_variant_str_to_int, value=self.mutation_variant
        )

        algo_specific_options = {
            "gen": self.stopping_maxiter,
            "F": self.weight_coefficient,
            "CR": self.crossover_probability,
            "variant": mutation_variant,
            "ftol": self.convergence_criterion_tolerance,
            "xtol": self.convergence_relative_params_tolerance,
        }

        res = _minimize_pygmo(
            problem=problem,
            x0=x0,
            method="de",
            specific_options=algo_specific_options,
            population_size=population_size,
            n_cores=1,
            seed=self.seed,
            discard_start_params=self.discard_start_params,
        )
        return res


@mark.minimizer(
    name="pygmo_sea",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_PYGMO_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class PygmoSea(Algorithm):
    r"""Minimize a scalar function using the (N+1)-ES simple evolutionary
    algorithm.

    This algorithm represents the simplest evolutionary strategy: a population
    of :math:`\lambda` individuals produces one offspring per generation by
    mutating its best individual uniformly at random within the bounds. Should
    the offspring be better than the worst individual in the population, it will
    substitute it. See :cite:`Oliveto2007` for a time-complexity analysis of
    this kind of evolutionary algorithm.

    Each generation computes the objective function only once, so progress is
    slow but the algorithm is extremely simple and robust. It is mainly useful
    as a baseline for other global, derivative-free optimizers on multimodal
    problems.

    The algorithm requires finite bounds on all parameters.

    """

    population_size: int | None = None
    """Size of the population. If None, it is set to ``10 * (n_params + 1)``, but
    at least 10."""

    seed: int | None = None
    """Seed used by the internal random number generator."""

    discard_start_params: bool = False
    """If True, the start parameters are not guaranteed to be part of the initial
    population.

    This saves one criterion function evaluation that cannot be done in parallel
    with other evaluations. Default False.

    """

    stopping_maxiter: PositiveInt = (
        10_000  # Each generation will compute the objective once
    )
    """Number of generations to evolve (the ``gen`` parameter in pygmo).

    Each generation computes the objective function once.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=10
        )

        algo_specific_options = {
            "gen": self.stopping_maxiter,
        }

        res = _minimize_pygmo(
            problem=problem,
            x0=x0,
            method="sea",
            specific_options=algo_specific_options,
            population_size=population_size,
            n_cores=1,
            seed=self.seed,
            discard_start_params=self.discard_start_params,
        )
        return res


@mark.minimizer(
    name="pygmo_sga",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_PYGMO_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class PygmoSga(Algorithm):
    """Minimize a scalar function using a simple genetic algorithm.

    A simple genetic algorithm evolves a population of candidate solutions by
    repeatedly applying selection, crossover, and mutation operators. The pygmo
    implementation offers several strategies for each of these operators. A
    detailed description of the algorithm can be found in the `pagmo2
    documentation <https://esa.github.io/pagmo2/docs/cpp/algorithms/sga.html>`_.
    See also :cite:`Oliveto2007` for a time-complexity analysis of evolutionary
    algorithms.

    Genetic algorithms are global, derivative-free, population based optimizers
    that are suited for multimodal problems where no derivatives are available
    and many function evaluations are affordable. For smooth problems with only
    one optimum, local optimizers are much more efficient.

    The algorithm requires finite bounds on all parameters.

    """

    population_size: int | None = None
    """Size of the population. If None, it is set to ``10 * (n_params + 1)``, but
    at least 64."""

    seed: int | None = None
    """Seed used by the internal random number generator."""

    discard_start_params: bool = False
    """If True, the start parameters are not guaranteed to be part of the initial
    population.

    This saves one criterion function evaluation that cannot be done in parallel
    with other evaluations. Default False.

    """

    stopping_maxiter: PositiveInt = STOPPING_MAX_ITERATIONS_GENETIC
    """Number of generations to evolve (the ``gen`` parameter in pygmo)."""

    # TODO: Refine type to fix range [0,1]
    crossover_probability: NonNegativeFloat = 0.9
    """Crossover probability. It must lie in [0, 1]. The default value is taken
    from pygmo."""

    crossover_strategy: Literal[
        "exponential",
        "sbx",
        "single",
        "binomial",
    ] = "exponential"
    """The crossover strategy.

    One of "exponential", "binomial", "single" or "sbx" (simulated binary
    crossover). Default is "exponential".

    """

    # TODO: Refine type to fix range [1,100]
    eta_c: PositiveFloat | None = None
    """Distribution index for "sbx" crossover.

    It must lie in [1, 100]. This parameter is ignored (and a warning is raised)
    if another crossover strategy is selected. If None, pygmo's default of 1.0
    is used.

    """

    # TODO: Refine type to fix range [0,1]
    mutation_probability: NonNegativeFloat = 0.02
    """Mutation probability. It must lie in [0, 1]. The default value is taken
    from pygmo."""

    mutation_strategy: Literal["uniform", "polynomial"] = "polynomial"
    """Mutation strategy. Must be "uniform" or "polynomial". Default is
    "polynomial"."""

    # TODO: Refine type to fix range [0,1]
    mutation_polynomial_distribution_index: NonNegativeFloat | None = None
    """Distribution index used by polynomial mutation (part of the ``param_m``
    parameter in pygmo).

    According to the pygmo documentation, it must lie in [1, 100]. It is only
    used if ``mutation_strategy`` is "polynomial"; otherwise a warning is raised
    and the value is ignored. If None, 1.0 is used.

    """

    # TODO: Refine type to fix range [0,1]
    mutation_gaussian_width: NonNegativeFloat | None = None
    """Width of the gaussian used by gaussian mutation (part of the ``param_m``
    parameter in pygmo).

    It must lie in [0, 1]. It is only used if the mutation strategy is
    "gaussian"; otherwise a warning is raised and the value is ignored.

    """

    selection_strategy: Literal["tournament", "truncated"] = "tournament"
    """Selection strategy. Must be "tournament" or "truncated". Default is
    "tournament"."""

    # TODO: Check if should be NonNegativeInt
    selection_truncated_n_best: int | None = None
    """Number of best individuals to use in the "truncated" selection mechanism.

    It is only used if ``selection_strategy`` is "truncated"; otherwise a
    warning is raised and the value is ignored. If None, pygmo's default of 2 is
    used.

    """

    # TODO Check if should be NonNegativeInt
    selection_tournament_size: int | None = None
    """Size of the tournament in the "tournament" selection mechanism.

    It is only used if ``selection_strategy`` is "tournament"; otherwise a
    warning is raised and the value is ignored. If None, pygmo's default of 2 is
    used.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=64
        )

        if self.eta_c is not None and self.crossover_strategy != "sbx":
            warnings.warn(
                f"You specified crossover strategy {self.crossover_strategy}"
                "and eta_c. However, eta_c is ignored because it is only used when "
                "the crossover_strategy is set to sbx."
            )
        eta_c = 1.0 if self.eta_c is None else self.eta_c

        if (
            self.mutation_polynomial_distribution_index is not None
        ) and self.mutation_strategy != "polynomial":
            warnings.warn(
                "You specified a mutation_polynomial_distribution_index but"
                "did not choose polynomial as your mutation_strategy. Thus, "
                "mutation_polynomial_distribution_index will be ignored."
            )
        if (
            self.mutation_gaussian_width is not None
            and self.mutation_strategy != "gaussian"
        ):
            warnings.warn(
                "You specified a mutation_gaussian_width but "
                "did not choose gaussion as your mutation_strategy. "
                "Thus, mutation_gaussian_width will be ignored."
            )
        if (
            self.selection_strategy != "truncated"
            and self.selection_truncated_n_best is not None
        ):
            warnings.warn(
                "You specified selection_truncated_n_best but "
                "did not specify truncated as your selection strategy. "
                "Therefore, selection_truncated_n_best is ignored."
            )
        if (
            self.selection_strategy != "tournament"
            and self.selection_tournament_size is not None
        ):
            warnings.warn(
                "You specified selection_tournament_size but "
                "did not specify tournament as your selection strategy. "
                "Therefore, selection_tournament_size is ignored."
            )

        if (
            self.mutation_strategy == "gaussian"
            and self.mutation_gaussian_width is not None
        ):
            param_m = self.mutation_gaussian_width
        elif (
            self.mutation_strategy == "polynomial"
            and self.mutation_polynomial_distribution_index is not None
        ):
            param_m = self.mutation_polynomial_distribution_index
        else:
            param_m = 1.0

        if (
            self.selection_strategy == "truncated"
            and self.selection_truncated_n_best is not None
        ):
            param_s = self.selection_truncated_n_best
        elif (
            self.selection_strategy == "tournament"
            and self.selection_tournament_size is not None
        ):
            param_s = self.selection_tournament_size
        else:
            param_s = 2

        algo_specific_options = {
            "gen": self.stopping_maxiter,
            "cr": self.crossover_probability,
            "eta_c": eta_c,
            "m": self.mutation_probability,
            "param_m": param_m,
            "crossover": self.crossover_strategy,
            "mutation": self.mutation_strategy,
            "selection": self.selection_strategy,
            "param_s": param_s,
        }

        res = _minimize_pygmo(
            problem=problem,
            x0=x0,
            method="sga",
            specific_options=algo_specific_options,
            population_size=population_size,
            n_cores=1,
            seed=self.seed,
            discard_start_params=self.discard_start_params,
        )
        return res


@mark.minimizer(
    name="pygmo_sade",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_PYGMO_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class PygmoSade(Algorithm):
    r"""Minimize a scalar function using self-adaptive differential evolution.

    The original differential evolution algorithm (pygmo_de) can be
    significantly improved by introducing the idea of parameter
    self-adaptation. Many different proposals have been made to self-adapt both
    the crossover probability :math:`CR` and the weight coefficient :math:`F`
    of the original differential evolution algorithm. pygmo's implementation
    supports two different mechanisms. The first one, proposed by
    :cite:`Brest2006`, does not make use of the differential evolution
    operators to produce new values for :math:`F` and :math:`CR` and, strictly
    speaking, is thus not self-adaptation, but rather parameter control. The
    resulting differential evolution variant is often referred to as jDE. The
    second variant is inspired by the ideas introduced by :cite:`Elsayed2011`
    and uses a variation of the selected DE operator to produce new :math:`CR`
    and :math:`F` parameters for each individual. This variant is referred to
    as iDE.

    Like plain differential evolution, this is a global, derivative-free,
    population based algorithm for multimodal problems. The self-adaptation
    removes the need to tune :math:`F` and :math:`CR` by hand.

    The algorithm requires finite bounds on all parameters.

    """

    population_size: int | None = None
    """Size of the population. If None, it is set to ``10 * (n_params + 1)``, but
    at least 64."""

    seed: int | None = None
    """Seed used by the internal random number generator."""

    discard_start_params: bool = False
    """If True, the start parameters are not guaranteed to be part of the initial
    population.

    This saves one criterion function evaluation that cannot be done in parallel
    with other evaluations. Default False.

    """

    jde: bool = True
    r"""Whether to use the jDE self-adaptation variant (:cite:`Brest2006`) to
    control the :math:`F` and :math:`CR` parameters.

    If True, jDE is used, otherwise iDE (:cite:`Elsayed2011`).

    """

    stopping_maxiter: PositiveInt = STOPPING_MAX_ITERATIONS_GENETIC
    """Number of generations to evolve (the ``gen`` parameter in pygmo)."""

    mutation_variant: Literal[
        "best/1/exp",
        "rand/1/exp",
        "rand-to-best/1/exp",
        "best/2/exp",
        "rand/2/exp",
        "best/1/bin",
        "rand/1/bin",
        "rand-to-best/1/bin",
        "best/2/bin",
        "rand/2/bin",
        "rand/3/exp",
        "rand/3/bin",
        "best/3/exp",
        "best/3/bin",
        "rand-to-current/2/exp",
        "rand-to-current/2/bin",
        "rand-to-best-and-current/2/exp",
        "rand-to-best-and-current/2/bin",
    ] = "rand/1/exp"
    """Mutation variant used to create a new candidate individual.

    The default is "rand/1/exp". The first ten variants are the classical
    mutation variants introduced in the original DE algorithm; the remaining
    ones are considered in the work by :cite:`Elsayed2011`. The following
    variants are available (the pygmo integer code given in parentheses is
    accepted as well):

    - "best/1/exp" (1)
    - "rand/1/exp" (2)
    - "rand-to-best/1/exp" (3)
    - "best/2/exp" (4)
    - "rand/2/exp" (5)
    - "best/1/bin" (6)
    - "rand/1/bin" (7)
    - "rand-to-best/1/bin" (8)
    - "best/2/bin" (9)
    - "rand/2/bin" (10)
    - "rand/3/exp" (11)
    - "rand/3/bin" (12)
    - "best/3/exp" (13)
    - "best/3/bin" (14)
    - "rand-to-current/2/exp" (15)
    - "rand-to-current/2/bin" (16)
    - "rand-to-best-and-current/2/exp" (17)
    - "rand-to-best-and-current/2/bin" (18)

    """

    keep_adapted_params: bool = False
    r"""When True, the adapted parameters :math:`CR` and :math:`F` are not reset
    between successive calls to the evolve method (the ``memory`` parameter in
    pygmo).

    Default is False.

    """

    ftol: NonNegativeFloat = 1e-6
    """Stop when the absolute difference between the best and the worst objective
    value in the population is smaller than this value.

    The default value is taken from pygmo.

    """

    xtol: NonNegativeFloat = 1e-6
    """Stop when the sum of absolute differences between the best and the worst
    parameter vector in the population is smaller than this value.

    The default value is taken from pygmo.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=64
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
            str_to_int=mutation_variant_str_to_int, value=self.mutation_variant
        )

        algo_specific_options = {
            "gen": self.stopping_maxiter,
            "variant": mutation_variant,
            "variant_adptv": 1 if self.jde else 2,
            "ftol": self.ftol,
            "xtol": self.xtol,
            "memory": self.keep_adapted_params,
        }

        res = _minimize_pygmo(
            problem=problem,
            x0=x0,
            method="sade",
            specific_options=algo_specific_options,
            population_size=population_size,
            n_cores=1,
            seed=self.seed,
            discard_start_params=self.discard_start_params,
        )
        return res


@mark.minimizer(
    name="pygmo_cmaes",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_PYGMO_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class PygmoCmaes(Algorithm):
    """Minimize a scalar function using the covariance matrix evolutionary
    strategy.

    CMA-ES is one of the most successful algorithms, classified as an
    evolutionary strategy, for derivative-free global optimization. The version
    supported by optimagic is the version described in :cite:`Hansen2006`.

    CMA-ES samples new candidate solutions from a multivariate normal
    distribution whose covariance matrix is adapted based on the success of
    previous generations. It is a good default choice among the population based
    pygmo optimizers for continuous problems that are multimodal, non-smooth, or
    noisy and for which no derivatives are available. Note that the algorithm is
    not elitist, i.e. the best solution found so far is not guaranteed to be
    part of the current population.

    In contrast to the pygmo version, optimagic always sets ``force_bounds`` to
    True. This avoids that ill defined parameter values are evaluated, at the
    cost of a somewhat worse covariance matrix adaptation.

    The algorithm requires finite bounds on all parameters.

    """

    population_size: int | None = None
    """Size of the population. If None, it is set to ``10 * (n_params + 1)``, but
    at least 64."""

    seed: int | None = None
    """Seed used by the internal random number generator."""

    discard_start_params: bool = False
    """If True, the start parameters are not guaranteed to be part of the initial
    population.

    This saves one criterion function evaluation that cannot be done in parallel
    with other evaluations. Default False.

    """

    stopping_maxiter: PositiveInt = STOPPING_MAX_ITERATIONS_GENETIC
    """Number of generations to evolve (the ``gen`` parameter in pygmo)."""

    # TODO: Refine type to fix range [0,1]
    backward_horizon: NonNegativeFloat | None = None
    """Backward time horizon for the evolution path (``cc`` in the pygmo
    documentation).

    It must lie between 0 and 1. If None, it is determined automatically by
    pygmo.

    """

    # TODO: Refine type to fix range [0,1]
    variance_loss_compensation: NonNegativeFloat | None = None
    """Makes partly up for the small variance loss in case the indicator is zero
    (``cs`` in the pygmo documentation and in the MATLAB code of
    :cite:`Hansen2006`).

    It must lie between 0 and 1. If None, it is determined automatically by
    pygmo.

    """

    # TODO: Refine type to fix range [0,1]
    learning_rate_rank_one_update: NonNegativeFloat | None = None
    """Learning rate for the rank-one update of the covariance matrix (``c1`` in
    the pygmo and pagmo documentation).

    It must lie between 0 and 1. If None, it is determined automatically by
    pygmo.

    """

    # TODO: Refine type to fix range [0,1]
    learning_rate_rank_mu_update: NonNegativeFloat | None = None
    """Learning rate for the rank-mu update of the covariance matrix (``cmu`` in
    the pygmo and pagmo documentation).

    It must lie between 0 and 1. If None, it is determined automatically by
    pygmo.

    """

    # TODO: Check if should be NonNegativeFloat
    initial_step_size: float = 0.5
    r"""Initial step size, :math:`\sigma^0` in the original paper.

    The default value is taken from pygmo.

    """

    ftol: NonNegativeFloat = 1e-6
    """Stop when the absolute difference between the best and the worst objective
    value in the population is smaller than this value.

    The default value is taken from pygmo.

    """

    xtol: NonNegativeFloat = 1e-6
    """Stopping tolerance in the parameter space.

    The algorithm stops when the sampling distribution has become so narrow that
    the typical parameter change falls below this value. The default value is
    taken from pygmo.

    """

    keep_adapted_params: bool = False
    """When True, the adapted parameters are not reset between successive calls
    to the evolve method (the ``memory`` parameter in pygmo).

    Default is False.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=64
        )

        algo_specific_options = {
            "gen": self.stopping_maxiter,
            "cc": self.backward_horizon if self.backward_horizon is not None else -1.0,
            "cs": self.variance_loss_compensation
            if self.variance_loss_compensation is not None
            else -1.0,
            "c1": self.learning_rate_rank_one_update
            if self.learning_rate_rank_one_update is not None
            else -1.0,
            "cmu": self.learning_rate_rank_mu_update
            if self.learning_rate_rank_mu_update is not None
            else -1.0,
            "sigma0": self.initial_step_size,
            "ftol": self.ftol,
            "xtol": self.xtol,
            "memory": self.keep_adapted_params,
            "force_bounds": True,
        }

        res = _minimize_pygmo(
            problem=problem,
            x0=x0,
            method="cmaes",
            specific_options=algo_specific_options,
            population_size=population_size,
            n_cores=1,
            seed=self.seed,
            discard_start_params=self.discard_start_params,
        )
        return res


@mark.minimizer(
    name="pygmo_simulated_annealing",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_PYGMO_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class PygmoSimulatedAnnealing(Algorithm):
    """Minimize a scalar function with the simulated annealing algorithm.

    This version of the simulated annealing algorithm is, essentially, an
    iterative random search procedure with adaptive moves along the coordinate
    directions. It permits uphill moves under the control of a metropolis
    criterion, in the hope to avoid the first local minima encountered. This
    version is the one proposed in :cite:`Corana1987`.

    Simulated annealing is a derivative-free global optimizer that is suited
    for multimodal problems where no derivatives are available. It should not
    be used for stochastic (noisy) problems.

    .. note::
        When selecting the starting and final temperature values it helps to
        think about the temperature as the deterioration in the objective
        function value that still has a 37% chance of being accepted.

    The algorithm requires finite bounds on all parameters.

    """

    population_size: int | None = None
    """Size of the population. If None, it is set to ``10 * (n_params + 1)``, but
    at least 64."""

    seed: int | None = None
    """Seed used by the internal random number generator."""

    discard_start_params: bool = False
    """If True, the start parameters are not guaranteed to be part of the initial
    population.

    This saves one criterion function evaluation that cannot be done in parallel
    with other evaluations. Default False.

    """

    start_temperature: PositiveFloat = 10.0
    """Starting temperature. Must be > 0. The default value is taken from
    pygmo."""

    # TODO: Check if type should be same as start_temperature
    end_temperature: float = 0.01
    """Final temperature.

    It must be positive and smaller than ``start_temperature``. Our default
    (0.01) is lower than the default in pygmo and pagmo (0.1).

    """

    # TODO: Check if type should be NonNegativeInt
    n_temp_adjustments: int = 10
    """Number of temperature adjustments in the annealing schedule."""

    # TODO: Check if type should be NonNegativeInt
    n_range_adjustments: int = 10
    """Number of adjustments of the search range performed at a constant
    temperature."""

    # TODO: Check if type should be NonNegativeInt
    bin_size: int = 10
    """Number of mutations that are used to compute the acceptance rate."""

    # TODO: Refine type to fix range [0,1]
    start_range: NonNegativeFloat = 1.0
    """Starting range for mutating the decision vector. It must lie in (0,
    1]."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=64
        )

        algo_specific_options = {
            "Ts": self.start_temperature,
            "Tf": self.end_temperature,
            "n_T_adj": self.n_temp_adjustments,
            "n_range_adj": self.n_range_adjustments,
            "bin_size": self.bin_size,
            "start_range": self.start_range,
        }

        res = _minimize_pygmo(
            problem=problem,
            x0=x0,
            method="simulated_annealing",
            specific_options=algo_specific_options,
            population_size=population_size,
            n_cores=1,
            seed=self.seed,
            discard_start_params=self.discard_start_params,
        )
        return res


@mark.minimizer(
    name="pygmo_pso",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_PYGMO_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class PygmoPso(Algorithm):
    r"""Minimize a scalar function using particle swarm optimization.

    Particle swarm optimization (PSO) is a population based algorithm inspired
    by the foraging behaviour of swarms, originally proposed by
    :cite:`Kennedy1995`. In PSO each point has memory of the position where it
    achieved the best performance :math:`x^l_i` (local memory) and of the best
    decision vector :math:`x^g` in a certain neighbourhood, and uses this
    information to update its position. For a survey on particle swarm
    optimization algorithms, see :cite:`Poli2007`.

    Each particle determines its future position :math:`x_{i+1} = x_i + v_i`
    where

    .. math:: v_{i+1} = \omega (v_i + \eta_1 \cdot \mathbf{r}_1 \cdot (x_i -
        x^{l}_i) + \eta_2 \cdot \mathbf{r}_2 \cdot (x_i - x^g))

    PSO is a global, derivative-free optimizer suited for multimodal problems
    where no derivatives are available and many function evaluations are
    affordable. This version updates one particle at a time and should not be
    used for stochastic problems; see pygmo_pso_gen for a generational variant
    that supports them and allows parallelization.

    The algorithm requires finite bounds on all parameters.

    """

    population_size: int | None = None
    """Size of the population. If None, it is set to ``10 * (n_params + 1)``, but
    at least 10."""

    seed: int | None = None
    """Seed used by the internal random number generator."""

    discard_start_params: bool = False
    """If True, the start parameters are not guaranteed to be part of the initial
    population.

    This saves one criterion function evaluation that cannot be done in parallel
    with other evaluations. Default False.

    """

    stopping_maxiter: PositiveInt = STOPPING_MAX_ITERATIONS_GENETIC
    """Number of generations to evolve (the ``gen`` parameter in pygmo)."""

    # TODO: Refine type to fix range [0,1]
    omega: NonNegativeFloat = 0.7298
    r"""Depending on the variant chosen, :math:`\omega` is the particles'
    inertia weight or the constriction coefficient.

    It must lie between 0 and 1. The default value is taken from pygmo.

    """

    # TODO: Refine type to fix range [0,4]
    force_of_previous_best: NonNegativeFloat = 2.05
    r""":math:`\eta_1` in the equation above.

    It is the magnitude of the force, applied to the particle's velocity, in
    the direction of its previous best position. It must lie between 0 and 4.
    The default value is taken from pygmo.

    """

    # TODO: Refine type to fix range [0,4]
    force_of_best_in_neighborhood: NonNegativeFloat = 2.05
    r""":math:`\eta_2` in the equation above.

    It is the magnitude of the force, applied to the particle's velocity, in
    the direction of the best position in its neighborhood. It must lie between
    0 and 4. The default value is taken from pygmo.

    """

    # TODO: Refine type to fix range [0,1]
    max_velocity: NonNegativeFloat = 0.5
    """Maximum allowed particle velocity as fraction of the box bounds.

    It must lie between 0 and 1. The default value is taken from pygmo.

    """

    algo_variant: Literal[
        "canonical_inertia",
        "social_and_cog_rand",
        "all_components_rand",
        "one_rand",
        "canonical_constriction",
        "fips",
    ] = "canonical_constriction"
    """Algorithm variant to be used:

    - "canonical_inertia" (1): canonical PSO with inertia weight
    - "social_and_cog_rand" (2): same social and cognitive random vector
    - "all_components_rand" (3): same random vector for all components
    - "one_rand" (4): only one random number per velocity update
    - "canonical_constriction" (5): canonical PSO with constriction factor
    - "fips" (6): fully informed particle swarm

    The pygmo integer code given in parentheses is accepted as well.

    """

    neighbor_definition: Literal[
        "gbest",
        "lbest",
        "Von Neumann",
        "Adaptive random",
    ] = "lbest"
    """Swarm topology that defines each particle's neighbors.

    One of "gbest" (1), "lbest" (2), "Von Neumann" (3) or "Adaptive random"
    (4). The pygmo integer code given in parentheses is accepted as well.

    """

    neighbor_param: int | None = None
    """The neighbourhood parameter.

    If the "lbest" topology is selected, it represents each particle's indegree
    (also outdegree) in the swarm topology. Particles have neighbours up to a
    radius of ``neighbor_param / 2`` in the ring. If the "Adaptive random"
    topology is selected, it represents each particle's maximum outdegree in
    the swarm topology; the minimum outdegree is 1 (the particle always
    connects back to itself). If the neighbor definition is "gbest" or "Von
    Neumann", this parameter is ignored and a warning is raised if it is set.
    If None, pygmo's default of 4 is used.

    """

    keep_velocities: bool = False
    """When True, the particle velocities are not reset between successive calls
    to the evolve method (the ``memory`` parameter in pygmo)."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if (
            self.neighbor_definition in ["gbest", "Von Neumann"]
            and self.neighbor_param is not None
        ):
            warnings.warn(
                "You gave a neighbor parameter but selected a neighbor_definition "
                "that ignores this parameter."
            )

        neighbor_param = 4 if self.neighbor_param is None else self.neighbor_param

        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=10
        )

        neighbor_definition_str_to_int = {
            "gbest": 1,
            "lbest": 2,
            "Von Neumann": 3,
            "Adaptive random": 4,
        }
        algo_variant_str_to_int = {
            "canonical_inertia": 1,
            "social_and_cog_rand": 2,
            "all_components_rand": 3,
            "one_rand": 4,
            "canonical_constriction": 5,
            "fips": 6,
        }

        algo_specific_options = {
            "gen": self.stopping_maxiter,
            "omega": self.omega,
            "eta1": self.force_of_previous_best,
            "eta2": self.force_of_best_in_neighborhood,
            "max_vel": self.max_velocity,
            "variant": _convert_str_to_int(algo_variant_str_to_int, self.algo_variant),
            "neighb_type": _convert_str_to_int(
                neighbor_definition_str_to_int, self.neighbor_definition
            ),
            "neighb_param": neighbor_param,
            "memory": self.keep_velocities,
        }

        res = _minimize_pygmo(
            problem=problem,
            x0=x0,
            method="pso",
            specific_options=algo_specific_options,
            population_size=population_size,
            n_cores=1,
            seed=self.seed,
            discard_start_params=self.discard_start_params,
        )

        return res


@mark.minimizer(
    name="pygmo_pso_gen",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_PYGMO_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=True,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class PygmoPsoGen(Algorithm):
    r"""Minimize a scalar function with generational particle swarm
    optimization.

    Particle swarm optimization (generational) is identical to pygmo_pso, but
    it updates the velocities of all particles before the new particle
    positions are computed (taking into consideration all updated particle
    velocities). The whole population is thus evaluated in one batch per
    generation, as opposed to the standard PSO, which evaluates a single
    particle at a time. Consequently, the generational PSO algorithm is suited
    for stochastic optimization problems and supports parallel function
    evaluation via ``n_cores``.

    For a survey on particle swarm optimization algorithms, see
    :cite:`Poli2007`.

    Each particle determines its future position :math:`x_{i+1} = x_i + v_i`
    where

    .. math:: v_{i+1} = \omega (v_i + \eta_1 \cdot \mathbf{r}_1 \cdot (x_i -
        x^{l}_i) + \eta_2 \cdot \mathbf{r}_2 \cdot (x_i - x^g))

    The algorithm requires finite bounds on all parameters.

    """

    population_size: int | None = None
    """Size of the population. If None, it is set to ``10 * (n_params + 1)``, but
    at least 10."""

    n_cores: PositiveInt = 1
    """Number of cores used for parallel function evaluation."""

    seed: int | None = None
    """Seed used by the internal random number generator."""

    discard_start_params: bool = False
    """If True, the start parameters are not guaranteed to be part of the initial
    population.

    This saves one criterion function evaluation that cannot be done in parallel
    with other evaluations. Default False.

    """

    stopping_maxiter: PositiveInt = STOPPING_MAX_ITERATIONS_GENETIC
    """Number of generations to evolve (the ``gen`` parameter in pygmo)."""

    # TODO: Refine type to fix range [0,1]
    omega: NonNegativeFloat = 0.7298
    r"""Depending on the variant chosen, :math:`\omega` is the particles'
    inertia weight or the constriction coefficient.

    It must lie between 0 and 1. The default value is taken from pygmo.

    """

    # TODO: Refine type to fix range [0,4]
    force_of_previous_best: NonNegativeFloat = 2.05
    r""":math:`\eta_1` in the equation above.

    It is the magnitude of the force, applied to the particle's velocity, in
    the direction of its previous best position. It must lie between 0 and 4.
    The default value is taken from pygmo.

    """

    # TODO: Refine type to fix range [0,4]
    force_of_best_in_neighborhood: NonNegativeFloat = 2.05
    r""":math:`\eta_2` in the equation above.

    It is the magnitude of the force, applied to the particle's velocity, in
    the direction of the best position in its neighborhood. It must lie between
    0 and 4. The default value is taken from pygmo.

    """

    # TODO: Refine type to fix range [0,1]
    max_velocity: NonNegativeFloat = 0.5
    """Maximum allowed particle velocity as fraction of the box bounds.

    It must lie between 0 and 1. The default value is taken from pygmo.

    """

    algo_variant: Literal[
        "canonical_inertia",
        "social_and_cog_rand",
        "all_components_rand",
        "one_rand",
        "canonical_constriction",
        "fips",
    ] = "canonical_constriction"
    """Algorithm variant to be used:

    - "canonical_inertia" (1): canonical PSO with inertia weight
    - "social_and_cog_rand" (2): same social and cognitive random vector
    - "all_components_rand" (3): same random vector for all components
    - "one_rand" (4): only one random number per velocity update
    - "canonical_constriction" (5): canonical PSO with constriction factor
    - "fips" (6): fully informed particle swarm

    The pygmo integer code given in parentheses is accepted as well.

    """

    neighbor_definition: Literal[
        "gbest",
        "lbest",
        "Von Neumann",
        "Adaptive random",
    ] = "lbest"
    """Swarm topology that defines each particle's neighbors.

    One of "gbest" (1), "lbest" (2), "Von Neumann" (3) or "Adaptive random"
    (4). The pygmo integer code given in parentheses is accepted as well.

    """

    neighbor_param: int | None = None
    """The neighbourhood parameter.

    If the "lbest" topology is selected, it represents each particle's indegree
    (also outdegree) in the swarm topology. Particles have neighbours up to a
    radius of ``neighbor_param / 2`` in the ring. If the "Adaptive random"
    topology is selected, it represents each particle's maximum outdegree in
    the swarm topology; the minimum outdegree is 1 (the particle always
    connects back to itself). If the neighbor definition is "gbest" or "Von
    Neumann", this parameter is ignored and a warning is raised if it is set.
    If None, pygmo's default of 4 is used.

    """

    keep_velocities: bool = False
    """When True, the particle velocities are not reset between successive calls
    to the evolve method (the ``memory`` parameter in pygmo)."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if (
            self.neighbor_definition in ["gbest", "Von Neumann"]
            and self.neighbor_param is not None
        ):
            warnings.warn(
                "You gave a neighbor parameter but selected a neighbor_definition "
                "that ignores this parameter."
            )
        neighbor_param = 4 if self.neighbor_param is None else self.neighbor_param
        neighbor_str_to_int = {
            "gbest": 1,
            "lbest": 2,
            "Von Neumann": 3,
            "Adaptive random": 4,
        }
        neighbor_type = _convert_str_to_int(
            neighbor_str_to_int, self.neighbor_definition
        )
        algo_variant_str_to_int = {
            "canonical_inertia": 1,
            "social_and_cog_rand": 2,
            "all_components_rand": 3,
            "one_rand": 4,
            "canonical_constriction": 5,
            "fips": 6,
        }
        algo_variant = _convert_str_to_int(algo_variant_str_to_int, self.algo_variant)

        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=10
        )

        algo_specific_options = {
            "gen": self.stopping_maxiter,
            "omega": self.omega,
            "eta1": self.force_of_previous_best,
            "eta2": self.force_of_best_in_neighborhood,
            "max_vel": self.max_velocity,
            "variant": algo_variant,
            "neighb_type": neighbor_type,
            "neighb_param": neighbor_param,
            "memory": self.keep_velocities,
        }

        res = _minimize_pygmo(
            problem=problem,
            x0=x0,
            method="pso_gen",
            specific_options=algo_specific_options,
            population_size=population_size,
            n_cores=self.n_cores,
            seed=self.seed,
            discard_start_params=self.discard_start_params,
        )
        return res


@mark.minimizer(
    name="pygmo_mbh",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_PYGMO_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class PygmoMbh(Algorithm):
    r"""Minimize a scalar function using generalized monotonic basin hopping.

    Monotonic basin hopping, or simply basin hopping, is an algorithm rooted in
    the idea of mapping the objective function :math:`f(x_0)` into the local
    minimum found when starting a local search from :math:`x_0`. This simple
    idea allows a substantial increase of efficiency in solving problems, such
    as the Lennard-Jones cluster or the MGA-1DSM interplanetary trajectory
    problem, that are conjectured to have a so-called funnel structure. See
    :cite:`Wales1997` for the paper introducing the basin hopping idea for a
    Lennard-Jones cluster optimization.

    pygmo provides an original generalization of this concept resulting in a
    meta-algorithm that operates on a population: the population is randomly
    perturbed, the inner algorithm is run on it, and the result is kept only if
    it improves upon the best known solution. When a population containing a
    single individual is used, the original method is recovered.

    Basin hopping is a good choice for derivative-free problems with many local
    optima in which good local optima are close to each other (a funnel
    structure). The algorithm requires finite bounds on all parameters.

    """

    population_size: int | None = None
    """Size of the population. If None, it is set to ``10 * (n_params + 1)``, but
    at least 250."""

    seed: int | None = None
    """Seed used by the internal random number generator."""

    discard_start_params: bool = False
    """If True, the start parameters are not guaranteed to be part of the initial
    population.

    This saves one criterion function evaluation that cannot be done in parallel
    with other evaluations. Default False.

    """

    inner_algorithm: pg.algorithm | None = None
    """A pygmo algorithm or a user-defined pygmo algorithm, either C++ or Python,
    that is used for the inner (local) searches.

    If None, the ``pygmo.compass_search`` algorithm is used.

    """

    # this is 30 instead of 5 in pygmo for our sum of squares test to pass
    stopping_max_inner_runs_without_improvement: PositiveInt = 30
    """Number of consecutive runs of the inner algorithm that need to result in
    no improvement for the algorithm to stop (the ``stop`` parameter in pygmo).

    optimagic uses a default of 30, whereas pygmo's default is 5.

    """

    perturbation: float = 0.01
    """The perturbation to be applied to each component of the decision vector
    when a new starting point is generated, expressed as a fraction of the width
    of the bounds.

    It must lie in (0, 1]. The default value is taken from pygmo.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        # the min default population size is this large to pass our sum of
        # squares tests.
        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=250
        )

        algo_specific_options = {
            "algo": self.inner_algorithm,
            "stop": self.stopping_max_inner_runs_without_improvement,
            "perturb": self.perturbation,
        }

        res = _minimize_pygmo(
            problem=problem,
            x0=x0,
            method="mbh",
            specific_options=algo_specific_options,
            population_size=population_size,
            seed=self.seed,
            discard_start_params=self.discard_start_params,
            n_cores=1,
        )
        return res


@mark.minimizer(
    name="pygmo_xnes",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_PYGMO_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class PygmoXnes(Algorithm):
    """Minimize a scalar function using exponential natural evolution
    strategies.

    Exponential natural evolution strategies (xNES) is an algorithm closely
    related to CMA-ES and based on the adaptation of a gaussian sampling
    distribution via the so-called natural gradient. Like CMA-ES, it is based
    on the idea of sampling new trial vectors from a multivariate distribution
    and using the new sampled points to update the distribution parameters.
    Naively, this could be done following the gradient of the expected fitness
    as approximated by a finite number of sampled points. While this idea
    offers a powerful lead on algorithmic construction, it has some major
    drawbacks that are solved in the so-called natural evolution strategies
    class of algorithms by adopting, instead, the natural gradient. xNES is one
    of the most performing variants in this class.

    See :cite:`Glasmachers2010` and the `pagmo documentation on xNES
    <https://esa.github.io/pagmo2/docs/cpp/algorithms/xnes.html>`_ for details.

    Like CMA-ES, xNES is a global, derivative-free, population based optimizer
    for continuous problems that are multimodal, non-smooth, or noisy. The
    algorithm is not elitist, i.e. the best solution found so far is not
    guaranteed to be part of the current population. In contrast to the pygmo
    version, optimagic always sets ``force_bounds`` to True, so the criterion
    function is never evaluated outside of the bounds, at the cost of a
    somewhat worse adaptation of the sampling distribution.

    The algorithm requires finite bounds on all parameters.

    """

    population_size: float | None = None
    """Size of the population. If None, it is set to ``10 * (n_params + 1)``, but
    at least 64."""

    seed: int | None = None
    """Seed used by the internal random number generator."""

    discard_start_params: bool = False
    """If True, the start parameters are not guaranteed to be part of the initial
    population.

    This saves one criterion function evaluation that cannot be done in parallel
    with other evaluations. Default False.

    """

    stopping_maxiter: PositiveInt = STOPPING_MAX_ITERATIONS_GENETIC
    """Number of generations to evolve (the ``gen`` parameter in pygmo)."""

    # TODO: Refine type to fix range [0,1]
    learning_rate_mean_update: NonNegativeFloat | None = 1.0
    r"""Learning rate for the mean update (:math:`\eta_\mu` in the pygmo
    documentation).

    It must lie between 0 and 1. If None, it is chosen automatically by pygmo.

    """

    # TODO: Refine type to fix range [0,1]
    learning_rate_step_size_update: NonNegativeFloat | None = None
    """Learning rate for the step-size update (``eta_sigma`` in the pygmo
    documentation).

    It must lie between 0 and 1. If None, it is chosen automatically by pygmo.

    """

    # TODO: Refine type to fix range [0,1]
    learning_rate_cov_matrix_update: NonNegativeFloat | None = None
    """Learning rate for the covariance matrix update (``eta_b`` in the pygmo
    documentation).

    It must lie between 0 and 1. If None, it is chosen automatically by pygmo.

    """

    # TODO: Refine type to fix range [0,1]
    initial_search_share: NonNegativeFloat | None = 1.0
    """Share of the given search space that will be initially searched
    (``sigma0`` in the pygmo documentation).

    The width of the initial sampling distribution along dimension ``i`` is
    ``initial_search_share * (upper_bound_i - lower_bound_i)``. It must lie
    between 0 and 1. Default is 1.

    """

    ftol: NonNegativeFloat = 1e-6
    """Stop when the absolute difference between the best and the worst objective
    value in the population is smaller than this value.

    The default value is taken from pygmo.

    """

    xtol: NonNegativeFloat = 1e-6
    """Stopping tolerance in the parameter space.

    The algorithm stops when the sampling distribution has become so narrow that
    the typical parameter change falls below this value. The default value is
    taken from pygmo.

    """

    keep_adapted_params: bool = False
    """When True, the adapted parameters are not reset between successive calls
    to the evolve method (the ``memory`` parameter in pygmo).

    Default is False.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=64
        )
        eta_mu = (
            -1
            if self.learning_rate_mean_update is None
            else self.learning_rate_mean_update
        )
        eta_sigma = (
            -1
            if self.learning_rate_step_size_update is None
            else self.learning_rate_step_size_update
        )
        eta_b = (
            -1
            if self.learning_rate_cov_matrix_update is None
            else self.learning_rate_cov_matrix_update
        )
        algo_specific_options = {
            "gen": self.stopping_maxiter,
            "eta_mu": eta_mu,
            "eta_sigma": eta_sigma,
            "eta_b": eta_b,
            "sigma0": self.initial_search_share,
            "ftol": self.ftol,
            "xtol": self.xtol,
            "memory": self.keep_adapted_params,
            "force_bounds": True,
        }

        res = _minimize_pygmo(
            problem=problem,
            x0=x0,
            method="xnes",
            specific_options=algo_specific_options,
            population_size=population_size,
            n_cores=1,
            seed=self.seed,
            discard_start_params=self.discard_start_params,
        )
        return res


@mark.minimizer(
    name="pygmo_gwo",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_PYGMO_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class PygmoGwo(Algorithm):
    """Minimize a scalar function using the grey wolf optimizer.

    The grey wolf optimizer was proposed by :cite:`Mirjalili2014`. The pygmo
    implementation that is wrapped by optimagic is based on the pseudo-code
    provided in that paper. The algorithm mimics the leadership hierarchy and
    hunting mechanism of grey wolves, with the three best solutions (called
    alpha, beta, and delta) guiding the movement of the remaining population.

    This algorithm is a classic example of a highly criticizable line of
    research that led in the first decades of our millennium to the development
    of an entire zoo of metaphors inspiring optimization heuristics. In the
    opinion of the pagmo developers, they are often but small variations of
    already existing heuristics rebranded with unnecessary and convoluted
    biological metaphors. In the case of GWO this is particularly evident as
    the position update rule is strikingly trivial and can also be easily seen
    as a product of an evolutionary metaphor or a particle swarm one. Such an
    update rule is also not particularly effective and results in a rather poor
    performance most of the time.

    The algorithm requires finite bounds on all parameters.

    """

    population_size: int | None = None
    """Size of the population. If None, it is set to ``10 * (n_params + 1)``, but
    at least 64."""

    seed: int | None = None
    """Seed used by the internal random number generator."""

    discard_start_params: bool = False
    """If True, the start parameters are not guaranteed to be part of the initial
    population.

    This saves one criterion function evaluation that cannot be done in parallel
    with other evaluations. Default False.

    """

    stopping_maxiter: PositiveInt = STOPPING_MAX_ITERATIONS_GENETIC
    """Number of generations to evolve (the ``gen`` parameter in pygmo)."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=64
        )
        algo_specific_options = {
            "gen": self.stopping_maxiter,
        }
        res = _minimize_pygmo(
            problem=problem,
            x0=x0,
            method="gwo",
            specific_options=algo_specific_options,
            population_size=population_size,
            n_cores=1,
            seed=self.seed,
            discard_start_params=self.discard_start_params,
        )
        return res


@mark.minimizer(
    name="pygmo_compass_search",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_PYGMO_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class PygmoCompassSearch(Algorithm):
    """Minimize a scalar function using compass search.

    Compass search is a direct (pattern) search method: at each step, the
    objective function is evaluated at points obtained by changing one
    parameter at a time by the current step size. If an improvement is found,
    the search moves there; otherwise, the step size is reduced. The algorithm
    is described in :cite:`Kolda2003`.

    In contrast to the other pygmo algorithms, compass search is a local,
    derivative-free optimizer. It is considered slow but reliable and is a
    reasonable choice for non-differentiable problems where only a local
    optimum is needed. It should not be used for stochastic problems.

    The algorithm requires finite bounds on all parameters.

    """

    population_size: int | None = None
    """Size of the initial population.

    Compass search itself is not population based, so this value only
    determines how many random initial points are evaluated when the population
    is created; the search then continues from the best of them. If a value is
    specified, a warning is raised. If None, 100 is used.

    """

    seed: int | None = None
    """Seed used by the internal random number generator."""

    discard_start_params: bool = False
    """If True, the start parameters are not guaranteed to be part of the initial
    population.

    This saves one criterion function evaluation that cannot be done in parallel
    with other evaluations. Default False.

    """

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL
    """Maximum number of function evaluations (the ``max_fevals`` parameter in
    pygmo)."""

    # TODO: Refine type to fix range (0,1]
    start_range: PositiveFloat = 0.1
    """The start range. Must lie in (0, 1]. The default value is taken from
    pygmo."""

    # TODO?: mus be in (0,start_range]
    stop_range: PositiveFloat = 0.01
    """The stop range; the search stops when the range falls below this value.

    Must lie in (0, start_range]. The default value is taken from pygmo.

    """

    # TODO: Refine type to fix range (0,1)
    reduction_coeff: PositiveFloat = 0.5
    """The range reduction coefficient by which the range is multiplied when no
    improvement is found.

    Must lie in (0, 1). The default value is taken from pygmo.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if self.population_size is not None:
            warnings.warn(
                f"You specified population size {self.population_size}. "
                "compass_search does not have a population so this argument is ignored."
            )
            population_size = self.population_size
        else:
            # if discard_start_params is False population_size - 1
            # must still be positive
            population_size = 100

        algo_specific_options = {
            "max_fevals": self.stopping_maxfun,
            "start_range": self.start_range,
            "stop_range": self.stop_range,
            "reduction_coeff": self.reduction_coeff,
        }

        res = _minimize_pygmo(
            problem=problem,
            x0=x0,
            method="compass_search",
            specific_options=algo_specific_options,
            population_size=population_size,
            n_cores=1,
            seed=self.seed,
            discard_start_params=self.discard_start_params,
        )
        return res


@mark.minimizer(
    name="pygmo_ihs",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_PYGMO_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class PygmoIhs(Algorithm):
    """Minimize a scalar function using the improved harmony search algorithm.

    Improved harmony search (IHS) was introduced by :cite:`Mahdavi2007`.
    Harmony search mimics the improvisation process of a music orchestra: new
    candidate solutions ("harmonies") are composed by recombining components of
    the solutions stored in the harmony memory (the population), by adjusting
    them ("pitch adjustment"), or by drawing random values. In the improved
    variant, the pitch adjustment rate and the adjustment bandwidth vary over
    the course of the optimization (linearly and exponentially, respectively).

    IHS is a derivative-free global optimizer for multimodal problems. Each
    generation creates and evaluates exactly one new candidate solution, so
    progress is slow but steady. IHS supports stochastic problems.

    The algorithm requires finite bounds on all parameters.

    """

    population_size: int | None = None
    """Size of the population (the harmony memory).

    If None, it is set to ``10 * (n_params + 1)``. If a value is specified, a
    warning is raised because the population size has no effect on the
    performance of IHS.

    """

    seed: int | None = None
    """Seed used by the internal random number generator."""

    discard_start_params: bool = False
    """If True, the start parameters are not guaranteed to be part of the initial
    population.

    This saves one criterion function evaluation that cannot be done in parallel
    with other evaluations. Default False.

    """

    stopping_maxiter: PositiveInt = STOPPING_MAX_ITERATIONS_GENETIC
    """Number of generations to evolve (the ``gen`` parameter in pygmo).

    Each generation computes the objective function once.

    """

    # TODO: Probably refine type to fix range [0,1]
    choose_from_memory_probability: NonNegativeFloat = 0.85
    """Probability of choosing from memory (similar to a crossover probability;
    the ``phmcr`` parameter in pygmo).

    It must lie between 0 and 1. The default value is taken from pygmo.

    """

    # TODO: Refine type to fix range [0,1]
    min_pitch_adjustment_rate: NonNegativeFloat = 0.35
    """Minimum pitch adjustment rate (similar to a mutation rate; the
    ``ppar_min`` parameter in pygmo).

    It must lie between 0 and 1. The default value is taken from pygmo.

    """

    # TODO: Refine type to fix range [0,1]
    max_pitch_adjustment_rate: NonNegativeFloat = 0.99
    """Maximum pitch adjustment rate (similar to a mutation rate; the
    ``ppar_max`` parameter in pygmo).

    It must lie between 0 and 1 and be larger than
    ``min_pitch_adjustment_rate``. The default value is taken from pygmo.

    """

    min_distance_bandwidth: PositiveFloat = 1e-5
    """Minimum distance bandwidth (similar to a mutation width; the ``bw_min``
    parameter in pygmo).

    It must be positive. The default value is taken from pygmo.

    """

    max_distance_bandwidth: PositiveFloat = 1.0
    """Maximum distance bandwidth (similar to a mutation width; the ``bw_max``
    parameter in pygmo).

    It must be larger than ``min_distance_bandwidth``. The default value is
    taken from pygmo.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if self.population_size is not None:
            warnings.warn("The population size has no effect on IHS' performance.")

        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=1
        )

        algo_specific_options = {
            "gen": self.stopping_maxiter,
            "phmcr": self.choose_from_memory_probability,
            "ppar_min": self.min_pitch_adjustment_rate,
            "ppar_max": self.max_pitch_adjustment_rate,
            "bw_min": self.min_distance_bandwidth,
            "bw_max": self.max_distance_bandwidth,
        }
        res = _minimize_pygmo(
            problem=problem,
            x0=x0,
            method="ihs",
            specific_options=algo_specific_options,
            population_size=population_size,
            n_cores=1,
            seed=self.seed,
            discard_start_params=self.discard_start_params,
        )
        return res


@mark.minimizer(
    name="pygmo_de1220",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_PYGMO_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class PygmoDe1220(Algorithm):
    r"""Minimize a scalar function using pygmo's self-adaptive differential
    evolution.

    de1220, also known as pDE, is pygmo's own flavor of self-adaptive
    differential evolution. In addition to self-adapting the crossover
    probability :math:`CR` and the weight coefficient :math:`F` with either the
    jDE (:cite:`Brest2006`) or the iDE (:cite:`Elsayed2011`) mechanism, each
    individual also carries and adapts the mutation variant that is used to
    create new candidates. See the `pagmo documentation on de1220
    <https://esa.github.io/pagmo2/docs/cpp/algorithms/de1220.html>`_ for
    details.

    Like the other differential evolution variants, it is a global,
    derivative-free, population based optimizer for multimodal problems. It is
    a good choice if you do not want to tune the mutation variant and the
    control parameters of differential evolution by hand.

    The algorithm requires finite bounds on all parameters.

    """

    population_size: int | None = None
    """Size of the population. If None, it is set to ``10 * (n_params + 1)``, but
    at least 64."""

    seed: int | None = None
    """Seed used by the internal random number generator."""

    discard_start_params: bool = False
    """If True, the start parameters are not guaranteed to be part of the initial
    population.

    This saves one criterion function evaluation that cannot be done in parallel
    with other evaluations. Default False.

    """

    jde: bool = True
    r"""Whether to use the jDE self-adaptation variant (:cite:`Brest2006`) to
    control the :math:`F` and :math:`CR` parameters.

    If True, jDE is used, otherwise iDE (:cite:`Elsayed2011`).

    """

    stopping_maxiter: PositiveInt = STOPPING_MAX_ITERATIONS_GENETIC
    """Number of generations to evolve (the ``gen`` parameter in pygmo)."""

    allowed_variants: List[str] | None = None
    """Mutation variants the self-adaptation is allowed to choose from.

    Each variant can be given as a string or as the pygmo integer code (in
    parentheses below). The first ten variants are the classical mutation
    variants introduced in the original DE algorithm; the remaining ones are
    considered in the work by :cite:`Elsayed2011`. If None, the default is
    ["rand/1/exp", "rand-to-best/1/exp", "rand/1/bin", "rand/2/bin",
    "best/3/exp", "best/3/bin", "rand-to-current/2/exp",
    "rand-to-current/2/bin"]. The following variants are available:

    - "best/1/exp" (1)
    - "rand/1/exp" (2)
    - "rand-to-best/1/exp" (3)
    - "best/2/exp" (4)
    - "rand/2/exp" (5)
    - "best/1/bin" (6)
    - "rand/1/bin" (7)
    - "rand-to-best/1/bin" (8)
    - "best/2/bin" (9)
    - "rand/2/bin" (10)
    - "rand/3/exp" (11)
    - "rand/3/bin" (12)
    - "best/3/exp" (13)
    - "best/3/bin" (14)
    - "rand-to-current/2/exp" (15)
    - "rand-to-current/2/bin" (16)
    - "rand-to-best-and-current/2/exp" (17)
    - "rand-to-best-and-current/2/bin" (18)

    """

    keep_adapted_params: bool = False
    r"""When True, the adapted parameters :math:`CR` and :math:`F` and the
    mutation variant are not reset between successive calls to the evolve
    method (the ``memory`` parameter in pygmo).

    Default is False.

    """

    ftol: NonNegativeFloat = 1e-6
    """Stop when the absolute difference between the best and the worst objective
    value in the population is smaller than this value.

    The default value is taken from pygmo.

    """

    xtol: NonNegativeFloat = 1e-6
    """Stop when the sum of absolute differences between the best and the worst
    parameter vector in the population is smaller than this value.

    The default value is taken from pygmo.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
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
        if self.allowed_variants is None:
            allowed_variant_codes = [2, 3, 7, 10, 13, 14, 15, 16]
        else:
            allowed_variant_codes = [
                _convert_str_to_int(variant_str_to_int, variant)
                for variant in self.allowed_variants
            ]

        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=64
        )

        algo_specific_options = {
            "gen": self.stopping_maxiter,
            "variant_adptv": 1 if self.jde else 2,
            "ftol": self.ftol,
            "xtol": self.xtol,
            "memory": self.keep_adapted_params,
            "allowed_variants": allowed_variant_codes,
        }

        res = _minimize_pygmo(
            problem=problem,
            x0=x0,
            method="de1220",
            specific_options=algo_specific_options,
            population_size=population_size,
            n_cores=1,
            seed=self.seed,
            discard_start_params=self.discard_start_params,
        )
        return res


# ====================================================================================


def _minimize_pygmo(
    problem: InternalOptimizationProblem,
    x0: NDArray[np.float64],
    method: str,
    specific_options: dict[str, Any],
    population_size: PositiveInt,
    n_cores: int,
    seed: int | None,
    discard_start_params: bool,
) -> InternalOptimizeResult:
    if not IS_PYGMO_INSTALLED:
        raise NotInstalledError(
            f"The {method} algorithm requires the pygmo package to be installed. "
            "You can install it with 'conda install -c conda-forge pygmo'. Visit "
            "https://esa.github.io/pygmo2/install.html for more detailed installation "
            "instructions."
        )

    bounds = problem.bounds
    if bounds is None or bounds.lower is None or bounds.upper is None:
        raise ValueError(f"{method} requires finitel bounds for all parameters.")
    elif not np.isfinite(bounds.lower).all() or not np.isfinite(bounds.upper).all():
        raise ValueError(f"{method} requires finite bounds for all parameters.")

    pygmo_problem = _create_pygmo_problem(problem, len(x0), n_cores)
    algo = _create_algorithm(method, specific_options, n_cores)
    pop = _create_population(
        problem=pygmo_problem,
        population_size=population_size,
        x=x0,
        seed=seed,
        discard_start_params=discard_start_params,
    )
    evolved = algo.evolve(pop)
    result = _process_pygmo_result(evolved)
    return result


def _create_pygmo_problem(
    problem: InternalOptimizationProblem, dim: int, n_cores: int
) -> pg.problem:
    import pygmo as pg

    class Problem:
        def fitness(self, x):
            return [problem.fun(x)]

        def get_bounds(self):
            return (problem.bounds.lower, problem.bounds.upper)

        def gradient(self, dv):  # noqa: ARG002
            raise ValueError("No pygmo optimizer should use a gradient.")

        def batch_fitness(self, dvs):
            x_list = list(dvs.reshape(-1, dim))
            eval_list = problem.batch_fun(x_list, n_cores=n_cores)
            evals = np.array(eval_list)
            return evals

    pygmo_problem = pg.problem(Problem())
    return pygmo_problem


def _create_algorithm(
    method: str, algo_options: dict[str, Any], n_cores: int
) -> pg.algorithm:
    """Create a pygmo algorithm."""
    import pygmo as pg

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


def _create_population(
    problem: InternalOptimizationProblem,
    population_size: int,
    x: NDArray[np.float64],
    seed: int | None,
    discard_start_params: bool,
) -> pg.population:
    import pygmo as pg

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


def _process_pygmo_result(evolved: pg.population) -> InternalOptimizeResult:
    result = InternalOptimizeResult(
        x=evolved.champion_x,
        fun=evolved.champion_f[0],
        success=True,
        message="Number of generations reached.",
        n_fun_evals=evolved.problem.get_fevals(),
        n_jac_evals=evolved.problem.get_gevals(),
    )

    return result


def _convert_str_to_int(str_to_int, value):
    if value in str_to_int:
        out = str_to_int[value]
    elif value not in str_to_int.values():
        raise ValueError(
            f"You specified {value} as value. "
            f"It must be one of {', '.join(str_to_int.keys())}"
        )
    else:
        out = value
    return out
