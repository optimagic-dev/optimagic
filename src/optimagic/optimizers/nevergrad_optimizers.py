"""Implement nevergrad optimizers."""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from optimagic import mark
from optimagic.config import IS_NEVERGRAD_INSTALLED
from optimagic.exceptions import NotInstalledError
from optimagic.optimization.algo_options import STOPPING_MAXFUN_GLOBAL
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.internal_optimization_problem import (
    InternalOptimizationProblem,
)
from optimagic.typing import AggregationLevel, NonNegativeInt, PositiveInt

if IS_NEVERGRAD_INSTALLED:
    import nevergrad as ng


@mark.minimizer(
    name="nevergrad_pso",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NEVERGRAD_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=True,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class NevergradPSO(Algorithm):
    transform: Literal["arctan", "gaussian", "identity"] = "arctan"
    population_size: int | None = None
    n_cores: int = 1
    seed: int | None = None
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL * 2  # imprecise algorithm
    inertia: float = 0.5 / math.log(2.0)
    cognitive: float = 0.5 + math.log(2.0)
    social: float = 0.5 + math.log(2.0)
    quasi_opp_init: bool = False
    speed_quasi_opp_init: bool = False
    special_speed_quasi_opp_init: bool = False

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_NEVERGRAD_INSTALLED:
            raise NotInstalledError(
                "The nevergrad_pso optimizer requires the 'nevergrad' package to be "
                "installed. You can install it with `pip install nevergrad`. "
                "Visit https://facebookresearch.github.io/nevergrad/getting_started.html"
                " for more detailed installation instructions."
            )

        instrum = ng.p.Instrumentation(
            ng.p.Array(
                shape=x0.shape, lower=problem.bounds.lower, upper=problem.bounds.upper
            )
        )

        if self.seed is not None:
            instrum.random_state.seed(self.seed)

        optimizer = ng.optimizers.ConfPSO(
            transform=self.transform,
            popsize=self.population_size,
            omega=self.inertia,
            phip=self.cognitive,
            phig=self.social,
            qo=self.quasi_opp_init,
            sqo=self.speed_quasi_opp_init,
            so=self.special_speed_quasi_opp_init,
        )(
            parametrization=instrum,
            budget=self.stopping_maxfun,
            num_workers=self.n_cores,
        )

        while optimizer.num_ask < self.stopping_maxfun:
            x_list = [
                optimizer.ask()
                for _ in range(
                    min(self.n_cores, self.stopping_maxfun - optimizer.num_ask)
                )
            ]
            losses = problem.batch_fun(
                [x.value[0][0] for x in x_list], n_cores=self.n_cores
            )
            for x, loss in zip(x_list, losses, strict=True):
                optimizer.tell(x, loss)

        recommendation = optimizer.provide_recommendation()

        result = InternalOptimizeResult(
            x=recommendation.value[0][0],
            fun=recommendation.loss,
            success=True,
            n_fun_evals=optimizer.num_ask,
            n_jac_evals=0,
            n_hess_evals=0,
        )

        return result


@mark.minimizer(
    name="nevergrad_oneplusone",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NEVERGRAD_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=True,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class NevergradOnePlusOne(Algorithm):
    """One Plus One Evolutionary algorithm from Nevergrad.

    Args:
        noise_handling: Method for handling the noise, can be
            - "random": A random point is reevaluated regularly using the one-fifth
              adaptation rule.
            - "optimistic": The best optimistic point is reevaluated regularly,
              embracing optimism in the face of uncertainty.
            - A float coefficient can be provided to tune the regularity of these
              reevaluations (default is 0.05). Eg: with 0.05, each evaluation has a 5%
              chance (i.e., 1 in 20) of being repeated (i.e., the same candidate
              solution is reevaluated to better estimate its performance).
              (Default: `None`).

        n_cores: Number of cores to use.

        stopping.maxfun: Maximum number of function evaluations.

        mutation: Type of mutation to apply. Available options are (Default:
            `"gaussian"`).
            - "gaussian": Standard mutation by adding a Gaussian random variable
              (with progressive widening) to the best pessimistic point.
            - "cauchy": Same as Gaussian but using a Cauchy distribution.
            - "discrete": Mutates a randomly drawn variable (mutation occurs with
              probability 1/d in d dimensions, hence ~1 variable per mutation).
            - "discreteBSO": Follows brainstorm optimization by gradually decreasing
              mutation rate from 1 to 1/d.
            - "fastga": Fast Genetic Algorithm mutations from the current best.
            - "doublefastga": Double-FastGA mutations from the current best
              :cite:`doerr2017`.
            - "rls": Randomized Local Search — mutates one and only one variable.
            - "portfolio": Random number of mutated bits, known as uniform mixing
              :cite:`dang2016`.
            - "lengler": Mutation rate is a function of dimension and iteration index.
            - "lengler{2|3|half|fourth}": Variants of the Lengler mutation rate
              adaptation.

        sparse: Whether to apply random mutations that set variables to zero.
            Default is `False`.

        smoother: Whether to suggest smooth mutations. Default is `False`.

        annealing: Annealing schedule to apply to mutation amplitude or
            temperature-based control. Options are:
            - "none": No annealing is applied.
            - "Exp0.9": Exponential decay with rate 0.9.
            - "Exp0.99": Exponential decay with rate 0.99.
            - "Exp0.9Auto": Exponential decay with rate 0.9, auto-scaled based on
              problem horizon.
            - "Lin100.0": Linear decay from 1 to 0 over 100 iterations.
            - "Lin1.0": Linear decay from 1 to 0 over 1 iteration.
            - "LinAuto": Linearly decaying annealing automatically scaled to the
              problem horizon. Default is `"none"`.

        super_radii: Whether to apply extended radii beyond standard bounds for
            candidate generation, enabling broader exploration. Default is `False`.

        roulette_size: Size of the roulette wheel used for selection in the
            evolutionary process. Affects the sampling diversity from past
            candidates. (Default: `64`)

        antismooth: Degree of anti-smoothing applied to prevent premature
            convergence in smooth landscapes. This alters the landscape by
            penalizing overly smooth improvements. (Default: `4`)

        crossover: Whether to include a genetic crossover step every other
            iteration. Default is `False`.

        crossover_type: Method used for genetic crossover between individuals in
            the population. Available options (Default: `"none"`):
            - "none": No crossover is applied.
            - "rand": Randomized selection of crossover point.
            - "max": Crossover at the point with maximum fitness gain.
            - "min": Crossover at the point with minimum fitness gain.
            - "onepoint": One-point crossover, splitting the genome at a single
              random point.
            - "twopoint": Two-point crossover, splitting the genome at two points
              and exchanging the middle section.

        tabu_length: Length of the tabu list used to prevent revisiting recently
            evaluated candidates in local search strategies. Helps in escaping
            local minima. (Default: `1000`)

        rotation: Whether to apply rotational transformations to the search
            space, promoting invariance to axis-aligned structures and enhancing
            search performance in rotated coordinate systems. (Default: `False`)

    """

    noise_handling: (
        Literal["random", "optimistic"]
        | tuple[Literal["random", "optimistic"], float]
        | None
    ) = None
    mutation: Literal[
        "gaussian",
        "cauchy",
        "discrete",
        "fastga",
        "rls",
        "doublefastga",
        "adaptive",
        "coordinatewise_adaptive",
        "portfolio",
        "discreteBSO",
        "lengler",
        "lengler2",
        "lengler3",
        "lenglerhalf",
        "lenglerfourth",
        "doerr",
        "lognormal",
        "xlognormal",
        "xsmalllognormal",
        "tinylognormal",
        "smalllognormal",
        "biglognormal",
        "hugelognormal",
    ] = "gaussian"
    annealing: (
        Literal[
            "none", "Exp0.9", "Exp0.99", "Exp0.9Auto", "Lin100.0", "Lin1.0", "LinAuto"
        ]
        | None
    ) = None
    sparse: bool = False
    super_radii: bool = False
    smoother: bool = False
    roulette_size: PositiveInt = 64
    antismooth: NonNegativeInt = 4
    crossover: bool = False
    crossover_type: (
        Literal["none", "rand", "max", "min", "onepoint", "twopoint"] | None
    ) = None
    tabu_length: NonNegativeInt = 1000
    rotation: bool = False
    seed: int | None = None
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL
    n_cores: PositiveInt = 1

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_NEVERGRAD_INSTALLED:
            raise NotInstalledError(
                "The nevergrad_oneplusone optimizer requires the 'nevergrad' package "
                "to be installed. You can install it with `pip install nevergrad`. "
                "Visit https://facebookresearch.github.io/nevergrad/getting_started.html"
                " for more detailed installation instructions."
            )

        instrum = ng.p.Array(
            init=x0, lower=problem.bounds.lower, upper=problem.bounds.upper
        )

        instrum.specify_tabu_length(tabu_length=self.tabu_length)
        instrum = ng.p.Instrumentation(instrum)

        if self.annealing is None:
            self.annealing = "none"
        if self.crossover_type is None:
            self.crossover_type = "none"
        if self.seed is not None:
            instrum.random_state.seed(self.seed)

        optimizer = ng.optimizers.ParametrizedOnePlusOne(
            noise_handling=self.noise_handling,
            mutation=self.mutation,
            crossover=self.crossover,
            rotation=self.rotation,
            annealing=self.annealing,
            sparse=self.sparse,
            smoother=self.smoother,
            super_radii=self.super_radii,
            roulette_size=self.roulette_size,
            antismooth=self.antismooth,
            crossover_type=self.crossover_type,
        )(
            parametrization=instrum,
            budget=self.stopping_maxfun,
            num_workers=self.n_cores,
        )

        while optimizer.num_ask < self.stopping_maxfun:
            x_list = [
                optimizer.ask()
                for _ in range(
                    min(self.n_cores, self.stopping_maxfun - optimizer.num_ask)
                )
            ]
            losses = problem.batch_fun(
                [x.value[0][0] for x in x_list], n_cores=self.n_cores
            )
            for x, loss in zip(x_list, losses, strict=True):
                optimizer.tell(x, loss)

        recommendation = optimizer.provide_recommendation()

        result = InternalOptimizeResult(
            x=recommendation.value[0][0],
            fun=recommendation.loss,
            success=True,
            n_fun_evals=optimizer.num_ask,
            n_jac_evals=0,
            n_hess_evals=0,
        )

        return result
