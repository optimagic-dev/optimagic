"""Implement optimizers from the nevergrad package."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from optimagic import mark
from optimagic.config import IS_BAYESOPTIM_INSTALLED, IS_NEVERGRAD_INSTALLED
from optimagic.exceptions import NotInstalledError
from optimagic.optimization.algo_options import (
    CONVERGENCE_FTOL_ABS,
    CONVERGENCE_FTOL_REL,
    CONVERGENCE_XTOL_ABS,
    STOPPING_MAXFUN_GLOBAL,
    STOPPING_MAXITER,
)
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.internal_optimization_problem import (
    InternalOptimizationProblem,
)
from optimagic.typing import (
    AggregationLevel,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)

if TYPE_CHECKING:
    from nevergrad.optimization.base import ConfiguredOptimizer


NEVERGRAD_NOT_INSTALLED_ERROR = (
    "This optimizer requires the 'nevergrad' package to be installed. "
    "You can install it with `pip install nevergrad`. "
    "Visit https://facebookresearch.github.io/nevergrad/getting_started.html "
    "for more detailed installation instructions."
)


@mark.minimizer(
    name="nevergrad_pso",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NEVERGRAD_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=False,
    supports_parallelism=True,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class NevergradPSO(Algorithm):
    """Minimize a scalar function using the Particle Swarm Optimization algorithm.

    The Particle Swarm Optimization algorithm was originally proposed by
    :cite:`Kennedy1995`.The implementation in Nevergrad is based on
    :cite:`Zambrano2013`.

    PSO solves an optimization problem by evolving a swarm of particles
    (candidate solutions) across the search space. Each particle adjusts its position
    based on its own experience (cognitive component) and the experiences
    of its neighbors or the swarm (social component), using velocity updates. The
    algorithm iteratively guides the swarm toward promising regions of the search
    space.

    """

    transform: Literal["arctan", "gaussian", "identity"] = "arctan"
    """The transform used to map from PSO optimization space to real space."""

    population_size: int | None = None
    """The number of particles in the swarm."""

    n_cores: int = 1
    """The number of CPU cores to use for parallel computation."""

    seed: int | None = None
    """Random seed for reproducibility."""

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL
    """Maximum number of function evaluations."""

    inertia: float = 0.5 / math.log(2.0)
    r"""Inertia weight ω.

    Controls the influence of a particle's previous velocity. Must be less than 1 to
    avoid divergence.

    """

    cognitive: float = 0.5 + math.log(2.0)
    r"""Cognitive coefficient :math:`\phi_p`.

    Controls the influence of a particle's own best known position. Typical values: 1.0
    to 3.0.

    """

    social: float = 0.5 + math.log(2.0)
    r"""Social coefficient.

    Denoted by :math:`\phi_g`. Controls the influence of the swarm's best known
    position. Typical values: 1.0 to 3.0.

    """

    quasi_opp_init: bool = False
    """Whether to use quasi-opposition initialization.

    Default is False.

    """

    speed_quasi_opp_init: bool = False
    """Whether to apply quasi-opposition initialization to speed.

    Default is False.

    """

    special_speed_quasi_opp_init: bool = False
    """Whether to use special quasi-opposition initialization for speed.

    Default is False.

    """

    sigma: float | None = None
    """Standard deviation for sampling initial population from N(0, σ²) in case bounds
    are not provided."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_NEVERGRAD_INSTALLED:
            raise NotInstalledError(NEVERGRAD_NOT_INSTALLED_ERROR)

        import nevergrad as ng

        configured_optimizer = ng.optimizers.ConfPSO(
            transform=self.transform,
            popsize=self.population_size,
            omega=self.inertia,
            phip=self.cognitive,
            phig=self.social,
            qo=self.quasi_opp_init,
            sqo=self.speed_quasi_opp_init,
            so=self.special_speed_quasi_opp_init,
        )

        res = _nevergrad_internal(
            problem=problem,
            x0=x0,
            configured_optimizer=configured_optimizer,
            stopping_maxfun=self.stopping_maxfun,
            n_cores=self.n_cores,
            seed=self.seed,
            sigma=self.sigma,
            nonlinear_constraints=problem.nonlinear_constraints,
        )

        return res


@mark.minimizer(
    name="nevergrad_cmaes",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NEVERGRAD_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=False,
    supports_parallelism=True,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class NevergradCMAES(Algorithm):
    """Minimize a scalar function using the Covariance Matrix Adaptation Evolution
    Strategy (CMA-ES) algorithm.

    The CMA-ES is a state-of-the-art evolutionary algorithm for difficult non-linear,
    non-convex, black-box optimization problems in continuous domains. It is typically
    applied to unconstrained or bounded problems with dimensionality between 3 and 100.
    CMA-ES adapts a multivariate normal distribution to approximate the objective
    function's shape by estimating a positive-definite covariance matrix, akin to the
    inverse Hessian in convex-quadratic problems, but without requiring derivatives.

    This implementation is a python wrapper over the original code.

    Original paper can be accessed at `cma-es
    <https://cma-es.github.io/>`_.

    """

    scale: NonNegativeFloat = 1.0
    """Scale of the search."""

    elitist: bool = False
    """Whether to switch to elitist mode (also known as (μ,λ)-CMA-ES).

    In elitist mode, the best point in the population is always retained.

    """

    population_size: int | None = None
    """Population size."""

    diagonal: bool = False
    """Use the diagonal version of CMA, which is more efficient for high-dimensional
    problems."""

    high_speed: bool = False
    """Use a metamodel for recommendation to speed up optimization."""

    fast_cmaes: bool = False
    """Use the fast CMA-ES implementation.

    Cannot be used with diagonal=True. Produces equivalent results and is preferable for
    high dimensions or when objective function evaluations are fast.

    """

    random_init: bool = False
    """If True, initialize the optimizer with random parameters."""

    n_cores: PositiveInt = 1
    """Number of cores to use for parallel function evaluation."""

    step_size_adaptive: bool | str = True
    """Whether to adapt the step size.

    Can be a boolean or a string specifying the adaptation strategy.

    """

    CSA_dampfac: PositiveFloat = 1.0
    """Damping factor for step size adaptation."""

    CMA_dampsvec_fade: PositiveFloat = 0.1
    """Damping rate for step size adaptation."""

    CSA_squared: bool = False
    """Whether to use squared step sizes in updates."""

    CMA_on: float = 1.0
    """Learning rate for the covariance matrix update."""

    CMA_rankone: float = 1.0
    """Multiplier for the rank-one update learning rate of the covariance matrix."""

    CMA_rankmu: float = 1.0
    """Multiplier for the rank-mu update learning rate of the covariance matrix."""

    CMA_cmean: float = 1.0
    """Learning rate for the mean update."""

    CMA_diagonal_decoding: float = 0.0
    """Learning rate for the diagonal update."""

    num_parents: int | None = None
    """Number of parents (μ) for recombination."""

    CMA_active: bool = True
    """Whether to use negative updates for the covariance matrix."""

    CMA_mirrormethod: Literal[0, 1, 2] = 2
    """Strategy for mirror sampling.

    0: Unconditional, 1: Selective, 2: Selective
    with delay.

    """

    CMA_const_trace: bool | Literal["arithm", "geom", "aeig", "geig"] = False
    """How to normalize the trace of the covariance matrix.

    False: No normalization,
    True: Normalize to 1. Other options: 'arithm', 'geom', 'aeig', 'geig'.

    """

    CMA_diagonal: int | bool = False
    """Number of iterations to use diagonal covariance matrix before switching to full
    matrix.

    If False, always use full matrix.

    """

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL
    """Maximum number of function evaluations before termination."""

    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    """Maximum number of iterations before termination."""

    stopping_maxtime: PositiveFloat = float("inf")
    """Maximum time in seconds before termination."""

    stopping_cov_mat_cond: NonNegativeFloat = 1e14
    """Maximum condition number of the covariance matrix before termination."""

    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_FTOL_ABS
    """Absolute tolerance on function value changes for convergence."""

    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    """Relative tolerance on function value changes for convergence."""

    convergence_xtol_abs: NonNegativeFloat = CONVERGENCE_XTOL_ABS
    """Absolute tolerance on parameter changes for convergence."""

    convergence_iter_noimprove: PositiveInt | None = None
    """Number of iterations without improvement before termination."""

    invariant_path: bool = False
    """Whether evolution path (pc) should be invariant to transformations."""

    eval_final_mean: bool = True
    """Whether to evaluate the final mean solution."""

    seed: int | None = None
    """Seed used by the internal random number generator for reproducibility."""

    sigma: float | None = None
    """Standard deviation for sampling initial population from N(0, σ²)in case bounds
    are not provided."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_NEVERGRAD_INSTALLED:
            raise NotInstalledError(NEVERGRAD_NOT_INSTALLED_ERROR)

        import nevergrad as ng

        cma_options = {
            "AdaptSigma": self.step_size_adaptive,
            "CSA_dampfac": self.CSA_dampfac,
            "CMA_dampsvec_fade": self.CMA_dampsvec_fade,
            "CSA_squared": self.CSA_squared,
            "CSA_invariant_path": self.invariant_path,
            "CMA_on": self.CMA_on,
            "CMA_rankone": self.CMA_rankone,
            "CMA_rankmu": self.CMA_rankmu,
            "CMA_cmean": self.CMA_cmean,
            "CMA_diagonal_decoding": self.CMA_diagonal_decoding,
            "CMA_mu": self.num_parents,
            "CMA_active": self.CMA_active,
            "CMA_mirrormethod": self.CMA_mirrormethod,
            "CMA_const_trace": self.CMA_const_trace,
            "CMA_diagonal": self.CMA_diagonal,
            "maxfevals": self.stopping_maxfun,
            "maxiter": self.stopping_maxiter,
            "timeout": self.stopping_maxtime,
            "tolconditioncov": self.stopping_cov_mat_cond,
            "tolfun": self.convergence_ftol_abs,
            "tolfunrel": self.convergence_ftol_rel,
            "tolx": self.convergence_xtol_abs,
            "tolstagnation": self.convergence_iter_noimprove,
            "eval_final_mean": self.eval_final_mean,
        }

        configured_optimizer = ng.optimizers.ParametrizedCMA(
            scale=self.scale,
            popsize=self.population_size,
            elitist=self.elitist,
            diagonal=self.diagonal,
            high_speed=self.high_speed,
            fcmaes=self.fast_cmaes,
            inopts=cma_options,
        )

        res = _nevergrad_internal(
            problem=problem,
            x0=x0,
            configured_optimizer=configured_optimizer,
            stopping_maxfun=self.stopping_maxfun,
            n_cores=self.n_cores,
            seed=self.seed,
            sigma=self.sigma,
            nonlinear_constraints=problem.nonlinear_constraints,
        )
        return res


@mark.minimizer(
    name="nevergrad_oneplusone",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NEVERGRAD_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=False,
    supports_parallelism=True,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class NevergradOnePlusOne(Algorithm):
    """Minimize a scalar function using the One-Plus-One Evolutionary algorithm.

    The One-Plus-One evolutionary algorithm iterates to find a set of parameters
    that minimizes the loss function. It does this by perturbing, or mutating,
    the parameters from the last iteration (the parent). If the new (child)
    parameters yield a better result, the child becomes the new parent whose
    parameters are perturbed, perhaps more aggressively. If the parent yields a
    better result, it remains the parent and the next perturbation is less
    aggressive.

    Originally proposed by :cite:`Rechenberg1973`. The implementation in
    Nevergrad is based on the one-fifth adaptation rule from :cite:`Schumer1968`.

    """

    noise_handling: (
        Literal["random", "optimistic"]
        | tuple[Literal["random", "optimistic"], float]
        | None
    ) = None
    """Method for handling noise.

    'random' reevaluates a random point, while 'optimistic' reevaluates the best
    optimistic point. A float coefficient can be provided to tune the regularity of
    these reevaluations.

    """

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
    """Type of mutation to apply.

    'gaussian' is the default. Other options include 'cauchy', 'discrete', 'fastga',
    'rls', and 'portfolio'.

    """

    annealing: (
        Literal[
            "none", "Exp0.9", "Exp0.99", "Exp0.9Auto", "Lin100.0", "Lin1.0", "LinAuto"
        ]
        | None
    ) = None
    """Annealing schedule for mutation amplitude.

    Can be 'none', exponential (e.g., 'Exp0.9'), or linear (e.g., 'Lin100.0').

    """

    sparse: bool = False
    """Whether to apply random mutations that set variables to zero."""

    super_radii: bool = False
    """Whether to apply extended radii beyond standard bounds for candidate generation,
    enabling broader exploration."""

    smoother: bool = False
    """Whether to suggest smooth mutations."""

    roulette_size: PositiveInt = 64
    """Size of the roulette wheel used for selection, affecting sampling diversity from
    past candidates."""

    antismooth: NonNegativeInt = 4
    """Degree of anti-smoothing to prevent premature convergence by penalizing overly
    smooth improvements."""

    crossover: bool = False
    """Whether to include a genetic crossover step every other iteration."""

    crossover_type: (
        Literal["none", "rand", "max", "min", "onepoint", "twopoint"] | None
    ) = None
    """Method for genetic crossover.

    Options include 'rand', 'onepoint', and 'twopoint'.

    """

    tabu_length: NonNegativeInt = 1000
    """Length of the tabu list to prevent revisiting recent candidates and help escape
    local minima."""

    rotation: bool = False
    """Whether to apply rotational transformations to the search space to enhance search
    performance."""

    seed: int | None = None
    """Seed for the random number generator for reproducibility."""

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL
    """Maximum number of function evaluations."""

    n_cores: PositiveInt = 1
    """Number of cores to use for parallel computation."""

    sigma: float | None = None
    """Standard deviation for sampling initial population from N(0, σ²)if bounds are not
    provided."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_NEVERGRAD_INSTALLED:
            raise NotInstalledError(NEVERGRAD_NOT_INSTALLED_ERROR)

        import nevergrad as ng

        configured_optimizer = ng.optimizers.ParametrizedOnePlusOne(
            noise_handling=self.noise_handling,
            mutation=self.mutation,
            crossover=self.crossover,
            rotation=self.rotation,
            annealing=self.annealing or "none",
            sparse=self.sparse,
            smoother=self.smoother,
            super_radii=self.super_radii,
            roulette_size=self.roulette_size,
            antismooth=self.antismooth,
            crossover_type=self.crossover_type or "none",
        )

        res = _nevergrad_internal(
            problem=problem,
            x0=x0,
            configured_optimizer=configured_optimizer,
            stopping_maxfun=self.stopping_maxfun,
            n_cores=self.n_cores,
            seed=self.seed,
            sigma=self.sigma,
            nonlinear_constraints=problem.nonlinear_constraints,
        )

        return res


@mark.minimizer(
    name="nevergrad_de",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NEVERGRAD_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=False,
    supports_parallelism=True,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class NevergradDifferentialEvolution(Algorithm):
    """Minimize a scalar function using the Differential Evolution optimizer.

    Differential Evolution is typically used for continuous optimization. It uses
    differences between points in the population for performing mutations in fruitful
    directions. It is a kind of covariance adaptation without any explicit covariance,
    making it very fast in high dimensions.

    """

    initialization: Literal["parametrization", "LHS", "QR", "QO", "SO"] = (
        "parametrization"
    )
    """Algorithm for initialization.

    'LHS' is Latin Hypercube Sampling, 'QR' is Quasi-Random.

    """

    scale: float | str = 1.0
    """Scale of random component of updates."""

    recommendation: Literal["pessimistic", "optimistic", "mean", "noisy"] = (
        "pessimistic"
    )
    """Criterion for selecting the best point to recommend."""

    crossover: (
        float
        | Literal[
            "dimension",
            "random",
            "onepoint",
            "twopoints",
            "rotated_twopoints",
            "parametrization",
        ]
    ) = 0.5
    """Crossover rate or strategy.

    Can be a float, 'dimension' (1/dim), 'random', 'onepoint', or 'twopoints'.

    """

    F1: PositiveFloat = 0.8
    """Differential weight #1 (scaling factor)."""

    F2: PositiveFloat = 0.8
    """Differential weight #2 (scaling factor)."""

    population_size: int | Literal["standard", "dimension", "large"] = "standard"
    """Population size.

    Can be an integer or a string like 'standard', 'dimension', or 'large' to set it
    automatically.

    """

    high_speed: bool = False
    """If True, uses a metamodel for recommendations to speed up optimization."""

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL
    """Maximum number of function evaluations before termination."""

    n_cores: PositiveInt = 1
    """Number of cores to use for parallel function evaluation."""

    seed: int | None = None
    """Seed for the random number generator for reproducibility."""

    sigma: float | None = None
    """Standard deviation for sampling initial population from N(0, σ²)if bounds are not
    provided."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_NEVERGRAD_INSTALLED:
            raise NotInstalledError(NEVERGRAD_NOT_INSTALLED_ERROR)

        import nevergrad as ng

        configured_optimizer = ng.optimizers.DifferentialEvolution(
            initialization=self.initialization,
            scale=self.scale,
            recommendation=self.recommendation,
            crossover=self.crossover,
            F1=self.F1,
            F2=self.F2,
            popsize=self.population_size,
            high_speed=self.high_speed,
        )

        res = _nevergrad_internal(
            problem=problem,
            x0=x0,
            configured_optimizer=configured_optimizer,
            stopping_maxfun=self.stopping_maxfun,
            n_cores=self.n_cores,
            seed=self.seed,
            sigma=self.sigma,
            nonlinear_constraints=problem.nonlinear_constraints,
        )
        return res


@mark.minimizer(
    name="nevergrad_bo",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NEVERGRAD_INSTALLED and IS_BAYESOPTIM_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=False,
    supports_parallelism=True,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class NevergradBayesOptim(Algorithm):
    """Minimize a scalar function using the Bayesian Optimization (BO) algorithm.

    This wrapper uses the BO and PCA-BO algorithms from the `bayes_optim` package
    :cite:`bayesoptimimpl`. PCA-BO (Principal Component Analysis for Bayesian
    Optimization) is a dimensionality reduction technique for black-box
    optimization. It applies PCA to the input space before performing Bayesian
    optimization, improving efficiency in high dimensions by focusing on
    directions of greatest variance.

    """

    init_budget: int | None = None
    """Number of initialization algorithm steps."""

    pca: bool = False
    """Whether to use the PCA transformation, defining PCA-BO rather than standard
    BO."""

    n_components: NonNegativeFloat = 0.95
    """Number of principal axes, representing the percentage of explained variance
    (e.g., 0.95 means 95% variance retained)."""

    prop_doe_factor: NonNegativeFloat | None = 1
    """Percentage of the initial budget used for Design of Experiments (DoE)."""

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL
    """Maximum number of function evaluations before termination."""

    n_cores: PositiveInt = 1
    """Number of cores to use for parallel function evaluation."""

    seed: int | None = None
    """Seed for the random number generator for reproducibility."""

    sigma: int | None = None
    """Standard deviation for sampling initial population from N(0, σ²)in case bounds
    are not provided."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_NEVERGRAD_INSTALLED:
            raise NotInstalledError(NEVERGRAD_NOT_INSTALLED_ERROR)

        import nevergrad as ng

        configured_optimizer = ng.optimizers.BayesOptim(
            init_budget=self.init_budget,
            pca=self.pca,
            n_components=self.n_components,
            prop_doe_factor=self.prop_doe_factor,
        )

        res = _nevergrad_internal(
            problem=problem,
            x0=x0,
            configured_optimizer=configured_optimizer,
            stopping_maxfun=self.stopping_maxfun,
            n_cores=self.n_cores,
            seed=self.seed,
            sigma=self.sigma,
            nonlinear_constraints=problem.nonlinear_constraints,
        )
        return res


@mark.minimizer(
    name="nevergrad_emna",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NEVERGRAD_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=False,
    supports_parallelism=True,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class NevergradEMNA(Algorithm):
    """Minimize a scalar function using the Estimation of Multivariate Normal Algorithm.

    EMNA is a distribution-based evolutionary algorithm that models the search
    space using a multivariate Gaussian. It learns the full covariance matrix,
    resulting in a cubic time complexity with respect to each sampling. It is
    efficient in parallel settings but other methods should be considered first.
    See :cite:`emnaimpl`.

    """

    isotropic: bool = True
    """If True, uses an isotropic (identity covariance) Gaussian.

    If False, uses a separable (diagonal covariance) Gaussian.

    """

    noise_handling: bool = True
    """If True, returns the best individual found.

    If False (recommended for noisy problems), returns the average of the final
    population.

    """

    population_size_adaptation: bool = False
    """If True, the population size is adjusted automatically based on the optimization
    landscape and noise level."""

    initial_popsize: int | None = None
    """Initial population size.

    Defaults to 4 times the problem dimension.

    """

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL
    """Maximum number of function evaluations before termination."""

    n_cores: PositiveInt = 1
    """Number of cores to use for parallel function evaluation."""

    seed: int | None = None
    """Seed for the random number generator for reproducibility."""

    sigma: float | None = None
    """Standard deviation for sampling initial population from N(0, σ²)in case bounds
    are not provided."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_NEVERGRAD_INSTALLED:
            raise NotInstalledError(NEVERGRAD_NOT_INSTALLED_ERROR)

        import nevergrad as ng

        configured_optimizer = ng.optimizers.EMNA(
            isotropic=self.isotropic,
            naive=self.noise_handling,
            population_size_adaptation=self.population_size_adaptation,
            initial_popsize=self.initial_popsize,
        )

        res = _nevergrad_internal(
            problem=problem,
            x0=x0,
            configured_optimizer=configured_optimizer,
            stopping_maxfun=self.stopping_maxfun,
            n_cores=self.n_cores,
            seed=self.seed,
            sigma=self.sigma,
            nonlinear_constraints=problem.nonlinear_constraints,
        )
        return res


@mark.minimizer(
    name="nevergrad_cga",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NEVERGRAD_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=False,
    supports_parallelism=True,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class NevergradCGA(Algorithm):
    """Minimize a scalar function using the Compact Genetic Algorithm.

    The Compact Genetic Algorithm (cGA) is a memory-efficient genetic algorithm
    that represents the population as a probability vector over gene values. It
    simulates the behavior of a simple GA with uniform crossover by updating
    probabilities instead of maintaining an explicit population. See :cite:`cgaimpl`.

    """

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL
    """Maximum number of function evaluations before termination."""

    n_cores: PositiveInt = 1
    """Number of cores to use for parallel function evaluation."""

    seed: int | None = None
    """Seed for the random number generator for reproducibility."""

    sigma: float | None = None
    """Standard deviation for sampling initial population from N(0, σ²)in case bounds
    are not provided."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_NEVERGRAD_INSTALLED:
            raise NotInstalledError(NEVERGRAD_NOT_INSTALLED_ERROR)

        import nevergrad as ng

        configured_optimizer = ng.optimizers.cGA

        res = _nevergrad_internal(
            problem=problem,
            x0=x0,
            configured_optimizer=configured_optimizer,
            stopping_maxfun=self.stopping_maxfun,
            n_cores=self.n_cores,
            seed=self.seed,
            sigma=self.sigma,
            nonlinear_constraints=problem.nonlinear_constraints,
        )
        return res


@mark.minimizer(
    name="nevergrad_eda",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NEVERGRAD_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=False,
    supports_parallelism=True,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class NevergradEDA(Algorithm):
    """Minimize a scalar function using the Estimation of Distribution Algorithm.

    Estimation of Distribution Algorithms (EDAs) optimize by building and sampling
    a probabilistic model of promising solutions. Instead of using traditional
    variation operators like crossover or mutation, EDAs update a distribution
    based on selected individuals and sample new candidates from it.
    Refer to :cite:`edaimpl`.

    """

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL
    """Maximum number of function evaluations before termination."""

    n_cores: PositiveInt = 1
    """Number of cores to use for parallel function evaluation."""

    seed: int | None = None
    """Seed for the random number generator for reproducibility."""

    sigma: float | None = None
    """Standard deviation for sampling initial population from N(0, σ²)in case bounds
    are not provided."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_NEVERGRAD_INSTALLED:
            raise NotInstalledError(NEVERGRAD_NOT_INSTALLED_ERROR)

        import nevergrad as ng

        configured_optimizer = ng.optimizers.EDA

        res = _nevergrad_internal(
            problem=problem,
            x0=x0,
            configured_optimizer=configured_optimizer,
            stopping_maxfun=self.stopping_maxfun,
            n_cores=self.n_cores,
            seed=self.seed,
            sigma=self.sigma,
            nonlinear_constraints=problem.nonlinear_constraints,
        )
        return res


@mark.minimizer(
    name="nevergrad_tbpsa",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NEVERGRAD_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=False,
    supports_parallelism=True,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class NevergradTBPSA(Algorithm):
    """Minimize a scalar function using the Test-based Population Size Adaptation
    algorithm.

    TBPSA adapts population size based on fitness trend detection using linear
    regression. If no significant improvement is found (via hypothesis testing),
    the population size is increased to improve robustness, making it effective
    for noisy optimization problems. For more details, refer to :cite:`tbpsaimpl`.

    """

    noise_handling: bool = True
    """If True, returns the best individual.

    If False (recommended for noisy problems), returns the average of the final
    population to reduce noise.

    """

    initial_popsize: int | None = None
    """Initial population size.

    If not specified, defaults to 4 times the problem dimension.

    """

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL
    """Maximum number of function evaluations before termination."""

    n_cores: PositiveInt = 1
    """Number of cores to use for parallel function evaluation."""

    seed: int | None = None
    """Seed for the random number generator for reproducibility."""

    sigma: float | None = None
    """Standard deviation for sampling initial population from N(0, σ²)in case bounds
    are not provided."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_NEVERGRAD_INSTALLED:
            raise NotInstalledError(NEVERGRAD_NOT_INSTALLED_ERROR)

        import nevergrad as ng

        configured_optimizer = ng.optimizers.ParametrizedTBPSA(
            naive=self.noise_handling,
            initial_popsize=self.initial_popsize,
        )

        res = _nevergrad_internal(
            problem=problem,
            x0=x0,
            configured_optimizer=configured_optimizer,
            stopping_maxfun=self.stopping_maxfun,
            n_cores=self.n_cores,
            seed=self.seed,
            sigma=self.sigma,
            nonlinear_constraints=problem.nonlinear_constraints,
        )
        return res


@mark.minimizer(
    name="nevergrad_randomsearch",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NEVERGRAD_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=False,
    supports_parallelism=True,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class NevergradRandomSearch(Algorithm):
    """Minimize a scalar function using the Random Search algorithm.

    This is a one-shot optimization method that provides random suggestions and serves
    as a simple baseline for other optimizers.

    """

    middle_point: bool = False
    """Enforces that the first suggested point is the zero vector."""

    opposition_mode: Literal["opposite", "quasi"] | None = None
    """Symmetrizes exploration with respect to the center.

    'opposite' enables full symmetry, while 'quasi' applies randomized symmetry.

    """

    sampler: Literal["parametrization", "gaussian", "cauchy"] = "parametrization"
    """The probability distribution for sampling points.

    'gaussian' and 'cauchy' are available alternatives.

    """

    scale: PositiveFloat | Literal["random", "auto", "autotune"] = "auto"
    """Scalar used to multiply suggested point values.

    Can be a float or a string for auto-scaling ('random', 'auto', 'autotune').

    """

    recommendation_rule: Literal[
        "average_of_best", "pessimistic", "average_of_exp_best"
    ] = "pessimistic"
    """Specifies how the final recommendation is chosen, e.g., 'pessimistic' (default)
    or 'average_of_best'."""

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL
    """Maximum number of function evaluations before termination."""

    n_cores: PositiveInt = 1
    """Number of cores to use for parallel function evaluation."""

    sigma: float | None = None
    """Standard deviation for sampling initial population from N(0, σ²)in case bounds
    are not provided."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_NEVERGRAD_INSTALLED:
            raise NotInstalledError(NEVERGRAD_NOT_INSTALLED_ERROR)

        import nevergrad as ng

        configured_optimizer = ng.optimizers.RandomSearchMaker(
            stupid=False,
            middle_point=self.middle_point,
            opposition_mode=self.opposition_mode,
            sampler=self.sampler,
            scale=self.scale,
            recommendation_rule=self.recommendation_rule,
        )

        res = _nevergrad_internal(
            problem=problem,
            x0=x0,
            configured_optimizer=configured_optimizer,
            stopping_maxfun=self.stopping_maxfun,
            n_cores=self.n_cores,
            seed=None,
            sigma=self.sigma,
            nonlinear_constraints=problem.nonlinear_constraints,
        )
        return res


@mark.minimizer(
    name="nevergrad_samplingsearch",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NEVERGRAD_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=False,
    supports_parallelism=True,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class NevergradSamplingSearch(Algorithm):
    """Minimize a scalar function using SamplingSearch.

    This is a one-shot optimization method that is better than random search because it
    uses low-discrepancy sequences to ensure more uniform coverage of the search space.
    It is recommended to use "Hammersley" as the sampler if the budget is known, and to
    set `scrambled=True` in high dimensions.

    """

    sampler: Literal["Halton", "LHS", "Hammersley"] = "Halton"
    """Choice of the low-discrepancy sampler used for generating points.

    'LHS' is Latin Hypercube Sampling.

    """

    scrambled: bool = False
    """If True, adds scrambling to the search sequence, which is highly recommended for
    high-dimensional problems."""

    middle_point: bool = False
    """If True, the first suggested point is the zero vector, useful for initializing at
    the center of the search space."""

    cauchy: bool = False
    """If True, uses the inverse Cauchy distribution instead of Gaussian when projecting
    samples to a real-valued space."""

    scale: bool | NonNegativeFloat = 1.0
    """A float multiplier to scale all generated points."""

    rescaled: bool = False
    """If True, rescales the sampling pattern to ensure better coverage of the
    boundaries."""

    recommendation_rule: Literal["average_of_best", "pessimistic"] = "pessimistic"
    """How the final recommendation is chosen.

    'pessimistic' is the default.

    """

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL
    """Maximum number of function evaluations before termination."""

    n_cores: PositiveInt = 1
    """Number of cores to use for parallel function evaluation."""

    seed: int | None = None
    """Seed for the random number generator for reproducibility."""

    sigma: float | None = None
    """Standard deviation for sampling initial population from N(0, σ²)in case bounds
    are not provided."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_NEVERGRAD_INSTALLED:
            raise NotInstalledError(NEVERGRAD_NOT_INSTALLED_ERROR)

        import nevergrad as ng

        configured_optimizer = ng.optimizers.SamplingSearch(
            sampler=self.sampler,
            scrambled=self.scrambled,
            middle_point=self.middle_point,
            cauchy=self.cauchy,
            scale=self.scale,
            rescaled=self.rescaled,
            recommendation_rule=self.recommendation_rule,
        )

        res = _nevergrad_internal(
            problem=problem,
            x0=x0,
            configured_optimizer=configured_optimizer,
            stopping_maxfun=self.stopping_maxfun,
            n_cores=self.n_cores,
            seed=self.seed,
            sigma=self.sigma,
            nonlinear_constraints=problem.nonlinear_constraints,
        )
        return res


# TODO https://facebookresearch.github.io/nevergrad/optimizers_ref.html#nevergrad.families.EvolutionStrategy


class Wizard(str, Enum):
    """Available portfolio optimizers from Nevergrad."""

    # REF https://openreview.net/pdf/bcf18ffaccd27991ddf707a37b164dbab4ec4771.pdf
    NGOpt = "NGOpt"
    NGOpt4 = "NGOpt4"
    NGOpt8 = "NGOpt8"
    NGOpt10 = "NGOpt10"
    NGOpt12 = "NGOpt12"
    NGOpt13 = "NGOpt13"
    NGOpt14 = "NGOpt14"
    NGOpt15 = "NGOpt15"
    NGOpt16 = "NGOpt16"
    NGOpt21 = "NGOpt21"
    NGOpt36 = "NGOpt36"
    NGOpt38 = "NGOpt38"
    NGOpt39 = "NGOpt39"
    NGOptRW = "NGOptRW"
    NGOptF = "NGOptF"
    NGOptF2 = "NGOptF2"
    NGOptF3 = "NGOptF3"
    NGOptF5 = "NGOptF5"

    NgIoh2 = "NgIoh2"
    NgIoh3 = "NgIoh3"
    NgIoh4 = "NgIoh4"
    NgIoh5 = "NgIoh5"
    NgIoh6 = "NgIoh6"
    NgIoh7 = "NgIoh7"
    NgIoh11 = "NgIoh11"
    NgIoh14 = "NgIoh14"
    NgIoh13 = "NgIoh13"
    NgIoh15 = "NgIoh15"
    NgIoh12 = "NgIoh12"
    NgIoh16 = "NgIoh16"
    NgIoh17 = "NgIoh17"
    NgIoh21 = "NgIoh21"
    NgIoh20 = "NgIoh20"
    NgIoh19 = "NgIoh19"
    NgIoh18 = "NgIoh18"
    NgIoh10 = "NgIoh10"
    NgIoh9 = "NgIoh9"
    NgIoh8 = "NgIoh8"
    NgIoh12b = "NgIoh12b"
    NgIoh13b = "NgIoh13b"
    NgIoh14b = "NgIoh14b"
    NgIoh15b = "NgIoh15b"

    NgDS = "NgDS"
    NgDS2 = "NgDS2"
    NGDSRW = "NGDSRW"
    NGO = "NGO"
    NgIohRW2 = "NgIohRW2"
    NgIohTuned = "NgIohTuned"

    CSEC = "CSEC"
    CSEC10 = "CSEC10"
    CSEC11 = "CSEC11"
    Wiz = "Wiz"


@mark.minimizer(
    name="nevergrad_wizard",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NEVERGRAD_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=False,
    supports_parallelism=True,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class NevergradWizard(Algorithm):
    """Minimize a scalar function using a Meta Optimizer from Nevergrad.

    These are meta-optimizers that intelligently combine multiple different
    optimization algorithms to solve a problem. The specific portfolio of
    optimizers can be selected via the `optimizer` parameter.

    """

    optimizer: Wizard = Wizard.NgIoh10  # rename algorithm_selection maybe
    """The specific Nevergrad meta-optimizer to use.

    Each option is a portfolio of different algorithms.

    """

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL
    """Maximum number of function evaluations before termination."""

    n_cores: PositiveInt = 1
    """Number of cores to use for parallel function evaluation."""

    seed: int | None = None
    """Seed for the random number generator for reproducibility."""

    sigma: float | None = None
    """Standard deviation for sampling initial population from N(0, σ²)in case bounds
    are not provided."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_NEVERGRAD_INSTALLED:
            raise NotInstalledError(NEVERGRAD_NOT_INSTALLED_ERROR)

        import nevergrad as ng

        configured_optimizer = getattr(ng.optimizers, self.optimizer)

        res = _nevergrad_internal(
            problem=problem,
            x0=x0,
            configured_optimizer=configured_optimizer,
            stopping_maxfun=self.stopping_maxfun,
            n_cores=self.n_cores,
            seed=self.seed,
            sigma=self.sigma,
            nonlinear_constraints=problem.nonlinear_constraints,
        )

        return res


class Portfolio(str, Enum):
    """Available portfolio optimizers in Nevergrad."""

    Carola1 = "Carola1"
    """
    CAROLA1 - Cost-effective Asymptotic Randomized Optimization with Limited Access.

    Method:

    1. COBYLA (budget b/2).
    2. CMA with Meta Model (budget b/2), starting from COBYLA’s best solution,
    """
    Carola2 = "Carola2"
    """
    CAROLA2 - see Carola1

    Method

    1. COBYLA (budget b/3) for fast approximation.
    2. CMA with meta-model (budget b/3), starting from COBYLA’s best solution,
        for robust local search.
    3. SQP (budget b/3), starting from the best solution so far,
        for fast refinement.
    """
    Carola3 = "Carola3"
    """
    CAROLA3 - CAROLA2 for the parallel case. see Carola2,

    Method
    1. Apply w copies of Carola2 in parallel, with budget b/w."""

    MultiBFGSPlus = "MultiBFGSPlus"
    LogMultiBFGSPlus = "LogMultiBFGSPlus"
    SqrtMultiBFGSPlus = "SqrtMultiBFGSPlus"
    MultiCobylaPlus = "MultiCobylaPlus"
    MultiSQPPlus = "MultiSQPPlus"
    BFGSCMAPlus = "BFGSCMAPlus"
    LogBFGSCMAPlus = "LogBFGSCMAPlus"
    SqrtBFGSCMAPlus = "SqrtBFGSCMAPlus"
    SQPCMAPlus = "SQPCMAPlus"
    LogSQPCMAPlus = "LogSQPCMAPlus"
    SqrtSQPCMAPlus = "SqrtSQPCMAPlus"

    MultiBFGS = "MultiBFGS"
    LogMultiBFGS = "LogMultiBFGS"
    SqrtMultiBFGS = "SqrtMultiBFGS"
    MultiCobyla = "MultiCobyla"
    ForceMultiCobyla = "ForceMultiCobyla"
    MultiSQP = "MultiSQP"
    BFGSCMA = "BFGSCMA"
    LogBFGSCMA = "LogBFGSCMA"
    SqrtBFGSCMA = "SqrtBFGSCMA"
    SQPCMA = "SQPCMA"
    LogSQPCMA = "LogSQPCMA"
    SqrtSQPCMA = "SqrtSQPCMA"
    FSQPCMA = "FSQPCMA"
    F2SQPCMA = "F2SQPCMA"
    F3SQPCMA = "F3SQPCMA"

    MultiDiscrete = "MultiDiscrete"
    CMandAS2 = "CMandAS2"
    CMandAS3 = "CMandAS3"
    MetaCMA = "MetaCMA"
    CMA = "CMA"
    PCEDA = "PCEDA"
    MPCEDA = "MPCEDA"
    MEDA = "MEDA"
    NoisyBandit = "NoisyBandit"
    Shiwa = "Shiwa"


@mark.minimizer(
    name="nevergrad_portfolio",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NEVERGRAD_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=False,
    supports_parallelism=True,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class NevergradPortfolio(Algorithm):
    """Minimize a scalar function using a Meta Optimizer from Nevergrad.

    This algorithm utilizes a combination of local and global optimizers to find
    the best solution. The specific portfolio of optimizers can be selected via
    the `optimizer` parameter.

    """

    optimizer: Portfolio = Portfolio.BFGSCMA
    """The specific Nevergrad meta-optimizer to use.

    Each option is a portfolio of different local and global algorithms.

    """

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL
    """Maximum number of function evaluations before termination."""

    n_cores: PositiveInt = 1
    """Number of cores to use for parallel function evaluation."""

    seed: int | None = None
    """Seed for the random number generator for reproducibility."""

    sigma: float | None = None
    """Standard deviation for sampling initial population from N(0, σ²) in case bounds
    are not provided."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_NEVERGRAD_INSTALLED:
            raise NotInstalledError(NEVERGRAD_NOT_INSTALLED_ERROR)

        import nevergrad as ng

        configured_optimizer = getattr(ng.optimizers, self.optimizer)

        res = _nevergrad_internal(
            problem=problem,
            x0=x0,
            configured_optimizer=configured_optimizer,
            stopping_maxfun=self.stopping_maxfun,
            n_cores=self.n_cores,
            seed=self.seed,
            sigma=self.sigma,
            nonlinear_constraints=problem.nonlinear_constraints,
        )

        return res


def _nevergrad_internal(
    problem: InternalOptimizationProblem,
    x0: NDArray[np.float64],
    n_cores: int,
    configured_optimizer: ConfiguredOptimizer,
    stopping_maxfun: int,
    seed: int | None,
    sigma: float | None,
    nonlinear_constraints: list[dict[str, Any]] | None,
) -> InternalOptimizeResult:
    """Internal helper function for nevergrad.

    Handle the optimization loop.

    Args:
        problem (InternalOptimizationProblem): Internal optimization problem to solve.
        x0 (np.ndarray): Initial parameter vector of shape (n_params,).
        n_cores (int):  Number of processes used to parallelize the function
            evaluations.
        configured_optimizer (ConfiguredOptimizer): Nevergrad optimizer instance
            configured with options.
        stopping_maxfun (int): Maximum number of function evaluations.
        seed (int): Random seed for reproducibility. Defaults to None.

    Returns:
        InternalOptimizeResult: Internal optimization result

    """

    import nevergrad as ng

    param = ng.p.Array(
        init=x0,
        lower=problem.bounds.lower,
        upper=problem.bounds.upper,
    )

    instrum = ng.p.Instrumentation(param)

    # In case bounds are not provided, the initial population is sampled
    # from a gaussian with mean = 0 and sd = 1,
    # which can be set through this method.
    param.set_mutation(sigma=sigma)

    if seed is not None:
        instrum.random_state.seed(seed)

    optimizer = configured_optimizer(
        parametrization=instrum, budget=stopping_maxfun, num_workers=n_cores
    )

    ### Skip handling of non_linear constraints until improve constraint handling.
    # if nonlinear_constraints:
    #     constraints = _process_nonlinear_constraints(nonlinear_constraints)
    ###

    # optimization loop using the ask-and-tell interface
    while optimizer.num_ask < stopping_maxfun:
        x_list = [
            optimizer.ask()
            for _ in range(min(n_cores, stopping_maxfun - optimizer.num_ask))
        ]

        losses = problem.batch_fun([x.value[0][0] for x in x_list], n_cores=n_cores)

        if not nonlinear_constraints:
            for x, loss in zip(x_list, losses, strict=True):
                optimizer.tell(x, loss)

    ### Skip handling of non_linear constraints until improve constraint handling.
    # else:
    # constraint_violations = _batch_constraint_evaluations(
    #     constraints, [x.value[0][0] for x in x_list], n_cores
    # )
    # for x, loss, cv in zip(x_list, losses, constraint_violations, strict=True):
    #     optimizer.tell(x, loss, cv)
    ###

    recommendation = optimizer.provide_recommendation()
    best_x = recommendation.value[0][0]
    loss = recommendation.loss

    # In case of CMA, loss is not provided by the optimizer, in that case,
    # evaluate it manually using problem.fun
    if loss is None:
        loss = problem.fun(best_x)

    result = InternalOptimizeResult(
        x=best_x,
        fun=loss,
        success=True,
        n_fun_evals=optimizer.num_ask,
        n_jac_evals=0,
        n_hess_evals=0,
    )

    return result


### Skip handling of non_linear constraints until improve constraint handling.

# def _process_nonlinear_constraints(
#     constraints: list[dict[str, Any]],
# ) -> list[dict[str, Any]]:
#     """Process stacked inequality constraints as single constraints.

#     Returns a list of single constraints.

#     """
#     processed_constraints = []
#     for c in constraints:
#         new = _vector_to_list_of_scalar(c)
#         processed_constraints.extend(new)
#     return processed_constraints


# def _get_constraint_evaluations(
#     constraints: list[dict[str, Any]], x: NDArray[np.float64]
# ) -> list[NDArray[np.float64]]:
#     """In optimagic, inequality constraints are internally defined as g(x) >= 0.
#    Nevergrad uses h(x) <= 0 hence a sign flip is required. Passed equality
#    constraints are treated as inequality constraints with lower bound equal to
#    value. Return a list of constraint evaluations at x.

#     """
#     results = [-c["fun"](x) for c in constraints]
#     results = [np.atleast_1d(i) for i in results]
#     return results


# def _batch_constraint_evaluations(
#     constraints: list[dict[str, Any]], x_list: list[Any], n_cores: int
# ) -> list[list[NDArray[np.float64]]]:
#     """Batch version of _get_constraint_evaluations."""
#     batch = process_batch_evaluator("joblib")
#     func = partial(_get_constraint_evaluations, constraints)
#     results = batch(func=func, arguments=[x for x in x_list], n_cores=n_cores)
#     return results
###
