"""Implement optimizers from the nevergrad package."""

import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from optimagic import mark
from optimagic.batch_evaluators import process_batch_evaluator
from optimagic.config import IS_NEVERGRAD_INSTALLED
from optimagic.exceptions import NotInstalledError
from optimagic.optimization.algo_options import STOPPING_MAXFUN_GLOBAL, STOPPING_MAXITER
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.internal_optimization_problem import (
    InternalOptimizationProblem,
)
from optimagic.parameters.nonlinear_constraints import _vector_to_list_of_scalar
from optimagic.typing import (
    AggregationLevel,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
)

if IS_NEVERGRAD_INSTALLED:
    import nevergrad as ng


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
    supports_parallelism=True,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=True,
    disable_history=False,
)
@dataclass(frozen=True)
class NevergradPSO(Algorithm):
    r"""Minimize a scalar function using the Particle Swarm Optimization algorithm.

    The Particle Swarm Optimization algorithm was originally proposed
    by :cite:`Kennedy1995`.The implementation in Nevergrad is based
    on :cite:`Zambrano2013`.

    PSO solves an optimization problem by evolving a swarm of particles
    (candidate solutions) across the search space. Each particle adjusts its
    position based on its own experience (cognitive component) and the experiences of
    its neighbors or the swarm (social component), using velocity updates. The algorithm
    iteratively guides the swarm toward promising regions of the search space.

    Args:
        transform (str): The transform used to map from PSO optimization space to
            real space. Options:
            - "arctan" (default)
            - "identity"
            - "gaussian"
        population_size (int): The number of particles in the swarm.
        n_cores (int): The number of CPU cores to use for parallel computation.
        seed (int, optional): Random seed for reproducibility.
        stopping_maxfun (int, optional): Maximum number of function evaluations.
        inertia (float): Inertia weight ω. Controls the influence of a particle's
            previous velocity. Must be less than 1 to avoid divergence.
            Default is 0.7213475204444817.
        cognitive (float): Cognitive coefficient :math:`\phi_p`. Controls the influence
            of a particle’s own best known position. Typical values: 1.0 to 3.0.
            Default is 1.1931471805599454.
        social (float): Social coefficient. Denoted by :math:`\phi_g`. Controls the
            influence of the swarm’s best known position. Typical values: 1.0 to 3.0.
            Default is 1.1931471805599454.
        quasi_opp_init (bool): Whether to use quasi-opposition initialization.
            Default is False.
        speed_quasi_opp_init (bool): Whether to apply quasi-opposition
            initialization to speed. Default is False.
        special_speed_quasi_opp_init (bool): Whether to use special quasi-opposition
            initialization for speed. Default is False.

    """

    transform: Literal["arctan", "gaussian", "identity"] = "arctan"
    population_size: int | None = None
    n_cores: int = 1
    seed: int | None = None
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL
    inertia: float = 0.5 / math.log(2.0)
    cognitive: float = 0.5 + math.log(2.0)
    social: float = 0.5 + math.log(2.0)
    quasi_opp_init: bool = False
    speed_quasi_opp_init: bool = False
    special_speed_quasi_opp_init: bool = False
    sigma: int | None = None

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_NEVERGRAD_INSTALLED:
            raise NotInstalledError(NEVERGRAD_NOT_INSTALLED_ERROR)

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
    supports_parallelism=True,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=True,
    disable_history=False,
)
@dataclass(frozen=True)
class NevergradCMAES(Algorithm):
    """Covariance Matrix Adaptation Evolution Strategy (CMA-ES) optimizer from
    Nevergrad.

    Args:
        scale: Scale of the search.
        elitist: Whether to switch to elitist mode (also known as (μ,λ)-CMA-ES).
            In elitist mode, the best point in the population is always retained.
        population_size: Population size.
        diagonal: Use the diagonal version of CMA, which is more efficient for
            high-dimensional problems.
        high_speed: Use a metamodel for recommendation to speed up optimization.
        fast_cmaes: Use the fast CMA-ES implementation. Cannot be used with
            diagonal=True.
            Produces equivalent results and is preferable for high dimensions or when
            objective function evaluations are fast.
        random_init: If True, initialize the optimizer with random parameters.
        n_cores: Number of cores to use for parallel function evaluation.
        step_size_adaptive: Whether to adapt the step size. Can be a boolean or a string
            specifying the adaptation strategy.
        step_size_damping_factor: Damping factor for step size adaptation.
        step_size_damping_rate: Damping rate for step size adaptation.
        step_size_update_squared: Whether to use squared step sizes in updates.
        learning_rate_cov_mat_update: Learning rate for the covariance matrix update.
        learning_rate_rank_one_update: Multiplier for the rank-one update learning rate
            of the covariance matrix.
        learning_rate_rank_mu_update: Multiplier for the rank-mu update learning rate
            of the covariance matrix.
        learning_rate_mean_update: Learning rate for the mean update.
        learning_rate_diagonal_update: Learning rate for the diagonal update.
        num_parents: Number of parents(μ) for recombination.
        negative_update: Whether to use negative updates for the covariance matrix.
        mirror_sampling_strategy: Strategy for mirror sampling. Possible values are:
            0: Unconditional mirroring
            1: Selective mirroring
            2: Selective mirroring with delay (default)
        normalize_cov_trace: How to normalize the trace of the covariance matrix.
            Valid values are:
            - False: No normalization
            - True: Normalize to 1
            - "arithm": Arithmetic mean normalization
            - "geom": Geometric mean normalization
            - "aeig": Arithmetic mean of eigenvalues
            - "geig": Geometric mean of eigenvalues
        diag_covariance_iters: Number of iterations to use diagonal covariance matrix
            before switching to full matrix. If False, always use full matrix.
        stopping_maxfun: Maximum number of function evaluations before termination.
        stopping_maxiter: Maximum number of iterations before termination.
        stopping_timeout: Maximum time in seconds before termination.
        stopping_cov_mat_cond: Maximum condition number of the covariance matrix before
            termination.
        convergence_ftol_abs: Absolute tolerance on function value changes for
            convergence.
        convergence_ftol_rel: Relative tolerance on function value changes for
            convergence.
        convergence_xtol_abs: Absolute tolerance on parameter changes for convergence.
        convergence_iter_noimprove: Number of iterations without improvement before
            termination.
        invariant_path: Whether evolution path (pc) should be invariant to
            transformations.
        eval_final_mean: Whether to evaluate the final mean solution.
        seed: Seed used by the internal random number generator for reproducibility.

    """

    scale: NonNegativeFloat = 1.0
    elitist: bool = False
    population_size: int | None = None
    diagonal: bool = False
    high_speed: bool = False
    fast_cmaes: bool = False
    random_init: bool = False
    n_cores: PositiveInt = 1
    step_size_adaptive: bool | str = True
    CSA_dampfac: float = 1.0
    CMA_dampsvec_fade: float = 0.1
    CSA_squared: bool = False
    CMA_on: float = 1.0
    CMA_rankone: float = 1.0
    CMA_rankmu: float = 1.0
    CMA_cmean: float = 1.0
    CMA_diagonal_decoding: float = 0.0
    num_parents: int | None = None
    CMA_active: bool = True
    CMA_mirrormethod: Literal[0, 1, 2] = 2
    CMA_const_trace: bool | Literal["arithm", "geom", "aeig", "geig"] = False
    CMA_diagonal: int | bool = False
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL
    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    stopping_maxtime: float = float("inf")
    stopping_cov_mat_cond: NonNegativeFloat = 1e14
    convergence_ftol_abs: NonNegativeFloat = 1e-11
    convergence_ftol_rel: NonNegativeFloat = 0.0
    convergence_xtol_abs: NonNegativeFloat = 1e-11
    convergence_iter_noimprove: PositiveInt | None = None
    invariant_path: bool = False
    eval_final_mean: bool = True
    seed: int | None = None
    sigma: int | None = None

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_NEVERGRAD_INSTALLED:
            raise NotInstalledError(NEVERGRAD_NOT_INSTALLED_ERROR)

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
    supports_parallelism=True,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=True,
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
    annealing: Literal[
        "none", "Exp0.9", "Exp0.99", "Exp0.9Auto", "Lin100.0", "Lin1.0", "LinAuto"
    ] = "none"
    sparse: bool = False
    super_radii: bool = False
    smoother: bool = False
    roulette_size: PositiveInt = 64
    antismooth: NonNegativeInt = 4
    crossover: bool = False
    crossover_type: Literal["none", "rand", "max", "min", "onepoint", "twopoint"] = (
        "none"
    )
    tabu_length: NonNegativeInt = 1000
    rotation: bool = False
    seed: int | None = None
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL
    n_cores: PositiveInt = 1
    sigma: int | None = None

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_NEVERGRAD_INSTALLED:
            raise NotInstalledError(NEVERGRAD_NOT_INSTALLED_ERROR)

        configured_optimizer = ng.optimizers.ParametrizedOnePlusOne(
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
    supports_parallelism=True,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=True,
    disable_history=False,
)
@dataclass(frozen=True)
class NevergradDifferentialEvolution(Algorithm):
    """Differential Evolution optimizer from Nevergrad.

    Differential evolution is typically used for continuous optimization. It uses
    differences between points in the population for doing mutations in fruitful
    directions, making it effective in high dimensions.

    Args:
        initialization: Algorithm/distribution used for initialization.
            Can be one of: "parametrization" (uses parametrization's sample method),
            "LHS" (Latin Hypercube Sampling), "QR" (Quasi-Random),
            "QO" (Quasi-Orthogonal), or "SO" (Sobol sequence).
        scale: Scale of random component of updates. Can be a float or a string.
        recommendation: Criterion for selecting the best point to recommend.
            Options: "pessimistic", "optimistic", "mean", or "noisy".
        crossover: Crossover rate or strategy. Can be:
            - float: Fixed crossover rate
            - "dimension": 1/dimension
            - "random": Random uniform rate per iteration
            - "onepoint": One-point crossover
            - "twopoints": Two-points crossover
            - "rotated_twopoints": Rotated two-points crossover
            - "parametrization": Use parametrization's recombine method
        F1: Differential weight #1 (mutation factor).
        F2: Differential weight #2 (mutation factor for best/current variants).
        popsize: Population size. Can be an integer or one of:
            - "standard": max(num_workers, 30)
            - "dimension": max(num_workers, 30, dimension + 1)
            - "large": max(num_workers, 30, 7 * dimension)
        high_speed: If True, uses a metamodel for recommendations to speed up
            optimization.
        stopping_maxfun: Maximum number of function evaluations before termination.
        n_cores: Number of cores to use for parallel function evaluation.
        seed: Seed for the random number generator for reproducibility.
        sigma: Standard deviation for sampling initial population from N(0, σ²) in
            case bounds are not provided.

    """

    initialization: Literal["parametrization", "LHS", "QR", "QO", "SO"] = (
        "parametrization"
    )
    scale: float | str = 1.0
    recommendation: Literal["pessimistic", "optimistic", "mean", "noisy"] = (
        "pessimistic"
    )
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
    F1: float = 0.8
    F2: float = 0.8
    population_size: int | Literal["standard", "dimension", "large"] = "standard"
    high_speed: bool = False
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL
    n_cores: PositiveInt = 1
    seed: int | None = None
    sigma: int | None = None

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_NEVERGRAD_INSTALLED:
            raise NotInstalledError(NEVERGRAD_NOT_INSTALLED_ERROR)

        configured_optimizer = ng.optimizers.DifferentialEvolution(
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
class NevergradBayesOptim(Algorithm):
    """Bayes Optim."""

    init_budget: int | None = None
    pca: bool = False
    n_components: NonNegativeFloat = 0.95
    prop_doe_factor: NonNegativeFloat | None = 1
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL
    n_cores: PositiveInt = 1
    seed: int | None = None
    sigma: int | None = None

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_NEVERGRAD_INSTALLED:
            raise NotInstalledError(NEVERGRAD_NOT_INSTALLED_ERROR)

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
    supports_parallelism=True,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=True,
    disable_history=False,
)
@dataclass(frozen=True)
class NevergradEMNA(Algorithm):
    """Estimation of Multivariate Normal Algorithm."""

    isotropic: bool = True
    naive: bool = True
    population_size_adaptation: bool = False
    initial_popsize: int | None = None
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL
    n_cores: PositiveInt = 1
    seed: int | None = None
    sigma: int | None = None

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_NEVERGRAD_INSTALLED:
            raise NotInstalledError(NEVERGRAD_NOT_INSTALLED_ERROR)

        configured_optimizer = ng.optimizers.EMNA(
            isotropic=self.isotropic,
            naive=self.naive,
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
    name="nevergrad_axp",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NEVERGRAD_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=True,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=True,
    disable_history=False,
)
@dataclass(frozen=True)
class NevergradAXPlatform(Algorithm):
    """Estimation of distribution algorithm."""

    isotropic: bool = True
    naive: bool = True
    population_size_adaptation: bool = False
    initial_popsize: int | None = None
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL
    n_cores: PositiveInt = 1
    seed: int | None = None
    sigma: int | None = None

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_NEVERGRAD_INSTALLED:
            raise NotInstalledError(NEVERGRAD_NOT_INSTALLED_ERROR)

        configured_optimizer = ng.optimizers.EMNA(
            isotropic=self.isotropic,
            naive=self.naive,
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
    name="nevergrad_eda",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NEVERGRAD_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=True,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=True,
    disable_history=False,
)
@dataclass(frozen=True)
class NevergradEDA(Algorithm):
    """Estimation of distribution algorithm."""

    isotropic: bool = True
    naive: bool = True
    population_size_adaptation: bool = False
    initial_popsize: int | None = None
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL
    n_cores: PositiveInt = 1
    seed: int | None = None
    sigma: int | None = None

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_NEVERGRAD_INSTALLED:
            raise NotInstalledError(NEVERGRAD_NOT_INSTALLED_ERROR)

        configured_optimizer = ng.optimizers.EMNA(
            isotropic=self.isotropic,
            naive=self.naive,
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
    name="nevergrad_tbpsa",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NEVERGRAD_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=True,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=True,
    disable_history=False,
)
@dataclass(frozen=True)
class NevergradTBPSA(Algorithm):
    """Estimation of distribution algorithm."""

    isotropic: bool = True
    naive: bool = True
    population_size_adaptation: bool = False
    initial_popsize: int | None = None
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL
    n_cores: PositiveInt = 1
    seed: int | None = None
    sigma: int | None = None

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_NEVERGRAD_INSTALLED:
            raise NotInstalledError(NEVERGRAD_NOT_INSTALLED_ERROR)

        configured_optimizer = ng.optimizers.EMNA(
            isotropic=self.isotropic,
            naive=self.naive,
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
    name="nevergrad_spsa",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NEVERGRAD_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=True,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=True,
    disable_history=False,
)
@dataclass(frozen=True)
class NevergradSPSA(Algorithm):
    """Estimation of distribution algorithm."""

    isotropic: bool = True
    naive: bool = True
    population_size_adaptation: bool = False
    initial_popsize: int | None = None
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL
    n_cores: PositiveInt = 1
    seed: int | None = None
    sigma: int | None = None

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_NEVERGRAD_INSTALLED:
            raise NotInstalledError(NEVERGRAD_NOT_INSTALLED_ERROR)

        configured_optimizer = ng.optimizers.EMNA(
            isotropic=self.isotropic,
            naive=self.naive,
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
    name="nevergrad_randomsearch",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NEVERGRAD_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=True,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=True,
    disable_history=False,
)
@dataclass(frozen=True)
class NevergradRandomSearch(Algorithm):
    """Minimize a scalar function using the Random Search algorithm."""

    stupid: bool = False
    middle_point: bool = False
    opposition_mode: Literal["opposite", "quasi", None] = None
    sampler: Literal["parametrization", "gaussian", "cauchy"] = "parametrization"
    scale: float | Literal["random", "auto", "autotune"] = "auto"
    recommendation_rule: Literal[
        "average_of_best", "pessimistic", "average_of_exp_best"
    ] = "pessimistic"
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL
    n_cores: PositiveInt = 1
    seed: int | None = None
    sigma: int | None = None

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_NEVERGRAD_INSTALLED:
            raise NotInstalledError(NEVERGRAD_NOT_INSTALLED_ERROR)

        configured_optimizer = ng.optimizers.RandomSearchMaker(
            stupid=self.stupid,
            middle_point=self.middle_point,
            opposition_mode=self.opposition_mode,
            sampler=self.sampler,
            scale=self.scale,
            recommendation_rule=self.recommendation_rule,
            stopping_maxfun=self.stopping_maxfun,
            n_cores=self.n_cores,
            seed=self.seed,
            sigma=self.sigma,
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
    name="nevergrad_samplingsearch",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NEVERGRAD_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=True,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=True,
    disable_history=False,
)
@dataclass(frozen=True)
class NevergradSamplingSearch(Algorithm):
    """SamplingSearch."""

    sampler: Literal["Halton", "LHS", "Hammersley"] = "Halton"
    scrambled: bool = False
    middle_point: bool = False
    cauchy: bool = False
    scale: bool | NonNegativeFloat = 1.0
    rescaled: bool = False
    recommendation_rule: Literal["average_of_best", "pessimistic"] = "pessimistic"
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL
    n_cores: PositiveInt = 1
    seed: int | None = None
    sigma: int | None = None

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_NEVERGRAD_INSTALLED:
            raise NotInstalledError(NEVERGRAD_NOT_INSTALLED_ERROR)

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


@mark.minimizer(
    name="nevergrad_NGOpt",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NEVERGRAD_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=True,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=True,
    disable_history=False,
)
@dataclass(frozen=True)
class NevergradNGOpt(Algorithm):
    """
    Minimize a scalar function using a Meta Optimizer from Nevergrad.
    Each meta optimizer combines multiples optimizers to solve a problem.
    Args:
        optimizer: One of
            - NGOpt
            - NGOpt4
            - NGOpt8
            - NGOpt10
            - NGOpt12
            - NGOpt13
            - NGOpt14
            - NGOpt15
            - NGOpt16
            - NGOpt21
            - NGOpt36
            - NGOpt38
            - NGOpt39
            - NGOptRW
            - NGOptF
            - NGOptF2
            - NGOptF3
            - NGOptF5
            - NgIoh2
            - NgIoh3
            - NgIoh4
            - NgIoh5
            - NgIoh6
            - NgIoh7
            - NgIoh8
            - NgIoh9
            - NgIoh10
            - NgIoh11
            - NgIoh12
            - NgIoh13
            - NgIoh14
            - NgIoh15
            - NgIoh16
            - NgIoh17
            - NgIoh18
            - NgIoh19
            - NgIoh20
            - NgIoh21
            - NgIoh12b
            - NgIoh13b
            - NgIoh14b
            - NgIoh15b
            - NgIohRW2
            - NgIohTuned
            - NgDS
            - NgDS2
            - NGDSRW
            - NGO
            - CSEC
            - CSEC10
            - CSEC11
            - Wiz
        stopping_maxfun: Maximum number of function evaluations before termination.
        n_cores: Number of cores to use for parallel function evaluation.
        seed: Seed for the random number generator for reproducibility.
        sigma: Standard deviation for sampling initial population from N(0, σ²) in
            case bounds are not provided.

    """

    optimizer: Literal[
        "NGOpt",
        "NGOpt4",
        "NGOpt8",
        "NGOpt10",
        "NGOpt12",
        "NGOpt13",
        "NGOpt14",
        "NGOpt15",
        "NGOpt16",
        "NGOpt21",
        "NGOpt36",
        "NGOpt38",
        "NGOpt39",
        "NGOptRW",
        "NGOptF",
        "NGOptF2",
        "NGOptF3",
        "NGOptF5",
        "NgIoh2",
        "NgIoh3",
        "NgIoh4",
        "NgIoh5",
        "NgIoh6",
        "NgIoh7",
        "NgIoh11",
        "NgIoh14",
        "NgIoh13",
        "NgIoh15",
        "NgIoh12",
        "NgIoh16",
        "NgIoh17",
        "NgIoh21",
        "NgIoh20",
        "NgIoh19",
        "NgIoh18",
        "NgIoh10",
        "NgIoh9",
        "NgIoh8",
        "NgIoh12b",
        "NgIoh13b",
        "NgIoh14b",
        "NgIoh15b",
        "NgDS",
        "NgDS2",
        "NGDSRW",
        "NGO",
        "NgIohRW2",
        "NgIohTuned",
        "CSEC",
        "CSEC10",
        "CSEC11",
        "Wiz",
    ] = "NGOpt"
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL
    n_cores: PositiveInt = 1
    seed: int | None = None
    sigma: int | None = None

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_NEVERGRAD_INSTALLED:
            raise NotInstalledError(NEVERGRAD_NOT_INSTALLED_ERROR)

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


@mark.minimizer(
    name="nevergrad_meta",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NEVERGRAD_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=True,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=True,
    disable_history=False,
)
@dataclass(frozen=True)
class NevergradMeta(Algorithm):
    """Minimize a scalar function using a Meta Optimizer from Nevergrad.
    Utilizes a combination of local and global optimizers to find the best solution.
    Local optimizers like BFGS are wrappers over scipy implementations.
    Each meta optimizer combines multiples optimizers to solve a problem.
    Args:
        optimizer: One of
            - MultiBFGSPlus
            - LogMultiBFGSPlus
            - SqrtMultiBFGSPlus
            - MultiCobylaPlus
            - MultiSQPPlus
            - BFGSCMAPlus
            - LogBFGSCMAPlus
            - SqrtBFGSCMAPlus
            - SQPCMAPlus
            - LogSQPCMAPlus
            - SqrtSQPCMAPlus
            - MultiBFGS
            - LogMultiBFGS
            - SqrtMultiBFGS
            - MultiCobyla
            - ForceMultiCobyla
            - MultiSQP
            - BFGSCMA
            - LogBFGSCMA
            - SqrtBFGSCMA
            - SQPCMA
            - LogSQPCMA
            - SqrtSQPCMA
            - FSQPCMA
            - F2SQPCMA
            - F3SQPCMA
            - MultiDiscrete
            - CMandAS2
            - CMandAS3
            - MetaCMA
            - CMA
            - PCEDA
            - MPCEDA
            - MEDA
            - NoisyBandit
            - SPSA
            - Shiwa
            - Carola3
        stopping_maxfun: Maximum number of function evaluations before termination.
        n_cores: Number of cores to use for parallel function evaluation.
        seed: Seed for the random number generator for reproducibility.
        sigma: Standard deviation for sampling initial population from N(0, σ²) in
            case bounds are not provided.
    """

    optimizer: Literal[
        "MultiBFGSPlus",
        "LogMultiBFGSPlus",
        "SqrtMultiBFGSPlus",
        "MultiCobylaPlus",
        "MultiSQPPlus",
        "BFGSCMAPlus",
        "LogBFGSCMAPlus",
        "SqrtBFGSCMAPlus",
        "SQPCMAPlus",
        "LogSQPCMAPlus",
        "SqrtSQPCMAPlus",
        "MultiBFGS",
        "LogMultiBFGS",
        "SqrtMultiBFGS",
        "MultiCobyla",
        "ForceMultiCobyla",
        "MultiSQP",
        "BFGSCMA",
        "LogBFGSCMA",
        "SqrtBFGSCMA",
        "SQPCMA",
        "LogSQPCMA",
        "SqrtSQPCMA",
        "FSQPCMA",
        "F2SQPCMA",
        "F3SQPCMA",
        "MultiDiscrete",
        "CMandAS2",
        "CMandAS3",
        "MetaCMA",
        "CMA",
        "PCEDA",
        "MPCEDA",
        "MEDA",
        "NoisyBandit",
        "SPSA",
        "Shiwa",
        "Carola3",
    ] = "Shiwa"
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL
    n_cores: PositiveInt = 1
    seed: int | None = None
    sigma: int | None = None

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_NEVERGRAD_INSTALLED:
            raise NotInstalledError(NEVERGRAD_NOT_INSTALLED_ERROR)

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
    configured_optimizer: "ng.optimization.base.ConfiguredOptimizer",
    stopping_maxfun: int,
    seed: int | None,
    sigma: int | None,
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

    param = ng.p.Array(
        init=x0,
    )

    param.set_bounds(
        lower=problem.bounds.lower,
        upper=problem.bounds.upper,
    )

    # In case bounds are not provided, the initial population is sampled
    # from a gaussian with mean = 0 and sigma = 1,
    # which can be set through this method.
    param.set_mutation(sigma=sigma)

    instrum = ng.p.Instrumentation(param)

    if seed is not None:
        instrum.random_state.seed(seed)

    optimizer = configured_optimizer(
        parametrization=instrum, budget=stopping_maxfun, num_workers=n_cores
    )

    if nonlinear_constraints:
        constraints = _process_nonlinear_constraints(nonlinear_constraints)

    # optimization loop using the ask-and-tell interface
    while optimizer.num_ask < stopping_maxfun:
        x_list = [
            optimizer.ask().value[0][0]
            for _ in range(min(n_cores, stopping_maxfun - optimizer.num_ask))
        ]

        losses = problem.batch_fun([x for x in x_list], n_cores=n_cores)

        if not nonlinear_constraints:
            for x, loss in zip(x_list, losses, strict=True):
                optimizer.tell(x, loss)
        else:
            constraint_violations = _batch_constraint_evaluations(
                constraints, x_list, n_cores
            )
            for x, loss, cv in zip(x_list, losses, constraint_violations, strict=True):
                optimizer.tell(x, loss, cv)

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


def _process_nonlinear_constraints(
    constraints: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Process stacked inequality constraints as single constraints.

    Returns a list of single constraints.

    """
    processed_constraints = []
    for c in constraints:
        new = _vector_to_list_of_scalar(c)
        processed_constraints.extend(new)
    return processed_constraints


def _get_constraint_evaluations(
    constraints: list[dict[str, Any]], x: NDArray[np.float64]
) -> list[NDArray[np.float64]]:
    """In optimagic, inequality constraints are internally defined as g(x) >= 0.

    Nevergrad uses h(x) <= 0 hence a sign flip is required. Passed equality constraints
    are treated as inequality constraints with lower bound equal to value. Return a list
    of constraint evaluations at x.

    """
    results = [-c["fun"](x) for c in constraints]
    results = [np.atleast_1d(i) for i in results]
    return results


def _batch_constraint_evaluations(
    constraints: list[dict[str, Any]], x_list: list[Any], n_cores: int
) -> list[list[NDArray[np.float64]]]:
    """Batch version of _get_constraint_evaluations."""
    batch = process_batch_evaluator("joblib")
    func = partial(_get_constraint_evaluations, constraints)
    results = batch(func=func, arguments=[x for x in x_list], n_cores=n_cores)
    return results
