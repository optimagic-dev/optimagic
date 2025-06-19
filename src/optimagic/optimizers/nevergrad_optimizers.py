"""Implement nevergrad optimizers."""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from optimagic import mark
from optimagic.config import IS_NEVERGRAD_INSTALLED
from optimagic.exceptions import NotInstalledError
from optimagic.optimization.algo_options import STOPPING_MAXFUN_GLOBAL, STOPPING_MAXITER
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.internal_optimization_problem import (
    InternalOptimizationProblem,
)
from optimagic.typing import (
    AggregationLevel,
    NonNegativeFloat,
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
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class NevergradPSO(Algorithm):
    """Particle Swarm Optimization (PSO) optimizer from Nevergrad.

    Args:
        transform: The transform to use to map from PSO optimization space to R-space.
            Available options are:
            - "arctan": Uses arctan transformation for bounded optimization
            - "identity": No transformation (for unbounded optimization)
            - "gaussian": Uses Gaussian transformation for bounded optimization
        population_size: Population size of the particle swarm. If None, it will be
            set to 4 * n_parameters. Larger values may improve optimization quality
            but increase computational cost.
        n_cores: Number of cores to use for parallel function evaluation.
        seed: Seed used by the internal random number generator for reproducibility.
        stopping_maxfun: Maximum number of function evaluations before termination.
        inertia: Inertia weight (ω). Controls the influence of the particle's previous
            velocity on its movement. Must be smaller than 1 to prevent divergence.
        cognitive: Cognitive coefficient (φₚ). Controls the influence of the particle's
            own best known position.
        social: Social coefficient (φ₉). Controls the influence of the swarm's best
            known position.
        quasi_opp_init: If True, uses quasi-opposition initialization for particle
            positions. This can improve optimization by providing better initial
            coverage
            of the search space.
        speed_quasi_opp_init: If True, uses quasi-opposition initialization for particle
            velocities. This can help in exploring the search space more effectively.
        special_speed_quasi_opp_init: If True, uses a special quasi-opposition
            initialization
            for velocities. This is an experimental feature that may improve performance
            on certain problems.

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
    supports_nonlinear_constraints=False,
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
    stopping_cov_mat_cond: float = 1e14
    convergence_ftol_abs: float = 1e-11
    convergence_ftol_rel: float = 0.0
    convergence_xtol_abs: float = 1e-11
    convergence_iter_noimprove: int | None = None
    invariant_path: bool = False
    eval_final_mean: bool = True
    seed: int | None = None

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
        )
        return res


def _nevergrad_internal(
    problem: InternalOptimizationProblem,
    x0: NDArray[np.float64],
    n_cores: int,
    configured_optimizer: ng.optimization.base.ConfiguredOptimizer,
    stopping_maxfun: int,
    seed: int | None,
) -> InternalOptimizeResult:
    """Internal helper function for nevergrad.

    Handles the optimization loop for Nevergrad optimizers, including parameter bounds,
    parallel function evaluation, and result.

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

    if not (
        problem.bounds.lower is not None
        and problem.bounds.upper is not None
        and np.all(np.isfinite(problem.bounds.lower))
        and np.all(np.isfinite(problem.bounds.upper))
    ):
        raise ValueError("Bounds cannot be None or infinite.")

    param = ng.p.Array(
        init=x0,
        lower=problem.bounds.lower,
        upper=problem.bounds.upper,
    )

    instrum = ng.p.Instrumentation(param)

    if seed is not None:
        instrum.random_state.seed(seed)

    optimizer = configured_optimizer(
        parametrization=instrum, budget=stopping_maxfun, num_workers=n_cores
    )

    # optimization loop using the ask-and-tell interface
    while optimizer.num_ask < stopping_maxfun:
        x_list = [
            optimizer.ask()
            for _ in range(min(n_cores, stopping_maxfun - optimizer.num_ask))
        ]
        losses = problem.batch_fun([x.value[0][0] for x in x_list], n_cores=n_cores)
        for x, loss in zip(x_list, losses, strict=True):
            optimizer.tell(x, loss)

    recommendation = optimizer.provide_recommendation()
    best_x = recommendation.value[0][0]
    loss = recommendation.loss

    # in some cases, loss is not provided by the optimizer, in that case,
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
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class NevergradDifferentialEvolution(Algorithm):
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
        )
        return res
