from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from optimagic import mark
from optimagic.config import IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED
from optimagic.optimization.algo_options import (
    CONVERGENCE_FTOL_ABS,
    STOPPING_MAXFUN_GLOBAL,
    STOPPING_MAXITER,
    get_population_size,
)
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.internal_optimization_problem import (
    InternalBounds,
    InternalOptimizationProblem,
)
from optimagic.parameters.conversion import Converter
from optimagic.typing import (
    AggregationLevel,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    PyTree,
    YesNoBool,
)
from optimagic.typing import (
    UnitIntervalFloat as ProbabilityFloat,
)

if TYPE_CHECKING:
    from gradient_free_optimizers.optimizers.base_optimizer import BaseOptimizer


@dataclass(frozen=True)
class GFOCommonOptions:
    """Common options for all optimizers from GFO."""

    n_grid_points: PositiveInt | PyTree = 200
    """Number of grid points per dimension.

    If an integer is provided, it will be used for all dimensions.

    """

    n_init: PositiveInt = 20
    """Number of initialization steps to run.

    Accordingly, N positions will be initialized at the vertices and remaining
    initialized randmoly in the search space.

    """

    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    """Maximum number of iterations."""

    stopping_maxtime: NonNegativeFloat | None = None
    """Maximum time in seconds before termination."""

    stopping_funval: float | None = None
    """"Stop the optimization if the objective function is less than this value."""

    convergence_iter_noimprove: PositiveInt = 10000  # do not want to trigger this
    """Number of iterations without improvement before termination."""

    convergence_ftol_abs: NonNegativeFloat | None = CONVERGENCE_FTOL_ABS
    """Converge if the absolute change in the objective function is less than this
    value."""

    convergence_ftol_rel: NonNegativeFloat | None = None
    """Converge if the relative change in the objective function is less than this
    value."""

    caching: bool = True
    """Whether to cache evaluated param and function values in a dictionary for
    lookup."""

    warm_start: list[PyTree] | None = None
    """List of additional start points for the optimization run."""

    verbosity: Literal["progress_bar", "print_results", "print_times"] | bool = False
    """Determines what part of the optimization information will be printed."""

    seed: int | None = None
    """Random seed for reproducibility."""


@mark.minimizer(
    name="gfo_powells_method",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
    is_global=False,
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
class GFOPowellsMethod(Algorithm, GFOCommonOptions):
    """Minimize a scalar function using Powell's Method.

    This powell's method implementation works by optimizing each search space dimension
    at a time with a hill climbing algorithm. It works by setting the search space range
    for all dimensions except one to a single value. The hill climbing algorithms
    searches the best position within this dimension.

    """

    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    """Maximum number of iterations."""

    iters_p_dim: PositiveInt = 10
    """Number of iterations the algorithm will let the hill-climbing algorithm search to
    find the best position before it changes to the next dimension of the search space.

    Typical range: 5 to 15.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        opt = gfo.PowellsMethod
        optimizer = partial(
            opt,
            iters_p_dim=self.iters_p_dim,
        )

        res = _gfo_internal(
            common_options=self,
            problem=problem,
            x0=x0,
            optimizer=optimizer,
        )
        return res


@mark.minimizer(
    name="gfo_random_annealing",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
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
class GFORandomAnnealing(Algorithm, GFOCommonOptions):
    """Minimize a scalar function using Random Annealing.

    The random annealing algorithm is based on hill climbing and derived on the regular
    simulated annealing algorithm. It takes the idea of a temperature and annealing to
    change the step-size over time.

    """

    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    """Maximum number of iterations."""

    epsilon: NonNegativeFloat = 0.03
    """The step-size of the hill climbing algorithm.

    Increasing epsilon also
    increases the average step-size, because its proportional to the
    standard-deviation of the distribution of the hill-climbing-based algorithm.
    Typical range: 0.01 to 0.3.

    """

    distribution: Literal["normal", "laplace", "logistic", "gumbel"] = "normal"
    """The mathematical distribution the algorithm draws samples from."""

    n_neighbours: PositiveInt = 3
    """The number of positions the algorithm explores from its current postion before
    setting its current position to the best of those neighbour positions.

    Typical range: 1 to 10.

    """

    start_temp: NonNegativeFloat = 10.0
    """The start temperature is set to the given value at the start of the optimization
    run and gets changed by the annealing_rate over time.

    This start_temp is multiplied with epsilon to change the step-size of this hill-
    climbing-based algorithm over time. Typical range: 3 to 25.

    """

    annealing_rate: ProbabilityFloat = 0.97
    """Rate at which the temperature-value of the algorithm decreases.

    An
    annealing rate above 1 increases the temperature over time.
    Typical range: 0.9 to 0.99.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        opt = gfo.RandomAnnealingOptimizer
        optimizer = partial(
            opt,
            epsilon=self.epsilon,
            distribution=self.distribution,
            n_neighbours=self.n_neighbours,
            start_temp=self.start_temp,
            annealing_rate=self.annealing_rate,
        )

        res = _gfo_internal(
            common_options=self,
            problem=problem,
            x0=x0,
            optimizer=optimizer,
        )
        return res


# ==================================================================================
# Grid Search
# ==================================================================================


@mark.minimizer(
    name="gfo_grid_search",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
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
class GFOGridSearch(Algorithm, GFOCommonOptions):
    """Minimize a scalar function using Grid Search.

    The grid search explores the search space by starting from a corner and
    progressing `step_size`-steps per iteration. Increasing the `step_size`
    enables a more uniform exploration of the search space.

    """

    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    """Maximum number of iterations."""

    step_size: PositiveInt = 5
    """The number of steps the grid search takes after each iteration.

    If this
    parameter is set to 3 the grid search won't select the next position, but
    the one it would normally select after 3 iterations. This way we get a
    sparse grid after the first pass through the search space.
    Typical range: 1 to 1000.

    """

    direction: Literal["diagonal", "orthogonal"] = "diagonal"
    """The direction the grid-search will walk through the search-space."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        opt = gfo.GridSearchOptimizer
        optimizer = partial(
            opt,
            step_size=self.step_size,
            direction=self.direction,
        )

        res = _gfo_internal(
            common_options=self,
            problem=problem,
            x0=x0,
            optimizer=optimizer,
        )
        return res


# ==================================================================================
# Global Optimizers
# ==================================================================================


@mark.minimizer(
    name="gfo_lipschitz_optimization",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
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
class GFOLipschitzOptimization(Algorithm, GFOCommonOptions):
    """Minimize a scalar function using Lipschitz Optimization.

    The lipschitz optimization (often called lipo) is a global optimization algorithm
    that calculates an upper bound based on all previously explored positions in the
    search space.

    """

    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    """Maximum number of iterations."""

    max_sample_size: PositiveInt = 10000000
    """A first pass of randomly sampling, before all possible positions are generated
    for the sequence-model-based optimization.

    It samples the search
    space directly and takes effect if the search-space is very large. This is
    necessary to avoid a memory overload.
    Typical range: 1,000,000 to 100,000,000.

    """

    replacement: YesNoBool = True
    """This parameter determines if a position is replaced into the list of possible
    positions after it was selected and evaluated by the sequential model."""

    sampling: Any = 1000000
    """The sampling-parameter is a second pass of randomly sampling.

    It samples from the list of all possible positions (not directly from the search-
    space). This might be necessary, because the predict-method of the surrogate model
    could overload the memory.

    """

    warm_start_smbo: list[PyTree] | None = None
    """A warm start for the sequential-model-based-optimization can be provided as a
    pandas dataframe that contains search-data with the results from a previous
    optimization run."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        opt = gfo.LipschitzOptimizer
        optimizer = partial(
            opt,
            max_sample_size=self.max_sample_size,
            replacement=self.replacement,
            sampling={"random": self.sampling},
            warm_start_smbo=self.warm_start_smbo,
        )

        res = _gfo_internal(
            common_options=self,
            problem=problem,
            x0=x0,
            optimizer=optimizer,
        )
        return res


@mark.minimizer(
    name="gfo_direct_algorithm",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
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
class GFODirectAlgorithm(Algorithm, GFOCommonOptions):
    """Minimize a scalar function using the DIRECT Algorithm.

    The DIRECT algorithm works by separating the search-space into smaller rectangle-
    shaped subspaces and evaluating their center positions. The algorithm decides which
    subspace to further separate by calculating an upper-bound within each subspace.

    """

    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    """Maximum number of iterations."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        opt = gfo.DirectAlgorithm
        optimizer = partial(opt)

        res = _gfo_internal(
            common_options=self,
            problem=problem,
            x0=x0,
            optimizer=optimizer,
        )
        return res


@mark.minimizer(
    name="gfo_pattern_search",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
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
class GFOPatternSearch(Algorithm, GFOCommonOptions):
    """Minimize a scalar function using Pattern Search.

    The pattern search works by initializing a cross-shaped collection of positions in
    the search space. Those positions explore the search-space by moving the collection
    of positions as a whole towards optima or shrinking the cross.

    """

    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    """Maximum number of iterations."""

    n_positions: PositiveInt = 4
    """Number of positions that the pattern consists of.

    If the value of
    `n_positions` is large the algorithm will take a lot of time to choose the next
    position to move to, but the choice will probably be a good one. It might be a
    prudent approach to increase `n_positions` of the search-space has a lot of
    dimensions, because there are more possible directions to move to.
    Typical range: 2 to 8.

    """

    pattern_size: NonNegativeFloat = 0.25
    """The initial size of the patterns in percentage of the size of the search space in
    the corresponding dimension.

    Typical range: 0.1 to 0.5.

    """

    reduction: NonNegativeFloat = 0.9
    """The factor that reduces the size of the pattern if no better position is found.

    Typical range: 0.75 to 0.99.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        opt = gfo.PatternSearch
        optimizer = partial(
            opt,
            n_positions=self.n_positions,
            pattern_size=self.pattern_size,
            reduction=self.reduction,
        )

        res = _gfo_internal(
            common_options=self,
            problem=problem,
            x0=x0,
            optimizer=optimizer,
        )
        return res


@mark.minimizer(
    name="gfo_random_search",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
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
class GFORandomSearch(Algorithm, GFOCommonOptions):
    """Minimize a scalar function using Random Search.

    The random search explores by choosing a new position at random after each
    iteration. The implementation in gradient_free_optimizers is purely random across
    the search space in each step.

    """

    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    """Maximum number of iterations."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        opt = gfo.RandomSearchOptimizer
        optimizer = partial(opt)

        res = _gfo_internal(
            common_options=self,
            problem=problem,
            x0=x0,
            optimizer=optimizer,
        )
        return res


@mark.minimizer(
    name="gfo_randomrestarthillclimbing",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
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
class GFORandomRestartHillClimbing(Algorithm, GFOCommonOptions):
    """Minimize a scalar function using the Random Restart Hill Climbing algorithm.

    This algorithm is a Python implementation of the Random Restart Hill Climbing
    algorithm through the gradient_free_optimizers package.

    The random restart hill climbing works by starting a hill climbing search and
    jumping to a random new position after n_iter_restart iterations. Those restarts
    should prevent the algorithm getting stuck in local optima.

    """

    epsilon: PositiveFloat = 0.03
    """The step-size of the hill climbing algorithm.If step_size is too large the newly
    selected positions will be at the edge of the search space.

    If its value is very low it might not find new positions.

    """

    distribution: Literal["normal", "laplace", "logistic", "gumbel"] = "normal"
    """The mathematical distribution the algorithm draws samples from.

    All available distributions are taken from the numpy-package.

    """

    n_neighbours: PositiveInt = 3
    """The number of positions the algorithm explores from its current postion before
    setting its current position to the best of those neighbour positions.

    If the value of n_neighbours is large the hill-climbing-based algorithm will take a
    lot of time to choose the next position to move to, but the choice will probably be
    a good one. It might be a prudent approach to increase n_neighbours of the search-
    space has a lot of dimensions, because there are more possible directions to move
    to.

    """

    n_iter_restart: PositiveInt = 10
    """The number of iterations the algorithm performs before jumping to a random
    position."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        opt = gfo.RandomRestartHillClimbingOptimizer
        optimizer = partial(
            opt,
            epsilon=self.epsilon,
            distribution=self.distribution,
            n_neighbours=self.n_neighbours,
            n_iter_restart=self.n_iter_restart,
        )
        res = _gfo_internal(
            common_options=self,
            problem=problem,
            x0=x0,
            optimizer=optimizer,
        )

        return res


# ==================================================================================
# Surrogate Model Based Optimizers
# ==================================================================================


@mark.minimizer(
    name="gfo_tree_structured_parzen_estimators",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
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
class GFOTreeStructuredParzenEstimators(Algorithm, GFOCommonOptions):
    """Minimize a scalar function using Tree-structured Parzen Estimators (TPE).

    Tree of Parzen Estimators chooses new positions by calculating an acquisition
    function. It assesses all possible positions by calculating the ratio of their
    probability being among the best positions and the worst positions. Those
    probabilities are determined with a kernel density estimator, which is trained on
    already evaluated positions.

    """

    stopping_maxiter: PositiveInt = 50
    """Maximum number of iterations."""

    gamma_tpe: NonNegativeFloat = 0.5  # non default
    """This parameter determines the separation of the explored positions into good and
    bad.

    It must be in the range between 0 and 1. A value of 0.2 means,
    that the best 20% of the known positions are put into the list of best known
    positions, while the rest is put into the list of worst known positions.
    Typical range: 0.05 to 0.75.

    """

    replacement: YesNoBool = True
    """This parameter determines if a position is replaced into the list of possible
    positions after it was selected and evaluated by the sequential model."""

    sampling: Any = 1000000
    """The sampling-parameter is a second pass of randomly sampling.

    It samples from the list of all possible positions (not directly from the search-
    space). This might be necessary, because the predict-method of the surrogate model
    could overload the memory.

    """

    max_sample_size: PositiveInt = 10000000
    """A first pass of randomly sampling, before all possible positions are generated
    for the sequence-model-based optimization.

    It samples the search
    space directly and takes effect if the search-space is very large. This is
    necessary to avoid a memory overload.
    Typical range: 1,000,000 to 100,000,000.

    """

    warm_start_smbo: list[PyTree] | None = None
    """A warm start for the sequential-model-based-optimization can be provided as a
    pandas dataframe that contains search-data with the results from a previous
    optimization run."""

    rand_rest_p: NonNegativeFloat = 0.0
    """Probability for the optimization algorithm to jump to a random position in an
    iteration step.

    It is set to 0 per default. The idea of this parameter is to give the possibility to
    inject randomness into algorithms that don't normally support it. Typical range:
    0.01 to 0.1.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        opt = gfo.TreeStructuredParzenEstimators
        optimizer = partial(
            opt,
            gamma_tpe=self.gamma_tpe,
            max_sample_size=self.max_sample_size,
            replacement=self.replacement,
            sampling={"random": self.sampling},
            warm_start_smbo=self.warm_start_smbo,
            rand_rest_p=self.rand_rest_p,
        )

        res = _gfo_internal(
            common_options=self,
            problem=problem,
            x0=x0,
            optimizer=optimizer,
        )
        return res


@mark.minimizer(
    name="gfo_forest_optimization",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
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
class GFOForestOptimization(Algorithm, GFOCommonOptions):
    """Minimize a scalar function using Forest Optimization.

    The forest-optimizer calculates      the expected improvement of the position space
    with a tree-based model. This optimization technique is very similar to bayesian-
    optimization in every part, except its surrogate model.

    """

    stopping_maxiter: PositiveInt = 50
    """Maximum number of iterations."""

    xi: NonNegativeFloat = 0.3
    """Parameter for the expected uncertainty of the estimation.

    It is a
    parameter that belongs to the expected-improvement acquisition-function.
    Typical range: 0.1 to 0.9.

    """

    max_sample_size: PositiveInt = 10000000
    """A first pass of randomly sampling, before all possible positions are generated
    for the sequence-model-based optimization.

    It samples the search
    space directly and takes effect if the search-space is very large. This is
    necessary to avoid a memory overload.
    Typical range: 1,000,000 to 100,000,000.

    """

    tree_regressor: Literal["extra_tree", "random_forest", "gradient_boost"] = (
        "extra_tree"
    )

    replacement: YesNoBool = True
    """This parameter determines if a position is replaced into the list of possible
    positions after it was selected and evaluated by the sequential model."""

    sampling: Any = 1000000
    """The sampling-parameter is a second pass of randomly sampling.

    It samples from the list of all possible positions (not directly from the search-
    space). This might be necessary, because the predict-method of the surrogate model
    could overload the memory.

    """

    warm_start_smbo: list[PyTree] | None = None
    """A warm start for the sequential-model-based-optimization can be provided as a
    pandas dataframe that contains search-data with the results from a previous
    optimization run."""

    rand_rest_p: NonNegativeFloat = 0.0
    """Probability for the optimization algorithm to jump to a random position in an
    iteration step.

    It is set to 0 per default. The idea of this parameter is to give the possibility to
    inject randomness into algorithms that don't normally support it. Typical range:
    0.01 to 0.1.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        opt = gfo.ForestOptimizer
        optimizer = partial(
            opt,
            xi=self.xi,
            max_sample_size=self.max_sample_size,
            tree_regressor=self.tree_regressor,
            replacement=self.replacement,
            sampling={"random": self.sampling},
            warm_start_smbo=self.warm_start_smbo,
            rand_rest_p=self.rand_rest_p,
        )

        res = _gfo_internal(
            common_options=self,
            problem=problem,
            x0=x0,
            optimizer=optimizer,
        )
        return res


@mark.minimizer(
    name="gfo_bayesian_optimization",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
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
class GFOBayesianOptimization(Algorithm, GFOCommonOptions):
    """Minimize a scalar function using Bayesian Optimization.

    Bayesian optimization chooses new positions by calculating the expected improvement
    of every position in the search space based on a gaussian process that trains on
    already evaluated positions.

    """

    stopping_maxiter: PositiveInt = 50
    """Maximum number of iterations."""

    xi: NonNegativeFloat = 0.3
    """Parameter for the expected uncertainty of the estimation.

    It is a
    parameter that belongs to the expected-improvement acquisition-function.
    Typical range: 0.1 to 0.9.

    """

    replacement: YesNoBool = True
    """This parameter determines if a position is replaced into the list of possible
    positions after it was selected and evaluated by the sequential model."""

    sampling: Any = 1000000
    """The sampling-parameter is a second pass of randomly sampling.

    It samples from the list of all possible positions (not directly from the search-
    space). This might be necessary, because the predict-method of the surrogate model
    could overload the memory.

    """

    max_sample_size: PositiveInt = 10000000
    """A first pass of randomly sampling, before all possible positions are generated
    for the sequence-model-based optimization.

    It samples the search
    space directly and takes effect if the search-space is very large. This is
    necessary to avoid a memory overload.
    Typical range: 1,000,000 to 100,000,000.

    """

    warm_start_smbo: list[PyTree] | None = None
    """A warm start for the sequential-model-based-optimization can be provided as a
    pandas dataframe that contains search-data with the results from a previous
    optimization run."""

    rand_rest_p: NonNegativeFloat = 0.0
    """Probability for the optimization algorithm to jump to a random position in an
    iteration step.

    It is set to 0 per default. The idea of this parameter is to give the possibility to
    inject randomness into algorithms that don't normally support it. Typical range:
    0.01 to 0.1.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        opt = gfo.BayesianOptimizer
        optimizer = partial(
            opt,
            xi=self.xi,
            max_sample_size=self.max_sample_size,
            replacement=self.replacement,
            sampling={"random": self.sampling},
            warm_start_smbo=self.warm_start_smbo,
            rand_rest_p=self.rand_rest_p,
        )

        res = _gfo_internal(
            common_options=self,
            problem=problem,
            x0=x0,
            optimizer=optimizer,
        )
        return res


# ==================================================================================
# Population Based
# ==================================================================================


@mark.minimizer(
    name="gfo_pso",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
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
class GFOParticleSwarmOptimization(Algorithm, GFOCommonOptions):
    """Minimize a scalar function using the Particle Swarm Optimization algorithm.

    This algorithm is a Python implementation of the Particle Swarm Optimization
    algorithm through the gradient_free_optimizers package.

    Particle Swarm Optimization is a global population based algorithm.
    The algorithm simulates a swarm of particles across the search space.
    Each particle adjusts its position based on its own experience (cognitive weight)
    and the experiences of its neighbors or the swarm (social weight), using
    velocity updates.
    The algorithm iteratively guides the swarm toward promising regions of the
    search space. The velocity of a particle is calculated by the following
    equation:

    .. math::
        v_{n+1} = \\omega \\cdot v_n + c_k \\cdot r_1 \\cdot (p_{best}-p_n)
        + c_s \\cdot r_2 \\cdot (g_{best} - p_n)

    """

    population_size: PositiveInt = 10
    """Size of the population."""

    initial_population: list[PyTree] | None = None
    """The user-provided inital population."""

    inertia: NonNegativeFloat = 0.5
    """The inertia of the movement of the individual particles in the population."""

    cognitive_weight: NonNegativeFloat = 0.5
    """A factor of the movement towards the personal best position of the individual
    particles in the population."""

    social_weight: NonNegativeFloat = 0.5
    """A factor of the movement towards the global best position of the individual
    particles in the population."""

    rand_rest_p: NonNegativeFloat = 0
    """Probability for the optimization algorithm to jump to a random position in an
    iteration step."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=20
        )

        opt = gfo.ParticleSwarmOptimizer
        optimizer = partial(
            opt,
            population=population_size,
            inertia=self.inertia,
            cognitive_weight=self.cognitive_weight,
            social_weight=self.social_weight,
            rand_rest_p=self.rand_rest_p,
        )

        res = _gfo_internal(
            common_options=self,
            problem=problem,
            x0=x0,
            optimizer=optimizer,
        )

        return res


@mark.minimizer(
    name="gfo_parallel_tempering",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
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
class GFOParallelTempering(Algorithm, GFOCommonOptions):
    """Minimize a scalar function using the Parallel Tempering algorithm.

    This algorithm is a Python implementation of the Parallel Tempering
    algorithm through the gradient_free_optimizers package.

    Parallel Tempering is a global optimization algorithm that is inspired by
    metallurgical annealing. It runs multiple optimization chains at different
    "temperatures" in parallel. Periodically, swaps between these chains are
    attempted. Swaps between chains at different temperatures allow the optimizer
    to overcome local optima.

    The acceptance probability of a new position :math:`p_{new}` over an old one
    :math:`p_{old}` is given by:

    .. math::
        AP = e^{-\\frac{f(p_{new}) - f(p_{old})}{T}}

    where :math:`T` is the current temperature.

    """

    population_size: PositiveInt = 10
    """Size of the population, i.e., number of parallel chains."""

    initial_population: list[PyTree] | None = None
    """The user-provided inital population."""

    n_iter_swap: PositiveInt = 10
    """The number of iterations the algorithm performs before switching temperatures of
    the individual optimizers in the population."""

    rand_rest_p: NonNegativeFloat = 0
    """Probability for the optimization algorithm to jump to a random position in an
    iteration step."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=20
        )

        opt = gfo.ParallelTemperingOptimizer
        optimizer = partial(
            opt,
            population=population_size,
            n_iter_swap=self.n_iter_swap,
            rand_rest_p=self.rand_rest_p,
        )

        res = _gfo_internal(
            common_options=self,
            problem=problem,
            x0=x0,
            optimizer=optimizer,
        )

        return res


@mark.minimizer(
    name="gfo_spiral_optimization",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
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
class GFOSpiralOptimization(Algorithm, GFOCommonOptions):
    """Minimize a scalar function using the Spiral Optimization algorithm.

    This algorithm is a Python implementation of the Spiral Optimization
    algorithm through the gradient_free_optimizers package.

    Spiral Optimization is a global optimization algorithm inspired by the dynamics
    of spiral phenomena. It uses a multi-point search strategy that moves towards
    the current best solution in a logarithmic spiral trajectory.

    The spiral model for a two-dimensional search space is defined as:

    .. math::
        x_{i}(k+1) = S_n(r, \\theta) x_{i}(k) - (S_n(r, \\theta) - I_n) x^{*}

    where :math:`x^{*}` is the current center of the spiral (best solution),
    :math:`S_n(r, \\theta)` is a spiral rotation-scaling matrix, :math:`r` is the
    convergence rate, and :math:`\\theta` is the angle of rotation.

    """

    population_size: PositiveInt = 10
    """Size of the population for population-based optimization algorithms.

    Each
    member of the population is a separate optimizer. All population based
    optimizers in this package calculate the new positions one member at a time.
    So if the optimizer performs 10 iterations and has a population size of 10,
    then each member of the population would move once to a new position.
    Typical range: 4 to 25.

    """

    initial_population: list[PyTree] | None = None
    """The user-provided inital population."""

    decay_rate: NonNegativeFloat = 0.99
    """The `r` is called in the spiral-optimization equation and is usually referred to
    as a step-size, but behaves more like a modification factor of the radius of the
    spiral movement of the particles in this implementation.

    Lower values accelerate the convergence of the particles to the best known position,
    while values above 1 eventually lead to a movement where the particles spiral away
    from each other. Typical range: 0.85 to 1.15.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=20
        )

        opt = gfo.SpiralOptimization
        optimizer = partial(
            opt,
            population=population_size,
            decay_rate=self.decay_rate,
        )

        res = _gfo_internal(
            common_options=self,
            problem=problem,
            x0=x0,
            optimizer=optimizer,
        )

        return res


@mark.minimizer(
    name="gfo_genetic_algorithm",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
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
class GFOGeneticAlgorithm(Algorithm, GFOCommonOptions):
    """Minimize a scalar function using the Genetic Algorithm.

    This algorithm is a Python implementation of the Genetic Algorithm through the
    gradient_free_optimizers package.

    The Genetic Algorithm is an evolutionary algorithm inspired by the process of
    natural selection. It evolves a population of candidate solutions over generations
    using mechanisms like selection, crossover, and mutation to find the best solution.

    """

    population_size: PositiveInt = 10
    """Size of the population."""

    initial_population: list[PyTree] | None = None
    """The user-provided inital population."""

    mutation_rate: ProbabilityFloat = 0.5
    """Probability of a mutation event occurring in an individual."""

    crossover_rate: ProbabilityFloat = 0.5
    """Probability of a crossover event occurring between two parents."""

    n_parents: PositiveInt = 2

    offspring: PositiveInt = 10

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=20
        )

        opt = gfo.GeneticAlgorithmOptimizer
        optimizer = partial(
            opt,
            population=population_size,
            mutation_rate=self.mutation_rate,
            crossover_rate=self.crossover_rate,
            n_parents=self.n_parents,
            offspring=self.offspring,
        )

        res = _gfo_internal(
            common_options=self,
            problem=problem,
            x0=x0,
            optimizer=optimizer,
        )

        return res


@mark.minimizer(
    name="gfo_evolution_strategy",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
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
class GFOEvolutionStrategy(Algorithm, GFOCommonOptions):
    """Minimize a scalar function using the Evolution Strategy algorithm.

    This algorithm is a Python implementation of the Evolution Strategy algorithm
    through the gradient_free_optimizers package.

    Evolution Strategy is another type of evolutionary algorithm. It primarily relies on
    mutation to explore the search space. A population of parents generates offspring,
    and the fittest individuals from both parents and offspring are selected to form the
    next generation.

    """

    population_size: PositiveInt = 10
    """Size of the population."""

    stopping_maxiter: PositiveInt = STOPPING_MAXFUN_GLOBAL
    """Maximum number of iterations."""

    initial_population: list[PyTree] | None = None
    """The user-provided inital population."""

    mutation_rate: ProbabilityFloat = 0.7
    """Probability of a mutation event occurring in an individual."""

    crossover_rate: ProbabilityFloat = 0.3
    """Probability of a crossover event occurring between two parents."""

    rand_rest_p: NonNegativeFloat = 0
    """Probability for the optimization algorithm to jump to a random position in an
    iteration step."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=20
        )

        opt = gfo.EvolutionStrategyOptimizer
        optimizer = partial(
            opt,
            population=population_size,
            mutation_rate=self.mutation_rate,
            crossover_rate=self.crossover_rate,
            rand_rest_p=self.rand_rest_p,
        )

        res = _gfo_internal(
            common_options=self,
            problem=problem,
            x0=x0,
            optimizer=optimizer,
        )

        return res


@mark.minimizer(
    name="gfo_differential_evolution",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
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
class GFODifferentialEvolution(Algorithm, GFOCommonOptions):
    """Minimize a scalar function using the Differential Evolution algorithm.

    This algorithm is a Python implementation of the Differential Evolution
    algorithm through the gradient_free_optimizers package.

    Differential Evolution is a population-based optimization algorithm that
    creates new candidate solutions by combining existing ones. It creates new
    positions in the search space by adding the weighted difference between
    two population members to a third member.

    A new trial vector is generated according to:

    .. math::
        x_{trial} = x_{r1} + F \\cdot (x_{r2} - x_{r3})

    where :math:`r1, r2, r3` are random individuals from the population, and
    :math:`F` is the differential weight.

    """

    population_size: PositiveInt = 10
    """Size of the population."""

    initial_population: list[PyTree] | None = None
    """The user-provided inital population."""

    mutation_rate: ProbabilityFloat = 0.9
    """Probability of a mutation event occurring in an individual."""

    crossover_rate: ProbabilityFloat = 0.9
    """Probability of a crossover event occurring between two parents."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=4
        )

        opt = gfo.DifferentialEvolutionOptimizer
        optimizer = partial(
            opt,
            population=population_size,
            mutation_rate=self.mutation_rate,
            crossover_rate=self.crossover_rate,
        )

        res = _gfo_internal(
            common_options=self,
            problem=problem,
            x0=x0,
            optimizer=optimizer,
        )

        return res


# ==================================================================================
# Local optimizers
# ==================================================================================


@mark.minimizer(
    name="gfo_hillclimbing",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
    is_global=False,
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
class GFOHillClimbing(Algorithm, GFOCommonOptions):
    """Minimize a scalar function using the HillClimbing algorithm.

    This algorithm is a Python implementation of the HillClimbing algorithm through the
    gradient_free_optimizers package.

    Hill climbing is a local search algorithm suited for exploring combinatorial search
    spaces.

    It starts at an initial point, which is often chosen randomly and continues to move
    to positions within its neighbourhood with a better solution. It has no method
    against getting stuck in local optima.

    """

    epsilon: PositiveFloat = 0.03
    """The step-size of the hill climbing algorithm. If step_size is too large the newly
    selected positions will be at the edge of the search space.

    If its value is very low it might not find new positions.

    """

    distribution: Literal["normal", "laplace", "logistic", "gumbel"] = "normal"
    """The mathematical distribution the algorithm draws samples from.

    All available distributions are taken from the numpy-package.

    """

    n_neighbours: PositiveInt = 3
    """The number of positions the algorithm explores from its current postion before
    setting its current position to the best of those neighbour positions.

    If the value of n_neighbours is large the hill-climbing-based algorithm will take a
    lot of time to choose the next position to move to, but the choice will probably be
    a good one. It might be a prudent approach to increase n_neighbours of the search-
    space has a lot of dimensions, because there are more possible directions to move
    to.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        opt = gfo.HillClimbingOptimizer
        optimizer = partial(
            opt,
            epsilon=self.epsilon,
            distribution=self.distribution,
            n_neighbours=self.n_neighbours,
        )
        res = _gfo_internal(
            common_options=self,
            problem=problem,
            x0=x0,
            optimizer=optimizer,
        )

        return res


@mark.minimizer(
    name="gfo_stochastichillclimbing",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
    is_global=False,
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
class GFOStochasticHillClimbing(Algorithm, GFOCommonOptions):
    """Minimize a scalar function using the Stochastic Hill Climbing algorithm.

    This algorithm is a Python implementation of the StochasticHillClimbing algorithm
    through the gradient_free_optimizers package.

    Stochastic hill climbing extends the normal hill climbing by a simple method against
    getting stuck in local optima.

    """

    epsilon: PositiveFloat = 0.03
    """The step-size of the hill climbing algorithm.If step_size is too large the newly
    selected positions will be at the edge of the search space.

    If its value is very low it might not find new positions.

    """

    distribution: Literal["normal", "laplace", "logistic", "gumbel"] = "normal"
    """The mathematical distribution the algorithm draws samples from.

    All available distributions are taken from the numpy-package.

    """

    n_neighbours: PositiveInt = 3
    """The number of positions the algorithm explores from its current postion before
    setting its current position to the best of those neighbour positions.

    If the value of n_neighbours is large the hill-climbing-based algorithm will take a
    lot of time to choose the next position to move to, but the choice will probably be
    a good one. It might be a prudent approach to increase n_neighbours of the search-
    space has a lot of dimensions, because there are more possible directions to move
    to.

    """

    p_accept: NonNegativeFloat = 0.5
    """The probability factor used in the equation to calculate if a worse position is
    accepted as the new position.

    If the new score is not better than the previous one the algorithm accepts worse
    positions with probability p_accept.

    .. math::
        score_{normalized} = norm * \\frac{score_{current} - score_{new}}
        {score_{current} + score_{new}}
    .. math::
        p = \\exp^{-score_{normalized}}

    If p is less than p_accept the new position gets accepted anyways.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        opt = gfo.StochasticHillClimbingOptimizer
        optimizer = partial(
            opt,
            epsilon=self.epsilon,
            distribution=self.distribution,
            n_neighbours=self.n_neighbours,
            p_accept=self.p_accept,
        )
        res = _gfo_internal(
            common_options=self,
            problem=problem,
            x0=x0,
            optimizer=optimizer,
        )

        return res


@mark.minimizer(
    name="gfo_repulsinghillclimbing",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
    is_global=False,
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
class GFORepulsingHillClimbing(Algorithm, GFOCommonOptions):
    """Minimize a scalar function using the Repulsing Hill Climbing algorithm.

    This algorithm is a Python implementation of the Repulsing Hill Climbing algorithm
    through the gradient_free_optimizers package.

    The algorithm inherits from the Hill climbing which is a local search algorithm but
    always activates its methods to espace local optima.

    """

    epsilon: PositiveFloat = 0.03
    """The step-size of the hill climbing algorithm. If step_size is too large the newly
    selected positions will be at the edge of the search space.

    If its value is very low it might not find new positions.

    """

    distribution: Literal["normal", "laplace", "logistic", "gumbel"] = "normal"
    """The mathematical distribution the algorithm draws samples from.

    All available distributions are taken from the numpy-package.

    """

    n_neighbours: PositiveInt = 3
    """The number of positions the algorithm explores from its current position before
    setting its current position to the best of those neighbour positions."""

    repulsion_factor: PositiveFloat = 5
    """The algorithm increases the step size by multiplying it with the repulsion_factor
    for the next iteration. This way the algorithm escapes the region that does not
    offer better positions.

    .. math::
        \\epsilon = \\epsilon * {repulsion factor}

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        opt = gfo.RepulsingHillClimbingOptimizer
        optimizer = partial(
            opt,
            epsilon=self.epsilon,
            distribution=self.distribution,
            n_neighbours=self.n_neighbours,
            repulsion_factor=self.repulsion_factor,
        )
        res = _gfo_internal(
            common_options=self,
            problem=problem,
            x0=x0,
            optimizer=optimizer,
        )

        return res


@mark.minimizer(
    name="gfo_simulatedannealing",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
    is_global=False,
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
class GFOSimulatedAnnealing(Algorithm, GFOCommonOptions):
    """Minimize a scalar function using the Simulated Annealing algorithm.

    This algorithm is a Python implementation of Simulated Annealing through the
    gradient_free_optimizers package.

    Simulated annealing chooses its next possible position similar to hill climbing, but
    it accepts worse results with a probability that decreases with time. It simulates a
    temperature that decreases with each iteration, similar to a material cooling down.

    """

    epsilon: PositiveFloat = 0.03
    """The step-size of the algorithm.

    If step_size is too large the newly selected positions will be at the edge of the
    search space. If its value is very low it might not find new positions.

    """

    distribution: Literal["normal", "laplace", "logistic", "gumbel"] = "normal"
    """The mathematical distribution the algorithm draws samples from.

    All available distributions are taken from the numpy-package.

    """

    n_neighbours: PositiveInt = 3
    """The number of positions the algorithm explores from its current position before
    setting its current position to the best of those neighbour positions."""

    start_temp: PositiveFloat = 1
    """The start_temp is a factor for the probability p of accepting a worse position.

    .. math::
        p = \\exp^{-\\frac{score_{normalized}}{temp}}

    """

    annealing_rate: PositiveFloat = 0.97
    """Rate at which the temperatur-value of the algorithm decreases. An annealing rate
    above 1 increases the temperature over time.

    .. math::
        start\\_temp \\leftarrow start\\_temp * annealing\\_rate

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        opt = gfo.SimulatedAnnealingOptimizer
        optimizer = partial(
            opt,
            epsilon=self.epsilon,
            distribution=self.distribution,
            n_neighbours=self.n_neighbours,
            start_temp=self.start_temp,
            annealing_rate=self.annealing_rate,
        )
        res = _gfo_internal(
            common_options=self,
            problem=problem,
            x0=x0,
            optimizer=optimizer,
        )
        return res


@mark.minimizer(
    name="gfo_downhillsimplex",  # nelder_mead
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
    is_global=False,
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
class GFODownhillSimplex(Algorithm, GFOCommonOptions):
    """Minimize a scalar function using the Downhill Simplex algorithm.

    This algorithm is a Python implementation of the Downhill Simplex algorithm through
    the gradient_free_optimizers package.

    The Downhill simplex or Nelder mead algorithm works by grouping `number of
    dimensions + 1` positions into a simplex, which can explore the search-space by
    changing shape. The simplex changes shape by reflecting, expanding, contracting or
    shrinking via the alpha, gamma, beta or sigma parameters. It needs at least `number
    of dimensions + 1` initial positions to form a simplex in the search-space and the
    movement of the positions in the simplex are affected by each other.

    """

    simplex_reflection: PositiveFloat = 1
    """The reflection parameter of the simplex algorithm."""

    simplex_expansion: PositiveFloat = 2
    """The expansion parameter of the simplex algorithm."""

    simplex_contraction: PositiveFloat = 0.5
    """The contraction parameter of the simplex algorithm."""

    simplex_shrinking: PositiveFloat = 0.5
    """The shrinking parameter of the simplex algorithm."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        opt = gfo.DownhillSimplexOptimizer
        optimizer = partial(
            opt,
            alpha=self.simplex_reflection,
            gamma=self.simplex_expansion,
            beta=self.simplex_contraction,
            sigma=self.simplex_shrinking,
        )
        res = _gfo_internal(
            common_options=self,
            problem=problem,
            x0=x0,
            optimizer=optimizer,
        )
        return res


def _gfo_internal(
    common_options: GFOCommonOptions,
    problem: InternalOptimizationProblem,
    x0: NDArray[np.float64],
    optimizer: BaseOptimizer,
) -> InternalOptimizeResult:
    """Internal helper function.

    Define the search space and inital params, define the objective function and run
    optimization.

    """
    # Use common options from GFOCommonOptions
    common = common_options

    # set early stopping criterion
    early_stopping = {
        "n_iter_no_change": common.convergence_iter_noimprove,
        "tol_abs": common.convergence_ftol_abs,
        "tol_rel": common.convergence_ftol_rel,
    }

    # define search space, initial params, population, constraints
    opt = optimizer(
        search_space=_get_search_space_gfo(
            problem.bounds, common.n_grid_points, problem.converter
        ),
        initialize=_get_initialize_gfo(
            x0, common.n_init, common.warm_start, problem.converter
        ),
        constraints=_get_gfo_constraints(),
        random_state=common.seed,
    )

    # define objective function, negate to perform minimize
    def objective_function(para: dict[str, float]) -> float | NDArray[np.float64]:
        x = np.array(opt.conv.para2value(para))
        return -problem.fun(x)

    # negate in case of minimize
    stopping_funval = (
        -1 * common.stopping_funval if common.stopping_funval is not None else None
    )

    # run optimization
    opt.search(
        objective_function=objective_function,
        n_iter=common.stopping_maxiter,
        max_time=common.stopping_maxtime,
        max_score=stopping_funval,
        early_stopping=early_stopping,
        memory=common.caching,
        memory_warm_start=None,
        verbosity=common.verbosity,
    )

    return _process_result_gfo(opt)


def _get_search_space_gfo(
    bounds: InternalBounds, n_grid_points: PositiveInt | PyTree, converter: Converter
) -> dict[str, NDArray[np.float64]]:
    """Create search space.

    Args:
        bounds: Internal Bounds
        n_grid_points: number of grid points in each dimension
    Returns:
    dict: search_space dictionary

    """
    search_space = {}
    if bounds.lower is not None and bounds.upper is not None:
        dim = len(bounds.lower)
        upper = bounds.upper
        lower = bounds.lower

    if isinstance(n_grid_points, int):
        n_grid_points = [n_grid_points] * dim
    else:
        n_grid_points = converter.params_to_internal(n_grid_points)

    for i in range(dim):
        step = (upper[i] - lower[i]) / n_grid_points[i]
        search_space[f"x{i}"] = np.arange(lower[i], upper[i], step)
        # or
        # search_space[f"x{i}"] = np.linspace(lower[i], upper[i], n_grid_points[i])

    return search_space


def _process_result_gfo(opt: "BaseOptimizer") -> InternalOptimizeResult:
    """Process result.

    Args:
        opt: Optimizer instance after optimization run is complete

    Returns:
        InternalOptimizeResult: Internal optimization result.

    """
    res = InternalOptimizeResult(
        x=np.array(opt.best_value),
        fun=-opt.best_score,  # negate once again
        success=True,
        n_fun_evals=len(opt.eval_times),
        n_jac_evals=0,
        n_hess_evals=0,
        n_iterations=opt.n_iter_search,
    )

    return res


def _get_gfo_constraints() -> list[Any]:
    """Process constraints."""
    return []


def _get_initialize_gfo(
    x0: NDArray[np.float64],
    n_init: PositiveInt,
    warm_start: list[PyTree] | None,
    converter: Converter,
) -> dict[str, Any]:
    """Set initial params x0, additional start points for the optimization run or the
    initial_population.

    Args:
    x0: initial param

    Returns:
    dict: initialize dictionary with initial parameters set

    """
    init = _value2para(x0)
    x_list = [init]
    if warm_start is not None:
        internal_values = [converter.params_to_internal(x) for x in warm_start]
        warm_start = [_value2para(x) for x in internal_values]
        x_list += warm_start
    initialize = {"warm_start": x_list, "vertices": n_init}
    return initialize


def _value2para(x: NDArray[np.float64]) -> dict[str, float]:
    """Convert values to dict.

    Args:
        x: Array of parameter values

    Returns:
        dict: Dictionary of parameter values with key-value pair as { x{i} : x[i]}

    """
    para = {}
    for i in range(len(x)):
        para[f"x{i}"] = x[i]
    return para
