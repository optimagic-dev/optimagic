"""Implement ensmallen optimizers."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from optimagic import mark
from optimagic.config import IS_PYENSMALLEN_INSTALLED
from optimagic.optimization.algo_options import (
    CONVERGENCE_FTOL_REL,
    CONVERGENCE_GTOL_ABS,
    LIMITED_MEMORY_STORAGE_LENGTH,
    MAX_LINE_SEARCH_STEPS,
    STOPPING_MAXITER,
)
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.internal_optimization_problem import (
    InternalOptimizationProblem,
)
from optimagic.typing import AggregationLevel, NonNegativeFloat, PositiveInt

if IS_PYENSMALLEN_INSTALLED:
    import pyensmallen as pye

MIN_LINE_SEARCH_STEPS = 1e-20
"""The minimum step of the line search."""
MAX_LINE_SEARCH_TRIALS = 50
"""The maximum number of trials for the line search (before giving up)."""
ARMIJO_CONSTANT = 1e-4
"""Controls the accuracy of the line search routine for determining the Armijo
condition."""
WOLFE_CONDITION = 0.9
"""Parameter for detecting the Wolfe condition."""

STEP_SIZE = 0.001
"""Step size for each iteration."""
BATCH_SIZE = 32
"""Step size for each iteration."""
EXP_DECAY_RATE_FOR_FIRST_MOMENT = 0.9
"""Exponential decay rate for the first moment estimates."""
EXP_DECAY_RATE_FOR_WEIGHTED_INF_NORM = 0.999
"""Exponential decay rate for the first moment estimates."""


@mark.minimizer(
    name="ensmallen_lbfgs",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_PYENSMALLEN_INSTALLED,
    is_global=False,
    needs_jac=True,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class EnsmallenLBFGS(Algorithm):
    limited_memory_storage_length: PositiveInt = LIMITED_MEMORY_STORAGE_LENGTH
    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    armijo_constant: NonNegativeFloat = ARMIJO_CONSTANT  # needs review
    wolfe_condition: NonNegativeFloat = WOLFE_CONDITION  # needs review
    convergence_gtol_abs: NonNegativeFloat = CONVERGENCE_GTOL_ABS
    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    max_line_search_trials: PositiveInt = MAX_LINE_SEARCH_TRIALS
    min_step_for_line_search: NonNegativeFloat = MIN_LINE_SEARCH_STEPS
    max_step_for_line_search: NonNegativeFloat = MAX_LINE_SEARCH_STEPS

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        optimizer = pye.L_BFGS(
            numBasis=self.limited_memory_storage_length,
            maxIterations=self.stopping_maxiter,
            armijoConstant=self.armijo_constant,
            wolfe=self.wolfe_condition,
            minGradientNorm=self.convergence_gtol_abs,
            factr=self.convergence_ftol_rel,
            maxLineSearchTrials=self.max_line_search_trials,
            minStep=self.min_step_for_line_search,
            maxStep=self.max_step_for_line_search,
        )

        def objective_function(
            x: NDArray[np.float64], grad: NDArray[np.float64]
        ) -> np.float64:
            fun, jac = problem.fun_and_jac(x)
            grad[:] = jac
            return np.float64(fun)

        best_x = optimizer.optimize(objective_function, x0, report)

        res = InternalOptimizeResult(
            x=best_x,
            fun=problem.fun(best_x),
        )

        return res
