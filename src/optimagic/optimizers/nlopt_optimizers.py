"""Implement `nlopt` algorithms.

The documentation is heavily based on (nlopt documentation)[nlopt.readthedocs.io].

"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from optimagic import mark
from optimagic.config import IS_NLOPT_INSTALLED
from optimagic.optimization.algo_options import (
    CONVERGENCE_FTOL_ABS,
    CONVERGENCE_FTOL_REL,
    CONVERGENCE_XTOL_ABS,
    CONVERGENCE_XTOL_REL,
    STOPPING_MAXFUN,
    STOPPING_MAXFUN_GLOBAL,
)
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.internal_optimization_problem import (
    InternalOptimizationProblem,
)
from optimagic.parameters.nonlinear_constraints import (
    equality_as_inequality_constraints,
)
from optimagic.typing import (
    AggregationLevel,
    NonNegativeFloat,
    PositiveInt,
)

if IS_NLOPT_INSTALLED:
    import nlopt


@mark.minimizer(
    name="nlopt_bobyqa",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NLOPT_INSTALLED,
    is_global=False,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class NloptBOBYQA(Algorithm):
    convergence_xtol_rel: NonNegativeFloat = CONVERGENCE_XTOL_REL
    convergence_xtol_abs: NonNegativeFloat = CONVERGENCE_XTOL_ABS
    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_FTOL_ABS
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        res = _minimize_nlopt(
            problem=problem,
            x0=x0,
            is_global=self.algo_info.is_global,
            convergence_xtol_rel=self.convergence_xtol_rel,
            convergence_xtol_abs=self.convergence_xtol_abs,
            convergence_ftol_rel=self.convergence_ftol_rel,
            convergence_ftol_abs=self.convergence_ftol_abs,
            stopping_max_eval=self.stopping_maxfun,
            algorithm=nlopt.LN_BOBYQA,
        )

        return res


@mark.minimizer(
    name="nlopt_neldermead",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NLOPT_INSTALLED,
    is_global=False,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class NloptNelderMead(Algorithm):
    convergence_xtol_rel: NonNegativeFloat = CONVERGENCE_XTOL_REL
    convergence_xtol_abs: NonNegativeFloat = CONVERGENCE_XTOL_ABS
    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_FTOL_ABS
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        res = _minimize_nlopt(
            problem=problem,
            x0=x0,
            is_global=self.algo_info.is_global,
            convergence_xtol_rel=self.convergence_xtol_rel,
            convergence_xtol_abs=self.convergence_xtol_abs,
            convergence_ftol_abs=self.convergence_ftol_abs,
            stopping_max_eval=self.stopping_maxfun,
            algorithm=nlopt.LN_NELDERMEAD,
        )

        return res


@mark.minimizer(
    name="nlopt_praxis",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NLOPT_INSTALLED,
    is_global=False,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class NloptPRAXIS(Algorithm):
    convergence_xtol_rel: NonNegativeFloat = CONVERGENCE_XTOL_REL
    convergence_xtol_abs: NonNegativeFloat = CONVERGENCE_XTOL_ABS
    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_FTOL_ABS
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        res = _minimize_nlopt(
            problem=problem,
            x0=x0,
            is_global=self.algo_info.is_global,
            convergence_xtol_rel=self.convergence_xtol_rel,
            convergence_xtol_abs=self.convergence_xtol_abs,
            convergence_ftol_rel=self.convergence_ftol_rel,
            convergence_ftol_abs=self.convergence_ftol_abs,
            stopping_max_eval=self.stopping_maxfun,
            algorithm=nlopt.LN_PRAXIS,
        )

        return res


@mark.minimizer(
    name="nlopt_cobyla",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NLOPT_INSTALLED,
    is_global=False,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=True,
    disable_history=False,
)
@dataclass(frozen=True)
class NloptCOBYLA(Algorithm):
    convergence_xtol_rel: NonNegativeFloat = CONVERGENCE_XTOL_REL
    convergence_xtol_abs: NonNegativeFloat = CONVERGENCE_XTOL_ABS
    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_FTOL_ABS
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        res = _minimize_nlopt(
            problem=problem,
            x0=x0,
            is_global=self.algo_info.is_global,
            convergence_xtol_rel=self.convergence_xtol_rel,
            convergence_xtol_abs=self.convergence_xtol_abs,
            convergence_ftol_rel=self.convergence_ftol_rel,
            convergence_ftol_abs=self.convergence_ftol_abs,
            stopping_max_eval=self.stopping_maxfun,
            algorithm=nlopt.LN_COBYLA,
        )

        return res


@mark.minimizer(
    name="nlopt_sbplx",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NLOPT_INSTALLED,
    is_global=False,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class NloptSbplx(Algorithm):
    convergence_xtol_rel: NonNegativeFloat = CONVERGENCE_XTOL_REL
    convergence_xtol_abs: NonNegativeFloat = CONVERGENCE_XTOL_ABS
    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_FTOL_ABS
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        res = _minimize_nlopt(
            problem=problem,
            x0=x0,
            is_global=self.algo_info.is_global,
            convergence_xtol_rel=self.convergence_xtol_rel,
            convergence_xtol_abs=self.convergence_xtol_abs,
            convergence_ftol_rel=self.convergence_ftol_rel,
            convergence_ftol_abs=self.convergence_ftol_abs,
            stopping_max_eval=self.stopping_maxfun,
            algorithm=nlopt.LN_SBPLX,
        )

        return res


@mark.minimizer(
    name="nlopt_newuoa",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NLOPT_INSTALLED,
    is_global=False,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class NloptNEWUOA(Algorithm):
    convergence_xtol_rel: NonNegativeFloat = CONVERGENCE_XTOL_REL
    convergence_xtol_abs: NonNegativeFloat = CONVERGENCE_XTOL_ABS
    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_FTOL_ABS
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if problem.bounds.lower is None or problem.bounds.upper is None:
            algo = nlopt.LN_NEWUOA
        elif np.any(np.isfinite(problem.bounds.lower)) or np.any(
            np.isfinite(problem.bounds.upper)
        ):
            algo = nlopt.LN_NEWUOA_BOUND
        else:
            algo = nlopt.LN_NEWUOA
        res = _minimize_nlopt(
            problem=problem,
            x0=x0,
            is_global=self.algo_info.is_global,
            convergence_xtol_rel=self.convergence_xtol_rel,
            convergence_xtol_abs=self.convergence_xtol_abs,
            convergence_ftol_rel=self.convergence_ftol_rel,
            convergence_ftol_abs=self.convergence_ftol_abs,
            stopping_max_eval=self.stopping_maxfun,
            algorithm=algo,
        )

        return res


@mark.minimizer(
    name="nlopt_tnewton",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NLOPT_INSTALLED,
    is_global=False,
    needs_jac=True,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class NloptTNewton(Algorithm):
    convergence_xtol_rel: NonNegativeFloat = CONVERGENCE_XTOL_REL
    convergence_xtol_abs: NonNegativeFloat = CONVERGENCE_XTOL_ABS
    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_FTOL_ABS
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        res = _minimize_nlopt(
            problem=problem,
            x0=x0,
            is_global=self.algo_info.is_global,
            convergence_xtol_rel=self.convergence_xtol_rel,
            convergence_xtol_abs=self.convergence_xtol_abs,
            convergence_ftol_rel=self.convergence_ftol_rel,
            convergence_ftol_abs=self.convergence_ftol_abs,
            stopping_max_eval=self.stopping_maxfun,
            algorithm=nlopt.LD_TNEWTON,
        )

        return res


@mark.minimizer(
    name="nlopt_lbfgsb",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NLOPT_INSTALLED,
    is_global=False,
    needs_jac=True,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class NloptLBFGSB(Algorithm):
    convergence_xtol_rel: NonNegativeFloat = CONVERGENCE_XTOL_REL
    convergence_xtol_abs: NonNegativeFloat = CONVERGENCE_XTOL_ABS
    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_FTOL_ABS
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        res = _minimize_nlopt(
            problem=problem,
            x0=x0,
            is_global=self.algo_info.is_global,
            convergence_xtol_rel=self.convergence_xtol_rel,
            convergence_xtol_abs=self.convergence_xtol_abs,
            convergence_ftol_rel=self.convergence_ftol_rel,
            convergence_ftol_abs=self.convergence_ftol_abs,
            stopping_max_eval=self.stopping_maxfun,
            algorithm=nlopt.LD_LBFGS,
        )

        return res


@mark.minimizer(
    name="nlopt_ccsaq",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NLOPT_INSTALLED,
    is_global=False,
    needs_jac=True,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class NloptCCSAQ(Algorithm):
    convergence_xtol_rel: NonNegativeFloat = CONVERGENCE_XTOL_REL
    convergence_xtol_abs: NonNegativeFloat = CONVERGENCE_XTOL_ABS
    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_FTOL_ABS
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        res = _minimize_nlopt(
            problem=problem,
            x0=x0,
            is_global=self.algo_info.is_global,
            convergence_xtol_rel=self.convergence_xtol_rel,
            convergence_xtol_abs=self.convergence_xtol_abs,
            convergence_ftol_rel=self.convergence_ftol_rel,
            convergence_ftol_abs=self.convergence_ftol_abs,
            stopping_max_eval=self.stopping_maxfun,
            algorithm=nlopt.LD_CCSAQ,
        )

        return res


@mark.minimizer(
    name="nlopt_mma",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NLOPT_INSTALLED,
    is_global=False,
    needs_jac=True,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=True,
    disable_history=False,
)
@dataclass(frozen=True)
class NloptMMA(Algorithm):
    convergence_xtol_rel: NonNegativeFloat = CONVERGENCE_XTOL_REL
    convergence_xtol_abs: NonNegativeFloat = CONVERGENCE_XTOL_ABS
    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_FTOL_ABS
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        nonlinear_constraints = equality_as_inequality_constraints(
            problem.nonlinear_constraints
        )

        res = _minimize_nlopt(
            problem=problem,
            x0=x0,
            is_global=self.algo_info.is_global,
            convergence_xtol_rel=self.convergence_xtol_rel,
            convergence_xtol_abs=self.convergence_xtol_abs,
            convergence_ftol_rel=self.convergence_ftol_rel,
            convergence_ftol_abs=self.convergence_ftol_abs,
            stopping_max_eval=self.stopping_maxfun,
            algorithm=nlopt.LD_MMA,
            nonlinear_constraints=nonlinear_constraints,
        )

        return res


@mark.minimizer(
    name="nlopt_var",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NLOPT_INSTALLED,
    is_global=False,
    needs_jac=True,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class NloptVAR(Algorithm):
    convergence_xtol_rel: NonNegativeFloat = CONVERGENCE_XTOL_REL
    convergence_xtol_abs: NonNegativeFloat = CONVERGENCE_XTOL_ABS
    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_FTOL_ABS
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN
    rank_1_update: bool = True

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if self.rank_1_update:
            algo = nlopt.LD_VAR1
        else:
            algo = nlopt.LD_VAR2
        res = _minimize_nlopt(
            problem=problem,
            x0=x0,
            is_global=self.algo_info.is_global,
            convergence_xtol_rel=self.convergence_xtol_rel,
            convergence_xtol_abs=self.convergence_xtol_abs,
            convergence_ftol_rel=self.convergence_ftol_rel,
            convergence_ftol_abs=self.convergence_ftol_abs,
            stopping_max_eval=self.stopping_maxfun,
            algorithm=algo,
        )

        return res


@mark.minimizer(
    name="nlopt_slsqp",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NLOPT_INSTALLED,
    is_global=False,
    needs_jac=True,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=True,
    disable_history=False,
)
@dataclass(frozen=True)
class NloptSLSQP(Algorithm):
    convergence_xtol_rel: NonNegativeFloat = CONVERGENCE_XTOL_REL
    convergence_xtol_abs: NonNegativeFloat = CONVERGENCE_XTOL_ABS
    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_FTOL_ABS
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        res = _minimize_nlopt(
            problem=problem,
            x0=x0,
            is_global=self.algo_info.is_global,
            convergence_xtol_rel=self.convergence_xtol_rel,
            convergence_xtol_abs=self.convergence_xtol_abs,
            convergence_ftol_rel=self.convergence_ftol_rel,
            convergence_ftol_abs=self.convergence_ftol_abs,
            stopping_max_eval=self.stopping_maxfun,
            algorithm=nlopt.LD_SLSQP,
        )

        return res


@mark.minimizer(
    name="nlopt_direct",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NLOPT_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class NloptDirect(Algorithm):
    convergence_xtol_rel: NonNegativeFloat = CONVERGENCE_XTOL_REL
    convergence_xtol_abs: NonNegativeFloat = CONVERGENCE_XTOL_ABS
    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_FTOL_ABS
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL
    locally_biased: bool = False
    random_search: bool = False
    unscaled_bounds: bool = False

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if (
            not self.locally_biased
            and not self.random_search
            and not self.unscaled_bounds
        ):
            algo = nlopt.GN_DIRECT
        elif (
            self.locally_biased and not self.random_search and not self.unscaled_bounds
        ):
            algo = nlopt.GN_DIRECT_L
        elif self.locally_biased and not self.random_search and self.unscaled_bounds:
            algo = nlopt.GN_DIRECT_L_NOSCAL
        elif self.locally_biased and self.random_search and not self.unscaled_bounds:
            algo = nlopt.GN_DIRECT_L_RAND
        elif self.locally_biased and self.random_search and self.unscaled_bounds:
            algo = nlopt.GN_DIRECT_L_RAND_NOSCAL
        elif (
            not self.locally_biased and not self.random_search and self.unscaled_bounds
        ):
            algo = nlopt.GN_DIRECT_NOSCAL
        res = _minimize_nlopt(
            problem=problem,
            x0=x0,
            is_global=self.algo_info.is_global,
            convergence_xtol_rel=self.convergence_xtol_rel,
            convergence_xtol_abs=self.convergence_xtol_abs,
            convergence_ftol_rel=self.convergence_ftol_rel,
            convergence_ftol_abs=self.convergence_ftol_abs,
            stopping_max_eval=self.stopping_maxfun,
            algorithm=algo,
        )
        return res


@mark.minimizer(
    name="nlopt_esch",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NLOPT_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class NloptESCH(Algorithm):
    convergence_xtol_rel: NonNegativeFloat = CONVERGENCE_XTOL_REL
    convergence_xtol_abs: NonNegativeFloat = CONVERGENCE_XTOL_ABS
    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_FTOL_ABS
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        res = _minimize_nlopt(
            problem=problem,
            x0=x0,
            is_global=self.algo_info.is_global,
            convergence_xtol_rel=self.convergence_xtol_rel,
            convergence_xtol_abs=self.convergence_xtol_abs,
            convergence_ftol_rel=self.convergence_ftol_rel,
            convergence_ftol_abs=self.convergence_ftol_abs,
            stopping_max_eval=self.stopping_maxfun,
            algorithm=nlopt.GN_ESCH,
        )

        return res


@mark.minimizer(
    name="nlopt_isres",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NLOPT_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=True,
    disable_history=False,
)
@dataclass(frozen=True)
class NloptISRES(Algorithm):
    convergence_xtol_rel: NonNegativeFloat = CONVERGENCE_XTOL_REL
    convergence_xtol_abs: NonNegativeFloat = CONVERGENCE_XTOL_ABS
    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_FTOL_ABS
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        res = _minimize_nlopt(
            problem=problem,
            x0=x0,
            is_global=self.algo_info.is_global,
            convergence_xtol_rel=self.convergence_xtol_rel,
            convergence_xtol_abs=self.convergence_xtol_abs,
            convergence_ftol_rel=self.convergence_ftol_rel,
            convergence_ftol_abs=self.convergence_ftol_abs,
            stopping_max_eval=self.stopping_maxfun,
            algorithm=nlopt.GN_ISRES,
        )

        return res


@mark.minimizer(
    name="nlopt_crs2_lm",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NLOPT_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class NloptCRS2LM(Algorithm):
    convergence_xtol_rel: NonNegativeFloat = CONVERGENCE_XTOL_REL
    convergence_xtol_abs: NonNegativeFloat = CONVERGENCE_XTOL_ABS
    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_FTOL_ABS
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL
    population_size: PositiveInt | None = None

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if self.population_size is None:
            population_size = 10 * (len(x0) + 1)
        else:
            population_size = self.population_size
        res = _minimize_nlopt(
            problem=problem,
            x0=x0,
            is_global=self.algo_info.is_global,
            convergence_xtol_rel=self.convergence_xtol_rel,
            convergence_xtol_abs=self.convergence_xtol_abs,
            convergence_ftol_rel=self.convergence_ftol_rel,
            convergence_ftol_abs=self.convergence_ftol_abs,
            stopping_max_eval=self.stopping_maxfun,
            algorithm=nlopt.GN_CRS2_LM,
            population_size=population_size,
        )

        return res


def _minimize_nlopt(
    problem,
    x0,
    algorithm,
    is_global,
    *,
    convergence_xtol_rel=None,
    convergence_xtol_abs=None,
    convergence_ftol_rel=None,
    convergence_ftol_abs=None,
    stopping_max_eval=None,
    population_size=None,
    nonlinear_constraints=None,
):
    """Run actual nlopt optimization argument, set relevant attributes."""

    def func(x, grad):
        if grad.size > 0:
            fun, jac = problem.fun_and_jac(x)
            grad[:] = jac
        else:
            fun = problem.fun(x)
        return fun

    if nonlinear_constraints is None:
        nonlinear_constraints = problem.nonlinear_constraints
    opt = nlopt.opt(algorithm, x0.shape[0])
    if convergence_ftol_rel is not None:
        opt.set_ftol_rel(convergence_ftol_rel)
    if convergence_ftol_abs is not None:
        opt.set_ftol_abs(convergence_ftol_abs)
    if convergence_xtol_rel is not None:
        opt.set_xtol_rel(convergence_xtol_rel)
    if convergence_xtol_abs is not None:
        opt.set_xtol_abs(convergence_xtol_abs)
    if problem.bounds.lower is not None:
        opt.set_lower_bounds(problem.bounds.lower)
    if problem.bounds.upper is not None:
        opt.set_upper_bounds(problem.bounds.upper)
    if stopping_max_eval is not None:
        opt.set_maxeval(stopping_max_eval)
    if population_size is not None:
        opt.set_population(population_size)
    if nonlinear_constraints:
        for constr in _get_nlopt_constraints(nonlinear_constraints, filter_type="eq"):
            opt.add_equality_mconstraint(constr["fun"], constr["tol"])
        for constr in _get_nlopt_constraints(nonlinear_constraints, filter_type="ineq"):
            opt.add_inequality_mconstraint(constr["fun"], constr["tol"])
    opt.set_min_objective(func)
    solution_x = opt.optimize(x0)
    return _process_nlopt_results(opt, solution_x, is_global)


def _process_nlopt_results(nlopt_obj, solution_x, is_global):
    messages = {
        1: "Convergence achieved ",
        2: (
            "Optimizer stopped because maximum value of criterion function was reached"
        ),
        3: (
            "Optimizer stopped because convergence_ftol_rel or "
            "convergence_ftol_abs was reached"
        ),
        4: (
            "Optimizer stopped because convergence_xtol_rel or "
            "convergence_xtol_abs was reached"
        ),
        5: "Optimizer stopped because max_criterion_evaluations was reached",
        6: "Optimizer stopped because max running time was reached",
        -1: "Optimizer failed",
        -2: "Invalid arguments were passed",
        -3: "Memory error",
        -4: "Halted because roundoff errors limited progress",
        -5: "Halted because of user specified forced stop",
    }
    success = nlopt_obj.last_optimize_result() in [1, 2, 3, 4]
    if is_global and not success:
        success = None
    processed = InternalOptimizeResult(
        x=solution_x,
        fun=nlopt_obj.last_optimum_value(),
        n_fun_evals=nlopt_obj.get_numevals(),
        success=success,
        message=messages[nlopt_obj.last_optimize_result()],
    )

    return processed


def _get_nlopt_constraints(constraints, filter_type):
    """Transform internal nonlinear constraints to NLOPT readable format."""
    filtered = [c for c in constraints if c["type"] == filter_type]
    nlopt_constraints = [_internal_to_nlopt_constaint(c) for c in filtered]
    return nlopt_constraints


def _internal_to_nlopt_constaint(c):
    """Sign flip description:

    In optimagic, inequality constraints are internally defined as g(x) >= 0. NLOPT uses
    h(x) <= 0, which is why we need to flip the sign.

    """
    tol = c["tol"]
    if np.isscalar(tol):
        tol = np.tile(tol, c["n_constr"])

    def _constraint(result, x, grad):
        result[:] = -c["fun"](x)  # see docstring for sign flip
        if grad.size > 0:
            grad[:] = -c["jac"](x)  # see docstring for sign flip

    new_constr = {
        "fun": _constraint,
        "tol": tol,
    }
    return new_constr
