"""Implementation of parallelosation of Nelder-Mead algorithm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, cast

import numpy as np
from numpy.typing import NDArray

from optimagic import mark
from optimagic.batch_evaluators import process_batch_evaluator
from optimagic.optimization.algo_options import (
    CONVERGENCE_SECOND_BEST_FTOL_ABS,
    CONVERGENCE_SECOND_BEST_XTOL_ABS,
    STOPPING_MAXITER,
)
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.internal_optimization_problem import (
    InternalOptimizationProblem,
)
from optimagic.typing import AggregationLevel, NonNegativeFloat, PositiveInt

InitSimplexLiteral = Literal["pfeffer", "nash", "gao_han", "varadhan_borchers"]
InitSimplexCallable = Callable[[NDArray[np.float64]], NDArray[np.float64]]
from optimagic.typing import BatchEvaluator, BatchEvaluatorLiteral


@mark.minimizer(
    name="neldermead_parallel",
    solver_type=AggregationLevel.SCALAR,
    is_available=True,
    is_global=False,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=False,
    supports_parallelism=True,
    supports_bounds=False,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=True,
)
@dataclass(frozen=True)
class NelderMeadParallel(Algorithm):
    """Minimize a scalar function using a parallel Nelder-Mead algorithm.

    This is a parallel Nelder-Mead algorithm following :cite:`LeeWiswall2007`,
    which generalizes the classic simplex algorithm of Nelder and Mead
    (:cite:`Nelder1965`) to parallel processors. In each iteration, instead of
    reflecting only the worst vertex of the simplex, the p worst vertices are
    reflected (and possibly expanded or contracted) simultaneously, so that up to
    p criterion evaluations can be carried out in parallel. With ``n_cores=1``,
    the algorithm behaves like a standard sequential Nelder-Mead algorithm.

    As a direct search method, it does not require derivatives of the objective
    function and can handle a moderate amount of noise or non-smoothness. It is a
    local optimizer for scalar objective functions and does not support bounds or
    constraints. Parallelization is most beneficial when a single criterion
    evaluation is expensive and the number of parameters is large relative to the
    number of available cores.

    The algorithm was implemented in optimagic by Jacek Barszczewski.

    .. note::
        This is a pure-Python implementation within optimagic. It is currently
        considered experimental.

    """

    init_simplex_method: InitSimplexLiteral | InitSimplexCallable = "gao_han"
    """Method to create the initial simplex.

    Either the name of one of the implemented methods, i.e. "pfeffer" (due to L.
    Pfeffer at Stanford; the default in MATLAB's fminsearch and SciPy), "nash"
    (:cite:`Nash1990`; the default in R's optim), "gao_han" (:cite:`Gao2012`) or
    "varadhan_borchers" (:cite:`Varadhan2016`), or a callable that takes the
    initial parameter vector as argument and returns the initial simplex as an
    array of shape (j + 1, j), where j is the number of parameters. The default is
    "gao_han". See :cite:`Wessing2019` for a discussion of the importance of the
    choice of the initial simplex.

    """

    n_cores: PositiveInt = 1
    """Degree of parallelization.

    The default is 1 (no parallelization). Values larger than or equal to the
    number of parameters are reduced to the number of parameters minus 1.

    """

    adaptive: bool = True
    """Adjust the parameters of the Nelder-Mead algorithm to the dimension of the
    problem, following :cite:`Gao2012`.

    The default is True.

    """

    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    """Maximum number of algorithm iterations."""

    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_SECOND_BEST_FTOL_ABS
    """Maximal absolute difference between the function value at the best vertex of
    the simplex and the function values at the remaining vertices.

    Convergence requires that this criterion and ``convergence_xtol_abs`` are
    satisfied at the same time.

    """

    convergence_xtol_abs: NonNegativeFloat = CONVERGENCE_SECOND_BEST_XTOL_ABS
    """Maximal absolute distance between the best vertex of the simplex and the
    remaining vertices.

    Convergence requires that this criterion and ``convergence_ftol_abs`` are
    satisfied at the same time.

    """

    batch_evaluator: BatchEvaluator | BatchEvaluatorLiteral = "joblib"
    """Batch evaluator used for parallel function evaluations.

    See :ref:`batch_evaluators` for details. The default is "joblib".

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        raw = neldermead_parallel(
            criterion=cast(
                Callable[[NDArray[np.float64]], float],
                problem.fun,
            ),
            x=x0,
            init_simplex_method=self.init_simplex_method,
            n_cores=self.n_cores,
            adaptive=self.adaptive,
            stopping_maxiter=self.stopping_maxiter,
            convergence_ftol_abs=self.convergence_ftol_abs,
            convergence_xtol_abs=self.convergence_xtol_abs,
            batch_evaluator=self.batch_evaluator,
        )

        res = InternalOptimizeResult(
            x=raw["solution_x"],
            fun=raw["solution_criterion"],
            n_iterations=raw["n_iterations"],
            success=raw["success"],
            message=raw["reached_convergence_criterion"],
        )

        return res


def neldermead_parallel(
    criterion,
    x,
    *,
    init_simplex_method="gao_han",
    n_cores=1,
    adaptive=True,
    stopping_maxiter=STOPPING_MAXITER,
    convergence_ftol_abs=CONVERGENCE_SECOND_BEST_FTOL_ABS,
    convergence_xtol_abs=CONVERGENCE_SECOND_BEST_XTOL_ABS,
    batch_evaluator="joblib",
):
    if x.ndim >= 1:
        x = x.ravel()  # check if the vector of initial values is one-dimensional

    j = len(x)  # size of the parameter vector

    if n_cores <= 1:
        p = 1  # if number of cores is nonpositive, set it to 1
    else:
        if n_cores >= j:  # number of parallelisation cannot be bigger than
            # the number of parameters minus 1
            p = int(j - 1)
        else:
            p = int(n_cores)

    # set parameters of Nelder-Mead algorithm
    # for a discussion about Nlder-Mead parameters see Gao F., Han L., Implementing the
    # Nelder-Mead siplex algorithm with adaptive parameters, Computational Optimization
    # and Applications, 2012
    alpha, gamma, beta, tau = _init_algo_params(adaptive, j)

    # construct initial simplex using one of feasible methods
    # see Wssing, Simon, Proper initialization is crucial for
    # the Nelder-Mead simplex search, Optimization Letters, 2019
    # for a discussion about the choice of initialization

    if not callable(init_simplex_method):
        s = globals()["_" + init_simplex_method](x)
    else:
        s = init_simplex_method(x)

    batch_evaluator = process_batch_evaluator(batch_evaluator)

    # calculate criterion values for the initial simplex
    f_s = np.array(batch_evaluator(func=criterion, arguments=s, n_cores=n_cores))[
        :, None
    ]

    # parallelized function
    def func_parallel(args):
        criterion, s_j, s_j_r, f_s_0, f_s_j, f_s_j_1, m = args  # read arguments

        f_s_j_r = criterion(
            s_j_r
        )  # calculate value of the criterion at the reflection point

        if f_s_j_r < f_s_0:  # if the reflection point is better than the best point
            s_j_e = m + gamma * (s_j_r - m)  # calculate expansion point
            f_s_j_e = criterion(
                s_j_e
            )  # calculate value of the criterion at the expansion point

            if f_s_j_e < f_s_0:  # if the expansion point is better than the best point
                return np.hstack(
                    [s_j_e, f_s_j_e, 0]
                )  # return the expansion point as a new point

            else:  # if the expansion point is worse than the best point
                return np.hstack(
                    [s_j_r, f_s_j_r, 0]
                )  # return the reflection point as a new point

        elif (
            f_s_j_r < f_s_j_1
        ):  # if reflection point is better than the next worst point
            return np.hstack(
                [s_j_r, f_s_j_r, 0]
            )  # return reflection point as a new point

        else:  # if the reflection point is worse than the next worst point
            if (
                f_s_j_r < f_s_j
            ):  # if value of the criterion at reflection point is better than
                # value of the criterion at initial point
                s_j_c = m + beta * (s_j_r - m)  # calculate outside contraction point
            else:
                s_j_c = m - beta * (s_j_r - m)  # calculate inside contraction point

            f_s_j_c = criterion(
                s_j_c
            )  # calculate a value of the criterion at contraction point

            if f_s_j_c < np.minimum(
                f_s_j, f_s_j_r
            ):  # if ta value of the criterion at contraction point is better
                # than original and refrelction point

                return np.hstack(
                    [s_j_c, f_s_j_c, 0]
                )  # return contraction point as as new point

            else:
                if f_s_j_r < f_s_j:
                    return np.hstack(
                        [s_j_r, f_s_j_r, 1]
                    )  # return reflection point as a new point

                else:  # if value of the criterion at contraction point is worse
                    # than the value uf the criterion at the reflection
                    # and the initial points
                    return np.hstack(
                        [s_j, f_s_j, 1]
                    )  # return the old point as a new point

    optimal = False  # optmisation condition, if True stop the algorithem
    iterations = 0  # number of criterion evaluations

    while not optimal:
        iterations += 1  # new iteration

        # sort points and arguments increasing
        row = np.argsort(f_s.ravel())
        s = np.take(s, row, axis=0)
        f_s = np.take(f_s, row, axis=0)

        # calculate centroid
        m = (s[:-p, :].sum(axis=0)) / (j - p + 1)

        # calculate reflaction points
        s_j_r = m + alpha * (m - s[-p:, :])

        # calculate new points of simplex
        s[-p:, :], f_s[-p:, :], shrink_count = np.split(
            np.vstack(
                batch_evaluator(
                    func=func_parallel,
                    arguments=tuple(
                        (
                            criterion,
                            s[j + 1 - p + i, :],
                            s_j_r[i, :],
                            f_s[0, :],
                            f_s[j + 1 - p + i, :],
                            f_s[j - p + i, :],
                            m,
                        )
                        for i in range(p)
                    ),
                    n_cores=p,
                )
            ),
            [-2, -1],
            axis=1,
        )

        # shrink simplex if there is no improvement in every process
        if shrink_count.sum() == p:
            s = (
                tau * s[0:1, :] + (1 - tau) * s
            )  # new simplex is a linear combination of the best point
            # and remaining points
            # evaluate function at new simplex
            f_s = np.array(
                batch_evaluator(
                    func=criterion,
                    arguments=s,
                    n_cores=n_cores,
                )
            )[:, None]

        # termination criteria
        if (
            np.max(np.abs(f_s[0, :] - f_s[1:, :])) <= convergence_ftol_abs
            and np.max(np.abs(s[0, :] - s[1:,])) <= convergence_xtol_abs
        ):
            optimal = True
            converge = True
            reason_to_stop = "Termination codition satisfied"
        elif (
            iterations >= stopping_maxiter
        ):  # if maximum amount of iteration is exceeded
            optimal = True
            converge = False
            reason_to_stop = "Maximum number of interation exceeded"

    # save results
    result = {
        "solution_x": s[np.nonzero(f_s == f_s.min())[0][0], :],
        "solution_criterion": f_s.min(),
        "n_iterations": iterations,
        "success": converge,
        "reached_convergence_criterion": reason_to_stop,
    }
    return result


# set parameters of Nelder-Mead algorithm
# for a discussion about Nlder-Mead parameters see Gao F., Han L., Implementing the
# Nelder-Mead siplex algorithm with adaptive parameters, Computational Optimization
# and Applications, 2012
def _init_algo_params(adaptive, j):
    if adaptive:
        # algorithem parameters alla Gao-Han (adaptive)
        return (
            1,
            1 + 2 / j,
            0.75 - 1 / (2 * j),
            1 - 1 / j,
        )
    else:
        # standard setting of Nelder-Mead
        return (
            1,
            2,
            0.5,
            0.5,
        )


# initial structure of the simplex
def _init_simplex(x):
    s = np.vstack(
        [
            x,
        ]
        * (len(x) + 1)
    ).astype(np.float64)

    return s


# initilize due to L. Pfeffer at Standford (Matlab fminsearch and SciPy default option)
def _pfeffer(x):
    s = _init_simplex(x)

    # method parameters
    c_p = 1.05

    # initial simplex
    np.fill_diagonal(s[1:, :], x * c_p * (x != 0) + 0.00025 * (x == 0))

    return s


# adopted from Nash (R default option)
# see Nash, J.C.: Compact numerical methods for computers: linear algebra and
# function minimisation, 2nd edn. Adam Hilger Ltd., Bristol (1990) for details
def _nash(x):
    s = _init_simplex(x)

    # method parameters
    c_n = 0.1

    # initial simplex
    np.fill_diagonal(s[1:, :], (x != 0) * (np.max(x) * c_n + x) + c_n * (x == 0))
    return s


# adopted from Gao F., Han L., Implementing the
# Nelder-Mead siplex algorithm with adaptive parameters, Computational Optimizatio
def _gao_han(x):
    s = _init_simplex(x)

    # method parameters
    c_h = np.minimum(np.maximum(np.max(x), 1), 10)
    j = len(x)

    # initial simplex
    s = (
        s
        + np.vstack(
            [
                np.array([[(1 - (j + 1) ** 0.5) / j]]) * np.ones([1, j]),
                np.eye(j),
            ]
        )
        * c_h
    )

    return s


# adopted by Varadhan and Borchers for the R package dfoptim
# see Varadhan, R., Borchers, H.W.: Dfoptim: derivative-free optimization (2016).
# https://CRAN.R-project. org/package=dfoptim. R package version 2016.7-1 for details
def _varadhan_borchers(x):
    s = _init_simplex(x)

    # method parameters
    j = len(x)
    c_s = np.maximum(1, ((x**2).sum()) ** 0.5)
    beta1 = c_s / (j * 2**0.5) * ((j + 1) ** 0.5 + j - 1)
    beta2 = c_s / (j * 2**0.5) * ((j + 1) ** 0.5 - 1)

    # initial simplex
    s[1:, :] = s[1:, :] + np.full([j, j], beta2) + np.eye(j) * (beta1 - beta2)

    return s
