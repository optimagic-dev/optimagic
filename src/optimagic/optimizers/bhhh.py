"""Implement the Berndt-Hall-Hall-Hausman (BHHH) algorithm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from optimagic import mark
from optimagic.optimization.algo_options import (
    CONVERGENCE_FTOL_ABS,
    STOPPING_MAXFUN,
    STOPPING_MAXITER,
)
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.internal_optimization_problem import (
    InternalOptimizationProblem,
)
from optimagic.optimizers._bhhh.bhhh_loop import bhhh_internal
from optimagic.optimizers._bhhh.line_search import LineSearch
from optimagic.typing import AggregationLevel, NonNegativeFloat, PositiveInt

CONVERGENCE_DECREMENT_TOL = 1e-8
"""Default tolerance for the decrement; about 1e-4 robust standard errors."""


@mark.minimizer(
    name="bhhh",
    solver_type=AggregationLevel.LIKELIHOOD,
    is_available=True,
    is_global=False,
    needs_jac=True,
    needs_hess=False,
    needs_bounds=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class BHHH(Algorithm):
    r"""Maximize a likelihood with the Berndt-Hall-Hall-Hausman (BHHH) algorithm.

    BHHH (:cite:`Berndt1974`) is a Newton-type method for maximum likelihood
    estimation. It replaces the Hessian of the log-likelihood by the outer product of
    gradients (OPG) of the likelihood contributions. By the information matrix equality
    (:cite:`Halbert1982`), the OPG approximates the negative Hessian close to the
    maximum of a correctly specified likelihood. BHHH only needs first derivatives of
    the contributions, is invariant to linear reparametrizations of the parameters and
    typically converges in a few dozen iterations.

    **How it works.** Internally, optimagic minimizes the sum of contributions
    :math:`F(x) = \sum_i f_i(x)`, i.e. :math:`f_i = -\ell_i` when you maximize a
    log-likelihood. Each iteration

    1. evaluates the Jacobian :math:`J` (n x k) of the contributions at the current
       parameters and forms the gradient :math:`g = J'1` and the OPG :math:`B = J'J`;
    2. computes the BHHH step :math:`p = -B^{-1} g`. With bounds, :math:`p` minimizes
       the quadratic model :math:`g'p + p'Bp/2` over all feasible steps
       (:cite:`Bertsekas1982`);
    3. chooses a step length along :math:`p` with a line search (see
       :class:`~optimagic.optimizers.bhhh.LineSearch`).

    Only the objective is evaluated at trial points. The derivatives are evaluated
    once per accepted point.

    Rank decisions use the OPG after scaling its diagonal to one. Eigenvalues of the
    scaled OPG that are smaller than k times machine precision times the largest
    eigenvalue are raised to that value instead of being dropped, which avoids
    stopping short of the optimum when parameters are nearly collinear
    (:cite:`Dennis1981`).

    **Stopping.** The algorithm stops successfully when the decrement
    :math:`\delta = g'B^{-1}g` is below ``convergence_decrement_tol``. Unlike
    criteria based on the gradient or on the change of the objective, the decrement
    does not depend on the units of the parameters or on the number of observations.
    For a likelihood, :math:`\sqrt{\delta}` approximates the distance to the maximum,
    measured in robust (sandwich) standard errors, also in misspecified models. With
    bounds, :math:`\delta` is twice the decrease of the quadratic model over the
    feasible steps. It is zero exactly at points that satisfy the first order
    conditions. The criteria ``convergence_ftol_abs`` and ``convergence_xtol_rel`` are
    disabled by default. A run is reported as successful if any convergence criterion
    is met.

    **Requirements on the objective function.** BHHH is only valid for log-likelihood
    functions written as a sum of contributions. Decorate your function with
    ``om.mark.likelihood`` and return one contribution per independent unit. The
    algorithm cannot check most of the following requirements. Violating them makes
    BHHH slow or makes it fail.

    - *Contributions of independent units.* The information matrix equality holds
      per independent unit. If observations within a unit are dependent (e.g. panel
      data with individual random effects), return one contribution per unit, i.e.
      the sum over its observations. Conversely, do not aggregate independent
      observations: with grouped or frequency data, pass one contribution per
      observation, not per group. In the experiments of :cite:`Bunch1988`, grouping
      made BHHH between 2 and about 20 times slower, and more so for larger groups.
    - *Sum, do not average.* Do not divide the contributions by the number of
      observations. Averaged contributions make the BHHH step n times too long. The
      line search corrects this, but needs extra evaluations of the objective in each
      iteration.
    - *No weights.* For a weighted log-likelihood :math:`\sum_i w_i \ell_i`, the right
      curvature estimate is :math:`\sum_i w_i s_i s_i'`, but the OPG of the weighted
      contributions is :math:`\sum_i w_i^2 s_i s_i'`. With integer frequency weights,
      repeat the observations instead. Other weights are not supported yet.
    - *Only likelihoods.* For least-squares problems, GMM objectives or likelihoods
      with a penalty added as an extra contribution, the information matrix equality
      does not hold and BHHH is only a heuristic. optimagic accepts least-squares
      problems for likelihood optimizers (the contributions are the squared
      residuals), but you should use a least-squares optimizer for them. If the
      residuals at the solution are small, the decrement does not go to zero and BHHH
      cannot converge.
    - *Identified parameters.* The OPG must be non-singular at the solution, so there
      must be at least as many contributions as parameters. BHHH warns about
      parameters that do not affect the objective at the start values and about a
      nearly singular OPG at the solution. A parameter whose derivatives are zero for
      all contributions is not moved by BHHH, e.g. the standard deviation of a random
      coefficient that is started at zero.
    - *Accurate derivatives.* The attainable decrement is limited by the accuracy of
      the derivatives. Use closed form derivatives, automatic differentiation or
      central finite differences (optimagic's default).
    - *Finite start values.* The contributions and their derivatives must be finite
      at the start values, and the functions must not raise errors there; otherwise
      the optimization stops with an error. This catches most bugs in the objective
      and derivative functions. Elsewhere, trial points at which the objective or its
      derivatives are not finite or raise an error (e.g. outside the stationarity
      region of a GARCH model) are rejected by the line search, whatever the value of
      ``error_handling``. The number of rejected points is reported in
      ``algorithm_output``, and the result message shows the last error.

    BHHH also converges for misspecified models, but more slowly, because the OPG no
    longer approximates the Hessian. Bounds and constraints that optimagic implements
    by reparametrization are supported; BHHH then works with the internal parameters.

    **Result.** ``jac`` is the gradient of the minimized objective with respect to the
    internal parameters. No ``hess_inv`` is returned, because the inverse OPG is the
    least reliable estimator of the covariance matrix; use
    :func:`~optimagic.estimate_ml` instead. ``algorithm_output`` contains the final
    decrement, the rank and condition number of the scaled OPG, the active bounds, the
    numbers of step reductions, step expansions and failed evaluations of the
    objective and the derivatives, and the reason for stopping.

    """

    globalization: Literal["line_search"] | LineSearch = "line_search"
    """How the step length along the BHHH step is chosen.

    Currently, the only option is a line search. Pass "line_search" to use it with its
    default options or a :class:`~optimagic.optimizers.bhhh.LineSearch` instance to
    tune it, e.g. ``LineSearch(expansion="none")``.

    """

    convergence_decrement_tol: NonNegativeFloat = CONVERGENCE_DECREMENT_TOL
    r"""Stop successfully if the decrement is at most this.

    The decrement is :math:`\delta = g'(J'J)^{-1}g`, where :math:`g` is the gradient and
    :math:`J` the Jacobian of the contributions. For a likelihood, the estimate is then
    about :math:`\sqrt{\delta}` robust standard errors away from the maximum. The
    default of 1e-8 corresponds to 1e-4 standard errors.

    """

    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_FTOL_ABS
    """Stop if an accepted step improves the objective by at most this.

    The objective is the sum of the contributions, so the tolerance is on the scale of
    log-likelihood differences, whatever the number of observations. Set to zero (the
    default) to disable the criterion.

    """

    convergence_xtol_rel: NonNegativeFloat = 0.0
    r"""Stop if the BHHH step is small relative to the parameters.

    The criterion is :math:`\max_i |p_i| / \max(1, |x_i|) \leq`
    ``convergence_xtol_rel``, where :math:`p` is the BHHH step before the line search
    and :math:`x` are the current internal parameters. It depends on the units of the
    parameters. Set to zero (the default) to disable the criterion.

    """

    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    """Maximum number of iterations, i.e. of accepted steps."""

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN
    """Maximum number of evaluations of the objective.

    Evaluations of the derivatives are not counted.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        return bhhh_internal(
            problem=problem,
            x0=x0,
            line_search_options=_resolve_globalization(self.globalization),
            decrement_tol=self.convergence_decrement_tol,
            ftol_abs=self.convergence_ftol_abs,
            xtol_rel=self.convergence_xtol_rel,
            maxiter=self.stopping_maxiter,
            maxfun=self.stopping_maxfun,
        )


def _resolve_globalization(
    globalization: Literal["line_search"] | LineSearch,
) -> LineSearch:
    """Replace the name of a globalization strategy by its default instance."""
    if isinstance(globalization, str):
        return LineSearch()
    return globalization
