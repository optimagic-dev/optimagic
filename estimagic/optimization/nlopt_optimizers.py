"""Implement `nlopt` algorithms.

The documentation is heavily based on (nlopt documentation)[nlopt.readthedocs.io].

"""
import numpy as np

from estimagic.config import IS_NLOPT_INSTALLED
from estimagic.optimization.algo_options import CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE
from estimagic.optimization.algo_options import CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE
from estimagic.optimization.algo_options import CONVERGENCE_RELATIVE_CRITERION_TOLERANCE
from estimagic.optimization.algo_options import CONVERGENCE_RELATIVE_PARAMS_TOLERANCE
from estimagic.optimization.algo_options import STOPPING_MAX_CRITERION_EVALUATIONS
from estimagic.optimization.algo_options import (
    STOPPING_MAX_CRITERION_EVALUATIONS_GLOBAL,
)


if IS_NLOPT_INSTALLED:
    import nlopt


DEFAULT_ALGO_INFO = {
    "primary_criterion_entry": "value",
    "parallelizes": False,
    "needs_scaling": False,
}


def nlopt_bobyqa(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_relative_params_tolerance=CONVERGENCE_RELATIVE_PARAMS_TOLERANCE,
    convergence_absolute_params_tolerance=CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE,
    convergence_relative_criterion_tolerance=CONVERGENCE_RELATIVE_CRITERION_TOLERANCE,
    convergence_absolute_criterion_tolerance=CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS,
):

    """Minimize a scalar function using the BOBYQA algorithm.

    The implementation is derived from the BOBYQA subroutine of M. J. D. Powell.

    The algorithm performs derivative free bound-constrained optimization using
    an iteratively constructed quadratic approximation for the objective function.
    Due to its use of quadratic appoximation, the algorithm may perform poorly
    for objective functions that are not twice-differentiable.

    For details see:
    M. J. D. Powell, "The BOBYQA algorithm for bound constrained optimization
    without derivatives," Department of Applied Mathematics and Theoretical
    Physics, Cambridge England, technical report NA2009/06 (2009).

    ``nlopt_bobyqa`` supports the following ``algo_options``:

    - convergence.relative_params_tolerance (float):  Stop when the relative movement
      between parameter vectors is smaller than this.
    - convergence.relative_criterion_tolerance (float): Stop when the relative
      improvement between two iterations is smaller than this.
    - stopping_max_criterion_evaluations (int): If the maximum number of function
      evaluation is reached, the optimization stops but we do not count this
      as convergence.

    """
    out = _minimize_nlopt(
        criterion_and_derivative,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=nlopt.LN_BOBYQA,
        algorithm_name="nlopt_bobyqa",
        convergence_xtol_rel=convergence_relative_params_tolerance,
        convergence_xtol_abs=convergence_absolute_params_tolerance,
        convergence_ftol_rel=convergence_relative_criterion_tolerance,
        convergence_ftol_abs=convergence_absolute_criterion_tolerance,
        stopping_max_eval=stopping_max_criterion_evaluations,
    )

    return out


def nlopt_neldermead(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_relative_params_tolerance=CONVERGENCE_RELATIVE_PARAMS_TOLERANCE,
    convergence_absolute_params_tolerance=CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE,
    convergence_relative_criterion_tolerance=0,
    convergence_absolute_criterion_tolerance=CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS,
):
    """Minimize a scalar function using the Nelder-Mead simplex algorithm.

    The basic algorithm is described in:
    J. A. Nelder and R. Mead, "A simplex method for function minimization,"
    The Computer Journal 7, p. 308-313 (1965).

    The difference between the nlopt implementation an the original implementation is
    that the nlopt version supports bounds. This is done by moving all new points that
    would lie outside the bounds exactly on the bounds.


    Args:
        convergence_relative_params_tolerance (float): Stop when the relative movement
            between parameter vectors is smaller than this.
        convergence_relative_criterion_tolerance (float): Stop when the relative
            improvement between two iterations is smaller than this.
            In contrast to other algorithms the relative criterion tolerance is set
            to zero by default because setting it to any non-zero value made the
            algorithm stop too early even on the most simple test functions.
        stopping_max_criterion_evaluations (int): If the maximum number of function
            evaluation is reached, the optimization stops but we do not count this
            as convergence.


    Returns:
        dict: See :ref:`internal_optimizer_output` for details.

    """

    out = _minimize_nlopt(
        criterion_and_derivative,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=nlopt.LN_NELDERMEAD,
        algorithm_name="nlopt_neldermead",
        convergence_xtol_rel=convergence_relative_params_tolerance,
        convergence_xtol_abs=convergence_absolute_params_tolerance,
        convergence_ftol_rel=convergence_relative_criterion_tolerance,
        convergence_ftol_abs=convergence_absolute_criterion_tolerance,
        stopping_max_eval=stopping_max_criterion_evaluations,
    )

    return out


def nlopt_praxis(
    criterion_and_derivative,
    x,
    *,
    convergence_relative_params_tolerance=CONVERGENCE_RELATIVE_PARAMS_TOLERANCE,
    convergence_absolute_params_tolerance=CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE,
    convergence_relative_criterion_tolerance=CONVERGENCE_RELATIVE_CRITERION_TOLERANCE,
    convergence_absolute_criterion_tolerance=CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS,
):
    """Minimize a scalar function using principal-axis method.

    This is a gradient-free local optimizer originally described in:
    Richard Brent, Algorithms for Minimization without Derivatives
    (Prentice-Hall, 1972). (Reprinted by Dover, 2002.). It assumes quadratic
    form of the optimized function and repeatedly updates a set of conjugate
    search directions.

    The algorithm, is not invariant to scaling of the objective function and may
    fail under its certain rank-preserving transformations (e.g., will lead to
    a non-quadratic shape of the objective function).

    The algorithm is not determenistic and it is not possible to achieve
    detereminancy via seed setting.

    The algorithm failed on a simple benchmark function with finite parameter bounds.
    Passing arguments `lower_bounds` and `upper_bounds` has been disabled for this
    algorithm.

    The difference between the nlopt implementation an the original implementation is
    that the nlopt version supports bounds. This is done by returning infinity (Inf)
    when the constraints are violated. The implementation of bound constraints
    is achieved at the const of significantly reduced speed of convergence.
    In case of bounded constraints, this method is dominated by `nlopt_bobyqa`
    and `nlopt_cobyla`.


    Args:
        convergence_relative_params_tolerance (float): Stop when the relative movement
            between parameter vectors is smaller than this.
        convergence_relative_criterion_tolerance (float): Stop when the relative
            improvement between two iterations is smaller than this.
            In contrast to other algorithms the relative criterion tolerance is set
            to zero by default because setting it to any non-zero value made the
            algorithm stop too early even on the most simple test functions.
        stopping_max_criterion_evaluations (int): If the maximum number of function
            evaluation is reached, the optimization stops but we do not count this
            as convergence.


    Returns:
        dict: See :ref:`internal_optimizer_output` for details.

    """
    out = _minimize_nlopt(
        criterion_and_derivative,
        x,
        lower_bounds=None,
        upper_bounds=None,
        algorithm=nlopt.LN_PRAXIS,
        algorithm_name="nlopt_praxis",
        convergence_xtol_rel=convergence_relative_params_tolerance,
        convergence_xtol_abs=convergence_absolute_params_tolerance,
        convergence_ftol_rel=convergence_relative_criterion_tolerance,
        convergence_ftol_abs=convergence_absolute_criterion_tolerance,
        stopping_max_eval=stopping_max_criterion_evaluations,
    )

    return out


def nlopt_cobyla(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_relative_params_tolerance=CONVERGENCE_RELATIVE_PARAMS_TOLERANCE,
    convergence_absolute_params_tolerance=CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE,
    convergence_relative_criterion_tolerance=CONVERGENCE_RELATIVE_CRITERION_TOLERANCE,
    convergence_absolute_criterion_tolerance=CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS,
):
    """Minimize a scalar function using the cobyla method.

    The alggorithm is derived from Powell's Constrained Optimization BY Linear
    Approximations (COBYLA) algorithm. It is a derivative-free optimizer with
    nonlinear inequality and equality constrains, described in:

    M. J. D. Powell, "A direct search optimization method that models the
    objective and constraint functions by linear interpolation," in Advances in
    Optimization and Numerical Analysis, eds. S. Gomez and J.-P. Hennart (Kluwer
    Academic: Dordrecht, 1994), p. 51-67

    It constructs successive linear approximations of the objective function and
    constraints via a simplex of n+1 points (in n dimensions), and optimizes these
    approximations in a trust region at each step.

    The the nlopt implementation differs from the original implementation in a
    a few ways:
    - Incorporates all of the NLopt termination criteria.
    - Adds explicit support for bound constraints.
    - Allows the algorithm to increase the trust-reion radius if the predicted
    imptoovement was approximately right and the simplex is satisfactory.
    - Pseudo-randomizes simplex steps in the algorithm, aimproving robustness by
    avoiding accidentally taking steps that don't improve conditioning, preserving
    the deterministic nature of the algorithm.
    - Supports unequal initial-step sizes in the different parameters.


    Args:
        convergence_relative_params_tolerance (float): Stop when the relative movement
            between parameter vectors is smaller than this.
        convergence_relative_criterion_tolerance (float): Stop when the relative
            improvement between two iterations is smaller than this.
            In contrast to other algorithms the relative criterion tolerance is set
            to zero by default because setting it to any non-zero value made the
            algorithm stop too early even on the most simple test functions.
        stopping_max_criterion_evaluations (int): If the maximum number of function
            evaluation is reached, the optimization stops but we do not count this
            as convergence.


    Returns:
        dict: See :ref:`internal_optimizer_output` for details.

    """

    out = _minimize_nlopt(
        criterion_and_derivative,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=nlopt.LN_COBYLA,
        algorithm_name="nlopt_cobyla",
        convergence_xtol_rel=convergence_relative_params_tolerance,
        convergence_xtol_abs=convergence_absolute_params_tolerance,
        convergence_ftol_rel=convergence_relative_criterion_tolerance,
        convergence_ftol_abs=convergence_absolute_criterion_tolerance,
        stopping_max_eval=stopping_max_criterion_evaluations,
    )

    return out


def nlopt_sbplx(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_relative_params_tolerance=CONVERGENCE_RELATIVE_PARAMS_TOLERANCE,
    convergence_absolute_params_tolerance=CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE,
    convergence_relative_criterion_tolerance=CONVERGENCE_RELATIVE_CRITERION_TOLERANCE,
    convergence_absolute_criterion_tolerance=CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS,
):
    """Minimize a scalar function using the "Subplex" algorithm.

    The alggorithm is a reimplementation of  Tom Rowan's "Subplex" algorithm.
    See: T. Rowan, "Functional Stability Analysis of Numerical Algorithms",
    Ph.D. thesis, Department of Computer Sciences, University of Texas at
    Austin, 1990.

    Subplex is a variant of Nedler-Mead that uses Nedler-Mead on a sequence of
    subspaces. It is climed to be more efficient and robust than the original
    Nedler-Mead algorithm.

    The difference between this re-implementation and the original algorithm
    of Rowan, is that it explicitly supports bound constraints providing big
    improvement in the case where the optimum lies against one of the constraints.



    Args:
        convergence_relative_params_tolerance (float): Stop when the relative movement
            between parameter vectors is smaller than this.
        convergence_relative_criterion_tolerance (float): Stop when the relative
            improvement between two iterations is smaller than this.
            In contrast to other algorithms the relative criterion tolerance is set
            to zero by default because setting it to any non-zero value made the
            algorithm stop too early even on the most simple test functions.
        stopping_max_criterion_evaluations (int): If the maximum number of function
            evaluation is reached, the optimization stops but we do not count this
            as convergence.


    Returns:
        dict: See :ref:`internal_optimizer_output` for details.

    """

    out = _minimize_nlopt(
        criterion_and_derivative,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=nlopt.LN_SBPLX,
        algorithm_name="nlopt_sbplx",
        convergence_xtol_rel=convergence_relative_params_tolerance,
        convergence_xtol_abs=convergence_absolute_params_tolerance,
        convergence_ftol_rel=convergence_relative_criterion_tolerance,
        convergence_ftol_abs=convergence_absolute_criterion_tolerance,
        stopping_max_eval=stopping_max_criterion_evaluations,
    )

    return out


def nlopt_newuoa(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_relative_params_tolerance=CONVERGENCE_RELATIVE_PARAMS_TOLERANCE,
    convergence_absolute_params_tolerance=CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE,
    convergence_relative_criterion_tolerance=CONVERGENCE_RELATIVE_CRITERION_TOLERANCE,
    convergence_absolute_criterion_tolerance=CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS,
):
    """Minimize a scalar function using the NEWUOA algorithm.

    The algorithm is derived from the NEWUOA subroutine of M.J.D Powell which
    uses iteratively constructed quadratic approximation of the objctive fucntion
    to perform derivative-free unconstrained optimization. Fore more details see:
    M. J. D. Powell, "The NEWUOA software for unconstrained optimization without
    derivatives," Proc. 40th Workshop on Large Scale Nonlinear Optimization
    (Erice, Italy, 2004).

    The algorithm in `nlopt` has been modified to support bound constraints. If all
    of the bound constraints are infinite, this function calls the `nlopt.LN_NEWUOA`
    optimizers for uncsonstrained optimization. Otherwise, the `nlopt.LN_NEWUOA_BOUND`
    optimizer for constrained problems.

    `NEWUOA` requires the dimension n of the parameter space to be `≥ 2`, i.e. the
    implementation does not handle one-dimensional optimization problems.


    Args:
        convergence_relative_params_tolerance (float): Stop when the relative movement
            between parameter vectors is smaller than this.
        convergence_relative_criterion_tolerance (float): Stop when the relative
            improvement between two iterations is smaller than this.
            In contrast to other algorithms the relative criterion tolerance is set
            to zero by default because setting it to any non-zero value made the
            algorithm stop too early even on the most simple test functions.
        stopping_max_criterion_evaluations (int): If the maximum number of function
            evaluation is reached, the optimization stops but we do not count this
            as convergence.


    Returns:
        dict: See :ref:`internal_optimizer_output` for details.

    """
    if np.any(np.isfinite(lower_bounds)) or np.any(np.isfinite(upper_bounds)):
        algo = nlopt.LN_NEWUOA_BOUND
    else:
        algo = nlopt.LN_NEWUOA

    out = _minimize_nlopt(
        criterion_and_derivative,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=algo,
        algorithm_name="nlopt_newuoa",
        convergence_xtol_rel=convergence_relative_params_tolerance,
        convergence_xtol_abs=convergence_absolute_params_tolerance,
        convergence_ftol_rel=convergence_relative_criterion_tolerance,
        convergence_ftol_abs=convergence_absolute_criterion_tolerance,
        stopping_max_eval=stopping_max_criterion_evaluations,
    )

    return out


def nlopt_tnewton(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_relative_params_tolerance=CONVERGENCE_RELATIVE_PARAMS_TOLERANCE,
    convergence_absolute_params_tolerance=CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE,
    convergence_relative_criterion_tolerance=CONVERGENCE_RELATIVE_CRITERION_TOLERANCE,
    convergence_absolute_criterion_tolerance=CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS,
):
    """Minimize a scalar function using the "TNEWTON" algorithm.

    The alggorithm is based on a Fortran implementation of a preconditioned
    inexact truncated Newton algorithm written by Prof. Ladislav Luksan.

    Truncated Newton methods are a set of algorithms designed to solve large scale
    optimization problems. The algorithms use (inaccurate) approximations of the
    solutions to Newton equations, using conjugate gradient methodds, to handle the
    expensive calculations of derivatives during each iteration.

    Detailed description of algorithms is given in: R. S. Dembo and T. Steihaug,
    "Truncated Newton algorithms for large-scale optimization," Math. Programming
    26, p. 190-212 (1983), http://doi.org/10.1007/BF02592055.



    Args:
        convergence_relative_params_tolerance (float): Stop when the relative movement
            between parameter vectors is smaller than this.
        convergence_relative_criterion_tolerance (float): Stop when the relative
            improvement between two iterations is smaller than this.
            In contrast to other algorithms the relative criterion tolerance is set
            to zero by default because setting it to any non-zero value made the
            algorithm stop too early even on the most simple test functions.
        stopping_max_criterion_evaluations (int): If the maximum number of function
            evaluation is reached, the optimization stops but we do not count this
            as convergence.


    Returns:
        dict: See :ref:`internal_optimizer_output` for details.

    """

    out = _minimize_nlopt(
        criterion_and_derivative,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=nlopt.LD_TNEWTON,
        algorithm_name="nlopt_tnewton",
        convergence_xtol_rel=convergence_relative_params_tolerance,
        convergence_xtol_abs=convergence_absolute_params_tolerance,
        convergence_ftol_rel=convergence_relative_criterion_tolerance,
        convergence_ftol_abs=convergence_absolute_criterion_tolerance,
        stopping_max_eval=stopping_max_criterion_evaluations,
    )

    return out


def nlopt_lbfgs(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_relative_params_tolerance=CONVERGENCE_RELATIVE_PARAMS_TOLERANCE,
    convergence_absolute_params_tolerance=CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE,
    convergence_relative_criterion_tolerance=CONVERGENCE_RELATIVE_CRITERION_TOLERANCE,
    convergence_absolute_criterion_tolerance=CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS,
):
    """Minimize a scalar function using the "LBFGS" algorithm.

    The alggorithm is based on a Fortran implementation of low storage BFGS algorithm
    written by Prof. Ladislav Luksan.

    LFBGS is an approximation of the original Broyden–Fletcher–Goldfarb–Shanno algorithm
    based on limited use of memory. Memory efficiency is obtained by preserving a limi-
    ted number (<10) of past updates of candidate points and gradient values and using
    them to approximate the hessian matrix.

    Detailed description of algorithms is given in:
    J. Nocedal, "Updating quasi-Newton matrices with limited storage," Math. Comput.
    35, 773-782 (1980).
    D. C. Liu and J. Nocedal, "On the limited memory BFGS method for large scale
    optimization," ''Math. Programming' 45, p. 503-528 (1989).


    Args:
        convergence_relative_params_tolerance (float): Stop when the relative movement
            between parameter vectors is smaller than this.
        convergence_relative_criterion_tolerance (float): Stop when the relative
            improvement between two iterations is smaller than this.
            In contrast to other algorithms the relative criterion tolerance is set
            to zero by default because setting it to any non-zero value made the
            algorithm stop too early even on the most simple test functions.
        stopping_max_criterion_evaluations (int): If the maximum number of function
            evaluation is reached, the optimization stops but we do not count this
            as convergence.


    Returns:
        dict: See :ref:`internal_optimizer_output` for details.

    """

    out = _minimize_nlopt(
        criterion_and_derivative,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=nlopt.LD_TNEWTON,
        algorithm_name="nlopt_tnewton",
        convergence_xtol_rel=convergence_relative_params_tolerance,
        convergence_xtol_abs=convergence_absolute_params_tolerance,
        convergence_ftol_rel=convergence_relative_criterion_tolerance,
        convergence_ftol_abs=convergence_absolute_criterion_tolerance,
        stopping_max_eval=stopping_max_criterion_evaluations,
    )

    return out


def nlopt_ccsaq(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_relative_params_tolerance=CONVERGENCE_RELATIVE_PARAMS_TOLERANCE,
    convergence_absolute_params_tolerance=CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE,
    convergence_relative_criterion_tolerance=CONVERGENCE_RELATIVE_CRITERION_TOLERANCE,
    convergence_absolute_criterion_tolerance=CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS,
):

    """Minimize a scalar function using CCSAQ algorithm.

    CCSAQ uses the quadratic variant of the conservative convex separable approximation.
    The algorithm performs gradient based local optimization with equality (but not
    inequality) constraints. At each candidate point x, a quadratic approximation
    to the criterion faunction is computed using the value of gradient at point x. A
    penalty term is incorporated to render optimizaion convex and conservative. The
    algorithm is "globally convergent" in the sense that it is guaranteed to con-
    verge to a local optimum from any feasible starting point.

    The implementation is based on CCSA algorithm described in:
    Krister Svanberg, "A class of globally convergent optimization methods based
    on conservative convex separable approximations," SIAM J. Optim. 12 (2), p.
    555-573 (2002)



    ``nlopt_ccsaq`` supports the following ``algo_options``:

    - convergence.relative_params_tolerance (float):  Stop when the relative movement
      between parameter vectors is smaller than this.
    - convergence.relative_criterion_tolerance (float): Stop when the relative
      improvement between two iterations is smaller than this.
    - stopping_max_criterion_evaluations (int): If the maximum number of function
      evaluation is reached, the optimization stops but we do not count this
      as convergence.


    """
    out = _minimize_nlopt(
        criterion_and_derivative,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=nlopt.LD_CCSAQ,
        algorithm_name="nlopt_ccsaq",
        convergence_xtol_rel=convergence_relative_params_tolerance,
        convergence_xtol_abs=convergence_absolute_params_tolerance,
        convergence_ftol_rel=convergence_relative_criterion_tolerance,
        convergence_ftol_abs=convergence_absolute_criterion_tolerance,
        stopping_max_eval=stopping_max_criterion_evaluations,
    )

    return out


def nlopt_mma(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_relative_params_tolerance=CONVERGENCE_RELATIVE_PARAMS_TOLERANCE,
    convergence_absolute_params_tolerance=CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE,
    convergence_relative_criterion_tolerance=CONVERGENCE_RELATIVE_CRITERION_TOLERANCE,
    convergence_absolute_criterion_tolerance=CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS,
):

    """Minimize a scalar function using the method of moving asymptotes (MMA).

    The implementation is based on an algorithm described in:
    Krister Svanberg, "A class of globally convergent optimization methods based
    on conservative convex separable approximations," SIAM J. Optim. 12 (2), p.
    555-573 (2002)

    The algorithm performs gradient based local optimization with equality (but
    not inequality) constraints. At each candidate point x, an approximation to the
    criterion faunction is computed using the value of gradient at point x. A quadratic
    penalty term is incorporated to render optimizaion convex and conservative. The
    algorithm is "globally convergent" in the sense that it is guaranteed to con-
    verge to a local optimum from any feasible starting point.



    ``nlopt_mma`` supports the following ``algo_options``:

    - convergence.relative_params_tolerance (float):  Stop when the relative movement
      between parameter vectors is smaller than this.
    - convergence.relative_criterion_tolerance (float): Stop when the relative
      improvement between two iterations is smaller than this.
    - stopping_max_criterion_evaluations (int): If the maximum number of function
      evaluation is reached, the optimization stops but we do not count this
      as convergence.

    """
    out = _minimize_nlopt(
        criterion_and_derivative,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=nlopt.LD_MMA,
        algorithm_name="nlopt_mma",
        convergence_xtol_rel=convergence_relative_params_tolerance,
        convergence_xtol_abs=convergence_absolute_params_tolerance,
        convergence_ftol_rel=convergence_relative_criterion_tolerance,
        convergence_ftol_abs=convergence_absolute_criterion_tolerance,
        stopping_max_eval=stopping_max_criterion_evaluations,
    )

    return out


def nlopt_var(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_relative_params_tolerance=CONVERGENCE_RELATIVE_PARAMS_TOLERANCE,
    convergence_absolute_params_tolerance=CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE,
    convergence_relative_criterion_tolerance=CONVERGENCE_RELATIVE_CRITERION_TOLERANCE,
    convergence_absolute_criterion_tolerance=CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS,
    rank_variant=1,
):

    """Minimize a scalar function limited memory switching variable-metric method.

    The algorithm relies on saving only limited number M of past updates of the
    gradient to approximate the inverse hessian. The large is M, the more memory is
    consumed

    Detailed explanation of the algorithm, including its two variations of  rank-2 and
    rank-1 methods can be found in the following paper:
    J. Vlcek and L. Luksan, "Shifted limited-memory variable metric methods for
    large-scale unconstrained minimization," J. Computational Appl. Math. 186,
    p. 365-390 (2006).

    ``nlopt_svmm`` supports the following ``algo_options``:

    - convergence.relative_params_tolerance (float):  Stop when the relative movement
      between parameter vectors is smaller than this.
    - convergence.relative_criterion_tolerance (float): Stop when the relative
      improvement between two iterations is smaller than this.
    - stopping_max_criterion_evaluations (int): If the maximum number of function
      evaluation is reached, the optimization stops but we do not count this
      as convergence.

    """
    if rank_variant == 1:
        algo = nlopt.LD_VAR1
    elif rank_variant == 2:
        algo = nlopt.LD_VAR2
    else:
        raise ValueError(
            "nlopt supports only rank-1 and rank-2 methods of shifting variable-"
            "metric method"
        )
    out = _minimize_nlopt(
        criterion_and_derivative,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=algo,
        algorithm_name="nlopt_var",
        convergence_xtol_rel=convergence_relative_params_tolerance,
        convergence_xtol_abs=convergence_absolute_params_tolerance,
        convergence_ftol_rel=convergence_relative_criterion_tolerance,
        convergence_ftol_abs=convergence_absolute_criterion_tolerance,
        stopping_max_eval=stopping_max_criterion_evaluations,
    )

    return out


def nlopt_slsqp(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_relative_params_tolerance=CONVERGENCE_RELATIVE_PARAMS_TOLERANCE,
    convergence_absolute_params_tolerance=CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE,
    convergence_relative_criterion_tolerance=CONVERGENCE_RELATIVE_CRITERION_TOLERANCE,
    convergence_absolute_criterion_tolerance=CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS,
):
    """Optimize a scalar function based on SLSQP method.

    SLSQP solves gradient based nonlinearly constrained optimization problems.
    The algorithm treats the optimization problem as a sequence of constrained
    least-squares problems.

    The implementation is based on the procedure described in:
    Dieter Kraft, "A software package for sequential quadratic programming",
    Technical Report DFVLR-FB 88-28, Institut für Dynamik der Flugsysteme,
    Oberpfaffenhofen, July 1988.
    Dieter Kraft, "Algorithm 733: TOMP–Fortran modules for optimal control
    calculations," ACM Transactions on Mathematical Software, vol. 20, no. 3,
    pp. 262-281 (1994).

    ``nlopt_slsqp`` supports the following ``algo_options``:

    - convergence.relative_params_tolerance (float):  Stop when the relative movement
      between parameter vectors is smaller than this.
    - convergence.relative_criterion_tolerance (float): Stop when the relative
      improvement between two iterations is smaller than this.
    - stopping_max_criterion_evaluations (int): If the maximum number of function
      evaluation is reached, the optimization stops but we do not count this
      as convergence.

    """
    out = _minimize_nlopt(
        criterion_and_derivative,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=nlopt.LD_SLSQP,
        algorithm_name="nlopt_slsqp",
        convergence_xtol_rel=convergence_relative_params_tolerance,
        convergence_xtol_abs=convergence_absolute_params_tolerance,
        convergence_ftol_rel=convergence_relative_criterion_tolerance,
        convergence_ftol_abs=convergence_absolute_criterion_tolerance,
        stopping_max_eval=stopping_max_criterion_evaluations,
    )
    return out


def nlopt_direct(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_relative_params_tolerance=CONVERGENCE_RELATIVE_PARAMS_TOLERANCE,
    convergence_absolute_params_tolerance=CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE,
    convergence_relative_criterion_tolerance=CONVERGENCE_RELATIVE_CRITERION_TOLERANCE,
    convergence_absolute_criterion_tolerance=CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS_GLOBAL,
    nlopt_direct_version=0,
):
    """Optimize a scalar function based on DIRECT method.

    DIRECT is the DIviding RECTangles algorithm for global optimization, described in:
    D. R. Jones, C. D. Perttunen, and B. E. Stuckmann, "Lipschitzian optimization
    without the lipschitz constant," J. Optimization Theory and Applications, vol.
    79, p. 157 (1993).

    Variations of the algorithm include locally biased routines (distinguished by _L
    suffix) that prove to be more efficients for functions that have few local minima.
    See the following for the DIRECT_L variant:

    J. M. Gablonsky and C. T. Kelley, "A locally-biased form of the DIRECT algorithm,"
    J. Global Optimization, vol. 21 (1), p. 27-37 (2001).

    Locally biased algorithms can be implmented both with deterministic and random
    (distinguished by _RAND suffix) search algorithm.

    Finally, both original and locally biased variants can be implemented with and
    without the rescaling of the bound constraints.

    Argument nlopt_direct_vresion (int) determines which variant is implmented:
    - DIRECT: 0
    - DIRECT_L: 1
    - DIRECT_L_NOSCAL: 2
    - DIRECT_L_RAND: 3
    - DIRECT_L_RAND_NOSCAL: 4
    - DIRECT_RAND: 5

    ``nlopt_direct`` supports the following ``algo_options``:

    - convergence.relative_params_tolerance (float):  Stop when the relative movement
      between parameter vectors is smaller than this.
    - convergence.relative_criterion_tolerance (float): Stop when the relative
      improvement between two iterations is smaller than this.
    - stopping_max_criterion_evaluations_global (int): If the maximum number of function
      evaluation is reached.


    """
    if nlopt_direct_version == 0:
        algo = nlopt.GN_DIRECT
    elif nlopt_direct_version == 1:
        algo = nlopt.GN_DIRECT_L
    elif nlopt_direct_version == 2:
        algo = nlopt.GN_DIRECT_L_NOSCAL
    elif nlopt_direct_version == 3:
        algo = nlopt.GN_DIRECT_L_RAND
    elif nlopt_direct_version == 4:
        algo = nlopt.GN_DIRECT_L_RAND_NOSCAL
    elif nlopt_direct_version == 5:
        algo = nlopt.GN_DIRECT_NOSCAL
    out = _minimize_nlopt(
        criterion_and_derivative,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=algo,
        algorithm_name="nlopt_direct",
        convergence_xtol_rel=convergence_relative_params_tolerance,
        convergence_xtol_abs=convergence_absolute_params_tolerance,
        convergence_ftol_rel=convergence_relative_criterion_tolerance,
        convergence_ftol_abs=convergence_absolute_criterion_tolerance,
        stopping_max_eval=stopping_max_criterion_evaluations,
    )
    return out


def nlopt_esch(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_relative_params_tolerance=CONVERGENCE_RELATIVE_PARAMS_TOLERANCE,
    convergence_absolute_params_tolerance=CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE,
    convergence_relative_criterion_tolerance=CONVERGENCE_RELATIVE_CRITERION_TOLERANCE,
    convergence_absolute_criterion_tolerance=CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS_GLOBAL,
):
    """Optimize a scalar function using the ESCH algorithm.

    ESCH is an evolutionary algorithm that supports bound constraints only. Specifi
    cally, it does not support nonlinear constraints.

    More information on this method can be found in:
    C. H. da Silva Santos, M. S. Goncalves, and H. E. Hernandez-Figueroa, "Designing
    Novel Photonic Devices by Bio-Inspired Computing," IEEE Photonics Technology
    Letters 22 (15), pp. 1177–1179 (2010).
    C. H. da Silva Santos, "Parallel and Bio-Inspired Computing Applied to Analyze
    Microwave and Photonic Metamaterial Strucutures," Ph.D. thesis, University of
    Campinas, (2010).
    H.-G. Beyer and H.-P. Schwefel, "Evolution Strategies: A Comprehensive Introduction,
    "Journal Natural Computing, 1 (1), pp. 3–52 (2002).
    Ingo Rechenberg, "Evolutionsstrategie – Optimierung technischer Systeme nach
    Prinzipien der biologischen Evolution," Ph.D. thesis (1971), Reprinted by
    Fromman-Holzboog (1973).

    ``nlopt_escht`` supports the following ``algo_options``:

    - convergence.relative_params_tolerance (float):  Stop when the relative movement
      between parameter vectors is smaller than this.
    - convergence.relative_criterion_tolerance (float): Stop when the relative
      improvement between two iterations is smaller than this.
    - stopping_max_criterion_evaluations_global (int): If the maximum number of function
      evaluation is reached.


    """
    out = _minimize_nlopt(
        criterion_and_derivative,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=nlopt.GN_ESCH,
        algorithm_name="nlopt_esch",
        convergence_xtol_rel=convergence_relative_params_tolerance,
        convergence_xtol_abs=convergence_absolute_params_tolerance,
        convergence_ftol_rel=convergence_relative_criterion_tolerance,
        convergence_ftol_abs=convergence_absolute_criterion_tolerance,
        stopping_max_eval=stopping_max_criterion_evaluations,
    )
    return out


def nlopt_isres(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_relative_params_tolerance=CONVERGENCE_RELATIVE_PARAMS_TOLERANCE,
    convergence_absolute_params_tolerance=CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE,
    convergence_relative_criterion_tolerance=CONVERGENCE_RELATIVE_CRITERION_TOLERANCE,
    convergence_absolute_criterion_tolerance=CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS_GLOBAL,
):
    """Optimize a scalar function using the ISRES algorithm.

    ISRES is an implementation of "Improved Stochastic Evolution Strategy" written
    for solving optimization problems with non-linear constraints. The algorithm
    is supposed to be a global method, in that it has heuristics to avoid local
    minima. However, no convergence proof is available.

    The original method and a refined version can be found, respecively, in:
    Thomas Philip Runarsson and Xin Yao, "Search biases in constrained evolutionary
    optimization," IEEE Trans. on Systems, Man, and Cybernetics Part C: Applications
    and Reviews, vol. 35 (no. 2), pp. 233-243 (2005).
    Thomas P. Runarsson and Xin Yao, "Stochastic ranking for constrained evolutionary
    optimization," IEEE Trans. Evolutionary Computation, vol. 4 (no. 3), pp. 284-294
    (2000).


    ``nlopt_isres`` supports the following ``algo_options``:

    - convergence.relative_params_tolerance (float):  Stop when the relative movement
      between parameter vectors is smaller than this.
    - convergence.relative_criterion_tolerance (float): Stop when the relative
      improvement between two iterations is smaller than this.
    - stopping_max_criterion_evaluations_global (int): If the maximum number of function
      evaluation is reached.


    """
    out = _minimize_nlopt(
        criterion_and_derivative,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=nlopt.GN_ISRES,
        algorithm_name="nlopt_isres",
        convergence_xtol_rel=convergence_relative_params_tolerance,
        convergence_xtol_abs=convergence_absolute_params_tolerance,
        convergence_ftol_rel=convergence_relative_criterion_tolerance,
        convergence_ftol_abs=convergence_absolute_criterion_tolerance,
        stopping_max_eval=stopping_max_criterion_evaluations,
    )
    return out


def nlopt_crs2_lm(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_relative_params_tolerance=CONVERGENCE_RELATIVE_PARAMS_TOLERANCE,
    convergence_absolute_params_tolerance=CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE,
    convergence_relative_criterion_tolerance=CONVERGENCE_RELATIVE_CRITERION_TOLERANCE,
    convergence_absolute_criterion_tolerance=CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS_GLOBAL,
    random_search_population_size=None,
):
    """Optimize a scalar function using the CRS2_LM algorithm.

    This implementation of controlled random search method with local mutation is
    based on:
    P. Kaelo and M. M. Ali, "Some variants of the controlled random search algorithm
    for global optimization," J. Optim. Theory Appl. 130 (2), 253-264 (2006).

    The original CRS method is described in:
    W. L. Price, "A controlled random search procedure for global optimization,"
    in Towards Global Optimization 2, p. 71-84 edited by L. C. W. Dixon and G. P.
    Szego (North-Holland Press, Amsterdam, 1978).
    W. L. Price, "Global optimization by controlled random search," J. Optim. Theory
    Appl. 40 (3), p. 333-348 (1983).

    CRS class of algorithms starts with random population of points and evolves the
    points "randomly". The size of the initial population can be set via the param-
    meter random_search_population_size. If the user doesn't specify a value, it is
    set to the nlopt default of 10*(n+1).

    ``nlopt_isres`` supports the following ``algo_options``:

    - convergence.relative_params_tolerance (float):  Stop when the relative movement
      between parameter vectors is smaller than this.
    - convergence.relative_criterion_tolerance (float): Stop when the relative
      improvement between two iterations is smaller than this.
    - stopping_max_criterion_evaluations_global (int): If the maximum number of function
      evaluation is reached.
    - random_search_population_size(int): The size of the population of the starting
      points.


    """
    if not random_search_population_size:
        random_search_population_size = ((len(x) + 1),)
    out = _minimize_nlopt(
        criterion_and_derivative,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=nlopt.GN_CRS2_LM,
        algorithm_name="nlopt_crs2_lm",
        convergence_xtol_rel=convergence_relative_params_tolerance,
        convergence_xtol_abs=convergence_absolute_params_tolerance,
        convergence_ftol_rel=convergence_relative_criterion_tolerance,
        convergence_ftol_abs=convergence_absolute_criterion_tolerance,
        stopping_max_eval=stopping_max_criterion_evaluations,
        population_size=random_search_population_size,
    )
    return out


def _minimize_nlopt(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    algorithm,
    algorithm_name,
    *,
    convergence_xtol_rel=None,
    convergence_xtol_abs=None,
    convergence_ftol_rel=None,
    convergence_ftol_abs=None,
    stopping_max_eval=None,
    population_size=None,
):
    """Run actual nlopt optimization argument, set relevant attributes."""
    algo_info = DEFAULT_ALGO_INFO.copy()
    algo_info["name"] = algorithm_name

    def func(x, grad):
        if grad.size > 0:
            criterion, derivative = criterion_and_derivative(
                x,
                task="criterion_and_derivative",
                algorithm_info=algo_info,
            )
            grad[:] = derivative
        else:
            criterion = criterion_and_derivative(
                x,
                task="criterion",
                algorithm_info=algo_info,
            )
        return criterion

    opt = nlopt.opt(algorithm, x.shape[0])
    if convergence_ftol_rel is not None:
        opt.set_ftol_rel(convergence_ftol_rel)
    if convergence_ftol_abs is not None:
        opt.set_ftol_abs(convergence_ftol_abs)
    if convergence_xtol_rel is not None:
        opt.set_xtol_rel(convergence_xtol_rel)
    if convergence_xtol_abs is not None:
        opt.set_xtol_abs(convergence_xtol_abs)
    if lower_bounds is not None:
        opt.set_lower_bounds(lower_bounds)
    if upper_bounds is not None:
        opt.set_upper_bounds(upper_bounds)
    if stopping_max_eval is not None:
        opt.set_maxeval(stopping_max_eval)
    if population_size is not None:
        opt.set_population(population_size)
    opt.set_min_objective(func)
    solution_x = opt.optimize(x)
    return _process_nlopt_results(opt, solution_x)


def _process_nlopt_results(nlopt_obj, solution_x):
    messages = {
        1: "Convergence achieved ",
        2: (
            "Optimizer stopped because maximum value of criterion function was reached"
        ),
        3: (
            "Optimizer stopped because convergence_relative_criterion_tolerance or "
            + "convergence_absolute_criterion_tolerance was reached"
        ),
        4: (
            "Optimizer stopped because convergence_relative_params_tolerance or "
            + "convergence_absolute_params_tolerance was reached"
        ),
        5: "Optimizer stopped because max_criterion_evaluations was reached",
        6: "Optimizer stopped because max running time was reached",
        -1: "Optimizer failed",
        -2: "Invalid arguments were passed",
        -3: "Memory error",
        -4: "Halted because roundoff errors limited progress",
        -5: "Halted because of user specified forced stop",
    }
    processed = {
        "solution_x": solution_x,
        "solution_criterion": nlopt_obj.last_optimum_value(),
        "solution_derivative": None,
        "solution_hessian": None,
        "n_criterion_evaluations": nlopt_obj.get_numevals(),
        "n_derivative_evaluations": None,
        "n_iterations": None,
        "success": nlopt_obj.last_optimize_result() in [1, 2, 3, 4],
        "message": messages[nlopt_obj.last_optimize_result()],
        "reached_convergence_criterion": None,
    }
    return processed
