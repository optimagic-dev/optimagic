"""Implement cyipopt's Interior Point Optimizer."""
import functools

from estimagic.config import IS_CYIPOPT_INSTALLED
from estimagic.optimization.algo_options import CONVERGENCE_RELATIVE_CRITERION_TOLERANCE
from estimagic.optimization.algo_options import STOPPING_MAX_ITERATIONS
from estimagic.optimization.scipy_optimizers import get_scipy_bounds
from estimagic.optimization.scipy_optimizers import process_scipy_result

try:
    import cyipopt
except ImportError:
    pass

DEFAULT_ALGO_INFO = {
    "primary_criterion_entry": "value",
    "parallelizes": False,
    "needs_scaling": False,
}


def ipopt(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_relative_criterion_tolerance=CONVERGENCE_RELATIVE_CRITERION_TOLERANCE,
    #
    mu_strategy="monotone",
    s_max=100.0,
    #
    stopping_max_iterations=STOPPING_MAX_ITERATIONS,
    stopping_max_wall_time_seconds=1e20,
    stopping_max_cpu_time=1e20,
    #
    dual_inf_tol=1.0,
    constr_viol_tol=0.0001,
    compl_inf_tol=0.0001,
    #
    acceptable_iter=15,
    acceptable_tol=1e-6,
    acceptable_dual_inf_tol=1e-10,
    acceptable_constr_viol_tol=0.01,
    acceptable_compl_inf_tol=0.01,
    acceptable_obj_change_tol=1e20,
):
    """Minimize a scalar function using the Interior Point Optimizer.

    This implementation of the Interior Point Optimizer (:cite:`Waechter2005`,
    :cite:`Waechter2005a`, :cite:`Waechter2005b`, :cite:`Nocedal2009`) relies on
    `cyipopt <https://cyipopt.readthedocs.io/en/latest/index.html>`_, a Python
    wrapper for the `Ipopt optimization package
    <https://coin-or.github.io/Ipopt/index.html>`_.

    There are two levels of termination criteria. If the usual "desired"
    tolerances (see tol, dual_inf_tol etc) are satisfied at an iteration, the
    algorithm immediately terminates with a success message. On the other hand,
    if the algorithm encounters "acceptable_iter" many iterations in a row that
    are considered "acceptable", it will terminate before the desired
    convergence tolerance is met. This is useful in cases where the algorithm
    might not be able to achieve the "desired" level of accuracy.

    - convergence.relative_criterion_tolerance (float): The algorithm terminates
      successfully, if the (scaled) non linear programming error becomes smaller
      than this value.

    - stopping.max_iterations (int):  If the maximum number of iterations is
      reached, the optimization stops, but we do not count this as successful
      convergence. The difference to ``max_criterion_evaluations`` is that one
      iteration might need several criterion evaluations, for example in a line
      search or to determine if the trust region radius has to be shrunk.
    - stopping.max_wall_time_seconds (float): Maximum number of walltime clock
      seconds.
    - stopping.max_cpu_time (float): Maximum number of CPU seconds. A limit on
      CPU seconds that Ipopt can use to solve one problem. If during the
      convergence check this limit is exceeded, Ipopt will terminate with a
      corresponding message. The valid range for this real option is 0 <
      max_cpu_time and its default value is 1e20.

    - dual_inf_tol (float): Desired threshold for the dual infeasibility.
      Absolute tolerance on the dual infeasibility. Successful termination
      requires that the max-norm of the (unscaled) dual infeasibility is less
      than this threshold. The valid range for this real option is 0 <
      dual_inf_tol and its default value is 1.
    - constr_viol_tol (float): Desired threshold for the constraint and variable
      bound violation. Absolute tolerance on the constraint and variable bound
      violation. Successful termination requires that the max-norm of the
      (unscaled) constraint violation is less than this threshold. If option
      bound_relax_factor is not zero 0, then Ipopt relaxes given variable
      bounds. The value of constr_viol_tol is used to restrict the absolute
      amount of this bound relaxation. The valid range for this real option is 0
      < constr_viol_tol and its default value is 0.0001.
    - compl_inf_tol (float): Desired threshold for the complementarity
      conditions. Absolute tolerance on the complementarity. Successful
      termination requires that the max-norm of the (unscaled) complementarity
      is less than this threshold. The valid range for this real option is 0 <
      compl_inf_tol and its default value is 0.0001.

    - acceptable_iter (int): Number of "acceptable" iterates before triggering
      termination. If the algorithm encounters this many successive "acceptable"
      iterates (see above on the acceptable heuristic), it terminates, assuming
      that the problem has been solved to best possible accuracy given
      round-off. If it is set to zero, this heuristic is disabled. The valid
      range for this integer option is 0 ≤ acceptable_iter.
    - acceptable_tol (float):"Acceptable" convergence tolerance (relative).
      Determines which (scaled) overall optimality error is considered to be
      "acceptable". The valid range for this real option is 0 < acceptable_tol.
    - acceptable_dual_inf_tol (float):  "Acceptance" threshold for the dual
      infeasibility. Absolute tolerance on the dual infeasibility. "Acceptable"
      termination requires that the (max-norm of the unscaled) dual
      infeasibility is less than this threshold; see also acceptable_tol. The
      valid range for this real option is 0 < acceptable_dual_inf_tol and its
      default value is 10+10.
    - acceptable_constr_viol_tol (float): "Acceptance" threshold for the
      constraint violation. Absolute tolerance on the constraint violation.
      "Acceptable" termination requires that the max-norm of the (unscaled)
      constraint violation is less than this threshold; see also acceptable_tol.
      The valid range for this real option is 0 < acceptable_constr_viol_tol and
      its default value is 0.01.
    - acceptable_compl_inf_tol (float): "Acceptance" threshold for the
      complementarity conditions. Absolute tolerance on the complementarity.
      "Acceptable" termination requires that the max-norm of the (unscaled)
      complementarity is less than this threshold; see also acceptable_tol. The
      valid range for this real option is 0 < acceptable_compl_inf_tol and its
      default value is 0.01.
    - acceptable_obj_change_tol (float): "Acceptance" stopping criterion based on
      objective function change. If the relative change of the objective
      function (scaled by Max(1,|f(x)|)) is less than this value, this part of
      the acceptable tolerance termination is satisfied; see also
      acceptable_tol. This is useful for the quasi-Newton option, which has
      trouble to bring down the dual infeasibility. The valid range for this
      real option is 0 ≤ acceptable_obj_change_tol and its default value is
      10+20.

    - mu_strategy (str): which barrier parameter update strategy is to be used.
      Can be "monotone" or "adaptive". Default is "monotone", i.e. use the
      monotone (Fiacco-McCormick) strategy.
    - s_max (float): Scaling threshold for the NLP error.

    The following options are not supported through cyipopt: - mu_oracle

    """
    if not IS_CYIPOPT_INSTALLED:
        raise NotImplementedError(
            "The cyipopt package is not installed and required for 'ipopt'. You can "
            "install the package with: `conda install -c conda-forge cyipopt`"
        )
    if acceptable_tol <= convergence_relative_criterion_tolerance:
        raise ValueError(
            "The acceptable tolerance must be larger than the desired tolerance."
        )
    if mu_strategy not in ["monotone", "adaptive"]:
        raise ValueError(
            f"Unknown barrier strategy: {mu_strategy}. It must be 'monotone' or "
            "'adaptive'."
        )

    algo_info = DEFAULT_ALGO_INFO.copy()
    algo_info["name"] = "ipopt"

    gradient = functools.partial(
        criterion_and_derivative, task="derivative", algorithm_info=algo_info
    )

    func = functools.partial(
        criterion_and_derivative,
        task="criterion",
        algorithm_info=algo_info,
    )

    options = {
        "s_max": float(s_max),
        "max_iter": stopping_max_iterations,
        "max_wall_time": float(stopping_max_wall_time_seconds),
        "max_cpu_time": float(stopping_max_cpu_time),
        "dual_inf_tol": float(dual_inf_tol),
        "constr_viol_tol": float(constr_viol_tol),
        "compl_inf_tol": float(compl_inf_tol),
        #
        "acceptable_iter": int(acceptable_iter),
        "acceptable_tol": float(acceptable_tol),
        "acceptable_dual_inf_tol": float(acceptable_dual_inf_tol),
        "acceptable_constr_viol_tol": float(acceptable_constr_viol_tol),
        "acceptable_compl_inf_tol": float(acceptable_compl_inf_tol),
        "acceptable_obj_change_tol": float(acceptable_obj_change_tol),
        #
        "mu_strategy": mu_strategy,
        "print_level": 0,  # disable verbosity
    }

    raw_res = cyipopt.minimize_ipopt(
        fun=func,
        x0=x,
        bounds=get_scipy_bounds(lower_bounds, upper_bounds),
        jac=gradient,
        constraints=(),
        tol=convergence_relative_criterion_tolerance,
        options=options,
    )

    res = process_scipy_result(raw_res)
    return res
