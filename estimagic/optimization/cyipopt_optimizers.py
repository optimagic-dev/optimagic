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
    stopping_max_iterations=STOPPING_MAX_ITERATIONS,
    barrier_strategy="monotone",
):
    """Minimize a scalar function using the Interior Point Optimizer.

    This implementation of the Interior Point Optimizer (:cite:`Waechter2005`,
    :cite:`Waechter2005a`, :cite:`Waechter2005b`, :cite:`Nocedal2009`) relies on
    `cyipopt <https://cyipopt.readthedocs.io/en/latest/index.html>`_, a Python wrapper
    for the `Ipopt optimization package <https://coin-or.github.io/Ipopt/index.html>`_.

    - convergence.relative_criterion_tolerance (float): The algorithm terminates
      successfully, if the (scaled) non linear programming error becomes smaller than
      this value.
    - stopping.max_iterations (int):  If the maximum number of iterations is reached,
      the optimization stops, but we do not count this as successful convergence. The
      difference to ``max_criterion_evaluations`` is that one iteration might need
      several criterion evaluations, for example in a line search or to determine if the
      trust region radius has to be shrunk.
    - barrier_strategy (str): which barrier parameter update strategy is to be used. Can
      be "monotone" or "adaptive". Default is "monotone", i.e. use the monotone
      (Fiacco-McCormick) strategy.

    """
    if not IS_CYIPOPT_INSTALLED:
        raise NotImplementedError(
            "The cyipopt package is not installed and required for 'ipopt'. You can "
            "install the package with: `conda install -c conda-forge cyipopt`"
        )

    if barrier_strategy not in ["monotone", "adaptive"]:
        raise ValueError(
            f"Unknown barrier strategy: {barrier_strategy}. It must be 'monotone' or "
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
        "acceptable_iter": 0,  # disable the "acceptable" heuristic
        "max_iter": stopping_max_iterations,
        "mu_strategy": barrier_strategy,
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
