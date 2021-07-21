"""Implement cyipopt's Interior Point Optimizer."""
import functools

from estimagic.config import IS_CYIPOPT_INSTALLED
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
):
    """Minimize a scalar function using the Interior Point Optimizer.

    This implementation of the Interior Point Optimizer (:cite:`Waechter2005`,
    :cite:`Waechter2005a`, :cite:`Waechter2005b`, :cite:`Nocedal2009`) relies on
    `cyipopt <https://cyipopt.readthedocs.io/en/latest/index.html>`_, a Python wrapper
    for the `Ipopt optimization package <https://coin-or.github.io/Ipopt/index.html>`_.

    """
    if not IS_CYIPOPT_INSTALLED:
        raise NotImplementedError(
            "The cyipopt package is not installed and required for 'ipopt'. You can "
            "install the package with: `conda install -c conda-forge cyipopt`"
        )

    algo_info = DEFAULT_ALGO_INFO.copy()
    algo_info["name"] = "ipopt"

    func = functools.partial(
        criterion_and_derivative,
        task="criterion",
        algorithm_info=algo_info,
    )

    raw_res = cyipopt.minimize_ipopt(
        fun=func, x0=x, bounds=get_scipy_bounds(lower_bounds, upper_bounds)
    )

    res = process_scipy_result(raw_res)
    return res
