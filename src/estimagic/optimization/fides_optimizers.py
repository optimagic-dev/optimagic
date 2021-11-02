"""Implement the fides optimizer."""
import logging
from functools import partial

from estimagic.config import IS_FIDES_INSTALLED

if IS_FIDES_INSTALLED:
    from fides import hessian_approximation
    from fides import Optimizer


def fides(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    hessian_update_strategy="bfgs",
):
    """Minimize a scalar function using the Fides Optimizer.

    - hessian_update_strategy (str): Hessian Update Strategy to employ. You can provide
      a lowercase or uppercase string or a
      fides.hession_approximation.HessianApproximation class instance. The available
      update strategies are:

        - **BB**: Broydens "bad" method as introduced :cite:`Broyden1965`.
        - **BFGS**: Broyden-Fletcher-Goldfarb-Shanno update strategy.
        - **BG**: Broydens "good" method as introduced in :cite:`Broyden1965`.
        - **Broyden**: BroydenClass Update scheme as described in :cite:`Nocedal1999`,
          Chapter 6.3.
        - **DFP**: Davidon-Fletcher-Powell update strategy.
        - **FX**: Hybrid method HY2 as introduced by :cite:`Fletcher1987`.
        - **GNSBFGS**: Hybrid Gauss-Newton Structured BFGS method as introduced
          by :cite:`Zhou2010`.
        - **IterativeHessianApproximation**: Iterative update schemes that only
          use the search direction steps and differences in the gradient for
          update.
        - **SR1**: Symmetric Rank 1 update strategy as described in :cite:`Nocedal1999`,
          Chapter 6.2.
        - **SSM**: Structured Secant Method as introduced by :cite:`Dennis1989`,
          which is compatible with BFGS, DFP update schemes.
        - **TSSM**: Totally Structured Secant Method as introduced by
          :cite:`Huschens1994`, which uses a self-adjusting update method for
          the second order term.

        Or you can pass a class instance directly.

      See the
      `Fides' Documentation
      <https://fides-optimizer.readthedocs.io/en/latest/generated/fides.hessian_approximation.html>`_ (# noqa: E501)
      for more details.

    Returns:
        dict: See :ref:`internal_optimizer_output` for details.

    """
    if not IS_FIDES_INSTALLED:
        raise NotImplementedError(
            "The fides package is not installed. You can install it with "
            "`pip install fides`."
        )

    algo_info = {
        "primary_criterion_entry": "value",
        "parallelizes": False,
        "needs_scaling": False,
        "name": "fides",
    }

    fun = partial(
        criterion_and_derivative,
        task="criterion_and_derivative",
        algorithm_info=algo_info,
    )

    if isinstance(hessian_update_strategy, str):
        hessian_class = getattr(hessian_approximation, hessian_update_strategy.upper())
        hessian_instance = hessian_class()
    elif not isinstance(
        hessian_update_strategy, hessian_approximation.HessianApproximation
    ):
        raise ValueError(
            "You must provide a hessian_update_strategy that is either a string or a "
            "fides.hessian_approximation.HessianApproximation class object."
        )

    opt = Optimizer(
        fun=fun,
        lb=lower_bounds,
        ub=upper_bounds,
        hessian_update=hessian_instance,
        verbose=logging.ERROR,
    )
    raw_res = opt.minimize(x)
    res = _process_fides_res(raw_res, opt)
    return res


def _process_fides_res(raw_res, opt):
    fval, x, grad, hess = raw_res
    res = {
        "solution_criterion": fval,
        "solution_x": x,
        "solution_derivative": grad,
        "solution_hessian": hess,
        "success": opt.converged,
        "n_iterations": opt.iteration,
    }
    return res
