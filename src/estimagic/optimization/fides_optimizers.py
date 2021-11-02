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

        - **bb**: Broydens "bad" method as introduced :cite:`Broyden1965`.
        - **bfgs**: Broyden-Fletcher-Goldfarb-Shanno update strategy.
        - **bg**: Broydens "good" method as introduced in :cite:`Broyden1965`.
        - You can use a general BroydenClass Update scheme using the Broyden class from
          `fides.hessian_approximation`. This is a generalization of BFGS/DFP methods
          where the parameter :math:`phi` controls the convex combination between the
          two. This is a rank 2 update strategy that preserves positive-semidefiniteness
          and symmetry (if :math:`\\phi \\in [0,1]`). It is described in
          :cite:`Nocedal1999`, Chapter 6.3.
        - **dfp**: Davidon-Fletcher-Powell update strategy.
        - **fx**: Hybrid method HY2 as introduced by :cite:`Fletcher1987`.
        - **gnsbfgs**: Hybrid Gauss-Newton Structured BFGS method as introduced by
          :cite:`Zhou2010`.
        - **sr1**: Symmetric Rank 1 update strategy as described in :cite:`Nocedal1999`,
          Chapter 6.2.
        - **ssm**: Structured Secant Method as introduced by :cite:`Dennis1989`, which
          is compatible with BFGS, DFP update schemes.
        - **tssm**: Totally Structured Secant Method as introduced by
          :cite:`Huschens1994`, which uses a self-adjusting update method for the second
          order term.

      Or you can pass a class instance directly. See the `Fides' Documentation
      <https://fides-optimizer.readthedocs.io/en/latest/generated/fides.hessian_approximation.html>`_
      (# noqa: E501) for more details.


    Returns: dict: See :ref:`internal_optimizer_output` for details.

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
        if hessian_update_strategy in ("broyden", "Broyden"):
            raise ValueError(
                "You cannot use the Broyden update strategy without specifying the "
                "interpolation parameter phi. Import the Broyden class from "
                "`fides.hessian_approximation`, create an instance of it with your "
                "desired value of phi and pass this instance instead."
            )
        else:
            hessian_name = hessian_update_strategy.upper()
        hessian_class = getattr(hessian_approximation, hessian_name)
        hessian_instance = hessian_class()
    elif isinstance(
        hessian_update_strategy, hessian_approximation.HessianApproximation
    ):
        hessian_instance = hessian_update_strategy
    else:
        raise ValueError(
            "You must provide a hessian_update_strategy that is either a string or an "
            "instance of the fides.hessian_approximation.HessianApproximation class."
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
