"""Implement the fides optimizer."""
import logging

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
    hessian_update_strategy="BFGS",
):
    """Minimize a scalar function using the Fides Optimizer.

    Args:
        hessian_update_strategy (str): Hessian Update Strategy to employ. See the
            [Fides' Documentation](https://fides-optimizer.readthedocs.io/en/latest/generated/fides.hessian_approximation.html) # noqa: E501
            for available strategies.

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

    def fun(x):
        criterion = criterion_and_derivative(
            x=x, task="criterion", algorithm_info=algo_info
        )
        derivative = criterion_and_derivative(
            x=x, task="derivative", algorithm_info=algo_info
        )
        return criterion, derivative

    hessian_class = getattr(hessian_approximation, hessian_update_strategy)
    hessian_instance = hessian_class()

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
