import numpy as np
from scipy.optimize import minimize


def minimize_scipy_np(func, x0, bounds, algo_name, algo_options=None, gradient=None):
    """Interface for scipy.

    Args:
        func (callable): Objective function.
        x0 (np.ndarray): Starting values of the parameters.
        bounds (Tuple[np.ndarray]): A tuple containing two NumPy arrays where the first
            corresponds to the lower and the second to the upper bound. Unbounded
            parameters are represented by infinite values. The arrays have the same
            length as the parameter vector.
        algo_name (str): One of the optimizers of the scipy package which receives the
            same inputs as the ``"method"`` keyword of the original function.
        algo_options (dict): Options for the optimizer.
        gradient (callable): Gradient function.

    Returns:
        results (dict): Dictionary with processed optimization results.

    """
    # Scipy works with `None` instead of infinite values for unconstrained parameters
    # and requires a list of tuples for each parameter with lower and upper bound.
    bounds = np.column_stack(bounds)
    mask = ~np.isfinite(bounds)
    bounds = bounds.astype("object")
    bounds[mask] = None
    bounds = tuple(bounds)

    scipy_results_obj = minimize(
        func, x0, jac=gradient, method=algo_name, bounds=bounds, options=algo_options,
    )
    results = _process_scipy_results(scipy_results_obj)

    return results


def _process_scipy_results(scipy_results_obj):
    results = {**scipy_results_obj}
    # Harmonized results
    results["fitness"] = results.pop("fun", None)
    results["n_evaluations"] = results.pop("nfev", None)

    # Other results.
    results["jacobian"] = results.pop("jac", None)
    results["hessian"] = results.pop("hess", None)
    results["n_evaluations_jacobian"] = results.pop("njev", None)
    results["n_evaluations_hessian"] = results.pop("nhev", None)
    results["n_iterations"] = results.pop("nit", None)
    results["max_constraints_violations"] = results.pop("maxcv", None)
    results["hessian_inverse"] = results.pop("hess_inv", None)

    return results
