import numpy as np
from scipy.optimize import minimize


def minimize_scipy_np(func, x0, bounds, algo_name, algo_options):
    # Scipy works with `None` instead of infinite values for unconstrained parameters
    # and requires a list of tuples for each parameter with lower and upper bound.
    bounds = np.column_stack(bounds).astype(float)
    bounds[~np.isfinite(bounds)] = None
    bounds = tuple(bounds)

    scipy_results_obj = minimize(
        func, x0, method=algo_name, bounds=bounds, options=algo_options,
    )
    results = _process_scipy_results(scipy_results_obj)

    return results


def _process_scipy_results(scipy_results_obj):
    results = {**scipy_results_obj}
    results["fitness"] = results.pop("fun", None)
    results["jacobian"] = results.pop("jac", None)
    results["hessian"] = results.pop("hess", None)
    results["n_evaluations"] = results.pop("nfev", None)
    results["n_evaluations_jacobian"] = results.pop("njev", None)
    results["n_evaluations_hessian"] = results.pop("nhev", None)
    results["n_iterations"] = results.pop("nit", None)
    results["max_constraints_violations"] = results.pop("maxcv", None)
    results["hessian_inverse"] = results.pop("hess_inv", None)

    return results
