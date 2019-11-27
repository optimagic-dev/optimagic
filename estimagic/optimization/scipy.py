import numpy as np
from scipy.optimize import minimize


def minimize_scipy(
    internal_criterion, params, internal_params, algo_name, algo_options
):
    # Scipy works with `None` instead of infinite values for unconstrained parameters.
    bounds = (
        params.query("_internal_free")[["lower", "upper"]]
        .replace([-np.inf, np.inf], [None, None])
        .to_numpy()
    )

    scipy_results_obj = minimize(
        internal_criterion,
        internal_params,
        method=algo_name,
        bounds=bounds,
        options=algo_options,
    )
    results = _process_scipy_results(scipy_results_obj)

    return results


def _process_scipy_results(scipy_results_obj):
    """Convert optimization results into json serializable dictionary.

    Args:
        scipy_results_obj (scipy.optimize.OptimizeResult): Result from numerical
            optimizer.

    Todo: Is the list conversion necessary? If so, apply to all processing steps.

    """
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
