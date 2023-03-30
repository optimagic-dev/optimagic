from functools import partial

import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint, minimize

from estimagic.optimization.tiktak import draw_exploration_sample


def solve_multistart(model, x_candidate, lower_bounds, upper_bounds):
    np.random.seed(12345)
    start_values = draw_exploration_sample(
        x=x_candidate,
        lower=lower_bounds,
        upper=upper_bounds,
        n_samples=100,
        sampling_distribution="uniform",
        sampling_method="sobol",
        seed=1234,
    )

    def crit(x):
        return model.predict(x)

    bounds = Bounds(lower_bounds, upper_bounds)

    best_crit = np.inf
    accepted_x = None
    critvals = []
    for x in start_values:
        res = minimize(
            crit,
            x,
            method="L-BFGS-B",
            bounds=bounds,
        )
        if res.fun <= best_crit:
            accepted_x = res.x
        critvals.append(res.fun)

    return {
        "x": accepted_x,
        "std": np.std(critvals),
        "n_iterations": None,
        "success": None,
    }


def slsqp_sphere(model, x_candidate):
    crit, grad = get_crit_and_grad(model)
    constraints = get_constraints()

    res = minimize(
        crit,
        x_candidate,
        method="slsqp",
        jac=grad,
        constraints=constraints,
    )

    return {
        "x": res.x,
        "success": res.success,
        "n_iterations": res.nit,
    }


def get_crit_and_grad(model):
    def _crit(x, c, g, h):
        return c + x @ g + 0.5 * x @ h @ x

    def _grad(x, g, h):
        return g + x @ h

    crit = partial(_crit, c=model.intercept, g=model.linear_terms, h=model.square_terms)
    grad = partial(_grad, g=model.linear_terms, h=model.square_terms)

    return crit, grad


def get_constraints():
    def _constr_fun(x):
        return x @ x

    def _constr_jac(x):
        return 2 * x

    constr = NonlinearConstraint(
        fun=_constr_fun,
        lb=-np.inf,
        ub=1,
        jac=_constr_jac,
    )

    return (constr,)
