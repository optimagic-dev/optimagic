import numpy as np
from estimagic.optimization.tiktak import draw_exploration_sample
from estimagic.optimization.tranquilo.models import evaluate_model
from scipy.optimize import Bounds
from scipy.optimize import minimize


def solve_thorough(model, lower_bounds, upper_bounds):
    np.random.seed(12345)
    start_values = draw_exploration_sample(
        x=np.zeros(len(lower_bounds)),
        lower=lower_bounds,
        upper=upper_bounds,
        n_samples=100,
        sampling_distribution="uniform",
        sampling_method="sobol",
        seed=1234,
    )

    def crit(x):
        return evaluate_model(model, x)

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
