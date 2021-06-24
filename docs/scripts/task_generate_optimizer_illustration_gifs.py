from pathlib import Path

import gif
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
import seaborn as sns
from scipy.optimize import minimize

gif.options.matplotlib["dpi"] = 300


# ======================================================================================
# Define example function
# ======================================================================================

WEIGHTS = [
    9.003014962148157,
    -3.383000146393776,
    -0.6037887934635748,
    1.6984454347036886,
    -0.9447426232680957,
    0.2669069434366247,
    -0.04446368897497234,
    0.00460781796708519,
    -0.0003000790127508276,
    1.1934114174145725e-05,
    -2.6471293419570505e-07,
    2.5090819960943964e-09,
]


def example_criterion(x):
    x = _unpack_x(x)
    exponents = np.arange(len(WEIGHTS))
    return WEIGHTS @ x ** exponents


def example_gradient(x):
    x = _unpack_x(x)
    exponents = np.arange(len(WEIGHTS))
    return (WEIGHTS * exponents) @ x ** (exponents - 1).clip(0)


def example_hessian(x):
    x = _unpack_x(x)
    exponents = np.arange(len(WEIGHTS))
    return (WEIGHTS * exponents * (exponents - 1)) @ x ** (exponents - 2).clip(0)


def _unpack_x(x):
    if hasattr(x, "__len__"):
        assert len(x) == 1

    if isinstance(x, pd.DataFrame):
        res = x["value"].to_numpy()[0]
    elif isinstance(x, pd.Series):
        res = x.to_numpy()[0]
    elif isinstance(x, (np.ndarray, list, tuple)):
        res = x[0]
    else:
        res = float(x)
    return res


# ======================================================================================
# Define tools
# ======================================================================================


def minimize_with_history(fun, x0, method, jac=None, hess=None):
    """Dumbed down scipy minimize that returns full history.

    This is really only meant for illustration in this notebook. In particular,
    the following restrictions apply:

    - Only works for 1 dimensional problems
    - does not support all arguments

    """
    history = []

    def wrapped_fun(x, history=history):
        history.append(_unpack_x(x))
        return fun(x)

    res = minimize(wrapped_fun, x0, method=method, jac=jac, hess=hess)
    res.history = history
    return res


def plot_function():
    x_grid = np.linspace(0, 20, 100)
    y_grid = [example_criterion(x) for x in x_grid]
    fig, ax = plt.subplots()
    sns.lineplot(x=x_grid, y=y_grid, ax=ax)
    sns.despine()
    return fig, ax


# ======================================================================================
# Make convergence gifs
# ======================================================================================
DOCS_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = DOCS_ROOT / "source" / "_static" / "images"

algorithms = ["Cobyla", "L-BFGS-B", "Nelder-Mead", "trust-ncg"]

PARMETRIZATON = [(STATIC_DIR / f"{algo.lower()}.gif", algo) for algo in algorithms]


@pytask.mark.parametrize("produces, algorithm", PARMETRIZATON)
def task_create_convergence_gif(produces, algorithm):
    start_x = np.array([2])
    hessian = example_hessian if algorithm == "trust-ncg" else NotImplementedError
    res = minimize_with_history(
        example_criterion, start_x, method=algorithm, jac=example_gradient, hess=hessian
    )

    # repeat the last point to show it longer in the gif
    points = res.history + [res.history[-1]] * 5

    @gif.frame
    def _plot_history(points):
        fig, ax = plot_function()
        sns.rugplot(points, ax=ax)
        plt.plot(
            points[-1],
            example_criterion(points[-1]),
            marker="*",
        )
        sns.despine()

    frames = []
    for i in range(len(points)):
        frames.append(_plot_history(points[: i + 1]))

    gif.save(frames, produces, duration=5, unit="s", between="startend")
