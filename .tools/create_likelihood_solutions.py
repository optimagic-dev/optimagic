"""Compute the reference solutions of the likelihood benchmark sets.

Writes ``src/optimagic/benchmarking/likelihood/solutions.py``. Run it after changing a
model or an instance of the likelihood benchmark sets:

    python .tools/create_likelihood_solutions.py [instance_name ...]

Without arguments, all solutions are recomputed. With instance names, only those are
recomputed and the others are kept.

Each solution is the best of several local optimizations (L-BFGS-B from the good start,
the bad start, the true params and a few perturbed starts), polished with projected
Newton steps on a numerical Hessian of the analytic gradient.

"""

import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, optimize

from optimagic.benchmarking.likelihood.models import SimulatedLikelihood
from optimagic.benchmarking.likelihood.problems import LIKELIHOOD_SETS
from optimagic.config import OPTIMAGIC_ROOT

OUTPUT_PATH = OPTIMAGIC_ROOT / "benchmarking" / "likelihood" / "solutions.py"

# a variable is treated as active if it is on its bound and the gradient does not
# point into the feasible set by more than this
ACTIVE_TOLERANCE = 1e-6


def main(names: list[str]) -> None:
    instances = {
        instance.name: instance
        for instances in LIKELIHOOD_SETS.values()
        for instance in instances
    }
    if names:
        unknown = set(names) - set(instances)
        if unknown:
            raise ValueError(f"Unknown instances: {sorted(unknown)}")
        from optimagic.benchmarking.likelihood.solutions import LIKELIHOOD_SOLUTIONS

        solutions = dict(LIKELIHOOD_SOLUTIONS)
    else:
        names = list(instances)
        solutions = {}

    with ProcessPoolExecutor() as executor:
        results = executor.map(compute_reference_solution, names)
        for name, (solution, report) in zip(names, results, strict=True):
            solutions[name] = solution
            print(f"{name:52s} {report}", flush=True)

    ordered = {name: solutions[name] for name in instances if name in solutions}
    OUTPUT_PATH.write_text(_create_module_code(ordered))


def compute_reference_solution(name: str) -> tuple[dict[str, Any], str]:
    """Compute the reference solution of one instance.

    Returns:
        The solution, with the entries "params", "value", "opg_hessian_ratio" and
        "other_local_optima", and a one line report.

    """
    instance = next(
        instance
        for instances in LIKELIHOOD_SETS.values()
        for instance in instances
        if instance.name == name
    )
    likelihood = instance.build()
    lower, upper = _get_bounds(likelihood)

    starts = [likelihood.good_start, likelihood.bad_start]
    start_names = ["good", "bad"]
    if likelihood.true_params is not None:
        starts.append(np.clip(likelihood.true_params, lower, upper))
        start_names.append("true")
    rng = np.random.default_rng(0)
    n_perturbed = 3 if likelihood.n_params <= 25 else 1
    for _ in range(n_perturbed):
        scale = 0.5 * (1 + np.abs(likelihood.good_start))
        perturbed = likelihood.good_start + scale * rng.normal(size=likelihood.n_params)
        starts.append(np.clip(perturbed, lower, upper))
        start_names.append("perturbed")

    optima = []
    for start in starts:
        x = _minimize_lbfgsb(likelihood, start, lower, upper)
        x = _polish(likelihood, x, lower, upper)
        optima.append((_value(likelihood, x), x))
    values = np.array([value for value, _ in optima])
    best_value, best_x = min(optima, key=lambda pair: pair[0])

    free = ~_is_active(likelihood, best_x, lower, upper)
    jac = likelihood.jac(best_x)[:, free]
    opg = jac.T @ jac
    hessian = _numerical_hessian(likelihood, best_x, lower, upper)[np.ix_(free, free)]
    ratio = linalg.eigh(opg, hessian, eigvals_only=True)
    decrement = _opg_decrement(likelihood, best_x, free)

    is_worse = values > best_value + 1e-6 * (1 + abs(best_value))
    solution = {
        "params": best_x,
        "value": best_value,
        "opg_hessian_ratio": (float(ratio.min()), float(ratio.max())),
        "other_local_optima": bool(is_worse.any()),
    }
    worse_starts = [
        name for name, worse in zip(start_names, is_worse, strict=True) if worse
    ]
    report = (
        f"value={best_value:.6f} decrement={decrement:.1e} "
        f"n_active={np.sum(~free)} ratio=[{ratio.min():.3f}, {ratio.max():.3f}] "
        f"worse_local_optima_from={worse_starts}"
    )
    return solution, report


def _get_bounds(
    likelihood: SimulatedLikelihood,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    k = likelihood.n_params
    lower = likelihood.lower_bounds
    upper = likelihood.upper_bounds
    lower = np.full(k, -np.inf) if lower is None else lower
    upper = np.full(k, np.inf) if upper is None else upper
    return lower, upper


def _value(likelihood: SimulatedLikelihood, x: NDArray[np.float64]) -> float:
    with np.errstate(all="ignore"):
        value = float(likelihood.fun(x).sum())
    return value if np.isfinite(value) else np.inf


def _gradient(
    likelihood: SimulatedLikelihood, x: NDArray[np.float64]
) -> NDArray[np.float64]:
    with np.errstate(all="ignore"):
        return likelihood.jac(x).sum(axis=0)


def _minimize_lbfgsb(
    likelihood: SimulatedLikelihood,
    start: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> NDArray[np.float64]:
    def fun_and_grad(x: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        value = _value(likelihood, x)
        if not np.isfinite(value):
            return 1e300, np.zeros_like(x)
        return value, _gradient(likelihood, x)

    res = optimize.minimize(
        fun_and_grad,
        start,
        jac=True,
        method="L-BFGS-B",
        bounds=list(zip(lower, upper, strict=True)),
        options={"maxiter": 100_000, "maxfun": 200_000, "ftol": 1e-15, "gtol": 1e-10},
    )
    return res.x


def _is_active(
    likelihood: SimulatedLikelihood,
    x: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> NDArray[np.bool_]:
    gradient = _gradient(likelihood, x)
    at_lower = (x <= lower) & (gradient >= -ACTIVE_TOLERANCE)
    at_upper = (x >= upper) & (gradient <= ACTIVE_TOLERANCE)
    return at_lower | at_upper


def _numerical_hessian(
    likelihood: SimulatedLikelihood,
    x: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Differentiate the analytic gradient; one-sided next to bounds."""
    k = len(x)
    hessian = np.empty((k, k))
    for j in range(k):
        h = 1e-5 * max(1.0, abs(x[j]))
        step = np.zeros(k)
        step[j] = h
        if x[j] - h < lower[j]:
            column = (_gradient(likelihood, x + step) - _gradient(likelihood, x)) / h
        elif x[j] + h > upper[j]:
            column = (_gradient(likelihood, x) - _gradient(likelihood, x - step)) / h
        else:
            forward = _gradient(likelihood, x + step)
            backward = _gradient(likelihood, x - step)
            column = (forward - backward) / (2 * h)
        hessian[:, j] = column
    return 0.5 * (hessian + hessian.T)


def _polish(
    likelihood: SimulatedLikelihood,
    x: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Polish a local solution, trying to put variables close to a bound on it.

    Close to a bound, the criterion can be so flat that L-BFGS-B stops before reaching
    the bound (e.g. the stochastic frontier with wrong skew). Ties in the criterion
    value are resolved in favor of the solution on the bound.

    """
    x = _polish_with_newton(likelihood, x, lower, upper)
    distance = np.minimum(x - lower, upper - x)
    close = (distance > 0) & (distance < 1e-4 * (1 + np.abs(x)))
    if close.any():
        snapped = np.where(close & (x - lower < upper - x), lower, x)
        snapped = np.where(close & (x - lower >= upper - x), upper, snapped)
        snapped = _polish_with_newton(likelihood, snapped, lower, upper, fixed=close)
        value = _value(likelihood, x)
        is_tie = _value(likelihood, snapped) <= value + 1e-12 * (1 + abs(value))
        if is_tie and _is_active(likelihood, snapped, lower, upper)[close].all():
            x = snapped
    return x


def _polish_with_newton(
    likelihood: SimulatedLikelihood,
    x: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    fixed: NDArray[np.bool_] | None = None,
    max_iterations: int = 30,
) -> NDArray[np.float64]:
    """Polish a solution with projected Newton steps on the free variables.

    Variables that are active or fixed are not changed.

    Far from the solution, steps are accepted if they decrease the criterion. Close to
    the solution, where the decrease is below the precision of the criterion value,
    full steps are accepted if they decrease the decrement in the OPG metric.

    """
    fixed = np.zeros(len(x), dtype=bool) if fixed is None else fixed
    value = _value(likelihood, x)
    for _ in range(max_iterations):
        free = ~(_is_active(likelihood, x, lower, upper) | fixed)
        gradient = _gradient(likelihood, x)[free]
        hessian = _numerical_hessian(likelihood, x, lower, upper)[np.ix_(free, free)]
        eigenvalues, eigenvectors = np.linalg.eigh(hessian)
        floor = 1e-12 * np.abs(eigenvalues).max()
        eigenvalues = np.maximum(np.abs(eigenvalues), floor)
        direction = np.zeros_like(x)
        direction[free] = -eigenvectors @ ((eigenvectors.T @ gradient) / eigenvalues)
        decrement = -gradient @ direction[free]
        if decrement < 1e-24:
            break

        if decrement < 1e-8:
            candidate = np.clip(x + direction, lower, upper)
            if _opg_decrement(likelihood, candidate, free) >= _opg_decrement(
                likelihood, x, free
            ):
                break
            x, value = candidate, _value(likelihood, candidate)
            continue

        step_length = 1.0
        while step_length > 1e-8:
            candidate = np.clip(x + step_length * direction, lower, upper)
            candidate_value = _value(likelihood, candidate)
            if candidate_value <= value:
                break
            step_length /= 2
        else:
            break
        x, value = candidate, candidate_value
    return x


def _opg_decrement(
    likelihood: SimulatedLikelihood, x: NDArray[np.float64], free: NDArray[np.bool_]
) -> float:
    """Compute g @ inv(J.T @ J) @ g on the free variables."""
    with np.errstate(all="ignore"):
        jac = likelihood.jac(x)[:, free]
    gradient = jac.sum(axis=0)
    return float(gradient @ np.linalg.lstsq(jac, np.ones(len(jac)), rcond=None)[0])


def _create_module_code(solutions: dict[str, dict[str, Any]]) -> str:
    lines = [
        '"""Reference solutions of the likelihood benchmark sets.',
        "",
        "Created by .tools/create_likelihood_solutions.py; do not edit by hand.",
        "",
        '"""',
        "",
        "from typing import Any",
        "",
        "# fmt: off",
        "LIKELIHOOD_SOLUTIONS: dict[str, dict[str, Any]] = {",
    ]
    for name, solution in solutions.items():
        lines.append(f'    "{name}": {{')
        lines.append('        "params": [')
        params = [repr(float(p)) for p in solution["params"]]
        for i in range(0, len(params), 3):
            lines.append("            " + ", ".join(params[i : i + 3]) + ",")
        lines.append("        ],")
        lines.append(f'        "value": {float(solution["value"])!r},')
        low, high = solution["opg_hessian_ratio"]
        lines.append(f'        "opg_hessian_ratio": ({low:.4g}, {high:.4g}),')
        lines.append(f'        "other_local_optima": {solution["other_local_optima"]},')
        lines.append("    },")
    lines += ["}", "# fmt: on", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main(sys.argv[1:])
