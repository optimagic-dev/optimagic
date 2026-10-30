"""Step subproblem of the BHHH algorithm.

Each iteration minimizes the quadratic model g'p + p'Bp / 2 of the objective, where g
is the gradient and B the OPG with floored eigenvalues. With bounds, the model is
minimized over the box of feasible steps, as in Bertsekas' projected Newton method
(:cite:`Bertsekas1982`) and the bounded variants of TOMS 717. Clipping or projecting
the unconstrained step instead can stall at non-optimal points.

"""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import lsq_linear

from optimagic.optimizers._bhhh.linalg import OPGFactorization, unconstrained_step


def compute_step(
    factorization: OPGFactorization,
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    """Minimize the quadratic OPG model subject to bounds on the step.

    The decrement is twice the reduction of the model achieved by the step. Without
    binding bounds it equals g'B⁻¹g. It is zero exactly at points that satisfy the
    first order conditions of the bound constrained problem.

    Args:
        factorization: The factorization of the OPG.
        lower: Lower bounds on the step, shape (k,). Entries are non-positive and can
            be -inf.
        upper: Upper bounds on the step, shape (k,). Entries are non-negative and can
            be inf.

    Returns:
        - The step, shape (k,).
        - The decrement.

    """
    step, decrement = unconstrained_step(factorization)
    if np.all((step >= lower) & (step <= upper)):
        return step, decrement
    return _box_constrained_step(factorization, lower, upper)


def _box_constrained_step(
    factorization: OPGFactorization,
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    """Minimize the quadratic OPG model over a box with bounded-variable least squares.

    With the scaled step q = Dp and the factor A = Λ^(1/2)V' of the scaled OPG, the
    model g'p + p'Bp / 2 equals ||Aq + c||² / 2 - ||c||² / 2 with c = Λ^(-1/2)V'D⁻¹g.
    Working in scaled parameters keeps A well conditioned.

    """
    scale = factorization.scale
    lam = factorization.floored_eigenvalues
    v = factorization.eigenvectors
    scaled_gradient = factorization.scaled_gradient
    a = np.sqrt(lam)[:, None] * v.T
    c = (v.T @ scaled_gradient) / np.sqrt(lam)
    res = lsq_linear(a, -c, bounds=(lower * scale, upper * scale), method="bvls")
    scaled_step = res.x
    # Computing the decrease directly avoids cancellation between ||c||² and the
    # residual sum of squares when the gradient pushes hard against a bound.
    decrement = float(
        -2 * scaled_gradient @ scaled_step - np.sum((a @ scaled_step) ** 2)
    )
    step = np.clip(scaled_step / scale, lower, upper)
    # Unscaling can move steps on a bound slightly into the interior.
    step = np.where(res.active_mask == -1, lower, step)
    step = np.where(res.active_mask == 1, upper, step)
    return step, max(decrement, 0.0)


def max_feasible_step_length(
    x: NDArray[np.float64],
    direction: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> float:
    """Return the largest t such that x + t * direction is within the bounds.

    Args:
        x: Current parameters within the bounds, shape (k,).
        direction: Search direction, shape (k,).
        lower: Lower bounds on the parameters, shape (k,). Can contain -inf.
        upper: Upper bounds on the parameters, shape (k,). Can contain inf.

    Returns:
        The largest feasible step length; inf if the direction never leaves the box.

    """
    with np.errstate(divide="ignore", invalid="ignore"):
        to_upper = np.where(direction > 0, (upper - x) / direction, np.inf)
        to_lower = np.where(direction < 0, (lower - x) / direction, np.inf)
    return float(
        min(np.min(to_upper, initial=np.inf), np.min(to_lower, initial=np.inf))
    )
