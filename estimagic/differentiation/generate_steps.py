import numpy as np

from estimagic.optimization.utilities import namedtuple_from_kwargs


def generate_steps(
    x,
    method,
    n_steps,
    target,
    base_steps,
    lower_bounds,
    upper_bounds,
    step_ratio,
    min_steps,
):
    """Generate steps for finite differences with or without Richardson Extrapolation.

    Args:
        x (np.ndarray): 1d array at which the derivative is evaluated
        method (str): One of ["central", "forward", "backward"].
        n_steps (int): Number of steps needed. For central methods, this is
            the number of steps per direction. It is one if no Richardson extrapolation
            is used.
        target (str): One of ["gradient", "jacobian", "hessian"]. This is used to choose
            the appropriate rule of thumb for the base_steps.
        base_steps (np.ndarray or None): 1d array of the same length as x with the
            absolute value of the first step. If the base_steps conflicts with bounds,
            generate_steps will modify it. If base step is None, it will be
            determined as according to the rule of thumb outlined below as long as
            this does not conflict with min_steps
        lower_bounds (np.ndarray or None): 1d array with lower bounds for x.
        upper_bounds (np.ndarray or None): 1d array with upper bounds for x.
        step_ratio (float): Ratio between two consecutive steps in the
            same direction. Has to be larger than one. step ratio
            is only used if n_steps > 1.
        min_steps (np.ndarray or None): Minimal possible step sizes that can be chosen
            to accomodate bounds. Needs to have same length as x. I None, min_steps is
            set to base_step, i.e step size is not decreased beyond what is optimal
            according to the rule of thumb.

    Returns:
        steps (namedtuple): Namedtuple with the fields pos and neg. Each field
            contains a numpy array of shape (n_steps, len(x)) with the steps in
            the corresponding direction. The steps always symmetric, in the sense
            that steps.neg[i, j] = - steps.pos[i, j] unless one of them is NaN.

    steps can be used to construct x-vectors at which the the function has to be
    evaluated. How the vectors are constructed from the steps differs between first
    (gradient, jacobian) and second derivatves (hessian). Note that both arrays are
    returned even for one-sided methods, because bounds might make it necessary to
    flip the direction of the method.

    The rule of thumb for the generation of base_stepss is:
    - gradient and jacobian: np.finfo(float) ** (1 / 2) * np.maximum(np.abs(x), 0.1)
    - hessian: np.finfo(float) ** (1 / 3) * np.maximum(np.abs(x), 0.1)
    Where EPS is machine accuracy retrieved by np.finfo(float).eps. This rule of
    thumb is also used in statsmodels and scipy.

    The step generation is bound aware and will try to find a good solution if
    any step would violate a bound. For this, we use the following rules until
    no bounds are violated:

    1. If a one sided method is used, flip to the direction with more distance
        to the bound.
    2. Decrease the base_steps, unless this would mean to go below min_steps. By default
        min_steps is equal to base_steps, so no squeezing happens unless explicitly
        requested by setting a smaller min_step.
    3. Set the conflicting steps to NaN, which means that this step won't be
        usable in the calculation of derivatives. All derivative functions can
        handle NaNs and will produce the best possible derivative estimate given
        the remaining steps. If all steps of one parameter are set to NaN, no
        derivative estimate will be produced for that parameter. If this happens,
        the user will be warned but no error will be raised.

    """
    base_steps = _calculate_or_validate_base_steps(base_steps, x, target, min_steps)
    min_steps = base_steps if min_steps is None else min_steps

    upper_bounds = np.full(len(x), np.inf) if upper_bounds is None else upper_bounds
    lower_bounds = np.full(len(x), -np.inf) if lower_bounds is None else lower_bounds

    upper_step_bounds = upper_bounds - x
    lower_step_bounds = lower_bounds - x

    pos = step_ratio ** np.arange(n_steps) * base_steps.reshape(-1, 1)
    neg = -pos.copy()

    if method in ["forward", "backward"]:
        pos, neg = _set_unused_side_to_nan(
            x, pos, neg, method, lower_step_bounds, upper_step_bounds
        )

    pos, neg = _rescale_to_accomodate_bounds(
        base_steps, pos, neg, lower_step_bounds, upper_step_bounds, min_steps
    )

    pos[pos > upper_step_bounds.reshape(-1, 1)] = np.nan
    neg[neg < lower_step_bounds.reshape(-1, 1)] = np.nan

    steps = namedtuple_from_kwargs(pos=pos.T, neg=neg.T)

    return steps


def _calculate_or_validate_base_steps(base_steps, x, target, min_steps):
    if base_steps is None:
        eps = np.finfo(float).eps
        if target == "hessian":
            base_steps = eps ** (1 / 3) * np.maximum(np.abs(x), 0.1)
        elif target in ["gradient", "jacobian"]:
            base_steps = eps ** (1 / 2) * np.maximum(np.abs(x), 0.1)
        else:
            raise ValueError(f"Invalid target: {target}.")
        if min_steps is not None:
            base_steps[base_steps < min_steps] = min_steps
    elif base_steps.shape != x.shape:
        raise ValueError("base_steps has to have the same shape as x.")
    elif min_steps is not None and (base_steps <= min_steps).any():
        raise ValueError("base_steps must be larger than min_steps.")
    return base_steps


def _set_unused_side_to_nan(x, pos, neg, method, lower_step_bounds, upper_step_bounds):
    pos = pos.copy()
    neg = neg.copy()
    larger_side = np.where(
        (upper_step_bounds - x) >= (x - lower_step_bounds), np.ones_like(x), -1
    )
    max_abs_step = pos[:, -1]
    if method == "forward":
        side = np.where(
            (upper_step_bounds - x) >= max_abs_step, np.ones_like(x), larger_side
        )
    else:
        side = np.where(
            (x - lower_step_bounds) >= max_abs_step, (-np.ones_like(x)), larger_side
        )

    pos[side == -1] = np.nan
    neg[side == 1] = np.nan
    return pos, neg


def _rescale_to_accomodate_bounds(
    base_steps, pos, neg, lower_step_bounds, upper_step_bounds, min_steps
):
    """Res

    """
    pos_needed_scaling = _fillna(upper_step_bounds / np.nanmax(pos, axis=1), 1).clip(
        0, 1
    )
    neg_needed_scaling = _fillna(lower_step_bounds / np.nanmin(neg, axis=1), 1).clip(
        0, 1
    )
    needed_scaling = np.minimum(pos_needed_scaling, neg_needed_scaling)

    min_possible_scaling = min_steps / base_steps

    scaling = np.maximum(needed_scaling, min_possible_scaling).reshape(-1, 1)

    pos = pos * scaling
    neg = neg * scaling
    return pos, neg


def _fillna(x, val):
    return np.where(np.isnan(x), val, x)
