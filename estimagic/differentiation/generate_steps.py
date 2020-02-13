import numpy as np

from estimagic.optimization.utilities import namedtuple_from_kwargs


def generate_steps(
    x,
    method,
    n_steps,
    target,
    base_step=None,
    lower_bound=None,
    upper_bound=None,
    step_ratio=2.0,
    min_step=1e-8,
):
    """Generate steps for numerical differentiation.

    Args:
        x (np.ndarray): 1d array at which the derivative is evaluated
        method (str): One of ["central", "forward", "backward"], default "central".
        n_steps (int): Number of steps needed. For central methods, this is
            the number of steps per direction.
        target (str): One of ["gradient", "jacobian", "hessian"], used to generate
            the base step if not provided.
        base_step (np.ndarray, optional): 1d array of the same length as x with the
            absolute value of the first step. If the base_step conflicts with bounds,
            generate_steps will modify it. If base step is not provided, it will be
            determined as according to the rule of thumb outlined below as long as
            this does not conflict with min_step
        lower_bound (np.ndarray): 1d array with lower bounds for each parameter.
        upper_bound (np.ndarray): 1d array with upper bounds for each parameter.
        step_ratio (float or array): Ratio between two consecutive steps in the
            same direction. default 2.0. Has to be larger than one. step ratio
            is only used if n_steps > 1.
        min_step (float, array or "optimal"): Minimal possible step size that can
            be chosen to accomodate bounds. Default 1e-8 which is square-root of
            machine accurracy for 64 bit floats. If min_step is an array, it has to
            be have the same shape as x. If "optimal", step size is not decreased
            beyond what is optimal according to the rule of thumb.

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

    The rule of thumb for the generation of base_steps is:
    - gradient and jacobian: np.finfo(float) ** (1 / 2) * np.maximum(np.abs(x), 0.1)
    - hessian: np.finfo(float) ** (1 / 3) * np.maximum(np.abs(x), 0.1)
    Where EPS is machine accuracy retrieved by np.finfo(float).eps. This rule of
    thumb is also used in statsmodels and scipy.

    The step generation is bound aware and will try to find a good solution if
    any step would violate a bound. For this, we use the following rules until
    no bounds are violated:

    1. If a one sided method is used, flip to the direction with more distance
        to the bound.
    2. Decrease the base step, unless this would mean to go below min_step.
    3. Set the conflicting steps to NaN, which means that this step won't be
        usable in the calculation of derivatives. All derivative functions can
        handle NaNs and will produce the best possible derivative estimate given
        the remaining steps. If all steps of one parameter are set to NaN, no
        derivative estimate will be produced for that parameter. If this happens,
        the user will be warned but no error will be raised.

    """
    base_step = _calculate_or_validate_base_step(base_step, x, target, min_step)
    min_step = base_step if min_step == "optimal" else min_step

    pos = step_ratio ** np.arange(n_steps) * base_step.reshape(-1, 1)
    neg = -pos.copy()

    if method in ["forward", "backward"]:
        pos, neg = _set_unused_side_to_nan(
            x, pos, neg, method, lower_bound, upper_bound
        )

    pos, neg = _rescale_to_accomodate_bounds(
        base_step, pos, neg, lower_bound, upper_bound, min_step
    )

    pos[pos > upper_bound.reshape(-1, 1)] = np.nan
    neg[neg < lower_bound.reshape(-1, 1)] = np.nan

    steps = namedtuple_from_kwargs(pos=pos.T, neg=neg.T)

    return steps


def _calculate_or_validate_base_step(base_step, x, target, min_step):
    if base_step is None:
        eps = np.finfo(float).eps
        if target == "hessian":
            base_step = eps ** (1 / 3) * np.maximum(np.abs(x), 0.1)
        elif target in ["gradient", "jacobian"]:
            base_step = eps ** (1 / 2) * np.maximum(np.abs(x), 0.1)
        else:
            raise ValueError(f"Invalid target: {target}.")
        base_step[base_step < min_step] = min_step
    elif base_step.shape != x.shape:
        raise ValueError("base_step has to have the same shape as x.")
    elif (base_step <= min_step).any():
        raise ValueError("base_step must be larger than min_step.")
    return base_step


def _set_unused_side_to_nan(x, pos, neg, method, lower_bound, upper_bound):
    pos = pos.copy()
    neg = neg.copy()
    larger_side = np.where((upper_bound - x) >= (x - lower_bound), np.ones_like(x), -1)
    max_abs_step = pos[:, -1]
    if method == "forward":
        side = np.where((upper_bound - x) >= max_abs_step, np.ones_like(x), larger_side)
    else:
        side = np.where(
            (x - lower_bound) >= max_abs_step, (-np.ones_like(x)), larger_side
        )

    pos[side == -1] = np.nan
    neg[side == 1] = np.nan
    return pos, neg


def _rescale_to_accomodate_bounds(
    base_step, pos, neg, lower_bound, upper_bound, min_step
):
    pos_needed_scaling = _fillna(upper_bound / np.nanmax(pos, axis=1), 1).clip(0, 1)
    neg_needed_scaling = _fillna(lower_bound / np.nanmin(neg, axis=1), 1).clip(0, 1)
    needed_scaling = np.minimum(pos_needed_scaling, neg_needed_scaling)

    min_possible_scaling = min_step / base_step

    scaling = np.maximum(needed_scaling, min_possible_scaling).reshape(-1, 1)

    pos = pos * scaling
    neg = neg * scaling
    return pos, neg


def _fillna(x, val):
    return np.where(np.isnan(x), val, x)
