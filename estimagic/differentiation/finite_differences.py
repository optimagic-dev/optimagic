"""Finite difference formulae for jacobians and hessians.

All functions in this module should not only work for the simple case
of one positive and/or one negative step, but also for the Richardson
Extrapolation case with several positive and/or several negative steps.

Since steps and evals contain NaNs, we have to make sure that the functions do not raise
warnings or errors for that case.

"""


def jacobian(evals, steps, f0, method):
    """Calculate a Jacobian estimate with finite differences according to method.

    Notation: f:R^dim_x -> R^dim_f. We compute the derivative at x0, with f0 = f(x0).

    Args:
        evals (namedtuple): It has the fields called pos and neg for evaluations with
            positive and negative steps, respectively. Each field is a numpy array
            of shape (n_steps, dim_f, dim_x). It contains np.nan for evaluations that
            failed or were not attempted because a one-sided derivative rule was chosen.
        steps (namedtuple): Namedtuple with the fields pos and neg. Each field
            contains a numpy array of shape (n_steps, dim_x) with the steps in
            the corresponding direction. The steps are always symmetric, in the sense
            that steps.neg[i, j] = - steps.pos[i, j] unless one of them is NaN.
        f0 (numpy.ndarray): Numpy array of length dim_f with the output of the function
            at the user supplied parameters.
        method (str): One of ["forward", "backward", "central"]

    Returns:
        jac (numpy.ndarray): Numpy array of shape (n_steps, dim_f, dim_x) with estimated
            Jacobians. I.e. there are n_step jacobian estimates.

    """
    n_steps, dim_f, dim_x = evals.pos.shape
    if method == "forward":
        diffs = evals.pos - f0.reshape(1, dim_f, 1)
        jac = diffs / steps.pos.reshape(n_steps, 1, dim_x)
    elif method == "backward":
        diffs = evals.neg - f0.reshape(1, dim_f, 1)
        jac = diffs / steps.neg.reshape(n_steps, 1, dim_x)
    elif method == "central":
        diffs = evals.pos - evals.neg
        deltas = steps.pos - steps.neg
        jac = diffs / deltas.reshape(n_steps, 1, dim_x)
    else:
        raise ValueError("Method has to be 'forward', 'backward' or 'central'.")
    return jac
