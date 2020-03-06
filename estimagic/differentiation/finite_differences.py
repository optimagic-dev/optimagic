"""Finite difference formulae for jacobians and hessians.

To harmonize the interface, all functions in this module take the
three arguments: the function evaluations `evals`, the steps and the function value at the position of interest `f0`, even though f0 is not used
for central methods.

All functions in this module should not only work for the simple case
of one positive and/or one negative step, but also for the Richardson
Extrapolation case with several positive and/or several negative steps.

Since steps and evals contain NaNs, we have to make sure that the functions do not raise
warnings or errors for that case.

Naming conventions:

- n_steps: number of steps. This is one if no Richardson extrapolation is used.
- dim_f: Length of the output of the functions that is being differentiated.
- dim_x: Length of the parameter vector with respect to which derivatives are taken.

"""


def jacobian_forward(evals, steps, f0):
    """Calculate a Jacobian estimate with forward differences.

    Args:
        evals (namedtuple): It has the fields called pos and neg for evaluations with
            positive and negative steps, respectively. Each field is a numpy array
            of shape (n_steps, dim_f, dim_x). It contains np.nan for evaluations that
            failed or were not attempted because a one-sided derivative rule was chosen.
        steps (namedtuple): Namedtuple with the fields pos and neg. Each field
            contains a numpy array of shape (n_steps, dim_x with the steps in
            the corresponding direction. The steps always symmetric, in the sense
            that steps.neg[i, j] = - steps.pos[i, j] unless one of them is NaN.
        f0 (np.ndarray): Numpy array of length dim_f with the output of the function at
            the user supplied parameters.

    Returns:
        np.ndarray: Numpy array of shape (n_steps, dim_f, dim_x) with estimated
            Jacobians. I.e. there are n_step jacobian estimates.

    """
    n_steps, dim_f, dim_x = evals.pos.shape
    diffs = evals.pos - f0.reshape(1, dim_f, 1)
    return diffs / steps.pos.reshape(n_steps, 1, dim_x)


def jacobian_backward(evals, steps, f0):
    """Calculate a Jacobian estimate with backward differences.

    Args:
        evals (namedtuple): It has the fields called pos and neg for evaluations with
            positive and negative steps, respectively. Each field is a numpy array
            of shape (n_steps, dim_f, dim_x). It contains np.nan for evaluations that
            failed or were not attempted because a one-sided derivative rule was chosen.
        steps (namedtuple): Namedtuple with the fields pos and neg. Each field
            contains a numpy array of shape (n_steps, dim_x with the steps in
            the corresponding direction. The steps always symmetric, in the sense
            that steps.neg[i, j] = - steps.pos[i, j] unless one of them is NaN.
        f0 (np.ndarray): Numpy array of length dim_f with the output of the function at
            the user supplied parameters.

    Returns:
        np.ndarray: Numpy array of shape (n_steps, dim_f, dim_x) with estimated
            Jacobians. I.e. there are n_step jacobian estimates.

    """
    n_steps, dim_f, dim_x = evals.neg.shape
    diffs = evals.neg - f0.reshape(1, dim_f, 1)
    return diffs / steps.neg.reshape(n_steps, 1, dim_x)


def jacobian_central(evals, steps, f0):
    """Calculate a Jacobian estimate with central differences.

    Args:
        evals (namedtuple): It has the fields called pos and neg for evaluations with
            positive and negative steps, respectively. Each field is a numpy array
            of shape (n_steps, dim_f, dim_x). It contains np.nan for evaluations that
            failed or were not attempted because a one-sided derivative rule was chosen.
        steps (namedtuple): Namedtuple with the fields pos and neg. Each field
            contains a numpy array of shape (n_steps, dim_x with the steps in
            the corresponding direction. The steps always symmetric, in the sense
            that steps.neg[i, j] = - steps.pos[i, j] unless one of them is NaN.
        f0 (np.ndarray): Numpy array of length dim_f with the output of the function at
            the user supplied parameters. This is only an argument to have a unified interface.. 
            It is not used for calculating the Jacobian with central differences.

    Returns:
        np.ndarray: Numpy array of shape (n_steps, dim_f, dim_x) with estimated
            Jacobians. I.e. there are n_step jacobian estimates.

    """
    n_steps, dim_f, dim_x = evals.pos.shape
    diffs = evals.pos - evals.neg
    deltas = steps.pos - steps.neg
    return diffs / deltas.reshape(n_steps, 1, dim_x)
