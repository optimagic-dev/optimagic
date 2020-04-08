"""Copy code from numdifftools.extrapolation.Richardson"""
import numpy as np
from scipy.linalg import pinv
from scipy.ndimage.filters import convolve1d


def richardson_extrapolation(
    sequence, step_sizes, order=1, exponentiation_step=1, num_terms=2, step_ratio=2.0
):
    """Apply Richardson extrapolation to sequence.

    Suppose you have a series expansion

        L = f(h) + a0 * h^p_0 + a1 * h^p_1+ a2 * h^p_2 + ... ,

    where p_i = order + exponentiation_step * i  and f(h) -> L as h -> 0, but f(0) != L.

    With ``order`` = 1 and ``exponentiation_step`` = 1 we therefore get
        L = f(h) + a0 * h^1 + a1 * h^2 + a2 * h^3 + ...

    If we evaluate the right hand side for different stepsizes h we can fit a polynomial
    to that sequence of approximations and use the estimated intercept as a better
    approximation for L. Further, we can compute estimation error of our approximation.

    Args:
        sequence (np.ndarray): The sequence of which we want to approximate the limit.
            Has dimension (k x n x m), where k denotes the number of sequence elements
            and an element ``sequence[l, :, :]`` denotes the (n x m) dimensional element

        step_sizes (np.ndarray): The step sizes used to construct the sequences.
            The array is of length k where ``step_sizes[l]`` was used to construct
            ``sequence[l, :, :]`` for l = 1, ..., k.

        order (int): Initial order of the approximation error of sequence elements.
            For central differences derivative approximation ``order`` = 1.

        exponentiation_step (int): ?

        num_terms (int?): ? Related to the number of final outputs?

        step_ratio (float): ?

    Returns:
        limit (np.ndarray): The refined limit.
        error (np.ndarray): The error approximation of ``limit``.
        step_sizes (np.ndarray): ?

    """
    sequence_length = sequence.shape[0]
    num_steps = len(step_sizes)

    assert sequence_length == num_steps

    rule = _rule(
        sequence_length=sequence_length,
        num_terms=num_terms,
        step_ratio=step_ratio,
        exponentiation_step=exponentiation_step,
        order=order,
    )

    nr = rule.size - 1
    m = sequence_length - nr
    mm = np.minimum(sequence_length, m + 1)

    new_sequence = convolve1d(sequence, rule[::-1], axis=0, origin=nr // 2)

    abserr = _estimate_error(
        new_sequence=new_sequence[:mm],
        old_sequence=sequence,
        step_sizes=step_sizes,
        rule=rule,
    )

    limit = new_sequence[:m]
    error = abserr[:m]

    return limit, error, step_sizes[:m]


def _rule(sequence_length, num_terms, step_ratio, exponentiation_step, order):
    """

    Args:
        sequence_length:
        num_terms:
        step_ratio:
        exponentiation_step:
        order:

    Returns:

    """
    num_terms = np.minimum(num_terms, sequence_length - 1)
    if num_terms > 0:
        r_mat = _r_matrix(step_ratio, exponentiation_step, num_terms, order)
        return pinv(r_mat)[0]
    return np.ones((1,))


def _r_matrix(step_ratio, exponentiation_step, num_terms, order):
    """

    Args:
        step_ratio (float):
        exponentiation_step (int):
        num_terms (int):
        order (int):

    Returns:
        r_mat (np.ndarray):

    """
    rows, cols = np.ogrid[: num_terms + 1, :num_terms]
    r_mat = np.ones((num_terms + 1, num_terms + 1))
    r_mat[:, 1:] = (1.0 / step_ratio) ** (rows @ (exponentiation_step * cols + order))
    return r_mat


def _estimate_error(new_sequence, old_sequence, step_sizes, rule):
    """

    Args:
        new_sequence:
        old_sequence:
        step_sizes:
        rule:

    Returns:

    """
    m = new_sequence.shape[0]
    mo = old_sequence.shape[0]

    cov1 = np.sum(rule ** 2)  # 1 spare dof
    fact = np.maximum(12.7062047361747 * np.sqrt(cov1), np.finfo(float).eps * 10.0)

    if mo < 2:
        return (np.abs(new_sequence) * np.finfo(float).eps + step_sizes) * fact
    if m < 2:
        delta = np.diff(old_sequence, axis=0)
        tol = np.maximum(np.abs(old_sequence[:-1]), np.abs(old_sequence[1:])) * fact
        err = np.abs(delta)
        converged = err <= tol
        abserr = err[-m:] + np.where(
            converged[-m:], tol[-m:] * 10, abs(new_sequence - old_sequence[-m:]) * fact
        )
        return abserr

    err = np.abs(np.diff(new_sequence, axis=0)) * fact
    tol = (
        np.maximum(np.abs(new_sequence[1:]), np.abs(new_sequence[:-1]))
        * np.finfo(float).eps
        * fact
    )
    converged = err <= tol
    abserr = err + np.where(
        converged, tol * 10, abs(new_sequence[:-1] - old_sequence[-m + 1 :]) * fact
    )
    return abserr


def _convolve(sequence, rule, **kwds):
    """Probably not needed for estimagic.

    Wrapper around scipy.ndimage.convolve1d that allows complex input.
    """
    dtype = np.result_type(float, np.ravel(sequence)[0])
    seq = np.asarray(sequence, dtype=dtype)
    if np.iscomplexobj(seq):
        return convolve1d(seq.real, rule, **kwds) + 1j * convolve1d(
            seq.imag, rule, **kwds
        )
    return convolve1d(seq, rule, **kwds)
