"""Copy code from numdifftools.extrapolation.Richardson

Notes:
    - correlate with reversed weight is *not* the same as convolve1d
    - Richardson matrix is matrix with weights from equation 25 in numdifftools docs
    - student t quantile of numdifftools does not coincide with that of scipy.stats.t

Problems:
    - If we do left, right, or central differences is decided beforehand and stored
      as information in the steps. But for the extrapolation to work we have to choose
      an order and exponentiation_step for all steps?
    - For sequences which are more than one dimensional, do we select the best limit
      approximation elementwise with respect to the estimation error?

Series expansions:

    Central Differences.
        Derivative approximation via central difference is given by
            g(h) := [f(x + h) - f(x - h)] / 2h = f'(x) + r(x, h)

        If we expand the remainder term r(x, h) we get
            r(x, h) = a0*(h**2) + a1*(h**4) + a2*(h**6) + ...
        with a0 = f''(x) / 2!, a1 = f'''(x) / 3! etc.

        Rearanging terms we can write
            L := f'(x) = g(h) - r(x, h) = g(h) + O(h**2)

"""
import numpy as np
from scipy.linalg import pinv
from scipy.ndimage.filters import convolve1d
from scipy.stats import t


EPS = np.finfo(float).eps
TQUANTILE = t(df=1).ppf(0.975)  # 12.7062047361747


def richardson_extrapolation(
    sequence, steps, num_terms=2, order=1, exponentiation_step=1,
):
    """Apply Richardson extrapolation to sequence.

    Suppose you have a series expansion

        L = g(h) + a0 * h^p_0 + a1 * h^p_1+ a2 * h^p_2 + ... ,

    where p_i = order + exponentiation_step * i  and g(h) -> L as h -> 0, but g(0) != L.

    With ``order`` = 1 and ``exponentiation_step`` = 1 we therefore get
        L = g(h) + a0 * h^1 + a1 * h^2 + a2 * h^3 + ...

    If we evaluate the right hand side for different stepsizes h we can fit a polynomial
    to that sequence of approximations and use the estimated intercept as a better
    approximation for L. Further, we can compute estimation errors of our approximation.

    Args:
        sequence (np.ndarray): The sequence of which we want to approximate the limit.
            Has dimension (k x n x m), where k denotes the number of sequence elements
            and an element ``sequence[l, :, :]`` denotes the (n x m) dimensional element

        steps (namedtuple): Namedtuple with the field names pos and neg. Each field
            contains a numpy array of shape (n_steps, len(x)) with the steps in
            the corresponding direction. The steps are always symmetric, in the sense
            that steps.neg[i, j] = - steps.pos[i, j] unless one of them is NaN.

        num_terms (int): Number of terms needed to construct one estimate.
            !!! NEEDS MORE EXPLAINATION !!!

        order (int): Initial order of the approximation error of sequence elements.
            For central difference derivative approximation ``order`` = 2.

        exponentiation_step (int): Step representing the growth of the exponent in
            the series expansions of the limit.
            For central difference derivative approximation ``exponentiation_step`` = 2.

    Returns:
        limit (np.ndarray): The refined limit.
        error (np.ndarray): The error approximation of ``limit``.

    """
    seq_len = sequence.shape[0]

    steps = steps.pos
    n_steps = steps.shape[0]

    assert seq_len == n_steps, (
        "Length of ``steps`` must coincide with " "length of ``sequence``. "
    )

    assert num_terms > 0, "``num_terms`` must be greater than zero."

    assert seq_len - 1 >= num_terms, (
        "``num_terms`` cannot be greater than " "``seq_len`` - 1. "
    )

    # compute step ratio robust
    i = 0
    step_ratio = np.nan
    while np.isnan(step_ratio):
        step_ratio = steps[1].ravel()[i] / steps[0].ravel()[i]
        i += 1

    richardson_coef = _richardson_coefficients(
        num_terms, step_ratio, exponentiation_step, order,
    )

    new_sequence = convolve1d(
        input=sequence, weights=richardson_coef[::-1], axis=0, origin=num_terms // 2
    )

    m = seq_len - num_terms
    mm = m + 1 if num_terms >= 2 else seq_len

    abserr = _estimate_error(new_sequence[:mm], sequence, richardson_coef,)

    limit = new_sequence[:m]
    error = abserr[:m]

    return limit, error


def _richardson_coefficients(num_terms, step_ratio, exponentiation_step, order):
    """Compute Richardson coefficients.

    Let e := ``exponentiation_step``, r := ``step_ratio``, o := ``order`` and
    n := ``num_terms``. We build a matrix of the form

            [[1      1                  ...         1                ],
             [1    1/(s)**(2*o)         ...  1/(s)**(2*(o+n))        ],
        R =  [1    1/(s**2)**(2*o)      ...  1/(s**2)**(2*(o+n))     ],
             [...                       ...        ...               ],
             [1    1/(s**(n+1))**(2*o)  ...  1/(s**(n+1))**(2*(o+n)) ]]

    which is the weighting matrix in equation 24 in https://tinyurl.com/ybtfj4pm.
    We then return the first row of R^{-1} as the coefficients, as can be seen in
    equation 25 in https://tinyurl.com/ybtfj4pm.

    Args:
        num_terms (int): Number of terms needed to construct one estimate.
            !!! NEEDS MORE EXPLAINATION !!!

        step_ratio (float): Ratio between two consecutive steps. Order is chosen such
            that ``step_ratio`` >= 1.

        exponentiation_step (int): Step representing the growth of the exponent in
            the series expansions of the limit.
            For central difference derivative approximation ``exponentiation_step`` = 2.

        order (int): Initial order of the approximation error of sequence elements.
            For central difference derivative approximation ``order`` = 2.

    Returns:
        coef (np.ndarray): Array with Richardson coefficients of length num_terms + 1.

    Example:
    >>> import numpy as np
    >>> num_terms = 2
    >>> step_ratio = 2.
    >>> exponentiation_step = 1
    >>> order = 1
    >>> _richardson_coefficients(num_terms, step_ratio, exponentiation_step, order)
    array([ 0.33333333, -2.        ,  2.66666667])

    """
    rows, cols = np.ogrid[: num_terms + 1, :num_terms]

    coef_mat = np.ones((num_terms + 1, num_terms + 1))
    coef_mat[:, 1:] = (1.0 / step_ratio) ** (
        rows @ (exponentiation_step * cols + order)
    )

    coef = pinv(coef_mat)[0]
    return coef


def _estimate_error(new_seq, old_seq, richardson_coef):
    """Estimate error of multiple Richardson limit approximation.

    Args:
        new_seq (np.ndarray): Multiple estimates of the limit of ``old_seq``. The last
            two dimensions coincide with those of ``old_seq``. The first dimensions
            denotes the number of different estimates.

        old_seq (np.ndarray): The sequence of which we want to approximate the limit.
            Has dimension (k x n x m), where k denotes the number of sequence elements
            and an element ``sequence[l, :, :]`` denotes the (n x m) dimensional element

        richardson_coef (np.ndarray):

    Returns:
        abserr (np.ndarray): The error estimate for each limit approximation in
            ``new_seq``.

    """
    new_seq_len = new_seq.shape[0]

    unnormalized_covariance = np.sum(richardson_coef ** 2)
    fact = np.maximum(TQUANTILE * np.sqrt(unnormalized_covariance), EPS * 10.0)

    if new_seq_len <= 1:
        delta = np.diff(old_seq, axis=0)
        tol = np.maximum(np.abs(old_seq[:-1]), np.abs(old_seq[1:])) * fact
        err = np.abs(delta)
        converged = err <= tol
        abserr = err[-new_seq_len:] + np.where(
            converged[-new_seq_len:],
            tol[-new_seq_len:] * 10,
            abs(new_seq - old_seq[-new_seq_len:]) * fact,
        )
    else:
        err = np.abs(np.diff(new_seq, axis=0)) * fact
        tol = np.maximum(np.abs(new_seq[1:]), np.abs(new_seq[:-1])) * EPS * fact
        converged = err <= tol
        abserr = err + np.where(
            converged, tol * 10, abs(new_seq[:-1] - old_seq[-new_seq_len + 1 :]) * fact,
        )

    return abserr
