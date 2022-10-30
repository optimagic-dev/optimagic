"""Auxiliary functions for the quadratic GQTPAR trust-region subsolver."""
import numpy as np
from estimagic.optimization.subsolvers._numba_potrf import (
    compute_cholesky_factorization,
)
from numba import njit


@njit
def get_initial_guess_for_lambdas_fast(
    model_gradient,
    model_hessian,
):
    """Return good initial guesses for lambda, its lower and upper bound.

    Given a trust-region radius, good initial guesses for the damping factor lambda,
    along with its lower bound and upper bound, are computed.

    The values are chosen accordingly to the guidelines on
    section 7.3.8 (p. 192) from :cite:`Conn2000`.

    Args:
        main_model (NamedTuple): Named tuple containing the parameters of the
            main model, i.e.:
            - ``linear_terms``, a np.ndarray of shape (n,) and
            - ``square_terms``, a np.ndarray of shape (n,n).

    Returns:
        (dict): Dictionary containing the initial guess for the damping
            factor lambda, along with its lower and upper bound.
            The respective keys are:
            - "candidate"
            - "upper_bound"
            - "lower_bound"
    """
    gradient_norm = np.linalg.norm(model_gradient)

    hessian_infinity_norm = np.linalg.norm(model_hessian, np.Inf)
    hessian_frobenius_norm = np.linalg.norm(model_hessian)

    hessian_gershgorin_lower, hessian_gershgorin_upper = _compute_gershgorin_bounds(
        model_hessian
    )

    lambda_lower_bound = max(
        0,
        -min(np.diag(model_hessian)),
        gradient_norm
        - min(hessian_gershgorin_upper, hessian_frobenius_norm, hessian_infinity_norm),
    )
    lambda_upper_bound = max(
        0,
        gradient_norm
        + min(-hessian_gershgorin_lower, hessian_frobenius_norm, hessian_infinity_norm),
    )

    if lambda_lower_bound == 0:
        lambda_candidate = 0
    else:
        lambda_candidate = _get_new_lambda_candidate(
            lower_bound=lambda_lower_bound, upper_bound=lambda_upper_bound
        )

    return lambda_candidate, lambda_lower_bound, lambda_upper_bound


@njit
def add_lambda_and_factorize_hessian_fast(model_hessian, lambda_candidate):
    """Add lambda to hessian and factorize it into its upper triangular matrix.

    Args:
        model_hessian (np.ndarray): The array with square terms of the main surrogate
        model.
        lambda_candidate (float): Initial guess for the damping factor lambda.
    Returns:
        hessian_plus_lambda (np.ndarray): The square terms of the main model
                plus the identity matrix times lambda
        hessian_upper_triangular (np.ndarray): Factorization of the hessian from the
                main model into its upper triangular matrix. The diagonal is filled
                and the lower lower triangular contains zeros.
        factorization_info (int): Integer flag indicating whether hessian factorization
            is successfull (0) and, if not, order of the first non positive definite
            diagonal block.
    """
    n = model_hessian.shape[0]

    hessian_plus_lambda = model_hessian + lambda_candidate * np.eye(n)
    hessian_upper_triangular, factorization_info = compute_cholesky_factorization(
        hessian_plus_lambda,
    )
    return hessian_plus_lambda, hessian_upper_triangular, factorization_info


@njit
def find_new_candidate_and_update_parameters_fast(
    model_gradient,
    model_hessian,
    hessian_upper_triangular,
    hessian_plus_lambda,
    hessian_already_factorized,
    lambda_candidate,
    lambda_lower_bound,
    lambda_upper_bound,
    stopping_k_easy,
    stopping_k_hard,
    converged,
):
    """Find new candidate vector and update transformed hessian and lambdas."""
    x_candidate = np.linalg.solve(
        hessian_upper_triangular,
        np.linalg.solve(hessian_upper_triangular.T, -model_gradient),
    )

    x_norm = np.linalg.norm(x_candidate)

    if x_norm <= 1 and lambda_candidate == 0:
        converged = True

    w = np.linalg.solve(hessian_upper_triangular.T, x_candidate)
    w_norm = np.linalg.norm(w)

    newton_step = _compute_newton_step(lambda_candidate, x_norm, w_norm)

    if x_norm < 1:
        (
            x_candidate,
            hessian_plus_lambda,
            hessian_already_factorized,
            lambda_new_candidate,
            lambda_new_lower_bound,
            lambda_new_upper_bound,
            converged,
        ) = _update_candidate_and_parameters_when_candidate_within_trustregion_fast(
            x_candidate=x_candidate,
            model_hessian=model_hessian,
            hessian_upper_triangular=hessian_upper_triangular,
            hessian_plus_lambda=hessian_plus_lambda,
            hessian_already_factorized=hessian_already_factorized,
            lambda_candidate=lambda_candidate,
            lambda_lower_bound=lambda_lower_bound,
            newton_step=newton_step,
            stopping_criterion=stopping_k_hard,
            converged=converged,
        )

    else:
        (
            lambda_new_candidate,
            lambda_new_lower_bound,
            converged,
        ) = _update_lambdas_when_candidate_outside_trustregion(
            lambda_candidate,
            newton_step,
            x_norm,
            stopping_k_easy,
            converged,
        )
        lambda_new_upper_bound = lambda_upper_bound
    return (
        x_candidate,
        hessian_plus_lambda,
        hessian_upper_triangular,
        hessian_already_factorized,
        lambda_new_candidate,
        lambda_new_lower_bound,
        lambda_new_upper_bound,
        converged,
    )


@njit
def check_for_interior_convergence_and_update_fast(
    x_candidate,
    hessian_upper_triangular,
    lambda_candidate,
    lambda_lower_bound,
    lambda_upper_bound,
    stopping_criterion,
    converged,
):
    """Check for interior convergence, update candidate vector and lambdas."""
    if lambda_candidate == 0:
        x_candidate = np.zeros(len(x_candidate))
        converged = True

    s_min, z_min = estimate_smallest_singular_value(hessian_upper_triangular)
    step_len = 2

    if step_len**2 * s_min**2 <= stopping_criterion * lambda_candidate:
        x_candidate = step_len * z_min
        converged = True

    lambda_new_lower_bound = max(lambda_lower_bound, lambda_upper_bound - s_min**2)
    lambda_new_upper_bound = lambda_candidate

    lambda_new_candidate = _get_new_lambda_candidate(
        lower_bound=lambda_new_lower_bound, upper_bound=lambda_new_upper_bound
    )
    return (
        x_candidate,
        lambda_new_candidate,
        lambda_new_lower_bound,
        lambda_new_upper_bound,
        converged,
    )


@njit
def update_lambdas_when_factorization_unsuccessful_fast(
    hessian_upper_triangular,
    hessian_plus_lambda,
    lambda_candidate,
    lambda_lower_bound,
    lambda_upper_bound,
    factorization_info,
):
    """Update lambdas in the case that factorization of hessian not successful."""
    delta, v = _compute_terms_to_make_leading_submatrix_singular(
        hessian_upper_triangular=hessian_upper_triangular,
        hessian_plus_lambda=hessian_plus_lambda,
        k=factorization_info,
    )
    v_norm = np.linalg.norm(v)

    lambda_new_lower_bound = max(
        lambda_lower_bound, lambda_candidate + delta / v_norm**2
    )
    lambda_new_candidate = _get_new_lambda_candidate(
        lower_bound=lambda_new_lower_bound, upper_bound=lambda_upper_bound
    )
    return lambda_new_candidate, lambda_new_lower_bound


@njit
def _get_new_lambda_candidate(lower_bound, upper_bound):
    """Update current lambda so that it lies within its bounds.

    Args:
        lambdas_new (NamedTuple): Named tuple containing the current candidate
            value for the damping factor lambda, its lower bound and upper bound.

    Returns:
        float: New candidate for the damping factor lambda.
    """
    lambda_new_candidate = max(
        np.sqrt(lower_bound * upper_bound),
        lower_bound + 0.01 * (upper_bound - lower_bound),
    )

    return lambda_new_candidate


@njit
def _compute_gershgorin_bounds(model_hessian):
    """Compute upper and lower Gregoshgorin bounds for a square matrix.

    The Gregoshgorin bounds are the upper and lower bounds for the
    eigenvalues of the square hessian matrix (i.e. the square terms of
    the main model). See :cite:`Conn2000`.

    Args:
        model_hessian (np.ndarray): square terms of the model.

    Returns:
        - lower_gershgorin (float): Lower Gregoshgorin bound.
        - upper_gershgorin (float): Upper Gregoshgorin bound.
    """

    hessian_diag = np.diag(model_hessian)
    hessian_diag_abs = np.abs(hessian_diag)
    hessian_row_sums = np.sum(np.abs(model_hessian), axis=1)

    lower_gershgorin = np.min(hessian_diag + hessian_diag_abs - hessian_row_sums)
    upper_gershgorin = np.max(hessian_diag - hessian_diag_abs + hessian_row_sums)

    return lower_gershgorin, upper_gershgorin


@njit
def _compute_newton_step(lambda_candidate, p_norm, w_norm):
    """Compute the Newton step.

    Args:
        lambda_candidate (np.ndarray): The current candidate value for the damping
            factor lambda.
        p_norm (float): Frobenius (i.e. L2-norm) of the candidate vector.
        w_norm (float): Frobenius (i.e. L2-norm) of vector w, which is the solution
            to the following triangular system: U.T w = p.

    Returns:
        float: Newton step computed according to formula (4.44) p.87
            from Nocedal and Wright (2006).
    """
    return lambda_candidate + (p_norm / w_norm) ** 2 * (p_norm - 1)


@njit
def _update_candidate_and_parameters_when_candidate_within_trustregion_fast(
    x_candidate,
    model_hessian,
    hessian_upper_triangular,
    hessian_plus_lambda,
    hessian_already_factorized,
    lambda_candidate,
    lambda_lower_bound,
    newton_step,
    stopping_criterion,
    converged,
):
    """Update candidate vector, hessian, and lambdas when x outside trust-region."""
    n = len(x_candidate)

    s_min, z_min = estimate_smallest_singular_value(hessian_upper_triangular)
    step_len = _compute_smallest_step_len_for_candidate_vector(x_candidate, z_min)

    quadratic_term = x_candidate.T @ hessian_plus_lambda @ x_candidate

    relative_error = (step_len**2 * s_min**2) / (quadratic_term + lambda_candidate)
    if relative_error <= stopping_criterion:
        x_candidate = x_candidate + step_len * z_min
        converged = True

    lambda_new_lower_bound = max(lambda_lower_bound, lambda_candidate - s_min**2)

    hessian_plus_lambda = model_hessian + newton_step * np.eye(n)
    _, factorization_unsuccessful = compute_cholesky_factorization(
        hessian_plus_lambda,
    )

    if factorization_unsuccessful == 0:
        hessian_already_factorized = True
        lambda_new_candidate = newton_step
    else:
        hessian_already_factorized = hessian_already_factorized
        lambda_new_lower_bound = max(lambda_new_lower_bound, newton_step)
        lambda_new_candidate = _get_new_lambda_candidate(
            lower_bound=lambda_new_lower_bound, upper_bound=lambda_candidate
        )
    lambda_upper_bound = lambda_candidate

    return (
        x_candidate,
        hessian_plus_lambda,
        hessian_already_factorized,
        lambda_new_candidate,
        lambda_new_lower_bound,
        lambda_upper_bound,
        converged,
    )


@njit
def _update_lambdas_when_candidate_outside_trustregion(
    lambda_candidate, newton_step, p_norm, stopping_criterion, converged
):
    """Update lambas in the case that candidate vector lies outside trust-region."""
    relative_error = abs(p_norm - 1)

    if relative_error <= stopping_criterion:
        converged = True

    lambda_new_candidate = newton_step
    lambda_new_lower_bound = lambda_candidate
    return lambda_new_candidate, lambda_new_lower_bound, converged


@njit
def _compute_smallest_step_len_for_candidate_vector(x_candidate, z_min):
    """Compute the smallest step length for the candidate vector.

    Choose step_length with the smallest magnitude.
    The reason for this choice is explained at p. 6 in :cite:`More1983`,
    just before the formula for tau.

    Args:
        x_candidate (np.ndarray): Candidate vector of shape (n,).
        z_min (float): Smallest singular value of the hessian matrix.

    Returns:
        float: Step length with the smallest magnitude.
    """
    ta, tb = _solve_scalar_quadratic_equation(x_candidate, z_min)
    if np.abs(ta) > np.abs(tb):
        step_len = tb
    elif np.abs(ta) == np.abs(tb):
        if ta >= tb:
            step_len = tb
        else:
            step_len = ta
    else:
        step_len = ta

    return step_len


@njit
def _solve_scalar_quadratic_equation(z, d):
    """Return the sorted values that solve the scalar quadratic equation.

    Solve the scalar quadratic equation ||z + t d|| == trustregion_radius.
    This is like a line-sphere intersection.


    Computation of the ``aux`` step, ``ta`` and ``tb`` is mathematically equivalent
    to equivalent the following calculation:

    ta = (-b - sqrt_discriminant) / (2*a)
    tb = (-b + sqrt_discriminant) / (2*a)

    but produces smaller round-off errors.
    For more details, look at "Matrix Computation" p.97.

    Args:
        z (np.ndarray): Eigenvector of the upper triangular hessian matrix.
        d (float): Smallest singular value of the upper triangular of the
            hessian matrix.

    Returns
        Tuple: The two values of t, sorted from low to high.
        - (float) Lower value of t.
        - (float) Higher value of t.
    """
    a = d.T @ d
    b = 2 * z.T @ d
    c = z.T @ z - 1
    ta, tb = np.sort(np.roots(np.array([a, b, c])))
    return ta, tb


@njit
def _compute_terms_to_make_leading_submatrix_singular(
    hessian_upper_triangular, hessian_plus_lambda, k
):
    """Compute terms that make the leading submatrix of the hessian singular.

    The "hessian" here refers to the matrix

        H + lambda * I(n),

    where H is the initial hessian, lambda is the current damping factor,
    I the identity matrix, and m the number of rows/columns of the symmetric
    hessian matrix.

    Args:
        hessian_plust_lambda(np.ndarray): Symmetric k by k hessian matrix, which is not
            positive definite.
        upper_triangular (np.ndarray) Upper triangular matrix resulting of an
            incomplete Cholesky decomposition of the hessian matrix.
        k (int): Positive integer such that the leading k by k submatrix from
            hessian is the first non-positive definite leading submatrix.

    Returns:
        Tuple:
        - delta(float): Amount that should be added to the element (k, k) of
            the leading k by k submatrix of the hessian to make it singular.
        - v (np.ndarray): A vector such that ``v.T B v = 0``. Where B is the
            hessian after ``delta`` is added to its element (k, k).
    """
    n = len(hessian_plus_lambda)

    delta = (
        np.sum(hessian_upper_triangular[: k - 1, k - 1] ** 2)
        - hessian_plus_lambda[k - 1, k - 1]
    )

    v = np.zeros(n)
    v[k - 1] = 1

    if k != 1:
        v[: k - 1] = np.linalg.solve(
            hessian_upper_triangular[: k - 1, : k - 1],
            -hessian_upper_triangular[: k - 1, k - 1],
        )

    return delta, v


@njit
def estimate_smallest_singular_value(u):
    """Given upper triangular matrix ``U`` estimate the smallest singular
    value and the correspondent right singular vector in O(n**2) operations.
    Parameters
    ----------
    U : ndarray
        Square upper triangular matrix.
    Returns
    -------
    s_min : float
        Estimated smallest singular value of the provided matrix.
    z_min : ndarray
        Estimatied right singular vector.
    Notes
    -----
    The procedure is based on [1] and is done in two steps. First, it finds
    a vector ``e`` with components selected from {+1, -1} such that the
    solution ``w`` from the system ``U.T w = e`` is as large as possible.
    Next it estimate ``U v = w``. The smallest singular value is close
    to ``norm(w)/norm(v)`` and the right singular vector is close
    to ``v/norm(v)``.
    The estimation will be better more ill-conditioned is the matrix.
    References
    ----------
    .. [1] Cline, A. K., Moler, C. B., Stewart, G. W., Wilkinson, J. H.
           An estimate for the condition number of a matrix.  1979.
           SIAM Journal on Numerical Analysis, 16(2), 368-375.
    """

    u = np.atleast_2d(u)
    m, n = u.shape

    if m != n:
        raise ValueError("A square triangular matrix should be provided.")

    # A vector `e` with components selected from {+1, -1}
    # is selected so that the solution `w` to the system
    # `U.T w = e` is as large as possible. Implementation
    # based on algorithm 3.5.1, p. 142, from reference [2]
    # adapted for lower triangular matrix.

    p = np.zeros(n)
    w = np.empty(n)

    # Implemented according to:  Golub, G. H., Van Loan, C. F. (2013).
    # "Matrix computations". Forth Edition. JHU press. pp. 140-142.
    for k in range(n):
        wp = (1 - p[k]) / u.T[k, k]
        wm = (-1 - p[k]) / u.T[k, k]
        pp = p[k + 1 :] + u.T[k + 1 :, k] * wp
        pm = p[k + 1 :] + u.T[k + 1 :, k] * wm

        if abs(wp) + np.linalg.norm(pp, 1) >= abs(wm) + np.linalg.norm(pm, 1):
            w[k] = wp
            p[k + 1 :] = pp
        else:
            w[k] = wm
            p[k + 1 :] = pm

    # The system `U v = w` is solved using backward substitution.
    v = np.linalg.solve(u, w)

    v_norm = np.linalg.norm(v)
    w_norm = np.linalg.norm(w)

    # Smallest singular value
    s_min = w_norm / v_norm

    # Associated vector
    z_min = v / v_norm

    return s_min, z_min
