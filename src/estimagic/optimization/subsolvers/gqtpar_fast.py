"""Auxiliary functions for the quadratic GQTPAR trust-region subsolver."""
import numpy as np
from numba import njit
from scipy.linalg import cho_solve
from scipy.linalg import solve_triangular
from scipy.linalg.lapack import dpotrf as compute_cholesky_factorization


def gqtpar_fast(model, *, k_easy=0.1, k_hard=0.2, maxiter=200):
    """Solve the quadratic trust-region subproblem via nearly exact iterative method.

    This subproblem solver is mainly based on Conn et al. (2000) "Trust region methods"
    (:cite:`Conn2000`), pp. 169-200.

    But ideas from Nocedal and Wright (2006) "Numerical optimization"
    (:cite:`Nocedal2006`), pp. 83-91, who implement a similar algorithm,
    were also used.

    The original algorithm was developed by More and Sorensen (1983) (:cite:`More1983`)
    and is known as "GQTPAR".

    The vector x* is a global solution to the quadratic subproblem:

        min_x f + g @ x + 0.5 * x.T @ H @ x,

        if and only if ||x|| <= trustregion_radius
        and if there is a scalar lambda >= 0, such that:

    1) (H + lambda * I(n)) x* = -g
    2) lambda (trustregion_radius - ||x*||) = 0
    3) H + lambda * I is positive definite

    where g denotes the gradient and H the hessian of the quadratic model,
    respectively.

    k_easy and k_hard are stopping criteria for the iterative subproblem solver.
    See pp. 194-197 in :cite:`Conn2000` for a more detailed description.

    Args:
        main_model (NamedTuple): NamedTuple containing the parameters of the
            main model, i.e.:
            - ``linear_terms``, a np.ndarray of shape (n,) and
            - ``square_terms``, a np.ndarray of shape (n,n).
        trustregion_radius (float): Trustregion radius, often referred to as delta.
        k_easy (float): topping criterion for the "easy" case.
        k_hard (float): Stopping criterion for the "hard" case.
        maxiter (int): Maximum number of iterations to perform. If reached,
            terminate.

    Returns:
        (dict): Result dictionary containing the following keys:
            - ``x`` (np.ndarray): Solution vector of the subproblem of shape (n,)
            - ``criterion`` (float): Minimum function value associated with the
                solution.
    """
    hessian_already_factorized = False
    model_gradient = model.linear_terms
    model_hessian = model.square_terms
    x_candidate = np.zeros(len(model_gradient))

    # Small floating point number signaling that for vectors smaller
    # than that backward substituition is not reliable.
    # See Golub, G. H., Van Loan, C. F. (2013), "Matrix computations", p.165.
    zero_threshold = (
        model_hessian.shape[0] * np.finfo(float).eps * _norm(model_hessian, np.Inf)
    )
    stopping_criteria = {
        "k_easy": k_easy,
        "k_hard": k_hard,
    }

    gradient_norm = _norm(model_gradient, -1)
    (
        lambda_candidate,
        lambda_lower_bound,
        lambda_upper_bound,
    ) = _get_initial_guess_for_lambdas(model_gradient, model_hessian)

    converged = False

    for _niter in range(maxiter):

        if hessian_already_factorized:
            hessian_already_factorized = False
        else:
            (
                hessian_plus_lambda,
                hessian_upper_triangular,
                factorization_info,
            ) = _add_lambda_and_factorize_hessian(model_hessian, lambda_candidate)

        if factorization_info == 0 and gradient_norm > zero_threshold:
            (
                x_candidate,
                hessian_plus_lambda,
                hessian_already_factorized,
                lambda_candidate,
                lambda_lower_bound,
                lambda_upper_bound,
                converged,
            ) = _find_new_candidate_and_update_parameters(
                model_gradient,
                model_hessian,
                hessian_upper_triangular,
                hessian_plus_lambda,
                hessian_already_factorized,
                lambda_candidate,
                lambda_lower_bound,
                lambda_upper_bound,
                stopping_criteria,
                converged,
            )

        elif factorization_info == 0 and gradient_norm <= zero_threshold:
            (
                x_candidate,
                lambda_candidate,
                lambda_lower_bound,
                lambda_upper_bound,
                converged,
            ) = _check_for_interior_convergence_and_update(
                x_candidate,
                hessian_upper_triangular,
                lambda_candidate,
                lambda_lower_bound,
                lambda_upper_bound,
                stopping_criteria,
                converged,
            )

        else:
            (
                lambda_candidate,
                lambda_lower_bound,
            ) = _update_lambdas_when_factorization_unsuccessful(
                hessian_upper_triangular,
                hessian_plus_lambda,
                lambda_candidate,
                lambda_lower_bound,
                lambda_upper_bound,
                factorization_info,
            )

        if converged:
            break

    f_min = (
        model_gradient.T @ x_candidate
        + 0.5 * x_candidate.T @ model_hessian @ x_candidate
    )
    result = {
        "x": x_candidate,
        "criterion": f_min,
        "n_iterations": _niter,
        "success": converged,
    }

    return result


@njit
def _get_initial_guess_for_lambdas(model_gradient, model_hessian):
    """Return good initial guesses for lambda, its lower and upper bound.

    The values are chosen accordingly to the guidelines on
    section 7.3.8 (p. 192) from :cite:`Conn2000`.

    Args:
        model_gradient (np.ndarray): 1d array, of len n, of linear terms of the
            surrogate model.
        model_hessian (np.ndarray): 2d array, of shape (n,n), of square terms of the
            surrogate model.

    Returns:
        lambda_candidate (float): initial guess for damping factor
        lambda_lower_bound (float): initial guess for the lower bound of the damping
            factor.
        lambda_upper_bound(float): initial guess for the upper bound of the damping
            factor.
    """
    gradient_norm = _norm(model_gradient, -1.0)
    model_hessian = model_hessian

    hessian_infinity_norm = _norm(model_hessian, np.Inf)
    hessian_frobenius_norm = _norm(model_hessian, -1.0)

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


def _add_lambda_and_factorize_hessian(model_hessian, lambda_candidate):
    """Add lambda to hessian and factorize it into its upper triangular matrix.

    Args:
        model_hessian (np.ndarray): 2d array, of shape (n,n), of square terms of the
            surrogate model.
        lambda_candidate (float): dampig factor.
    Returns:
        hessian_plus_lambda (np.ndarray):  The square terms of the main model
                plus the identity matrix times lambda. 2d array of shape (n, n).
        hessian_upper_triangular (np.ndarray): Factorization of the hessian from the
                main model into its upper triangular matrix. The diagonal is filled
                and the lower lower triangular contains zeros.
        factorization_info (int): success flag returned by scipy.dpotrf
    """

    hessian_plus_lambda = model_hessian + lambda_candidate * _identity(
        model_hessian.shape[0]
    )
    hessian_upper_triangular, factorization_info = compute_cholesky_factorization(
        hessian_plus_lambda,
        lower=False,
        overwrite_a=False,
        clean=True,
    )

    return hessian_plus_lambda, hessian_upper_triangular, factorization_info


def _find_new_candidate_and_update_parameters(
    model_gradient,
    model_hessian,
    hessian_upper_triangular,
    hessian_plus_lambda,
    hessian_already_factorized,
    lambda_candidate,
    lambda_lower_bound,
    lambda_upper_bound,
    stopping_criteria,
    converged,
):
    """Find new candidate vector and update transformed hessian and lambdas."""
    x_candidate = cho_solve(
        (hessian_upper_triangular, False),
        -model_gradient,
    )
    x_norm = _norm(x_candidate, -1.0)

    if x_norm <= 1 and lambda_candidate == 0:
        converged = True

    w = solve_triangular(
        hessian_upper_triangular,
        x_candidate,
        trans="T",
    )
    w_norm = _norm(w, -1.0)

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
        ) = _update_candidate_and_parameters_when_candidate_within_trustregion(
            x_candidate,
            model_hessian,
            hessian_upper_triangular,
            hessian_plus_lambda,
            hessian_already_factorized,
            lambda_candidate,
            lambda_lower_bound,
            newton_step,
            stopping_criteria,
            converged,
        )

    else:
        if abs(x_norm - 1) <= stopping_criteria["k_easy"]:
            converged = True
        lambda_new_candidate = newton_step
        lambda_new_lower_bound = lambda_candidate
        lambda_new_upper_bound = lambda_upper_bound
    return (
        x_candidate,
        hessian_plus_lambda,
        hessian_already_factorized,
        lambda_new_candidate,
        lambda_new_lower_bound,
        lambda_new_upper_bound,
        converged,
    )


def _check_for_interior_convergence_and_update(
    x_candidate,
    hessian_upper_triangular,
    lambda_candidate,
    lambda_lower_bound,
    lambda_upper_bound,
    stopping_criteria,
    converged,
):
    """Check for interior convergence, update candidate vector and lambdas."""
    if lambda_candidate == 0:
        x_candidate = np.zeros(len(x_candidate))
        converged = True

    s_min, z_min = _estimate_smallest_singular_value(hessian_upper_triangular)
    step_len = 2

    if step_len**2 * s_min**2 <= stopping_criteria["k_hard"] * lambda_candidate:
        x_candidate = step_len * z_min
        converged = True

    lambda_lower_bound = max(lambda_lower_bound, lambda_upper_bound - s_min**2)
    lambda_new_candidate = _get_new_lambda_candidate(
        lower_bound=lambda_lower_bound, upper_bound=lambda_candidate
    )
    return (
        x_candidate,
        lambda_new_candidate,
        lambda_lower_bound,
        lambda_candidate,
        converged,
    )


def _update_lambdas_when_factorization_unsuccessful(
    hessian_upper_triangular,
    hessian_plus_lambda,
    lambda_candidate,
    lambda_lower_bound,
    lambda_upper_bound,
    factorization_info,
):
    """Update lambdas in the case that factorization of hessian not successful."""
    delta, v = _compute_terms_to_make_leading_submatrix_singular(
        hessian_upper_triangular,
        hessian_plus_lambda,
        factorization_info,
    )
    v_norm = _norm(v, -1.0)

    lambda_lower_bound = max(lambda_lower_bound, lambda_candidate + delta / v_norm**2)
    lambda_new_candidate = _get_new_lambda_candidate(
        lower_bound=lambda_lower_bound, upper_bound=lambda_upper_bound
    )
    return lambda_new_candidate, lambda_lower_bound


@njit
def _get_new_lambda_candidate(lower_bound, upper_bound):
    """Update current lambda so that it lies within its bounds.

    Args:
        lower_boud (float): lower bound of the current candidate dumping factor.
        upper_bound(float): upper bound of the current candidate dumping factor.

    Returns:
        float: New candidate for the damping factor lambda.
    """
    lambda_new_candidate = max(
        np.sqrt(max(0, lower_bound * upper_bound)),
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
        model_hessian (np.ndarray): 2d array, of shape (n,n), with square terms of the
            surrogate model
    Returns:
        lower_gershgorin (float): Lower Gregoshgorin bound.
        upper_gershgorin (float): Upper Gregoshgorin bound.
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
        lambda_candidate (float): Damping factor.
        p_norm (float): Frobenius (i.e. L2-norm) of the candidate vector.
        w_norm (float): Frobenius (i.e. L2-norm) of vector w, which is the solution
            to the following triangular system: U.T w = p.

    Returns:
        float: Newton step computed according to formula (4.44) p.87
            from Nocedal and Wright (2006).
    """
    return lambda_candidate + (p_norm / w_norm) ** 2 * (p_norm - 1)


def _update_candidate_and_parameters_when_candidate_within_trustregion(
    x_candidate,
    model_hessian,
    hessian_upper_triangular,
    hessian_plus_lambda,
    hessian_already_factorized,
    lambda_candidate,
    lambda_lower_bound,
    newton_step,
    stopping_criteria,
    converged,
):
    """Update candidate vector, hessian, and lambdas when x outside trust-region."""

    s_min, z_min = _estimate_smallest_singular_value(hessian_upper_triangular)
    step_len = _compute_smallest_step_len_for_candidate_vector(x_candidate, z_min)

    quadratic_term = x_candidate.T @ hessian_plus_lambda @ x_candidate

    relative_error = (step_len**2 * s_min**2) / (quadratic_term + lambda_candidate)
    if relative_error <= stopping_criteria["k_hard"]:
        x_candidate = x_candidate + step_len * z_min
        converged = True

    lambda_new_lower_bound = max(lambda_lower_bound, lambda_candidate - s_min**2)

    hessian_plus_lambda = model_hessian + newton_step * _identity(len(x_candidate))
    _, factorization_unsuccessful = compute_cholesky_factorization(
        hessian_plus_lambda,
        lower=False,
        overwrite_a=False,
        clean=True,
    )

    if factorization_unsuccessful == 0:
        hessian_already_factorized = True
        lambda_new_candidate = newton_step
    else:
        lambda_new_lower_bound = max(lambda_new_lower_bound, newton_step)
        lambda_new_candidate = _get_new_lambda_candidate(
            lower_bound=lambda_new_lower_bound, upper_bound=lambda_candidate
        )

    lambda_new_upper_bound = lambda_candidate

    return (
        x_candidate,
        hessian_plus_lambda,
        hessian_already_factorized,
        lambda_new_candidate,
        lambda_new_lower_bound,
        lambda_new_upper_bound,
        converged,
    )


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
    a = z_min @ z_min
    b = 2 * x_candidate.T @ z_min
    c = x_candidate.T @ x_candidate - 1
    ta, tb = np.roots(np.array([a, b, c]))
    if abs(ta) <= abs(tb):
        step_len = ta
    else:
        step_len = tb
    return step_len


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
        hessian_upper_triangular (np.ndarray) Upper triangular matrix resulting of an
            incomplete Cholesky decomposition of the hessian matrix.
        hessian_plus_lambda (np.ndarray): Symmetric k by k hessian matrix, which is not
            positive definite.
        k (int): Positive integer such that the leading k by k submatrix from
            hessian is the first non-positive definite leading submatrix.

    Returns:
        Tuple:
        - delta(float): Amount that should be added to the element (k, k) of
            the leading k by k submatrix of the hessian to make it singular.
        - v (np.ndarray): A vector such that ``v.T B v = 0``. Where B is the
            hessian after ``delta`` is added to its element (k, k).
    """
    hessian_plus_lambda = hessian_plus_lambda
    upper_triangular = hessian_upper_triangular

    delta = (
        np.sum(upper_triangular[: k - 1, k - 1] ** 2)
        - hessian_plus_lambda[k - 1, k - 1]
    )

    v = np.zeros(len(hessian_plus_lambda))
    v[k - 1] = 1

    if k != 1:
        v[: k - 1] = solve_triangular(
            upper_triangular[: k - 1, : k - 1],
            -upper_triangular[: k - 1, k - 1],
        )

    return delta, v


@njit
def _estimate_condition(u):
    """Return largest possible solution w to the system u.T w = e.

    u is an upper triangular matrix, and components of e are selected from {+1, -1}.

    Args:
        u (np.ndarray): Upper triangular matrix of shape (n,n).
    Returns:
        w (np.ndarray): 1d array of len n.

    """
    u = np.atleast_2d(u)

    if u.shape[0] != u.shape[1]:
        raise ValueError("A square triangular matrix should be provided.")

    # A vector `e` with components selected from {+1, -1}
    # is selected so that the solution `w` to the system
    # `U.T w = e` is as large as possible. Implementation
    # based on algorithm 3.5.1, p. 142, from reference [2]
    # adapted for lower triangular matrix.
    m = u.shape[0]
    p = np.zeros(m)
    w = np.zeros(m)

    # Implemented according to:  Golub, G. H., Van Loan, C. F. (2013).
    # "Matrix computations". Forth Edition. JHU press. pp. 140-142.
    for k in range(m):
        wp = (1 - p[k]) / u.T[k, k]
        wm = (-1 - p[k]) / u.T[k, k]
        pp = p[k + 1 :] + u.T[k + 1 :, k] * wp
        pm = p[k + 1 :] + u.T[k + 1 :, k] * wm

        if abs(wp) + _norm(pp, 1) >= abs(wm) + _norm(pm, 1):
            w[k] = wp
            p[k + 1 :] = pp
        else:
            w[k] = wm
            p[k + 1 :] = pm
    return w


def _estimate_smallest_singular_value(upper_triangular):
    """Estimate the smallest singular vlue and the correspondent right singular vector.

    Given an upper triangular matrix `u`, performs in O(n**2) operations and returns
    estimated values of smalles singular value and the correspondent right singular
    vector.

    Based on estimate_smallest_singular_value from scipy.optimize._trustregion_exact,
    jitting some calculations in a separate function and calling them here.

    Args:
        upper_triangular (np.ndarray) : Square upper triangular matrix of shape (n,n)

    Returns:
        s_min (float): Estimated smallest singular value of the provided matrix.
        z_min (np.ndarray): Estimatied right singular vector.

    Notes:
        The procedure is based on [1] and is done in two steps. First, it finds
        a vector ``e`` with components selected from {+1, -1} such that the
        solution ``w`` from the system ``U.T w = e`` is as large as possible.
        Next it estimate ``U v = w``. The smallest singular value is close
        to ``norm(w)/norm(v)`` and the right singular vector is close
        to ``v/norm(v)``.
        The estimation will be better more ill-conditioned is the matrix.

    References:
    .. [1] Cline, A. K., Moler, C. B., Stewart, G. W., Wilkinson, J. H.
        An estimate for the condition number of a matrix.  1979.
        SIAM Journal on Numerical Analysis, 16(2), 368-375.

    """
    w = _estimate_condition(upper_triangular)

    # The system `U v = w` is solved using backward substitution.
    v = solve_triangular(upper_triangular, w)

    v_norm = _norm(v, -1.0)
    w_norm = _norm(w, -1.0)

    # Smallest singular value
    s_min = w_norm / v_norm

    # Associated vector
    z_min = v / v_norm

    return s_min, z_min


@njit
def _norm(a, order):
    """A wrapper to jit np.linalg.norm."""
    if order == -1:
        out = np.linalg.norm(a)
    else:
        out = np.linalg.norm(a, order)
    return out


@njit
def _identity(dim):
    """A wrapper to jit np.eye."""
    return np.eye(dim)
