"""Implementation of a nearly exact solver for the trust-region subproblem."""
import math

import numpy as np
from scipy.linalg import cho_solve
from scipy.linalg import norm
from scipy.linalg import solve_triangular
from scipy.linalg.lapack import dpotrf as compute_cholesky_factorization
from scipy.optimize._trustregion_exact import estimate_smallest_singular_value


def solve_trustregion_subproblem(fun, trustregion_radius):
    """Solve the quadratic trust-region subproblem via nearly exact iterative method.

    This subproblem solver is mainly based on

    .. [1] A.R. Conn, N.I. Gould, and P.L. Toint (2000) "Trust region methods",
        Siam, pp. 169-200.

    But ideas from

    .. [2] J. Nocedal and  S. Wright (2006) "Numerical optimization",
        Springer Science & Business Media. pp. 83-91

    and

    .. [3] J.J. More and D.C. Sorensen (1983) "Computing a trust region step",
        SIAM Journal on Scientific and Statistical Computing, vol. 4(3) pp. 553-572,

    who implement similar algorithms, were also used.
    The original algorithm was developed by [3] and is known as "GQTPAR".

    The direction vector ``p*`` is a global solution to the quadratic subproblem:

        min f + g p + 0.5 * p H p,

        s.t. norm(p) <= trustregion_radius

        if and only if norm(``p*``) <= trustregion radius and there is a scalar
        lambda >= 0, such that:

    1) (H + lambda * I(n)) p* = -g
    2) lambda (trustregion_radius - norm(p*)) = 0
    3) H + lambda * I is positive definite

    where g denotes the gradient vector and H the hessian matrix of the main model,
    respectively.

    k_easy and k_hard are stopping criteria for the iterative subproblem solver.
    See pp. 194-197 in [1] for a more detailed description.

    Args:
        fun (callable): Criterion function to be minimized.
        trustregion_radius (float): Trustregion radius, often referred to as "delta".

    Returns:
        result (dict): Result dictionary containing the following keys:
            - "p_solution" (np.ndarray): Solution vector of the subproblem
                with optimal direction p.
            - "q_min" (float): Minimum function value of the subproblem.
    """
    model_gradient = fun.keywords["linear_terms"]
    model_hessian = {
        "initial": fun.keywords["square_terms"],
        "info_already_factorized": False,
    }

    n = model_gradient.shape[0]

    # Small floating point number signaling that for vectors smaller
    # than that backward substituition is not reliable.
    # See Golub, G. H., Van Loan, C. F. (2013), "Matrix computations", p.165.
    close_to_zero = n * np.finfo(float).eps * norm(model_hessian["initial"], np.Inf)

    stopping_criteria = {
        "k_easy": 0.1,
        "k_hard": 0.2,
    }

    gnorm = norm(model_gradient)

    lambdas = get_initial_guess_for_lambdas(
        gnorm,
        model_hessian["initial"],
        trustregion_radius,
    )

    converged = False
    niter = 0

    while True:

        if model_hessian["info_already_factorized"]:
            model_hessian["info_already_factorized"] = False
        else:
            model_hessian, factorization_info = add_lambda_and_factorize_hessian(
                model_hessian, lambdas
            )

        niter += 1

        if factorization_info == 0 and gnorm > close_to_zero:
            # Factorization successful
            (
                p_candidate,
                model_hessian,
                lambdas,
                converged,
            ) = find_new_candidate_and_update_parameters(
                model_gradient,
                model_hessian,
                lambdas,
                trustregion_radius,
                stopping_criteria,
                converged,
            )

        elif factorization_info == 0 and gnorm <= close_to_zero:
            p_candidate, lambdas, converged = check_for_interior_convergence_and_update(
                p_candidate,
                model_hessian,
                lambdas,
                trustregion_radius,
                stopping_criteria,
                converged,
            )

        else:
            # Factorization not successful
            lambdas = update_lambdas_when_factorization_unsuccessful(
                model_hessian,
                lambdas,
                factorization_info,
            )

        if converged is True:
            break

    q_min, _ = fun(p_candidate)

    result = {"p_solution": p_candidate, "q_min": q_min}

    return result


def get_initial_guess_for_lambdas(
    gnorm,
    hessian,
    trustregion_radius,
):
    """Return good initial guesses for lambda, its lower and upper bound.

    Given a trust-region radius, good initial guesses for the damping factor lambda,
    along with its lower bound and upper bound, are computed.

    The values are chosen accordingly to the guidelines on
    section 7.3.8 (p. 192) from Conn et al. (2000).

    Args:
        gnorm (float): Gradient norm.
        hessian (np.ndarray): Square hessian matrix.
        trustregion_radius (float): trust-region radius.

    Returns:
        dict: Dictionary containing the initial (current) guess for the damping
            factor lambda, along with its lower and upper bound.
            The respective keys are:
            - "current"
            - "upper"
            - "lower"
    """
    lambdas = {}

    hessian_infinity_norm = norm(hessian, np.Inf)
    hessian_frobenius_norm = norm(hessian, "fro")

    hessian_gershgorin_lower, hessian_gershgorin_upper = _compute_gershgorin_bounds(
        hessian
    )

    lambdas["upper_bound"] = max(
        0,
        gnorm / trustregion_radius
        + min(-hessian_gershgorin_lower, hessian_frobenius_norm, hessian_infinity_norm),
    )

    lambdas["lower_bound"] = max(
        0,
        -min(hessian.diagonal()),
        gnorm / trustregion_radius
        - min(hessian_gershgorin_upper, hessian_frobenius_norm, hessian_infinity_norm),
    )

    if lambdas["lower_bound"] == 0:
        lambdas["current"] = 0
    else:
        lambdas["current"] = _update_current_lambda(lambdas)

    return lambdas


def add_lambda_and_factorize_hessian(model_hessian, lambdas):
    """Add lambda to hessian matrix and factorize into its upper triangular.

    Args:
        model_hessian (dict): Dictionary of the hessian matrix containing the
            following keys:
            - "initial" (np.ndarray): The initial square hessian matrix supplied
                as input to the subproblem.
            - "info_already_factorized" (bool): Boolean indicating whether the hessian
                has already been factorized for the current iteration.

    Returns:
        Tuple:
        - model_hessian_updated (dict): Dictionary of the hessian matrix and its
            transformations containing the following keys:
            - "initial" (np.ndarray): The initial square hessian matrix supplied
                as input to the subproblem.
            - "initial_plus_lambda" (np.ndarray): The initial hessian matrix
                plus lambda times the identity matrix, computed as

                    H + lambda * I(n),

                where `H` denotes the initial hessian, `I` the identity matrix,
                and `n` the number of rows/columns.
            - "upper_triangular" (np.ndarray): Factorization of the initial hessian
                into its upper triangular matrix.
            - "info_already_factorized" (bool): Boolean indicating whether the hessian
                has already been factorized for the current iteration.

        - lambdas (dict): Dictionary containing the current damping factor lambda,
            its lower and upper bound.
            The respective keys are:
            - "current"
            - "upper"
            - "lower"
    """
    model_hessian_updated = model_hessian.copy()
    n = model_hessian["initial"].shape[0]

    model_hessian_updated["initial_plus_lambda"] = model_hessian["initial"] + lambdas[
        "current"
    ] * np.eye(n)
    (
        model_hessian_updated["upper_triangular"],
        factorization_info,
    ) = compute_cholesky_factorization(
        model_hessian_updated["initial_plus_lambda"],
        lower=False,
        overwrite_a=False,
        clean=True,
    )

    return model_hessian_updated, factorization_info


def find_new_candidate_and_update_parameters(
    model_gradient,
    model_hessian,
    lambdas,
    trustregion_radius,
    stopping_criteria,
    converged,
):
    """Find new candidate vector and update transformed hessian and lambdas.

    Args:
        model_gradient (np.ndarray): Gradient.
        model_hessian (dict): Dictionary of the hessian matrix and its
            transformations containing the following keys:
            - "initial" (np.ndarray): The initial square hessian matrix supplied
                as input to the subproblem.
            - "initial_plus_lambda" (np.ndarray): The initial hessian matrix
                plus lambda times the identity matrix, computed as

                    H + lambda * I(n),

                where `H` denotes the initial hessian, `I` the identity matrix,
                and `n` the number of rows/columns.
            - "upper_triangular" (np.ndarray): Factorization of the initial hessian
                into its upper triangular matrix.
            - "info_already_factorized" (bool): Boolean indicating whether the hessian
                has already been factorized for the current iteration.
        lambdas (dict): Dictionary containing the current damping
            factor, lambda, as well as its lower and upper bound.
            The respective keys are:
            - "current"
            - "upper"
            - "lower"
        trustregion_radius (float): Trust-region radius.
        stopping_criteria (dict): Dictionary of the two stopping criteria
            containing the following keys:
            - "k_easy" (float): Stopping criterion in the "easy" case.
            - "k_hard" (float): Stopping criterion in the "hard" case.
            See pp. 194-197 from reference _[1] for more details.
        converged (bool): Boolean indicating whether the subproblem has converged
            If True, the iteration will be stopped and the solution vector returned.

    Returns:
        Tuple:
        - p_candidate (np.ndarray): Current candidate vector.
        - model_hessian_updated (dict): Dictionary of the hessian matrix and its
            updated transformations containing the following keys:
            - "initial" (np.ndarray): The initial square hessian matrix supplied
                as input to the subproblem.
            - "initial_plus_lambda" (np.ndarray): The initial hessian matrix
                plus lambda times the identity matrix, computed as

                    H + lambda * I(n),

                where `H` denotes the initial hessian, `I` the identity matrix,
                and `n` the number of rows/columns.
            - "upper_triangular" (np.ndarray): Factorization of the initial hessian
                into its upper triangular matrix.
            - "info_already_factorized" (bool): Boolean indicating whether the hessian
                has already been factorized for the current iteration.
        - lambdas_updated (dict): Updated dictionary containing the current damping
            factor, lambda, as well as its lower and upper bound.
            The respective keys are:
            - "current"
            - "upper"
            - "lower"
        - converged (bool): Boolean indicating whether the subproblem has converged
            If True, the iteration will be stopped and the solution vector returned.
    """
    model_hessian_updated = model_hessian.copy()
    lambdas_updated = lambdas.copy()

    # Solve `U.T U p = -g`
    p_candidate = cho_solve((model_hessian["upper_triangular"], False), -model_gradient)
    p_norm = norm(p_candidate)

    if p_norm <= trustregion_radius and lambdas["current"] == 0:
        converged = True

    # Solve `U.T w = p`
    w = solve_triangular(model_hessian["upper_triangular"], p_candidate, trans="T")
    w_norm = norm(w)

    newton_step = _compute_newton_step(lambdas, p_norm, w_norm, trustregion_radius)

    if p_norm < trustregion_radius:
        (
            p_candidate,
            model_hessian_updated,
            lambdas_updated,
            converged,
        ) = _update_candidate_and_parameters_when_candidate_within_trustregion(
            p_candidate,
            model_hessian,
            lambdas,
            newton_step,
            trustregion_radius,
            stopping_criteria,
            converged,
        )

    else:
        lambdas_updated, converged = _update_lambdas_when_candidate_outside_trustregion(
            lambdas,
            newton_step,
            p_norm,
            trustregion_radius,
            stopping_criteria,
            converged,
        )

    return (
        p_candidate,
        model_hessian_updated,
        lambdas_updated,
        converged,
    )


def check_for_interior_convergence_and_update(
    p_candidate,
    model_hessian,
    lambdas,
    trustregion_radius,
    stopping_criteria,
    converged,
):
    """Check for interior convergence, update candidate vector and lambdas.


    Args:
        p_candidate (np.ndarray): Current candidate vector.
        model_hessian (dict): Dictionary of the hessian matrix and its
            transformations containing the following keys:
            - "initial" (np.ndarray): The initial square hessian matrix supplied
                as input to the subproblem.
            - "initial_plus_lambda" (np.ndarray): The initial hessian matrix
                plus lambda times the identity matrix, computed as

                    H + lambda * I(n),

                where `H` denotes the initial hessian, `I` the identity matrix,
                and `n` the number of rows/columns.
            - "upper_triangular" (np.ndarray): Factorization of the initial hessian
                into its upper triangular matrix.
            - "info_already_factorized" (bool): Boolean indicating whether the hessian
                has already been factorized for the current iteration.
        lambdas (dict): Dictionary containing the current damping
            factor, lambda, as well as its lower and upper bound.
            The respective keys are:
            - "current"
            - "upper"
            - "lower"
        trustregion_radius (float): Trust-region radius.
        stopping_criteria (dict): Dictionary of the two stopping criteria
            containing the following keys:
            - "k_easy" (float): Stopping criterion in the "easy" case.
            - "k_hard" (float): Stopping criterion in the "hard" case.
            See pp. 194-197 from reference _[1] for more details.
        converged (bool): Boolean indicating whether the subproblem has converged
            If True, the iteration will be stopped and the solution vector returned.


    Returns:
        Tuple:
        - p_candidate (np.ndarray): Current, possibly updated, candidate vector.
        - lambdas_updated (dict): Updated dictionary containing the current damping
            factor lambda, its lower and upper bound.
            The respective keys are:
            - "current"
            - "upper"
            - "lower"
        - converged (bool): Boolean indicating whether the subproblem has converged
            If True, the iteration will be stopped and the solution vector returned.
    """
    lambdas_updated = lambdas.copy()

    n = p_candidate.shape[0]

    if lambdas["current"] == 0:
        p_candidate = np.zeros(n)
        converged = True

    s_min, z_min = estimate_smallest_singular_value(model_hessian["upper_triangular"])
    step_len = trustregion_radius + 1

    if (
        step_len ** 2 * s_min ** 2
        <= stopping_criteria["k_hard"] * lambdas["current"] * trustregion_radius ** 2
    ):
        p_candidate = step_len * z_min
        converged = True

    lambdas_updated["upper_bound"] = lambdas["current"]
    lambdas_updated["lower_bound"] = max(
        lambdas["lower_bound"], lambdas["current"] - s_min ** 2
    )
    lambdas_updated["current"] = _update_current_lambda(lambdas_updated)

    return p_candidate, lambdas_updated, converged


def update_lambdas_when_factorization_unsuccessful(
    model_hessian, lambdas, factorization_info
):
    """Update lambdas in the case that factorization of hessian not successful.

    Args:
        model_hessian (dict): Dictionary of the hessian matrix and its
            transformations containing the following keys:
            - "initial" (np.ndarray): The initial square hessian matrix supplied
                as input to the subproblem.
            - "initial_plus_lambda" (np.ndarray): The initial hessian matrix
                plus lambda times the identity matrix, computed as

                    H + lambda * I(n),

                where `H` denotes the initial hessian, `I` the identity matrix,
                and `n` the number of rows/columns.
            - "upper_triangular" (np.ndarray): Factorization of the initial hessian
                into its upper triangular matrix.
            - "info_already_factorized" (bool): Boolean indicating whether the hessian
                has already been factorized for the current iteration.
        lambdas (dict): Dictionary containing the current damping
            factor, lambda, as well as its lower and upper bound.
            The respective keys are:
            - "current"
            - "upper"
            - "lower"
        factorization_info (int): Integer denoting the success of the cholesky
            factorization of H + lambda * I(n)
            (see ``model_hessian["initial_plus_lambda"]``).
            If == 0, the factorization was successful. Otherwise, it was not
            successful and factorization_info represents a positive integer
            such that the leading k by k submatrix of H + lambda * I(n)
            is the first non-positive definite leading submatrix.

    Returns:
        lambdas_updated (dict): Updated dictionary containing the current damping
            factor lambda, its lower and upper bound.
            The respective keys are:
            - "current"
            - "upper"
            - "lower"
    """
    lambdas_updated = lambdas.copy()

    delta, v = _compute_terms_to_make_leading_submatrix_singular(
        model_hessian["initial_plus_lambda"],
        model_hessian["upper_triangular"],
        factorization_info,
    )
    v_norm = norm(v)

    lambdas_updated["lower_bound"] = max(
        lambdas["lower_bound"], lambdas["current"] + delta / v_norm ** 2
    )

    lambdas_updated["current"] = _update_current_lambda(lambdas_updated)

    return lambdas_updated


def _update_current_lambda(lambdas):
    """Update current lambda so that it lies within its bounds.

    Args:
        lambdas (dict): Dictionary containing the current damping
            factor, lambda, as well as its lower and upper bound.
            The respective keys are:
            - "current"
            - "upper"
            - "lower"

    Returns:
        lambdas_updated (dict): Updated dictionary containing the current damping
            factor lambda, its lower and upper bound.
            The respective keys are:
            - "current"
            - "upper"
            - "lower"
    """
    lambda_current = max(
        np.sqrt(lambdas["lower_bound"] * lambdas["upper_bound"]),
        lambdas["lower_bound"]
        + 0.01 * (lambdas["upper_bound"] - lambdas["lower_bound"]),
    )

    return lambda_current


def _compute_gershgorin_bounds(hessian):
    """Compute upper and lower Gregoshgorin bounds for a square matrix.

    The Gregoshgorin bounds are the upper and lower bounds for the
    eigenvalues of the square matrix ``hessian``.
    See ref. [1].

    References:

    .. [1] Conn, A. R., Gould, N. I., & Toint, P. L.
           trust-region methods. 2000. Siam. pp. 19.

    Args:
        hessian (np.ndarray): Square hessian matrix.

    Returns:
        Tuple:
        - lower_bound (np.ndarray): Lower Gregoshgorin bound.
        - upper_bound (np.ndarray): Upper Gregoshgorin bound.
    """
    hessian_diag = np.diag(hessian)
    hessian_diag_abs = np.abs(hessian_diag)
    hessian_row_sums = np.sum(np.abs(hessian), axis=1)

    lower_bound = np.min(hessian_diag + hessian_diag_abs - hessian_row_sums)
    upper_bound = np.max(hessian_diag - hessian_diag_abs + hessian_row_sums)

    return lower_bound, upper_bound


def _compute_newton_step(lambdas, p_norm, w_norm, tr_radius):
    """Compute the Newton step.

    Args:
        lambdas (dict): Dictionary containing the current damping
            factor, lambda, as well as its lower and upper bound.
            The respective keys are:
            - "current"
            - "upper"
            - "lower"
        p_norm (float): Frobenius (i.e. L2-norm) of the candidate vector.
        w_norm (float): Frobenius (i.e. L2-norm) of vector w, which is the solution
            to the following triangular system: U.T w = p.
        trustregion_radius (float): Trust-region radius.

    Returns:
        float: Newton step computed according to formula (4.44) p.87
            from Nocedal and Wright (2006).
    """
    delta_lambda = (p_norm / w_norm) ** 2 * (p_norm - tr_radius) / tr_radius
    newton_step = lambdas["current"] + delta_lambda

    return newton_step


def _update_candidate_and_parameters_when_candidate_within_trustregion(
    p_candidate,
    model_hessian,
    lambdas,
    newton_step,
    trustregion_radius,
    stopping_criteria,
    converged,
):
    """Update candidate vector, hessian, and lambdas when p outside trust-region.

    Args:
        p_candidate (np.ndarray): Current candidate vector.
        model_hessian (dict): Dictionary of the hessian matrix and its
            transformations containing the following keys:
            - "initial" (np.ndarray): The initial square hessian matrix supplied
                as input to the subproblem.
            - "initial_plus_lambda" (np.ndarray): The initial hessian matrix
                plus lambda times the identity matrix, computed as

                    H + lambda * I(n),

                where `H` denotes the initial hessian, `I` the identity matrix,
                and `n` the number of rows/columns.
            - "upper_triangular" (np.ndarray): Factorization of the initial hessian
                into its upper triangular matrix.
            - "info_already_factorized" (bool): Boolean indicating whether the hessian
                has already been factorized for the current iteration.
        lambdas (dict): Dictionary containing the current damping
            factor, lambda, as well as its lower and upper bound.
            The respective keys are:
            - "current"
            - "upper"
            - "lower"
        newton_step (float): Newton step computed according to formula (4.44)
            p.87 from Nocedal and Wright (2006).
        trustregion_radius (float): Trust-region radius.
        stopping_criteria (dict): Dictionary of the two stopping criteria
            containing the following keys:
            - "k_easy" (float): Stopping criterion in the "easy" case.
            - "k_hard" (float): Stopping criterion in the "hard" case.
            See pp. 194-197 from reference _[1] for more details.
        converged (bool): Boolean indicating whether the subproblem has converged
            If True, the iteration will be stopped and the solution vector returned.

    Returns:
        Tuple:
        - p_candidate (np.ndarray): Current candidate vector.
        - model_hessian_updated (dict): Dictionary of the hessian matrix and its
            updated transformations containing the following keys:
            - "initial" (np.ndarray): The initial square hessian matrix supplied
                as input to the subproblem.
            - "initial_plus_lambda" (np.ndarray): The initial hessian matrix
                plus lambda times the identity matrix, computed as

                    H + lambda * I(n),

                where `H` denotes the initial hessian, `I` the identity matrix,
                and `n` the number of rows/columns.
            - "upper_triangular" (np.ndarray): Factorization of the initial hessian
                into its upper triangular matrix.
            - "info_already_factorized" (bool): Boolean indicating whether the hessian
                has already been factorized for the current iteration.
        - lambdas_updated (dict): Updated dictionary containing the current damping
            factor lambda, its lower and upper bound.
            The respective keys are:
            - "current"
            - "upper"
            - "lower"
        - converged (bool): Boolean indicating whether the subproblem has converged
            If True, the iteration will be stopped and the solution vector returned.
    """
    model_hessian_updated = model_hessian.copy()
    lambdas_updated = lambdas.copy()

    n = model_hessian["initial"].shape[0]

    s_min, z_min = estimate_smallest_singular_value(model_hessian["upper_triangular"])

    step_len = _compute_smallest_step_len_for_candidate_vector(
        p_candidate, z_min, trustregion_radius
    )

    quadratic_term = np.dot(
        p_candidate, np.dot(model_hessian["initial_plus_lambda"], p_candidate)
    )

    relative_error = (step_len ** 2 * s_min ** 2) / (
        quadratic_term + lambdas["current"] * trustregion_radius ** 2
    )
    if relative_error <= stopping_criteria["k_hard"]:
        p_candidate = p_candidate + step_len * z_min
        converged = True

    lambdas_updated["lower_bound"] = max(
        lambdas["lower_bound"], lambdas["current"] - s_min ** 2
    )
    lambdas_updated["upper_bound"] = lambdas["current"]

    model_hessian_updated["initial_plus_lambda"] = model_hessian[
        "initial"
    ] + newton_step * np.eye(n)
    _, factorization_unsuccessful = compute_cholesky_factorization(
        model_hessian_updated["initial_plus_lambda"],
        lower=False,
        overwrite_a=False,
        clean=True,
    )

    if factorization_unsuccessful == 0:
        # Factorization was successful
        lambdas_updated["current"] = newton_step
        model_hessian_updated["info_already_factorized"] = True
    else:
        lambdas_updated["lower_bound"] = max(
            lambdas_updated["lower_bound"], newton_step
        )
        lambdas_updated["current"] = _update_current_lambda(lambdas_updated)

    return p_candidate, model_hessian_updated, lambdas_updated, converged


def _update_lambdas_when_candidate_outside_trustregion(
    lambdas, newton_step, p_norm, trustregion_radius, stopping_criteria, converged
):
    """Update lambas in the case that candidate vector lies outside trust-region.

    Args:
        lambdas (dict): Dictionary containing the current damping
            factor, lambda, as well as its lower and upper bound.
            The respective keys are:
            - "current"
            - "upper"
            - "lower"
        newton_step (float): Newton step computed according to formula (4.44)
            p.87 from Nocedal and Wright (2006).
        p_norm (float): Frobenius (i.e. L2-norm) of the candidate vector.
        trustregion_radius (float): Trust-region radius.
        stopping_criteria (dict): Dictionary of the two stopping criteria
            containing the following keys:
            - "k_easy" (float): Stopping criterion in the "easy" case.
            - "k_hard" (float): Stopping criterion in the "hard" case.
            See pp. 194-197 from reference _[1] for more details.
        converged (bool): Boolean indicating whether the subproblem has converged
            If True, the iteration will be stopped and the solution vector returned.

    Returns:
        Tuple:
        - lambdas_updated (dict): Updated dictionary containing the current damping
            factor lambda, its lower and upper bound.
            The respective keys are:
            - "current"
            - "upper"
            - "lower"
        - converged (bool): Boolean indicating whether the subproblem has converged
            If True, the iteration will be stopped and the solution vector returned.
    """
    lambdas_updated = lambdas.copy()

    relative_error = abs(p_norm - trustregion_radius) / trustregion_radius

    if relative_error <= stopping_criteria["k_easy"]:
        converged = True

    lambdas_updated["lower_bound"] = lambdas["current"]
    lambdas_updated["current"] = newton_step

    return lambdas_updated, converged


def _compute_smallest_step_len_for_candidate_vector(
    p_candidate, z_min, trustregion_radius
):
    """Compute the smallest step length for the candidate vector.

    Choose ``step_len`` with the smallest magnitude.
    The reason for this choice is explained at p. 6 in More and D.C. Sorensen (1983),
    just before the formula for `tau`.

    Args:
        p_candidate (np.ndarray): Candidate vector for the direction p.
        z_min (float): Smallest singular value of the hessian matrix.
        trustregion_radius (float): Trust-region radius.

    Returns:
        (float) Step length with the smallest magnitude.
    """
    ta, tb = _solve_scalar_quadratic_equation(p_candidate, z_min, trustregion_radius)
    step_len = min([ta, tb], key=abs)

    return step_len


def _solve_scalar_quadratic_equation(z, d, trustregion_radius):
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
        trustregion_radius (float): Trust-region radius.

    Returns
        Tuple: The two values of t, sorted from low to high.
        - (float) Lower value of t.
        - (float) Higher value of t.
    """
    a = np.dot(d, d)
    b = 2 * np.dot(z, d)
    c = np.dot(z, z) - trustregion_radius ** 2
    sqrt_discriminant = math.sqrt(b * b - 4 * a * c)

    aux = b + math.copysign(sqrt_discriminant, b)
    ta = -aux / (2 * a)
    tb = -2 * c / aux

    return sorted([ta, tb])


def _compute_terms_to_make_leading_submatrix_singular(hessian, upper_triangular, k):
    """Compute terms that make the leading submatrix of the hessian singular.

    The "hessian" here refers to the matrix

        H + lambda * I(n),

    where H is the initial hessian, lambda is the current damping factor,
    I the identity matrix, and m the number of rows/columns of the symmetric
    hessian matrix.

    Args:
        hessian (np.ndarray): Symmetric k by k hessian matrix, which is not
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
    n = len(hessian)

    delta = np.sum(upper_triangular[: k - 1, k - 1] ** 2) - hessian[k - 1, k - 1]

    v = np.zeros(n)
    v[k - 1] = 1

    if k != 1:
        v[: k - 1] = solve_triangular(
            upper_triangular[: k - 1, : k - 1], -upper_triangular[: k - 1, k - 1]
        )

    return delta, v
