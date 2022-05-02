"""Collection of solvers for a quadratic trust-region subproblem."""
from collections import namedtuple
from copy import copy

import numpy as np
from estimagic.optimization._trustregion_bounded_newton_quadratic import (
    apply_bounds_to_x_candidate,
)
from estimagic.optimization._trustregion_bounded_newton_quadratic import (
    check_for_convergence_conjugate_gradient,
)
from estimagic.optimization._trustregion_bounded_newton_quadratic import (
    compute_conjugate_gradient_step,
)
from estimagic.optimization._trustregion_bounded_newton_quadratic import (
    compute_predicted_reduction_from_conjugate_gradient_step,
)
from estimagic.optimization._trustregion_bounded_newton_quadratic import (
    take_preliminary_steepest_descent_step_and_check_for_solution,
)
from estimagic.optimization._trustregion_bounded_newton_quadratic import (
    update_trustregion_radius_conjugate_gradient,
)
from estimagic.optimization._trustregion_gqtpar_quadratic import (
    add_lambda_and_factorize_hessian,
)
from estimagic.optimization._trustregion_gqtpar_quadratic import (
    check_for_interior_convergence_and_update,
)
from estimagic.optimization._trustregion_gqtpar_quadratic import (
    find_new_candidate_and_update_parameters,
)
from estimagic.optimization._trustregion_gqtpar_quadratic import (
    get_initial_guess_for_lambdas,
)
from estimagic.optimization._trustregion_gqtpar_quadratic import (
    update_lambdas_when_factorization_unsuccessful,
)


def minimize_bntr_quadratic(
    model,
    lower_bounds,
    upper_bounds,
    *,
    maxiter,
    maxiter_steepest_descent,
    step_size_newton,
    ftol_abs,
    ftol_scaled,
    xtol,
    gtol_abs,
    gtol_rel,
    gtol_scaled,
    steptol
):
    """Minimize a bounded trust-region subproblem via Newton Conjugate Gradient method.

    The BNTR (Bounded Newton Trust Rregion) algorithm uses an active-set approach
    to solve the symmetric system of equations:

        Hessian @ x = - gradient

    only for the inactive parameters of x that lie within the bounds. The active-set
    estimation employed here is based on Bertsekas (:cite:`Bertsekas1982`).

    In the main loop, BNTR globalizes the Newton step using a trust-region method
    based on the predicted versus actual reduction in the criterion function.
    The trust-region radius is increased only if the accepted step is at the
    trust-region boundary.


    Args:
        model (namedtuple): Named tuple containing the parameters of the
            main model, i.e.:
            - "linear_terms", a np.ndarray of shape (n,) and
            - "square_terms", a np.ndarray of shape (n,n).
        lower_bounds (np.ndarray): Lower bound on parameter vector x.
            Must have same length as the initial guess of the
            parameter vector. Equal to -1 if not provided by the user.
        upper_bounds (np.ndarray): Upper bounds on parameter vector x.
        maxiter (int): Maximum number of iterations. If reached, terminate.
        maxiter_steepest_descent (int): Maximum number of steepest descent iterations
            to perform when the trust-region subsolver BNTR is used.
        step_size_newton (float): Parameter to scale the size of the newton step
            before the preliminary steepest descent step is taken.
        ftol_abs (float): Convergence tolerance for the absolute difference
            between f(k+1) - f(k).
        ftol_scaled (float): Convergence tolerance for the scaled difference
            between f(k+1) - f(k).
        xtol_sub (float): Convergence tolerance for the absolute difference
            between max(x(k+1) - x(k)).
        gtol_abs_sub (float): Convergence tolerance for the absolute gradient norm.
        gtol_rel_sub (float): Convergence tolerance for the relative gradient norm.
        gtol_scaled_sub (float): Convergence tolerance for the scaled gradient norm.

    Returns:
        (dict): Result dictionary containing the following keys:
            - "x" (np.ndarray): Solution vector of the subproblem of shape (n,)
            - "criterion" (float): Minimum function value associated with the
                solution.
            - n_iterations (int): Number of iterations the algorithm ran before
                termination.
            - "success" (bool): Boolean indicating whether a solution has been found
                before reaching maxiter.
    """
    x_candidate = np.zeros(model.linear_terms.shape[0])

    (
        x_candidate,
        f_candidate,
        gradient_candidate,
        hessian_inactive,
        trustregion_radius,
        active_bounds_info,
        converged,
    ) = take_preliminary_steepest_descent_step_and_check_for_solution(
        x_candidate,
        model,
        lower_bounds,
        upper_bounds,
        maxiter_steepest_descent,
        step_size_newton,
        gtol_abs,
        gtol_rel,
        gtol_scaled,
    )

    for _niter in range(maxiter):
        x_old = np.copy(x_candidate)
        f_old = copy(f_candidate)

        accept_step = False

        while accept_step is False and converged is False:
            (
                x_inactive_conjugate_gradient,
                gradient_inactive_conjugate_gradient,
                x_norm_conjugate_gradient,
                conjugate_gradient_step,
            ) = compute_conjugate_gradient_step(
                x_candidate,
                gradient_candidate,
                hessian_inactive,
                lower_bounds,
                upper_bounds,
                active_bounds_info,
                trustregion_radius,
            )

            # Temporarily accept the step and project it into the bounds
            x_candidate = x_candidate - conjugate_gradient_step
            x_candidate = apply_bounds_to_x_candidate(
                x_candidate, lower_bounds, upper_bounds
            )
            f_candidate = evaluate_model_criterion(
                x_candidate, model.linear_terms, model.square_terms
            )

            predicted_reduction = (
                compute_predicted_reduction_from_conjugate_gradient_step(
                    x_inactive_conjugate_gradient,
                    gradient_inactive_conjugate_gradient,
                    conjugate_gradient_step,
                    gradient_candidate,
                    hessian_inactive,
                    active_bounds_info,
                )
            )

            actual_reduction = f_old - f_candidate

            trustregion_radius_old = copy(trustregion_radius)

            (
                trustregion_radius,
                accept_step,
            ) = update_trustregion_radius_conjugate_gradient(
                f_candidate,
                predicted_reduction,
                actual_reduction,
                x_norm_conjugate_gradient,
                trustregion_radius,
            )

            if accept_step:
                gradient_candidate = evaluate_model_gradient(x_candidate, model)
            else:
                x_candidate = np.copy(x_old)
                f_candidate = np.copy(f_old)

                if trustregion_radius_old == trustregion_radius:
                    break

            converged = check_for_convergence_conjugate_gradient(
                x_candidate,
                f_candidate,
                x_old,
                f_old,
                gradient_candidate,
                model,
                lower_bounds,
                upper_bounds,
                trustregion_radius,
                ftol_abs=ftol_abs,
                ftol_scaled=ftol_scaled,
                xtol=xtol,
                gtol_abs=gtol_abs,
                gtol_rel=gtol_rel,
                gtol_scaled=gtol_scaled,
                steptol=steptol,
            )

        if converged is True:
            break

    result = {
        "x": x_candidate,
        "criterion": f_candidate,
        "n_iterations": _niter,
        "success": converged,
    }

    return result


def minimize_gqtpar_quadratic(model, *, k_easy=0.1, k_hard=0.2, maxiter=200):
    """Solve the quadratic trust-region subproblem via nearly exact iterative method.

    This subproblem solver is mainly based on Conn et al. (2000) "Trust region methods"
    (:cite:`Conn2000`), pp. 169-200.

    But ideas from Nocedal and Wright (2006) "Numerical optimization"
    (:cite:`Nocedal2006`), pp. 83-91, who implement a similar algorithm,
    were also used.

    The original algorithm was developed by More and Sorensen (1983) (:cite:`More1983`)
    and is known as "GQTPAR".

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
    See pp. 194-197 in :cite:`Conn2000` for a more detailed description.

    Args:
        main_model (namedtuple): Named tuple containing the parameters of the
            main model, i.e.:
            - "linear_terms", a np.ndarray of shape (n,) and
            - "square_terms", a np.ndarray of shape (n,n).
        trustregion_radius (float): Trustregion radius, often referred to as delta.
        k_easy (float): topping criterion for the "easy" case.
        k_hard (float): Stopping criterion for the "hard" case.
        maxiter (int): Maximum number of iterations to perform. If reached,
            terminate.

    Returns:
        (dict): Result dictionary containing the following keys:
            - "x" (np.ndarray): Solution vector of the subproblem of shape (n,)
            - "criterion" (float): Minimum function value associated with the
                solution.
    """
    # Small floating point number signaling that for vectors smaller
    # than that backward substituition is not reliable.
    # See Golub, G. H., Van Loan, C. F. (2013), "Matrix computations", p.165.
    zero_threshold = (
        model.square_terms.shape[0]
        * np.finfo(float).eps
        * np.linalg.norm(model.square_terms, np.Inf)
    )
    stopping_criteria = {
        "k_easy": k_easy,
        "k_hard": k_hard,
    }

    HessianInfo = namedtuple(
        "HessianInfo", ["hessian_plus_lambda", "upper_triangular", "already_factorized"]
    )
    hessian_info = HessianInfo(
        hessian_plus_lambda=None,
        upper_triangular=None,
        already_factorized=False,
    )

    gradient_norm = np.linalg.norm(model.linear_terms)
    lambdas = get_initial_guess_for_lambdas(model)

    converged = False

    for _niter in range(maxiter):

        if hessian_info.already_factorized is True:
            hessian_info = hessian_info._replace(already_factorized=False)
        else:
            hessian_info, factorization_info = add_lambda_and_factorize_hessian(
                model, hessian_info, lambdas
            )

        if factorization_info == 0 and gradient_norm > zero_threshold:
            (
                x_candidate,
                hessian_info,
                lambdas,
                converged,
            ) = find_new_candidate_and_update_parameters(
                model,
                hessian_info,
                lambdas,
                stopping_criteria,
                converged,
            )

        elif factorization_info == 0 and gradient_norm <= zero_threshold:
            x_candidate, lambdas, converged = check_for_interior_convergence_and_update(
                x_candidate,
                hessian_info,
                lambdas,
                stopping_criteria,
                converged,
            )

        else:
            lambdas = update_lambdas_when_factorization_unsuccessful(
                hessian_info,
                lambdas,
                factorization_info,
            )

        if converged is True:
            break

    f_min = evaluate_model_criterion(
        x_candidate, model.linear_terms, model.square_terms
    )

    result = {
        "x": x_candidate,
        "criterion": f_min,
        "n_iterations": _niter,
        "success": converged,
    }

    return result


def evaluate_model_criterion(
    x,
    gradient,
    hessian,
):
    """Evaluate the criterion function value of the main model.

    Args:
        x (np.ndarray): Parameter vector of shape (n,).
        gradient (np.ndarray): Gradient of shape (n,) for which the main model
            shall be evaluated.
        hessian (np.ndarray): Hessian of shape (n, n) for which the main model
            shall be evaulated.

    Returns:
        (float): Criterion value of the main model.
    """
    return np.dot(gradient, x) + 0.5 * np.dot(np.dot(x, hessian), x)


def evaluate_model_gradient(x, model):
    """Evaluate the derivative of the main model.

    Args:
       main_model (namedtuple): Named tuple containing the parameters of the
            main model, i.e.:
            - "linear_terms", a np.ndarray of shape (n,) and
            - "square_terms", a np.ndarray of shape (n,n).

    Returns:
        (np.ndarray): Derivative of the main model of shape (n,).
    """
    return model.linear_terms + np.dot(model.square_terms, x)
