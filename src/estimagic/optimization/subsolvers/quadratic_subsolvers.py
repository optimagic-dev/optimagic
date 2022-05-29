"""Collection of solvers for a quadratic trust-region subproblem."""
import numpy as np
from estimagic.optimization.subsolvers.bounded_newton_quadratic import (
    apply_bounds_to_x_candidate,
)
from estimagic.optimization.subsolvers.bounded_newton_quadratic import (
    check_for_convergence,
)
from estimagic.optimization.subsolvers.bounded_newton_quadratic import (
    compute_conjugate_gradient_step,
)
from estimagic.optimization.subsolvers.bounded_newton_quadratic import (
    compute_predicted_reduction_from_conjugate_gradient_step,
)
from estimagic.optimization.subsolvers.bounded_newton_quadratic import (
    find_hessian_submatrix_where_bounds_inactive,
)
from estimagic.optimization.subsolvers.bounded_newton_quadratic import (
    get_information_on_active_bounds,
)
from estimagic.optimization.subsolvers.bounded_newton_quadratic import (
    take_preliminary_gradient_descent_step_and_check_for_solution,
)
from estimagic.optimization.subsolvers.bounded_newton_quadratic import (
    update_trustregion_radius_conjugate_gradient,
)
from estimagic.optimization.subsolvers.gqtpar_quadratic import (
    add_lambda_and_factorize_hessian,
)
from estimagic.optimization.subsolvers.gqtpar_quadratic import (
    check_for_interior_convergence_and_update,
)
from estimagic.optimization.subsolvers.gqtpar_quadratic import (
    find_new_candidate_and_update_parameters,
)
from estimagic.optimization.subsolvers.gqtpar_quadratic import (
    get_initial_guess_for_lambdas,
)
from estimagic.optimization.subsolvers.gqtpar_quadratic import HessianInfo
from estimagic.optimization.subsolvers.gqtpar_quadratic import (
    update_lambdas_when_factorization_unsuccessful,
)

# ======================================================================================
# Subsolver BNTR
# ======================================================================================


def minimize_bntr_quadratic(
    model,
    lower_bounds,
    upper_bounds,
    *,
    conjugate_gradient_method,
    maxiter,
    maxiter_gradient_descent,
    gtol_abs,
    gtol_rel,
    gtol_scaled,
    gtol_abs_conjugate_gradient,
    gtol_rel_conjugate_gradient,
):
    """Minimize a bounded trust-region subproblem via Newton Conjugate Gradient method.

    The BNTR (Bounded Newton Trust Rregion) algorithm uses an active-set approach
    to solve the symmetric system of equations:

        hessian @ x = - gradient

    only for the inactive parameters of x that lie within the bounds. The active-set
    estimation employed here is based on Bertsekas (:cite:`Bertsekas1982`).

    In the main loop, BNTR globalizes the Newton step using a trust-region method
    based on the predicted versus actual reduction in the criterion function.
    The trust-region radius is increased only if the accepted step is at the
    trust-region boundary.


    Args:
        model (NamedTuple): NamedTuple containing the parameters of the
            main model, i.e.:
            - ``linear_terms`` (np.ndarray): 1d array of shape (n,)
            - ``square_terms`` (np.ndarray): 2d array of shape (n,n).
        lower_bounds (np.ndarray): 1d array of shape (n,) with lower bounds
            for the parameter vector x.
        upper_bounds (np.ndarray): 1d array of shape (n,) with upper bounds
            for the parameter vector x.
        conjugate_gradient_method (str): Method for computing the conjugate gradient
            step. Available conjugate gradient methods are:
                - "cg"
                - "steihaug_toint"
                - "trsbox" (default)
        maxiter (int): Maximum number of iterations. If reached, terminate.
        maxiter_gradient_descent (int): Maximum number of steepest descent iterations
            to perform when the trust-region subsolver BNTR is used.
        gtol_abs (float): Convergence tolerance for the absolute gradient norm.
        gtol_rel (float): Convergence tolerance for the relative gradient norm.
        gtol_scaled (float): Convergence tolerance for the scaled gradient norm.
        gtol_abs_conjugate_gradient (float): Convergence tolerance for the absolute
            gradient norm in the conjugate gradient step of the trust-region
            subproblem ("BNTR").
        gtol_rel_conjugate_gradient (float): Convergence tolerance for the relative
            gradient norm in the conjugate gradient step of the trust-region
            subproblem ("BNTR").

    Returns:
        (dict): Result dictionary containing the following keys:
            - ``x`` (np.ndarray): Solution vector of the subproblem of shape (n,)
            - ``criterion`` (float): Minimum function value associated with the
                solution.
            - ``n_iterations`` (int): Number of iterations the algorithm ran before
                termination.
            - ``success`` (bool): Boolean indicating whether a solution has been found
                before reaching maxiter.
    """
    options_update_radius = {
        "eta1": 1.0e-4,
        "eta2": 0.25,
        "eta3": 0.50,
        "eta4": 0.90,
        "alpha1": 0.25,
        "alpha2": 0.50,
        "alpha3": 1.00,
        "alpha4": 2.00,
        "alpha5": 4.00,
        "min_radius": 1e-10,
        "max_radius": 1e10,
        "default_radius": 100,
    }

    x_candidate = np.zeros_like(model.linear_terms)

    (
        x_candidate,
        f_candidate,
        gradient_unprojected,
        hessian_bounds_inactive,
        trustregion_radius,
        active_bounds_info,
        converged,
        convergence_reason,
    ) = take_preliminary_gradient_descent_step_and_check_for_solution(
        x_candidate,
        model,
        lower_bounds,
        upper_bounds,
        maxiter_gradient_descent,
        gtol_abs,
        gtol_rel,
        gtol_scaled,
    )

    for niter in range(maxiter + 1):
        if converged is True:
            break

        x_old = x_candidate
        f_old = f_candidate
        accept_step = False

        while not accept_step and not converged:
            gradient_bounds_inactive = gradient_unprojected[active_bounds_info.inactive]
            hessian_bounds_inactive = find_hessian_submatrix_where_bounds_inactive(
                model, active_bounds_info
            )
            (
                conjugate_gradient_step,
                conjugate_gradient_step_inactive_bounds,
                cg_step_norm,
            ) = compute_conjugate_gradient_step(
                x_candidate,
                gradient_bounds_inactive,
                hessian_bounds_inactive,
                lower_bounds,
                upper_bounds,
                active_bounds_info,
                trustregion_radius,
                conjugate_gradient_method=conjugate_gradient_method,
                gtol_abs_conjugate_gradient=gtol_abs_conjugate_gradient,
                gtol_rel_conjugate_gradient=gtol_rel_conjugate_gradient,
                options_update_radius=options_update_radius,
            )

            x_unbounded = x_candidate + conjugate_gradient_step
            x_candidate = apply_bounds_to_x_candidate(
                x_unbounded, lower_bounds, upper_bounds
            )

            predicted_reduction = (
                compute_predicted_reduction_from_conjugate_gradient_step(
                    conjugate_gradient_step,
                    conjugate_gradient_step_inactive_bounds,
                    gradient_unprojected,
                    gradient_bounds_inactive,
                    hessian_bounds_inactive,
                    active_bounds_info,
                )
            )

            f_candidate = evaluate_model_criterion(
                x_candidate, model.linear_terms, model.square_terms
            )
            actual_reduction = f_old - f_candidate

            trustregion_radius_old = trustregion_radius
            (
                trustregion_radius,
                accept_step,
            ) = update_trustregion_radius_conjugate_gradient(
                f_candidate,
                predicted_reduction,
                actual_reduction,
                cg_step_norm,
                trustregion_radius,
                options_update_radius,
            )

            if accept_step:
                gradient_unprojected = evaluate_model_gradient(x_candidate, model)

                active_bounds_info = get_information_on_active_bounds(
                    x_candidate,
                    gradient_unprojected,
                    lower_bounds,
                    upper_bounds,
                )
            else:
                x_candidate = x_old
                f_candidate = f_old

                if trustregion_radius == trustregion_radius_old:
                    converged = True
                    break

            converged, convergence_reason = check_for_convergence(
                x_candidate,
                f_candidate,
                gradient_unprojected,
                model,
                lower_bounds,
                upper_bounds,
                converged,
                convergence_reason,
                niter,
                maxiter=maxiter,
                gtol_abs=gtol_abs,
                gtol_rel=gtol_rel,
                gtol_scaled=gtol_scaled,
            )

    result = {
        "x": x_candidate,
        "criterion": f_candidate,
        "n_iterations": niter,
        "success": converged,
        "message": convergence_reason,
    }

    return result


# ======================================================================================
# Subsolver GQTPAR
# ======================================================================================


def minimize_gqtpar_quadratic(model, *, k_easy=0.1, k_hard=0.2, maxiter=200):
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
    hessian_info = HessianInfo()

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
        x (np.ndarray): Candidate vector of shape (n,).
        gradient (np.ndarray): Gradient of shape (n,) for which the main model
            shall be evaluated.
        hessian (np.ndarray): Hessian of shape (n, n) for which the main model
            shall be evaulated.

    Returns:
        float: Criterion value of the main model.
    """
    return gradient.T @ x + 0.5 * x.T @ hessian @ x


def evaluate_model_gradient(x, model):
    """Evaluate the derivative of the main model.

    Args:
        x (np.ndarray): Candidate vector of shape (n,).
        main_model (NamedTuple): NamedTuple containing the parameters of the
            main model, i.e.:
            - ``linear_terms``", a np.ndarray of shape (n,) and
            - ``square_terms``, a np.ndarray of shape (n,n).

    Returns:
        np.ndarray: Derivative of the main model of shape (n,).
    """
    return model.linear_terms + model.square_terms @ x
