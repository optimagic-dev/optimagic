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
from estimagic.optimization.subsolvers.bounded_newton_quadratic_fast import (
    apply_bounds_to_x_candidate_fast,
)
from estimagic.optimization.subsolvers.bounded_newton_quadratic_fast import (
    check_for_convergence_fast,
)
from estimagic.optimization.subsolvers.bounded_newton_quadratic_fast import (
    compute_conjugate_gradient_step_fast,
)
from estimagic.optimization.subsolvers.bounded_newton_quadratic_fast import (
    compute_predicted_reduction_from_conjugate_gradient_step_fast,
)
from estimagic.optimization.subsolvers.bounded_newton_quadratic_fast import (
    find_hessian_submatrix_where_bounds_inactive_fast,
)
from estimagic.optimization.subsolvers.bounded_newton_quadratic_fast import (
    get_information_on_active_bounds_fast,
)
from estimagic.optimization.subsolvers.bounded_newton_quadratic_fast import (
    take_preliminary_gradient_descent_step_and_check_for_solution_fast,
)
from estimagic.optimization.subsolvers.bounded_newton_quadratic_fast import (
    update_trustregion_radius_conjugate_gradient_fast,
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
from numba import njit

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
        "default_radius": 100.00,
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
        if converged:
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
                gradient_unprojected = evaluate_model_gradient(
                    x_candidate, model.linear_terms, model.square_terms
                )

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
# Subsolver Fast BNTR
# ======================================================================================


def minimize_bntr_quadratic_fast(
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

    This function serves as a wrapper around the faster, numba-implementation of the
    original BNTR algorithm.

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

    model_gradient = model.linear_terms
    model_hessian = model.square_terms
    (
        x_candidate,
        f_candidate,
        niter,
        converged,
        convergence_reason,
    ) = _minimize_bntr_fast(
        model_gradient=model_gradient,
        model_hessian=model_hessian,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        conjugate_gradient_method=conjugate_gradient_method,
        maxiter=maxiter,
        maxiter_gradient_descent=maxiter_gradient_descent,
        gtol_abs=gtol_abs,
        gtol_rel=gtol_rel,
        gtol_scaled=gtol_scaled,
        gtol_abs_conjugate_gradient=gtol_abs_conjugate_gradient,
        gtol_rel_conjugate_gradient=gtol_rel_conjugate_gradient,
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

        if hessian_info.already_factorized:
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

        if converged:
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


@njit
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
    out = gradient.T @ x + 0.5 * x.T @ hessian @ x
    return out


@njit
def evaluate_model_gradient(x, gradient, hessian):
    """Evaluate the derivative of the main model.

    Args:
        x (np.ndarray): Candidate vector of shape (n,).
        gradient (np.ndarray): Gradient of shape (n,) for which the main model
            shall be evaluated.
        hessian (np.ndarray): Hessian of shape (n, n) for which the main model
            shall be evaulated.


    Returns:
        np.ndarray: Derivative of the main model of shape (n,).
    """
    out = gradient + hessian @ x
    return out


@njit
def _minimize_bntr_fast(
    model_gradient,
    model_hessian,
    lower_bounds,
    upper_bounds,
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

    Thi is the faster,  numba implmementation of the original BNTR algorithm that
    gets wrapped in minimize_bntr_fast

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
        model_gradient (np.ndarray): 1d array of shape (n,) of the linear terms of
            surrogate model.
        model_hessian (np.ndarray): 2d array of shape (n,n) of the square terms of
            the surrogate model.
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
        x (np.ndarray): Solution vector of the subproblem of shape (n,)
        criterion (float): Minimum function value associated with the
            solution.
        n_iterations (int): Number of iterations the algorithm ran before
            termination.
        success (bool): Boolean indicating whether a solution has been found
            before reaching maxiter.
    """

    (
        x_candidate,
        f_candidate,
        gradient_unprojected,
        hessian_bounds_inactive,
        trustregion_radius,
        active_lower_bounds,
        active_upper_bounds,
        active_fixed_bounds,
        active_all_bounds,
        inactive_bounds,
        converged,
        convergence_reason,
    ) = take_preliminary_gradient_descent_step_and_check_for_solution_fast(
        model_gradient,
        model_hessian,
        lower_bounds,
        upper_bounds,
        maxiter_gradient_descent,
        gtol_abs,
        gtol_rel,
        gtol_scaled,
    )

    for niter in range(maxiter + 1):
        if converged:
            break

        x_old = x_candidate
        f_old = f_candidate
        accept_step = False

        while not accept_step and not converged:
            gradient_bounds_inactive = gradient_unprojected[inactive_bounds]
            hessian_bounds_inactive = find_hessian_submatrix_where_bounds_inactive_fast(
                model_hessian, inactive_bounds
            )
            (
                conjugate_gradient_step,
                conjugate_gradient_step_inactive_bounds,
                cg_step_norm,
            ) = compute_conjugate_gradient_step_fast(
                x_candidate,
                gradient_bounds_inactive,
                hessian_bounds_inactive,
                lower_bounds,
                upper_bounds,
                inactive_bounds=inactive_bounds,
                active_lower_bounds=active_lower_bounds,
                active_upper_bounds=active_upper_bounds,
                active_fixed_bounds=active_fixed_bounds,
                trustregion_radius=trustregion_radius,
                conjugate_gradient_method=conjugate_gradient_method,
                gtol_abs_conjugate_gradient=gtol_abs_conjugate_gradient,
                gtol_rel_conjugate_gradient=gtol_rel_conjugate_gradient,
                default_radius=100.00,
                min_radius=1e-10,
                max_radius=1e10,
            )

            x_unbounded = x_candidate + conjugate_gradient_step
            x_candidate = apply_bounds_to_x_candidate_fast(
                x_unbounded, lower_bounds, upper_bounds
            )

            predicted_reduction = (
                compute_predicted_reduction_from_conjugate_gradient_step_fast(
                    conjugate_gradient_step,
                    conjugate_gradient_step_inactive_bounds,
                    gradient_unprojected,
                    gradient_bounds_inactive,
                    hessian_bounds_inactive,
                    inactive_bounds,
                    active_all_bounds,
                )
            )

            f_candidate = evaluate_model_criterion(
                x_candidate, model_gradient, model_hessian
            )
            actual_reduction = f_old - f_candidate

            trustregion_radius_old = trustregion_radius
            (
                trustregion_radius,
                accept_step,
            ) = update_trustregion_radius_conjugate_gradient_fast(
                f_candidate,
                predicted_reduction,
                actual_reduction,
                cg_step_norm,
                trustregion_radius,
                min_radius=1e-10,
                max_radius=1e10,
                eta1=1.0e-4,
                eta2=0.25,
                eta3=0.50,
                eta4=0.90,
                alpha1=0.25,
                alpha2=0.50,
                alpha3=1.00,
                alpha4=2.00,
                alpha5=4.00,
            )

            if accept_step:
                gradient_unprojected = evaluate_model_gradient(
                    x_candidate, model_gradient, model_hessian
                )

                (
                    active_lower_bounds,
                    active_upper_bounds,
                    active_fixed_bounds,
                    active_all_bounds,
                    inactive_bounds,
                ) = get_information_on_active_bounds_fast(
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

            converged, convergence_reason = check_for_convergence_fast(
                x_candidate,
                f_candidate,
                gradient_unprojected,
                model_gradient,
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
    return x_candidate, f_candidate, niter, converged, convergence_reason
