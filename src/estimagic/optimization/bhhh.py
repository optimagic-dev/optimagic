import numpy as np


def minimize_bhhh(
    criterion_and_derivative,
    x,
    convergence_absolute_gradient_tolerance=1e-8,
    stopping_max_iterations=100,
):
    """
    Minimize scalar function of one or more variables via the BHHH algorithm.

    Args:
        criterion_and_derivative (callable): The objective function to be minimized.
        x (np.ndarray): Initial guess. Array of real elements of size (n,),
            where `n` is the number of parameters.
        convergence_absolute_gradient_tolerance (float): Tolerance for termination.
        stopping_max_iterations (int): Maximum number of iterations to perform.

    Returns:
        x_hat(np.ndarray): The solution vector of size (n,) containing fitted
            parameter values.
    """
    old_fval, old_jac = criterion_and_derivative(x, task="criterion_and_derivative")

    # Approxmiate Hessian as the outer product of the Jacobian
    hess_approx = np.dot(old_jac.T, old_jac)

    jac_sum = np.sum(hess_approx, axis=0)
    direction = np.linalg.solve(hess_approx, jac_sum)  # Current direction set
    grad_direction = np.dot(jac_sum, direction)

    # Initialize step size
    lambda0 = 1
    lambdak = lambda0

    for _ in range(stopping_max_iterations):
        xk = x + lambdak * direction

        fval = criterion_and_derivative(xk, task="criterion")

        # If previous step was accepted
        if lambdak == lambda0:
            jac = criterion_and_derivative(xk, task="derivative")
            hess_approx = np.dot(jac.T, jac)

        # Line search
        if np.sum(fval) > np.sum(old_fval):
            lambdak /= 2

            if lambdak <= 0.01:
                # Accept step
                x = xk
                old_fval = fval

                # Reset step size
                lambdak = lambda0

        # If decrease in likelihood, calculate new direction vector
        else:
            # Accept step
            x = xk
            old_fval = fval

            jac = criterion_and_derivative(xk, task="derivative")
            jac_sum = np.sum(jac, axis=0)
            direction = np.linalg.solve(hess_approx, jac_sum)
            grad_direction = np.dot(jac_sum, direction)

            # Reset stepsize
            lambdak = lambda0

        if grad_direction < convergence_absolute_gradient_tolerance:
            break

    x_hat = x

    return x_hat
