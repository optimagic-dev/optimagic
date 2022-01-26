"""
Implementation of parallelosation of Nelder-Mead algorithm
"""
from estimagic import batch_evaluators
from estimagic.optimization.algo_options import (
    CONVERGENCE_SECOND_BEST_ABSOLUTE_CRITERION_TOLERANCE,
)
from estimagic.optimization.algo_options import (
    CONVERGENCE_SECOND_BEST_ABSOLUTE_PARAMS_TOLERANCE,
)
from estimagic.optimization.algo_options import STOPPING_MAX_ITERATIONS
from numpy import abs
from numpy import argsort
from numpy import array
from numpy import eye
from numpy import fill_diagonal
from numpy import float64
from numpy import full
from numpy import hstack
from numpy import max
from numpy import maximum
from numpy import minimum
from numpy import nonzero
from numpy import ones
from numpy import split
from numpy import take
from numpy import vstack


def neldermead_parallel(
    criterion,
    x,
    *,
    init_simplex_method="Gao-Han",
    n_cores=1,
    adaptive=True,
    maxiter=STOPPING_MAX_ITERATIONS,
    convergence_abs_criterion_tol=CONVERGENCE_SECOND_BEST_ABSOLUTE_CRITERION_TOLERANCE,
    convergence_abs_params_tol=CONVERGENCE_SECOND_BEST_ABSOLUTE_PARAMS_TOLERANCE,
    batch_evaluator="joblib",
):
    """
    Parallel Nelder-Mead algorithm following Lee D., Wiswall M., A parallel
    implementation of the simplex function minimization routine,
    Computational Economics, 2007.

    Parameters
    ----------
    criterion (callable): A function that takes a Numpy array_like as an argument
        and return scalar floating point or a :class:`numpy.ndarray`

    x (array_like): 1-D array of initial value of parameters

    init_simplex_method (string): Name of the method to create initial simplex.
        The default is "Gao-Han".

    n_cores (int): Degrees of parallization. The default is 1 (no parallelization).

    adaptive (bool): Adjust parameters of Nelder-Mead algorithm to accounf
        for simplex size.
        The default is True.

    maxiter (int): Maximum number of algorithm iterations.
        The default is STOPPING_MAX_ITERATIONS.

    convergence_abs_criterion_tol (float): maximal difference between function
        value evaluated on simplex points.
        The default is CONVERGENCE_SECOND_BEST_ABSOLUTE_CRITERION_TOLERANCE.

    convergence_abs_params_tol (float): maximal distance between points in the simplex.
        The default is CONVERGENCE_SECOND_BEST_ABSOLUTE_PARAMS_TOLERANCE.

    batch_evaluator (string or callbale): See :ref:`batch_evaluators` for
        details. Default "joblib".

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    if x.ndim >= 1:
        x = x.ravel()  # check if the vector of initial values is one-dimensional
    j = len(x)  # size of the parameter vector

    if n_cores <= 1:
        p = 1  # if number of cores is nonpositive, set it to 1
    else:
        if (
            n_cores >= j
        ):  # number of parallelisation cannot be bigger than number of parameters
            p = int(j - 1)
        else:
            p = int(n_cores)

    # set parameters of Nelder-Mead algorithm
    # for a discussion about Nlder-Mead parameters see Gao F., Han L., Implementing the
    # Nelder-Mead siplex algorithm with adaptive parameters, Computational Optimization
    # and Applications, 2012
    if adaptive:
        # algorithem parameters alla Gao-Han (adaptive)
        alpha, gamma, beta, tau = (
            1,
            1 + 2 / j,
            0.75 - 1 / (2 * j),
            1 - 1 / j,
        )
    else:
        # standard setting of Nelder-Mead
        alpha, gamma, beta, tau = (
            1,
            2,
            0.5,
            0.5,
        )

    # construct initial simplex using one of feasible methods
    s = vstack(
        [
            x,
        ]
        * (j + 1)
    ).astype(float64)

    # see Wssing, Simon, Proper initialization is crucial for
    # the Nelder–Mead simplex search, Optimization Letters, 2019
    # for a discussion about the choice of initialization
    if init_simplex_method == "Pfeffer":

        c_p = 0.05

        # initial simplex
        fill_diagonal(s[1:, :], x * c_p * (x != 0) + 0.00025 * (x == 0))

    elif init_simplex_method == "Nash":  # alla Nash (1990)

        c_n = 0.1

        # initial simplex
        fill_diagonal(s[1:, :], (x != 0) * (max(x) * c_n + x) + c_n / 100 * (x == 0))

    elif init_simplex_method == "Gao-Han":  # alla Gao & Han (2012)

        c_h = minimum(maximum(max(x), 1), 10)

        # initial simplex
        s = (
            s
            + vstack(
                [
                    array([[(1 - (j + 1) ** 0.5) / j]]) * ones([1, j]),
                    eye(j),
                ]
            )
            * c_h
        )

    elif init_simplex_method == "Varadhan-Borchers":  # alla Spendley et al. (1962)

        c_s = maximum(1, ((x ** 2).sum()) ** 0.5)
        beta1 = c_s / (j * 2 ** 0.5) * ((j + 1) ** 0.5 + j - 1)
        beta2 = c_s / (j * 2 ** 0.5) * ((j + 1) ** 0.5 + -1)

        # initial simplex
        s[1:, :] = s[1:, :] + full([j, j], beta2) + eye(j) * (beta1 - beta2)

    # check if batch is callable
    if not callable(batch_evaluator):
        batch_evaluator = getattr(
            batch_evaluators, f"{batch_evaluator}_batch_evaluator"
        )

    # calculate criterion values for the initial simplex
    f_s = array(batch_evaluator(func=criterion, arguments=s, n_cores=n_cores))[:, None]

    # parallelized function
    def func_parallel(args):

        criterion, s_j, s_j_r, f_s_0, f_s_j, f_s_j_1, m = args  # read arguments

        f_s_j_r = criterion(
            s_j_r
        )  # calculate value of the criterion at the reflection point

        if f_s_j_r < f_s_0:  # if the reflection point is better than the best point

            s_j_e = m + gamma * (s_j_r - m)  # calculate expansion point
            f_s_j_e = criterion(
                s_j_e
            )  # calculate value of the criterion at the expansion point

            if f_s_j_e < f_s_0:  # if the expansion point is better than the best point

                return hstack(
                    [s_j_e, f_s_j_e, 0]
                )  # return the expansion point as a new point

            else:  # if the expansion point is worse than the best point

                return hstack(
                    [s_j_r, f_s_j_r, 0]
                )  # return the reflection point as a new point

        elif (
            f_s_j_r < f_s_j_1
        ):  # if reflection point is better than the next worst point

            return hstack([s_j_r, f_s_j_r, 0])  # return reflection point as a new point

        else:  # if the reflection point is worse than the next worst point

            if (
                f_s_j_r < f_s_j
            ):  # if value of the criterion at reflection point is better than
                # value of the criterion at initial point
                s_j_c = m + beta * (s_j_r - m)  # calculate outside contraction point
            else:
                s_j_c = m - beta * (s_j_r - m)  # calculate inside contraction point

            f_s_j_c = criterion(
                s_j_c
            )  # calculate a value of the criterion at contraction point

            if f_s_j_c < minimum(
                f_s_j, f_s_j_r
            ):  # if ta value of the criterion at contraction point is better
                # than original and refrelction point

                return hstack(
                    [s_j_c, f_s_j_c, 0]
                )  # return contraction point as as new point

            else:
                if f_s_j_r < f_s_j:

                    return hstack(
                        [s_j_r, f_s_j_r, 1]
                    )  # return reflection point as a new point

                else:  # if value of the criterion at contraction point is worse
                    # than the value uf the criterion at the reflection
                    # and the initial points
                    return hstack(
                        [s_j, f_s_j, 1]
                    )  # return the old point as a new point

    optimal = False  # optmisation condition, if True stop the algorithem
    iterations = 0  # number of criterion evaluations

    while not optimal:

        iterations += 1  # new iteration

        # sort points and arguments increasing
        row = argsort(f_s.ravel())
        s = take(s, row, axis=0)
        f_s = take(f_s, row, axis=0)

        # calculate centroid
        m = (s[:-p, :].sum(axis=0)) / (j - p + 1)

        # calculate reflaction points
        s_j_r = m + alpha * (m - s[-p:, :])

        # calculate new points of simplex
        s[-p:, :], f_s[-p:, :], shrink_count = split(
            vstack(
                batch_evaluator(
                    func=func_parallel,
                    arguments=tuple(
                        (
                            criterion,
                            s[j + 1 - p + i, :],
                            s_j_r[i, :],
                            f_s[0, :],
                            f_s[j + 1 - p + i, :],
                            f_s[j - p + i, :],
                            m,
                        )
                        for i in range(p)
                    ),
                    n_cores=p,
                )
            ),
            [-2, -1],
            axis=1,
        )

        # shrink simplex if there is no improvement in every process
        if shrink_count.sum() == p:
            s = (
                tau * s[0:1, :] + (1 - tau) * s
            )  # new simplex is a linear combination of the best point
            # and remaining points
            # evaluate function at new simplex
            f_s = array(
                batch_evaluator(
                    func=criterion,
                    arguments=s,
                    n_cores=n_cores,
                )
            )[:, None]

        # termination criteria
        if (
            max(abs(f_s[0, :] - f_s[1:, :])) <= convergence_abs_criterion_tol
            and max(
                abs(
                    s[0, :]
                    - s[
                        1:,
                    ]
                )
            )
            <= convergence_abs_params_tol
        ):
            optimal = True
            converge = True
            reason_to_stop = "Termination codition satisfied"
        elif iterations >= maxiter:  # if maximum amount of iteration is exceeded
            optimal = True
            converge = False
            reason_to_stop = "Maximum number of interation exceeded"
        continue

    # save results
    result = {
        "solution_x": s[nonzero(f_s == f_s.min())[0][0], :],
        "solution_criterion": f_s.min(),
        "n_iterations": iterations,
        "success": converge,
        "reached_convergence_criterion": reason_to_stop,
    }
    return result
