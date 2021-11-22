import copy
from functools import partial

import numpy as np
from estimagic.optimization.pounders_auxiliary import add_more_points
from estimagic.optimization.pounders_auxiliary import calc_res
from estimagic.optimization.pounders_auxiliary import compute_fnorm
from estimagic.optimization.pounders_auxiliary import find_nearby_points
from estimagic.optimization.pounders_auxiliary import get_params_quadratic_model
from estimagic.optimization.pounders_auxiliary import improve_model
from estimagic.optimization.pounders_auxiliary import solve_subproblem


def pounders(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    convergence_absolute_gradient_tolerance=1e-4,
    stopping_max_iterations=200,
    trustregion_initial_radius=0.1,
    trustregion_minimal_radius=0.1,
    trustregion_maximal_radius=0.1,
    trustregion_shrinking_factor_not_successful=0.5,
    trustregion_expansion_factor_successful=2,
    theta1=1e-5,
    theta2=1e-4,
    trustregion_threshold_successful=0,
    trustregion_threshold_very_successful=0.1,
    c1=None,
    c2=10,
    trustregion_subproblem_solver="trust-constr",
    trustregion_subproblem_options=None,
):
    algorithm_info = {
        "primary_criterion_entry": "root_contributions",
        "parallelizes": False,
        "needs_scaling": True,
        "name": "pounders",
    }
    criterion = partial(
        criterion_and_derivative, algorithm_info=algorithm_info, task="criterion"
    )
    n_errors = len(
        criterion_and_derivative.keywords["first_criterion_evaluation"]["output"][
            "root_contributions"
        ]
    )

    if c1 is None:
        c1 = np.sqrt(x.shape[0])

    if trustregion_subproblem_options is None:
        trustregion_subproblem_options = {}

    default_options = {"ftol": 1e-6, "xtol": 1e-6, "gtol": 1e-6}
    trustregion_subproblem_options = {
        **default_options,
        **trustregion_subproblem_options,
    }

    rslt = internal_solve_pounders(
        criterion=criterion,
        x0=x,
        nobs=n_errors,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        gtol=convergence_absolute_gradient_tolerance,
        maxiter=stopping_max_iterations,
        delta=trustregion_initial_radius,
        delta_min=trustregion_minimal_radius,
        delta_max=trustregion_maximal_radius,
        gamma0=trustregion_shrinking_factor_not_successful,
        gamma1=trustregion_expansion_factor_successful,
        theta1=theta1,
        theta2=theta2,
        eta0=trustregion_threshold_successful,
        eta1=trustregion_threshold_very_successful,
        c1=c1,
        c2=c2,
        solver_sub=trustregion_subproblem_solver,
        ftol_sub=trustregion_subproblem_options["ftol"],
        xtol_sub=trustregion_subproblem_options["xtol"],
        gtol_sub=trustregion_subproblem_options["gtol"],
    )

    return rslt


def internal_solve_pounders(
    x0,
    nobs,
    criterion,
    delta,
    delta_min,
    delta_max,
    gamma0,
    gamma1,
    theta1,
    theta2,
    eta0,
    eta1,
    c1,
    c2,
    maxiter,
    gtol,
    ftol_sub,
    xtol_sub,
    gtol_sub,
    solver_sub,
    lower_bounds,
    upper_bounds,
):
    """Minimize criterion function using POUNDERS.
    Args:
        x0 (np.ndarray): Initial guess of the parameter vector. Starting points.
        nobs (int): Number of observations/evaluation points.
        criterion (callable): Criterion function to be minimized.
        delta (float): Delta, initial trust-region radius.
        delta_min (float): Minimum value for delta.
        delta_max (float): Maximum value for delta.
        gamma0 (float): Gamma_0.
        gamma1 (float): Gamma_1.
        theta1 (float): Theta_1.
        theta2 (float): Theta_2.
        eta0 (float): Eta_0.
        eta1 (float): Eta_1.
        c1 (float): C_1. Equal to sqrt(*nparams*) by default.
        c2 (int)): C_2. Equal to 10 by default.
        maxiter (int): Maximum number of iterations. If reached, terminate.
        gtol_sub (float): Gradient norm used in the subproblem.
        solver_sub (str): Bound-constraint minimizer for the subproblem.
            Currently, three solvers from the scipy library are supported.
            - "trust-constr"
            - "L-BFGS-B"
            - "SLSQP"
        lower_bounds (np.ndarray): Lower bounds.
            Must have same length as the initial guess of the
            parameter vector. Equal to -1 if not provided by the user.
        upper_bounds (np.ndarray): Upper bounds.
            Must have same length as the initial guess of the
            parameter vector. Equal to 1 if not provided by the user.

    Returns:
        Tuple:
        - solution (np.ndarray): Solution vector.
        - gradient (np.ndarray): Gradient associated with the solution vector.
    """
    n = x0.shape[0]  # number of model parameters
    maxinterp = 2 * n + 1  # max number of interpolation points

    xhist = np.zeros((maxiter + 1, n))
    fhist = np.zeros((maxiter + 1, nobs))
    fnorm = np.zeros(maxiter + 1)
    hess = np.zeros((nobs, n, n))
    model_indices = np.zeros(maxinterp, dtype=int)

    last_mpoints = 0
    niter = 0

    if lower_bounds is not None and upper_bounds is not None:
        if np.max(x0 - upper_bounds) > 1e-10:
            raise ValueError("Starting points > upper bounds.")
        if np.max(lower_bounds - x0) > 1e-10:
            raise ValueError("Starting points < lower bounds.")
        if np.max(x0 + delta - upper_bounds) > 1e-10:
            raise ValueError("Starting points + delta > upper bounds.")

    xhist[0] = x0
    fhist[0, :] = criterion(x0)
    fnorm[0] = compute_fnorm(criterion_value=fhist[0, :])

    minnorm = fnorm[0]
    minindex = 0

    # Increment parameters separately by delta
    for i in range(n):
        x1 = copy.deepcopy(x0)
        x1[i] += delta

        xhist[i + 1, :] = x1
        fhist[i + 1, :] = criterion(x1)
        fnorm[i + 1] = compute_fnorm(criterion_value=fhist[i + 1, :])

        if fnorm[i + 1] < minnorm:
            minnorm = fnorm[i + 1]
            minindex = i + 1

    xmin = xhist[minindex, :]
    fmin = fhist[minindex, :]
    fnorm_min = minnorm

    # centering around new trust-region and normalize to [-1, 1]
    indices_not_min = [i for i in range(n + 1) if i != minindex]
    xk = (xhist[indices_not_min, :] - xmin) / delta
    fdiff = fhist[indices_not_min, :] - fmin

    # Determine the initial quadratic model
    fdiff = np.linalg.solve(xk, fdiff)

    jac_res = np.dot(fdiff, fmin)
    hess_res = np.dot(fdiff, fdiff.T)
    gnorm = np.linalg.norm(jac_res)
    gnorm *= delta

    valid = True
    reason = True
    nhist = n + 1
    mpoints = n + 1

    while reason is True:
        niter += 1

        # Solve the subproblem min{Q(s): ||s|| <= 1.0}
        rslt = solve_subproblem(
            solution=xhist[minindex, :],
            delta=delta,
            jac_res=jac_res,
            hess_res=hess_res,
            ftol=ftol_sub,
            xtol=xtol_sub,
            gtol=gtol_sub,
            solver=solver_sub,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

        qmin = -rslt.fun
        xplus = xmin + rslt.x * delta
        xhist[nhist, :] = xplus
        fhist[nhist, :] = criterion(xhist[nhist, :])
        fnorm[nhist] = compute_fnorm(criterion_value=fhist[nhist, :])
        rho = (fnorm[minindex] - fnorm[nhist]) / qmin

        nhist += 1

        # Update the center
        if (rho >= eta1) or (rho > eta0 and valid is True):
            # Update model to reflect new base point
            x1 = (xplus - xmin) / delta

            fdiff += np.dot(hess, x1).T
            fmin += 0.5 * np.dot(np.dot(x1, hess), x1) + np.dot(x1, fdiff)

            fnorm_min += np.dot(x1, jac_res) + 0.5 * np.dot(hess_res, x1)
            jac_res += np.dot(hess_res, x1)

            minindex = nhist - 1
            minnorm = fnorm[minindex]

            # Change current center
            xmin = xhist[minindex, :]

        # Evaluate at a model improving point if necessary
        # Note: valid is True in first iteration
        qmat = np.zeros((n, n))
        if valid is False:
            q_is_I = 1
            mpoints = 0
            qmat, model_indices, mpoints, q_is_I = find_nearby_points(
                xhist=xhist,
                xmin=xmin,
                qmat=qmat,
                q_is_I=q_is_I,
                delta=delta,
                theta1=theta1,
                c=c1,
                model_indices=model_indices,
                n=n,
                mpoints=mpoints,
                nhist=nhist,
            )

            if mpoints < n:
                addallpoints = 1
                xhist, fhist, fnorm, model_indices, mpoints, nhist = improve_model(
                    xhist=xhist,
                    fhist=fhist,
                    fnorm=fnorm,
                    jac_res=jac_res,
                    hess_res=hess_res,
                    qmat=qmat,
                    model_indices=model_indices,
                    minindex=minindex,
                    mpoints=mpoints,
                    addallpoints=addallpoints,
                    n=n,
                    nhist=nhist,
                    delta=delta,
                    criterion=criterion,
                    lower_bounds=lower_bounds,
                    upper_bounds=upper_bounds,
                )

        # Update the trust region radius
        delta_old = delta
        xnorm_sub = np.sqrt(np.sum(rslt.x ** 2))

        if rho >= eta1 and xnorm_sub > 0.5 * delta:
            delta = min(delta * gamma1, delta_max)
        elif valid is True:
            delta = max(delta * gamma0, delta_min)

        # Compute the next interpolation set
        q_is_I = 1
        mpoints = 0
        qmat, model_indices, mpoints, q_is_I = find_nearby_points(
            xhist=xhist,
            xmin=xmin,
            qmat=qmat,
            q_is_I=q_is_I,
            delta=delta,
            theta1=theta1,
            c=c1,
            model_indices=model_indices,
            n=n,
            mpoints=mpoints,
            nhist=nhist,
        )

        if mpoints == n:
            valid = True
        else:
            valid = False
            qmat, model_indices, mpoints, q_is_I = find_nearby_points(
                xhist=xhist,
                xmin=xmin,
                qmat=qmat,
                q_is_I=q_is_I,
                delta=delta,
                theta1=theta1,
                c=c2,
                model_indices=model_indices,
                n=n,
                mpoints=mpoints,
                nhist=nhist,
            )

            if n > mpoints:
                # Model not valid. Add geometry points
                addallpoints = n - mpoints
                xhist, fhist, fnorm, model_indices, mpoints, nhist = improve_model(
                    xhist=xhist,
                    fhist=fhist,
                    fnorm=fnorm,
                    jac_res=jac_res,
                    hess_res=hess_res,
                    qmat=qmat,
                    model_indices=model_indices,
                    minindex=minindex,
                    mpoints=mpoints,
                    addallpoints=addallpoints,
                    n=n,
                    nhist=nhist,
                    delta=delta,
                    criterion=criterion,
                    lower_bounds=lower_bounds,
                    upper_bounds=upper_bounds,
                )

        model_indices[1 : mpoints + 1] = model_indices[:mpoints]
        mpoints += 1
        model_indices[0] = minindex

        L, Z, N, M, mpoints = add_more_points(
            xhist=xhist,
            xmin=xmin,
            model_indices=model_indices,
            minindex=minindex,
            delta=delta,
            c2=c2,
            theta2=theta2,
            n=n,
            maxinterp=maxinterp,
            mpoints=mpoints,
            nhist=nhist,
        )

        xk = (xhist[model_indices[:mpoints]] - xmin) / delta_old
        res = np.zeros((maxinterp, nobs))

        for j in range(nobs):
            xk_hess = np.dot(xk, hess[j, :, :])

            for i in range(mpoints):
                res[i, j] = (
                    -fmin[j]
                    - np.dot(fdiff[:, j], xk[i, :])
                    - 0.5 * np.dot(xk_hess[i, :], xk[i, :])
                    + fhist[model_indices[i], j]
                )

        jac_quadratic, hess_quadratic = get_params_quadratic_model(
            L=L, Z=Z, N=N, M=M, res=res, mpoints=mpoints, n=n, nobs=nobs
        )
        fdiff = jac_quadratic.T + (delta / delta_old) * fdiff
        hess = hess_quadratic + (delta / delta_old) ** 2 * hess

        fmin = fhist[minindex]
        fnorm_min = fnorm[minindex]
        jac_res, hess_res = calc_res(fdiff=fdiff, fmin=fmin, hess=hess)

        gradient = jac_res
        gnorm = np.linalg.norm(gradient)
        gnorm *= delta

        if gnorm < gtol:
            reason = False

        if niter > maxiter:
            reason = False

        # Test for repeated model
        last_model_indices = np.zeros(maxinterp, dtype=int)
        if mpoints == last_mpoints:
            same = True
        else:
            same = False

        for i in range(mpoints):
            if same:
                if model_indices[i] == last_model_indices[i]:
                    same = True
                else:
                    same = False
            last_model_indices[i] = model_indices[i]

        last_mpoints = mpoints
        if (same is True) and (delta == delta_old):
            # Identical model used in successive iterations
            reason = False

    rslt_dict = {
        "solution_x": xhist[minindex, :],
        "solution_criterion": fhist[minindex, :],
        "n_iterations": niter,
        "message": "Under development.",
    }

    return rslt_dict
