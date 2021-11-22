from functools import partial

import numpy as np
from scipy.linalg import qr_multiply
from scipy.optimize import Bounds
from scipy.optimize import minimize


def compute_fnorm(criterion_value):
    """Returns norm of the criterion function value.
    Args:
        criterion_value (np.ndarray): Value of the criterion function.
    Returns:
        (float): Norm of the criterion function
    """
    return np.dot(criterion_value, criterion_value)


def calc_res(fdiff, fmin, hess):
    """Calculate residuals of the Jacobian and Hessian.
    Args:
        fdiff (np.ndarray): Difference between the criterion function values and *fmin*.
            Shape (*n*, *nobs*)
        fmin (np.ndarray): Values of criterion function associated with
            parameter vector x that yields the lowest criterion function norm.
        hess (np.ndarray): Hessian matrix. Shape (*nobs*, *n*, *n*).
    Returns:
        Tuple:
        - jac_res (np.ndarray): Residuals of the Jacobian. Shape (*n*,).
        - hess_res (np.ndarray): Residuals of the Hessian. Shape (*n*, *n*).
    """
    jac_res = np.dot(fdiff, fmin)
    hess_res = np.dot(fdiff, fdiff.T)

    dim_array = np.ones((1, hess.ndim), int).ravel()
    dim_array[0] = -1
    fmin_reshaped = fmin.reshape(dim_array)

    hess_res += np.sum(fmin_reshaped * hess, axis=0)

    return jac_res, hess_res


def solve_subproblem(
    solution,
    delta,
    jac_res,
    hess_res,
    ftol,
    xtol,
    gtol,
    solver,
    lower_bounds,
    upper_bounds,
):
    """Solve the subproblem.

    Args:
        solution (np.ndarray): Current solution vector.
        delta (float): Current trust region radius.
        jac_res (np.ndarray): Residuals of the Jacobian. Shape (*n*,).
        hess_res (np.ndarray): Residuals of the Hessian. Shape (*n*, *n*).
        gtol (float): Gradient tolerance. Stopping criterion.
        solver (str): Minimizer used to solve the bound-constraint subproblem.
            Currently, three solvers from the scipy library are supported.
            - "trust-constr"
            - "L-BFGS-B"
            - "SLSQP"
        lower_bounds (np.ndarray): Lower bounds for the subproblem.
            Must have same length as the initial guess of the
            parameter vector. Equal to -1 if not provided by the user.
        upper_bounds (np.ndarray): Upper bounds for the subproblem.
            Must have same length as the initial guess of the
            parameter vector. Equal to 1 if not provided by the user.

    Returns:
        Dict[str, np.ndarray]: Result dictionary.
    """
    # Initial guess
    n = jac_res.shape[0]
    x0 = np.zeros(n)

    # Normalize bounds. If not specified, use unit cube [-1, 1]
    if lower_bounds is not None:
        lower_bounds = (lower_bounds - solution) / delta
        lower_bounds[lower_bounds < -1] = -1
    else:
        lower_bounds = -np.ones(n)

    if upper_bounds is not None:
        upper_bounds = (upper_bounds - solution) / delta
        upper_bounds[upper_bounds > 1] = 1
    else:
        upper_bounds = np.ones(n)

    # Check if bounds valid
    if np.max(lower_bounds - upper_bounds) > 1e-10:
        raise ValueError("Upper bounds < lower bounds in subproblem.")
    if np.max(lower_bounds - x0) > 1e-10:
        raise ValueError("Initial guess < lower bounds in subproblem.")
    if np.max(x0 - upper_bounds) > 1e-10:
        raise ValueError("Initial guess > upper bounds in subproblem.")

    bounds = Bounds(lower_bounds, upper_bounds)

    if solver == "trust-constr":
        solver_args = {"hess": "2-point"}
        options = {"xtol": xtol, "gtol": gtol}
    elif solver == "L-BFGS-B":  # , "SLSQP"]:
        solver_args = {}
        options = {"ftol": ftol, "gtol": gtol}
    else:
        raise ValueError("Specififed subproblem solver is not supported.")

    evaluate_subproblem = partial(
        _evaluate_obj_and_grad, jac_res=jac_res, hess_res=hess_res
    )

    rslt = minimize(
        evaluate_subproblem,
        x0,
        method=solver,
        jac=True,
        bounds=bounds,
        **solver_args,
        options=options,
    )

    # Test bounds post-solution
    if np.max(lower_bounds - rslt.x) > 1e-5:
        raise ValueError("Subproblem solution < lower bounds.")
    if np.max(rslt.x - upper_bounds) > 1e-5:
        raise ValueError("Subproblem solution > upper bounds.")

    return rslt


def find_nearby_points(
    xhist,
    xmin,
    qmat,
    q_is_I,
    delta,
    theta1,
    c,
    model_indices,
    n,
    mpoints,
    nhist,
):
    """Find nearby points.

    Args:
        xhist (np.ndarray): Array storing all candidates of the parameter vector.
        xmin (np.ndarray): Values of parameter vector x that yield the lowest
            criterion function norm.
        qmat (np.ndarray): Q matrix.
        q_is_I (int): Indicator whether to calculate the QR decomposition of
            *qmat* and multiply *qmat* with vector *xk_plus*.
        delta (float): Delta, current trust-region radius.
        theta1 (float): Theta_1.
        c (float): C.
        model_indices (np.ndarray): Indices related to *xhist*, i.e. the
            candidates of x that are currently in the model. Shape (2 *n* + 1,).
        n (int): Number of parameters.
        mpoints (int): Current number of model points.
        nhist (int): Current number candidate solutions for x.

    Returns:
        Tuple:
        - qmat (np.ndarray): Q matrix.
        - model_indices (np.ndarray):
        - mpoints (int): Current number of model points.
        - q_is_I (int): Indicator whether to calculate the QR decomposition of
            *qmat* and multiply *qmat* with vector *xk_plus*.
            Relevant for next call of *find_nearby_points*.
    """
    for i in range(nhist - 1, -1, -1):
        xk = (xhist[i, :] - xmin) / delta
        normd = np.linalg.norm(xk)

        xk_plus = xk

        if normd <= c:
            if q_is_I == 0:
                xk_plus, _ = qr_multiply(qmat, xk_plus)

            proj = np.linalg.norm(xk_plus[mpoints:])

            # Add this index to the model
            if proj >= theta1:
                qmat = np.zeros((n, n))
                model_indices[mpoints] = i
                mpoints += 1
                qmat[:, mpoints - 1] = xk
                q_is_I = 0

            if mpoints == n:
                break

    return qmat, model_indices, mpoints, q_is_I


def improve_model(
    xhist,
    fhist,
    fnorm,
    jac_res,
    hess_res,
    qmat,
    model_indices,
    minindex,
    mpoints,
    addallpoints,
    n,
    nhist,
    delta,
    criterion,
    lower_bounds,
    upper_bounds,
):
    """Improve the model.

    Args:
        xhist (np.ndarray): Array storing all candidates of the parameter
            vector. Shape (1000, *n*).
        fhist (np.ndarray): Array storing all evaluations of the criterion
            function. Shape(1000, *nobs*).
        fnorm (np.ndarray): Array storing norm of the criterion function.
            Shape (1000,):
        jac_res (np.ndarray): Residuals of the Jacobian. Shape (*n*,).
        hess_res (np.ndarray): Residuals of the Hessian. Shape (*n*, *n*).
        qmat (np.ndarray): Q matrix.
        model_indices (np.ndarray): Indices related to *xhist*, i.e. the
            candidates of x that are currently in the model. Shape (2 *n* + 1,).
        minindex (int): Index in *xhist* associated with the parameter vector
            that yields the lowest criterion function norm.
        mpoints (int): Current number of model points.
        addallpoints (int): If equal to 0, add points. Else, don't.
        n (int): Number of parameters.
        nhist (int): Current number candidate solutions for x.
        delta (float): Delta, current trust-region radius.
        criterion (callable): Criterion function.
        lower_bounds (np.ndarray): Lower bounds.
            Must have same length as the initial guess of the
            parameter vector. Equal to -1 if not provided by the user.
        upper_bounds (np.ndarray): Upper bounds.
            Must have same length as the initial guess of the
            parameter vector. Equal to 1 if not provided by the user.

    Returns:
        Tuple:
        - xhist (np.ndarray): Array storing all candidates of the parameter
            vector. Shape (1000, *n*).
        - fhist (np.ndarray): Array storing all evaluations of the criterion
            function. Shape(1000, *nobs*).
        - fnorm (np.ndarray): Array storing norm of the criterion function.
            Shape (1000,)
        - mpoints (int): Current number of model points.
        - nhist (int): Current number candidate solutions for x.
    """
    minindex_internal = 0
    minvalue = np.inf
    work = np.zeros(3)

    qtmp, _ = qr_multiply(qmat, np.eye(3), mode="right")

    for i in range(mpoints, n):
        dp = np.dot(qtmp[:, i], jac_res)

        # Model says use the other direction!
        if dp > 0:
            qtmp[:, i] *= -1

        jac_res_new = jac_res + 0.5 * np.dot(hess_res, qtmp[:, i])
        work[i] = np.dot(qtmp[:, i], jac_res_new)

        if (i == mpoints) or (work[i] < minvalue):
            minindex_internal = i
            minvalue = work[i]

        if addallpoints != 0:
            xhist, fhist, fnorm, model_indices, mpoints, nhist = _add_point(
                xhist=xhist,
                fhist=fhist,
                fnorm=fnorm,
                qtmp=qtmp,
                model_indices=model_indices,
                minindex=minindex,
                index=i,
                mpoints=mpoints,
                nhist=nhist,
                delta=delta,
                criterion=criterion,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
            )

    if addallpoints == 0:
        xhist, fhist, fnorm, model_indices, mpoints, nhist = _add_point(
            xhist=xhist,
            fhist=fhist,
            fnorm=fnorm,
            qtmp=qtmp,
            model_indices=model_indices,
            minindex=minindex,
            index=minindex_internal,
            mpoints=mpoints,
            nhist=nhist,
            delta=delta,
            criterion=criterion,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

    return xhist, fhist, fnorm, model_indices, mpoints, nhist


def add_more_points(
    xhist,
    xmin,
    model_indices,
    minindex,
    delta,
    c2,
    theta2,
    n,
    maxinterp,
    mpoints,
    nhist,
):
    """Add more points.
    Args:
        xhist (np.ndarray): Array storing all candidates of the parameter
            vector. Shape (1000, *n*).
        xmin (np.ndarray): Values of parameter vector x that yield the lowest
            criterion function norm.
        model_indices (np.ndarray): Indices related to *xhist*, i.e. the
            candidates of x that are currently in the model. Shape (2 *n* + 1,).
        minindex (int): Index in *xhist* associated with the parameter vector
            that yields the lowest criterion function norm.
        delta (float): Delta, current trust-region radius.
        c2 (int): C_2. Equal to 10 by default.
        theta2 (float): Theta_2.
        n (int): Number of parameters.
        maxinterp (int): Maximum number of interpolation points.
        mpoints (int): Current number of model points.
        nhist (int): Current number candidate solutions for x.
    Returns:
        Tuple:
        - L (np.ndarray): L matrix. Shape(*maxinterp*, *n* (*n* + 1) / 2).
        - Z (np.ndarray): Z matrix. Shape(*maxinterp*, len(*n* + 1 : *mpoints*)).
        - N (np.ndarray): N matrix. Shape(*maxinterp*, *n* + 1).
        - M (np.ndarray): M matrix. Shape(*maxinterp*, *n* (*n* + 1) / 2).
        - mpoints (int): Current number of model points.
    """
    M = np.zeros((maxinterp, n + 1))
    N = np.zeros((maxinterp, int(n * (n + 1) / 2)))
    M[:, 0] = 1

    for i in range(n + 1):
        M[i, 1:] = (xhist[model_indices[i], :] - xmin) / delta
        N[i, :] = _evaluate_phi(x=M[i, 1:])

    # Now we add points until we have maxinterp starting with the most recent ones
    point = nhist - 1
    mpoints = n + 1

    while (mpoints < maxinterp) and (point >= 0):
        # Reject any points already in the model
        reject = 0

        for i in range(n + 1):
            if point == model_indices[i]:
                reject = 1
                break

        if reject == 0:
            workxvec = xhist[point]
            workxvec = workxvec - xhist[minindex]
            normd = np.linalg.norm(workxvec)
            normd /= delta

            if normd > c2:
                reject = 1

        else:
            point -= 1
            continue

        M[mpoints, 1:] = (xhist[point] - xmin) / delta
        N[mpoints, :] = _evaluate_phi(x=M[mpoints, 1:])

        Q_tmp = np.zeros((7, 7))
        Q_tmp[:7, : n + 1] = M

        L_tmp, _ = qr_multiply(
            Q_tmp[: mpoints + 1, :],
            N.T[: int(n * (n + 1) / 2), : mpoints + 1],
            mode="right",
        )
        beta = np.linalg.svd(L_tmp.T[n + 1 :], compute_uv=False)

        if beta[min(mpoints - n, int(n * (n + 1) / 2)) - 1] > theta2:
            # Accept point
            model_indices[mpoints] = point
            L = L_tmp

            mpoints += 1

        point -= 1

    cq, _ = qr_multiply(
        Q_tmp[:mpoints, :], np.eye(maxinterp)[:, :mpoints], mode="right"
    )
    Z = cq[:, n + 1 : mpoints]

    if mpoints == (n + 1):
        L = np.zeros((maxinterp, int(n * (n + 1) / 2)))
        L[:n, :n] = np.eye(n)

    return L, Z, N, M, mpoints


def get_params_quadratic_model(
    L,
    Z,
    N,
    M,
    res,
    mpoints,
    n,
    nobs,
):
    """Get parameters of quadratic model.

    Computes the parameters of the quadratic model Q(x) = c + g'x + 0.5 x G x'
    that satisfies the interpolation conditions Q(X[:,j]) = f(j)
    for j= 1,..., m and with a Hessian matrix of least Frobenius norm.
    Args:
        L (np.ndarray): L matrix. Shape(*maxinterp*, *n* (*n* + 1) / 2).
        Z (np.ndarray): Z matrix. Shape(:*mpoints*, *n* + 1 : *mpoints*).
        N (np.ndarray): N matrix. Shape(*maxinterp*, *n* + 1).
        M (np.ndarray): M matrix. Shape(*maxinterp*, *n* (*n* + 1) / 2).
        res (np.ndarray): Array of residuals of the current iteration.
            Shape (*maxinterp*, *nobs*).
        mpoints (int): Current number of model points.
        n (int): Number of parameters.
        nobs (int): Number of observations.

    Returns:
        Tuple:
        - jac_quadratic (np.ndarray): Jacobian of the quadratic model.
            Shape (*nobs*, *n*).
        - hess_quadratic (np.ndarray): Hessian of the quadratic model.
            Shape (*nobs*, *n*, *n*).
    """
    jac_quadratic = np.zeros((nobs, n))
    hess_quadratic = np.zeros((nobs, n, n))

    if mpoints == (n + 1):
        omega = np.zeros(n)
        beta = np.zeros(int(n * (n + 1) / 2))
    else:
        L_tmp = np.dot(L[:, n + 1 : mpoints].T, L[:, n + 1 : mpoints])

    for k in range(nobs):
        if mpoints != (n + 1):
            # Solve L'*L*Omega = Z' * RES_k
            omega = np.dot(Z[:mpoints, :].T, res[:mpoints, k])
            omega = np.linalg.solve(np.atleast_2d(L_tmp), np.atleast_1d(omega))

            beta = np.dot(np.atleast_2d(L[:, n + 1 : mpoints]), omega)

        rhs = res[:mpoints, k] - np.dot(N[:mpoints, :], beta)

        alpha = np.linalg.solve(M[: n + 1, : n + 1], rhs[: n + 1])
        jac_quadratic[k, :] = alpha[1 : (n + 1)]

        num = 0
        for i in range(n):
            hess_quadratic[k, i, i] = beta[num]
            num += 1
            for j in range(i + 1, n):
                hess_quadratic[k, j, i] = beta[num] / np.sqrt(2)
                hess_quadratic[k, i, j] = beta[num] / np.sqrt(2)
                num += 1

    return jac_quadratic, hess_quadratic


def _evaluate_obj_and_grad(
    x,
    jac_res,
    hess_res,
):
    """Returns the objective and gradient of the subproblem.
    Args:
        x (np.ndarray): Parameter vector.
        jac_res (np.ndarray): Residuals of the Jacobian. Shape (*n*,).
        hess_res (np.ndarray): Residuals of the Hessian. Shape (*n*, *n*).
    Returns:
        Tuple:
        - obj (float): Value of the objective function.
        - grad (np.ndarray): Gradient vector. Shape (*n*,).
    """
    grad = np.dot(hess_res, x)
    obj = 0.5 * np.dot(x, grad) + np.dot(jac_res, x)
    grad += jac_res

    return obj, grad


def _evaluate_phi(x):
    """Evaluate phi.
    Phi = .5*[x(1)^2  sqrt(2)*x(1)*x(2) ... sqrt(2)*x(1)*x(n) ...
        ... x(2)^2 sqrt(2)*x(2)*x(3) .. x(n)^2]
    Args:
        x (np.ndarray): Parameter vector of shape (*n*,).
        n (int): Number of parameters.
    Returns:
        (np.ndarray): Phi vector. Shape (*n* (*n* + 1) / 2,)
    """
    n = x.shape[0]
    phi = np.zeros(int(n * (n + 1) / 2))

    j = 0
    for i in range(n):
        phi[j] = 0.5 * x[i] * x[i]
        j += 1

        for k in range(i + 1, n):
            phi[j] = x[i] * x[k] / np.sqrt(2)
            j += 1

    return phi


def _add_point(
    xhist,
    fhist,
    fnorm,
    qtmp,
    model_indices,
    minindex,
    index,
    mpoints,
    nhist,
    delta,
    criterion,
    lower_bounds,
    upper_bounds,
):
    """Add point to the model

    Args:
        xhist (np.ndarray): Array storing all candidates of the parameter
            vector. Shape (1000, *n*).
        fhist (np.ndarray): Array storing all evaluations of the criterion
            function. Shape(1000, *nobs*).
        fnorm (np.ndarray): Array storing norm of the criterion function.
            Shape (1000,):
        qtmp (np.ndarray): Q matrix containing the parameter vector to add
            to *xhist*. Shape (*n*, *n*).
        model_indices (np.ndarray): Indices related to *xhist*, i.e. the
            candidates of x that are currently in the model. Shape (2 *n* + 1,).
        minindex (int): Index in *xhist* associated with the parameter vector
            that yields the lowest criterion function norm.
        index (int): Index relating to the parameter vector in *qtmp* that
            is added to *xhist*.
        mpoints (int): Current number of model points.
        nhist (int): Current number candidate solutions for x.
        delta (float): Delta, current trust-region radius.
        criterion (callable): Criterion function.
        lower_bounds (np.ndarray): Lower bounds.
            Must have same length as the initial guess of the
            parameter vector. Equal to -1 if not provided by the user.
        upper_bounds (np.ndarray): Upper bounds.
            Must have same length as the initial guess of the
            parameter vector. Equal to 1 if not provided by the user.

    Returns:
        Tuple:
        - xhist (np.ndarray): Array storing all candidates of the parameter
            vector. Shape (1000, *n*).
        - fhist (np.ndarray): Array storing all evaluations of the criterion
            function. Shape(1000, *nobs*).
        - fnorm (np.ndarray): Array storing norm of the criterion function.
            Shape (1000,)
        - model_indices (np.ndarray): Indices related to *xhist*, i.e. the
            candidates of x that are currently in the model. Shape (2 *n* + 1,).
        - mpoints (int): Current number of model points.
        - nhist (int): Current number candidate solutions for x.
    """
    # Create new vector in history: X[newidx] = X[index] + delta * X[index]
    xhist[nhist] = qtmp[:, index]
    xhist[nhist, :] = delta * xhist[nhist, :] + xhist[minindex]

    # Project into feasible region
    if lower_bounds is not None and upper_bounds is not None:
        xhist[nhist] = np.median(
            np.stack([lower_bounds, xhist[nhist], upper_bounds]), axis=0
        )

    # Compute value of new vector
    res = criterion(xhist[nhist])
    fsum = compute_fnorm(res)
    fhist[nhist, :] = res
    fnorm[nhist] = fsum

    # Add new vector to the model
    model_indices[mpoints] = nhist
    mpoints += 1
    nhist += 1

    return xhist, fhist, fnorm, model_indices, mpoints, nhist
