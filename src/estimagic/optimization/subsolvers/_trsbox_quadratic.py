"""Implementation of the quadratic trustregion solver TRSBOX."""
import numpy as np


def minimize_trust_trsbox(
    model_gradient,
    model_hessian,
    trustregion_radius,
    *,
    lower_bounds,
    upper_bounds,
):
    """Minimize a qaudratic trust-region subproblem using the trsbox algorithm.

    Solve the quadratic trust-region subproblem:
      min_x   g.T @ x + 0.5 * x.T @ hess @ x
        s.t.   ||x|| <= trustregion_radius
               lower_bounds <= x <= upper_bounds

    approximately, using an active-set approach, where g denotes the gradient
    and hess the hessian of the quadratic model (i.e. the linear terms and
    square_terms), respectively.
    The subproblem is assumed to be centered, i.e. ``x_center`` is the zero vector.

    The trsbox algorithm applies a conjugate gradient step in its main loop.

    This algorithm is an implementation of the quadratic trsbox routine from
    M. J. D. Powell (2009) "The BOBYQA algorithm for bound constrained
    optimization without derivatives." (cite:`Powell2009`).
    Some modifications to the termination conditions are from the generalized
    routine DFBOLS by Zhang et al. (:cite:`Zhang2010`).


    Args:
        model_gradient (np.ndarray): 1d array of shape (n,) containing the
            gradient (i.e. linear terms) of the quadratic model.
        model_hessian (np.ndarray): 2d array of shape (n, n) containing the
            hessian (i.e .square terms) of the quadratic model.
        lower_bounds (np.ndarray): 1d array of shape (n,) with lower bounds
            for the parameter vector x.
        upper_bounds (np.ndarray): 1d array of shape (n,) with upper bounds
            for the parameter vector x.
        trustregion_radius (float): Radius of the trust-region.

    Returns:
        np.ndarray: Solution vector for the quadratic trust-region subproblem
            of shape (n,).
    """
    n = len(model_gradient)
    x_center = np.zeros(n)

    n_iter = 0
    n_fixed_variables = 0

    x_bounded = np.zeros(n)
    x_bounded[(x_center <= lower_bounds) & (model_gradient >= 0.0)] = -1
    x_bounded[(x_center >= upper_bounds) & (model_gradient <= 0.0)] = 1

    x_candidate = np.zeros(n)
    gradient_projected = np.zeros(n)
    gradient_candidate = model_gradient

    total_reduction = 0.0
    delta_sq = trustregion_radius**2
    curve_min = -1.0
    beta = 0.0

    need_alt_trust_step = False
    max_iter = 100 * n**2

    # Main Conjugate Gradient loop
    for _ in range(max_iter):
        gradient_projected[x_bounded != 0] = 0.0
        if beta == 0.0:
            gradient_projected[x_bounded == 0] = -gradient_candidate[x_bounded == 0]
        else:
            gradient_projected[x_bounded == 0] = (
                beta * gradient_projected[x_bounded == 0]
                - gradient_candidate[x_bounded == 0]
            )
        gradient_projected_sumsq = _sumsq(gradient_projected)

        if gradient_projected_sumsq == 0.0:
            need_alt_trust_step = False
            break

        if beta == 0.0:
            gradient_sumsq = gradient_projected_sumsq
            max_iter = n_iter + n - n_fixed_variables

        if n_iter == 0:
            gradient_sumsq_initial = gradient_sumsq

        if gradient_sumsq <= min(
            1.0e-6 * gradient_sumsq_initial, 1.0e-18
        ) or gradient_sumsq * delta_sq <= min(1.0e-6 * total_reduction**2, 1.0e-18):
            need_alt_trust_step = False
            break

        hess_g = model_hessian @ gradient_projected
        g_x = gradient_projected[x_bounded == 0].T @ x_candidate[x_bounded == 0]
        g_hess_g = gradient_projected[x_bounded == 0].T @ hess_g[x_bounded == 0]
        raw_distance = delta_sq - _sumsq(x_candidate[x_bounded == 0])

        if raw_distance <= 0.0:
            need_alt_trust_step = True
            break

        step_len, distance_to_boundary = _take_unconstrained_step_to_boundary(
            raw_distance, gradient_sumsq, gradient_projected_sumsq, g_x, g_hess_g
        )

        if step_len <= 1.0e-30:
            need_alt_trust_step = False
            break

        step_len, index_bound_active = _reduce_step_len_to_preserve_bounds(
            x_candidate, gradient_projected, step_len, lower_bounds, upper_bounds
        )

        (
            x_candidate,
            gradient_candidate,
            current_reduction,
            total_reduction,
            curve_min,
            gradient_sumsq,
            gradient_sumsq_old,
            n_iter,
        ) = _update_candidate_vectors_and_reduction(
            x_candidate,
            x_bounded,
            gradient_candidate,
            gradient_projected,
            step_len,
            total_reduction,
            curve_min,
            index_bound_active,
            gradient_projected_sumsq,
            gradient_sumsq,
            g_hess_g,
            hess_g,
            n_iter,
        )

        # Restart the conjugate gradient method if it has hit a new bound.
        if index_bound_active is not None:
            n_fixed_variables += 1
            x_bounded[index_bound_active] = (
                1 if gradient_projected[index_bound_active] >= 0.0 else -1
            )
            delta_sq = delta_sq - x_candidate[index_bound_active] ** 2
            if delta_sq <= 0.0:
                need_alt_trust_step = True
                break

            beta = 0.0
            continue  # new CG iteration

        if step_len >= distance_to_boundary:
            need_alt_trust_step = True
            break

        if n_iter == max_iter or current_reduction <= 1.0e-6 * total_reduction:
            need_alt_trust_step = False
            break

        beta = gradient_sumsq / gradient_sumsq_old
        continue  # new CG iteration

    # d, xbdi, xopt, gnew, H, sl, su, nact, qred
    if need_alt_trust_step:
        curve_min = 0.0
        x_candidate, gradient_candidate = _perform_alternative_trustregion_step(
            d=x_candidate,
            xbdi=x_bounded,
            gnew=gradient_candidate,
            hess=model_hessian,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            nact=n_fixed_variables,
            qred=total_reduction,
        )
    else:
        x_candidate = _apply_bounds_to_candidate_vector(
            x_candidate, lower_bounds, upper_bounds, x_bounded
        )

    return x_candidate


def _perform_alternative_trustregion_step(
    d, xbdi, gnew, hess, lower_bounds, upper_bounds, nact, qred
):
    """Perform the alternative trust-region step."""
    n = len(d)
    max_iter = 100 * n**2

    for _ in range(max_iter):
        if nact >= n - 1:
            d = _apply_bounds_to_candidate_vector(d, lower_bounds, upper_bounds, xbdi)
            break

        s = np.zeros(n)
        s[xbdi == 0] = d[xbdi == 0]
        dredsq = _sumsq(d[xbdi == 0])
        dredg = d[xbdi == 0].T @ gnew[xbdi == 0]
        gredsq = _sumsq(gnew[xbdi == 0])

        hs = hess @ s
        hred = hs

        # Let the search direction s be a linear combination of the reduced d
        # and the reduced g that is orthogonal to the reduced d.
        restart_alt_loop = False

        for _ in range(max_iter):
            temp = gredsq * dredsq - dredg**2
            if temp <= 1.0e-4 * qred**2:
                restart_alt_loop = False
                break

            temp = np.sqrt(temp)
            s = np.zeros(n)
            s[xbdi == 0] = (dredg * d[xbdi == 0] - dredsq * gnew[xbdi == 0]) / temp
            sredg = -temp

            (
                xbdi,
                iact,
                nact,
                xsav,
                angbd,
                free_variable_reached_bound,
            ) = _calc_upper_bound_on_tangent(
                d, s, xbdi, lower_bounds, upper_bounds, nact
            )

            if free_variable_reached_bound:
                restart_alt_loop = True
                break

            hs = hess @ s

            shs = np.sum(s[xbdi == 0] * hs[xbdi == 0])
            dhs = np.sum(d[xbdi == 0] * hs[xbdi == 0])
            dhd = np.sum(d[xbdi == 0] * hred[xbdi == 0])

            rdprev, rdnext, redmax, angt, isav, iu = _seek_greatest_criterion_reduction(
                angbd, shs, dhs, dhd, dredg, sredg
            )

            # Return if the reduction is zero. Otherwise, set the sine and cosine
            # of the angle of the alternative iteration, and calculate SDEC.
            if isav == -1:
                restart_alt_loop = False
                break

            if isav < iu - 1:
                temp = (rdnext - rdprev) / (2.0 * redmax - rdprev - rdnext)
                angt = angbd * (float(isav + 1) + 0.5 * temp) / float(iu)

            cth = (1.0 - angt**2) / (1.0 + angt**2)
            sth = 2.0 * angt / (1.0 + angt**2)
            temp = shs + angt * (angt * dhd - 2.0 * dhs)
            sdec = sth * (angt * dredg - sredg - 0.5 * sth * temp)

            if sdec <= 0.0:
                restart_alt_loop = False
                break

            d, gnew, dredg, gredsq, hred = _update_candidate_vectors_and_reduction_alt(
                d, s, xbdi, gnew, cth, sth, hs, hred
            )

            qred += sdec
            if iact is not None and isav == iu - 1:
                nact += 1
                xbdi[iact] = xsav
                restart_alt_loop = True
                break

            if sdec <= 0.01 * qred:
                restart_alt_loop = False
                break

            continue

        if restart_alt_loop:
            continue
        else:
            break

    d = _apply_bounds_to_candidate_vector(d, lower_bounds, upper_bounds, xbdi)

    return d, gnew


def _apply_bounds_to_candidate_vector(d, sl, su, xbdi):
    """Force candidate vector to lie within bounds."""
    xnew = np.clip(sl, d, su)
    xnew[xbdi == -1] = sl[xbdi == -1]
    xnew[xbdi == 1] = su[xbdi == 1]

    return xnew


def _take_unconstrained_step_to_boundary(
    raw_distance, gradient_sumsq, gradient_projected_sumsq, g_x, g_hess_g
):
    """Take unconstrained step, ignoring bounds, up to boundary."""
    temp = np.sqrt(gradient_projected_sumsq * raw_distance + g_x**2)

    distance_to_boundary = (
        raw_distance / (temp + g_x)
        if g_x >= 0.0
        else (temp - g_x) / gradient_projected_sumsq
    )
    step_len = (
        distance_to_boundary
        if g_hess_g <= 0.0
        else min(distance_to_boundary, gradient_sumsq / g_hess_g)
    )

    return step_len, distance_to_boundary


def _update_candidate_vectors_and_reduction(
    x_candidate,
    x_bounded,
    gradient_candidate,
    gradient_projected,
    step_len,
    total_reduction,
    curve_min,
    index_bound_active,
    gradient_projected_sumsq,
    gradient_sumsq,
    g_hess_g,
    hess_g,
    n_iter,
):
    """Update candidate vectors and the associated criterion reduction."""
    current_reduction = 0.0

    if step_len > 0.0:
        n_iter += 1
        current_min = g_hess_g / gradient_projected_sumsq

        if index_bound_active is None and current_min > 0.0:
            curve_min = (
                min(curve_min, current_min) if curve_min != -1.0 else current_min
            )

        gradient_sumsq_old = gradient_sumsq

        gradient_candidate_new = gradient_candidate + step_len * hess_g
        x_candidate_new = x_candidate + step_len * gradient_projected

        gradient_sumsq = _sumsq(gradient_candidate_new[x_bounded == 0])

        current_reduction = max(
            step_len * (gradient_sumsq_old - 0.5 * step_len * g_hess_g), 0.0
        )
        total_reduction = total_reduction + current_reduction

    return (
        x_candidate_new,
        gradient_candidate_new,
        current_reduction,
        total_reduction,
        curve_min,
        gradient_sumsq,
        gradient_sumsq_old,
        n_iter,
    )


def _reduce_step_len_to_preserve_bounds(
    x_candidate, gradient_projected, step_len, lower_bounds, upper_bounds
):
    """Reduce step length, where boundary is hit, to preserve simple bounds."""
    n = len(x_candidate)
    index_bound_active = None

    for i in range(n):
        if gradient_projected[i] != 0.0:
            temp = (
                upper_bounds[i] - x_candidate[i]
                if gradient_projected[i] > 0.0
                else lower_bounds[i] - x_candidate[i]
            ) / gradient_projected[i]

            if temp < step_len:
                step_len = temp
                index_bound_active = i

    return step_len, index_bound_active


def _calc_upper_bound_on_tangent(d, s, xbdi, lower_bounds, upper_bounds, nact):
    """Calculate upper bound on tangent of half the angle to the boundary."""
    n = len(d)

    free_variable_reached_bound = False
    angbd = 1.0
    iact = None
    xsav = None

    for i in range(n):
        if xbdi[i] == 0:
            tempa = d[i] - lower_bounds[i]
            tempb = upper_bounds[i] - d[i]
            if tempa <= 0.0:
                nact += 1
                xbdi[i] = -1
                free_variable_reached_bound = True
                break

            elif tempb <= 0.0:
                nact += 1
                xbdi[i] = 1
                free_variable_reached_bound = True
                break

            ssq = d[i] ** 2 + s[i] ** 2

            temp = ssq - lower_bounds[i] ** 2
            if temp > 0.0:
                temp = np.sqrt(temp) - s[i]
                if angbd * temp > tempa:
                    angbd = tempa / temp
                    iact = i
                    xsav = -1

            temp = ssq - upper_bounds[i] ** 2
            if temp > 0.0:
                temp = np.sqrt(temp) + s[i]
                if angbd * temp > tempb:
                    angbd = tempb / temp
                    iact = i
                    xsav = 1

    return xbdi, iact, nact, xsav, angbd, free_variable_reached_bound


def _seek_greatest_criterion_reduction(angbd, shs, dhs, dhd, dredg, sredg):
    """Seek the greatest reduction in the criterion function.

    The largest reduction is found by looking at a range of equally spaced values
    of ``angt`` in the interval [0, ``angbd``], where ``angt`` is the tangent of
    half the angle of the alternative iteration."""
    rdprev = None
    rdnext = None

    redmax = 0.0
    isav = -1
    redsav = 0.0
    temp = 0.0
    iu = int(17 * angbd + 3.1)

    for i in range(iu):
        angt = angbd * float(i + 1) / float(iu)
        sth = 2.0 * angt / (1.0 + angt**2)
        temp = shs + angt * (angt * dhd - 2.0 * dhs)
        rednew = sth * (angt * dredg - sredg - 0.5 * sth * temp)

        if rednew > redmax:
            redmax = rednew
            isav = i
            rdprev = redsav
        elif i == isav + 1:
            rdnext = rednew
        redsav = rednew

    return rdprev, rdnext, redmax, angt, isav, iu


def _update_candidate_vectors_and_reduction_alt(d, s, xbdi, gnew, cth, sth, hs, hred):
    """Update candidate vectors and the associated criterion reduction.

    If the angle of the alternative iteration is restricted by a bound on a
    free variable, that variable is fixed at the bound.
    """
    gnew += (cth - 1.0) * hred + sth * hs
    d[xbdi == 0] = cth * d[xbdi == 0] + sth * s[xbdi == 0]
    dredg = d[xbdi == 0] @ gnew[xbdi == 0]
    gredsq = _sumsq(gnew[xbdi == 0])
    hred = cth * hred + sth * hs

    return d, gnew, dredg, gredsq, hred


def _sumsq(x):
    """Calculate the sum of squares of a vector."""
    return np.dot(x, x)
