import warnings
from functools import partial
from typing import NamedTuple

import numpy as np
from estimagic.decorators import mark_minimizer
from estimagic.optimization.tranquilo.adjust_radius import adjust_radius
from estimagic.optimization.tranquilo.aggregate_models import get_aggregator
from estimagic.optimization.tranquilo.filter_points import get_sample_filter
from estimagic.optimization.tranquilo.fit_models import get_fitter
from estimagic.optimization.tranquilo.models import ModelInfo
from estimagic.optimization.tranquilo.models import ScalarModel
from estimagic.optimization.tranquilo.options import Bounds
from estimagic.optimization.tranquilo.options import ConvOptions
from estimagic.optimization.tranquilo.options import RadiusOptions
from estimagic.optimization.tranquilo.options import TrustRegion
from estimagic.optimization.tranquilo.sample_points import get_sampler
from estimagic.optimization.tranquilo.solve_subproblem import get_subsolver
from estimagic.optimization.tranquilo.tranquilo_history import History


def _tranquilo(
    criterion,
    x,
    functype,
    lower_bounds=None,
    upper_bounds=None,
    disable_convergence=False,
    n_points_factor=1.0,
    stopping_max_iterations=200,
    sample_filter="keep_all",
):
    # ==================================================================================
    # hardcoded stuff that needs to be made flexible
    # ==================================================================================
    warnings.warn(
        "Tranquilo is extremely experimental. algo_options and results will change "
        "frequently and without notice. Do not use."
    )
    maxiter = stopping_max_iterations

    sampler = "sphere"
    sampling_rng = np.random.default_rng(925408)
    sampler_options = {}

    radius_options = RadiusOptions()

    fitter = "ols"
    fit_options = {}

    if functype == "scalar":
        model_info = ModelInfo()
        target_sample_size = int(0.5 * len(x) * (len(x) + 1)) + len(x) + 1
    else:
        model_info = ModelInfo(has_squares=False, has_interactions=False)
        target_sample_size = len(x) + 1

    target_sample_size = int(n_points_factor * target_sample_size)

    subsolver = "bntr"
    solver_options = {}

    if functype == "scalar":
        aggregator = "identity"
    elif functype == "likelihood":
        aggregator = "information_equality_linear"
    elif functype == "least_squares":
        aggregator = "least_squares_linear"
    else:
        raise ValueError(f"Invalid functype: {functype}")

    conv_options = ConvOptions()

    # ==================================================================================

    history = History(functype=functype)
    bounds = Bounds(lower=lower_bounds, upper=upper_bounds)
    trustregion = TrustRegion(center=x, radius=radius_options.initial_radius)
    sample_points = get_sampler(sampler, bounds=bounds, user_options=sampler_options)
    filter_points = get_sample_filter(sample_filter)

    aggregate_vector_model = get_aggregator(
        aggregator=aggregator,
        functype=functype,
        model_info=model_info,
    )

    fit_model = get_fitter(
        fitter=fitter,
        user_options=fit_options,
        model_info=model_info,
    )

    solve_subproblem = get_subsolver(
        solver=subsolver,
        user_options=solver_options,
        bounds=bounds,
    )

    first_eval = criterion(x)
    history.add_entries(x, first_eval)

    state = State(
        index=0,
        model=None,
        rho=None,
        radius=trustregion.radius,
        x=history.get_xs(0),
        fvec=history.get_fvecs(0),
        fval=history.get_fvals(0),
    )

    converged, msg = False, None
    states = [state]
    for _ in range(maxiter):
        old_indices = history.get_indices_in_trustregion(trustregion)
        old_xs = history.get_xs(old_indices)

        filtered_xs, filtered_indices = filter_points(
            xs=old_xs,
            indices=old_indices,
            state=state,
        )

        if state.index not in filtered_indices:
            raise ValueError()

        new_xs = sample_points(
            trustregion=trustregion,
            target_size=target_sample_size,
            existing_xs=filtered_xs,
            rng=sampling_rng,
        )

        new_fvecs = [criterion(_x) for _x in new_xs]
        new_indices = np.arange(history.get_n_fun(), history.get_n_fun() + len(new_xs))
        history.add_entries(new_xs, new_fvecs)

        model_indices = np.hstack([filtered_indices, new_indices])

        model_xs = history.get_xs(model_indices)
        model_fvecs = history.get_fvecs(model_indices)

        centered_xs = (model_xs - trustregion.center) / trustregion.radius

        vector_model = fit_model(centered_xs, model_fvecs)

        scalar_model = aggregate_vector_model(
            vector_model=vector_model,
            fvec_center=state.fvec,
        )

        sub_sol = solve_subproblem(model=scalar_model, trustregion=trustregion)

        candidate_x = sub_sol["x"]

        candidate_fvec = criterion(candidate_x)
        candidate_index = history.get_n_fun()
        history.add_entries(candidate_x, candidate_fvec)
        candidate_fval = history.get_fvals(-1)
        actual_improvement = -(candidate_fval - state.fval)

        rho = _calculate_rho(
            actual_improvement=actual_improvement,
            expected_improvement=sub_sol["expected_improvement"],
        )

        new_radius = adjust_radius(
            radius=trustregion.radius,
            rho=rho,
            step=candidate_x - state.x,
            options=radius_options,
        )
        if actual_improvement > 0:
            trustregion = trustregion._replace(center=candidate_x, radius=new_radius)
            state = State(
                index=candidate_index,
                model=scalar_model,
                rho=rho,
                radius=new_radius,
                x=candidate_x,
                fvec=history.get_fvecs(candidate_index),
                fval=history.get_fvals(candidate_index),
            )

        else:
            trustregion = trustregion._replace(radius=new_radius)
            state = state._replace(
                model=scalar_model,
                rho=rho,
                radius=new_radius,
            )

        states.append(state)

        if actual_improvement > 0 and not disable_convergence:
            converged, msg = _is_converged(states=states, options=conv_options)
            if converged:
                break

    res = {
        "solution_x": state.x,
        "solution_criterion": state.fval,
        "states": states,
        "message": msg,
    }

    return res


def _calculate_rho(actual_improvement, expected_improvement):
    if expected_improvement == 0 and actual_improvement > 0:
        rho = np.inf
    elif expected_improvement == 0:
        rho = -np.inf
    else:
        rho = actual_improvement / expected_improvement

    return rho


class State(NamedTuple):
    index: int
    model: ScalarModel
    rho: float
    radius: float
    x: np.ndarray
    fvec: np.ndarray
    fval: np.ndarray


def _is_converged(states, options):
    old, new = states[-2:]

    f_change_abs = np.abs(old.fval - new.fval)
    f_change_rel = f_change_abs / max(np.abs(old.fval), 0.1)
    x_change_abs = np.linalg.norm(old.x - new.x)
    x_change_rel = np.linalg.norm((old.x - new.x) / np.clip(np.abs(old.x), 0.1, np.inf))
    g_norm_abs = np.linalg.norm(new.model.linear_terms)
    g_norm_rel = g_norm_abs / max(g_norm_abs, 0.1)

    converged = True
    if g_norm_rel <= options.gtol_rel:
        msg = "Relative gradient norm smaller than tolerance."
    elif g_norm_abs <= options.gtol_abs:
        msg = "Absolute gradient norm smaller than tolerance."
    elif f_change_rel <= options.ftol_rel:
        msg = "Relative criterion change smaller than tolerance."
    elif f_change_abs <= options.ftol_abs:
        msg = "Absolute criterion change smaller than tolerance."
    elif x_change_rel <= options.xtol_rel:
        msg = "Relative params change smaller than tolerance."
    elif x_change_abs <= options.xtol_abs:
        msg = "Absolute params change smaller than tolerance."
    else:
        converged = False
        msg = None

    return converged, msg


tranquilo = mark_minimizer(
    func=partial(_tranquilo, functype="scalar"),
    name="tranquilo",
    primary_criterion_entry="value",
    needs_scaling=True,
    is_available=True,
    is_global=False,
)

tranquilo_ls = mark_minimizer(
    func=partial(_tranquilo, functype="least_squares"),
    primary_criterion_entry="root_contributions",
    name="tranquilo_ls",
    needs_scaling=True,
    is_available=True,
    is_global=False,
)
