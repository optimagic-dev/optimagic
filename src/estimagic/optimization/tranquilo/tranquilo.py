import functools
from functools import partial
from typing import NamedTuple

import numpy as np

from estimagic.decorators import mark_minimizer
from estimagic.optimization.tranquilo.adjust_radius import adjust_radius
from estimagic.optimization.tranquilo.filter_points import (
    drop_worst_points,
)
from estimagic.optimization.tranquilo.models import (
    ScalarModel,
    VectorModel,
)
from estimagic.optimization.tranquilo.process_arguments import process_arguments
from estimagic.optimization.tranquilo.region import Region


# wrapping gives us the signature and docstring of process arguments
@functools.wraps(process_arguments)
def _tranquilo(*args, **kwargs):
    internal_kwargs = process_arguments(*args, **kwargs)
    return _internal_tranquilo(**internal_kwargs)


def _internal_tranquilo(
    evaluate_criterion,
    x,
    noisy,
    conv_options,
    stop_options,
    radius_options,
    batch_size,
    target_sample_size,
    stagnation_options,
    search_radius_factor,
    n_evals_per_point,
    n_evals_at_start,
    trustregion,
    sampling_rng,
    history,
    sample_points,
    solve_subproblem,
    filter_points,
    fit_model,
    aggregate_model,
    estimate_variance,
    accept_candidate,
):
    eval_info = {0: n_evals_at_start}

    evaluate_criterion(eval_info)

    _init_fvec = history.get_fvecs(0).mean(axis=0)

    _init_vector_model = VectorModel(
        intercepts=_init_fvec,
        linear_terms=np.zeros((len(_init_fvec), len(x))),
        square_terms=np.zeros((len(_init_fvec), len(x), len(x))),
        shift=trustregion.center,
        scale=trustregion.radius,
    )

    _init_model = aggregate_model(_init_vector_model)

    state = State(
        trustregion=trustregion,
        model_indices=[0],
        model=_init_model,
        vector_model=_init_vector_model,
        index=0,
        x=x,
        fval=np.mean(history.get_fvals(0)),
        rho=np.nan,
        accepted=True,
        new_indices=[0],
        old_indices_discarded=[],
        old_indices_used=[],
        candidate_index=0,
        candidate_x=x,
    )

    states = [state]

    # ==================================================================================
    # main optimization loop
    # ==================================================================================
    converged, msg = False, None
    for _ in range(stop_options.max_iter):
        # ==============================================================================
        # find, filter and count points
        # ==============================================================================

        search_region = state.trustregion._replace(
            radius=search_radius_factor * state.trustregion.radius
        )

        old_indices = history.get_x_indices_in_region(search_region)

        old_xs = history.get_xs(old_indices)

        model_xs, model_indices = filter_points(
            xs=old_xs,
            indices=old_indices,
            state=state,
            target_size=target_sample_size,
        )

        # ==========================================================================
        # sample points if necessary and do simple iteration
        # ==========================================================================
        new_xs = sample_points(
            trustregion=state.trustregion,
            n_points=max(0, target_sample_size - len(model_xs)),
            existing_xs=model_xs,
            rng=sampling_rng,
        )

        new_indices = history.add_xs(new_xs)

        eval_info = {i: n_evals_per_point for i in new_indices}

        evaluate_criterion(eval_info)

        model_indices = _concatenate_indices(model_indices, new_indices)

        model_xs = history.get_xs(model_indices)
        model_data = history.get_model_data(
            x_indices=model_indices,
            average=True,
        )

        vector_model = fit_model(
            *model_data,
            region=state.trustregion,
            old_model=state.vector_model,
            weights=None,
        )

        scalar_model = aggregate_model(
            vector_model=vector_model,
        )

        sub_sol = solve_subproblem(model=scalar_model, trustregion=state.trustregion)

        _relative_step_length = (
            np.linalg.norm(sub_sol.x - state.x) / state.trustregion.radius
        )

        # ==========================================================================
        # If we have enough points, drop points until the relative step length
        # becomes large enough
        # ==========================================================================

        if len(model_xs) > target_sample_size:
            while (
                _relative_step_length < stagnation_options.min_relative_step_keep
                and len(model_xs) > target_sample_size
            ):
                model_xs, model_indices = drop_worst_points(
                    xs=model_xs,
                    indices=model_indices,
                    state=state,
                    n_to_drop=1,
                )

                model_data = history.get_model_data(
                    x_indices=model_indices,
                    average=True,
                )

                vector_model = fit_model(
                    *model_data,
                    region=state.trustregion,
                    old_model=state.vector_model,
                    weights=None,
                )

                scalar_model = aggregate_model(
                    vector_model=vector_model,
                )

                sub_sol = solve_subproblem(
                    model=scalar_model, trustregion=state.trustregion
                )

                _relative_step_length = (
                    np.linalg.norm(sub_sol.x - state.x) / state.trustregion.radius
                )

        # ==========================================================================
        # If step length is still too small, replace the worst point with a new one
        # ==========================================================================

        sample_counter = 0
        while _relative_step_length < stagnation_options.min_relative_step:
            if stagnation_options.drop:
                model_xs, model_indices = drop_worst_points(
                    xs=model_xs,
                    indices=model_indices,
                    state=state,
                    n_to_drop=stagnation_options.sample_increment,
                )

            new_xs = sample_points(
                trustregion=state.trustregion,
                n_points=stagnation_options.sample_increment,
                existing_xs=model_xs,
                rng=sampling_rng,
            )

            new_indices = history.add_xs(new_xs)

            eval_info = {i: n_evals_per_point for i in new_indices}

            evaluate_criterion(eval_info)

            model_indices = _concatenate_indices(model_indices, new_indices)
            model_xs = history.get_xs(model_indices)
            model_data = history.get_model_data(
                x_indices=model_indices,
                average=True,
            )

            vector_model = fit_model(
                *model_data,
                region=state.trustregion,
                old_model=state.vector_model,
                weights=None,
            )

            scalar_model = aggregate_model(
                vector_model=vector_model,
            )

            sub_sol = solve_subproblem(
                model=scalar_model, trustregion=state.trustregion
            )

            _relative_step_length = (
                np.linalg.norm(sub_sol.x - state.x) / state.trustregion.radius
            )

            sample_counter += 1
            if sample_counter >= stagnation_options.max_trials:
                break

        # ==============================================================================
        # fit noise model based on previous acceptance samples
        # ==============================================================================

        if noisy:
            scalar_noise_variance = estimate_variance(
                trustregion=state.trustregion,
                history=history,
                model_type="scalar",
            )
        else:
            scalar_noise_variance = None

        # ==============================================================================
        # acceptance decision
        # ==============================================================================

        acceptance_result = accept_candidate(
            subproblem_solution=sub_sol,
            state=state,
            wrapped_criterion=evaluate_criterion,
            noise_variance=scalar_noise_variance,
            history=history,
        )

        # ==============================================================================
        # update state with information on this iteration
        # ==============================================================================

        state = state._replace(
            model_indices=model_indices,
            model=scalar_model,
            new_indices=np.setdiff1d(model_indices, old_indices),
            old_indices_used=np.intersect1d(model_indices, old_indices),
            old_indices_discarded=np.setdiff1d(old_indices, model_indices),
            **acceptance_result._asdict(),
        )

        states.append(state)

        # ==============================================================================
        # update state for beginning of next iteration
        # ==============================================================================

        new_radius = adjust_radius(
            radius=state.trustregion.radius,
            rho=acceptance_result.rho,
            step_length=acceptance_result.step_length,
            options=radius_options,
        )

        new_trustregion = state.trustregion._replace(
            center=acceptance_result.x, radius=new_radius
        )

        state = state._replace(trustregion=new_trustregion)

        # ==============================================================================
        # convergence check
        # ==============================================================================

        if acceptance_result.accepted and not conv_options.disable:
            converged, msg = _is_converged(states=states, options=conv_options)
            if converged:
                break

        if history.get_n_fun() >= stop_options.max_eval:
            converged = False
            msg = "Maximum number of criterion evaluations reached."
            break

    # ==================================================================================
    # results processing
    # ==================================================================================
    res = {
        "solution_x": state.x,
        "solution_criterion": state.fval,
        "states": states,
        "message": msg,
        "tranquilo_history": history,
    }

    return res


class State(NamedTuple):
    trustregion: Region
    """The trustregion at the beginning of the iteration."""

    # Information about the model used to make the acceptance decision in the iteration
    model_indices: np.ndarray
    """The indices of points used to build the current surrogate model `State.model`.

    The points can be retrieved through calling `history.get_xs(model_indices)`.

    """

    model: ScalarModel
    """The current surrogate model.

    The solution to the subproblem with this model as the criterion is stored as
    `State.candidate_x`.

    """

    vector_model: VectorModel

    # candidate information
    candidate_index: int
    """The index of the candidate point in the history.

    This corresponds to the index of the point in the history that solved the
    subproblem.

    """

    candidate_x: np.ndarray
    """The candidate point.

    Is the same as `history.get_xs(candidate_index)`.

    """

    # accepted parameters and function values at the end of the iteration
    index: int
    """The index of the accepted point in the history."""

    x: np.ndarray
    """The accepted point.

    Is the same as  `history.get_xs(index)`.

    """

    fval: np.ndarray  # this is an estimate for noisy functions
    """The function value at the accepted point.

    If `noisy=False` this is the same as `history.get_fval(index)`. Otherwise, this is
    an average.

    """

    # success information
    rho: float
    """The calculated rho in the current iteration."""

    accepted: bool
    """Whether the candidate point was accepted."""

    # information on existing and new points
    new_indices: np.ndarray
    """The indices of new points generated for the model fitting in this iteration."""

    old_indices_used: np.ndarray
    """The indices of existing points used to build the model in this iteration."""

    old_indices_discarded: np.ndarray
    """The indices of existing points not used to build the model in this iteration."""

    # information on step length
    step_length: float = None
    """The euclidian distance between `State.x` and `State.trustregion.center`."""

    relative_step_length: float = None
    """The step_length divided by the radius of the trustregion."""


def _is_converged(states, options):
    old, new = states[-2:]

    f_change_abs = np.abs(old.fval - new.fval)
    f_change_rel = f_change_abs / max(np.abs(old.fval), 1)
    x_change_abs = np.linalg.norm(old.x - new.x)
    x_change_rel = np.linalg.norm((old.x - new.x) / np.clip(np.abs(old.x), 1, np.inf))
    g_norm_abs = np.linalg.norm(new.model.linear_terms)
    g_norm_rel = g_norm_abs / max(g_norm_abs, 1)

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


def _concatenate_indices(first, second):
    first = np.atleast_1d(first).astype(int)
    second = np.atleast_1d(second).astype(int)
    return np.hstack((first, second))
