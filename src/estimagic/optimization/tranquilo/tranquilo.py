import functools
import numbers
from functools import partial
from typing import NamedTuple

import numpy as np

from estimagic.decorators import mark_minimizer
from estimagic.optimization.tranquilo.acceptance_decision import get_acceptance_decider
from estimagic.optimization.tranquilo.adjust_radius import adjust_radius
from estimagic.optimization.tranquilo.aggregate_models import get_aggregator
from estimagic.optimization.tranquilo.bounds import Bounds
from estimagic.optimization.tranquilo.estimate_variance import get_variance_estimator
from estimagic.optimization.tranquilo.filter_points import (
    drop_worst_points,
    get_sample_filter,
)
from estimagic.optimization.tranquilo.fit_models import get_fitter
from estimagic.optimization.tranquilo.models import (
    ScalarModel,
    VectorModel,
    n_free_params,
)
from estimagic.optimization.tranquilo.new_history import History
from estimagic.optimization.tranquilo.options import (
    AcceptanceOptions,
    ConvOptions,
    RadiusOptions,
    StagnationOptions,
    StopOptions,
)
from estimagic.optimization.tranquilo.process_arguments import process_arguments
from estimagic.optimization.tranquilo.region import Region
from estimagic.optimization.tranquilo.sample_points import get_sampler
from estimagic.optimization.tranquilo.solve_subproblem import get_subsolver
from estimagic.optimization.tranquilo.wrap_criterion import get_wrapped_criterion


# wrapping gives us the signature and docstring of process arguments
@functools.wraps(process_arguments)
def _new_tranquilo(*args, **kwargs):
    internal_kwargs = process_arguments(*args, **kwargs)
    return _internal_tranquilo(**internal_kwargs)


def _internal_tranquilo(
    evaluate_criterion,  # noqa: ARG001
    x,  # noqa: ARG001
    noisy,  # noqa: ARG001
    conv_options,  # noqa: ARG001
    stop_options,  # noqa: ARG001
    radius_options,  # noqa: ARG001
    batch_size,  # noqa: ARG001
    target_sample_size,  # noqa: ARG001
    stagnation_options,  # noqa: ARG001
    search_radius_factor,  # noqa: ARG001
    n_evals_per_point,  # noqa: ARG001
    n_evals_at_start,  # noqa: ARG001
    trustregion,  # noqa: ARG001
    sampling_rng,  # noqa: ARG001
    history,  # noqa: ARG001
    sample_points,  # noqa: ARG001
    solve_subproblem,  # noqa: ARG001
    filter_points,  # noqa: ARG001
    fit_model,  # noqa: ARG001
    aggregate_model,  # noqa: ARG001
    estimate_variance,  # noqa: ARG001
    accept_candidate,  # noqa: ARG001
):
    pass


def _tranquilo(
    criterion,
    x,
    functype,
    lower_bounds=None,
    upper_bounds=None,
    stopping_max_iterations=200,
    stopping_max_criterion_evaluations=2_000,
    random_seed=925408,
    sampler=None,
    sample_filter="keep_all",
    filter_options=None,
    fitter=None,
    subsolver=None,
    sample_size=None,
    surrogate_model=None,
    radius_options=None,
    sampler_options=None,
    fit_options=None,
    solver_options=None,
    conv_options=None,
    batch_evaluator="joblib",
    n_cores=1,
    infinity_handling="relative",
    search_radius_factor=None,
    noisy=False,
    sample_size_factor=None,
    acceptance_decider=None,
    acceptance_options=None,
    variance_estimator="classic",
    variance_estimation_options=None,
    stagnation_options=None,
    n_evals_per_point=1,
    disable_convergence=False,
    n_evals_at_start=None,
):
    # ==================================================================================
    # set default values for optional arguments
    # ==================================================================================
    n_evals_at_start = 5 if noisy else 1
    sampling_rng = np.random.default_rng(random_seed)

    if radius_options is None:
        radius_options = RadiusOptions()
    if sampler_options is None:
        sampler_options = {}
    if fit_options is None:
        if functype == "scalar":
            fit_options = {"residualize": True}
        else:
            fit_options = {}
    if solver_options is None:
        solver_options = {}

    model_type = _process_surrogate_model(
        surrogate_model=surrogate_model,
        functype=functype,
    )

    bounds = Bounds(lower=lower_bounds, upper=upper_bounds)

    sampler = "optimal_hull" if sampler is None else sampler

    if subsolver is None:
        if bounds.has_any:
            subsolver = "bntr_fast"
        else:
            subsolver = "gqtpar_fast"

    if search_radius_factor is None:
        search_radius_factor = 4.25 if functype == "scalar" else 5.0

    target_sample_size = _process_sample_size(
        user_sample_size=sample_size,
        model_type=model_type,
        x=x,
        sample_size_factor=sample_size_factor,
    )

    if fitter is None:
        if functype == "scalar":
            fitter = "tranquilo"
        else:
            fitter = "ols"

    if functype == "scalar":
        aggregator = "identity"
    elif functype == "likelihood":
        aggregator = "information_equality_linear"
    elif functype == "least_squares":
        aggregator = "least_squares_linear"
    else:
        raise ValueError(f"Invalid functype: {functype}")

    if conv_options is None:
        conv_options = ConvOptions()

    if acceptance_decider is None:
        acceptance_decider = "noisy" if noisy else "classic"

    if acceptance_options is None:
        acceptance_options = AcceptanceOptions()

    if stagnation_options is None:
        stagnation_options = StagnationOptions()

    stop_options = StopOptions(
        max_iter=stopping_max_iterations,
        max_eval=stopping_max_criterion_evaluations,
        max_time=np.inf,
    )

    # ==================================================================================
    # initialize compoments for the solver
    # ==================================================================================

    history = History(functype=functype)
    history.add_xs(x)

    evaluate_criterion = get_wrapped_criterion(
        criterion=criterion,
        batch_evaluator=batch_evaluator,
        n_cores=n_cores,
        history=history,
    )

    sample_points = get_sampler(sampler, user_options=sampler_options)

    filter_points = get_sample_filter(sample_filter, user_options=filter_options)

    aggregate_model = get_aggregator(
        aggregator=aggregator,
        functype=functype,
        model_type=model_type,
    )

    fit_model = get_fitter(
        fitter=fitter,
        fitter_options=fit_options,
        model_type=model_type,
        infinity_handling=infinity_handling,
    )

    solve_subproblem = get_subsolver(
        solver=subsolver,
        user_options=solver_options,
        bounds=bounds,
    )

    estimate_variance = get_variance_estimator(
        fitter=variance_estimator,
        user_options=variance_estimation_options,
    )
    # ==================================================================================
    # initialize the optimizer state
    # ==================================================================================

    accept_candidate = get_acceptance_decider(
        acceptance_decider=acceptance_decider,
        acceptance_options=acceptance_options,
    )

    eval_info = {0: n_evals_at_start}

    evaluate_criterion(eval_info)

    _init_fvec = history.get_fvecs(0).mean(axis=0)
    _init_radius = radius_options.initial_radius * np.max(np.abs(x))
    trustregion = Region(center=x, radius=_init_radius, bounds=bounds)

    _init_vector_model = VectorModel(
        intercepts=_init_fvec,
        linear_terms=np.zeros((len(_init_fvec), len(x))),
        square_terms=np.zeros((len(_init_fvec), len(x), len(x))),
        region=trustregion,
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

        if acceptance_result.accepted and not disable_convergence:
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


def _process_surrogate_model(surrogate_model, functype):
    if surrogate_model is None:
        if functype == "scalar":
            surrogate_model = "quadratic"
        else:
            surrogate_model = "linear"

    if isinstance(surrogate_model, str):
        if surrogate_model not in ("linear", "quadratic"):
            raise ValueError(
                f"Invalid surrogate model: {surrogate_model} must be in ('linear', "
                "'quadratic')"
            )
    else:
        raise TypeError(f"Invalid surrogate model: {surrogate_model}")

    return surrogate_model


def _process_sample_size(user_sample_size, model_type, x, sample_size_factor):
    if user_sample_size is None:
        if model_type == "quadratic":
            out = 2 * len(x) + 1
        else:
            out = len(x) + 1

    elif isinstance(user_sample_size, str):
        user_sample_size = user_sample_size.replace(" ", "")
        if user_sample_size in ["linear", "n+1"]:
            out = n_free_params(dim=len(x), model_type="linear")
        elif user_sample_size in ["powell", "2n+1", "2*n+1"]:
            out = 2 * len(x) + 1
        elif user_sample_size == "quadratic":
            out = n_free_params(dim=len(x), model_type="quadratic")
        else:
            raise ValueError(f"Invalid sample size: {user_sample_size}")

    elif isinstance(user_sample_size, numbers.Number):
        out = int(user_sample_size)
    else:
        raise TypeError(f"invalid sample size: {user_sample_size}")

    if sample_size_factor is not None:
        out = int(out * sample_size_factor)

    return out


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
