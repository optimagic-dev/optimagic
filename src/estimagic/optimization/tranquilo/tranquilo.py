import numbers
import warnings
from functools import partial
from typing import NamedTuple

import numpy as np

from estimagic.decorators import mark_minimizer
from estimagic.optimization.tranquilo.acceptance_decision import get_acceptance_decider
from estimagic.optimization.tranquilo.adjust_radius import adjust_radius
from estimagic.optimization.tranquilo.aggregate_models import get_aggregator
from estimagic.optimization.tranquilo.bounds import Bounds, _any_finite
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
)
from estimagic.optimization.tranquilo.region import Region
from estimagic.optimization.tranquilo.sample_points import get_sampler
from estimagic.optimization.tranquilo.solve_subproblem import get_subsolver
from estimagic.optimization.tranquilo.wrap_criterion import get_wrapped_criterion


def _tranquilo(
    criterion,
    x,
    functype,
    lower_bounds=None,
    upper_bounds=None,
    disable_convergence=False,
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
    silence_experimental_warning=False,
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
):
    """Find the local minimum to a noisy optimization problem.

    Args:
        criterion (callable): Function that return values of the objective function.
        x (np.ndarray): Initial guess for the parameter vector.
        functype (str): String indicating whether the criterion is a scalar, a
            likelihood function or a least-square type of function. Valid arguments
            are:
            - "scalar"
            - "likelihood"
            - "least_squares"
        lower_bounds (np.ndarray or NoneTeyp): 1d array of shape (n,) with lower bounds
            for the parameter vector x.
        upper_bounds (np.ndarray or NoneTeyp): 1d array of shape (n,) with upper bounds
            for the parameter vector x.
        disable_convergence (bool): If True, check for convergence criterion and stop
            the iterations.
        stopping_max_iterations (int): Maximum number of iterations to run.
        random_seed (int): The seed used in random number generation.
        sample_filter (str): The method used to filter points in the current trust
            region.
        sampler (str): The sampling method used to sample points from the current
            trust region.
        fitter (str): The method used to fit the surrogate model.
        subsolver (str): The algorithm used for solving the nested surrogate model.
        sample_size (str): Target sample size. One of:
            - "linear": n + 1
            - "powell": 2 * n + 1
            - "quadratic: 0.5 * n * (n + 1) + n + 1
        surrogate_model (str): Type of surrogate model to fit. Both a "linear" and
            "quadratic" surrogate model are supported.
        radius_options (NamedTuple or NoneType): Options for trust-region radius
            management.
        sampler_options (dict or NoneType): Additional keyword arguments passed to the
            sampler function.
        fit_options (dict or NoneType): Additional keyword arguments passed to the
            fitter function.
        solver_options (dict or NoneType): Additional keyword arguments passed to the
            sub-solver function.
        conv_options (NamedTuple or NoneType): Criteria for successful convergence.
        batch_evaluator (str or callabler)
        n_cores (int): Number of cores.

    Returns:
        (dict): Results dictionary with the following items:
            - solution_x (np.ndarray): Solution vector of shape (n,).
            - solution_criterion (np.ndarray): Values of the criterion function at the
                solution vector. Shape (n_obs,).
            - states (list): The history of optimization as a list of the State objects.
            - message (str or NoneType): Message stating which convergence criterion,
                if any has been reached at the end of optimization

    """
    # ==================================================================================
    # experimental warning
    # ==================================================================================
    if not silence_experimental_warning:
        warnings.warn(
            "Tranquilo is extremely experimental. algo_options and results will change "
            "frequently and without notice. Do not use."
        )

    # ==================================================================================
    # set default values for optional arguments
    # ==================================================================================

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

    if sampler is None:
        sampler = "optimal_sphere"

    if subsolver is None:
        if _any_finite(bounds.lower, bounds.upper):
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

    sample_points = get_sampler(
        sampler,
        model_info=model_type,
        user_options=sampler_options,
    )

    filter_points = get_sample_filter(sample_filter, user_options=filter_options)

    aggregate_vector_model = get_aggregator(
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

    acceptance_decider = get_acceptance_decider(
        acceptance_decider=acceptance_decider,
        acceptance_options=acceptance_options,
    )

    eval_info = {0: acceptance_options.n_initial} if noisy else {0: n_evals_per_point}

    evaluate_criterion(eval_info)

    _init_fvec = history.get_fvecs(0).mean(axis=0)
    _init_radius = radius_options.initial_radius * np.max(np.abs(x))
    _init_region = Region(center=x, radius=_init_radius, bounds=bounds)

    _init_vector_model = VectorModel(
        intercepts=_init_fvec,
        linear_terms=np.zeros((len(_init_fvec), len(x))),
        square_terms=np.zeros((len(_init_fvec), len(x), len(x))),
        region=_init_region,
    )

    _init_model = aggregate_vector_model(_init_vector_model)

    state = State(
        trustregion=_init_region,
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
    for _ in range(stopping_max_iterations):
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

        scalar_model = aggregate_vector_model(
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

                scalar_model = aggregate_vector_model(
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

            scalar_model = aggregate_vector_model(
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

        acceptance_result = acceptance_decider(
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

        if history.get_n_fun() >= stopping_max_criterion_evaluations:
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
