import numpy as np

from estimagic.optimization.algo_options import (
    CONVERGENCE_RELATIVE_CRITERION_TOLERANCE,
    CONVERGENCE_RELATIVE_GRADIENT_TOLERANCE,
)
from estimagic.optimization.tranquilo.acceptance_decision import get_acceptance_decider
from estimagic.optimization.tranquilo.aggregate_models import get_aggregator
from estimagic.optimization.tranquilo.bounds import Bounds
from estimagic.optimization.tranquilo.estimate_variance import get_variance_estimator
from estimagic.optimization.tranquilo.filter_points import get_sample_filter
from estimagic.optimization.tranquilo.fit_models import get_fitter
from estimagic.optimization.tranquilo.history import History
from estimagic.optimization.tranquilo.options import (
    ConvOptions,
    StagnationOptions,
    StopOptions,
    get_default_acceptance_decider,
    get_default_aggregator,
    get_default_batch_size,
    get_default_model_fitter,
    get_default_residualize,
    get_default_model_type,
    get_default_n_evals_at_start,
    get_default_radius_options,
    get_default_sample_size,
    get_default_search_radius_factor,
    update_option_bundle,
)
from estimagic.optimization.tranquilo.region import Region
from estimagic.optimization.tranquilo.sample_points import get_sampler
from estimagic.optimization.tranquilo.solve_subproblem import get_subsolver
from estimagic.optimization.tranquilo.wrap_criterion import get_wrapped_criterion


def process_arguments(
    # functype, will be partialled out
    functype,
    # problem description
    criterion,
    x,
    lower_bounds=None,
    upper_bounds=None,
    *,
    # basic options
    noisy=False,
    # convergence options
    disable_convergence=False,
    convergence_absolute_criterion_tolerance=0.0,
    convergence_absolute_gradient_tolerance=0.0,
    convergence_absolute_params_tolerance=0.0,
    convergence_relative_criterion_tolerance=CONVERGENCE_RELATIVE_CRITERION_TOLERANCE,
    convergence_relative_gradient_tolerance=CONVERGENCE_RELATIVE_GRADIENT_TOLERANCE,
    convergence_relative_params_tolerance=1e-8,
    convergence_min_trust_region_radius=0.0,
    # stopping options
    stopping_max_criterion_evaluations=2_000,
    stopping_max_iterations=200,
    stopping_max_time=np.inf,
    # single advanced options
    batch_evaluator="joblib",
    n_cores=1,
    batch_size=None,
    sample_size=None,
    model_type=None,
    search_radius_factor=None,
    n_evals_per_point=1,
    n_evals_at_start=None,
    seed=925408,
    # bundled advanced options
    radius_options=None,
    stagnation_options=None,
    # component names and related options
    sampler="optimal_hull",
    sampler_options=None,
    sample_filter="keep_all",
    sample_filter_options=None,
    model_fitter=None,
    model_fitter_options=None,
    cube_subsolver="bntr_fast",
    sphere_subsolver="gqtpar_fast",
    subsolver_options=None,
    acceptance_decider=None,
    acceptance_decider_options=None,
    variance_estimator="classic",
    variance_estimator_options=None,
    infinity_handler="relative",
    residualize=None,
):
    # process convergence options
    conv_options = ConvOptions(
        disable=bool(disable_convergence),
        ftol_abs=float(convergence_absolute_criterion_tolerance),
        gtol_abs=float(convergence_absolute_gradient_tolerance),
        xtol_abs=float(convergence_absolute_params_tolerance),
        ftol_rel=float(convergence_relative_criterion_tolerance),
        gtol_rel=float(convergence_relative_gradient_tolerance),
        xtol_rel=float(convergence_relative_params_tolerance),
        min_radius=float(convergence_min_trust_region_radius),
    )

    # process stopping options
    stop_options = StopOptions(
        max_iter=int(stopping_max_iterations),
        max_eval=int(stopping_max_criterion_evaluations),
        max_time=float(stopping_max_time),
    )

    # process simple options with static defaults
    x = _process_x(x)
    noisy = _process_noisy(noisy)
    n_cores = _process_n_cores(n_cores)
    stagnation_options = update_option_bundle(StagnationOptions(), stagnation_options)
    n_evals_per_point = int(n_evals_per_point)
    sampling_rng = _process_seed(seed)
    n_evals_at_start = _process_n_evals_at_start(
        n_evals_at_start,
        noisy,
    )

    # process options that depend on arguments with static defaults
    search_radius_factor = _process_search_radius_factor(search_radius_factor, functype)
    batch_size = _process_batch_size(batch_size, n_cores)
    radius_options = update_option_bundle(get_default_radius_options(x), radius_options)
    model_type = _process_model_type(model_type, functype)
    acceptance_decider = _process_acceptance_decider(acceptance_decider, noisy)

    # process options that depend on arguments with dependent defaults
    target_sample_size = _process_sample_size(
        sample_size=sample_size,
        model_type=model_type,
        x=x,
    )
    model_fitter = _process_model_fitter(
        model_fitter, model_type=model_type, sample_size=target_sample_size, x=x
    )
    residualize = _process_residualize(residualize, model_fitter=model_fitter)

    # initialize components
    history = History(functype=functype)
    history.add_xs(x)
    evaluate_criterion = get_wrapped_criterion(
        criterion=criterion,
        batch_evaluator=batch_evaluator,
        n_cores=n_cores,
        history=history,
    )
    _bounds = Bounds(lower_bounds, upper_bounds)
    trustregion = Region(
        center=x,
        radius=radius_options.initial_radius,
        bounds=_bounds,
    )

    sample_points = get_sampler(sampler, sampler_options)

    solve_subproblem = get_subsolver(
        cube_solver=cube_subsolver,
        sphere_solver=sphere_subsolver,
        user_options=subsolver_options,
    )

    filter_points = get_sample_filter(
        sample_filter=sample_filter,
        user_options=sample_filter_options,
    )

    fit_model = get_fitter(
        fitter=model_fitter,
        fitter_options=model_fitter_options,
        model_type=model_type,
        infinity_handling=infinity_handler,
        residualize=residualize,
    )

    aggregate_model = get_aggregator(
        aggregator=get_default_aggregator(functype=functype, model_type=model_type),
    )

    estimate_variance = get_variance_estimator(
        variance_estimator,
        variance_estimator_options,
    )

    accept_candidate = get_acceptance_decider(
        acceptance_decider,
        acceptance_decider_options,
    )

    # put everything in a dict
    out = {
        "evaluate_criterion": evaluate_criterion,
        "x": x,
        "noisy": noisy,
        "conv_options": conv_options,
        "stop_options": stop_options,
        "radius_options": radius_options,
        "batch_size": batch_size,
        "target_sample_size": target_sample_size,
        "stagnation_options": stagnation_options,
        "search_radius_factor": search_radius_factor,
        "n_evals_per_point": n_evals_per_point,
        "n_evals_at_start": n_evals_at_start,
        "trustregion": trustregion,
        "sampling_rng": sampling_rng,
        "history": history,
        "sample_points": sample_points,
        "solve_subproblem": solve_subproblem,
        "filter_points": filter_points,
        "fit_model": fit_model,
        "aggregate_model": aggregate_model,
        "estimate_variance": estimate_variance,
        "accept_candidate": accept_candidate,
    }

    return out


def _process_x(x):
    return np.asarray(x, dtype=np.float64)


def _process_noisy(noisy):
    return bool(noisy)


def _process_n_cores(n_cores):
    return int(n_cores)


def _process_batch_size(batch_size, n_cores):
    if batch_size is None:
        batch_size = get_default_batch_size(n_cores)

    elif batch_size < n_cores:
        raise ValueError("batch_size must be at least as large as n_cores.")

    return int(batch_size)


def _process_sample_size(sample_size, model_type, x):
    if sample_size is None:
        out = get_default_sample_size(model_type=model_type, x=x)
    elif callable(sample_size):
        out = sample_size(x=x, model_type=model_type)
    else:
        out = int(sample_size)
    return out


def _process_model_type(model_type, functype):
    out = get_default_model_type(functype) if model_type is None else model_type

    if out not in ["linear", "quadratic"]:
        raise ValueError("model_type must be either 'linear' or 'quadratic'.")

    return out


def _process_search_radius_factor(search_radius_factor, functype):
    if search_radius_factor is None:
        out = get_default_search_radius_factor(functype)
    else:
        out = float(search_radius_factor)

    if out <= 0:
        raise ValueError("search_radius_factor must be positive.")

    return out


def _process_seed(seed):
    return np.random.default_rng(seed)


def _process_acceptance_decider(acceptance_decider, noisy):
    if acceptance_decider is None:
        out = get_default_acceptance_decider(noisy)
    else:
        out = acceptance_decider

    return out


def _process_model_fitter(model_fitter, model_type, sample_size, x):
    if model_fitter is None:
        out = get_default_model_fitter(model_type, sample_size=sample_size, x=x)
    else:
        out = model_fitter

    return out


def _process_residualize(residualize, model_fitter):
    if residualize is None:
        out = get_default_residualize(model_fitter)
    else:
        if not isinstance(residualize, bool):
            raise ValueError("residualize must be a boolean.")
        out = residualize

    return out


def _process_n_evals_at_start(n_evals, noisy):
    if n_evals is None:
        out = get_default_n_evals_at_start(noisy)
    else:
        out = int(n_evals)

    if out < 1:
        raise ValueError("n_initial_acceptance_evals must be non-negative.")

    return out
