from typing import NamedTuple
from estimagic.optimization.tranquilo.models import n_free_params

import numpy as np


def get_default_radius_options(x):
    return RadiusOptions(initial_radius=0.1 * np.max(np.abs(x)))


def get_default_batch_size(n_cores):
    return n_cores


def get_default_acceptance_decider(noisy):
    return "noisy" if noisy else "classic"


def get_default_sample_size(model_type, x):
    if model_type == "quadratic":
        out = 2 * len(x) + 1
    else:
        out = len(x) + 1

    return out


def get_default_model_fitter(model_type, sample_size, x):
    n_params = n_free_params(dim=len(x), model_type=model_type)
    if model_type == "linear" or sample_size >= n_params:
        fitter = "ols"
    else:
        fitter = "tranquilo"
    return fitter


def get_default_residualize(model_fitter):
    return model_fitter == "tranquilo"


def get_default_subsolver(bounds, cube_subsolver, sphere_subsolver):
    return cube_subsolver if bounds.has_any else sphere_subsolver


def get_default_search_radius_factor(functype):
    return 4.25 if functype == "scalar" else 5.0


def get_default_model_type(functype):
    return "quadratic" if functype == "scalar" else "linear"


def get_default_aggregator(functype, model_type):
    if functype == "scalar" and model_type == "quadratic":
        aggregator = "identity"
    elif functype == "least_squares" and model_type == "linear":
        aggregator = "least_squares_linear"
    elif functype == "likelihood" and model_type == "linear":
        aggregator = "information_equality_linear"
    else:
        allowed_combinations = {
            "scalar": "quadratic",
            "least_squares": "linear",
            "likelihood": "linear",
        }
        raise NotImplementedError(
            "The requested combination of functype and model_type is not supported. "
            f"Allowed combinations are: {list(allowed_combinations.items())}."
        )

    return aggregator


def get_default_n_evals_at_start(noisy):
    return 5 if noisy else 1


class StopOptions(NamedTuple):
    """Criteria for stopping without successful convergence."""

    max_iter: int
    max_eval: int
    max_time: float


class ConvOptions(NamedTuple):
    """Criteria for successful convergence."""

    disable: bool
    ftol_abs: float
    gtol_abs: float
    xtol_abs: float
    ftol_rel: float
    gtol_rel: float
    xtol_rel: float
    min_radius: float


class RadiusOptions(NamedTuple):
    """Options for trust-region radius management."""

    initial_radius: float
    min_radius: float = 1e-6
    max_radius: float = 1e6
    rho_decrease: float = 0.1
    rho_increase: float = 0.1
    shrinking_factor: float = 0.5
    expansion_factor: float = 2.0
    large_step: float = 0.5
    max_radius_to_step_ratio: float = np.inf


class AcceptanceOptions(NamedTuple):
    confidence_level: float = 0.8
    power_level: float = 0.8
    n_initial: int = 5
    n_min: int = 5
    n_max: int = 100
    min_improvement: float = 0.0


class StagnationOptions(NamedTuple):
    min_relative_step_keep: float = 0.125
    min_relative_step: float = 0.05
    sample_increment: int = 1
    max_trials: int = 1
    drop: bool = True


class SubsolverOptions(NamedTuple):
    maxiter: int = 20
    maxiter_gradient_descent: int = 5
    conjugate_gradient_method: str = "cg"
    gtol_abs: float = 1e-8
    gtol_rel: float = 1e-8
    gtol_scaled: float = 0.0
    gtol_abs_conjugate_gradient: float = 1e-8
    gtol_rel_conjugate_gradient: float = 1e-6
    k_easy: float = 0.1
    k_hard: float = 0.2


class FitterOptions(NamedTuple):
    l2_penalty_linear: float = 0.0
    l2_penalty_square: float = 0.1
    p_intercept: float = 0.05
    p_linear: float = 0.4
    p_square: float = 1.0


class VarianceEstimatorOptions(NamedTuple):
    max_distance_factor: float = 3.0
    min_n_evals: int = 3


class FilterOptions(NamedTuple):
    strictness: float = 1e-10
    shape: str = "sphere"


class SamplerOptions(NamedTuple):
    distribution: str = None
    hardness: float = 1
    algorithm: str = "scipy_lbfgsb"
    algo_options: dict = None
    criterion: str = None
    n_points_randomsearch: int = 1
    return_info: bool = False


def update_option_bundle(default_options, user_options=None):
    """Update default options with user options.

    The user option is converted to the type of the default option if possible.

    Args:
        default_options (NamedTuple): Options that behave like a `typing.NamedTuple`,
            i.e. have _fields as well as _asdict and _replace methods.
        user_options (NamedTuple, Dict or None): User options as a dict or NamedTuple.
            The default options will be updated by the user options.

    """
    if user_options is None:
        return default_options

    # convert user options to dict
    if not isinstance(user_options, dict):
        user_options = user_options._asdict()

    # check that all user options are valid
    invalid_fields = set(user_options) - set(default_options._fields)
    if invalid_fields:
        raise ValueError(
            f"The following user options are not valid: {invalid_fields}. "
            f"Valid options are {default_options._fields}."
        )

    # convert types if possible
    typed = {}
    for k, v in user_options.items():
        target_type = type(getattr(default_options, k))
        if isinstance(v, target_type):
            typed[k] = v
        else:
            typed[k] = target_type(v)

    # update default options
    out = default_options._replace(**typed)

    return out
