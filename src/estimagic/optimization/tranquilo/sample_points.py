import inspect
import warnings
from functools import partial

import estimagic as em
import numpy as np
import scipy as sp
from estimagic.optimization.tranquilo.options import Bounds
from scipy.spatial.distance import pdist


def get_sampler(sampler, bounds, user_options=None):
    """Get sampling function partialled options.

    Args:
        sampler (str or callable): Name of a sampling method or sampling function.
            The arguments of sampling functions need to be: ``lower_bounds``,
            ``upper_bounds``, ``target_size``, ``existing_xs``, ``existing_fvals``.
            Sampling functions need to return a dictionary with the entry "points"
            (and arbitrary additional information). See ``reference_sampler`` for
            details.
        bounds (Bounds): A NamedTuple with attributes ``lower`` and ``upper``
        user_options (dict): Additional keyword arguments for the sampler. Options that
            are not used by the sampler are ignored with a warning.

    Returns:
        callable: Function that depends on trustregion, target_size, existing_xs and
            existing_fvals and returns a new sample.

    """
    user_options = {} if user_options is None else user_options

    built_in_samplers = {
        "naive": _naive_sampler,
        "sphere": _sphere_sampler,
        "optimal_sphere": _optimal_sphere_sampler,
    }

    if isinstance(sampler, str) and sampler in built_in_samplers:
        _sampler = built_in_samplers[sampler]
        _sampler_name = sampler
    elif callable(sampler):
        _sampler = sampler
        _sampler_name = getattr(sampler, "__name__", "your sampler")
    else:
        raise ValueError(
            f"Invalid sampler: {sampler}. Must be one of {list(built_in_samplers)} "
            "or a callable."
        )

    args = set(inspect.signature(_sampler).parameters)

    mandatory_args = {
        "bounds",
        "trustregion",
        "target_size",
        "existing_xs",
        "rng",
    }

    problematic = mandatory_args - args
    if problematic:
        raise ValueError(
            f"The following mandatory arguments are missing in {_sampler_name}: "
            f"{problematic}"
        )

    valid_options = args - mandatory_args

    reduced = {key: val for key, val in user_options.items() if key in valid_options}
    ignored = {
        key: val for key, val in user_options.items() if key not in valid_options
    }

    if ignored:
        warnings.warn(
            "The following options were ignored because they are not compatible "
            f"with {_sampler_name}:\n\n {ignored}"
        )

    out = partial(
        _sampler,
        bounds=bounds,
        **reduced,
    )

    return out


def _naive_sampler(
    trustregion,
    target_size,
    rng,
    existing_xs=None,
    bounds=None,
):
    """Naive random generation of trustregion points.

    This is just a reference implementation to illustrate the interface of trustregion
    samplers.

    All arguments but seed are mandatory, even if not used.

    Samplers should not make unnecessary checks on input compatibility (e.g. that the
    shapes of existing_xs and existing_fvals match). This will be done automatically
    outside of the sampler.

    Args:
        trustregion (TrustRegion): NamedTuple with attributes center and radius.
        target_size (int): Target number of points in the combined sample of existing_xs
            and newly sampled points. The sampler does not have to guarantee that this
            number will actually be reached.
        existing_xs (np.ndarray or None): 2d numpy array in which each row is an
            x vector at which the criterion function has already been evaluated, that
            satisfies lower_bounds <= existing_xs <= upper_bounds.
        rng (numpy.random.Generator): Random number generator.
        bounds (Bounds or None): NamedTuple

    """
    n_points = _get_effective_n_points(target_size, existing_xs)
    n_params = len(trustregion.center)
    region_bounds = _get_effective_bounds(trustregion, bounds)

    points = rng.uniform(
        low=region_bounds.lower,
        high=region_bounds.upper,
        size=(n_points, n_params),
    )

    return points


def _sphere_sampler(
    trustregion,
    target_size,
    rng,
    existing_xs=None,
    bounds=None,
):
    n_points = _get_effective_n_points(target_size, existing_xs)
    n_params = len(trustregion.center)
    raw = rng.normal(size=(n_points, n_params))
    denom = np.linalg.norm(raw, axis=1).reshape(-1, 1)

    points = trustregion.radius * raw / denom + trustregion.center

    if bounds is not None and (bounds.lower is not None or bounds.upper is not None):
        bounds = _get_effective_bounds(trustregion, bounds)
        points = np.clip(
            points,
            a_min=bounds.lower,
            a_max=bounds.upper,
        )

    return points


# ======================================================================================
# Optimal sphere sampler
# ======================================================================================


def _optimal_sphere_sampler(
    trustregion,
    target_size,
    rng,
    existing_xs=None,
    bounds=None,
    algorithm="scipy_lbfgsb",
):
    n_points = _get_effective_n_points(target_size, existing_xs)
    n_params = len(trustregion.center)

    x0 = _sphere_sampler(trustregion, target_size=n_points, rng=rng, bounds=bounds)

    res = em.minimize(
        criterion=_optimal_sphere_criterion,
        params=x0,
        algorithm=algorithm,
        criterion_kwargs={
            "existing_xs": existing_xs,
            "n_points": n_points,
            "n_params": n_params,
        },
        lower_bounds=-np.ones_like(x0),
        upper_bounds=np.ones_like(x0),
    )

    points = _x_from_internal(res.params, n_points, n_params)
    return points


def _optimal_sphere_criterion(x, existing_xs, n_points, n_params):
    x = _x_from_internal(x, n_points, n_params)
    if existing_xs is not None:
        sample = np.row_stack([x, existing_xs])
    else:
        sample = x
    return sp.special.logsumexp(-pdist(sample) ** 2)


def _x_from_internal(x, n_points, n_params):
    x = x.reshape(n_points, n_params)
    denom = np.linalg.norm(x, axis=1).reshape(-1, 1)
    x = x / denom
    return x


# ======================================================================================
# Helper functions
# ======================================================================================


def _get_effective_bounds(trustregion, bounds):
    lower_bounds = trustregion.center - trustregion.radius
    upper_bounds = trustregion.center + trustregion.radius

    if bounds is not None and bounds.lower is not None:
        lower_bounds = np.clip(lower_bounds, bounds.lower, np.inf)

    if bounds is not None and bounds.upper is not None:
        upper_bounds = np.clip(upper_bounds, -np.inf, bounds.upper)

    return Bounds(lower=lower_bounds, upper=upper_bounds)


def _get_effective_n_points(target_size, existing_xs):
    if existing_xs is not None:
        n_points = max(0, target_size - len(existing_xs))
    else:
        n_points = target_size
    return n_points
