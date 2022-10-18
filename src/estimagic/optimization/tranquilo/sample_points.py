import inspect
import warnings
from functools import partial

import estimagic as em
import numpy as np
from estimagic.optimization.tranquilo.options import Bounds
from numba import njit
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
        "box": _box_sampler,
        "optimal_box": _optimal_box_sampler,
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
    samplers. Mathematically it samples uniformaly from inside the box defined by the
    intersection of the trustregion and the bounds.

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
        bounds (Bounds or None): NamedTuple.

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


def _box_sampler(
    trustregion,
    target_size,
    rng,
    existing_xs=None,
    bounds=None,
):
    """Random generation of trustregion points on the hull of a box.

    Mathematically it samples randomly from the convex hull of the box defined by the
    intersection of the trustregion and the bounds.

    Args:
        trustregion (TrustRegion): NamedTuple with attributes center and radius.
        target_size (int): Target number of points in the combined sample of existing_xs
            and newly sampled points. The sampler does not have to guarantee that this
            number will actually be reached.
        existing_xs (np.ndarray or None): 2d numpy array in which each row is an
            x vector at which the criterion function has already been evaluated, that
            satisfies lower_bounds <= existing_xs <= upper_bounds.
        rng (numpy.random.Generator): Random number generator.
        bounds (Bounds or None): NamedTuple.

    """
    n_points = _get_effective_n_points(target_size, existing_xs)
    n_params = len(trustregion.center)
    bounds = _get_effective_bounds(trustregion, bounds)

    points = rng.uniform(
        low=np.zeros(n_params),
        high=np.ones(n_params),
        size=(n_points, n_params),
    )

    points = _project_points_onto_unit_box(points)
    points = (bounds.upper - bounds.lower) * points + bounds.lower
    return points


def _optimal_box_sampler(
    trustregion,
    target_size,
    rng,
    existing_xs=None,
    bounds=None,
    algorithm="scipy_lbfgsb",
    multistart=False,
):
    n_points = _get_effective_n_points(target_size, existing_xs)
    n_params = len(trustregion.center)
    bounds = _get_effective_bounds(trustregion, bounds)

    # start params
    x0 = rng.uniform(
        low=np.zeros(n_params),
        high=np.ones(n_params),
        size=(n_points, n_params),
    )

    res = em.maximize(
        criterion=_optimal_box_criterion,
        params=x0,
        algorithm=algorithm,
        criterion_kwargs={"existing_xs": existing_xs, "bounds": bounds},
        lower_bounds=np.zeros_like(x0),
        upper_bounds=np.ones_like(x0),
        multistart=multistart,
    )

    points = _project_points_onto_unit_box(res.params)
    points = (bounds.upper - bounds.lower) * points + bounds.lower
    return points


def _sphere_sampler(
    trustregion,
    target_size,
    rng,
    existing_xs=None,
    bounds=None,
):
    """Random generation of points on a sphere.

    Mathematically it samples uniformly from the sphere defined by the
    trustregion and then projects these points onto the binding bounds.

    Args:
        trustregion (TrustRegion): NamedTuple with attributes center and radius.
        target_size (int): Target number of points in the combined sample of existing_xs
            and newly sampled points. The sampler does not have to guarantee that this
            number will actually be reached.
        existing_xs (np.ndarray or None): 2d numpy array in which each row is an
            x vector at which the criterion function has already been evaluated, that
            satisfies lower_bounds <= existing_xs <= upper_bounds.
        rng (numpy.random.Generator): Random number generator.
        bounds (Bounds or None): NamedTuple.

    Returns:
        np.ndarray: Generated points. Has shape (target_size, len(trustregion.center)).

    """
    n_points = _get_effective_n_points(target_size, existing_xs)
    n_params = len(trustregion.center)

    points = rng.normal(size=(n_points, n_params))
    points = _project_onto_unit_sphere(points)
    points = trustregion.radius * points + trustregion.center

    if bounds is not None and (bounds.lower is not None or bounds.upper is not None):
        bounds = _get_effective_bounds(trustregion, bounds)
        points = _project_onto_bounds(points, bounds)

    return points


def _optimal_sphere_sampler(
    trustregion,
    target_size,
    rng,
    existing_xs=None,
    bounds=None,
    algorithm="scipy_lbfgsb",
    multistart=False,
):
    """Optimal generation of points on a sphere.

    Mathematically the points are chosen to maximize the minimal pairwise distance
    between any two sample points under the constraint that points lie on convex
    hull of the intersection of the sphere defined by the trustregion and the bounds.

    Args:
        trustregion (TrustRegion): NamedTuple with attributes center and radius.
        target_size (int): Target number of points in the combined sample of existing_xs
            and newly sampled points. The sampler does not have to guarantee that this
            number will actually be reached.
        existing_xs (np.ndarray or None): 2d numpy array in which each row is an
            x vector at which the criterion function has already been evaluated, that
            satisfies lower_bounds <= existing_xs <= upper_bounds.
        rng (numpy.random.Generator): Random number generator.
        bounds (Bounds or None): NamedTuple.

    Returns:
        np.ndarray: Generated points. Has shape (target_size, len(trustregion.center)).

    """
    n_points = _get_effective_n_points(target_size, existing_xs)

    bounds = _rescale_bounds(bounds, trustregion=trustregion)

    # start params
    x0 = _sphere_sampler(trustregion, target_size=n_points, rng=rng, bounds=bounds)

    res = em.maximize(
        criterion=_optimal_sphere_criterion,
        params=x0,
        algorithm=algorithm,
        criterion_kwargs={"existing_xs": existing_xs, "bounds": bounds},
        lower_bounds=-np.ones_like(x0),
        upper_bounds=np.ones_like(x0),
        multistart=multistart,
    )

    points = _project_onto_unit_sphere(res.params)
    points = _project_onto_bounds(points, bounds)
    points = trustregion.radius * points + trustregion.center

    return points


# ======================================================================================
# Criteria
# ======================================================================================


def _optimal_box_criterion(x, existing_xs, bounds):
    x = _project_points_onto_unit_box(x)
    x = (bounds.upper - bounds.lower) * x + bounds.lower

    if existing_xs is not None:
        sample = np.row_stack([x, existing_xs])
    else:
        sample = x

    dist = (pdist(sample) ** 2).min()
    return dist


def _optimal_sphere_criterion(x, existing_xs, bounds):
    x = _project_onto_unit_sphere(x)
    x = _project_onto_bounds(x, bounds)

    if existing_xs is not None:
        sample = np.row_stack([x, existing_xs])
    else:
        sample = x

    dist = (pdist(sample) ** 2).min()
    return dist


# ======================================================================================
# Helper functions
# ======================================================================================


def _project_onto_bounds(sample, bounds):
    new_sample = np.clip(sample, a_min=bounds.lower, a_max=bounds.upper)
    return new_sample


def _project_onto_unit_sphere(x):
    denom = np.linalg.norm(x, axis=1).reshape(-1, 1)
    projected = x / denom
    return projected


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


def _rescale_bounds(bounds, trustregion):
    if bounds is not None and bounds.lower is not None:
        lower = bounds.lower / trustregion.radius - trustregion.center
    else:
        lower = -1.0

    if bounds is not None and bounds.upper is not None:
        upper = bounds.upper / trustregion.radius - trustregion.center
    else:
        upper = 1.0
    bounds = Bounds(lower=lower, upper=upper)
    return bounds


# ======================================================================================
# Numba implementation of logsumexp and its derivative
# ======================================================================================


@njit
def logsumexp_and_softmax(x):
    _exp = np.exp(x)
    _sum_exp = np.sum(_exp)
    _logsumexp = np.log(_sum_exp)
    _softmax = _exp / _sum_exp
    return _logsumexp, _softmax


@njit
def logsumexp_and_derivative(x):

    n_points, dim = x.shape
    dim_out = n_points * (n_points - 1) // 2

    dists = np.zeros(dim_out)
    dists_jac = np.zeros((dim_out, n_points, dim))  # jac of pairwise-distances
    counter = 0

    for i in range(n_points):
        for j in range(i + 1, n_points):

            _dist = 0.0
            for k in range(dim):
                _diff = x[i, k] - x[j, k]
                _dist += _diff**2
                dists_jac[counter, i, k] = 2 * _diff
                dists_jac[counter, j, k] = -2 * _diff

            dists[counter] = _dist
            counter += 1

    func_val, smax = logsumexp_and_softmax(-dists)
    jac_reshaped = dists_jac.reshape(dim_out, -1).T
    der = -(jac_reshaped @ smax).reshape((n_points, dim))

    return func_val, der


@njit
def _project_points_onto_unit_box(points):
    points = points.copy()
    n_points = len(points)
    for i in range(n_points):
        point = points[i]
        _upper_cost = 1 - point.max()
        _lower_cost = point.min()
        if _upper_cost < _lower_cost:
            points[i, point.argmax()] = 1
        else:
            points[i, point.argmin()] = 0
    return points
