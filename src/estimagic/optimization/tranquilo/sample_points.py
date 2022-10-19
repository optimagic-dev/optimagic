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
        "cube": partial(_hull_sampler, ord=np.inf),
        "sphere": partial(_hull_sampler, ord=2),
        "optimal_cube": partial(_optimal_hull_sampler, ord=np.inf),
        "optimal_sphere": partial(_optimal_hull_sampler, ord=2),
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
    samplers. Mathematically it samples uniformaly from inside the cube defined by the
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
    bounds = _get_effective_bounds(trustregion, bounds)

    points = rng.uniform(
        low=bounds.lower,
        high=bounds.upper,
        size=(n_points, n_params),
    )
    return points


def _hull_sampler(
    trustregion,
    target_size,
    rng,
    ord,  # noqa: A002
    existing_xs=None,
    bounds=None,
):
    """Random generation of trustregion points on the hull of general sphere / cube.

    Args:
        trustregion (TrustRegion): NamedTuple with attributes center and radius.
        target_size (int): Target number of points in the combined sample of existing_xs
            and newly sampled points. The sampler does not have to guarantee that this
            number will actually be reached.
        rng (numpy.random.Generator): Random number generator.
        ord (int): Type of norm to use when scaling the sampled points. For 2 it will
            result in sphere sampling, for np.inf in cube sampling.
        existing_xs (np.ndarray or None): 2d numpy array in which each row is an
            x vector at which the criterion function has already been evaluated, that
            satisfies lower_bounds <= existing_xs <= upper_bounds.
        bounds (Bounds or None): NamedTuple.

    """
    n_points = _get_effective_n_points(target_size, existing_xs)
    n_params = len(trustregion.center)
    bounds = _get_effective_bounds(trustregion, bounds)

    points = rng.normal(size=(n_points, n_params))
    points = _internal_to_external(
        points, bounds=bounds, trustregion=trustregion, ord=ord
    )

    return points


def _optimal_hull_sampler(
    trustregion,
    target_size,
    rng,
    ord,  # noqa: A002
    existing_xs=None,
    bounds=None,
    algorithm="scipy_lbfgsb",
    multistart=False,
):
    """Optimal generation of points on a hull.

    Args:
        trustregion (TrustRegion): NamedTuple with attributes center and radius.
        target_size (int): Target number of points in the combined sample of existing_xs
            and newly sampled points. The sampler does not have to guarantee that this
            number will actually be reached.
        rng (numpy.random.Generator): Random number generator.
        ord (int): Type of norm to use when scaling the sampled points. For 2 it will
            result in sphere sampling, for np.inf in cube sampling.
        existing_xs (np.ndarray or None): 2d numpy array in which each row is an
            x vector at which the criterion function has already been evaluated, that
            satisfies lower_bounds <= existing_xs <= upper_bounds.
        bounds (Bounds or None): NamedTuple.
        algorithm (str): Optimization algorithm.
        multistart (bool): Whether to use multistart in the optimization.

    Returns:
        np.ndarray: Generated points. Has shape (target_size, len(trustregion.center)).

    """
    n_points = _get_effective_n_points(target_size, existing_xs)
    bounds = _get_effective_bounds(trustregion, bounds)

    # start params
    x0 = _hull_sampler(
        trustregion, target_size=n_points, rng=rng, ord=ord, bounds=bounds
    )
    x0 = (x0 - trustregion.center) / trustregion.radius

    res = em.maximize(
        criterion=_pairwise_distance_crit,
        params=x0,
        algorithm=algorithm,
        criterion_kwargs={
            "existing_xs": existing_xs,
            "trustregion": trustregion,
            "bounds": bounds,
            "ord": ord,
        },
        lower_bounds=-np.ones_like(x0),
        upper_bounds=np.ones_like(x0),
        multistart=multistart,
    )

    points = _internal_to_external(res.params, bounds, trustregion, ord=ord)
    return points


# ======================================================================================
# Criterion functions
# ======================================================================================


def _internal_to_external(x, bounds, trustregion, ord):  # noqa: A002
    points = _project_onto_unit_hull(x, ord=ord)
    points = trustregion.radius * points + trustregion.center
    points = _clip_on_bounds(points, bounds)
    return points


def _pairwise_distance_crit(x, existing_xs, trustregion, bounds, ord):  # noqa: A002
    x = _internal_to_external(x, bounds, trustregion, ord=ord)

    if existing_xs is not None:
        sample = np.row_stack([x, existing_xs])
    else:
        sample = x

    dist = (pdist(sample) ** 2).min()
    return dist


# ======================================================================================
# Helper functions
# ======================================================================================


def _clip_on_bounds(sample, bounds):
    new_sample = np.clip(sample, a_min=bounds.lower, a_max=bounds.upper)
    return new_sample


def _project_onto_unit_hull(x, ord):  # noqa: A002
    norm = np.linalg.norm(x, axis=1, ord=ord).reshape(-1, 1)
    projected = x / norm
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
