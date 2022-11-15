import inspect
import warnings
from functools import partial

import estimagic as em
import numpy as np
from estimagic.optimization.tranquilo.options import Bounds
from scipy.spatial.distance import pdist
from scipy.special import gammainc
from scipy.special import logsumexp


def get_sampler(
    sampler, bounds, model_info=None, radius_factors=None, user_options=None
):
    """Get sampling function partialled options.

    Args:
        sampler (str or callable): Name of a sampling method or sampling function.
            The arguments of sampling functions need to be: ``trustregion``,
            ``n_points``, ``rng``, ``existing_xs`` and ``bounds``.
            Sampling functions need to return a dictionary with the entry "points"
            (and arbitrary additional information). See ``reference_sampler`` for
            details.
        bounds (Bounds): A NamedTuple with attributes ``lower`` and ``upper``
        user_options (dict): Additional keyword arguments for the sampler. Options that
            are not used by the sampler are ignored with a warning. If sampler is
            'hull_sampler' or 'optimal_hull_sampler' the user options must contain the
            argument 'order', which is a positive integer.

    Returns:
        callable: Function that depends on trustregion, n_points, existing_xs and
            existing_fvals, model_info and  and returns a new sample.

    """
    user_options = {} if user_options is None else user_options

    built_in_samplers = {
        "box": _box_sampler,
        "ball": _ball_sampler,
        "hull_sampler": _hull_sampler,
        "optimal_hull_sampler": _optimal_hull_sampler,
        "cube": partial(_hull_sampler, order=np.inf),
        "sphere": partial(_hull_sampler, order=2),
        "optimal_cube": partial(_optimal_hull_sampler, order=np.inf),
        "optimal_sphere": partial(_optimal_hull_sampler, order=2),
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

    if "hull_sampler" in _sampler_name and "order" not in user_options:
        msg = (
            "The hull_sampler and optimal_hull_sampler require the argument 'order' to "
            "be prespecfied in the user_options dictionary. Order is a positive "
            "integer. For order = 2 the hull_sampler equals the sphere_sampler, and "
            "for order = np.inf it equals the cube_sampler."
        )
        raise ValueError(msg)

    args = set(inspect.signature(_sampler).parameters)

    mandatory_args = {
        "bounds",
        "trustregion",
        "n_points",
        "existing_xs",
        "rng",
    }

    optional_kwargs = {
        "model_info": model_info,
        "radius_factors": radius_factors,
    }

    optional_kwargs = {k: v for k, v in optional_kwargs.items() if k in args}

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
        **optional_kwargs,
        **reduced,
    )

    return out


def _box_sampler(
    trustregion,
    n_points,
    rng,
    existing_xs=None,
    bounds=None,
):
    """Naive random generation of trustregion points inside a box.

    This is just a reference implementation to illustrate the interface of trustregion
    samplers. Mathematically it samples uniformaly from inside the cube defined by the
    intersection of the trustregion and the bounds.

    All arguments but seed are mandatory, even if not used.

    Samplers should not make unnecessary checks on input compatibility (e.g. that the
    shapes of existing_xs and existing_fvals match). This will be done automatically
    outside of the sampler.

    Args:
        trustregion (TrustRegion): NamedTuple with attributes center and radius.
        n_points (int): how many new points to sample
        rng (numpy.random.Generator): Random number generator.
        existing_xs (np.ndarray or None): 2d numpy array in which each row is an
            x vector at which the criterion function has already been evaluated, that
            satisfies lower_bounds <= existing_xs <= upper_bounds.
        bounds (Bounds or None): NamedTuple.

    """
    n_params = len(trustregion.center)
    effective_bounds = _get_effective_bounds(trustregion, bounds=bounds)

    points = rng.uniform(
        low=effective_bounds.lower,
        high=effective_bounds.upper,
        size=(n_points, n_params),
    )
    return points


def _ball_sampler(trustregion, n_points, rng, existing_xs=None, bounds=None):
    """Naive random generation of trustregion points inside a ball.

    Mathematically it samples uniformaly from inside the ball defined by the
    intersection of the trustregion and the bounds.

    Code is adapted from https://tinyurl.com/y3p2dz6b.

    Args:
        trustregion (TrustRegion): NamedTuple with attributes center and radius.
        n_points (int): how many new points to sample
        rng (numpy.random.Generator): Random number generator.
        existing_xs (np.ndarray or None): 2d numpy array in which each row is an
            x vector at which the criterion function has already been evaluated, that
            satisfies lower_bounds <= existing_xs <= upper_bounds.
        bounds (Bounds or None): NamedTuple.

    """
    n_params = len(trustregion.center)
    effective_bounds = _get_effective_bounds(trustregion, bounds=bounds)

    raw = rng.normal(size=(n_points, n_params))
    norm = np.linalg.norm(raw, axis=1, ord=2)
    scale = gammainc(n_params / 2, norm**2 / 2) ** (1 / n_params) / norm
    points = raw * scale.reshape(-1, 1)

    out = _map_into_feasible_trustregion(points, bounds=effective_bounds)
    return out


def _hull_sampler(
    trustregion,
    n_points,
    rng,
    order,
    distribution=None,
    existing_xs=None,
    bounds=None,
):
    """Random generation of trustregion points on the hull of general sphere / cube.

    Points are sampled randomly on a hull (of a sphere for order=2 and of a cube for
    order=np.inf). These points are then mapped into the feasible region, which is
    defined by the intersection of the trustregion and the bounds.

    Args:
        trustregion (TrustRegion): NamedTuple with attributes center and radius.
        n_points (int): how many new points to sample
        rng (numpy.random.Generator): Random number generator.
        order (int): Type of norm to use when scaling the sampled points. For 2 it will
            result in sphere sampling, for np.inf in cube sampling.
        distribution (str): Distribution to use for initial sample before points are
            projected onto unit hull. Must be in {'normal', 'uniform'}.
        existing_xs (np.ndarray or None): 2d numpy array in which each row is an
            x vector at which the criterion function has already been evaluated, that
            satisfies lower_bounds <= existing_xs <= upper_bounds.
        bounds (Bounds or None): NamedTuple.

    """
    n_params = len(trustregion.center)
    effective_bounds = _get_effective_bounds(trustregion, bounds=bounds)

    if distribution is None:
        distribution = "normal" if order <= 3 else "uniform"
    points = _draw_from_distribution(distribution, rng=rng, size=(n_points, n_params))
    points = _project_onto_unit_hull(points, order=order)
    points = _map_into_feasible_trustregion(points, bounds=effective_bounds)
    return points


def _optimal_hull_sampler(
    trustregion,
    n_points,
    rng,
    model_info,
    radius_factors,
    order,
    distribution=None,
    hardness=1,
    existing_xs=None,
    bounds=None,
    algorithm="scipy_lbfgsb",
    algo_options=None,
    criterion=None,
):
    """Optimal generation of trustregion points on the hull of general sphere / cube.

    Points are sampled optimally on a hull (of a sphere for order=2 and of a cube for
    order=np.inf), where the criterion that is maximized is the minimum distance of all
    pairs of points, except for pairs of existing points. These points are then mapped
    into the feasible region, which is defined by the intersection of the trustregion
    and the bounds. Instead of using a hard minimum we return the soft minimum, whose
    accuracy we govern by the hardness factor. For more information on the soft-minimum,
    seek: https://tinyurl.com/mrythbk4.

    Args:
        trustregion (TrustRegion): NamedTuple with attributes center and radius.
        n_points (int): how many new points to sample
        rng (numpy.random.Generator): Random number generator.
        order (int): Type of norm to use when scaling the sampled points. For 2 it will
            result in sphere sampling, for np.inf in cube sampling.
        distribution (str): Distribution to use for initial sample before points are
            projected onto unit hull. Must be in {'normal', 'uniform'}.
        hardness (float): Positive scaling factor. As hardness tends to infinity the
            soft minimum (logsumexp) approaches the hard minimum. Default is 1. A
            detailed explanation is given in the docstring.
        existing_xs (np.ndarray or None): 2d numpy array in which each row is an
            x vector at which the criterion function has already been evaluated, that
            satisfies lower_bounds <= existing_xs <= upper_bounds.
        bounds (Bounds or None): NamedTuple.
        algorithm (str): Optimization algorithm.
        algo_options (dict): Algorithm specific configuration of the optimization. See
            :ref:`list_of_algorithms` for supported options of each algorithm. Default
            sets ``stopping_max_iterations=n_params``.
        criterion (str or None): "distance", "determinant" or None.
            - "distance": maximize the minimal distance between points, excluding
              distances between existing points. This is a fast and relatively simple
              optimization problem and yields the same points as "determinant" in
              many circumstances.
            - "determinant": maximize the determinant of the x'x where x is the matrix
              of points. This is known as d-optimality in the optimal design literature
              and as fekete points in the function approximation literature. This
              criterion has the best theoretical properties but is very hard to
              optimize. Thus the practical performance can be bad.
            - None: Use the "determinant" criterion if only one point is added and the
              "distance" criterion if multiple points are added.

    Returns:
        np.ndarray: Generated points. Has shape (n_points, len(trustregion.center)).

    """
    n_params = len(trustregion.center)

    if n_points <= 0:
        return np.array([])

    if criterion is None:
        if n_points == 1:
            criterion = "determinant"
        else:
            criterion = "distance"

    algo_options = {} if algo_options is None else algo_options
    if "stopping_max_iterations" not in algo_options:
        algo_options["stopping_max_iterations"] = 2 * n_params + 1

    effective_bounds = _get_effective_bounds(trustregion, bounds=bounds)

    if existing_xs is not None:
        # map existing points into unit space for easier optimization
        existing_xs_unit = _map_from_feasible_trustregion(existing_xs, effective_bounds)

        if criterion == "distance":
            dist_to_center = np.linalg.norm(existing_xs_unit, axis=1)
            not_centric = dist_to_center >= radius_factors.centric
            if not_centric.any():
                existing_xs_unit = existing_xs_unit[not_centric]
            else:
                existing_xs_unit = None

    else:
        existing_xs_unit = None

    # start params
    if distribution is None:
        distribution = "normal" if order <= 3 else "uniform"
    x0 = _draw_from_distribution(distribution, rng=rng, size=(n_points, n_params))
    x0 = _project_onto_unit_hull(x0, order=order)
    x0 = x0.flatten()  # flatten so that em.maximize uses fast path

    # This would raise an error because there are zero pairs to calculate the
    # pairwise distance
    if existing_xs_unit is None and n_points == 1:
        opt_params = x0
    else:

        func_dict = {
            "determinant": _determinant_on_hull,
            "distance": _minimal_pairwise_distance_on_hull,
        }

        criterion_kwargs = {
            "existing_xs": existing_xs_unit,
            "order": order,
            "n_params": n_params,
        }

        if criterion == "distance":
            criterion_kwargs["hardness"] = hardness

        res = em.maximize(
            criterion=func_dict[criterion],
            params=x0,
            algorithm=algorithm,
            criterion_kwargs=criterion_kwargs,
            lower_bounds=-np.ones_like(x0),
            upper_bounds=np.ones_like(x0),
            algo_options=algo_options,
        )

        opt_params = res.params

    points = _project_onto_unit_hull(opt_params.reshape(-1, n_params), order=order)
    points = _map_into_feasible_trustregion(points, bounds=effective_bounds)
    return points


# ======================================================================================
# Helper functions
# ======================================================================================


def _minimal_pairwise_distance_on_hull(x, existing_xs, order, hardness, n_params):
    """Compute minimal pairwise distance of new and existing points.

    Instead of optimizing the distance of points in the feasible trustregion, this
    criterion function leads to the maximization of the minimum distance of the points
    in the unit space. These can then be mapped into the feasible trustregion. We do not
    consider the distances between existing points. Instead of using a hard minimum we
    return the soft minimum, whose accuracy we govern by the hardness factor. For more
    information on the soft-minimum, seek: https://tinyurl.com/mrythbk4.

    Args:
        x (np.ndarray): Flattened 1d array of internal points. Each value is in [-1, 1].
        existing_xs (np.ndarray or None): 2d numpy array in which each row is an
            x vector at which the criterion function has already been evaluated, that
            satisfies -1 <= existing_xs <= 1.
        order (int): Type of norm to use when scaling the sampled points. For 2 we
            project onto the hull of a sphere, for np.inf onto the hull of a cube.
        hardness (float): Positive scaling factor. As hardness tends to infinity the
            soft minimum (logsumexp) approaches the hard minimum. Default is 1. A
            detailed explanation is given in the docstring.
        n_params (int): Dimensionality of the problem.

    Returns:
        float: The criterion value.

    """
    x = x.reshape(-1, n_params)
    x = _project_onto_unit_hull(x, order=order)

    if existing_xs is not None:
        sample = np.row_stack([x, existing_xs])
        n_existing_pairs = len(existing_xs) * (len(existing_xs) - 1) // 2
        slc = slice(0, -n_existing_pairs) if n_existing_pairs else slice(None)
    else:
        sample = x
        slc = slice(None)

    dist = pdist(sample) ** 2

    # drop distances between existing points. They could introduce flat spots.
    dist = dist[slc]

    # soft minimum
    crit_value = -logsumexp(-hardness * dist)
    return crit_value


def _determinant_on_hull(x, existing_xs, order, n_params):
    """Compute d-optimality criterion of new and existing points.

    Instead of optimizing the distance of points in the feasible trustregion, this
    criterion function leads to the maximization of the minimum distance of the points
    in the unit space.

    Args:
        x (np.ndarray): Flattened 1d array of internal points. Each value is in [-1, 1].
        existing_xs (np.ndarray or None): 2d numpy array in which each row is an
            x vector at which the criterion function has already been evaluated, that
            satisfies -1 <= existing_xs <= 1.
        order (int): Type of norm to use when scaling the sampled points. For 2 we
            project onto the hull of a sphere, for np.inf onto the hull of a cube.
        n_params (int): Dimensionality of the problem.

    Returns:
        float: The criterion value.

    """
    x = x.reshape(-1, n_params)
    n_samples = len(x)

    x = _project_onto_unit_hull(x, order=order)

    if existing_xs is not None:
        sample = np.row_stack([x, existing_xs])
    else:
        sample = x

    crit_value = np.linalg.det(sample.T @ sample / n_samples)

    return crit_value


def _draw_from_distribution(distribution, rng, size):
    """Draw points from distribution.

    Args:
        distribution (str): Distribution to use for initial sample before points are
            projected onto unit hull. Must be in {'normal', 'uniform'}.
        rng (np.random.Generator): Random number generator.
        size (Union[int, tuple[int]]): Output shape.

    Returns:
        np.ndarray: Randomly drawn points.

    """
    if distribution == "normal":
        draw = rng.normal(size=size)
    elif distribution == "uniform":
        draw = rng.uniform(-1, 1, size=size)
    else:
        raise ValueError(
            f"distribution is {distribution}, but needs to be in ('normal', 'uniform')."
        )
    return draw


def _map_into_feasible_trustregion(points, bounds):
    """Map points from the unit space into trustregion defined by bounds.

    Args:
        points (np.ndarray): 2d array of points to be mapped. Each value is in [-1, 1].
        bounds (Bounds): A NamedTuple with attributes ``lower`` and ``upper``, where
            lower and upper define the rectangle that is the feasible trustregion.

    Returns:
        np.ndarray: Points in trustregion.

    """
    out = (bounds.upper - bounds.lower) * (points + 1) / 2 + bounds.lower
    return out


def _map_from_feasible_trustregion(points, bounds):
    """Map points from a feasible trustregion definde by bounds into unit space.

    Args:
        points (np.ndarray): 2d array of points to be mapped. Each value is in [-1, 1].
        bounds (Bounds): A NamedTuple with attributes ``lower`` and ``upper``, where
            lower and upper define the rectangle that is the feasible trustregion.

    Returns:
        np.ndarray: Points in unit space.

    """
    out = 2 * (points - bounds.lower) / (bounds.upper - bounds.lower) - 1
    return out


def _project_onto_unit_hull(x, order):
    """Project points from the unit space onto the hull of a geometric figure.

    Args:
        x (np.ndarray): 2d array of points to be projects. Each value is in [-1, 1].
        order (int): Type of norm to use when scaling the sampled points. For 2 we
            project onto the hull of a sphere, for np.inf onto the hull of a cube.

    Returns:
        np.ndarray: The projected points.

    """
    norm = np.linalg.norm(x, axis=1, ord=order).reshape(-1, 1)
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
