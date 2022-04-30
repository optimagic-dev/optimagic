import itertools

import numpy as np
from numba import njit


def lhs_sampler(
    lower_bounds,
    upper_bounds,
    target_size,
    existing_xs=None,
    existing_fvals=None,  # noqa: U100
    centered=False,
    criterion="maximin",
    target="linear",
    n_iter=1_000,
    seed=1234,
    return_crit_vals=False,
):
    """Latin-Hypercube generation of trustregion sampling.

    Args:
        lower_bounds (np.ndarray): 1d array with lower bounds for the sampled points.
        upper_bounds (pn.ndarray): 1d sample with upper bounds for the sampled points.
        target_size (int): Target number of points in the combined sample of existing_xs
            and newly sampled points. The sampler does not have to guarantee that this
            number will actually be reached.
        existing_xs (np.ndarray or None): 2d numpy array in which each row is an
            x vector at which the criterion function has already been evaluated, that
            satisfies lower_bounds <= existing_xs <= upper_bounds.
        existing_fvals (np.ndarray): 1d numpy array with same length as existing_xs
            that contains the corresponding function evaluations.
        centered (bool): Center the point within the multi-dimensional grid. Default is
            False. If True, used random permutations of coordinates to lower the
            centered discrepancy.
        criterion (str): Must be in {"a-optimal", "d-optimal", "e-optimal", "g-optimal",
            "maximin"}. Default "maximin".
        target (str): One of "linear" or "quadratic". Determines in which space the
            criterion is evaluated. For "quadratic" the Latin-Hypercube sample X is
            mapped onto Y(X) = [X, X ** 2] and the criterion value for X is computed
            using Y(X). This can be useful if one chooses an optimality criterion that
            minimizes e.g. the variance of the least-squares estimator, while using a
            quadratic or polynomial model. Default is "linear".
        n_iter (int): Iterations considered in random search.
        seed (int): Seed for a random number generator.
        return_crit_vals (bool): Return values of the criterion. Default False. Mainly
            useful for testing.

    Returns:
        dict: A dictionary containing "points" (a numpy array where each row is a
            newly sampled point) and potentially other information about the sampling.

    """
    np.random.seed(seed)

    if existing_xs is not None:
        n_points = max(0, target_size - len(existing_xs))
    else:
        n_points = target_size

    n_params = len(lower_bounds)
    dtype = np.min_scalar_type(target_size)

    # create sample on grid [0, 1, ..., target_size] ** n_params
    if existing_xs is None:
        candidates = _create_upscaled_sample(n_params, n_points, n_iter, seed, dtype)
    else:
        existing_upscaled = _scale_up_points(
            existing_xs, target_size, lower_bounds, upper_bounds
        )
        empty_bins = _get_empty_bin_info(existing_upscaled, target_size, dtype)
        new_points = _extend_upscaled_sample(
            empty_bins, target_size, n_iter, seed, dtype
        )
        existing_upscaled = np.tile(existing_upscaled, (n_iter, 1, 1))
        candidates = np.concatenate((existing_upscaled, new_points), axis=1)

    if n_points > 0:
        # perturb sample
        candidates = candidates.astype(np.float64)
        perturbation = 0.5 if centered else np.random.uniform(size=(n_points, n_params))
        candidates[:, -n_points:, :] += perturbation

        # map sample into cube [lower_bounds, upper_bounds]
        candidates = _scale_down_points(
            candidates, target_size, lower_bounds, upper_bounds
        )

        crit_vals = calculate_criterion(candidates, criterion, target)
        argmin_id = np.argmin(crit_vals)

        # we return only newly sampled points
        points = candidates[argmin_id, -n_points:, :]
    else:
        points = np.array([]).reshape(0, n_params)
        crit_vals = None

    out = {"points": points}
    if return_crit_vals:
        out["crit_vals"] = crit_vals
    return out


@njit
def _create_upscaled_sample(n_params, n_points, n_designs, seed, dtype):
    """Create an upscaled Latin-Hypercube sample.

    Args:
        n_params (int): Number of parameters (dimensions).
        n_points (int): Number of points to sample.
        n_designs (int): Number of different hypercubes to sample.
        seed (int): Seed for a random number generator. Default 0.
        dtype (type): Data type of arrays. Default np.unint8.

    Returns:
        sample (np.ndarray): Latin-Hypercube sample of shape (n_designs, n_samples,
            n_dim)

    """
    np.random.seed(seed)

    index = np.arange(n_points, dtype=dtype)
    sample = np.empty((n_designs, n_params, n_points), dtype=dtype)

    for i in range(n_designs):
        for j in range(n_params):
            np.random.shuffle(index)
            sample[i, j] = index

    sample = np.transpose(sample, (0, 2, 1))
    return sample


@njit
def _scale_up_points(points, n_points, lower_bounds, upper_bounds):
    """Scale Latin-Hypercube sample up.

    Latin hypercubes are sampled using index arrays, which distributes the array entries
    over range(n_points). To get a sample from a specific region, e.g. [0, 1] ** 2, the
    samples have to be scaled down. This function performs the inverse operation. For
    the down-scaling operation see :func:`_scale_down_points`.

    Args:
        points (np.ndarray): Latin-Hypercube sample or subset thereof.
        n_points (int): Number of points of the originally sampled Latin-Hypercube.
        lower_bounds (np.ndarray): 1d array with lower bounds for the sampled points.
        upper_bounds (pn.ndarray): 1d sample with upper bounds for the sampled points.

    Returns:
        scaled (np.ndarray): The scaled-up version of points.

    """
    scaled = ((points - lower_bounds) * n_points) / (upper_bounds - lower_bounds)
    return scaled


@njit
def _scale_down_points(points, n_points, lower_bounds, upper_bounds):
    """Scale Latin-Hypercube sample down.

    Latin hypercubes are sampled using index arrays, which distributes the array entries
    over range(n_points). To get a sample from a specific region, e.g. [0, 1] ** 2, the
    samples have to be scaled down. For the inverse operation see function
    :func:`_scale_up_points`.

    Args:
        points (np.ndarray): Latin-Hypercube sample or subset thereof.
        n_points (int): Number of points of the originally sampled Latin-Hypercube.
        lower_bounds (np.ndarray): 1d array with lower bounds for the sampled points.
        upper_bounds (pn.ndarray): 1d sample with upper bounds for the sampled points.

    Returns:
        scaled (np.ndarray): The scaled-down version of points.

    """
    scaled = lower_bounds + (points / n_points) * (upper_bounds - lower_bounds)
    return scaled


@njit
def _get_empty_bin_info(existing_upscaled, n_points, dtype):
    """Find empty bins in space populated by existing points.

    Args:
        existing_upscaled (np.ndarray): Upscaled points, with integer dtype.
        n_points (int): Number of points originally sampled.
        dtype (type): Data type of arrays.

    Returns:
        out (np.ndarray): Empty bins for each dimension. Has shape
            (len(existing_upscaled) - n_points, n_dim). Non-empty bins are marked by -1,
            and occur since not all dimensions must have the same number of empty bins.

    """
    existing_upscaled = existing_upscaled.astype(dtype)
    n_dim = existing_upscaled.shape[1]
    all_bins = set(np.arange(n_points, dtype=dtype))

    empty_bins = []
    for j in range(n_dim):
        filled_bins = set(existing_upscaled[:, j])
        empty_bins.append(sorted(all_bins - filled_bins))

    max_empty = max(map(len, empty_bins))

    out = np.full((max_empty, n_dim), -1)
    for j, empty in enumerate(empty_bins):
        out[: len(empty), j] = empty

    return out


@njit
def _extend_upscaled_sample(empty_bins, n_points, n_designs, seed, dtype):
    """Extend a subset of sample to a full Latin-Hypercube sample.

    Args:
        empty_bins (np.ndarray): Dimensionality of the problem.
        n_points (int): Number of (total) sample points.
        n_designs (int): Number of different Latin-Hypercubes to sample.
        seed (int): Seed for a random number generator.
        dtype (type): Data type of arrays.

    Returns:
        sample (np.ndarray): Latin Hypercube sample of shape (n_designs, n_samples,
            n_dim)

    """
    np.random.seed(seed)

    mask = empty_bins == -1
    n_new_points, n_dim = empty_bins.shape

    sample = np.empty((n_designs, n_dim, n_new_points), dtype=dtype)

    for j in range(n_dim):
        empty = empty_bins[:, j].copy()
        n_duplicates = mask[:, j].sum()
        empty[mask[:, j]] = np.random.choice(n_points, size=n_duplicates, replace=False)
        for k in range(n_designs):
            np.random.shuffle(empty)
            sample[k, j] = empty

    sample = np.transpose(sample, (0, 2, 1))
    return sample


def calculate_criterion(x, criterion, target):
    """Compute optimality criterion of data matrix along axis.

    Implements criteria that measure the dispersion of sample points in a space.
    Optimization with respect to the criteria leads to certain properties of, for
    example, the estimated parametes in a least squares regression. See
    https://en.wikipedia.org/wiki/Optimal_design for more information on the specific
    criteria.

    Brief explaination:
    - a: minimizes the average variance of the least squares estimator
    - d: maximizes the determinant information matrix of the least squares estimator
    - e: maximizes the minimum eigenvalue of the information matrix
    - g: minimizes the maximum variance of the predicted value in the least squares case
    - maximin: maximizes the minimum (l-infinity) distance between any two points

    Args:
        x (np.ndarray): Data matrix. If 3d, computes the criterion value along the first
            dimension.
        criterion (str): Criterion type, must be in {'a-optimal', 'd-optimal',
            'e-optimal', 'g-optimal', 'maximin'}.
        target (str): One of "linear", "quadratic" or "polynomial". Determines in which
            space the criterion is evaluated. For "quadratic" the Latin-Hypercube sample
            X is mapped onto Y(X) = [X, X ** 2] and the criterion value for X is
            computed using Y(X). For "polynomial" Y(X) will also include all cross
            terms. This can be useful if one chooses an optimality criterion that
            minimizes e.g. the variance of the least-squares estimator, while using a
            quadratic or polynomial model.

    Returns:
        crit_vals (np.ndarray): Criterion values.

    """
    implemented_criteria = {
        "a-optimal",
        "d-optimal",
        "e-optimal",
        "g-optimal",
        "maximin",
    }
    if criterion not in implemented_criteria:
        raise ValueError(f"Invalid criterion. Must be in {implemented_criteria}.")

    implemented_targets = {"linear", "quadratic"}
    if target not in implemented_targets:
        raise ValueError(f"Invalid target. Must be in {implemented_targets}.")

    # pre-processing
    if x.ndim == 2:
        x = np.expand_dims(x, axis=0)

    if target == "quadratic":
        x = np.concatenate((x, x**2), axis=2)

    # common computations
    prod = np.matmul(x.transpose(0, 2, 1), x)

    if criterion in {"a-optimal", "g-optimal"}:
        if x.shape[1] < x.shape[2]:
            inv = np.linalg.inv(prod + 0.01 * np.eye(x.shape[2]))
        else:
            is_invertible = np.linalg.cond(prod) < 1 / np.finfo(float).eps
            inv = np.linalg.inv(prod[is_invertible])
            crit_vals = np.tile(np.inf, x.shape[0])

    # compute criteria
    if criterion == "a-optimal":
        crit_vals[is_invertible] = inv.trace(axis1=1, axis2=2)
    elif criterion == "g-optimal":
        hat_mat = np.matmul(
            np.matmul(x[is_invertible], inv), x[is_invertible].transpose(0, 2, 1)
        )
        crit_vals[is_invertible] = np.max(np.diagonal(hat_mat.T), axis=1)
    elif criterion == "d-optimal":
        crit_vals = -np.linalg.det(prod)  # minus because we maximize
    elif criterion == "e-optimal":
        eig_vals = np.linalg.eig(prod)[0]
        crit_vals = -np.min(eig_vals, axis=1)  # minus because we maximize
    elif criterion == "maximin":
        distances = []
        for row1, row2 in list(itertools.combinations(range(x.shape[1]), 2)):
            dist = np.linalg.norm(x[:, row1, :] - x[:, row2, :], axis=1, ord=np.inf)
            distances.append(dist)
        crit_vals = -np.vstack(distances).min(axis=0)  # minus because we maximize

    return crit_vals
