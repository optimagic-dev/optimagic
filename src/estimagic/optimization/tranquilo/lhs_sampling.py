import itertools
from itertools import combinations

import numpy as np


def get_next_trust_region_points_latin_hypercube(
    center,
    radius,
    n_points,
    existing_points=None,
    optimality_criterion="maximin",
    lhs_design="centered",
    target="linear",
    n_iter=10_000,
):
    """Generate new points at which the criterion should be evaluated.

    Important note
    --------------
    If existing points are passed the returned sample of points may exceed the number
    of requested points, and the sample is will not be a valid Latin-Hypercube. The
    sample will, however, include a subset of points that constitutes a valid
    Latin-Hyercube.

    Description
    -----------
    Generates an (optimal) Latin hypercube sample taking into account already existing
    points. Optimality is defined via different criteria, see
    :func:`compute_optimality_criterion`. The best sample is chosen via random search.

    Args:
        center (np.ndarray): Center of the current trust region.
        radius (float): Radius of the current trust region.
        n_points (int): Number of points in the trust region at which criterion values
            are known. The actual number can be larger than this if the existing points
            are badly spaced.
        existing_points (np.ndarray): 2d Array where each row is a parameter vector at
            which the criterion has already been evaluated.
        optimality_criterion (str): One of "a-optimal", "d-optimal", "e-optimal",
            "g-optimal" or "maximin". Default "maximin".
        lhs_design (str): One of "random", "centered". Determines how sample points are
            spaced inside bins. Default 'centered'.
        target (str): One of "linear" or "quadratic". Determines in which space the
            criterion is evaluated. For "quadratic" the Latin-Hypercube sample X is
            mapped onto Y(X) = [X, X ** 2] and the criterion value for X is computed
            using Y(X).  This can be useful if one chooses an optimality criterion that
            minimizes e.g.  the variance of the least-squares estimator, while using a
            quadratic or polynomial model. Default is "linear".
        n_iter (int): Iterations considered in random search.

    Returns:
        out (dict): Dictionary with entries:
        - 'points' (np.ndarray): The (optimal) Latin hypercube sample. Has shape
        (n_points, len(center)).
        - 'crit_vals' (np.ndarray): 1d array of length n_iter, containing the criterion
        values assigned to each candidate sample. Mainly used for debugging.

    """
    if lhs_design not in {"centered", "random"}:
        raise ValueError(
            "Invalid Latin hypercube design. Must be in {'random', 'centered'}"
        )

    n_dim = len(center)
    dtype = np.uint8 if n_points < 256 else np.uint16

    if existing_points is None:
        candidates = _create_upscaled_lhs_sample(n_dim, n_points, n_iter, dtype)
        n_new_points = n_points
    else:
        existing_upscaled = _scale_up_points(existing_points, center, radius, n_points)
        empty_bins = _get_empty_bin_info(existing_upscaled, n_points)
        candidates = _extend_upscaled_lhs_sample(empty_bins, n_points, n_iter, dtype)
        existing_upscaled = np.tile(existing_upscaled, (n_iter, 1, 1))
        candidates = np.concatenate((existing_upscaled, candidates), axis=1)
        n_new_points = n_points - len(existing_points)

    candidates = candidates.astype(float)
    if lhs_design == "centered" and n_new_points > 0:
        candidates[:, -n_new_points:, :] += 0.5
    elif lhs_design == "random" and n_new_points > 0:
        candidates[:, -n_new_points:, :] += np.random.uniform(
            size=(n_new_points, n_dim)
        )

    candidates = _scale_down_points(candidates, center, radius, n_points)

    crit_vals = compute_optimality_criterion(
        candidates, criterion=optimality_criterion, target=target
    )
    points = candidates[np.argmin(crit_vals)]

    out = {"points": points, "crit_vals": crit_vals}
    return out


def compute_optimality_criterion(x, criterion, target):
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
        for row1, row2 in list(combinations(range(x.shape[1]), 2)):
            dist = np.linalg.norm(x[:, row1, :] - x[:, row2, :], axis=1, ord=np.inf)
            distances.append(dist)
        crit_vals = -np.vstack(distances).min(axis=0)  # minus because we maximize

    return crit_vals


def get_existing_points(old_sample, new_center, new_radius):
    """Locate subset of points in new region.

    Args:
        old_sample (np.ndarray): Previously sampled Latin Hypercube.
        new_center (np.ndarray): Center of the new trust region. Must
            fufill len(new_center) == old_sample.shape[1].
        new_radius (float): Radius of the new trust region.

    Returns:
        existing (np.ndarray): Points in old_sample that fall into
            the new region. If there are no points, returns None.

    """
    lower = new_center - new_radius
    upper = new_center + new_radius

    existing = []
    for row in old_sample:
        if (lower <= row).all() and (upper >= row).all():
            existing.append(row)

    existing = np.array(existing) if len(existing) > 0 else None
    return existing


def _create_upscaled_lhs_sample(n_dim, n_points, n_designs, dtype=np.uint8):
    """Create an upscaled Latin hypercube sample (LHS).

    Args:
        n_dim (int): Dimensionality of the problem.
        n_points (int): Number of sample points.
        n_designs (int): Number of different hypercubes to sample.
        dtype (np.uint8 or np.unt16): Data type of arrays. Default np.unint8.

    Returns:
        sample (np.ndarray): Latin Hypercube sample of shape (n_designs, n_samples,
            n_dim)

    """
    index = np.arange(n_points, dtype=dtype)
    sample = np.empty((n_designs, n_dim, n_points), dtype=dtype)

    for i, j in itertools.product(range(n_designs), range(n_dim)):
        np.random.shuffle(index)
        sample[i, j] = index

    sample = np.swapaxes(sample, 1, 2)
    return sample


def _scale_up_points(points, center, radius, n_points):
    """Scale Latin Hypercube sample up.

    Latin hypercubes are sampled using index arrays, which distributes the array entries
    over range(n_points). To get a sample from a specific region, e.g. [0, 1] ** 2, the
    samples have to be scaled down. This function performs the inverse operation. For
    the down-scaling operation see :func:`_scale_down_points`.

    Args:
        points (np.ndarray): Previously sampled Latin Hypercube, 2d.
        center (np.ndarray): Center of the new trust region, 1d.
            Must fufill len(center) == points.shape[1].
        radius (float): Radius of the new trust region.

    Returns:
        scaled (np.ndarray): The scaled-up version of points.

    """
    lower = center - radius
    scaled = (points - lower) * n_points / (2 * radius)
    return scaled


def _scale_down_points(points, center, radius, n_points):
    """Scale Latin Hypercube sample up.

    Latin hypercubes are sampled using index arrays, which distributes the array entries
    over range(n_points). To get a sample from a specific region, e.g. [0, 1] ** 2, the
    samples have to be scaled down. For the inverse operation see function
    :func:`_scale_up_points`.

    Args:
        points (np.ndarray): Previously sampled Latin Hypercube. If 3d the scaling will
            be performed along the last 2 dimenions.
        center (np.ndarray): Center of the new trust region, 1d.
            Must fufill len(center) == points.shape[1].
        radius (float): Radius of the new trust region.

    Returns:
        scaled (np.ndarray): The scaled-up version of points.

    """
    lower = center - radius
    scaled = points / n_points * (2 * radius) + lower
    return scaled


def _get_empty_bin_info(existing_upscaled, n_points):
    """Find empty bins in space populated by existing points.

    Args:
        existing_upscaled (np.ndarray): Upscaled points.
        n_points (int): Number of points originally sampled.

    Returns:
        out (np.ndarray): Empty bins for each dimension. Has shape
            (len(existing_upscaled) - n_points, n_dim). Non-empty bins are marked by -1,
            and occur since not all dimensions must have the same number of empty bins.

    """
    n_dim = existing_upscaled.shape[1]
    empty_bins = []
    all_bins = set(range(n_points))

    for j in range(n_dim):
        filled_bins = set(np.floor(existing_upscaled[:, j].astype(int)))
        empty_bins.append(sorted(all_bins - filled_bins))

    max_empty = max(map(len, empty_bins))

    out = np.full((max_empty, n_dim), -1)
    for j, empty in enumerate(empty_bins):
        out[: len(empty), j] = empty

    return out


def _extend_upscaled_lhs_sample(empty_bins, n_points, n_designs, dtype=np.uint8):
    """Extend a sample to a full Latin hypercube sample (LHS).

    Args:
        empty_bins (np.ndarray): Dimensionality of the problem.
        n_points (int): Number of (total) sample points.
        n_designs (int): Number of different hypercubes to sample.
        dtype (np.uint8 or np.unt16): Data type of arrays. Default np.unint8.

    Returns:
        sample (np.ndarray): Latin Hypercube sample of shape (n_designs, n_samples,
            n_dim)

    """
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

    sample = sample.swapaxes(1, 2)
    return sample
