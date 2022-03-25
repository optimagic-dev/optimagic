import numpy as np


def reference_sampler(
    lower_bounds,
    upper_bounds,
    target_size,
    existing_xs=None,
    existing_fvals=None,  # noqa: U100
    seed=1234,
):
    """Naive random generation of trustregion sampling.

    This is just a reference implementation to illustrate the interface of trustregion
    samplers.

    All arguments but seed are mandatory, even if not used.

    Samplers should not make unnecessary checks on input compatibility (e.g. that the
    shapes of existing_xs and existing_fvals match). This will be done automatically
    outside of the sampler.

    Args:
        lower_bounds (np.ndarray): 1d array with lower bounds for the sampled points.
            These must be respected!
        upper_bounds (pn.ndarray): 1d sample with upper bounds for the sampled points.
            These must be respected!
        target_size (int): Target number of points in the combined sample of existing_xs
            and newly sampled points. The sampler does not have to guarantee that this
            number will actually be reached.
        existing_xs (np.ndarray or None): 2d numpy array in which each row is an
            x vector at which the criterion function has already been evaluated.
        existing_fvals (np.ndarray): 1d numpy array with same length as existing_xs
            that contains the corresponding function evaluations.
        bounds (Bounds): NamedTuple with attributes ``lower`` and ``upper``.
        seed (int): Seed for a random number generator.

    Returns:
        dict: A dictionary containing "points" (a numpy array where each row is a
            sampled point) and potentially other information about the sampling.

    """

    if existing_xs is not None:
        n_points = max(1, target_size - len(existing_xs))
    else:
        n_points = target_size

    n_params = len(lower_bounds)

    np.random.seed(seed)
    points = np.random.uniform(
        low=lower_bounds, high=upper_bounds, size=(n_points, n_params)
    )

    out = {"points": points, "message": "Everything is great!"}
    return out
