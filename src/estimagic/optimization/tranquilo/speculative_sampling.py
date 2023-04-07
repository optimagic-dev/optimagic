def sample_from_next_trustregion(
    next_center,
    old_region,
    n_points,
    sample_points,
    rng,
):
    next_trustregion = old_region._replace(center=next_center)
    xs = sample_points(trustregion=next_trustregion, n_points=n_points, rng=rng)
    return xs
