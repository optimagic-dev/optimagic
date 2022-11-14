from functools import partial

"""Functions to count the effective number of points in a given sample."""


def get_counter(counter, bounds):
    """Get a function that counts the effective number of points in a sample.

    The resulting function takes the following arguments:
    - xs (np.ndarray): A 2d numpy array containing a sample.
    - trustregion (TrustRegion): The current trustregion.

    Args:
        counter (str)
        bounds (Bounds)
    """
    if isinstance(counter, str):
        built_in_counters = {"count_all": count_all}
        counter = built_in_counters[counter]
    elif not callable(counter):
        raise TypeError("counter must be a string or callable.")

    out = partial(counter, bounds=bounds)
    return out


def count_all(xs, trustregion, bounds):  # noqa: U100
    return len(xs)


def count_clusters(xs, trustregion, bounds):  # noqa: U100
    raise NotImplementedError()
