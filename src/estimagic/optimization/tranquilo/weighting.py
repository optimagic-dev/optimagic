from functools import partial


def get_sample_weighter(weighter, bounds):
    """Get a function that calculates weights for points in a sample.

    The resulting function takes the following arguments:
    - xs (np.ndarray): A 2d numpy array containing a sample.
    - trustregion (TrustRegion): The current trustregion.

    Args:
        weighter (str)
        bounds (Bounds)

    """
    if isinstance(weighter, str):
        built_in_weighters = {"no_weights": no_weights}
        weighter = built_in_weighters[weighter]
    elif not callable(weighter):
        raise TypeError("weighter must be a string or callable.")

    out = partial(weighter, bounds=bounds)
    return out


def no_weights(xs, trustregion, bounds):
    return None
