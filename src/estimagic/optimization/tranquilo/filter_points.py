import numpy as np


def get_sample_filter(sample_filter="keep_all"):
    """Get filter function with partialled options.

    The filter function is applied to points inside the current trustregion before
    additional points are sampled.

    The resulting function only takes an array of shape n_points, n_params as argument.

    Args:
        filter (str or callable): The name of a built in filter or a function with the
            filter interface.

    Returns:
        callable: The filter

    """
    built_in_filters = {
        "discard_all": _discard_all,
        "keep_all": _keep_all,
        "drop_collinear": _drop_collinear,
    }

    if isinstance(sample_filter, str) and sample_filter in built_in_filters:
        out = built_in_filters[sample_filter]
    elif callable(sample_filter):
        out = sample_filter
    else:
        raise ValueError()

    return out


def _discard_all(xs, indices, state):
    return state.x.reshape(1, -1), np.array([state.index])


def _keep_all(xs, indices, state):
    return xs, indices


def _drop_collinear(xs, indices, state):
    """Make sure that the points that are kept are linearly independent."""
    raise NotImplementedError()
