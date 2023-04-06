import numpy as np
import scipy

from estimagic.optimization.tranquilo.clustering import cluster
from estimagic.optimization.tranquilo.get_component import get_component
from estimagic.optimization.tranquilo.volume import get_radius_after_volume_scaling
from estimagic.optimization.tranquilo.options import FilterOptions


def get_sample_filter(sample_filter="keep_all", user_options=None):
    """Get filter function with partialled options.

    The filter function is applied to points inside the current trustregion before
    additional points are sampled.

    The resulting function only takes an array of shape n_points, n_params as argument.

    Args:
        sample_filter (str or callable): The name of a built in filter or a function
            with the filter interface.
        user_options (dict or namedtuple): Additional options for the filter.

    Returns:
        callable: The filter

    """
    built_in_filters = {
        "discard_all": discard_all,
        "keep_all": keep_all,
        "clustering": keep_cluster_centers,
        "drop_excess": drop_excess,
    }

    out = get_component(
        name_or_func=sample_filter,
        component_name="sample_filter",
        func_dict=built_in_filters,
        user_options=user_options,
        default_options=FilterOptions(),
    )

    return out


def discard_all(state):
    return state.x.reshape(1, -1), np.array([state.index])


def keep_all(xs, indices):
    return xs, indices


def drop_excess(xs, indices, state, target_size):
    n_to_drop = max(0, len(xs) - target_size)

    if n_to_drop:
        xs, indices = drop_worst_points(xs, indices, state, n_to_drop)

    return xs, indices


def drop_worst_points(xs, indices, state, n_to_drop):
    """Drop the worst points from xs and indices.

    As long as there are points outside the trustregion, drop the point that is furthest
    away from the trustregion center.

    If all points are inside the trustregion, find the two points that are closest to
    each other. If one of them is the center, drop the other one. If none is the center,
    drop the one that is closer to the center.

    This reflects that we want to have points as far out as possible as long as they are
    inside the trustregion.

    The control flow is a bit complicated to avoid unnecessary or repeated computations
    of distances and pairwise distances.

    """
    n_dropped = 0

    if n_dropped < n_to_drop:
        dists = np.linalg.norm(xs - state.x, axis=1)

        while n_dropped < n_to_drop and (dists > state.trustregion.radius).any():
            drop_index = np.argmax(dists)
            xs = np.delete(xs, drop_index, axis=0)
            indices = np.delete(indices, drop_index)
            dists = np.delete(dists, drop_index, axis=0)
            n_dropped += 1

    if n_dropped < n_to_drop:
        pdists = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(xs))
        pdists[np.diag_indices_from(pdists)] = np.inf

        while n_dropped < n_to_drop:
            i, j = np.unravel_index(np.argmin(pdists), pdists.shape)

            if indices[i] == state.index:
                drop_index = j
            elif indices[j] == state.index:
                drop_index = i
            else:
                drop_index = i if dists[i] < dists[j] else j

            xs = np.delete(xs, drop_index, axis=0)
            indices = np.delete(indices, drop_index)
            dists = np.delete(dists, drop_index, axis=0)
            pdists = np.delete(pdists, drop_index, axis=0)
            pdists = np.delete(pdists, drop_index, axis=1)
            n_dropped += 1

    return xs, indices


def keep_cluster_centers(
    xs, indices, state, target_size, strictness=1e-10, shape="sphere"
):
    dim = xs.shape[1]
    scaling_factor = strictness / target_size
    cluster_radius = get_radius_after_volume_scaling(
        radius=state.trustregion.radius,
        dim=dim,
        scaling_factor=scaling_factor,
    )
    _, centers = cluster(x=xs, epsilon=cluster_radius, shape=shape)

    # do I need to make sure trustregion center is in there?
    out = xs[centers], indices[centers]
    return out
