import numpy as np
from numba import njit
from scipy.spatial.distance import pdist, squareform


def cluster(x, epsilon, shape="sphere"):
    """Find clusters in x.

    A cluster is a set of points that are all within a radius
    of eps around the central point of the cluster.

    Args:
        x (np.ndarray): 2d numpy array of shape (n, d) with n points in
            d-dimensional space.
        eps (float): Proximity radius that determines the size of clusters.
        shape (str): One of "sphere" or "cube". This is the shape of the clusters.
            If "sphere", the distances between the points is calculated with an l2 norm.
            If "cube", they are calculated with an infinity norm.

    Returns:
        np.ndarray: 1d integer numpy array containing the cluster of each point.
        np.ndarray: 1d integer numpy array containing the centers of each cluster.

    """
    if shape == "sphere":
        dists = squareform(pdist(x))
    else:
        raise NotImplementedError()

    labels, centers = _cluster(dists, epsilon)
    return labels, centers


@njit
def _cluster(dists, epsilon):
    n_points = len(dists)
    labels = np.full(n_points, -1)
    centers = np.full(n_points, -1)
    n_labeled = 0
    cluster_counter = 0

    while n_labeled < n_points:
        # find best centerpoint among remaining points

        # provoke an index error if forget to set this later
        candidate_center = 2 * n_points
        max_n_neighbors = 0
        for i in range(n_points):
            if labels[i] < 0:
                n_neighbors = 0
                for j in range(n_points):
                    if labels[j] < 0 and j != i and dists[i, j] <= epsilon:
                        n_neighbors += 1
                if n_neighbors == 0:
                    labels[i] = cluster_counter
                    centers[cluster_counter] = i
                    cluster_counter += 1
                    n_labeled += 1
                elif n_neighbors > max_n_neighbors:
                    max_n_neighbors = n_neighbors
                    candidate_center = i

        # if not all points are labeled, we can be sure a cluster center
        # was found
        if n_labeled < n_points:
            i = candidate_center
            for j in range(n_points):
                if labels[j] < 0 and dists[i, j] <= epsilon:
                    labels[j] = cluster_counter
                    n_labeled += 1

            centers[cluster_counter] = i
            cluster_counter += 1

    return labels, centers[:cluster_counter]
