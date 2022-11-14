import numpy as np
from numba import njit


class History:
    """Container to save and retrieve history entries.

    These entries are: xs, fvecs and fvals.

    fvals don't need to be added explicitly, as they are computed internally
    whenever new entries are added.

    """

    def __init__(self, functype):
        self.xs = None
        self.fvecs = None
        self.fvals = None
        self.n_fun = 0

        self.functype = functype

        if functype == "scalar":
            self.aggregate = lambda x: x.flatten()
        elif functype == "likelihood":
            self.aggregate = lambda x: x.sum(axis=-1)
        elif functype == "least_squares":
            self.aggregate = lambda x: (x**2).sum(axis=-1)
        else:
            raise ValueError(
                "funtype must be 'scalar', 'likelihood' or 'least_squares'."
            )

    def add_entries(self, xs, fvecs):
        """Add new parameter vectors and fvecs to the history.

        Args:
            xs (np.ndarray or list): 1d or 2d array or list of 1d arrays with
                parameter vectors.
            fvecs (np.ndarray or list): 1d or 2d array or list of 1d arrays with
                least square fvecs.
        """
        xs = np.atleast_2d(xs)

        n_new_points = len(xs) if xs.size != 0 else 0

        if n_new_points == 0:
            return

        if self.functype == "scalar":
            fvecs = np.reshape(fvecs, (-1, 1))
        else:
            fvecs = np.atleast_2d(fvecs)

        fvals = np.atleast_1d(self.aggregate(fvecs))

        if n_new_points != len(fvecs):
            raise ValueError()

        self.xs = _add_entries_to_array(self.xs, xs, self.n_fun)
        self.fvecs = _add_entries_to_array(self.fvecs, fvecs, self.n_fun)
        self.fvals = _add_entries_to_array(self.fvals, fvals, self.n_fun)

        self.n_fun += len(xs)

    def get_entries(self, index=None):
        """Retrieve xs, fvecs and fvals from the history.

        Args:
            index (None, int or np.ndarray): Specifies the subset of rows that will
                be returned.

        Returns:
            np.ndarray: 1d or 2d array with parameter vectors.
            np.ndarray: 1d or 2d array with fvecs.
            np.ndarray: Float or 1d array with criterion values.

        """
        names = ["xs", "fvecs", "fvals"]

        out = (getattr(self, name)[: self.n_fun] for name in names)

        # Reducing arrays to length n_fun ensures that invalid indices raise IndexError
        if index is not None:
            out = [arr[index] for arr in out]

        return tuple(out)

    def get_xs(self, index=None):
        """Retrieve xs from history.

        Args:
            index (None, int or np.ndarray): Specifies the subset of rows that will
                be returned.

        Returns:
            np.ndarray: 1d or 2d array with parameter vectors
        """
        out = self.xs[: self.n_fun]
        out = out[index] if index is not None else out

        return out

    def get_fvecs(self, index=None):
        """Retrieve fvecs from history.

        Args:
            index (None, int or np.ndarray): Specifies the subset of rows that will
                be returned.

        Returns:
            np.ndarray: 1d or 2d array with fvecs.
        """
        out = self.fvecs[: self.n_fun]
        out = out[index] if index is not None else out

        return out

    def get_fvals(self, index=None):
        """Retrieve fvals from history.

        Args:
            index (None, int or np.ndarray): Specifies the subset of rows that will
                be returned.

        Returns:
            np.ndarray: Float or 1d array with criterion values.
        """
        out = self.fvals[: self.n_fun]
        out = out[index] if index is not None else out

        return out

    def get_n_fun(self):
        return self.n_fun

    def get_indices_in_trustregion(self, trustregion, norm="infinity"):
        if norm not in ("infinity", "euclidean"):
            raise ValueError("norm must be 'infinity' or 'euclidean'")

        if self.get_n_fun() != 0:
            xs = self.get_xs()

            out = _find_indices_in_trust_region(
                xs, center=trustregion.center, radius=trustregion.radius
            )

            if norm != "infinity":
                # Important: Only screen the indices that are in the trustregion
                # according to infinity norm! Otherwise this would be very expensive!
                raise NotImplementedError

        else:
            out = np.array([])

        return out

    def __repr__(self):
        return f"History for {self.functype} function with {self.n_fun} entries."


def _add_entries_to_array(arr, new, position):
    if arr is None:
        shape = 100_000 if new.ndim == 1 else (100_000, new.shape[1])
        arr = np.full(shape, np.nan)

    n_new_points = len(new) if new.size != 0 else 0

    if len(arr) - position - n_new_points < 0:
        n_extend = max(len(arr), n_new_points)
        if arr.ndim == 2:
            extension_shape = (n_extend, arr.shape[1])
            arr = np.vstack([arr, np.full(extension_shape, np.nan)])
        else:
            arr = np.hstack([arr, np.full(n_extend, np.nan)])

    arr[position : position + n_new_points] = new

    return arr


@njit
def _find_indices_in_trust_region(xs, center, radius):
    """Get the row indices of all parameter vectors in a trust region.

    This is for square trust regions, i.e. balls in term of an infinity norm.

    Args:
        xs (np.ndarray): 2d numpy array where each row is a parameter vector.
        center (np.ndarray): 1d numpy array that marks the center of the trust region.
        radius (float): Radius of the trust region.

    Returns:
        np.ndarray: The indices of parameters in the trust region.

    """
    n_obs, dim = xs.shape
    out = np.zeros(n_obs).astype(np.int64)
    success_counter = 0
    upper = center + radius
    lower = center - radius
    for i in range(n_obs):
        success = True
        for j in range(dim):
            value = xs[i, j]
            if not (lower[j] <= value <= upper[j]) or np.isnan(value):
                success = False
                continue
        if success:
            out[success_counter] = i
            success_counter += 1

    return out[:success_counter]
