import numpy as np


class History:
    """Container to save and retrieve history entries.

    These entries are: xs, fvecs and fvals.

    fvals don't need to be added explicitly, as they are computed internally whenever
    new entries are added.

    """

    def __init__(self, functype):
        self.xs = None
        self.fvecs = None
        self.fvals = None
        self.n_xs = 0
        self.n_fun = 0
        self.index_mapper = {}

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

    def add_xs(self, xs):
        """Add new parameter vectors to the history and return their indices.

        Args:
            xs (np.ndarray or list): 1d or 2d array or list of 1d arrays with
                parameter vectors.

        Returns:
            np.ndarray: 1d array with indices of the added xs.

        """
        is_single = np.ndim(xs) == 1

        xs = np.atleast_2d(xs)

        n_new_points = len(xs) if xs.size != 0 else 0

        if n_new_points == 0:
            return []

        self.xs = _add_entries_to_array(self.xs, xs, self.n_xs)

        x_indices = np.arange(self.n_xs, self.n_xs + n_new_points)

        for x_index in x_indices:
            self.index_mapper[x_index] = []

        self.n_xs += n_new_points

        if is_single:
            x_indices = x_indices[0]

        return x_indices

    def add_evals(self, x_indices, evals):
        """Add new function evaluations to the history.

        Args:
            x_indices (int, list or np.ndarray): Indices of the xs at which the function
                was evaluated.
            evals (np.ndarray or list): 1d or 2d array or list of 1d arrays with
                least square fvecs.

        """
        x_indices = np.atleast_1d(x_indices)

        if not (x_indices < self.n_xs).all():
            raise ValueError(
                "You requested to store a function evaluation for an x vector that is "
                "not in the history."
            )

        n_new_points = len(x_indices)

        if n_new_points == 0:
            return

        if self.functype == "scalar":
            fvecs = np.reshape(evals, (-1, 1))
        else:
            fvecs = np.atleast_2d(evals)

        fvals = np.atleast_1d(self.aggregate(fvecs))

        if n_new_points != len(fvecs):
            raise ValueError()

        self.fvecs = _add_entries_to_array(self.fvecs, fvecs, self.n_fun)
        self.fvals = _add_entries_to_array(self.fvals, fvals, self.n_fun)

        f_indices = np.arange(self.n_fun, self.n_fun + n_new_points)

        for x_index, f_index in zip(x_indices, f_indices):
            self.index_mapper[x_index].append(f_index)

        self.n_fun += n_new_points

    def get_xs(self, x_indices=None):
        """Retrieve xs from history.

        Args:
            x_indices (int, slice or sequence): Specifies the subset of rows that will
                be returned. Anything that can be used to index into a 1d numpy array
                is allowed.

        Returns:
            np.ndarray: 1d or 2d array with parameter vectors

        """
        if isinstance(x_indices, np.ndarray):
            x_indices = x_indices.astype(int)

        out = self.xs[: self.n_xs]
        out = out[x_indices] if x_indices is not None else out

        return out

    def get_fvecs(self, x_indices):
        """Retrieve fvecs from history.

        Args:
            x_indices (int, slice or sequence): Specifies the subset of rows that will
                be returned. Anything that can be used to index into a 1d numpy array
                is allowed.

        Returns:
            np.ndarray or dict: If x_indices is a scalar, a single array is returned.
                Otherwise, a dictionary with x_indices as keys and arrays as values is
                returned.

        """
        out = _extract_from_indices(
            arr=self.fvecs[: self.n_fun],
            mapper=self.index_mapper,
            x_indices=x_indices,
            n_xs=self.n_xs,
        )
        return out

    def get_fvals(self, x_indices):
        """Retrieve fvals from history.

        Args:
            x_indices (int, slice or sequence): Specifies the subset of rows that will
                be returned. Anything that can be used to index into a 1d numpy array
                is allowed.

        Returns:
            np.ndarray or dict: If x_indices is a scalar, a single array is returned.
                Otherwise, a dictionary with x_indices as keys and arrays as values is
                returned.

        """
        out = _extract_from_indices(
            arr=self.fvals[: self.n_fun],
            mapper=self.index_mapper,
            x_indices=x_indices,
            n_xs=self.n_xs,
        )
        return out

    def get_model_data(self, x_indices, average=True):
        if np.isscalar(x_indices):
            x_indices = [x_indices]

        raw_xs = self.get_xs(x_indices)
        raw_fvecs = self.get_fvecs(x_indices)

        if average:
            fvecs = np.array([np.mean(fvec, axis=0) for fvec in raw_fvecs.values()])
            xs = raw_xs
        else:
            fvecs = np.vstack(list(raw_fvecs.values()))
            n_obs = np.array([len(fvec) for fvec in raw_fvecs.values()])
            xs = np.repeat(raw_xs, n_obs, axis=0)

        return xs, fvecs

    def get_n_fun(self):
        return self.n_fun

    def get_n_xs(self):
        return self.n_xs

    def get_x_indices_in_region(self, region):
        # early return if there are no entries
        if self.get_n_fun() == 0:
            return np.array([])
        xs = self.get_xs()
        mask = np.linalg.norm(xs - region.center, axis=1) <= region.radius
        out = np.arange(len(mask))[mask]
        return out

    def __repr__(self):
        return f"History for {self.functype} function with {self.n_fun} entries."


def _add_entries_to_array(arr, new, position):
    if arr is None:
        shape = 1_000 if new.ndim == 1 else (1_000, new.shape[1])
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


def _extract_from_indices(arr, mapper, x_indices, n_xs):
    """Retrieve fvecs or fvals from history.

    Args:
        arr (np.ndarray): 1d or 2d Array with function values.
        mapper (dict): Maps x indices to f indices.
        x_indices (None, int or np.ndarray): Specifies the subset of parameter
            vectors for which the function values will be returned.

    Returns:
        dict or np.ndarray: If x_indices is a scalar, a single array is returned.
            Otherwise, a dictionary with x_indices as keys and arrays as values is
            returned.

    """
    if isinstance(x_indices, np.ndarray):
        x_indices = x_indices.astype(int)

    is_single = np.isscalar(x_indices)
    if is_single:
        x_indices = [x_indices]

    indices = np.arange(n_xs)[x_indices].tolist()

    out = {i: arr[mapper[i]] for i in indices}

    if is_single:
        out = out[x_indices[0]]

    return out
