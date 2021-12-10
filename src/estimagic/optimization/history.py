import numpy as np


class LeastSquaresHistory:
    """Container to save and retrieve history entries for a least squares optimizer."""

    def __init__(self):
        self.xs = None
        self.residuals = None
        self.critvals = None
        self.n_fun = 0
        self.min_index = None
        self.min_critval = np.inf

    def add_entries(self, xs, residuals):
        """Add new parameter vectors and residuals to the history.

        Args:
            xs (np.ndarray or list): 1d or 2d array or list of 1d arrays with
                parameter vectors.
            residuals (np.ndarray or list): 1d or 2d array or list of 1d arrays with
                least square residuals.

        """
        xs = np.atleast_2d(xs)
        residuals = np.atleast_2d(residuals)
        critvals = np.atleast_1d((residuals ** 2).sum(axis=-1))

        argmin_candidate = critvals.argmin()
        min_candidate = critvals[argmin_candidate]
        if min_candidate < self.min_critval:
            self.min_index = argmin_candidate + self.n_fun

        if len(xs) != len(residuals):
            raise ValueError()

        self.xs = _add_entries_to_array(self.xs, xs, self.n_fun)
        self.residuals = _add_entries_to_array(self.residuals, residuals, self.n_fun)
        self.critvals = _add_entries_to_array(self.critvals, critvals, self.n_fun)

        self.n_fun += len(xs)

    def add_centered_entries(self, xs, residuals, center_info):
        """Add new parameter vectors and residuals to the history.

        Args:
            xs (np.ndarray or list): 1d or 2d array or list of 1d arrays with
                parameter vectors.
            residuals (np.ndarray or list): 1d or 2d array or list of 1d arrays with
                least square residuals.
            center_info (dict): Dictionary with the entries "x", "residuals" and
                "radius". The information is used to uncenter parameters and residuals
                before adding them to the history.

        """
        xs = np.atleast_2d(xs)
        residuals = np.atleast_2d(residuals)
        xs_uncentered = xs * center_info["radius"] + center_info["x"]
        residuals_uncentered = residuals + center_info["residuals"]
        self.add_entries(xs_uncentered, residuals_uncentered)

    def get_entries(self, index=None):
        """Retrieve xs, residuals and critvals from the history.

        Args:
            index (None, int or np.ndarray): Specifies the subset of rows that will
                be returned.

        Returns:
            np.ndarray: 1d or 2d array with parameter vectors
            np.ndarray: 1d or 2d array with residuals
            np.ndarray: Float or 1d array with criterion values.

        """
        names = ["xs", "residuals", "critvals"]

        out = (getattr(self, name)[: self.n_fun] for name in names)

        # reducing arrays to length n_fun ensures that invalid indices raise IndexError
        if index is not None:
            out = [arr[index] for arr in out]

        return tuple(out)

    def get_centered_entries(self, center_info, index=None):
        """Retrieve xs, residuals and critvals from the history.

        Args:
            center_info (dict): Dictionary with the entries "x", "residuals" and
                "radius". The information is used to center parameters, residuals
                and critvals.
            index (None, int or np.ndarray): Specifies the subset of rows that will
                be returned.

        Returns:
            np.ndarray: 1d or 2d array with centered parameter vectors
            np.ndarray: 1d or 2d array with centered residuals
            np.ndarray: Float or 1d array with centered criterion values.

        """
        xs_unc, residuals_unc, _ = self.get_entries(index=index)
        xs = (xs_unc - center_info["x"]) / center_info["radius"]
        residuals = residuals_unc - center_info["residuals"]
        critvals = (residuals ** 2).sum(axis=-1)
        return xs, residuals, critvals

    def get_n_fun(self):
        return self.n_fun

    def get_min_index(self):
        return self.min_index

    def get_best_entries(self):
        return self.get_entries(index=self.min_index)

    def get_best_centered_entries(self, center_info):
        return self.get_centered_entries(self, center_info, index=self.min_index)


def _add_entries_to_array(arr, new, position):
    if arr is None:
        shape = 1000 if new.ndim == 1 else (1000, new.shape[1])
        arr = np.full(shape, np.nan)

    if len(arr) - position - len(new) < 0:
        n_extend = max(len(arr), len(new))
        if arr.ndim == 2:
            extension_shape = (n_extend, arr.shape[1])
            arr = np.vstack([arr, np.full(extension_shape, np.nan)])
        else:
            arr = np.hstack([arr, np.full(n_extend, np.nan)])

    arr[position : position + len(new)] = new

    return arr
