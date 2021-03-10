import numpy as np
import pandas as pd

# ======================================================================================
# Define criterion functions
# ======================================================================================


def trid_scalar_criterion(params):
    x = params["value"].to_numpy()
    return ((params["value"] - 1) ** 2).sum() - (params["value"][1:] * x[:-1]).sum()


def trid_gradient(params):
    x = params["value"].to_numpy()
    l1 = np.insert(x, 0, 0)
    l1 = np.delete(l1, [-1])
    l2 = np.append(x, 0)
    l2 = np.delete(l2, [0])
    return 2 * (x - 1) - l1 - l2


def trid_pandas_gradient(params):
    x = params["value"].to_numpy()
    l1 = np.insert(x, 0, 0)
    l1 = np.delete(l1, [-1])
    l2 = np.append(x, 0)
    l2 = np.delete(l2, [0])
    return pd.Series(2 * (x - 1) - l1 - l2)


def trid_criterion_and_gradient(params):
    x = params["value"].to_numpy()
    l1 = np.insert(x, 0, 0)
    l1 = np.delete(l1, [-1])
    l2 = np.append(x, 0)
    l2 = np.delete(l2, [0])
    return ((params["value"] - 1) ** 2).sum() - (
        params["value"][1:] * x[:-1]
    ).sum(), 2 * (x - 1) - l1 - l2


def rotated_hyper_ellipsoid_scalar_criterion(params):
    val = 0
    for i in range(len(params["value"])):
        val += (params["value"][: i + 1] ** 2).sum()
    return val


def rotated_hyper_ellipsoid_gradient(params):
    x = params["value"].to_numpy()
    return np.arange(2 * len(x), 0, -2) * x


def rotated_hyper_ellipsoid_pandas_gradient(params):
    x = params["value"].to_numpy()
    return pd.Series(np.arange(2 * len(x), 0, -2) * x)


def rotated_hyper_ellipsoid_criterion_and_gradient(params):
    val = 0
    for i in range(len(params["value"])):
        val += (params["value"][: i + 1] ** 2).sum()
    x = params["value"].to_numpy()
    return val, np.arange(2 * len(x), 0, -2) * x


def rosenbrock_scalar_criterion(params):
    x = params["value"].to_numpy()
    r1 = ((x[1:] - x[:-1] ** 2) ** 2).sum() * 100
    r2 = ((x[:-1] - 1) ** 2).sum()
    return r1 + r2
