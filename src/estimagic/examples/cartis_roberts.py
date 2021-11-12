"""Define the medium scale CUTEst Benchmark Set.

This benchmark set is contains 60 test cases for nonlinear least squares
solvers. It was used to benchmark all modern model based non-linear
derivative free least squares solvers (e.g. POUNDERS, DFOGN, DFOLS).

The parameter dimensions are of medium scale, varying between 25 and 100.

The benchmark set is based on Table 3 in Cartis and Roberts (2019).
Implementation is based either on sources cited in the SIF files or
where available, on AMPL implementaions available here:
- https://vanderbei.princeton.edu/ampl/nlmodels/cute/index.html


"""
from functools import partial

import numpy as np

from estimagic.examples.more_wild import brown_almost_linear


def argtrig(x):
    dim_in = len(x)
    fvec = (
        dim_in
        - np.sum(np.cos(x))
        + np.arange(1, dim_in + 1) * (1 - np.cos(x) - np.sin(x))
    )
    return fvec


def artif(x):
    dim_in = len(x)
    xvec = np.concatenate([[0], x, [0]])
    fvec = np.zeros(dim_in)
    for i in range(dim_in):
        fvec[i] = -0.05 * (xvec[i + 1] + xvec[i + 2] + xvec[i]) + np.arctan(
            np.sin(np.mod(i + 1, 100) * xvec[i + 1])
        )
    return fvec


def arwhdne(x):
    dim_in = len(x)
    fvec = np.zeros(2 * (dim_in - 1))
    fvec[: dim_in - 1] = x[:-1] ** 2 + x[-1] ** 2
    fvec[dim_in - 1 :] = 4 * x[:-1] - 3
    return fvec


def bdvalues(x):
    dim_in = len(x)
    h = 1 / (dim_in + 1)
    xvec = np.concatenate([[0], x, [0]])
    fvec = np.zeros(dim_in)
    for i in range(2, dim_in + 2):
        fvec[i - 2] = (
            -xvec[i - 2]
            + 2 * xvec[i - 1]
            - xvec[i]
            + 0.5 * h ** 2 * (xvec[i - 1] + i * h + 1) ** 3
        )
    return fvec


def bratu_2d(x, alpha):
    p = x.shape[0] + 2
    h = 1 / (p - 1)
    c = h ** 2 * alpha
    xvec = np.zeros((x.shape[0] + 2, x.shape[1] + 2))
    xvec[1 : x.shape[0] + 1, 1 : x.shape[1] + 1] = x
    fvec = np.zeros_like(x)
    for i in range(2, p):
        for j in range(2, p):
            fvec[i - 2, j - 2] = (
                4 * xvec[i - 1, j - 1]
                - xvec[i, j - 1]
                - xvec[i - 2, j - 1]
                - xvec[i - 1, j]
                - xvec[i - 1, j - 2]
                - c * np.exp(xvec[i - 1, j - 1])
            )
    return fvec.flatten()


def bratu_3d(x, alpha):
    p = x.shape[0] + 2
    h = 1 / (p - 1)
    c = h ** 2 * alpha
    xvec = np.zeros((x.shape[0] + 2, x.shape[1] + 2, x.shape[2] + 2))
    xvec[1 : x.shape[0] + 1, 1 : x.shape[1] + 1, 1 : x.shape[2] + 1] = x
    fvec = np.zeros_like(x)
    for i in range(2, p):
        for j in range(2, p):
            for k in range(2, p):
                fvec[i - 2, j - 2, k - 2] = (
                    6 * xvec[i - 1, j - 1, k - 1]
                    - xvec[i, j - 1, k - 1]
                    - xvec[i - 2, j - 1, k - 1]
                    - xvec[i - 1, j, k - 1]
                    - xvec[i - 1, j - 2, k - 1]
                    - xvec[i - 1, j - 1, k]
                    - xvec[i - 1, j - 1, k - 2]
                    - c * np.exp(xvec[i, j, k])
                )
    return fvec.flatten()


def broydn_3d(x):
    xvec = np.zers(len(x) + 2)
    xvec[1 : len(x) + 1] = x
    fvec = (3 - 2 * xvec[1:-1]) * xvec[1:-1] - xvec[:-2] - 2 * xvec[2:] + 1
    return fvec


def get_start_points_bdvalues(n):
    h = 1 / (n + 1)
    x = np.zeros(n)
    for i in range(n):
        x[i] = (i + 1) * h * ((i + 1) * h - 1)
    return x


CARTIS_ROBERTS_PROBLEMS = {
    "argtrig": {
        "criterion": argtrig,
        "start_x": np.ones(100) / 100,
        "solution_x": None,
        "start_criterion": 32.99641,
        "solution_criterion": 0,
    },
    "artif": {
        "criterion": artif,
        "start_x": np.ones(100),
        "solution_x": None,
        "start_criterion": 36.59115,
        "solution_criterion": 0,
    },
    "arwhdne": {
        "criterion": arwhdne,
        "start_x": np.ones(100),
        "solution_x": None,
        "start_criterion": 495,
        "solution_criterion": 27.66203,
    },
    "bdvalues": {
        "criterion": bdvalues,
        "start_x": 1000 * get_start_points_bdvalues(100),
        "solution_x": None,
        "start_criterion": 1.943417e7,
        "solution_criterion": 0,
    },
    "bratu_2d": {
        "criterion": partial(bratu_2d, alpha=4),
        "start_x": np.zeros((8, 8)),
        "solution_x": None,
        "start_criterion": 0.1560738,
        "solution_criterion": 0,
    },
    "bratu_2d_t": {
        "criterion": partial(bratu_2d, alpha=6.80812),
        "start_x": np.zeros((8, 8)),
        "solution_x": None,
        "start_criterion": 0.4521311,
        "solution_criterion": 1.853474e-5,
    },
    "bratu_3d": {
        "criterion": partial(bratu_3d, alpha=6.80812),
        "start_x": np.zeros((3, 3, 3)),
        "solution_x": None,
        "start_criterion": 4.888529,
        "solution_criterion": 0,
    },
    "brownale": {
        "criterion": brown_almost_linear,
        "start_x": 0.5 * np.ones(100),
        "solution_x": None,
        "start_criterion": 2.524757e5,
        "solution_criterion": 0,
    },
}
