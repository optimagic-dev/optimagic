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
from estimagic.benchmarking.more_wild import brown_almost_linear
from estimagic.benchmarking.more_wild import watson


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
    x = x.reshape((int(np.sqrt(len(x))), int(np.sqrt(len(x)))))
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
    n = int(np.cbrt(len(x)))
    x = x.reshape((n, n, n))
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
    kappa_1 = 2
    kappa_2 = 1
    fvec = np.zeros_like(x)
    fvec[0] = -2 * x[1] + kappa_2 + (3 - kappa_1 * x[0]) * x[0]
    fvec[1 : len(x) - 1] = (
        -x[:-2] - 2 * x[2:] + kappa_2 + (3 - kappa_1 * x[1:-1]) * x[1:-1]
    )
    fvec[-1] = -x[-2] + kappa_2 + (3 - kappa_1 * x[-1]) * x[-1]
    return fvec


def broydn_bd(x):
    dim_in = len(x)
    fvec = np.zeros(dim_in)
    for i in range(1, 1 + dim_in):
        ji = []
        lb = np.max([1, i - 5])
        ub = np.min([dim_in, i + 1])
        for j in range(lb, ub + 1):
            if j != i:
                ji.append(j)
        fvec[i - 1] = x[i - 1] * (2 + 5 * x[i - 1] ** 2) - np.sum(
            x[np.array(ji) - 1] * (1 + x[np.array(ji) - 1])
        )
    return fvec


def cbratu_2d(x):
    n = int(np.sqrt(len(x) / 2))
    x = x.reshape((2, n, n))
    xvec = np.zeros((x.shape[0], x.shape[1] + 2, x.shape[2] + 2))
    xvec[0, 1 : x.shape[1] + 1, 1 : x.shape[2] + 1] = x[0, :, :]
    xvec[1, 1 : x.shape[1] + 1, 1 : x.shape[2] + 1] = x[1, :, :]
    p = x.shape[1] + 2
    h = 1 / (p - 1)
    alpha = 5
    c = h ** 2 * alpha
    fvec = np.zeros_like(x)
    for i in range(2, p):
        for j in range(2, p):
            fvec[0, i - 2, j - 2] = (
                4 * xvec[0, i - 1, j - 1]
                - xvec[0, i, j - 1]
                - xvec[0, i - 2, j - 1]
                - xvec[0, i - 1, j]
                - xvec[0, i - 1, j - 2]
                - c * np.exp(xvec[0, i - 1, j - 1]) * np.cos(xvec[0, i - 1, j - 1])
            )
            fvec[1, i - 2, j - 2] = (
                4 * xvec[1, i - 1, j - 1]
                - xvec[1, i, j - 1]
                - xvec[1, i - 2, j - 1]
                - xvec[1, i - 1, j]
                - xvec[1, i - 1, j - 2]
                - c * np.exp(xvec[1, i - 1, j - 1]) * np.sin(xvec[1, i - 1, j - 1])
            )
    return fvec.flatten()


def chandheq(x):
    dim_in = len(x)
    constant = 1
    w = np.ones(dim_in) / dim_in
    h = np.ones(dim_in)
    fvec = np.zeros(dim_in)
    for i in range(dim_in):
        fvec[i] = (-0.5 * constant * w * x[i] / (x[i] + x) * h[i] * h + h[i] - 1).sum()
    return fvec


def chemrcta(x):
    dim_in = int(len(x) / 2)
    x = x.reshape((2, dim_in))
    # define the out vector
    fvec = np.zeros(2 * dim_in)
    # define some auxuliary params
    pem = 1
    peh = 5.0
    d = 0.135
    b = 0.5
    beta = 2.0
    gamma = 25.0
    h = 1 / (dim_in - 1)
    cu1 = -h * pem
    cui1 = 1 / (h ** 2 * pem) + 1 / h
    cui = -1 / h - 2 / (h ** 2 * pem)
    ct1 = -h * peh
    cti1 = 1 / (h ** 2 * peh) + 1 / h
    cti = -beta - 1 / h - 2 / (h ** 2 * peh)
    fvec[0] = cu1 * x[0, 1] - x[0, 0] + h * pem
    fvec[1] = ct1 * x[1, 1] - x[1, 0] + h * peh
    for i in range(2, dim_in):
        fvec[i] = (
            -d * x[0, i - 1] * np.exp(gamma - gamma / x[1, i - 1])
            + (cui1) * x[0, i - 2]
            + cui * x[0, i - 1]
            + x[0, i] / (h ** 2 * pem)
        )
        fvec[dim_in - 2 + i] = (
            b * d * x[0, i - 1] * np.exp(gamma - gamma / x[1, i - 1])
            + beta * x[1, i - 1]
            + cti1 * x[1, i - 2]
            + cti * x[1, i - 1]
            + x[1, i] / (h ** 2 * peh)
        )
    fvec[-2] = x[0, -1] - x[0, -2]
    fvec[-1] = x[1, -1] - x[1, -2]
    return fvec


def chemrctb(x):
    dim_in = int(len(x))
    # define the out vector
    fvec = np.zeros(dim_in)
    # define some auxuliary params
    pe = 5.0
    d = 0.135
    b = 0.5
    gamma = 25.0
    h = 1 / (dim_in - 1)
    ct1 = -h * pe
    cti1 = 1 / (h ** 2 * pe) + 1 / h
    cti = -1 / h - 2 / (h ** 2 * pe)
    fvec[0] = ct1 * x[1] - x[0] + h * pe
    for i in range(2, dim_in):
        fvec[i - 1] = (
            d * (b + 1 - x[i - 1]) * np.exp(gamma - gamma / x[i - 1])
            + cti1 * x[i - 2]
            + cti * x[i - 1]
            + x[i] / (h ** 2 * pe)
        )
    fvec[-1] = x[-1] - x[-2]
    return fvec


def chnrsbne(x):
    alfa = np.array(
        [
            1.25,
            1.40,
            2.40,
            1.40,
            1.75,
            1.20,
            2.25,
            1.20,
            1.00,
            1.10,
            1.50,
            1.60,
            1.25,
            1.25,
            1.20,
            1.20,
            1.40,
            0.50,
            0.50,
            1.25,
            1.80,
            0.75,
            1.25,
            1.40,
            1.60,
            2.00,
            1.00,
            1.60,
            1.25,
            2.75,
            1.25,
            1.25,
            1.25,
            3.00,
            1.50,
            2.00,
            1.25,
            1.40,
            1.80,
            1.50,
            2.20,
            1.40,
            1.50,
            1.25,
            2.00,
            1.50,
            1.25,
            1.40,
            0.60,
            1.50,
        ]
    )
    dim_in = len(x)
    fvec = np.zeros(2 * (dim_in - 1))
    fvec[: dim_in - 1] = 4 * alfa[1:] * (x[:-1] - x[1:] ** 2)
    fvec[dim_in - 1 :] = x[1:] - 1
    return fvec


def drcavty(x, r):
    m = int(np.sqrt(len(x)))
    x = x.reshape((m, m))
    h = 1 / (m + 2)
    xvec = np.zeros((m + 4, m + 4))
    xvec[2 : m + 2, 2 : m + 2] = x
    xvec[-2, :] = -h / 2
    xvec[-1, :] = h / 2
    fvec = np.zeros_like(x)
    for i in range(m):
        for j in range(m):
            fvec[i, j] = (
                20 * xvec[i + 2, j + 2]
                - 8 * xvec[i + 1, j + 2]
                - 8 * xvec[i + 3, j + 2]
                - 8 * xvec[i + 2, j + 1]
                - 8 * xvec[i + 2, j + 3]
                + 2 * xvec[i + 1, j + 3]
                + 2 * xvec[i + 3, j + 2]
                + 2 * xvec[i + 1, j + 1]
                + 2 * xvec[i + 3, j + 3]
                + xvec[i, j + 2]
                + xvec[i + 4, j + 2]
                + xvec[i + 2, j]
                + xvec[i + 2, j + 4]
                + (r / 4)
                * (xvec[i + 2, j + 3] - xvec[i + 2, j + 1])
                * (
                    xvec[i, j + 2]
                    + xvec[i + 1, j + 1]
                    + xvec[i + 1, j + 3]
                    - 4 * xvec[i + 1, j + 2]
                    - 4 * xvec[i + 3, j + 2]
                    - xvec[i + 3, j + 2]
                    - xvec[i + 3, j + 3]
                    - xvec[i + 4, j + 2]
                )
                - (r / 4)
                * (xvec[i + 3, j + 2] - xvec[i + 1, j + 2])
                * (
                    xvec[i + 2, j]
                    + xvec[i + 1, j + 1]
                    + xvec[i + 3, j + 1]
                    - 4 * xvec[i + 2, j + 1]
                    - 4 * xvec[i + 2, j + 3]
                    - xvec[i + 1, j + 3]
                    - xvec[i + 3, j + 3]
                    - xvec[i + 2, j + 4]
                )
            )

    return fvec.flatten()


def freurone(x):
    dim_in = len(x)
    fvec = np.zeros((2, dim_in - 1))
    for i in range(dim_in - 1):
        fvec[0, i] = (5.0 - x[i + 1]) * x[i + 1] ** 2 + x[i] - 2 * x[i + 1] - 13.0
        fvec[1, i] = (1.0 + x[i + 1]) * x[i + 1] ** 2 + x[i] - 14 * x[i + 1] - 29.0
    return fvec.flatten()


def hatfldg(x):
    dim_in = len(x)
    fvec = np.zeros(dim_in)
    for i in range(1, dim_in - 1):
        fvec[i - 1] = x[i] * (x[i - 1] - x[i + 1]) + x[i] - x[12] + 1
    fvec[-2] = x[0] - x[12] + 1 - x[0] * x[1]
    fvec[-1] = x[-1] - x[12] + 1 + x[-2] * x[-1]
    return fvec


def integreq(x):
    dim_in = len(x)
    h = 1 / (dim_in + 1)
    t = np.arange(1, dim_in + 1) * h
    xvec = np.concatenate([[0], x, [0]])
    fvec = np.zeros_like(x)
    for i in range(1, dim_in):
        fvec[i - 1] = (
            xvec[i]
            + h
            * (
                (1 - t[i - 1]) * (t[:i] * (xvec[1 : i + 1] + t[:i] + 1) ** 3).sum()
                + t[i - 1] * ((1 - t[i:]) * (xvec[i + 1 : -1] + t[i:] + 1) ** 3).sum()
            )
            / 2
        )
    fvec[-1] = (
        xvec[-2]
        + h
        * (
            (1 - t[-1]) * (t * (xvec[1:-1] + t + 1) ** 3).sum()
            + t[-1] * ((1 - t[-1]) * (xvec[-2] + t[-1] + 1) ** 3)
        )
        / 2
    )
    return fvec


def msqrta(x):
    dim_in = int(np.sqrt(len(x)))
    xmat = x.reshape((dim_in, dim_in))
    bmat = 5 * xmat
    amat = np.zeros((dim_in, dim_in))
    for i in range(1, dim_in + 1):
        for j in range(1, dim_in + 1):
            amat[i - 1, j - 1] = (bmat[i - 1, :] * bmat[:, j - 1]).sum()
    fmat = np.zeros((dim_in, dim_in))
    for i in range(1, dim_in + 1):
        for j in range(1, dim_in + 1):
            fmat[i - 1, j - 1] = (xmat[i - 1, :] * xmat[:, j - 1]).sum() - amat[
                i - 1, j - 1
            ]
    return fmat.flatten()


def penalty_1(x, a=1e-5):
    fvec = np.sqrt(a) * (x - 2)
    fvec = np.concatenate([fvec, [x @ x - 1 / 4]])
    return fvec


def penalty_2(x, a=1e-10):
    dim_in = len(x)
    y = np.exp(np.arange(1, 2 * dim_in + 1) / 10) + np.exp(np.arange(2 * dim_in) / 10)
    fvec = np.zeros(2 * dim_in)
    fvec[0] = x[0] - 0.2
    fvec[1:dim_in] = np.sqrt(a) * (
        np.exp(x[1:] / 10) + np.exp(x[:-1] / 10) - y[1:dim_in]
    )
    fvec[dim_in:-1] = np.sqrt(a) * (np.exp(x[1:] / 10) - np.exp(-1 / 10))
    fvec[-1] = (np.arange(1, dim_in + 1)[::-1] * x ** 2).sum() - 1
    return fvec


def vardimne(x):
    dim_in = len(x)
    fvec = np.zeros(dim_in + 2)
    fvec[:-2] = x - 1
    fvec[-2] = (np.arange(1, dim_in + 1) * (x - 1)).sum()
    fvec[-1] = ((np.arange(1, dim_in + 1) * (x - 1)).sum()) ** 2
    return fvec


def yatpsq_1(x, dim_in):
    xvec = x[: dim_in ** 2]
    xvec = xvec.reshape((dim_in, dim_in))
    yvec = x[dim_in ** 2 : dim_in ** 2 + dim_in]
    zvec = x[dim_in ** 2 + dim_in : dim_in ** 2 + 2 * dim_in]
    fvec = np.zeros((dim_in, dim_in))
    for i in range(dim_in):
        for j in range(dim_in):
            fvec[i, j] = (
                xvec[i, j] ** 3
                - 10 * xvec[i, j] ** 2
                - (yvec[i] + zvec[j])
                * (xvec[i, j] * np.cos(xvec[i, j]) - np.sin(xvec[i, j]))
            )
    fvec = fvec.flatten()
    temp = (np.sin(xvec) / xvec).sum(axis=0) - 1
    fvec = np.concatenate([fvec, temp])
    temp = (np.sin(xvec) / xvec).sum(axis=1) - 1
    fvec = np.concatenate([fvec, temp])
    return fvec


def yatpsq_2(x, dim_in):
    xvec = x[: dim_in ** 2]
    xvec = xvec.reshape((dim_in, dim_in))
    yvec = x[dim_in ** 2 : dim_in ** 2 + dim_in]
    zvec = x[dim_in ** 2 + dim_in : dim_in ** 2 + 2 * dim_in]
    fvec = np.zeros((dim_in, dim_in))
    for i in range(dim_in):
        for j in range(dim_in):
            fvec[i, j] = xvec[i, j] - (yvec[i] + zvec[j]) * (1 + np.cos(xvec[i, j])) - 1
    fvec = fvec.flatten()
    temp = (np.sin(xvec) + xvec).sum(axis=0) - 1
    fvec = np.concatenate([fvec, temp])
    temp = (np.sin(xvec) + xvec).sum(axis=1) - 1
    fvec = np.concatenate([fvec, temp])
    return fvec


def get_start_points_msqrta(dim_in, flag=1):
    bmat = np.zeros((dim_in, dim_in))
    for i in range(1, dim_in + 1):
        for j in range(1, dim_in + 1):
            bmat[i - 1, j - 1] = np.sin(((i - 1) * dim_in + j) ** 2)
    if flag == 2:
        bmat[2, 0] = 0
    xmat = 0.2 * bmat
    return xmat.flatten()


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
        "start_x": np.zeros(64),
        "solution_x": None,
        "start_criterion": 0.1560738,
        "solution_criterion": 0,
    },
    "bratu_2d_t": {
        "criterion": partial(bratu_2d, alpha=6.80812),
        "start_x": np.zeros(64),
        "solution_x": None,
        "start_criterion": 0.4521311,
        "solution_criterion": 1.853474e-5,
    },
    "bratu_3d": {
        "criterion": partial(bratu_3d, alpha=6.80812),
        "start_x": np.zeros(27),
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
    "broydn_3d": {
        "criterion": broydn_3d,
        "start_x": -np.ones(100),
        "solution_x": None,
        "start_criterion": 111,
        "solution_criterion": 0,
    },
    "cbratu_2d": {
        "criterion": cbratu_2d,
        "start_x": np.zeros(2 * 5 * 5),
        "solution_x": None,
        "start_criterion": 0.4822531,
        "solution_criterion": 0,
    },
    "broydn_bd": {
        "criterion": broydn_bd,
        "start_x": np.ones(100),
        "solution_x": None,
        "start_criterion": 2404,
        "solution_criterion": 0,
    },
    "chandheq": {
        "criterion": chandheq,
        "start_x": np.arange(1, 101) / 100,
        "solution_x": None,
        "start_criterion": 6.923365,
        "solution_criterion": 0,
    },
    "chemrcta": {
        "criterion": chemrcta,
        "start_x": np.ones(100),
        "solution_x": None,
        "start_criterion": 3.0935,
        "solution_criterion": 0,
        "lower_bounds": np.concatenate([np.zeros(50), 1e-6 * np.ones(50)]),
    },
    "chemrctb": {
        "criterion": chemrctb,
        "start_x": np.ones(100),
        "solution_x": None,
        "start_criterion": 1.446513,
        "solution_criterion": 1.404424e-3,
        "lower_bounds": 1e-6 * np.ones(100),
    },
    "chnrsbne": {
        "criterion": chnrsbne,
        "start_x": -np.ones(50),
        "solution_x": None,
        "start_criterion": 7635.84,
        "solution_criterion": 0,
    },
    "drcavty1": {
        "criterion": partial(drcavty, r=500),
        "start_x": np.zeros(100),
        "solution_x": None,
        "start_criterion": 0.4513889,
        "solution_criterion": 0,
    },
    "drcavty2": {
        "criterion": partial(drcavty, r=1000),
        "start_x": np.zeros(100),
        "solution_x": None,
        "start_criterion": 0.4513889,
        "solution_criterion": 5.449602e-3,
    },
    "drcavty3": {
        "criterion": partial(drcavty, r=4500),
        "start_x": np.zeros(100),
        "solution_x": None,
        "start_criterion": 0.4513889,
        "solution_criterion": 0,
    },
    "freurone": {
        "criterion": freurone,
        "start_x": np.concatenate([np.array([0.5, -2]), np.zeros(98)]),
        "solution_x": None,
        "start_criterion": 9.95565e4,
        "solution_criterion": 1.196458e4,
    },
    "hatfldg": {
        "criterion": hatfldg,
        "start_x": np.ones(25),
        "solution_x": None,
        "start_criterion": 27,
        "solution_criterion": 10,
    },
    "integreq": {
        "criterion": integreq,
        "start_x": np.arange(1, 101) / 101 * (np.arange(1, 101) / 101 - 1),
        "solution_x": None,
        "start_criterion": 0.5730503,
        "solution_criterion": 0,
    },
    "msqrta": {
        "criterion": msqrta,
        "start_x": get_start_points_msqrta(10),
        "solution_x": None,
        "start_criterion": 212.7162,
        "solution_criterion": 0,
    },
    "msqrtb": {
        "criterion": msqrta,
        "start_x": get_start_points_msqrta(10, flag=2),
        "solution_x": None,
        "start_criterion": 205.0753,
        "solution_criterion": 0,
    },
    "penalty_1": {
        "criterion": penalty_1,
        "start_x": np.arange(1, 101),
        "solution_x": None,
        "start_criterion": 1.144806e11,
        "solution_criterion": 9.025000e-9,
    },
    "penalty_2": {
        "criterion": penalty_2,
        "start_x": np.ones(100) * 0.5,
        "solution_x": None,
        "start_criterion": 1.591383e6,
        "solution_criterion": 0.9809377,
    },
    "vardimne": {
        "criterion": vardimne,
        "start_x": 1 - np.arange(1, 101) / 100,
        "solution_x": None,
        "start_criterion": 1.310584e14,
        "solution_criterion": 0,
    },
    "watsonne": {
        "criterion": watson,
        "start_x": np.zeros(31),
        "solution_x": None,
        "start_criterion": 30,
        "solution_criterion": 0,
    },
    "yatpsq_1": {
        "criterion": partial(yatpsq_1, dim_in=10),
        "start_x": np.concatenate([np.ones(100) * 6, np.zeros(20)]),
        "solution_x": None,
        "start_criterion": 2.073643e6,
        "solution_criterion": 0,
    },
    "yatpsq_2": {
        "criterion": partial(yatpsq_2, dim_in=10),
        "start_x": np.concatenate([np.ones(100) * 10, np.zeros(20)]),
        "solution_x": None,
        "start_criterion": 1.831687e5,
        "solution_criterion": 0,
    },
}
