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

from estimagic.benchmarking.more_wild import (
    brown_almost_linear,
    linear_full_rank,
    linear_rank_one,
    watson,
)


# =====================================================================================
def luksan11(x):
    dim_in = len(x)

    fvec = np.zeros(2 * (dim_in - 1))
    fvec[::2] = 20 * x[:-1] / (1 + x[:-1] ** 2) - 10 * x[1:]
    fvec[1::2] = x[:-1] - 1

    return fvec


def luksan12(x):
    dim_in = len(x)
    n = (dim_in - 2) // 3
    fvec = np.zeros(2 * (dim_in - 2))

    fvec[: 6 * n : 6] = 10 * (x[:n] ** 2 - x[1 : n + 1])
    fvec[1 : 6 * n : 6] = x[2 : 3 * n + 2 : 3] - 1
    fvec[2 : 6 * n : 6] = (x[3 : 3 * n + 3 : 3] - 1) ** 2
    fvec[3 : 6 * n : 6] = (x[4 : 3 * n + 4 : 3] - 1) ** 3
    fvec[4 : 6 * n : 6] = (
        x[:n] ** 2 * x[3 : 3 * n + 3 : 3]
        + np.sin(x[3 : 3 * n + 3 : 3] - x[4 : 3 * n + 4 : 3])
        - 10
    )
    fvec[5 : 6 * n : 6] = (
        x[1 : n + 1] + (x[2 : 3 * n + 2 : 3] ** 4) * (x[3 : 3 * n + 3 : 3] ** 2) - 20
    )

    return fvec


def luksan13(x):
    dim_in = len(x)
    n = (dim_in - 2) // 3
    fvec = np.zeros(n * 7)

    i = np.arange(n)
    k = i * 7
    fvec[k] = 10 * (x[3 * i] ** 2 - x[3 * i + 1])
    fvec[k + 1] = 10 * (x[3 * i + 1] ** 2 - x[3 * i + 2])
    fvec[k + 2] = (x[3 * i + 2] - x[3 * i + 3]) ** 2
    fvec[k + 3] = (x[3 * i + 3] - x[3 * i + 4]) ** 3
    fvec[k + 4] = x[3 * i] + x[3 * i + 1] ** 2 + x[3 * i + 2] - 30
    fvec[k + 5] = x[3 * i + 1] - x[3 * i + 2] ** 2 + x[3 * i + 3] - 10
    fvec[k + 6] = x[3 * i + 1] * x[3 * i + 4] - 10

    return fvec


def luksan14(x):
    dim_in = len(x)
    dim_out = 7 * (dim_in - 2) // 3
    fvec = np.zeros(dim_out)

    for i in range(0, dim_in - 2, 3):
        k = (i // 3) * 7
        fvec[k : k + 7] = [
            10 * (x[i] ** 2 - x[i + 1]),
            x[i + 1] + x[i + 2] - 2,
            x[i + 3] - 1,
            x[i + 4] - 1,
            x[i] + 3 * x[i + 1],
            x[i + 2] + x[i + 3] - 2 * x[i + 4],
            10 * (x[i + 1] ** 2 - x[i + 4]),
        ]

    return fvec


def luksan15(x):
    dim_in = len(x)
    dim_out = (dim_in - 2) * 2
    temp = np.zeros((dim_out, 3))
    y = np.tile([35.8, 11.2, 6.2, 4.4], dim_out // 4)

    for p in range(1, 4):
        k = 0
        for i in range(0, dim_in - 2, 2):
            for j in range(1, 5):
                temp[k, p - 1] = (p**2 / j) * np.abs(
                    x[i] * (x[i + 1] ** 2) * (x[i + 2] ** 3) * (x[i + 3] ** 4)
                ) ** (1 / (p * j))

                k += 1

    fvec = y - np.sum(temp, axis=1)

    return fvec


def luksan16(x):
    dim_in = len(x)
    dim_out = (dim_in - 2) * 2
    temp = np.zeros((dim_out, 3))
    y = np.tile([35.8, 11.2, 6.2, 4.4], dim_out // 4)

    for p in range(1, 4):
        k = 0
        for i in range(0, dim_in - 2, 2):
            for j in range(1, 5):
                temp[k, p - 1] = (p**2 / j) * np.exp(
                    (x[i] + 2 * x[i + 1] + 3 * x[i + 2] + 4 * x[i + 3]) * (1 / (p * j))
                )
                k += 1

    fvec = y - np.sum(temp, axis=1)

    return fvec


def luksan17(x):
    dim_in = len(x)
    dim_out = (dim_in - 2) * 2
    temp = np.zeros((dim_out, 4))
    y = np.tile([30.6, 72.2, 124.4, 187.4], dim_out // 4)

    for q in range(1, 5):
        k = 0
        for i in range(-1, dim_in - 4, 2):
            for j in range(1, 5):
                temp[k, q - 1] += -j * q**2 * np.sin(x[i + q]) + j**2 * q * np.cos(
                    x[i + q]
                )
                k += 1

    fvec = y - np.sum(temp, axis=1)

    return fvec


def luksan21(x):
    dim_out = len(x)
    h = 1 / (dim_out + 1)
    fvec = np.zeros(dim_out)

    fvec[0] = 2 * x[0] + 0.5 * h**2 * (x[0] + h + 1) ** 3 - x[1] + 1
    for i in range(1, dim_out - 1):
        fvec[i] = (
            2 * x[i]
            + 0.5 * h**2 * (x[i] + h * (i + 1) + 1) ** 3
            - x[i - 1]
            - x[i + 1]
            + 1
        )
    fvec[-1] = 2 * x[-1] + 0.5 * h**2 * (x[-1] + h * dim_out + 1) ** 3 - x[-2] + 1

    return fvec


def luksan22(x):
    dim_out = 2 * len(x) - 2
    fvec = np.zeros(dim_out)

    fvec[0] = x[0] - 1
    fvec[1:-1:2] = 10 * (x[:-2] ** 2 - x[1:-1])
    fvec[2:-1:2] = 2 * np.exp(-((x[:-2] - x[1:-1]) ** 2)) + np.exp(
        -2 * (x[1:-1] - x[2:]) ** 2
    )
    fvec[-1] = -10 * (x[-2] ** 2)

    return fvec


def morebvne(x):
    dim_in = len(x)
    h = 1 / (dim_in + 1)
    i = np.arange(1, dim_in + 1)
    fvec = np.zeros(dim_in)

    fvec[0] = 2 * x[0] - x[1] + h**2 / 2 * (x[0] + i[0] * h + 1) ** 3
    fvec[1:-1] = (
        2 * x[1:-1] - x[:-2] - x[2:] + h**2 / 2 * (x[1:-1] + i[1:-1] * h + 1) ** 3
    )
    fvec[-1] = 2 * x[-2] - x[-2] + h**2 / 2 * (x[-1] + i[-1] * h + 1) ** 3

    return fvec


def flosp2(x, const, ra=1.0e7, dim_in=2):
    n = dim_in * 2 + 1
    xvec = np.ones((3, n, n))
    xvec[0] = x[: n**2].reshape(n, n)
    xvec[1] = x[n**2 : 2 * n**2].reshape(n, n)
    xvec[2, 1:-1, 1:-1] = x[2 * n**2 :].reshape(n - 2, n - 2)

    a = const[0]
    b = const[1]
    f = [1, 0, 0]
    g = [1, 0, 0]

    h = 1 / dim_in
    ax = 1
    axx = ax**2
    theta = 0.5 * np.pi
    pi1 = -0.5 * ax * ra * np.cos(theta)
    pi2 = 0.5 * ax * ra * np.sin(theta)

    fvec = np.zeros((n - 2, n - 2, n - 2))
    for j in range(1, n - 1):
        for i in range(1, n - 1):
            fvec[0, i - 1, j - 1] = (
                xvec[0, i, j] * -2 * (1 / h) ** 2
                + xvec[0, i + 1, j] * (1 / h) ** 2
                + xvec[0, i - 1, j] * (1 / h) ** 2
                + xvec[0, i, j] * -2 * axx * (1 / h) ** 2
                + xvec[0, i, j + 1] * axx * (1 / h) ** 2
                + xvec[0, i, j - 1] * ax * (1 / h) ** 2
                + xvec[1, i + 1, j] * -pi1 / (2 * h)
                + xvec[1, i - 1, j] * pi1 / (2 * h)
                + xvec[1, i, j + 1] * -pi2 / (2 * h)
                + xvec[1, i, j - 1] * pi2 / (2 * h)
            )
            fvec[1, i - 1, j - 1] = (
                xvec[2, i, j] * -2 * (1 / h) ** 2
                + xvec[2, i + 1, j] * (1 / h) ** 2
                + xvec[2, i - 1, j] * (1 / h) ** 2
                + xvec[2, i, j] * -2 * axx * (1 / h) ** 2
                + xvec[2, i, j + 1] * axx * (1 / h) ** 2
                + xvec[2, i, j - 1] * axx * (1 / h) ** 2
                + xvec[0, i, j] * axx * 0.25
            )
            fvec[2, i - 1, j - 1] = (
                xvec[1, i, j] * -2 * (1 / h) ** 2
                + xvec[1, i + 1, j] * (1 / h) ** 2
                + xvec[1, i - 1, j] * (1 / h) ** 2
                + xvec[1, i, j] * -2 * axx * (1 / h) ** 2
                + xvec[1, i, j + 1] * axx * (1 / h) ** 2
                + xvec[1, i, j - 1] * axx * (1 / h) ** 2
                - 0.25
                * ax
                * (1 / h) ** 2
                * (xvec[2, i, j + 1] - xvec[2, i, j - 1])
                * (xvec[1, i + 1, j] - xvec[1, i - 1, j])
                + 0.25
                * ax
                * (1 / h) ** 2
                * (xvec[2, i + 1, j] - xvec[2, i - 1, j])
                * (xvec[1, i, j + 1] - xvec[1, i, j - 1])
            )

    temp = np.zeros((n, n))
    temp[:, -1] = a[2]
    temp[:, 0] = b[2]
    temp[-1, 1:] = f[2]
    temp[0, :] = g[2]
    for k in range(n):
        temp[k, -1] += (
            xvec[1, k, -1] * 2 * a[0] * (1 / h)
            + xvec[1, k, -2] * -2 * a[0] * (1 / h)
            + xvec[1, k, -1] * a[1]
        )
        temp[k, 0] += (
            xvec[1, k, 1] * 2 * b[0] * (1 / h)
            + xvec[1, k, 0] * -2 * b[0] * (1 / h)
            + xvec[1, k, 0] * b[1]
        )
        temp[-1, k] += (
            xvec[1, -1, k] * 2 * f[0] * (1 / (ax * h))
            + xvec[1, -2, k] * -2 * f[0] * (1 / (ax * h))
            + xvec[1, -1, k] * f[1]
        )
        temp[0, k] += (
            xvec[1, 1, k] * 2 * g[0] * (1 / (ax * h))
            + xvec[1, 0, k] * -2 * g[0] * (1 / (ax * h))
            + xvec[1, 0, k] * g[1]
        )

    fvec = np.concatenate(
        [
            fvec.flatten(),
            np.concatenate((temp[0, :], temp[-1, :], temp[1:-1, 0], temp[1:-1, -1])),
        ]
    )

    temp = np.zeros((n, n))
    for k in range(n):
        temp[k, -1] += xvec[2, k, -1] * -2 * (1 / h) + xvec[2, k, -2] * 2 * (1 / h)
        temp[k, 0] += xvec[2, k, 1] * 2 * (1 / h) + xvec[2, k, 0] * -2 * (1 / h)
        temp[-1, k] += xvec[2, -1, k] * -2 * (1 / (ax * h)) + xvec[2, -2, k] * 2 * (
            1 / (ax * h)
        )
        temp[0, k] += xvec[2, 1, k] * 2 * (1 / (ax * h)) + xvec[2, 0, k] * -2 * (
            1 / (ax * h)
        )
    fvec = np.concatenate(
        [fvec, np.concatenate((temp[0, :], temp[-1, :], temp[1:-1, 0], temp[1:-1, -1]))]
    )

    return fvec


def oscigrne(x):
    dim_in = len(x)
    rho = 500

    fvec = np.zeros(dim_in)
    fvec[0] = 0.5 * x[0] - 0.5 - 4 * rho * (x[1] - 2.0 * x[0] ** 2 + 1.0) * x[0]
    fvec[1:-1] = (
        2 * rho * (x[1:-1] - 2.0 * x[:-2] ** 2 + 1.0)
        - 4 * rho * (x[2:] - 2.0 * x[:-2] ** 2 + 1.0) * x[1:-1]
    )
    fvec[-1] = 2 * rho * (x[-1] - 2.0 * x[-2] ** 2 + 1.0)

    return fvec


def spmsqrt(x):
    m = (len(x) + 2) // 3
    xmat = np.diag(x[2:-1:3], -1) + np.diag(x[::3], 0) + np.diag(x[1:-2:3], 1)

    b = np.zeros((m, m))
    b[0, 0] = np.sin(1)
    b[0, 1] = np.sin(4)
    k = 2
    for i in range(1, m - 1):
        k += 1
        b[i, i - 1] = np.sin(k**2)
        k += 1
        b[i, i] = np.sin(k**2)
        k += 1
        b[i, i + 1] = np.sin(k**2)
    k += 1
    b[-1, -2] = np.sin(k**2)
    k += 1
    b[-1, -1] = np.sin(k**2)

    fmat = np.zeros((m, m))
    fmat[0, 0] = xmat[0, 0] ** 2 + xmat[0, 1] * xmat[1, 0]
    fmat[0, 1] = xmat[0, 0] * xmat[0, 1] + xmat[0, 1] * xmat[1, 1]
    fmat[0, 2] = xmat[0, 1] * xmat[1, 2]

    fmat[1, 0] = xmat[1, 0] * xmat[0, 0] + xmat[1, 1] * xmat[1, 0]
    fmat[1, 1] = xmat[1, 0] * xmat[0, 1] + xmat[1, 1] ** 2 + xmat[1, 2] * xmat[2, 1]
    fmat[1, 2] = xmat[1, 1] * xmat[1, 2] + xmat[1, 2] * xmat[2, 2]
    fmat[1, 3] = xmat[1, 2] * xmat[2, 3]

    for i in range(2, m - 2):
        fmat[i, i - 2] = xmat[i, i - 1] * xmat[i - 1, i - 2]
        fmat[i, i - 1] = (
            xmat[i, i - 1] * xmat[i - 1, i - 1] + xmat[i, i] * xmat[i, i - 1]
        )
        fmat[i, i] = (
            xmat[i, i - 1] * xmat[i - 1, i]
            + xmat[i, i] ** 2
            + xmat[i, i + 1] * xmat[i + 1, i]
        )
        fmat[i, i + 1] = (
            xmat[i, i] * xmat[i, i + 1] + xmat[i, i + 1] * xmat[i + 1, i + 1]
        )
        fmat[i, i + 2] = xmat[i, i + 1] * xmat[i + 1, i + 2]

    fmat[-2, -4] = xmat[-2, -3] * xmat[-3, -4]
    fmat[-2, -3] = xmat[-2, -3] * xmat[-3, -3] + xmat[-2, -2] * xmat[-2, -3]
    fmat[-2, -2] = (
        xmat[-2, -3] * xmat[-3, -2] + xmat[-2, -2] ** 2 + xmat[-2, -1] * xmat[-1, -2]
    )
    fmat[-2, -1] = xmat[-2, -2] * xmat[-2, -1] + xmat[-2, -1] * xmat[-1, -1]

    fmat[-1, -3] = xmat[-1, -2] * xmat[-2, -3]
    fmat[-1, -2] = xmat[-1, -2] * xmat[-2, -2] + xmat[-1, -1] * xmat[-1, -2]
    fmat[-1, -1] = xmat[-1, -2] * xmat[-2, -1] + xmat[-1, -1] ** 2

    fmat[0, 0] -= b[0, 0] ** 2 + b[0, 1] * b[1, 0]
    for i in range(1, m - 1):
        fmat[i, i] -= (
            b[i, i] ** 2 + b[i - 1, i] * b[i, i - 1] + b[i + 1, i] * b[i, i + 1]
        )
    fmat[-1, -1] -= b[-1, -1] ** 2 + b[-2, -1] * b[-1, -2]
    for i in range(m - 1):
        fmat[i + 1, i] -= b[i + 1, i] * b[i, i] + b[i + 1, i + 1] * b[i + 1, i]
    for i in range(1, m):
        fmat[i - 1, i] -= b[i - 1, i] * b[i, i] + b[i - 1, i - 1] * b[i - 1, i]
    for i in range(1, m - 1):
        fmat[i + 1, i - 1] -= b[i + 1, i] * b[i, i - 1]
    for i in range(1, m - 1):
        fmat[i - 1, i + 1] -= b[i - 1, i] * b[i, i + 1]

    return fmat.flatten()


def semicon2(x):
    n = len(x) // 1
    ln = 9 * n // 10

    lambda_ = 0.2
    a = -0.00009
    b = 0.00001
    ua = 0.0
    ub = 700.0
    ca = 1e12
    cb = 1e13
    beta = 40.0

    h = (b - a) / (n + 1)
    lb = lambda_ * beta
    lua = lambda_ * ua
    lub = lambda_ * ub

    xvec = np.zeros(n + 2)
    xvec[0] = lua
    xvec[1:-1] = x
    xvec[-1] = lub

    fvec = np.zeros(n)
    for i in range(1, ln + 1):
        fvec[i - 1] = (
            xvec[i - 1]
            - 2 * xvec[i]
            + xvec[i + 1]
            + lambda_ * (h**2) * ca * np.exp(-lb * (xvec[i] - lua))
            - lambda_ * (h**2) * cb * np.exp(lb * (xvec[i] - lub))
            - lambda_ * (h**2) * ca
        )
    for i in range(ln + 1, n + 1):
        fvec[i - 1] = (
            xvec[i - 1]
            - 2 * xvec[i]
            + xvec[i + 1]
            - lambda_ * (h**2) * cb * np.exp(lb * (xvec[i] - lub))
            + lambda_ * (h**2) * ca * np.exp(-lb * (xvec[i] - lua))
            + lambda_ * (h**2) * cb
        )

    return fvec


# =====================================================================================


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
            + 0.5 * h**2 * (xvec[i - 1] + i * h + 1) ** 3
        )
    return fvec


def bratu_2d(x, alpha):
    x = x.reshape((int(np.sqrt(len(x))), int(np.sqrt(len(x)))))
    p = x.shape[0] + 2
    h = 1 / (p - 1)
    c = h**2 * alpha
    xvec = np.zeros((x.shape[0] + 2, x.shape[1] + 2))
    xvec[1 : x.shape[0] + 1, 1 : x.shape[1] + 1] = x
    fvec = np.zeros(x.shape)
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
    c = h**2 * alpha
    xvec = np.zeros((x.shape[0] + 2, x.shape[1] + 2, x.shape[2] + 2))
    xvec[1 : x.shape[0] + 1, 1 : x.shape[1] + 1, 1 : x.shape[2] + 1] = x
    fvec = np.zeros(x.shape)
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
    c = h**2 * alpha
    fvec = np.zeros(x.shape)
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
    cui1 = 1 / (h**2 * pem) + 1 / h
    cui = -1 / h - 2 / (h**2 * pem)
    ct1 = -h * peh
    cti1 = 1 / (h**2 * peh) + 1 / h
    cti = -beta - 1 / h - 2 / (h**2 * peh)
    fvec[0] = cu1 * x[0, 1] - x[0, 0] + h * pem
    fvec[1] = ct1 * x[1, 1] - x[1, 0] + h * peh
    for i in range(2, dim_in):
        fvec[i] = (
            -d * x[0, i - 1] * np.exp(gamma - gamma / x[1, i - 1])
            + (cui1) * x[0, i - 2]
            + cui * x[0, i - 1]
            + x[0, i] / (h**2 * pem)
        )
        fvec[dim_in - 2 + i] = (
            b * d * x[0, i - 1] * np.exp(gamma - gamma / x[1, i - 1])
            + beta * x[1, i - 1]
            + cti1 * x[1, i - 2]
            + cti * x[1, i - 1]
            + x[1, i] / (h**2 * peh)
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
    cti1 = 1 / (h**2 * pe) + 1 / h
    cti = -1 / h - 2 / (h**2 * pe)
    fvec[0] = ct1 * x[1] - x[0] + h * pe
    for i in range(2, dim_in):
        fvec[i - 1] = (
            d * (b + 1 - x[i - 1]) * np.exp(gamma - gamma / x[i - 1])
            + cti1 * x[i - 2]
            + cti * x[i - 1]
            + x[i] / (h**2 * pe)
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
    fvec = np.zeros(x.shape)
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
    fvec[-1] = (np.arange(1, dim_in + 1)[::-1] * x**2).sum() - 1
    return fvec


def vardimne(x):
    dim_in = len(x)
    fvec = np.zeros(dim_in + 2)
    fvec[:-2] = x - 1
    fvec[-2] = (np.arange(1, dim_in + 1) * (x - 1)).sum()
    fvec[-1] = ((np.arange(1, dim_in + 1) * (x - 1)).sum()) ** 2
    return fvec


def yatpsq_1(x, dim_in):
    xvec = x[: dim_in**2]
    xvec = xvec.reshape((dim_in, dim_in))
    yvec = x[dim_in**2 : dim_in**2 + dim_in]
    zvec = x[dim_in**2 + dim_in : dim_in**2 + 2 * dim_in]
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
    xvec = x[: dim_in**2]
    xvec = xvec.reshape((dim_in, dim_in))
    yvec = x[dim_in**2 : dim_in**2 + dim_in]
    zvec = x[dim_in**2 + dim_in : dim_in**2 + 2 * dim_in]
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
    return xmat.flatten().tolist()


def get_start_points_bdvalues(n, a=1):
    h = 1 / (n + 1)
    x = np.zeros(n)
    for i in range(n):
        x[i] = (i + 1) * h * ((i + 1) * h - 1)
    return (x * a).tolist()


def get_start_points_spmsqrt(m):
    b = np.zeros((m, m))
    b[0, 0] = np.sin(1)
    b[0, 1] = np.sin(4)
    k = 2
    for i in range(1, m - 1):
        k += 1
        b[i, i - 1] = np.sin(k**2)
        k += 1
        b[i, i] = np.sin(k**2)
        k += 1
        b[i, i + 1] = np.sin(k**2)
    k += 1
    b[-1, -2] = np.sin(k**2)
    k += 1
    b[-1, -1] = np.sin(k**2)

    x = np.zeros((m, m))
    x[:, :2] = 0.2 * b[:, :2]
    x[1:-1, :-2] = 0.2 * b[1:-1, :-2]
    x[1:-1, 1:-1] = 0.2 * b[1:-1, 1:-1]
    x[1:-1, 2:] = 0.2 * b[1:-1, 2:]
    x[-1, -2:] = 0.2 * b[-1, -2:]

    x_out = x[x != 0]

    return x_out.tolist()


solution_x_bdvalues = [
    -0.00501717,
    -0.00998312,
    -0.01489709,
    -0.01975833,
    -0.02456605,
    -0.02931945,
    -0.03401771,
    -0.03866001,
    -0.0432455,
    -0.04777331,
    -0.05224255,
    -0.05665232,
    -0.0610017,
    -0.06528975,
    -0.06951549,
    -0.07367795,
    -0.07777612,
    -0.08180898,
    -0.08577546,
    -0.08967451,
    -0.09350501,
    -0.09726585,
    -0.10095589,
    -0.10457394,
    -0.10811881,
    -0.11158927,
    -0.11498406,
    -0.1183019,
    -0.12154147,
    -0.12470143,
    -0.1277804,
    -0.13077697,
    -0.13368969,
    -0.1365171,
    -0.13925766,
    -0.14190984,
    -0.14447205,
    -0.14694265,
    -0.14931997,
    -0.15160232,
    -0.15378794,
    -0.15587503,
    -0.15786175,
    -0.15974621,
    -0.16152647,
    -0.16320056,
    -0.16476642,
    -0.16622197,
    -0.16756507,
    -0.1687935,
    -0.16990502,
    -0.17089728,
    -0.17176792,
    -0.17251447,
    -0.17313443,
    -0.1736252,
    -0.17398413,
    -0.17420848,
    -0.17429545,
    -0.17424214,
    -0.17404559,
    -0.17370274,
    -0.17321044,
    -0.17256546,
    -0.17176447,
    -0.17080403,
    -0.16968062,
    -0.16839059,
    -0.16693019,
    -0.16529558,
    -0.16348276,
    -0.16148763,
    -0.15930595,
    -0.15693338,
    -0.15436539,
    -0.15159735,
    -0.14862447,
    -0.14544178,
    -0.14204417,
    -0.13842638,
    -0.13458293,
    -0.13050819,
    -0.12619633,
    -0.12164132,
    -0.11683693,
    -0.1117767,
    -0.10645396,
    -0.10086179,
    -0.09499304,
    -0.0888403,
    -0.08239586,
    -0.07565179,
    -0.06859981,
    -0.06123136,
    -0.05353755,
    -0.04550917,
    -0.03713662,
    -0.02840998,
    -0.01931889,
    -0.00985262,
]

solution_x_bratu_2d = [
    0.07234633,
    0.11814877,
    0.1459185,
    0.15914495,
    0.15914495,
    0.1459185,
    0.11814877,
    0.07234633,
    0.11814877,
    0.19875438,
    0.24923944,
    0.27361473,
    0.27361473,
    0.24923944,
    0.19875438,
    0.11814877,
    0.1459185,
    0.24923944,
    0.31530971,
    0.34753593,
    0.34753593,
    0.31530971,
    0.24923944,
    0.1459185,
    0.15914495,
    0.27361473,
    0.34753593,
    0.3837784,
    0.3837784,
    0.34753593,
    0.27361473,
    0.15914495,
    0.15914495,
    0.27361473,
    0.34753593,
    0.3837784,
    0.3837784,
    0.34753593,
    0.27361473,
    0.15914495,
    0.1459185,
    0.24923944,
    0.31530971,
    0.34753593,
    0.34753593,
    0.31530971,
    0.24923944,
    0.1459185,
    0.11814877,
    0.19875438,
    0.24923944,
    0.27361473,
    0.27361473,
    0.24923944,
    0.19875438,
    0.11814877,
    0.07234633,
    0.11814877,
    0.1459185,
    0.15914495,
    0.15914495,
    0.1459185,
    0.11814877,
    0.07234633,
]

solution_x_bratu_2d_t = [
    0.1933024,
    0.33566336,
    0.43355494,
    0.48428111,
    0.48428111,
    0.43355494,
    0.33566336,
    0.1933024,
    0.33566336,
    0.59839893,
    0.78485783,
    0.88316504,
    0.88316504,
    0.78485783,
    0.59839893,
    0.33566336,
    0.43355494,
    0.78485783,
    1.04056365,
    1.17766089,
    1.17766089,
    1.04056365,
    0.78485783,
    0.43355494,
    0.48428111,
    0.88316504,
    1.17766089,
    1.33720634,
    1.33720634,
    1.17766089,
    0.88316504,
    0.48428111,
    0.48428111,
    0.88316504,
    1.17766089,
    1.33720634,
    1.33720634,
    1.17766089,
    0.88316504,
    0.48428111,
    0.43355494,
    0.78485783,
    1.04056365,
    1.17766089,
    1.17766089,
    1.04056365,
    0.78485783,
    0.43355494,
    0.33566336,
    0.59839893,
    0.78485783,
    0.88316504,
    0.88316504,
    0.78485783,
    0.59839893,
    0.33566336,
    0.1933024,
    0.33566336,
    0.43355494,
    0.48428111,
    0.48428111,
    0.43355494,
    0.33566336,
    0.1933024,
]


solution_x_bratu_3d = [
    0.24431369,
    0.27785366,
    0.19682155,
    0.27785366,
    0.32761664,
    0.23878408,
    0.19682155,
    0.23878408,
    0.18908409,
    0.27785366,
    0.32761664,
    0.23878408,
    0.32761664,
    0.39611483,
    0.29367471,
    0.23878408,
    0.29367471,
    0.2314289,
    0.19682155,
    0.23878408,
    0.18908409,
    0.23878408,
    0.29367471,
    0.2314289,
    0.18908409,
    0.2314289,
    0.18663237,
]

solution_x_broydn_3d = [
    -0.57076119,
    -0.68191013,
    -0.70248602,
    -0.70626058,
    -0.70695185,
    -0.70707842,
    -0.70710159,
    -0.70710583,
    -0.70710661,
    -0.70710675,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710678,
    -0.70710677,
    -0.70710675,
    -0.70710669,
    -0.70710654,
    -0.70710612,
    -0.70710498,
    -0.70710185,
    -0.70709332,
    -0.70707001,
    -0.70700634,
    -0.70683248,
    -0.70635771,
    -0.70506153,
    -0.7015252,
    -0.69189463,
    -0.66579752,
    -0.59603531,
    -0.4164123,
]


solution_x_cbratu_2d = [
    0.16692195,
    0.2529246,
    0.2796211,
    0.2529246,
    0.16692195,
    0.2529246,
    0.39198662,
    0.43607163,
    0.39198662,
    0.2529246,
    0.2796211,
    0.43607163,
    0.48598608,
    0.43607163,
    0.2796211,
    0.2529246,
    0.39198662,
    0.43607163,
    0.39198662,
    0.2529246,
    0.16692195,
    0.2529246,
    0.2796211,
    0.2529246,
    0.16692195,
]

solution_x_cbratu_2d = solution_x_cbratu_2d + [0] * 25

solution_x_broydn_bd = [
    -0.00000000e00,
    -0.00000000e00,
    -0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    -0.00000000e00,
    -0.00000000e00,
    -0.00000000e00,
    -0.00000000e00,
    -0.00000000e00,
    -0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    -0.00000000e00,
    -0.00000000e00,
    -0.00000000e00,
    -0.00000000e00,
    -0.00000000e00,
    -0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    -0.00000000e00,
    -0.00000000e00,
    -1.00000000e-10,
    -2.00000000e-10,
    -2.00000000e-10,
    -1.00000000e-10,
    4.00000000e-10,
    1.40000000e-09,
    3.00000000e-09,
    4.60000000e-09,
    4.70000000e-09,
    -1.00000000e-10,
    -1.43000000e-08,
    -4.23000000e-08,
    -8.25000000e-08,
    -1.17600000e-07,
    -1.00700000e-07,
    5.54000000e-08,
    4.68300000e-07,
    1.22420000e-06,
    2.22540000e-06,
    2.92120000e-06,
    1.96990000e-06,
    -2.95480000e-06,
    -1.47187000e-05,
    -3.48246000e-05,
    -5.90960000e-05,
    -7.05915000e-05,
    -3.15647000e-05,
    1.19032300e-04,
    4.48647900e-04,
    9.73368600e-04,
    1.53772420e-03,
    1.63273940e-03,
    2.14717600e-04,
    -4.30681910e-03,
    -1.36127680e-02,
    -2.81043041e-02,
    -4.39233903e-02,
    -4.73306566e-02,
    -8.45337580e-03,
    1.04321937e-01,
    2.74938066e-01,
    4.54655029e-01,
    6.22031184e-01,
    7.74293819e-01,
    9.11375485e-01,
    1.03226579e00,
    1.13635201e00,
    1.22498498e00,
    1.30019836e00,
    1.36374913e00,
    1.41711415e00,
    1.46168952e00,
    1.49882961e00,
    1.52972625e00,
    1.55537824e00,
    1.57663224e00,
    1.59421664e00,
    1.60875287e00,
    1.62076049e00,
    1.63067143e00,
    1.63884538e00,
    1.64557503e00,
    1.65102930e00,
    1.65461538e00,
    1.64858082e00,
    1.55247986e00,
]

solution_x_chemrctb = [
    0.05141945,
    0.05203209,
    0.05267567,
    0.05335175,
    0.05406197,
    0.05480806,
    0.05559182,
    0.05641517,
    0.05728009,
    0.05818869,
    0.05914317,
    0.06014585,
    0.06119916,
    0.06230566,
    0.06346804,
    0.06468911,
    0.06597184,
    0.06731935,
    0.0687349,
    0.07022193,
    0.07178406,
    0.07342507,
    0.07514894,
    0.07695987,
    0.07886224,
    0.08086068,
    0.08296004,
    0.08516541,
    0.08748215,
    0.08991588,
    0.09247251,
    0.09515826,
    0.09797963,
    0.10094348,
    0.104057,
    0.10732776,
    0.11076369,
    0.11437313,
    0.11816486,
    0.12214807,
    0.12633243,
    0.13072811,
    0.13534578,
    0.14019665,
    0.14529249,
    0.15064569,
    0.15626923,
    0.16217677,
    0.16838265,
    0.17490194,
    0.18175047,
    0.18894487,
    0.19650261,
    0.20444203,
    0.21278242,
    0.22154402,
    0.23074811,
    0.24041703,
    0.25057426,
    0.26124447,
    0.27245356,
    0.28422875,
    0.29659862,
    0.30959322,
    0.32324409,
    0.33758438,
    0.35264891,
    0.36847425,
    0.38509884,
    0.40256304,
    0.42090924,
    0.440182,
    0.46042812,
    0.48169675,
    0.50403953,
    0.52751072,
    0.55216731,
    0.57806915,
    0.60527915,
    0.63386338,
    0.66389124,
    0.69543563,
    0.72857315,
    0.76338427,
    0.79995348,
    0.83836951,
    0.87872535,
    0.921118,
    0.96564698,
    1.01240975,
    1.06148865,
    1.11291774,
    1.16660739,
    1.22219286,
    1.27878417,
    1.33468684,
    1.38740197,
    1.43443791,
    1.47507982,
    1.51238643,
]


solution_x_drcavty3 = [
    6.90580000e-06,
    -3.04054000e-05,
    -1.34595400e-04,
    -2.98301400e-04,
    -3.97564800e-04,
    -2.82615200e-04,
    -1.00791500e-04,
    1.18693000e-05,
    4.83418000e-05,
    3.86272000e-05,
    -3.61169000e-05,
    -1.56090300e-04,
    -3.44522400e-04,
    -5.22159200e-04,
    -5.02848100e-04,
    -1.96532500e-04,
    4.01814000e-05,
    1.66926300e-04,
    1.64254200e-04,
    9.75942000e-05,
    -1.53179900e-04,
    -3.26999400e-04,
    -5.35655500e-04,
    -5.17594800e-04,
    -2.45473400e-04,
    2.11398200e-04,
    3.85544900e-04,
    4.70161600e-04,
    3.19836200e-04,
    1.48115900e-04,
    -3.44263800e-04,
    -3.05706200e-04,
    -6.07866500e-04,
    -1.40639000e-04,
    3.54345200e-04,
    1.17906180e-03,
    1.27587890e-03,
    6.46781700e-04,
    2.97807400e-04,
    9.77706000e-05,
    -3.80139500e-04,
    5.85784900e-04,
    -4.34699000e-04,
    1.15040270e-03,
    2.93253490e-03,
    5.19921130e-03,
    3.26982700e-03,
    -1.15543100e-03,
    -3.31632400e-04,
    -9.65743000e-05,
    8.88011200e-04,
    4.55121760e-03,
    1.59257740e-03,
    4.02608170e-03,
    5.27395750e-03,
    -2.05009960e-03,
    -7.90681200e-04,
    1.29072190e-03,
    3.92764700e-04,
    -7.23810000e-05,
    1.10527329e-02,
    1.14289463e-02,
    1.01554380e-03,
    -4.10803130e-03,
    -1.39518580e-03,
    1.43680550e-03,
    -2.32410100e-04,
    3.02444440e-03,
    1.54672000e-04,
    -3.88632200e-04,
    4.87177720e-03,
    -1.17441400e-03,
    6.05647400e-04,
    -6.18932200e-04,
    -1.81334350e-03,
    5.15906690e-03,
    1.41277700e-04,
    6.31930020e-03,
    8.67670500e-04,
    1.30191470e-03,
    2.96133460e-03,
    3.64054300e-03,
    2.00721890e-03,
    5.74324870e-03,
    2.01317600e-04,
    5.60508670e-03,
    1.15676060e-03,
    8.20725550e-03,
    -9.88774500e-04,
    1.46054681e-02,
    4.93810300e-04,
    3.65006800e-04,
    6.47333900e-04,
    7.25182800e-04,
    1.71821900e-04,
    2.96466900e-04,
    -7.95212300e-04,
    1.80194150e-03,
    8.79835000e-04,
    1.17217338e-02,
]

solution_x_drcavty2 = [
    -8.30500000e-07,
    1.79025100e-04,
    4.69755400e-04,
    6.91706100e-04,
    7.63680500e-04,
    6.99211100e-04,
    5.59898000e-04,
    4.13496000e-04,
    2.89295400e-04,
    1.58674600e-04,
    1.44396300e-04,
    6.45348200e-04,
    1.19393250e-03,
    1.48581000e-03,
    1.49174680e-03,
    1.30666740e-03,
    1.04594130e-03,
    7.90114600e-04,
    5.54089300e-04,
    2.86541200e-04,
    4.86092200e-04,
    1.31996230e-03,
    1.90360630e-03,
    2.06459340e-03,
    1.95310610e-03,
    1.71807210e-03,
    1.41349750e-03,
    1.06405470e-03,
    6.90019300e-04,
    2.91278200e-04,
    1.00536400e-03,
    1.98412810e-03,
    2.39879170e-03,
    2.47713040e-03,
    2.38995890e-03,
    2.11023730e-03,
    1.58275540e-03,
    9.59023300e-04,
    3.65828300e-04,
    -2.52386000e-05,
    1.70639510e-03,
    2.47798200e-03,
    2.96272640e-03,
    3.23424450e-03,
    2.92194380e-03,
    1.83925430e-03,
    5.17510800e-04,
    -2.72294500e-04,
    -8.42981300e-04,
    -6.44882600e-04,
    3.47563010e-03,
    3.68554070e-03,
    4.85243520e-03,
    4.30556650e-03,
    2.59563830e-03,
    1.23414300e-04,
    -1.12148630e-03,
    -1.16433340e-03,
    -1.76218150e-03,
    -5.08449600e-04,
    1.02089465e-02,
    9.19876750e-03,
    7.38832940e-03,
    8.91347900e-04,
    -2.02918160e-03,
    -4.35306900e-04,
    1.74552680e-03,
    -3.82299000e-04,
    -1.89595900e-03,
    1.44318390e-03,
    8.39182720e-03,
    2.10036430e-03,
    -2.07708990e-03,
    -7.15986500e-04,
    -1.22269490e-03,
    2.85020860e-03,
    1.86361079e-02,
    -1.85665600e-04,
    5.77159200e-04,
    1.16139361e-02,
    7.12641800e-03,
    2.28174230e-03,
    -4.41730960e-03,
    1.19527564e-02,
    2.02136034e-02,
    1.78591365e-02,
    1.06707580e-01,
    3.00444810e-03,
    1.98001460e-02,
    1.37005246e-01,
    2.73846000e-03,
    7.47556450e-03,
    1.07964128e-02,
    1.81864591e-02,
    -1.19626975e-02,
    -5.17858661e-02,
    -2.97147410e-02,
    -5.84116800e-03,
    1.62672675e-01,
    9.02415668e-01,
]

solution_x_freurone = [
    12.26912153,
    -0.83186186,
    -1.50692279,
    -1.53467102,
    -1.53579843,
    -1.53584421,
    -1.53584607,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584616,
    -1.53584616,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584616,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584616,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584616,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584616,
    -1.53584615,
    -1.53584616,
    -1.53584615,
    -1.53584615,
    -1.53584615,
    -1.53584617,
    -1.53584665,
    -1.53585846,
    -1.53614941,
    -1.54330584,
]


solution_x_integreq = [
    -0.0049257,
    -0.00980164,
    -0.01462709,
    -0.01940127,
    -0.02412341,
    -0.0287927,
    -0.03340834,
    -0.03796949,
    -0.04247531,
    -0.04692493,
    -0.05131748,
    -0.05565204,
    -0.0599277,
    -0.06414352,
    -0.06829854,
    -0.07239178,
    -0.07642224,
    -0.08038888,
    -0.08429068,
    -0.08812655,
    -0.09189541,
    -0.09559613,
    -0.09922759,
    -0.1027886,
    -0.10627797,
    -0.10969449,
    -0.11303691,
    -0.11630394,
    -0.11949428,
    -0.1226066,
    -0.12563951,
    -0.12859164,
    -0.13146152,
    -0.13424771,
    -0.1369487,
    -0.13956294,
    -0.14208886,
    -0.14452484,
    -0.14686924,
    -0.14912034,
    -0.15127642,
    -0.15333569,
    -0.15529633,
    -0.15715646,
    -0.15891416,
    -0.16056747,
    -0.16211435,
    -0.16355275,
    -0.16488053,
    -0.16609551,
    -0.16719544,
    -0.16817804,
    -0.16904092,
    -0.16978168,
    -0.17039781,
    -0.17088676,
    -0.17124589,
    -0.17147249,
    -0.1715638,
    -0.17151695,
    -0.171329,
    -0.17099693,
    -0.17051763,
    -0.16988789,
    -0.16910442,
    -0.16816384,
    -0.16706265,
    -0.16579725,
    -0.16436394,
    -0.16275891,
    -0.16097822,
    -0.15901783,
    -0.15687354,
    -0.15454105,
    -0.15201592,
    -0.14929355,
    -0.14636922,
    -0.14323804,
    -0.13989497,
    -0.13633479,
    -0.13255212,
    -0.1285414,
    -0.12429688,
    -0.11981262,
    -0.11508247,
    -0.11010007,
    -0.10485884,
    -0.09935198,
    -0.09357242,
    -0.08751287,
    -0.08116576,
    -0.07452324,
    -0.06757719,
    -0.06031917,
    -0.05274045,
    -0.04483194,
    -0.03658422,
    -0.02798751,
    -0.01903165,
    -0.01008278,
]

solution_x_msqrta = [
    8.0e-10,
    -5.0e-10,
    -2.0e-10,
    -1.0e-10,
    1.0e-10,
    -5.0e-10,
    -1.0e-10,
    -4.0e-10,
    -2.0e-10,
    2.0e-10,
    2.1e-09,
    -1.0e-09,
    -7.0e-10,
    -4.0e-10,
    7.0e-10,
    -1.4e-09,
    -3.0e-10,
    -8.0e-10,
    4.0e-10,
    6.0e-10,
    2.2e-09,
    -1.6e-09,
    8.0e-10,
    -5.0e-10,
    -1.2e-09,
    -5.0e-10,
    -7.0e-10,
    -1.7e-09,
    -2.1e-09,
    2.0e-10,
    -1.2e-09,
    -3.0e-10,
    1.4e-09,
    -1.0e-10,
    -1.7e-09,
    1.4e-09,
    1.0e-10,
    -9.0e-10,
    -2.1e-09,
    -5.0e-10,
    -2.7e-09,
    6.0e-10,
    -1.0e-09,
    2.0e-10,
    1.2e-09,
    -4.0e-10,
    2.1e-09,
    8.0e-10,
    6.0e-10,
    4.0e-10,
    -2.9e-09,
    1.0e-09,
    -4.0e-10,
    3.0e-10,
    5.0e-10,
    6.0e-10,
    1.5e-09,
    9.0e-10,
    4.0e-10,
    -1.0e-10,
    2.9e-09,
    -1.4e-09,
    5.0e-10,
    -4.0e-10,
    -7.0e-10,
    -7.0e-10,
    -1.2e-09,
    -1.4e-09,
    -1.2e-09,
    2.0e-10,
    -1.3e-09,
    9.0e-10,
    -4.0e-10,
    3.0e-10,
    6.0e-10,
    2.0e-10,
    5.0e-10,
    1.1e-09,
    1.0e-09,
    -1.0e-10,
    2.5e-09,
    -1.1e-09,
    1.3e-09,
    -3.0e-10,
    -1.6e-09,
    3.0e-10,
    -1.6e-09,
    -1.4e-09,
    -1.7e-09,
    -3.0e-10,
    -3.1e-09,
    7.0e-10,
    4.0e-10,
    3.0e-10,
    -6.0e-10,
    1.4e-09,
    1.3e-09,
    2.0e-10,
    -8.0e-10,
    -5.0e-10,
]


solution_x_msqrtb = [
    1.2e-09,
    -3.0e-10,
    2.0e-10,
    -4.0e-10,
    -3.0e-10,
    6.0e-10,
    -2.0e-10,
    -8.0e-10,
    -6.0e-10,
    -0.0e00,
    8.0e-10,
    -1.6e-09,
    -1.3e-09,
    -5.0e-10,
    -0.0e00,
    -2.0e-09,
    3.0e-10,
    -9.0e-10,
    -1.0e-10,
    1.0e-09,
    2.3e-09,
    -0.0e00,
    1.3e-09,
    -6.0e-10,
    -1.3e-09,
    1.1e-09,
    -1.0e-09,
    -1.2e-09,
    -1.9e-09,
    0.0e00,
    -1.1e-09,
    1.9e-09,
    1.9e-09,
    7.0e-10,
    -5.0e-10,
    1.0e-09,
    -8.0e-10,
    1.3e-09,
    -4.0e-10,
    -6.0e-10,
    -3.1e-09,
    -1.0e-10,
    -2.2e-09,
    7.0e-10,
    2.2e-09,
    -7.0e-10,
    1.8e-09,
    1.3e-09,
    3.1e-09,
    -4.0e-10,
    -1.6e-09,
    1.7e-09,
    1.3e-09,
    8.0e-10,
    -0.0e00,
    1.0e-09,
    -3.0e-10,
    1.5e-09,
    3.0e-10,
    -7.0e-10,
    4.2e-09,
    -1.0e-09,
    5.0e-10,
    -1.4e-09,
    -9.0e-10,
    2.7e-09,
    -7.0e-10,
    -2.7e-09,
    -2.0e-09,
    -3.0e-10,
    -8.0e-10,
    5.0e-10,
    -4.0e-10,
    2.0e-10,
    1.2e-09,
    2.2e-09,
    7.0e-10,
    3.0e-10,
    1.3e-09,
    -1.1e-09,
    2.7e-09,
    4.0e-10,
    2.6e-09,
    -5.0e-10,
    -2.8e-09,
    -6.0e-10,
    -2.2e-09,
    -9.0e-10,
    -3.6e-09,
    8.0e-10,
    -2.4e-09,
    3.1e-09,
    2.9e-09,
    1.4e-09,
    -8.0e-10,
    1.1e-09,
    -1.1e-09,
    2.5e-09,
    -4.0e-10,
    -9.0e-10,
]

solution_x_penalty2 = [
    1.00248452e-01,
    -1.60000000e-09,
    -1.40000000e-09,
    -1.50000000e-09,
    -1.00000000e-09,
    -9.00000000e-10,
    -8.00000000e-10,
    -6.00000000e-10,
    -1.20000000e-09,
    -8.00000000e-10,
    -7.00000000e-10,
    -6.00000000e-10,
    -4.00000000e-10,
    -4.00000000e-10,
    -2.00000000e-10,
    -2.00000000e-10,
    -3.00000000e-10,
    -3.00000000e-10,
    -1.00000000e-10,
    -0.00000000e00,
    0.00000000e00,
    1.00000000e-10,
    1.00000000e-10,
    0.00000000e00,
    0.00000000e00,
    1.00000000e-10,
    1.00000000e-10,
    1.00000000e-10,
    -1.00000000e-10,
    -4.00000000e-10,
    -4.00000000e-10,
    -1.00000000e-09,
    -7.00000000e-10,
    -6.00000000e-10,
    -8.00000000e-10,
    -8.00000000e-10,
    -1.20000000e-09,
    -9.00000000e-10,
    -6.00000000e-10,
    -4.00000000e-10,
    -1.00000000e-10,
    2.00000000e-10,
    5.00000000e-10,
    1.00000000e-09,
    5.40000000e-09,
    2.20000000e-09,
    7.20000000e-09,
    8.50000000e-09,
    9.60000000e-09,
    1.00000000e-08,
    1.14000000e-08,
    5.80000000e-09,
    1.62000000e-08,
    1.82000000e-08,
    2.15000000e-08,
    2.35000000e-08,
    2.57000000e-08,
    2.90000000e-08,
    3.19000000e-08,
    3.09000000e-08,
    4.16000000e-08,
    4.66000000e-08,
    5.31000000e-08,
    6.20000000e-08,
    6.85000000e-08,
    7.99000000e-08,
    9.37000000e-08,
    1.08300000e-07,
    1.24300000e-07,
    1.46700000e-07,
    1.64900000e-07,
    1.87800000e-07,
    2.08200000e-07,
    2.40000000e-07,
    2.78900000e-07,
    3.24200000e-07,
    3.77700000e-07,
    4.22400000e-07,
    4.89200000e-07,
    5.68200000e-07,
    6.64900000e-07,
    7.72800000e-07,
    8.93500000e-07,
    1.05180000e-06,
    1.24180000e-06,
    1.46270000e-06,
    1.73540000e-06,
    2.06050000e-06,
    2.45940000e-06,
    2.96690000e-06,
    3.61230000e-06,
    4.44100000e-06,
    5.50230000e-06,
    6.96620000e-06,
    8.98120000e-06,
    1.18847000e-05,
    1.64570000e-05,
    2.42465000e-05,
    4.02062000e-05,
    4.21655000e-05,
]


solution_x_watson = [
    -0.00000000e00,
    1.00000000e00,
    -9.80000000e-09,
    3.33333416e-01,
    3.52440000e-06,
    1.33262416e-01,
    5.10786700e-04,
    5.27159393e-02,
    -4.88557280e-03,
    7.04208489e-02,
    -1.76310368e-01,
    3.59652497e-01,
    -3.34930081e-01,
    -1.06954704e-01,
    7.10806973e-01,
    -7.44769987e-01,
    1.39770112e-01,
    2.23466491e-01,
    -2.30955205e-02,
    -4.13852010e-03,
    -2.40655722e-01,
    2.11825083e-01,
    -7.89005230e-02,
    2.52472539e-02,
    5.33065585e-02,
    1.99652115e-01,
    -5.38278039e-01,
    2.27228847e-01,
    2.79127878e-01,
    -2.82103374e-01,
    7.20994063e-02,
]


solution_x_yatpsq_1 = [7.06817436] * 100 + [4011.71977601] * 10 + [-4045.83698215] * 10

solution_x_yatpsq_2 = [0.0500104219] * 100 + [31.74567612] * 10 + [-32.22096803] * 10

solution_x_arglble = [
    1.47780425,
    1.95560851,
    3.92938553,
    2.91121702,
    8.01627094,
    6.85877107,
    15.21786782,
    4.82243403,
    6.00860058,
    15.03254187,
    23.28986922,
    12.71754213,
    -26.58525249,
    29.43573564,
    15.37597436,
    8.64486806,
    -29.5319652,
    11.01720116,
    -42.30702246,
    29.06508375,
    -6.28404714,
    45.57973843,
    -20.97480585,
    24.43508427,
    -31.03091276,
    -54.17050498,
    -28.61801666,
    57.87147127,
    51.17010631,
    29.75194872,
    -20.71188224,
    16.28973612,
    150.20741347,
    -60.0639304,
    -83.05993998,
    21.03440232,
    91.61798856,
    -85.61404492,
    18.37256156,
    57.1301675,
    37.08398118,
    -13.56809428,
    121.49145482,
    90.15947687,
    12.46203868,
    -42.9496117,
    -29.66369732,
    47.87016853,
    142.29920239,
    -63.06182553,
    3.443138,
    -109.34100996,
    -194.46227627,
    -58.23603332,
    74.84482245,
    114.74294254,
    -32.27461444,
    101.34021261,
    48.55025892,
    58.50389743,
    66.92509996,
    -42.42376449,
    -96.64369487,
    31.57947224,
    -83.36761354,
    299.41482695,
    -67.29433552,
    -121.1278608,
    -35.15915764,
    -167.11987996,
    -88.92036545,
    41.06880464,
    242.22658666,
    182.23597711,
    72.43687845,
    -172.22808985,
    156.51103949,
    35.74512311,
    -129.889251,
    113.260335,
    61.76069205,
    73.16796235,
    -226.09539309,
    -28.13618857,
    -140.01767085,
    241.98290964,
    -188.31315279,
    179.31895373,
    -102.19596978,
    23.92407737,
    -20.6453831,
    -86.8992234,
    -260.60293104,
    -60.32739463,
    -187.46576175,
    94.74033707,
    -26.40609501,
    283.59840478,
    -81.00626161,
    -6.23327543,
]

CARTIS_ROBERTS_PROBLEMS = {
    "argtrig": {
        "criterion": argtrig,
        "start_x": [1 / 100] * 100,
        "solution_x": [0] * 100,
        "start_criterion": 32.99641,
        "solution_criterion": 0,
    },
    "artif": {
        "criterion": artif,
        "start_x": [1] * 100,
        "solution_x": None,
        "start_criterion": 36.59115,
        "solution_criterion": 0,
    },
    "arwhdne": {
        "criterion": arwhdne,
        "start_x": [1] * 100,
        "solution_x": [0.706011] * 99 + [0],
        "start_criterion": 495,
        "solution_criterion": 27.66203,
    },
    "bdvalues": {
        "criterion": bdvalues,
        "start_x": get_start_points_bdvalues(100, 1000),
        "solution_x": solution_x_bdvalues,
        "start_criterion": 1.943417e7,
        "solution_criterion": 0,
    },
    "bratu_2d": {
        "criterion": partial(bratu_2d, alpha=4),
        "start_x": [0] * 64,
        "solution_x": solution_x_bratu_2d,
        "start_criterion": 0.1560738,
        "solution_criterion": 0,
    },
    "bratu_2d_t": {
        "criterion": partial(bratu_2d, alpha=6.80812),
        "start_x": [0] * 64,
        "solution_x": solution_x_bratu_2d_t,
        "start_criterion": 0.4521311,
        "solution_criterion": 1.8534736e-05,
    },
    "bratu_3d": {
        "criterion": partial(bratu_3d, alpha=6.80812),
        "start_x": [0] * 27,
        "solution_x": solution_x_bratu_3d,
        "start_criterion": 4.888529,
        "solution_criterion": 0,
    },
    "brownale": {
        "criterion": brown_almost_linear,
        "start_x": [0.5] * 100,
        "solution_x": [1] * 100,
        "start_criterion": 2.524757e5,
        "solution_criterion": 0,
    },
    "broydn_3d": {
        "criterion": broydn_3d,
        "start_x": [-1] * 100,
        "solution_x": solution_x_broydn_3d,
        "start_criterion": 111,
        "solution_criterion": 0,
    },
    "cbratu_2d": {
        "criterion": cbratu_2d,
        "start_x": [0] * (2 * 5 * 5),
        "solution_x": solution_x_cbratu_2d,
        "start_criterion": 0.4822531,
        "solution_criterion": 0,
    },
    "broydn_bd": {
        "criterion": broydn_bd,
        "start_x": [1] * 100,
        "solution_x": solution_x_broydn_bd,
        "start_criterion": 2404,
        "solution_criterion": 0,
    },
    "chandheq": {
        "criterion": chandheq,
        "start_x": (np.arange(1, 101) / 100).tolist(),
        "solution_x": None,
        "start_criterion": 6.923365,
        "solution_criterion": 0,
    },
    "chemrcta": {
        "criterion": chemrcta,
        "start_x": [1] * 100,
        "solution_x": None,
        "start_criterion": 3.0935,
        "solution_criterion": 0,
        "lower_bounds": np.concatenate([np.zeros(50), 1e-6 * np.ones(50)]),
    },
    "chemrctb": {
        "criterion": chemrctb,
        "start_x": [1] * 100,
        "solution_x": solution_x_chemrctb,
        "start_criterion": 1.446513,
        "solution_criterion": 1.404424e-3,
        "lower_bounds": 1e-6 * np.ones(100),
    },
    "chnrsbne": {
        "criterion": chnrsbne,
        "start_x": [-1] * 50,
        "solution_x": [1] * 50,
        "start_criterion": 7635.84,
        "solution_criterion": 0,
    },
    "drcavty1": {
        "criterion": partial(drcavty, r=500),
        "start_x": [0] * 100,
        "solution_x": None,
        "start_criterion": 0.4513889,
        "solution_criterion": 0,
    },
    "drcavty2": {
        "criterion": partial(drcavty, r=1000),
        "start_x": [0] * 100,
        "solution_x": solution_x_drcavty2,
        "start_criterion": 0.4513889,
        "solution_criterion": 3.988378e-4,
    },
    "drcavty3": {
        "criterion": partial(drcavty, r=4500),
        "start_x": [0] * 100,
        "solution_x": solution_x_drcavty3,
        "start_criterion": 0.4513889,
        "solution_criterion": 0,
    },
    "freurone": {
        "criterion": freurone,
        "start_x": [0.5, -2] + [0] * 98,
        "solution_x": solution_x_freurone,
        "start_criterion": 9.95565e4,
        "solution_criterion": 1.196458e4,
    },
    "hatfldg": {
        "criterion": hatfldg,
        "start_x": [1] * 25,
        "solution_x": [0] * 11 + [-1, 1] + [0] * 12,
        "start_criterion": 27,
        "solution_criterion": 0,
    },
    "integreq": {
        "criterion": integreq,
        "start_x": (np.arange(1, 101) / 101 * (np.arange(1, 101) / 101 - 1)).tolist(),
        "solution_x": solution_x_integreq,
        "start_criterion": 0.5730503,
        "solution_criterion": 0,
    },
    "msqrta": {
        "criterion": msqrta,
        "start_x": get_start_points_msqrta(10),
        "solution_x": solution_x_msqrta,
        "start_criterion": 212.7162,
        "solution_criterion": 0,
    },
    "msqrtb": {
        "criterion": msqrta,
        "start_x": get_start_points_msqrta(10, flag=2),
        "solution_x": solution_x_msqrtb,
        "start_criterion": 205.0753,
        "solution_criterion": 0,
    },
    "penalty_1": {
        "criterion": penalty_1,
        "start_x": list(range(1, 101)),
        "solution_x": None,
        "start_criterion": 1.144806e11,
        "solution_criterion": 9.025000e-9,
    },
    "penalty_2": {
        "criterion": penalty_2,
        "start_x": [0.5] * 100,
        "solution_x": solution_x_penalty2,
        "start_criterion": 1.591383e6,
        "solution_criterion": 0.9809377,
    },
    "vardimne": {
        "criterion": vardimne,
        "start_x": [1 - i / 100 for i in range(1, 101)],
        "solution_x": [1] * 100,
        "start_criterion": 1.310584e14,
        "solution_criterion": 0,
    },
    "watsonne": {
        "criterion": watson,
        "start_x": [0] * 31,
        "solution_x": solution_x_watson,
        "start_criterion": 30,
        "solution_criterion": 0,
    },
    "yatpsq_1": {
        "criterion": partial(yatpsq_1, dim_in=10),
        "start_x": [6] * 100 + [0] * 20,
        "solution_x": solution_x_yatpsq_1,
        "start_criterion": 2.073643e6,
        "solution_criterion": 0,
    },
    "yatpsq_2": {
        "criterion": partial(yatpsq_2, dim_in=10),
        "start_x": [10] * 100 + [0] * 20,
        "solution_x": solution_x_yatpsq_2,
        "start_criterion": 1.831687e5,
        "solution_criterion": 0,
    },
    "arglale": {
        # arglale is the same as linear_full_rank with specific settings
        "criterion": partial(linear_full_rank, dim_out=400),
        "start_x": [1] * 100,
        "solution_x": [-0.99999952] * 100,
        "start_criterion": 700,
        "solution_criterion": 300,
    },
    "arglble": {
        # arglble is the same as linear_rank_one with specific settings
        "criterion": partial(linear_rank_one, dim_out=400),
        "start_x": [1] * 100,
        "solution_x": solution_x_arglble,
        "start_criterion": 5.460944e14,
        "solution_criterion": 99.62547,
    },
    "luksan11": {
        "criterion": luksan11,
        "start_x": [-0.8] * 100,
        "solution_x": [np.nan] * 100,
        "start_criterion": 626.0640,
        "solution_criterion": 0,
    },
    "luksan12": {
        "criterion": luksan12,
        "start_x": [-1] * 98,
        "solution_x": [np.nan] * 98,
        "start_criterion": 3.2160e4,
        "solution_criterion": 4292.197,
    },
    "luksan13": {
        "criterion": luksan13,
        "start_x": [-1] * 98,
        "solution_x": [np.nan] * 98,
        "start_criterion": 6.4352e4,
        "solution_criterion": 2.51886e4,
    },
    "luksan14": {
        "criterion": luksan14,
        "start_x": [-1] * 98,
        "solution_x": [np.nan] * 98,
        "start_criterion": 2.6880e4,
        "solution_criterion": 123.9235,
    },
    "luksan15": {
        "criterion": luksan15,
        "start_x": [-0.8, 1.2, -1.2, 0.8] * 25,
        "solution_x": [np.nan] * 100,
        "start_criterion": 2.701585e4,
        "solution_criterion": 3.569697,
    },
    "luksan16": {
        "criterion": luksan16,
        "start_x": [-0.8, 1.2, -1.2, 0.8] * 25,
        "solution_x": [np.nan] * 100,
        "start_criterion": 1.306848e4,
        "solution_criterion": 3.569697,
    },
    "luksan17": {
        "criterion": luksan17,
        "start_x": [-0.8, 1.2, -1.2, 0.8] * 25,
        "solution_x": [np.nan] * 100,
        "start_criterion": 1.687370e6,
        "solution_criterion": 0.4931613,
    },
    "luksan21": {
        "criterion": luksan21,
        "start_x": [ih * (ih - 1) for ih in np.arange(1, 101) * (1 / 101)],
        "solution_x": [np.nan] * 100,
        "start_criterion": 99.98751,
        "solution_criterion": 0,
    },
    "luksan22": {
        "criterion": luksan22,
        "start_x": [-1.2 if i % 2 == 0 else 1 for i in range(100)],
        "solution_x": [np.nan] * 100,
        "start_criterion": 2.487686e4,
        "solution_criterion": 872.9230,
    },
    "morebvne": {
        "criterion": morebvne,
        "start_x": [t * (t - 1) for t in np.arange(1, 101) * (1 / 101)],
        "solution_x": [np.nan] * 100,
        "start_criterion": 3.633100e-4,
        "solution_criterion": 0,
    },
    "flosp2hh": {
        "criterion": partial(flosp2, const=[[1, 0, -1], [1, 0, -1]], ra=1e7, dim_in=2),
        "start_x": [0] * 59,
        "solution_x": [np.nan] * 59,
        "start_criterion": 519,
        "solution_criterion": 1 / 3,
    },
    "flosp2hl": {
        "criterion": partial(flosp2, const=[[1, 0, -1], [1, 0, -1]], ra=1e3, dim_in=2),
        "start_x": [0] * 59,
        "solution_x": [np.nan] * 59,
        "start_criterion": 519,
        "solution_criterion": 1 / 3,
    },
    "flosp2hm": {
        "criterion": partial(flosp2, const=[[1, 0, -1], [1, 0, -1]], ra=1e5, dim_in=2),
        "start_x": [0] * 59,
        "solution_x": [np.nan] * 59,
        "start_criterion": 519,
        "solution_criterion": 1 / 3,
    },
    "flosp2th": {
        "criterion": partial(flosp2, const=[[0, 1, 0], [0, 1, 1]], ra=1e7, dim_in=2),
        "start_x": [0] * 59,
        "solution_x": [np.nan] * 59,
        "start_criterion": 516,
        "solution_criterion": 0,
    },
    "flosp2tl": {
        "criterion": partial(flosp2, const=[[0, 1, 0], [0, 1, 1]], ra=1e3, dim_in=2),
        "start_x": [0] * 59,
        "solution_x": [np.nan] * 59,
        "start_criterion": 516,
        "solution_criterion": 0,
    },
    "flosp2tm": {
        "criterion": partial(flosp2, const=[[0, 1, 0], [0, 1, 1]], ra=1e5, dim_in=2),
        "start_x": [0] * 59,
        "solution_x": [np.nan] * 59,
        "start_criterion": 516,
        "solution_criterion": 0,
    },
    "oscigrne": {
        "criterion": oscigrne,
        "start_x": [-2] + [1] * 99,
        "solution_x": [np.nan] * 100,
        "start_criterion": 6.120720e8,
        "solution_criterion": 0,
    },
    "spmsqrt": {
        "criterion": spmsqrt,
        "start_x": get_start_points_spmsqrt(34),
        "solution_x": [np.nan] * 100,
        "start_criterion": 74.33542,
        "solution_criterion": 0,
    },
    "semicn2u": {
        "criterion": semicon2,
        "start_x": [0] * 100,
        "solution_x": [np.nan] * 100,
        "start_criterion": 2.025037e4,
        "solution_criterion": 0,
    },
    "semicon2": {
        "criterion": semicon2,
        "start_x": [0] * 100,
        "solution_x": [np.nan] * 100,
        "start_criterion": 2.025037e4,
        "solution_criterion": 0,
        "lower_bounds": -5 * np.ones(100),
        "upper_bounds": 0.2 * 700 * np.ones(100),
    },
}
