"""Define the More-Wild Benchmark Set.

This benchmark set is contains 53 test cases for nonlinear least squares solvers.
The test cases are built out of 22 functions, originally derived from the CUTEr
Problems. It was used to benchmark all modern model based non-linear derivative
free least squares solvers (e.g. POUNDERS, DFOGN, DFOLS).

The parameter dimensions are quite small, varying between 2 and 12.

The benchmark set was first described In More and Wild, 2009. Fortran and Matlab Code
is available here. We use the following sources of information to construct the
benchmark set:

- https://www.mcs.anl.gov/~more/dfo/fortran/dfovec.f for the function implementation
- https://www.mcs.anl.gov/~more/dfo/fortran/dfoxs.f for the base starting points
- https://www.mcs.anl.gov/~more/dfo/fortran/dfo.dat for:
    - The mapping test cases to criterion functions (column 1)
    - The dimensionalities of parameter vectors (column 2)
    - The dimensionalities of the output (column 3)
    - Whether the base start vector is multiplied by a factor of ten or not (column 4).

"""
from functools import partial

import numpy as np


def linear_full_rank(x, dim_out):
    temp = 2 * x.sum() / dim_out + 1
    out = np.full(dim_out, -temp)
    out[: len(x)] += x
    return out


def linear_rank_one(x, dim_out):
    fvec = np.arange(1, dim_out + 1).astype("float") * sum - 1.0
    return fvec


def linear_rank_one_zero_columns_rows(x, dim_out):
    dim_in = len(x)
    sumry = sum(np.arange(2, dim_in) * x[1:-1])
    fvec = np.arange(dim_out) * float(sumry) - 1.0
    fvec[-1] = -1.0
    return fvec


def rosenbrock(x):
    fvec = np.zeros[1]
    fvec[0] = 10 * (x[1] - x[0] ** 2)
    fvec[1] = 1.0 - x[0]
    return fvec


def helical_valley(x):
    temp = 8 * np.arctan(1.0)
    temp1 = np.sign(x[1]) * 0.25
    if x[0] > 0:
        temp1 = np.arctan(x[1] / x[0]) / temp
    elif x[0] < 0:
        temp1 = np.arctan(x[1] / x[0]) / temp + 0.5
    temp2 = np.sqrt(x[0] ** 2 + x[1] ** 2)
    fvec = np.zeros[2]
    fvec[0] = 10 * (x[2] - 10 * temp1)
    fvec[1] = 10 * (temp2 - 1.0)
    fvec[2] = x[2]
    return fvec


def powell_singular(x):
    fvec = np.zeros[3]
    fvec[0] = x[0] + 10 * x[1]
    fvec[1] = np.sqrt(5.0) * (x[2] - x[3])
    fvec[2] = (x[1] - 2 * x[2]) ** 2
    fvec[3] = np.sqrt(10.0) * (x[0] - x[3]) ** 2
    return fvec


def freudenstein_roth(x):
    fvec = np.zeros(2)
    fvec[0] = -13 + x[0] + ((5 - x[1]) * x[1] - 2) * x[1]
    fvec[1] = -29 + x[0] + ((1.0 + x[1]) * x[1] - 14) * x[1]
    return fvec


def bard(x, y):
    fvec = np.zeros(len(y))
    for i in range(1, round(len(y) / 2) + 1):
        temp1 = float(i)
        temp2 = float(len(y) + 1 - i)
        fvec[i - 1] = y[i - 1] - (x[0] + temp1 / (x[1] * temp2 + x[2] * temp1))
    for i in range(round(len(y) / 2) + 1, len(y) + 1):
        temp1 = float(i)
        temp2 = float(len(y) + 1 - i)
        fvec[i - 1] = y[i - 1] - (x[0] + temp1 / (x[1] * temp2 + x[2] * temp2))
    return fvec


def kowalik_osborne(x, y1, y2):
    temp1 = y1 * (y1 + x[1])
    temp2 = y1 * (y1 + x[2]) + x[3]
    fvec = y2 - x[0] * temp1 / temp2
    return fvec


def meyer(x, y):
    temp = 5 * np.arange(1, len(y) + 1) + 45 + x[2]
    temp1 = x[1] / temp
    temp2 = np.exp(temp1)
    fvec = x[0] * temp2 - y
    return fvec


def watson(x):
    dim_in = len(x)
    fvec = np.zeros(31)
    for i in range(1, 30):
        temp = float(i) / 29
        sum_1 = sum(np.arange(1, dim_in) * temp ** np.arange(dim_in - 1) * x[1:])
        sum_2 = sum(temp ** np.arange(dim_in) * x)
        fvec[i - 1] = sum_1 - sum_2 ** 2 - 1.0
    fvec[29] = x[0]
    fvec[30] = x[1] - x[0] ** 2 - 1.0
    return fvec


def box_3d(x, dim_out):
    fvec = np.zeros(dim_out)
    for i in range(1, dim_out + 1):
        fvec[i - 1] = (
            np.exp(-float(i) / 10 * x[0])
            - np.exp(-float(i) / 10 * x[1])
            + (np.exp(-float(i)) - np.exp(-float(i) / 10)) * x[2]
        )
    return fvec


def jennrich_sampson(x, dim_out):
    fvec = (
        2 * (1.0 + np.range(1, dim_out + 1))
        - np.exp(np.range(1, dim_out + 1) * x[0])
        - np.exp(np.arange(1, dim_out + 1) * x[1])
    )
    return fvec


def brown_dennis(x, dim_out):
    fvec = np.zeros(dim_out)
    for i in range(1, dim_out + 1):
        temp = i / 5
        temp_1 = x[0] + temp * x[1] - np.exp(temp)
        temp_2 = x[2] + np.sin(temp) * x[3] - np.cos(temp)
        fvec[i - 1] = temp_1 ** 2 + temp_2 ** 2
    return fvec


def chebyquad(x, dim_out):
    fvec = np.zeros(dim_out)
    dim_in = len(x)
    for i in range(1, dim_in + 1):
        temp_1 = 1.0
        temp_2 = 2 * x[i - 1] - 1.0
        temp = 2 * temp_2
        for j in range(dim_out):
            fvec[j] = fvec[j] + temp_2
            temp_3 = temp * temp_2 - temp_1
            temp_1 = temp_2
            temp_2 = temp_3
        for i in range(1, dim_out + 1):
            fvec[i - 1] = fvec[i - 1] / dim_in
            if i % 2 == 0:
                fvec[i - 1] = fvec[i - 1] + 1 / (i ** 2 - 1.0)
    return fvec


def brown_almost_linear(x):
    dim_in = len(x)
    sumry = -float(dim_in + 1)
    prodct = 1.0
    for i in range(dim_in):
        sumry += x[i]
        prodct *= x[i]
    fvec = x + sumry
    fvec[dim_in - 1] = prodct - 1.0
    return fvec


def osborne_one(x, y):
    temp = 10 * np.arange(33).astype("float")
    temp_1 = np.exp(-x[3] * temp)
    temp_2 = np.exp(-x[4] * temp)
    fvec = y - (x[0] + x[1] * temp_1 + x[2] * temp_2)
    return fvec


def osborne_two(x, y):
    temp = np.arange(len(y)).astype("float") / 10
    temp_1 = np.exp(-x[4] * temp)
    temp_2 = np.exp(-x[5] * (temp - x[8]) ** 2)
    temp_3 = np.exp(-x[6] * (temp - x[9]) ** 2)
    temp_4 = np.exp(-x[7] * (temp - x[10]) ** 2)
    fvec = y - (x[0] * temp_1 + x[1] * temp_2 + x[2] * temp_3 + x[3] * temp_4)
    return fvec


def bdqrtic(x):
    dim_in = len(x)
    fvec = np.zeros(2 * (dim_in - 4))
    for i in range(dim_in - 4):
        fvec[i] = -4 * x[i] + 3
        fvec[dim_in - 4 + i] = (
            x[i] ** 2
            + 2 * x[i + 1] ** 2
            + 3 * x[i + 2] ** 2
            + 4 * x[i + 3] ** 2
            + 5 * x[dim_in - 1] ** 2
        )
    return fvec


def cube(x):
    dim_in = len(x)
    fvec = np.zeros(dim_in)
    fvec[0] = x[0] - 1.0
    for i in range(1, dim_in):
        fvec[i] = 10 * (x[i] - x[i - 1] ** 3)
    return fvec


def mancino(x):
    dim_in = len(x)
    fvec = np.zeros(dim_in)
    for i in range(dim_in):
        sumry = 0
        for j in range(dim_in):
            temp = np.sqrt(x[i] ** 2) + float(i + 1) / float(j + 1)
            sumry += temp * ((np.sin(np.log(temp))) ** 5 + (np.cos(np.log(temp))) ** 5)
        fvec[i] = 1400 * x[i] + (i + 1 - 50) ** 3 + sumry
    return fvec


def heart_eight(x, y):
    dim_y = len(y)
    fvec = np.zeros(dim_y)
    fvec[0] = x[0] + x[1] - y[0]
    fvec[1] = x[2] + x[3] - y[1]
    fvec[2] = x[4] * x[0] + x[5] * x[1] - x[6] * x[2] - x[7] * x[3] - y[2]
    fvec[3] = x[6] * x[0] + x[7] * x[1] + x[4] * x[2] + x[5] * x[3] - y[3]
    fvec[4] = (
        x[0] * (x[4] ** 2 - x[6] ** 2)
        - 2 * x[2] * x[4] * x[6]
        + x[1] * (x[5] ** 2 - x[7] ** 2)
        - 2 * x[3] * x[5] * x[7]
        - y[4]
    )
    fvec[5] = (
        x[2] * (x[4] ** 2 - x[6] ** 2)
        + 2 * x[0] * x[4] * x[6]
        + x[3] * (x[5] ** 2 - x[7] ** 2)
        + 2 * x[1] * x[5] * x[7]
        - y[5]
    )
    fvec[6] = (
        x[0] * x[4] * (x[4] ** 2 - 3 * x[6] ** 2)
        + x[2] * x[6] * (x[6] ** 2 - 3 * x[4] ** 2)
        + x[1] * x[5] * (x[5] ** 2 - 3 * x[7] ** 2)
        + x[3] * x[7] * (x[7] ** 2 - 3 * x[5] ** 2)
        - y[6]
    )
    fvec[7] = (
        x[2] * x[4] * (x[4] ** 2 - 3 * x[6] ** 2)
        - x[0] * x[6] * (x[6] ** 2 - 3 * x[4] ** 2)
        + x[3] * x[5] * (x[5] ** 2 - 3 * x[7] ** 2)
        - x[1] * x[7] * (x[7] ** 2 - 3 * x[5] ** 2)
        - y[7]
    )


MORE_WILD_PROBLEMS = {
    "linear_full_rank_good_start": {
        "criterion": partial(linear_full_rank, dim_out=45),
        "start_x": np.ones(9),
        "solution_x": None,
        "start_criterion": 72,
        "solution_criterion": 36,
    },
    "linear_full_rank_bad_start": {
        "criterion": partial(linear_full_rank, dim_out=45),
        "start_x": np.ones(9) * 10,
        "solution_x": None,
        "start_criterion": 1125,
        "solution_criterion": 36,
    },
}
