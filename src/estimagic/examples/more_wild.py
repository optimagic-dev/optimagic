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
    dim_in = len(x)
    sm = np.arange(1, dim_in + 1) @ x
    fvec = np.arange(1, dim_out + 1) * sm - 1.0
    return fvec


def linear_rank_one_zero_columns_rows(x, dim_out):
    dim_in = len(x)
    sm = (np.arange(2, dim_in) * x[1:-1]).sum()
    fvec = np.arange(dim_out) * sm - 1.0
    fvec[-1] = -1.0
    return fvec


def rosenbrock(x):
    fvec = np.zeros(2)
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
    fvec = np.zeros(3)
    fvec[0] = 10 * (x[2] - 10 * temp1)
    fvec[1] = 10 * (temp2 - 1.0)
    fvec[2] = x[2]
    return fvec


def powell_singular(x):
    fvec = np.zeros(4)
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
        temp = len(y) + 1 - i
        fvec[i - 1] = y[i - 1] - (x[0] + i / (x[1] * temp + x[2] * i))
    for i in range(round(len(y) / 2) + 1, len(y) + 1):
        temp = len(y) + 1 - i
        fvec[i - 1] = y[i - 1] - (x[0] + i / (x[1] * temp + x[2] * temp))
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
        temp = i / 29
        sum_1 = (np.arange(1, dim_in) * temp ** np.arange(dim_in - 1) * x[1:]).sum()
        sum_2 = (temp ** np.arange(dim_in) * x).sum()
        fvec[i - 1] = sum_1 - sum_2 ** 2 - 1.0
    fvec[29] = x[0]
    fvec[30] = x[1] - x[0] ** 2 - 1.0
    return fvec


def box_3d(x, dim_out):
    fvec = np.zeros(dim_out)
    for i in range(1, dim_out + 1):
        fvec[i - 1] = (
            np.exp(-i / 10 * x[0])
            - np.exp(-i / 10 * x[1])
            + (np.exp(-i) - np.exp(-i / 10)) * x[2]
        )
    return fvec


def jennrich_sampson(x, dim_out):
    fvec = (
        2 * (1.0 + np.arange(1, dim_out + 1))
        - np.exp(np.arange(1, dim_out + 1) * x[0])
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
        temp_3 = 2 * temp_2
        for j in range(dim_out):
            fvec[j] = fvec[j] + temp_2
            temp_4 = temp_3 * temp_2 - temp_1
            temp_1 = temp_2
            temp_2 = temp_4
    for i in range(1, dim_out + 1):
        fvec[i - 1] = fvec[i - 1] / dim_in
        if i % 2 == 0:
            fvec[i - 1] = fvec[i - 1] + 1 / (i ** 2 - 1.0)
    return fvec


def brown_almost_linear(x):
    dim_in = len(x)
    sm = -(dim_in + 1) + x.sum()
    product = x.prod()
    fvec = x + sm
    fvec[dim_in - 1] = product - 1.0
    return fvec


def osborne_one(x, y):
    temp = 10 * np.arange(len(y))
    temp_1 = np.exp(-x[3] * temp)
    temp_2 = np.exp(-x[4] * temp)
    fvec = y - (x[0] + x[1] * temp_1 + x[2] * temp_2)
    return fvec


def osborne_two(x, y):
    temp_array = np.zeros((4, len(y)))
    temp = np.arange(len(y)) / 10
    temp_array[0] = np.exp(-x[4] * temp)
    temp_array[1] = np.exp(-x[5] * (temp - x[8]) ** 2)
    temp_array[2] = np.exp(-x[6] * (temp - x[9]) ** 2)
    temp_array[3] = np.exp(-x[7] * (temp - x[10]) ** 2)
    fvec = y - (temp_array.T * x[:4]).T.sum(axis=0)
    return fvec


def bdqrtic(x):
    # the length of array x should be more then 5.
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
    fvec = 10 * (x - np.roll(x, 1) ** 3)
    fvec[0] = x[0] - 1.0
    return fvec


def mancino(x):
    dim_in = len(x)
    fvec = np.zeros(dim_in)
    for i in range(dim_in):
        sm = 0
        for j in range(dim_in):
            temp = np.sqrt(x[i] ** 2 + (i + 1) / (j + 1))
            sm += temp * ((np.sin(np.log(temp))) ** 5 + (np.cos(np.log(temp))) ** 5)
        fvec[i] = 1400 * x[i] + (i + 1 - 50) ** 3 + sm
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
    return fvec


def get_start_points_mancino(n):
    x = np.zeros(n)
    for i in range(1, n + 1):
        sm = 0
        for j in range(1, n + 1):
            sm += np.sqrt(i / j) * (
                (np.sin(np.log(np.sqrt(i / j)))) ** 5
                + (np.cos(np.log(np.sqrt(i / j)))) ** 5
            )
        x[i - 1] = -8.7110e-04 * ((i - 50) ** 3 + sm)
    return x


y_vec = np.array(
    [
        0.1400,
        0.1800,
        0.2200,
        0.2500,
        0.2900,
        0.3200,
        0.3500,
        0.3900,
        0.3700,
        0.5800,
        0.7300,
        0.9600,
        1.3400,
        2.1000,
        4.3900,
    ]
)

v_vec = np.array(
    [
        4.0000,
        2.0000,
        1.0000,
        0.5000,
        0.2500,
        0.1670,
        0.1250,
        0.1000,
        0.0833,
        0.0714,
        0.0625,
    ]
)

y2_vec = np.array(
    [
        0.1957,
        0.1947,
        0.1735,
        0.1600,
        0.0844,
        0.0627,
        0.0456,
        0.0342,
        0.0323,
        0.0235,
        0.0246,
    ]
)

y3_vec = np.array(
    [
        34780,
        28610,
        23650,
        19630,
        16370,
        13720,
        11540,
        9744,
        8261,
        7030,
        6005,
        5147,
        4427,
        3820,
        3307,
        2872,
    ]
)
y4_vec = np.array(
    [
        8.44e-1,
        9.08e-1,
        9.32e-1,
        9.36e-1,
        9.25e-1,
        9.08e-1,
        8.81e-1,
        8.5e-1,
        8.18e-1,
        7.84e-1,
        7.51e-1,
        7.18e-1,
        6.85e-1,
        6.58e-1,
        6.28e-1,
        6.03e-1,
        5.8e-1,
        5.58e-1,
        5.38e-1,
        5.22e-1,
        5.06e-1,
        4.9e-1,
        4.78e-1,
        4.67e-1,
        4.57e-1,
        4.48e-1,
        4.38e-1,
        4.31e-1,
        4.24e-1,
        4.2e-1,
        4.14e-1,
        4.11e-1,
        4.06e-1,
    ]
)
y5_vec = np.array(
    [
        1.366e0,
        1.191e0,
        1.112e0,
        1.013e0,
        9.91e-1,
        8.85e-1,
        8.31e-1,
        8.47e-1,
        7.86e-1,
        7.25e-1,
        7.46e-1,
        6.79e-1,
        6.08e-1,
        6.55e-1,
        6.16e-1,
        6.06e-1,
        6.02e-1,
        6.26e-1,
        6.51e-1,
        7.24e-1,
        6.49e-1,
        6.49e-1,
        6.94e-1,
        6.44e-1,
        6.24e-1,
        6.61e-1,
        6.12e-1,
        5.58e-1,
        5.33e-1,
        4.95e-1,
        5.0e-1,
        4.23e-1,
        3.95e-1,
        3.75e-1,
        3.72e-1,
        3.91e-1,
        3.96e-1,
        4.05e-1,
        4.28e-1,
        4.29e-1,
        5.23e-1,
        5.62e-1,
        6.07e-1,
        6.53e-1,
        6.72e-1,
        7.08e-1,
        6.33e-1,
        6.68e-1,
        6.45e-1,
        6.32e-1,
        5.91e-1,
        5.59e-1,
        5.97e-1,
        6.25e-1,
        7.39e-1,
        7.1e-1,
        7.29e-1,
        7.2e-1,
        6.36e-1,
        5.81e-1,
        4.28e-1,
        2.92e-1,
        1.62e-1,
        9.8e-2,
        5.4e-2,
    ]
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
    "linear_rank_one_good_start": {
        "criterion": partial(linear_rank_one, dim_out=35),
        "start_x": np.ones(7),
        "solution_x": None,
        "start_criterion": 1.165420e7,
        "solution_criterion": 8.380282,
    },
    "linear_rank_one_bad_start": {
        "criterion": partial(linear_rank_one, dim_out=35),
        "start_x": np.ones(7) * 10,
        "solution_x": None,
        "start_criterion": 1.168591e9,
        "solution_criterion": 8.380282,
    },
    "linear_rank_one_zero_columns_rows_good_start": {
        "criterion": partial(linear_rank_one_zero_columns_rows, dim_out=35),
        "start_x": np.ones(7),
        "solution_x": None,
        "start_criterion": 4.989195e6,
        "solution_criterion": 9.880597,
    },
    "linear_rank_one_zero_columns_rows_bad_start": {
        "criterion": partial(linear_rank_one_zero_columns_rows, dim_out=35),
        "start_x": np.ones(7) * 10,
        "solution_x": None,
        "start_criterion": 5.009356e8,
        "solution_criterion": 9.880597,
    },
    "rosenbrock_good_start": {
        "criterion": rosenbrock,
        "start_x": np.array([-1.2, 1]),
        "solution_x": None,
        "start_criterion": 24.2,
        "solution_criterion": 0,
    },
    "rosenbrock_bad_start": {
        "criterion": rosenbrock,
        "start_x": np.array([-1.2, 1]) * 10,
        "solution_x": None,
        "start_criterion": 1.795769e6,
        "solution_criterion": 0,
    },
    "helical_valley_good_start": {
        "criterion": helical_valley,
        "start_x": np.array([-1, 0, 0]),
        "solution_x": None,
        "start_criterion": 2500,
        "solution_criterion": 0,
    },
    "helical_valley_bad_start": {
        "criterion": helical_valley,
        "start_x": np.array([-10, 0, 0]),
        "solution_x": None,
        "start_criterion": 10600,
        "solution_criterion": 0,
    },
    "powell_singular_good_start": {
        "criterion": powell_singular,
        "start_x": np.array([3, -1, 0, 1]),
        "solution_x": None,
        "start_criterion": 215,
        "solution_criterion": 0,
    },
    "powell_singular_bad_start": {
        "criterion": powell_singular,
        "start_x": np.array([3, -1, 0, 1]) * 10,
        "solution_x": None,
        "start_criterion": 1.615400e6,
        "solution_criterion": 0,
    },
    "freudenstein_roth_good_start": {
        "criterion": freudenstein_roth,
        "start_x": np.array([0.5, -2]),
        "solution_x": None,
        "start_criterion": 400.5,
        "solution_criterion": 0,
    },
    "freudenstein_roth_bad_start": {
        "criterion": freudenstein_roth,
        "start_x": np.array([0.5, -2]) * 10,
        "solution_x": None,
        "start_criterion": 1.545754e8,
        "solution_criterion": 0,
    },
    "bard_good_start": {
        "criterion": partial(bard, y=y_vec),
        "start_x": np.ones(3),
        "solution_x": None,
        "start_criterion": 41.68170,
        "solution_criterion": 8.214877e-3,
    },
    "bard_bad_start": {
        "criterion": partial(bard, y=y_vec),
        "start_x": np.ones(3) * 10,
        "solution_x": None,
        "start_criterion": 1306.234,
        "solution_criterion": 8.214877e-3,
    },
    "kowalik_osborne": {
        "criterion": partial(
            kowalik_osborne,
            y1=v_vec,
            y2=y2_vec,
        ),
        "start_x": np.array([0.25, 0.39, 0.415, 0.39]),
        "solution_x": None,
        "start_criterion": 5.313172e-3,
        "solution_criterion": 3.075056e-4,
    },
    "meyer": {
        "criterion": partial(meyer, y=y3_vec),
        "start_x": np.array([0.02, 4000, 250]),
        "solution_x": None,
        "start_criterion": 1.693608e9,
        "solution_criterion": 87.94586,
    },
    "watson_6_good_start": {
        "criterion": watson,
        "start_x": 0.5 * np.ones(6),
        "solution_x": None,
        "start_criterion": 16.43083,
        "solution_criterion": 2.287670e-3,
    },
    "watson_6_bad_start": {
        "criterion": watson,
        "start_x": 5 * np.ones(6),
        "solution_x": None,
        "start_criterion": 2.323367e6,
        "solution_criterion": 2.287670e-3,
    },
    "watson_9_good_start": {
        "criterion": watson,
        "start_x": 0.5 * np.ones(9),
        "solution_x": None,
        "start_criterion": 26.90417,
        "solution_criterion": 1.399760e-6,
    },
    "watson_9_bad_start": {
        "criterion": watson,
        "start_x": 5 * np.ones(9),
        "solution_x": None,
        "start_criterion": 8.158877e6,
        "solution_criterion": 1.399760e-6,
    },
    "watson_12_good_start": {
        "criterion": watson,
        "start_x": 0.5 * np.ones(12),
        "solution_x": None,
        "start_criterion": 73.67821,
        "solution_criterion": 4.722381e-10,
    },
    "watson_12_bad_start": {
        "criterion": watson,
        "start_x": 5 * np.ones(12),
        "solution_x": None,
        "start_criterion": 2.059384e7,
        "solution_criterion": 4.722381e-10,
    },
    "box_3d": {
        "criterion": partial(box_3d, dim_out=10),
        "start_x": np.array([0, 10, 20]),
        "solution_x": None,
        "start_criterion": 1031.154,
        "solution_criterion": 0,
    },
    "jennrich_sampson": {
        "criterion": partial(jennrich_sampson, dim_out=10),
        "start_x": np.array([0.3, 0.4]),
        "solution_x": None,
        "start_criterion": 4171.306,
        "solution_criterion": 124.3622,
    },
    "brown_dennis_good_start": {
        "criterion": partial(brown_dennis, dim_out=20),
        "start_x": np.array([25, 5, -5, -1]),
        "solution_x": None,
        "start_criterion": 7.926693e6,
        "solution_criterion": 8.582220e4,
    },
    "brown_dennis_bad_start": {
        "criterion": partial(brown_dennis, dim_out=20),
        "start_x": np.array([25, 5, -5, -1]) * 10,
        "solution_x": None,
        "start_criterion": 3.081064e11,
        "solution_criterion": 8.582220e4,
    },
    "chebyquad_6": {
        "criterion": partial(chebyquad, dim_out=6),
        "start_x": np.arange(1, 7) / 7,
        "solution_x": None,
        "start_criterion": 4.642817e-2,
        "solution_criterion": 0,
    },
    "chebyquad_7": {
        "criterion": partial(chebyquad, dim_out=7),
        "start_x": np.arange(1, 8) / 8,
        "solution_x": None,
        "start_criterion": 3.377064e-2,
        "solution_criterion": 0,
    },
    "chebyquad_8": {
        "criterion": partial(chebyquad, dim_out=8),
        "start_x": np.arange(1, 9) / 9,
        "solution_x": None,
        "start_criterion": 3.861770e-2,
        "solution_criterion": 3.516874e-3,
    },
    "chebyquad_9": {
        "criterion": partial(chebyquad, dim_out=9),
        "start_x": np.arange(1, 10) / 10,
        "solution_x": None,
        "start_criterion": 2.888298e-2,
        "solution_criterion": 0,
    },
    "chebyquad_10": {
        "criterion": partial(chebyquad, dim_out=10),
        "start_x": np.arange(1, 11) / 11,
        "solution_x": None,
        "start_criterion": 3.376327e-2,
        "solution_criterion": 4.772714e-3,
    },
    "chebyquad_11": {
        "criterion": partial(chebyquad, dim_out=11),
        "start_x": np.arange(1, 12) / 12,
        "solution_x": None,
        "start_criterion": 2.674060e-2,
        "solution_criterion": 2.799762e-3,
    },
    "brown_almost_linear": {
        "criterion": brown_almost_linear,
        "start_x": 0.5 * np.ones(10),
        "solution_x": None,
        "start_criterion": 273.2480,
        "solution_criterion": 0,
    },
    "osborne_one": {
        "criterion": partial(osborne_one, y=y4_vec),
        "start_x": np.array([0.5, 1.5, 1, 0.01, 0.02]),
        "solution_x": None,
        "start_criterion": 16.17411,
        "solution_criterion": 5.464895e-5,
    },
    "osborne_two_good_start": {
        "criterion": partial(osborne_two, y=y5_vec),
        "start_x": np.array([1.3, 0.65, 0.65, 0.7, 0.6, 3, 5, 7, 2, 4.5, 5.5]),
        "solution_x": None,
        "start_criterion": 2.093420,
        "solution_criterion": 4.013774e-2,
    },
    "osborne_two_bad_start": {
        "criterion": partial(osborne_two, y=y5_vec),
        "start_x": 10 * np.array([1.3, 0.65, 0.65, 0.7, 0.6, 3, 5, 7, 2, 4.5, 5.5]),
        "solution_x": None,
        "start_criterion": 199.6847,
        "solution_criterion": 4.013774e-2,
    },
    "bdqrtic_8": {
        "criterion": bdqrtic,
        "start_x": np.ones(8),
        "solution_x": None,
        "start_criterion": 904,
        "solution_criterion": 10.23897,
    },
    "bdqrtic_10": {
        "criterion": bdqrtic,
        "start_x": np.ones(10),
        "solution_x": None,
        "start_criterion": 1356,
        "solution_criterion": 18.28116,
    },
    "bdqrtic_11": {
        "criterion": bdqrtic,
        "start_x": np.ones(11),
        "solution_x": None,
        "start_criterion": 1582,
        "solution_criterion": 22.26059,
    },
    "bdqrtic_12": {
        "criterion": bdqrtic,
        "start_x": np.ones(12),
        "solution_x": None,
        "start_criterion": 1808,
        "solution_criterion": 26.27277,
    },
    "cube_5": {
        "criterion": cube,
        "start_x": 0.5 * np.ones(5),
        "solution_x": None,
        "start_criterion": 56.5,
        "solution_criterion": 0,
    },
    "cube_6": {
        "criterion": cube,
        "start_x": 0.5 * np.ones(6),
        "solution_x": None,
        "start_criterion": 70.5625,
        "solution_criterion": 0,
    },
    "cube_8": {
        "criterion": cube,
        "start_x": 0.5 * np.ones(8),
        "solution_x": None,
        "start_criterion": 98.6875,
        "solution_criterion": 0,
    },
    "mancino_5_good_start": {
        "criterion": mancino,
        "start_x": get_start_points_mancino(5),
        "solution_x": None,
        "start_criterion": 2.539084e9,
        "solution_criterion": 0,
    },
    "mancino_5_bad_start": {
        "criterion": mancino,
        "start_x": 10 * get_start_points_mancino(5),
        "solution_x": None,
        "start_criterion": 6.873795e12,
        "solution_criterion": 0,
    },
    "mancino_8": {
        "criterion": mancino,
        "start_x": get_start_points_mancino(8),
        "solution_x": None,
        "start_criterion": 3.367961e9,
        "solution_criterion": 0,
    },
    "mancino_10": {
        "criterion": mancino,
        "start_x": get_start_points_mancino(10),
        "solution_x": None,
        "start_criterion": 3.735127e9,
        "solution_criterion": 0,
    },
    "mancino_12_good_start": {
        "criterion": mancino,
        "start_x": get_start_points_mancino(12),
        "solution_x": None,
        "start_criterion": 3.991072e9,
        "solution_criterion": 0,
    },
    "mancino_12_bad_start": {
        "criterion": mancino,
        "start_x": 10 * get_start_points_mancino(12),
        "solution_x": None,
        "start_criterion": 1.130015e13,
        "solution_criterion": 0,
    },
    "heart_eight_good_start": {
        "criterion": partial(
            heart_eight,
            y=np.array([-0.69, -0.044, -1.57, -1.31, -2.65, 2, -12.6, 9.48]),
        ),
        "start_x": np.array([-0.3, -0.39, 0.3, -0.344, -1.2, 2.69, 1.59, -1.5]),
        "solution_x": None,
        "start_criterion": 9.385672,
        "solution_criterion": 0,
    },
    "heart_eight_bad_start": {
        "criterion": partial(
            heart_eight,
            y=np.array([-0.69, -0.044, -1.57, -1.31, -2.65, 2, -12.6, 9.48]),
        ),
        "start_x": 10 * np.array([-0.3, -0.39, 0.3, -0.344, -1.2, 2.69, 1.59, -1.5]),
        "solution_x": None,
        "start_criterion": 3.365815e10,
        "solution_criterion": 0,
    },
    "brown_almost_linear_medium": {
        "criterion": brown_almost_linear,
        "start_x": 0.5 * np.ones(100),
        "solution_x": None,
        "start_criterion": 2.524757e5,
        "solution_criterion": 0,
    },
}
