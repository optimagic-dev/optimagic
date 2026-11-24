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

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from optimagic import mark
from optimagic.benchmarking.benchmark_problem import (
    BenchmarkFunction,
    BenchmarkProblem,
    BenchmarkSolution,
)


@mark.least_squares
def linear_full_rank(x, dim_out):
    temp = 2 * x.sum() / dim_out + 1
    out = np.full(dim_out, -temp)
    out[: len(x)] += x
    return out


@mark.least_squares
def linear_rank_one(x, dim_out):
    dim_in = len(x)
    sm = np.arange(1, dim_in + 1) @ x
    fvec = np.arange(1, dim_out + 1) * sm - 1.0
    return fvec


@mark.least_squares
def linear_rank_one_zero_columns_rows(x, dim_out):
    dim_in = len(x)
    sm = (np.arange(2, dim_in) * x[1:-1]).sum()
    fvec = np.arange(dim_out) * sm - 1.0
    fvec[-1] = -1.0
    return fvec


@mark.least_squares
def rosenbrock(x):
    fvec = np.zeros(2)
    fvec[0] = 10 * (x[1] - x[0] ** 2)
    fvec[1] = 1.0 - x[0]
    return fvec


@mark.least_squares
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


@mark.least_squares
def powell_singular(x):
    fvec = np.zeros(4)
    fvec[0] = x[0] + 10 * x[1]
    fvec[1] = np.sqrt(5.0) * (x[2] - x[3])
    fvec[2] = (x[1] - 2 * x[2]) ** 2
    fvec[3] = np.sqrt(10.0) * (x[0] - x[3]) ** 2
    return fvec


@mark.least_squares
def freudenstein_roth(x):
    fvec = np.zeros(2)
    fvec[0] = -13 + x[0] + ((5 - x[1]) * x[1] - 2) * x[1]
    fvec[1] = -29 + x[0] + ((1.0 + x[1]) * x[1] - 14) * x[1]
    return fvec


@mark.least_squares
def bard(x, y):
    fvec = np.zeros(len(y))
    for i in range(1, round(len(y) / 2) + 1):
        temp = len(y) + 1 - i
        fvec[i - 1] = y[i - 1] - (x[0] + i / (x[1] * temp + x[2] * i))
    for i in range(round(len(y) / 2) + 1, len(y) + 1):
        temp = len(y) + 1 - i
        fvec[i - 1] = y[i - 1] - (x[0] + i / (x[1] * temp + x[2] * temp))
    return fvec


@mark.least_squares
def kowalik_osborne(x, y1, y2):
    temp1 = y1 * (y1 + x[1])
    temp2 = y1 * (y1 + x[2]) + x[3]
    fvec = y2 - x[0] * temp1 / temp2
    return fvec


@mark.least_squares
def meyer(x, y):
    temp = 5 * np.arange(1, len(y) + 1) + 45 + x[2]
    temp1 = x[1] / temp
    temp2 = np.exp(temp1)
    fvec = x[0] * temp2 - y
    return fvec


@mark.least_squares
def watson(x):
    dim_in = len(x)
    fvec = np.zeros(31)
    for i in range(1, 30):
        temp = i / 29
        sum_1 = (np.arange(1, dim_in) * temp ** np.arange(dim_in - 1) * x[1:]).sum()
        sum_2 = (temp ** np.arange(dim_in) * x).sum()
        fvec[i - 1] = sum_1 - sum_2**2 - 1.0
    fvec[29] = x[0]
    fvec[30] = x[1] - x[0] ** 2 - 1.0
    return fvec


@mark.least_squares
def box_3d(x, dim_out):
    fvec = np.zeros(dim_out)
    for i in range(1, dim_out + 1):
        fvec[i - 1] = (
            np.exp(-i / 10 * x[0])
            - np.exp(-i / 10 * x[1])
            + (np.exp(-i) - np.exp(-i / 10)) * x[2]
        )
    return fvec


@mark.least_squares
def jennrich_sampson(x, dim_out):
    fvec = (
        2 * (1.0 + np.arange(1, dim_out + 1))
        - np.exp(np.arange(1, dim_out + 1) * x[0])
        - np.exp(np.arange(1, dim_out + 1) * x[1])
    )
    return fvec


@mark.least_squares
def brown_dennis(x, dim_out):
    fvec = np.zeros(dim_out)
    for i in range(1, dim_out + 1):
        temp = i / 5
        temp_1 = x[0] + temp * x[1] - np.exp(temp)
        temp_2 = x[2] + np.sin(temp) * x[3] - np.cos(temp)
        fvec[i - 1] = temp_1**2 + temp_2**2
    return fvec


@mark.least_squares
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
            fvec[i - 1] = fvec[i - 1] + 1 / (i**2 - 1.0)
    return fvec


@mark.least_squares
def brown_almost_linear(x):
    dim_in = len(x)
    sm = -(dim_in + 1) + x.sum()
    product = x.prod()
    fvec = x + sm
    fvec[dim_in - 1] = product - 1.0
    return fvec


@mark.least_squares
def osborne_one(x, y):
    temp = 10 * np.arange(len(y))
    temp_1 = np.exp(-x[3] * temp)
    temp_2 = np.exp(-x[4] * temp)
    fvec = y - (x[0] + x[1] * temp_1 + x[2] * temp_2)
    return fvec


@mark.least_squares
def osborne_two(x, y):
    temp_array = np.zeros((4, len(y)))
    temp = np.arange(len(y)) / 10
    temp_array[0] = np.exp(-x[4] * temp)
    temp_array[1] = np.exp(-x[5] * (temp - x[8]) ** 2)
    temp_array[2] = np.exp(-x[6] * (temp - x[9]) ** 2)
    temp_array[3] = np.exp(-x[7] * (temp - x[10]) ** 2)
    fvec = y - (temp_array.T * x[:4]).T.sum(axis=0)
    return fvec


@mark.least_squares
def bdqrtic(x):
    # the length of array x should be more than 5.
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


@mark.least_squares
def cube(x):
    fvec = 10 * (x - np.roll(x, 1) ** 3)
    fvec[0] = x[0] - 1.0
    return fvec


@mark.least_squares
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


@mark.least_squares
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


@mark.least_squares
def get_start_points_mancino(n, a=1):
    x = np.zeros(n)
    for i in range(1, n + 1):
        sm = 0
        for j in range(1, n + 1):
            sm += np.sqrt(i / j) * (
                (np.sin(np.log(np.sqrt(i / j)))) ** 5
                + (np.cos(np.log(np.sqrt(i / j)))) ** 5
            )
        x[i - 1] = -8.7110e-04 * ((i - 50) ** 3 + sm)
    return (x * a).tolist()


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


linear_full_rank_solution_x = [
    -0.9999999988839997,
    -1.0000000177422066,
    -1.0000000115935452,
    -1.0000000228208163,
    -1.0000000488884697,
    -0.9999999970458138,
    -0.999999957053959,
    -1.0000000040514776,
    -0.9999999708374043,
]


freudenstein_roth_solution_x = [11.4127789219781, -0.8968052599835741]


bard_solution_x = [0.08241056005476516, 1.1330360796060677, 2.3436951913379658]


kowalik_osborne_solution_x = [
    0.19280693401647758,
    0.19128233030789646,
    0.12305650338704374,
    0.1360623315234073,
]


meyer_solution_x = [0.005609636453940975, 6181.3463491557495, 345.22363473367955]

watson_6_solution_x = [
    -0.01572508595814696,
    1.0124348692251488,
    -0.23299161822960684,
    1.2604300607312298,
    -1.5137288869025518,
    0.9929964192277573,
]


# Note: only nlopt_neldermead got close to the correct optimal criterion value.
# Parameter values might be less precise than others but should be precise enough
# for all practical purposes.
watson_9_solution_x = [
    -1.5307729818292037e-05,
    0.9997897038761921,
    0.014763956456196943,
    0.14634240306061744,
    1.000820801996808,
    -2.617730533377693,
    4.104402503186126,
    -3.1436119083184844,
    1.0526263240326197,
]


# Note: only nlopt_nobyqa got close to the correct optimal criterion value.
# Parameter values might be less precise than others but should be precise enough
# for all practical purposes.
watson_12_solution_x = [
    -1.257374334661004e-07,
    1.000009574359581,
    -0.0005801330054146337,
    0.339181153679104,
    -0.01717885040751319,
    0.1133023927390161,
    0.19016852711009063,
    -0.21697797575421524,
    -0.20528305553311146,
    0.9344814896242725,
    -0.8979508634897754,
    0.3182351206188577,
]

brown_dennis_solution_x = [
    -11.594439969349615,
    13.203630099554186,
    -0.40343943943781074,
    0.2367787758603151,
]

chebyquad_6_solution_x = [
    0.06687659094608964,
    0.2887406731194441,
    0.36668229924164747,
    0.6333177007583523,
    0.7112593268805555,
    0.9331234090539102,
]


chebyquad_7_solution_x = [
    0.0580691496209753,
    0.23517161235742137,
    0.3380440947400461,
    0.49999999999999983,
    0.6619559052599537,
    0.7648283876425783,
    0.9419308503790245,
]


chebyquad_8_solution_x = [
    0.043152760689960816,
    0.19309084165259105,
    0.2663287079773684,
    0.5000000016286815,
    0.5000000007226908,
    0.8069091602434582,
    0.7336712939109635,
    0.9568472402172841,
]

chebyquad_9_solution_x = [
    0.04420534613578318,
    0.19949067230988682,
    0.23561910847105574,
    0.4160469078926057,
    0.5839530921074088,
    0.4999999999999922,
    0.800509327690123,
    0.7643808915289372,
    0.9557946538642177,
]


chebyquad_10_solution_x = [
    0.07474816709152399,
    0.17151817795786592,
    0.28643415454482585,
    0.35964645053932914,
    0.4707505262783716,
    0.6167383355304029,
    0.6167383367837294,
    0.7998108031241883,
    0.844854641539109,
    0.9670066274628275,
]


chebyquad_11_solution_x = [
    0.02995874447661457,
    0.1373112070822553,
    0.18836638791417698,
    0.3588431173822416,
    0.3588431160884765,
    0.5000000000242054,
    0.6411568833224512,
    0.6411568815391566,
    0.8116336110470005,
    0.8626887929155374,
    0.9700412549151204,
]

osborne_one_solution_x = [
    0.37541005253870485,
    1.9358469347077125,
    -1.4646871598379403,
    0.012867534697214533,
    0.02212269960299629,
]

osborne_two_solution_x = [
    1.3099771555174913,
    0.4315537955622272,
    0.6336616986693765,
    0.5994305344293098,
    0.7541832304802704,
    0.9042885759622441,
    1.365811821857166,
    4.823698851312894,
    2.398684862961737,
    4.568874597996633,
    5.675341470445994,
]


bdqrtic_8_solution_x = [
    0.616075443630495,
    0.4861767187980861,
    0.39190293828200784,
    0.32635052133139375,
    5.7665311977077046e-09,
    9.348707442258251e-09,
    7.066347917413364e-09,
    -2.030598138768078e-09,
]


bdqrtic_10_solution_x = [
    0.6255364749479968,
    0.4851009828850974,
    0.3671943518989714,
    0.28518847760113386,
    0.33016716122418716,
    0.37757199483645576,
    -3.24040819296658e-09,
    -1.8973118921921425e-08,
    -2.2244236071548075e-08,
    1.9263207246002433e-09,
]


bdqrtic_11_solution_x = [
    0.6251418193253757,
    0.4858196102070445,
    0.3712502347939938,
    0.28350403794642487,
    0.31694697562905494,
    0.33873300184720523,
    0.3759208995980027,
    -1.8942209640948616e-08,
    3.418631657404969e-08,
    -4.003185000628104e-09,
    3.166166094063382e-09,
]


bdqrtic_12_solution_x = [
    0.6248003622228653,
    0.48537650602979937,
    0.37165912289534886,
    0.2859718523039759,
    0.31552001728813406,
    0.3253724392486982,
    0.33781861543778574,
    0.37402021737899876,
    -4.429208872117422e-09,
    -1.008941638491605e-08,
    -2.5608732325955336e-08,
    4.485976896804288e-09,
]


mancino_5_solution_x = [
    84.28291101102532,
    79.20603967293438,
    74.3364141135311,
    69.6711474112178,
    65.20718113814442,
]

mancino_8_solution_x = [
    84.43334222593528,
    79.33454939399172,
    74.44387011026309,
    69.7592945870252,
    65.27853533617875,
    60.9988580578957,
    56.9169379354432,
    53.028761291567236,
]

mancino_10_solution_x = [
    84.53434289477315,
    79.42084435375007,
    74.51601545241338,
    69.81844699647671,
    65.32637991893166,
    61.03748806452533,
    56.94869518846038,
    53.056052319528746,
    49.35469508461959,
    45.83889035077595,
]


mancino_12_solution_x = [
    84.63591921594158,
    79.5076423105225,
    74.5885724920863,
    69.87791406833868,
    65.37444824684921,
    61.07626530788906,
    56.98054088428213,
    53.08338921660163,
    49.379816810523714,
    45.86378591833196,
    42.52838225939789,
    39.36606891417026,
]

heart_eight_solution_x = [
    -0.311626605565399,
    -0.37837339443458845,
    0.3282442301180765,
    -0.3722442301180588,
    -1.282227094270286,
    2.4943003120854743,
    1.5548658787873983,
    -1.384637842863253,
]


heart_eight_y_vec = np.array([-0.69, -0.044, -1.57, -1.31, -2.65, 2, -12.6, 9.48])


# ======================================================================================
# Benchmark functions
# ======================================================================================


@mark.least_squares
@dataclass(frozen=True)
class LinearFullRank(BenchmarkFunction):
    """Linear function with full rank."""

    dim_out: int
    """The number of residuals. Must not be smaller than the number of parameters."""

    differentiable = True
    convex = True
    multimodal = False

    def fun(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return linear_full_rank(x, dim_out=self.dim_out)

    def supports_n_params(self, n_params: int) -> bool:
        return 1 <= n_params <= self.dim_out


@mark.least_squares
@dataclass(frozen=True)
class LinearRankOne(BenchmarkFunction):
    """Linear function with rank one."""

    dim_out: int
    """The number of residuals. Must not be smaller than the number of parameters."""

    differentiable = True
    convex = True
    multimodal = False

    def fun(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return linear_rank_one(x, dim_out=self.dim_out)

    def supports_n_params(self, n_params: int) -> bool:
        return 1 <= n_params <= self.dim_out


@mark.least_squares
@dataclass(frozen=True)
class LinearRankOneZeroColumnsRows(BenchmarkFunction):
    """Linear function with rank one and zero columns and rows."""

    dim_out: int
    """The number of residuals. Must not be smaller than the number of parameters."""

    differentiable = True
    convex = True
    multimodal = False

    def fun(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return linear_rank_one_zero_columns_rows(x, dim_out=self.dim_out)

    def supports_n_params(self, n_params: int) -> bool:
        return 1 <= n_params <= self.dim_out


@mark.least_squares
@dataclass(frozen=True)
class Rosenbrock(BenchmarkFunction):
    """Rosenbrock function with two parameters."""

    differentiable = True
    convex = False
    multimodal = False
    min_n_params = 2
    max_n_params = 2

    def fun(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return rosenbrock(x)


@mark.least_squares
@dataclass(frozen=True)
class HelicalValley(BenchmarkFunction):
    """Helical valley function."""

    # the first residual jumps where x[0] changes its sign and x[1] is negative
    differentiable = False
    min_n_params = 3
    max_n_params = 3

    def fun(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return helical_valley(x)


@mark.least_squares
@dataclass(frozen=True)
class PowellSingular(BenchmarkFunction):
    """Powell singular function."""

    differentiable = True
    convex = True
    multimodal = False
    min_n_params = 4
    max_n_params = 4

    def fun(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return powell_singular(x)


@mark.least_squares
@dataclass(frozen=True)
class FreudensteinRoth(BenchmarkFunction):
    """Freudenstein and Roth function."""

    differentiable = True
    convex = False
    # has a local minimum with a function value of about 48.98 and a global minimum
    # with a function value of zero
    multimodal = True
    min_n_params = 2
    max_n_params = 2

    def fun(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return freudenstein_roth(x)


@mark.least_squares
@dataclass(frozen=True)
class Bard(BenchmarkFunction):
    """Bard function."""

    differentiable = True
    min_n_params = 3
    max_n_params = 3

    def fun(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return bard(x, y=y_vec)


@mark.least_squares
@dataclass(frozen=True)
class KowalikOsborne(BenchmarkFunction):
    """Kowalik and Osborne function."""

    differentiable = True
    min_n_params = 4
    max_n_params = 4

    def fun(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return kowalik_osborne(x, y1=v_vec, y2=y2_vec)


@mark.least_squares
@dataclass(frozen=True)
class Meyer(BenchmarkFunction):
    """Meyer function."""

    differentiable = True
    min_n_params = 3
    max_n_params = 3

    def fun(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return meyer(x, y=y3_vec)


@mark.least_squares
@dataclass(frozen=True)
class Watson(BenchmarkFunction):
    """Watson function."""

    differentiable = True
    min_n_params = 2
    max_n_params = 31

    def fun(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return watson(x)


@mark.least_squares
@dataclass(frozen=True)
class Box3D(BenchmarkFunction):
    """Box three-dimensional function."""

    dim_out: int
    """The number of residuals."""

    differentiable = True
    min_n_params = 3
    max_n_params = 3

    def fun(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return box_3d(x, dim_out=self.dim_out)


@mark.least_squares
@dataclass(frozen=True)
class JennrichSampson(BenchmarkFunction):
    """Jennrich and Sampson function."""

    dim_out: int
    """The number of residuals."""

    differentiable = True
    min_n_params = 2
    max_n_params = 2

    def fun(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return jennrich_sampson(x, dim_out=self.dim_out)


@mark.least_squares
@dataclass(frozen=True)
class BrownDennis(BenchmarkFunction):
    """Brown and Dennis function."""

    dim_out: int
    """The number of residuals."""

    differentiable = True
    min_n_params = 4
    max_n_params = 4

    def fun(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return brown_dennis(x, dim_out=self.dim_out)


@mark.least_squares
@dataclass(frozen=True)
class Chebyquad(BenchmarkFunction):
    """Chebyquad function."""

    dim_out: int
    """The number of residuals. Must not be smaller than the number of parameters."""

    differentiable = True

    def fun(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return chebyquad(x, dim_out=self.dim_out)

    def supports_n_params(self, n_params: int) -> bool:
        return 1 <= n_params <= self.dim_out


@mark.least_squares
@dataclass(frozen=True)
class BrownAlmostLinear(BenchmarkFunction):
    """Brown almost-linear function."""

    differentiable = True

    def fun(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return brown_almost_linear(x)


@mark.least_squares
@dataclass(frozen=True)
class OsborneOne(BenchmarkFunction):
    """Osborne 1 function."""

    differentiable = True
    min_n_params = 5
    max_n_params = 5

    def fun(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return osborne_one(x, y=y4_vec)


@mark.least_squares
@dataclass(frozen=True)
class OsborneTwo(BenchmarkFunction):
    """Osborne 2 function."""

    differentiable = True
    min_n_params = 11
    max_n_params = 11

    def fun(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return osborne_two(x, y=y5_vec)


@mark.least_squares
@dataclass(frozen=True)
class Bdqrtic(BenchmarkFunction):
    """Banded quartic function (BDQRTIC)."""

    differentiable = True
    min_n_params = 5

    def fun(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return bdqrtic(x)


@mark.least_squares
@dataclass(frozen=True)
class Cube(BenchmarkFunction):
    """Cube function."""

    differentiable = True
    min_n_params = 2

    def fun(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return cube(x)


@mark.least_squares
@dataclass(frozen=True)
class Mancino(BenchmarkFunction):
    """Mancino function."""

    differentiable = True

    def fun(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return mancino(x)


@mark.least_squares
@dataclass(frozen=True)
class HeartEight(BenchmarkFunction):
    """Heart function with eight parameters."""

    differentiable = True
    min_n_params = 8
    max_n_params = 8

    def fun(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return heart_eight(x, y=heart_eight_y_vec)


# ======================================================================================
# Benchmark problems
# ======================================================================================


def _problem(
    function: BenchmarkFunction,
    start_x: list[float],
    solution_x: list[float] | None,
    solution_fun: float,
) -> BenchmarkProblem:
    """Create a benchmark problem from the information in the problem table.

    Args:
        function: The benchmark function.
        start_x: The start parameters.
        solution_x: The parameters at the solution or None if they are not unique.
        solution_fun: The sum of squared residuals at the solution.

    Returns:
        The benchmark problem. The solution is marked as exact if solution_fun is zero
        because a sum of squares cannot be smaller. All other solutions are the best
        known values.

    """
    solution_params = None
    if solution_x is not None:
        solution_params = np.array(solution_x, dtype=np.float64)

    return BenchmarkProblem(
        function=function,
        start_params=np.array(start_x, dtype=np.float64),
        solution=BenchmarkSolution(
            fun=float(solution_fun), params=solution_params, exact=solution_fun == 0
        ),
    )


MORE_WILD_PROBLEMS: dict[str, BenchmarkProblem] = {
    "linear_full_rank_good_start": _problem(
        function=LinearFullRank(dim_out=45),
        start_x=[1] * 9,
        solution_x=linear_full_rank_solution_x,
        solution_fun=36,
    ),
    "linear_full_rank_bad_start": _problem(
        function=LinearFullRank(dim_out=45),
        start_x=[10] * 9,
        solution_x=linear_full_rank_solution_x,
        solution_fun=36,
    ),
    "linear_rank_one_good_start": _problem(
        function=LinearRankOne(dim_out=35),
        start_x=[1] * 7,
        # no unique solution
        solution_x=None,
        solution_fun=8.380281690143324,
    ),
    "linear_rank_one_bad_start": _problem(
        function=LinearRankOne(dim_out=35),
        start_x=[10] * 7,
        # no unique solution
        solution_x=None,
        solution_fun=8.380282,
    ),
    "linear_rank_one_zero_columns_rows_good_start": _problem(
        function=LinearRankOneZeroColumnsRows(dim_out=35),
        start_x=[1] * 7,
        # no unique solution
        solution_x=None,
        solution_fun=9.880597014926506,
    ),
    "linear_rank_one_zero_columns_rows_bad_start": _problem(
        function=LinearRankOneZeroColumnsRows(dim_out=35),
        start_x=[10] * 7,
        # no unique solution
        solution_x=None,
        solution_fun=9.880597014926506,
    ),
    "rosenbrock_good_start": _problem(
        function=Rosenbrock(),
        start_x=[-1.2, 1],
        solution_x=[1, 1],
        solution_fun=0,
    ),
    "rosenbrock_bad_start": _problem(
        function=Rosenbrock(),
        start_x=[-12, 10],
        solution_x=[1, 1],
        solution_fun=0,
    ),
    "helical_valley_good_start": _problem(
        function=HelicalValley(),
        start_x=[-1, 0, 0],
        solution_x=[1, 0, 0],
        solution_fun=0,
    ),
    "helical_valley_bad_start": _problem(
        function=HelicalValley(),
        start_x=[-10, 0, 0],
        solution_x=[1, 0, 0],
        solution_fun=0,
    ),
    "powell_singular_good_start": _problem(
        function=PowellSingular(),
        start_x=[3, -1, 0, 1],
        solution_x=[0] * 4,
        solution_fun=0,
    ),
    "powell_singular_bad_start": _problem(
        function=PowellSingular(),
        start_x=[30, -10, 0, 10],
        solution_x=[0] * 4,
        solution_fun=0,
    ),
    "freudenstein_roth_good_start": _problem(
        function=FreudensteinRoth(),
        start_x=[0.5, -2],
        solution_x=freudenstein_roth_solution_x,
        solution_fun=48.984253679240013,
    ),
    "freudenstein_roth_bad_start": _problem(
        function=FreudensteinRoth(),
        start_x=[5, -20],
        solution_x=freudenstein_roth_solution_x,
        solution_fun=48.984253679240013,
    ),
    "bard_good_start": _problem(
        function=Bard(),
        start_x=[1] * 3,
        solution_x=bard_solution_x,
        solution_fun=0.00821487730657897,
    ),
    "bard_bad_start": _problem(
        function=Bard(),
        start_x=[10] * 3,
        solution_x=bard_solution_x,
        solution_fun=0.00821487730657897,
    ),
    "kowalik_osborne": _problem(
        function=KowalikOsborne(),
        start_x=[0.25, 0.39, 0.415, 0.39],
        solution_x=kowalik_osborne_solution_x,
        solution_fun=0.00030750560384924,
    ),
    "meyer": _problem(
        function=Meyer(),
        start_x=[0.02, 4000, 250],
        solution_x=meyer_solution_x,
        solution_fun=87.945855170395831,
    ),
    "watson_6_good_start": _problem(
        function=Watson(),
        start_x=[0.5] * 6,
        solution_x=watson_6_solution_x,
        solution_fun=0.00228767005355236,
    ),
    "watson_6_bad_start": _problem(
        function=Watson(),
        start_x=[5] * 6,
        solution_x=watson_6_solution_x,
        solution_fun=0.00228767005355236,
    ),
    "watson_9_good_start": _problem(
        function=Watson(),
        start_x=[0.5] * 9,
        solution_x=watson_9_solution_x,
        solution_fun=1.399760e-6,
    ),
    "watson_9_bad_start": _problem(
        function=Watson(),
        start_x=[5] * 9,
        solution_x=watson_9_solution_x,
        solution_fun=1.399760e-6,
    ),
    "watson_12_good_start": _problem(
        function=Watson(),
        start_x=[0.5] * 12,
        solution_x=watson_12_solution_x,
        solution_fun=4.722381e-10,
    ),
    "watson_12_bad_start": _problem(
        function=Watson(),
        start_x=[5] * 12,
        solution_x=watson_12_solution_x,
        solution_fun=4.722381e-10,
    ),
    "box_3d": _problem(
        function=Box3D(dim_out=10),
        start_x=[0, 10, 20],
        solution_x=[1, 10, 1],
        solution_fun=0,
    ),
    "jennrich_sampson": _problem(
        function=JennrichSampson(dim_out=10),
        start_x=[0.3, 0.4],
        solution_x=[0.2578252135686162] * 2,
        solution_fun=124.3621823556148,
    ),
    "brown_dennis_good_start": _problem(
        function=BrownDennis(dim_out=20),
        start_x=[25, 5, -5, -1],
        solution_x=brown_dennis_solution_x,
        solution_fun=85822.20162635,
    ),
    "brown_dennis_bad_start": _problem(
        function=BrownDennis(dim_out=20),
        start_x=[250, 50, -50, -10],
        solution_x=brown_dennis_solution_x,
        solution_fun=85822.20162635,
    ),
    "chebyquad_6": _problem(
        function=Chebyquad(dim_out=6),
        start_x=[i / 7 for i in range(1, 7)],
        solution_x=chebyquad_6_solution_x,
        solution_fun=0,
    ),
    "chebyquad_7": _problem(
        function=Chebyquad(dim_out=7),
        start_x=[i / 8 for i in range(1, 8)],
        solution_x=chebyquad_7_solution_x,
        solution_fun=0,
    ),
    "chebyquad_8": _problem(
        function=Chebyquad(dim_out=8),
        start_x=[i / 9 for i in range(1, 9)],
        solution_x=chebyquad_8_solution_x,
        solution_fun=0.003516873725677,
    ),
    "chebyquad_9": _problem(
        function=Chebyquad(dim_out=9),
        start_x=[i / 10 for i in range(1, 10)],
        solution_x=chebyquad_9_solution_x,
        solution_fun=0,
    ),
    "chebyquad_10": _problem(
        function=Chebyquad(dim_out=10),
        start_x=[i / 11 for i in range(1, 11)],
        solution_x=chebyquad_10_solution_x,
        solution_fun=0.00477271369637536,
    ),
    "chebyquad_11": _problem(
        function=Chebyquad(dim_out=11),
        start_x=[i / 12 for i in range(1, 12)],
        solution_x=chebyquad_11_solution_x,
        solution_fun=0.00279976155186576,
    ),
    "brown_almost_linear": _problem(
        function=BrownAlmostLinear(),
        start_x=[0.5] * 10,
        solution_x=[1] * 10,
        solution_fun=0,
    ),
    "osborne_one": _problem(
        function=OsborneOne(),
        start_x=[0.5, 1.5, 1, 0.01, 0.02],
        solution_x=osborne_one_solution_x,
        solution_fun=0.00005464894697483,
    ),
    "osborne_two_good_start": _problem(
        function=OsborneTwo(),
        start_x=[1.3, 0.65, 0.65, 0.7, 0.6, 3, 5, 7, 2, 4.5, 5.5],
        solution_x=osborne_two_solution_x,
        solution_fun=0.0401377362935477,
    ),
    "osborne_two_bad_start": _problem(
        function=OsborneTwo(),
        start_x=[13, 6.5, 6.5, 7, 6, 30, 50, 70, 20, 45, 55],
        solution_x=osborne_two_solution_x,
        solution_fun=0.0401377362935477,
    ),
    "bdqrtic_8": _problem(
        function=Bdqrtic(),
        start_x=[1] * 8,
        solution_x=bdqrtic_8_solution_x,
        solution_fun=10.2389734213174,
    ),
    "bdqrtic_10": _problem(
        function=Bdqrtic(),
        start_x=[1] * 10,
        solution_x=bdqrtic_10_solution_x,
        solution_fun=18.28116175359353,
    ),
    "bdqrtic_11": _problem(
        function=Bdqrtic(),
        start_x=[1] * 11,
        solution_x=bdqrtic_11_solution_x,
        solution_fun=22.260591734883817,
    ),
    "bdqrtic_12": _problem(
        function=Bdqrtic(),
        start_x=[1] * 12,
        solution_x=bdqrtic_12_solution_x,
        solution_fun=26.2727663967939,
    ),
    "cube_5": _problem(
        function=Cube(),
        start_x=[0.5] * 5,
        solution_x=[1] * 5,
        solution_fun=0,
    ),
    "cube_6": _problem(
        function=Cube(),
        start_x=[0.5] * 6,
        solution_x=[1] * 6,
        solution_fun=0,
    ),
    "cube_8": _problem(
        function=Cube(),
        start_x=[0.5] * 8,
        solution_x=[1] * 8,
        solution_fun=0,
    ),
    "mancino_5_good_start": _problem(
        function=Mancino(),
        start_x=get_start_points_mancino(5),
        solution_x=mancino_5_solution_x,
        solution_fun=0,
    ),
    "mancino_5_bad_start": _problem(
        function=Mancino(),
        start_x=get_start_points_mancino(5, 10),
        solution_x=mancino_5_solution_x,
        solution_fun=0,
    ),
    "mancino_8": _problem(
        function=Mancino(),
        start_x=get_start_points_mancino(8),
        solution_x=mancino_8_solution_x,
        solution_fun=0,
    ),
    "mancino_10": _problem(
        function=Mancino(),
        start_x=get_start_points_mancino(10),
        solution_x=mancino_10_solution_x,
        solution_fun=0,
    ),
    "mancino_12_good_start": _problem(
        function=Mancino(),
        start_x=get_start_points_mancino(12),
        solution_x=mancino_12_solution_x,
        solution_fun=0,
    ),
    "mancino_12_bad_start": _problem(
        function=Mancino(),
        start_x=get_start_points_mancino(12, 10),
        solution_x=mancino_12_solution_x,
        solution_fun=0,
    ),
    "heart_eight_good_start": _problem(
        function=HeartEight(),
        start_x=[-0.3, -0.39, 0.3, -0.344, -1.2, 2.69, 1.59, -1.5],
        solution_x=heart_eight_solution_x,
        solution_fun=0,
    ),
    "heart_eight_bad_start": _problem(
        function=HeartEight(),
        start_x=[-3, -3.9, 3, -3.44, -12, 26.9, 15.9, -15],
        solution_x=heart_eight_solution_x,
        solution_fun=0,
    ),
    "brown_almost_linear_medium": _problem(
        function=BrownAlmostLinear(),
        start_x=[0.5] * 100,
        solution_x=[1] * 100,
        solution_fun=0,
    ),
}
