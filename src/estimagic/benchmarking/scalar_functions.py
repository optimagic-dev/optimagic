"""Define the Scalar Benchmark Set.

This benchmark set contains 78+ test cases for single objective optimization.
The test cases are built out of 78 functions, which originate from the python
implementation of benchmark test functions for single objective optimization by
Axel Thevenot.

We use the following sources of information to construct the
benchmark set:

- https://towardsdatascience.com/optimization-eye-pleasure-78-benchmark-test-functions-for-single-objective-optimization-92e7ed1d1f12
- https://github.com/AxelThevenot/Python_Benchmark_Test_Optimization_Function_Single_Objective
"""
import numpy as np
from estimagic.benchmarking import more_wild as mw


def ackley(x):
    out = (
        -20 * np.exp(-0.2 * np.sqrt(np.mean(x**2)))
        - np.exp(np.mean(np.cos(2 * np.pi * x)))
        + 20
        + np.exp(1)
    )
    return out


def ackley2(x):
    x_1, x_2 = x
    out = -200 * np.exp(-0.2 * np.sqrt(x_1**2 + x_2**2))
    return out


def ackley3(x):
    x_1, x_2 = x
    out = -200 * np.exp(-0.2 * np.sqrt(x_1**2 + x_2**2)) + 5 * np.exp(
        np.cos(3 * x_1) + np.sin(3 * x_2)
    )
    return out


def ackley4(x):
    x_1, x_2 = x
    out = np.sum(
        np.exp(-0.2) * np.sqrt(x_1**2 + x_2**2)
        + 3 * (np.cos(2 * x_1) + np.sin(2 * x_2))
    )
    return out


def adjiman(x):
    x_1, x_2 = x
    out = np.cos(x_1) * np.sin(x_2) - x_1 / (x_2**2 + 1)
    return out


def alpine1(x):
    out = np.sum(np.abs(x * np.sin(x) + 0.1 * x))
    return out


def alpine2(x):
    out = -np.prod(np.sqrt(x) * np.sin(x))
    return out


def bartels(x):
    x_1, x_2 = x
    out = (
        np.abs(x_1**2 + x_2**2 + x_1 * x_2)
        + np.abs(np.sin(x_1))
        + np.abs(np.cos(x_2))
    )
    return out


def beale(x):
    x_1, x_2 = x
    out = (
        (1.5 - x_1 + x_1 * x_2) ** 2
        + (2.25 - x_1 + x_1 * x_2**2) ** 2
        + (2.625 - x_1 + x_1 * x_2**3) ** 2
    )
    return out


def bird(x):
    x_1, x_2 = x
    out = (
        np.sin(x_1) * np.exp((1 - np.cos(x_2)) ** 2)
        + np.cos(x_2) * np.exp((1 - np.sin(x_1)) ** 2)
        + (x_1 - x_2) ** 2
    )
    return out


def bohachevsky1(x):
    x_1, x_2 = x
    out = (
        x_1**2
        + 2 * x_2**2
        - 0.3 * np.cos(3 * np.pi * x_1)
        - 0.4 * np.cos(4 * np.pi * x_2)
        + 0.7
    )
    return out


def bohachevsky2(x):
    x_1, x_2 = x
    out = (
        x_1**2
        + 2 * x_2**2
        - 0.3 * np.cos(3 * np.pi * x_1) * np.cos(4 * np.pi * x_2)
        + 0.3
    )
    return out


def bohachevsky3(x):
    x_1, x_2 = x
    out = (
        x_1**2
        + 2 * x_2**2
        - 0.3 * np.cos(3 * np.pi * x_1 + 4 * np.pi * x_2) * np.cos(4 * np.pi * x_2)
        + 0.3
    )
    return out


def booth(x):
    x_1, x_2 = x
    out = (x_1 + 2 * x_2 - 7) ** 2 + (2 * x_1 + x_2 - 5) ** 2
    return out


def branin(x):
    x_1, x_2 = x
    out = (
        (x_2 - 5.1 / (4 * np.pi**2) * x_1**2 + 5 / np.pi * x_1 - 6) ** 2
        + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x_1)
        + 10
    )
    return out


def brent(x):
    x_1, x_2 = x
    out = (x_1 + 10) ** 2 + (x_2 + 10) ** 2 + np.exp(-(x_1**2) - x_2**2)
    return out


def brown(x):
    x_1, x_2 = x
    out = np.sum((x_1**2) ** (x_2**2 + 1) + (x_2**2) ** (x_1**2 + 1))
    return out


def bukin6(x):
    x_1, x_2 = x
    out = 100 * np.sqrt(np.abs(x_2 - 0.01 * x_1**2)) + 0.01 * np.abs(x_1 + 10)
    return out


def colville(x):
    x_1, x_2, x_3, x_4 = x
    out = (
        100 * (x_1**2 - x_2) ** 2
        + (x_1 - 1) ** 2
        + (x_3 - 1) ** 2
        + 90 * (x_3**2 - x_4) ** 2
        + 10.1 * ((x_2 - 1) ** 2 + (x_4 - 1) ** 2)
        + 19.8 * (x_2 - 1) * (x_4 - 1)
    )
    return out


def crossintray(x):
    x_1, x_2 = x
    out = (
        -0.0001
        * (
            np.abs(np.sin(x_1) * np.sin(x_2))
            * np.exp(np.abs(100 - np.sqrt(x_1**2 + x_2**2) / np.pi))
            + 1
        )
        ** 0.1
    )
    return out


def dejong5(x):
    x_1, x_2 = x
    b = [-32, -16, 0, 16, 32]
    a = np.array([[x_1, x_2] for x_1 in b for x_2 in b])
    out = (
        0.002
        + np.sum(
            [
                1 / ((i + 1) + (x_1 - a1) ** 6 + (x_2 - a2) ** 6)
                for i, (a1, a2) in enumerate(a)
            ]
        )
    ) ** -1
    return out


def deckkersaarts(x):
    x_1, x_2 = x
    out = (
        1e5 * x_1**2
        + x_2**2
        - (x_1**2 + x_2**2) ** 2
        + 1e-5 * (x_1**2 + x_2**2) ** 4
    )
    return out


def dixonprice(x):
    d = x.shape[0]
    out = (x[0] - 1) ** 2 + np.sum(
        [(i + 1) * (2 * x[i] ** 2 - x[i - 1]) ** 2 for i in range(1, d)]
    )
    return out


def dropwave(x):
    x_1, x_2 = x
    out = -(1 + np.cos(12 * np.sqrt(x_1**2 + x_2**2))) / (
        0.5 * (x_1**2 + x_2**2) + 2
    )
    return out


def easom(x):
    x_1, x_2 = x
    out = (
        -np.cos(x_1) * np.cos(x_2) * np.exp(-((x_1 - np.pi) ** 2) - (x_2 - np.pi) ** 2)
    )
    return out


def eggcrate(x):
    x_1, x_2 = x
    out = x_1**2 + x_2**2 + 25 * (np.sin(x_1) ** 2 + np.sin(x_2) ** 2)
    return out


def eggholder(x):
    x_1, x_2 = x
    out = -(x_2 + 47) * np.sin(np.sqrt(np.abs(x_2 + x_1 / 2 + 47))) - x_1 * np.sin(
        np.sqrt(np.abs(x_1 - x_2 - 47))
    )
    return out


def exponential(x):
    out = -np.exp(-0.5 * np.sum(x**2))
    return out


def forrester(x):
    out = (6 * x - 2) ** 2 * np.sin(12 * x - 4)
    return out


def goldsteinprice(x):
    x_1, x_2 = x
    out = (
        1
        + (x_1 + x_2 + 1) ** 2
        * (19 - 14 * x_1 + 3 * x_1**2 - 14 * x_2 + 6 * x_1 * x_2 + 3 * x_2**2)
    ) * (
        30
        + (2 * x_1 - 3 * x_2) ** 2
        * (18 - 32 * x_1 + 12 * x_1**2 + 48 * x_2 - 36 * x_1 * x_2 + 27 * x_2**2)
    )
    return out


def gramacylee(x):
    out = np.sin(10 * np.pi * x) / (2 * x) + (x - 1) ** 4
    return out


def griewank(x):
    d = x.shape[0]
    i = np.arange(1, d + 1)
    out = 1 + np.sum(x**2 / 4000) - np.prod(np.cos(x / np.sqrt(i)))
    return out


def happycat(x):
    d = x.shape[0]
    out = (
        ((np.linalg.norm(x) ** 2 - d) ** 2) ** 0.5
        + (1 / d) * (0.5 * np.linalg.norm(x) ** 2 + np.sum(x))
        + 0.5
    )
    return out


def himmelblau(x):
    x_1, x_2 = x
    out = (x_1**2 + x_2 - 11) ** 2 + (x_1 + x_2**2 - 7) ** 2
    return out


def holdertable(x):
    x_1, x_2 = x
    out = -np.abs(
        np.sin(x_1)
        * np.cos(x_2)
        * np.exp(np.abs(1 - np.sqrt(x_1**2 + x_2**2) / np.pi))
    )
    return out


def keane(x):
    x_1, x_2 = x
    out = -(np.sin(x_1 - x_2) ** 2 * np.sin(x_1 + x_2) ** 2) / np.sqrt(
        x_1**2 + x_2**2
    )
    return out


def langermann(x):
    c = np.array([1, 2, 5, 2, 3])
    a = np.array([[3, 5], [5, 2], [2, 1], [1, 4], [7, 9]])
    out = -np.sum(
        [
            c[i]
            * np.exp(-1 / np.pi * np.sum((x - a[i]) ** 2))
            * np.cos(np.pi * np.sum((x - a[i]) ** 2))
            for i in range(5)
        ]
    )
    return out


def leon(x):
    x_1, x_2 = x
    out = 100 * (x_2 - x_1**3) ** 2 + (1 - x_1) ** 2
    return out


def levy13(x):
    x_1, x_2 = x
    out = (
        np.sin(3 * np.pi * x_1) ** 2
        + (x_1 - 1) ** 2 * (1 + np.sin(3 * np.pi * x_2) ** 2)
        + (x_2 - 1) ** 2 * (1 + np.sin(2 * np.pi * x_2) ** 2)
    )
    return out


def matyas(x):
    x_1, x_2 = x
    out = 0.26 * (x_1**2 + x_2**2) - 0.48 * x_1 * x_2
    return out


def mccormick(x):
    x_1, x_2 = x
    out = np.sin(x_1 + x_2) + (x_1 - x_2) ** 2 - 1.5 * x_1 + 2.5 * x_2 + 1
    return out


def michalewicz(x, m=10):
    d = x.shape[0]
    i = np.arange(1, d + 1)
    out = -np.sum(np.sin(x) * np.sin(i * x**2 / np.pi) ** (2 * m))
    return out


def periodic(x):
    out = 1 + np.sum(np.sin(x) ** 2) - 0.1 * np.exp(-np.sum(x**2))
    return out


def permzerodbeta(x, b=10):
    d = x.shape[0]
    out = np.sum(
        [
            (
                np.sum(
                    [
                        (((j + 1) + b) * (x[j] ** (i + 1) - (j + 1) ** -(i + 1)))
                        for j in range(d)
                    ]
                )
            )
            ** 2
            for i in range(d)
        ]
    )
    return out


def permdbeta(x, b=0.5):
    d = x.shape[0]
    j = np.arange(1, d + 1)
    out = np.sum(
        [np.sum((j**i + b) * ((x / j) ** i - 1)) ** 2 for i in range(1, d + 1)]
    )
    return out


def powell(x):
    d = x.shape[0]
    out = np.sum(np.abs(x) ** np.arange(2, d + 2))
    return out


def qing(x):
    d = x.shape[0]
    x_1 = np.power(x, 2)
    res = 0
    for i in range(d):
        out = res + np.power(x_1[i] - (i + 1), 2)
    return out


def quartic(x):
    np.random.seed(3)
    d = x.shape[0]
    out = np.sum(np.arange(1, d + 1) * x**4) + np.random.random()
    return out


def rastrigin(x):
    d = x.shape[0]
    out = 10 * d + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
    return out


def ridge(x, a=0.1, b=2):
    out = x[0] + b * np.sum(x[1:] ** 2) ** a
    return out


def rosenbrock(x):
    out = mw.rosenbrock(x) @ mw.rosenbrock(x)
    return out


def rotatedhyperellipsoid(x):
    d = x.shape[0]
    out = np.sum([np.sum(x[: i + 1] ** 2) for i in range(d)])
    return out


def salomon(x):
    res = 1 - np.cos(2 * np.pi * np.sqrt(np.sum(x**2)))
    out = res + 0.1 * np.sqrt(np.sum(x**2))
    return out


def schaffer1(x):
    x_1, x_2 = x
    out = (
        0.5
        + (np.sin((x_1**2 + x_2**2) ** 2) ** 2 - 0.5)
        / (1 + 0.001 * (x_1**2 + x_2**2)) ** 2
    )
    return out


def schaffer2(x):
    x_1, x_2 = x
    out = (
        0.5
        + (np.sin((x_1**2 + x_2**2)) ** 2 - 0.5)
        / (1 + 0.001 * (x_1**2 + x_2**2)) ** 2
    )
    return out


def schaffer3(x):
    x_1, x_2 = x
    out = (
        0.5
        + (np.sin(np.cos(np.abs(x_1**2 + x_2**2))) ** 2 - 0.5)
        / (1 + 0.001 * (x_1**2 + x_2**2)) ** 2
    )
    return out


def schaffer4(x):
    x_1, x_2 = x
    out = (
        0.5
        + (np.cos(np.sin(np.abs(x_1**2 + x_2**2))) ** 2 - 0.5)
        / (1 + 0.001 * (x_1**2 + x_2**2)) ** 2
    )
    return out


def schwefel(x):
    d = x.shape[0]
    out = 418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))))
    return out


def schwefel2_20(x):
    out = np.sum(np.abs(x))
    return out


def schwefel2_21(x):
    out = np.max(np.abs(x))
    return out


def schwefel2_22(x):
    out = np.sum(np.abs(x)) + np.prod(np.abs(x))
    return out


def schwefel2_23(x):
    out = np.sum(x**10)
    return out


def shekel(x):
    x_1, x_2, x_3, x_4 = x
    m = 10
    b = 1 / 10 * np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5])
    c = np.array(
        [
            [4, 4, 4, 4],
            [1, 1, 1, 1],
            [8, 8, 8, 8],
            [6, 6, 6, 6],
            [3, 7, 3, 7],
            [2, 9, 2, 9],
            [5, 3, 5, 3],
            [8, 1, 8, 1],
            [6, 2, 6, 2],
            [7, 3.6, 7, 3.6],
        ]
    )
    out = -np.sum(
        [
            (np.sum([(x[j] - c[i, j]) ** 2 for j in range(4)]) + b[i]) ** -1
            for i in range(m)
        ]
    )
    return out


def shubert(x):
    x_1, x_2 = x
    out = (np.sum([j * np.cos((j + 1) * x_1 + j) for j in range(6)])) * (
        np.sum([j * np.cos((j + 1) * x_2 + j) for j in range(6)])
    )
    return out


def shubert3(x):
    out = np.sum(np.sum([j * np.sin((j + 1) * x + j) for j in range(1, 5 + 1)]))
    return out


def shubert4(x):
    out = np.sum(np.sum([j * np.cos((j + 1) * x + j) for j in range(1, 5 + 1)]))
    return out


def sphere(x):
    out = np.sum(x**2)
    return out


def styblinskitank(x):
    out = 0.5 * np.sum(x**4 - 16 * x**2 + 5 * x)
    return out


def sumsquares(x):
    d = x.shape[0]
    i = np.arange(1, d + 1)
    out = np.sum(i * x**2)
    return out


def threehump(x):
    x_1, x_2 = x
    out = 2 * x_1**2 - 1.05 * x_1**4 + x_1**6 * (1 / 6) + x_1 * x_2 + x_2**2
    return out


def thevenot(x, m=5, b=15):
    res = np.exp(-np.sum((x / b) ** (2 * m)))
    out = res - 2 * np.exp(-np.prod(x**2)) * np.prod(np.cos(x) ** 2)
    return out


def trid(x):
    out = np.sum(x - 1) ** 2 - np.sum(x[1:] * x[:-1])
    return out


def wolfe(x):
    x_1, x_2, x_3 = x
    out = 4 / 3 * (x_1**2 + x_2**2 - x_1 * x_2) ** 0.75 + x_3
    return out


def xinsheyang(x):
    np.random.seed(3)
    d = x.shape[0]
    i = np.arange(1, d + 1)
    rand = np.random.random(d)
    out = np.sum(rand * np.abs(x) ** i)
    return out


def xinsheyang2(x):
    out = np.sum(np.abs(x)) * np.exp(-np.sum(np.sin(x**2)))
    return out


def xinsheyang3(x, m=5, beta=15):
    res = np.exp(-np.sum((x / beta) ** (2 * m)))
    out = res - 2 * np.exp(-np.sum(x**2)) * np.prod(np.cos(x) ** 2)
    return out


def xinsheyang4(x):
    out = np.sum(np.sin(x) ** 2 - np.exp(-np.sum(x) ** 2)) * np.exp(
        -np.sum(np.sin(np.sqrt(np.abs(x))) ** 2)
    )
    return out


# According to the website global minimum at f(0,..,0) = -1,
# but according to the internet global minimum at f(0,..,0) = 0
def zakharov(x):
    d = x.shape[0]
    i = np.arange(1, d + 1)
    out = np.sum(x**2) + np.sum(0.5 * i * x) ** 2 + np.sum(0.5 * i * x) ** 4
    return out


SCALAR_FUNCTIONS_PROBLEMS = {
    "ackley_good_start": {
        "criterion": ackley,
        "start_x": np.full(10, 3),
        "solution_x": np.zeros(10),
        "start_criterion": 9.023767278119472,
        "solution_criterion": 0,
    },
    "ackley_bad_start": {
        "criterion": ackley,
        "start_x": np.full(10, 30),
        "solution_x": np.zeros(10),
        "start_criterion": 19.950424956466673,
        "solution_criterion": 0,
    },
    "ackley2_good_start": {
        "criterion": ackley2,
        "start_x": np.full(2, 3),
        "solution_x": np.zeros(2),
        "start_criterion": -85.60889823804698,
        "solution_criterion": -200,
    },
    "ackley2_bad_start": {
        "criterion": ackley2,
        "start_x": np.full(2, 25),
        "solution_x": np.zeros(2),
        "start_criterion": -0.1698651409438339,
        "solution_criterion": -200,
    },
    "ackley3_good_start": {
        "criterion": ackley3,
        "start_x": np.full(2, 3),
        "solution_x": np.zeros(2),
        "start_criterion": -82.57324651934985,
        "solution_criterion": -186.40859085770478,
    },
    "ackley3_bad_start": {
        "criterion": ackley3,
        "start_x": np.full(2, 25),
        "solution_x": np.zeros(2),
        "start_criterion": 8.358584120180984,
        "solution_criterion": -186.40859085770478,
    },
    "ackley4_good_start": {
        "criterion": ackley4,
        "start_x": np.full(2, 3),
        "solution_x": np.array([-1.51, -0.755]),
        "start_criterion": 5.515844770158779,
        "solution_criterion": -4.5901006651507235,
    },
    "ackley4_bad_start": {
        "criterion": ackley4,
        "start_x": np.full(2, 25),
        "solution_x": np.array([-1.51, -0.755]),
        "start_criterion": 31.054276897735043,
        "solution_criterion": -4.5901006651507235,
    },
    "adjiman": {
        "criterion": adjiman,
        "start_x": np.array([-1, 1]),
        "solution_x": np.array([2, 0.10578]),
        "start_criterion": 0.954648713412841,
        "solution_criterion": -2.0218067833370204,
    },
    "alpine1_good_start": {
        "criterion": alpine1,
        "start_x": np.full(10, 2),
        "solution_x": np.zeros(10),
        "start_criterion": 20.18594853651364,
        "solution_criterion": 0,
    },
    "alpine1_bad_start": {
        "criterion": alpine1,
        "start_x": np.full(10, 10),
        "solution_x": np.zeros(10),
        "start_criterion": 44.40211108893698,
        "solution_criterion": 0,
    },
    "alpine2_good_start": {
        "criterion": alpine2,
        "start_x": np.full(10, 9),
        "solution_x": np.full(10, 7.917),
        "start_criterion": -8.345137486473694,
        "solution_criterion": -30491.15748225926,
    },
    "alpine2_bad_start": {
        "criterion": alpine2,
        "start_x": np.ones(10),
        "solution_x": np.full(10, 7.917),
        "start_criterion": -0.177988299732403,
        "solution_criterion": -30491.15748225926,
    },
    "bartels_good_start": {
        "criterion": bartels,
        "start_x": np.array([50, -50]),
        "solution_x": np.full(2, 0),
        "start_criterion": 2501.227340882196,
        "solution_criterion": 1,
    },
    "bartels_bad_start": {
        "criterion": bartels,
        "start_x": np.full(2, 400),
        "solution_x": np.full(2, 0),
        "start_criterion": 480001.3762156983,
        "solution_criterion": 1,
    },
    "beale_good_start": {
        "criterion": beale,
        "start_x": np.zeros(2),
        "solution_x": np.array([3, 0.5]),
        "start_criterion": 14.203125,
        "solution_criterion": 0,
    },
    "beale_bad_start": {
        "criterion": beale,
        "start_x": np.array([4, -4]),
        "solution_x": np.array([3, 0.5]),
        "start_criterion": 70459.203125,
        "solution_criterion": 0,
    },
    "bird_good_start": {
        "criterion": bird,
        "start_x": np.array([2, -6]),
        # no unique solution
        "solution_x": None,
        "start_criterion": 65.87884323549846,
        "solution_criterion": -106.764537,
    },
    "bird_bad_start": {
        "criterion": bird,
        "start_x": np.array([-1.8, 3]),
        # no unique solution
        "solution_x": None,
        "start_criterion": -76.76546590907114,
        "solution_criterion": -106.764537,
    },
    "bohachevsky1": {
        "criterion": bohachevsky1,
        "start_x": np.full(2, 50),
        "solution_x": np.zeros(2),
        "start_criterion": 7500,
        "solution_criterion": 0,
    },
    "bohachevsky2": {
        "criterion": bohachevsky2,
        "start_x": np.full(2, 50),
        "solution_x": np.zeros(2),
        "start_criterion": 7500,
        "solution_criterion": 0,
    },
    "bohachevsky3": {
        "criterion": bohachevsky3,
        "start_x": np.full(2, 25),
        "solution_x": np.zeros(2),
        "start_criterion": 1875.6,
        "solution_criterion": 0,
    },
    "booth": {
        "criterion": booth,
        "start_x": np.full(2, -5),
        "solution_x": np.array([1, 3]),
        "start_criterion": 884,
        "solution_criterion": 0,
    },
    "branin_good_start": {
        "criterion": branin,
        "start_x": np.array([0, 6]),
        # no unique solution
        "solution_x": None,
        "start_criterion": 19.602112642270264,
        "solution_criterion": 0.39788735772973816,
    },
    "branin_bad_start": {
        "criterion": branin,
        "start_x": np.array([6, 14]),
        # no unique solution
        "solution_x": None,
        "start_criterion": 185.59494487252852,
        "solution_criterion": 0.39788735772973816,
    },
    "brent": {
        "criterion": brent,
        "start_x": np.full(2, 5),
        "solution_x": np.full(2, -10),
        "start_criterion": 450,
        "solution_criterion": 0,
    },
    "brown": {
        "criterion": brown,
        "start_x": np.full(2, 0.75),
        "solution_x": np.zeros(2),
        "start_criterion": 0.8139475940290111,
        "solution_criterion": 0,
    },
    "bukin6_start_1": {
        "criterion": bukin6,
        "start_x": np.array([-14, -4]),
        "solution_x": np.array([-10, 1]),
        "start_criterion": 244.17111231467405,
        "solution_criterion": 0,
    },
    "bukin6_start_2": {
        "criterion": bukin6,
        "start_x": np.array([-6, 0.35]),
        "solution_x": np.array([-10, 1]),
        "start_criterion": 10.040000000000004,
        "solution_criterion": 0,
    },
    "colville_good_start": {
        "criterion": colville,
        "start_x": np.zeros(4),
        "solution_x": np.full(4, 1),
        "start_criterion": 42,
        "solution_criterion": 0,
    },
    "colville_bad_start": {
        "criterion": colville,
        "start_x": np.full(4, 9),
        "solution_x": np.full(4, 1),
        "start_criterion": 987648,
        "solution_criterion": 0,
    },
    "crossintray_good_start_1": {
        "criterion": crossintray,
        "start_x": np.zeros(2),
        # no unique solution
        "solution_x": None,
        "start_criterion": -0.0001,
        "solution_criterion": -2.06261218,
    },
    "crossintray_good_start_2": {
        "criterion": crossintray,
        "start_x": np.full(2, 1),
        # no unique solution
        "solution_x": None,
        "start_criterion": -2.03424158303853,
        "solution_criterion": -2.06261218,
    },
    "crossintray_bad_start": {
        "criterion": crossintray,
        "start_x": np.full(2, 4),
        # no unique solution
        "solution_x": None,
        "start_criterion": -1.7399663465548592,
        "solution_criterion": -2.06261218,
    },
    "dejong5_good_start": {
        "criterion": dejong5,
        "start_x": np.full(2, -40, dtype=np.int64),
        "solution_x": np.full(2, -32, dtype=np.int64),
        "start_criterion": 499.5200236514581,
        "solution_criterion": 0.998003838818649,
    },
    "dejong5_bad_start": {
        "criterion": dejong5,
        "start_x": np.full(2, 32, dtype=np.int64),
        "solution_x": np.full(2, -32, dtype=np.int64),
        "start_criterion": 23.809436615621898,
        "solution_criterion": 0.998003838818649,
    },
    "deckkersaarts": {
        "criterion": deckkersaarts,
        "start_x": np.array([-6, 0.35]),
        # no unique solution
        "solution_x": None,
        "start_criterion": 3598712.3134376854,
        "solution_criterion": 25628.906250000004,
    },
    "dixonprice_start_1": {
        "criterion": dixonprice,
        "start_x": np.full(2, 2),
        "solution_x": np.array([1, 1 / np.sqrt(2)]),
        "start_criterion": 73,
        "solution_criterion": 0,
    },
    "dixonprice_start_2": {
        "criterion": dixonprice,
        "start_x": np.full(2, 8),
        "solution_x": np.array([1, 1 / np.sqrt(2)]),
        "start_criterion": 28849,
        "solution_criterion": 0,
    },
    "dropwave_good_start": {
        "criterion": dropwave,
        "start_x": np.full(2, 0.5),
        "solution_x": np.zeros(2),
        "start_criterion": -0.18213578404209926,
        "solution_criterion": -1,
    },
    "dropwave_bad_start": {
        "criterion": dropwave,
        "start_x": np.full(2, 5),
        "solution_x": np.zeros(2),
        "start_criterion": -1.645731876039932e-05,
        "solution_criterion": -1,
    },
    "easom": {
        "criterion": easom,
        "start_x": np.zeros(2),
        "solution_x": np.full(2, np.pi),
        "start_criterion": -2.675287991074243e-09,
        "solution_criterion": -1,
    },
    "eggcrate_good_start": {
        "criterion": eggcrate,
        "start_x": np.full(2, 1),
        "solution_x": np.zeros(2),
        "start_criterion": 37.40367091367856,
        "solution_criterion": 0,
    },
    "eggcrate_bad_start": {
        "criterion": eggcrate,
        "start_x": np.full(2, 4),
        "solution_x": np.zeros(2),
        "start_criterion": 60.637500845215335,
        "solution_criterion": 0,
    },
    "eggholder_good_start": {
        "criterion": eggholder,
        "start_x": np.full(2, 400),
        "solution_x": np.array([512, 404.2319]),
        "start_criterion": -350.2552101265452,
        "solution_criterion": -959.6406627106155,
    },
    "eggholder_bad_start_1": {
        "criterion": eggholder,
        "start_x": np.zeros(2),
        "solution_x": np.array([512, 404.2319]),
        "start_criterion": -25.460337185286313,
        "solution_criterion": -959.6406627106155,
    },
    "eggholder_bad_start_2": {
        "criterion": eggholder,
        "start_x": np.array([-400, 400]),
        "solution_x": np.array([512, 404.2319]),
        "start_criterion": -291.18686186178957,
        "solution_criterion": -959.6406627106155,
    },
    "eggholder_bad_start_3": {
        "criterion": eggholder,
        "start_x": np.full(2, -400),
        "solution_x": np.array([512, 404.2319]),
        "start_criterion": -135.9429868637175,
        "solution_criterion": -959.6406627106155,
    },
    "exponential": {
        "criterion": exponential,
        "start_x": np.full(2, 0.5),
        "solution_x": np.zeros(2),
        "start_criterion": -0.7788007830714049,
        "solution_criterion": -1,
    },
    "forrester": {
        "criterion": forrester,
        "start_x": 0.4,
        "solution_x": 0.757249,
        "start_criterion": 0.11477697454392392,
        "solution_criterion": -6.020740055735769,
    },
    "goldsteinprice_good_start": {
        "criterion": goldsteinprice,
        "start_x": np.zeros(2),
        "solution_x": np.array([0, -1]),
        "start_criterion": 600,
        "solution_criterion": 3,
    },
    "goldsteinprice_bad_start": {
        "criterion": goldsteinprice,
        "start_x": np.full(2, 2),
        "solution_x": np.array([0, -1]),
        "start_criterion": 76728,
        "solution_criterion": 3,
    },
    "gramacylee_good_start": {
        "criterion": gramacylee,
        "start_x": 0.65,
        "solution_x": 0.548563444,
        "start_criterion": 0.7842370192307692,
        "solution_criterion": -0.8690111349894997,
    },
    "gramacylee_bad_start": {
        "criterion": gramacylee,
        "start_x": 2,
        "solution_x": 0.548563444,
        "start_criterion": 0.9999999999999993,
        "solution_criterion": -0.8690111349894997,
    },
    "griewank_good_start": {
        "criterion": griewank,
        "start_x": np.full(10, 3),
        "solution_x": np.zeros(10),
        "start_criterion": 1.0225294825857276,
        "solution_criterion": 0,
    },
    "griewank_bad_start": {
        "criterion": griewank,
        "start_x": np.full(10, 15),
        "solution_x": np.zeros(10),
        "start_criterion": 1.5627598318159033,
        "solution_criterion": 0,
    },
    "happycat_good_start": {
        "criterion": happycat,
        "start_x": np.zeros(10),
        "solution_x": np.full(10, -1),
        "start_criterion": 10.5,
        "solution_criterion": 0,
    },
    "happycat_bad_start": {
        "criterion": happycat,
        "start_x": np.full(10, 2),
        "solution_x": np.full(10, -1),
        "start_criterion": 34.50000000000001,
        "solution_criterion": 0,
    },
    "himmelblau_good_start": {
        "criterion": himmelblau,
        "start_x": np.zeros(2),
        # no unique solution
        "solution_x": None,
        "start_criterion": 170,
        "solution_criterion": 0,
    },
    "himmelblau_bad_start": {
        "criterion": himmelblau,
        "start_x": np.full(2, 6),
        # no unique solution
        "solution_x": None,
        "start_criterion": 2186,
        "solution_criterion": 0,
    },
    "holertable_good_start": {
        "criterion": holdertable,
        "start_x": np.full(2, 6),
        # no unique solution
        "solution_x": None,
        "start_criterion": -1.4699802363264503,
        "solution_criterion": -19.208502567767606,
    },
    "holdertable_bad_start": {
        "criterion": holdertable,
        "start_x": np.zeros(2),
        # no unique solution
        "solution_x": None,
        "start_criterion": 0,
        "solution_criterion": -19.208502567767606,
    },
    "keane_good_start": {
        "criterion": keane,
        "start_x": np.full(2, 0.1),
        # no unique solution
        "solution_x": None,
        "start_criterion": 0,
        "solution_criterion": -0.6736675210466049,
    },
    "keane_bad_start": {
        "criterion": keane,
        "start_x": np.full(2, 9),
        # no unique solution
        "solution_x": None,
        "start_criterion": 0,
        "solution_criterion": -0.6736675210466049,
    },
    "leon_good_start": {
        "criterion": leon,
        "start_x": np.full(2, -1),
        "solution_x": np.ones(2),
        "start_criterion": 4,
        "solution_criterion": 0,
    },
    "leon_bad_start": {
        "criterion": leon,
        "start_x": np.array([2, -2]),
        "solution_x": np.ones(2),
        "start_criterion": 10001,
        "solution_criterion": 0,
    },
    "levy13_good_start": {
        "criterion": levy13,
        "start_x": np.zeros(2),
        "solution_x": np.ones(2),
        "start_criterion": 2,
        "solution_criterion": 0,
    },
    "levy13_bad_start_1": {
        "criterion": levy13,
        "start_x": np.full(2, 8),
        "solution_x": np.ones(2),
        "start_criterion": 98,
        "solution_criterion": 0,
    },
    "levy13_bad_start_2": {
        "criterion": levy13,
        "start_x": np.array([8, -8]),
        "solution_x": np.ones(2),
        "start_criterion": 130,
        "solution_criterion": 0,
    },
    "langermann_start_1": {
        "criterion": langermann,
        "start_x": np.array([8, 1]),
        "solution_x": np.array([2.00299219, 1.006096]),
        "start_criterion": -0.08296460236001565,
        "solution_criterion": -5.1621261599638375,
    },
    "langermann_start_2": {
        "criterion": langermann,
        "start_x": np.full(2, 8),
        "solution_x": np.array([2.00299219, 1.006096]),
        "start_criterion": -1.5872521647730398,
        "solution_criterion": -5.1621261599638375,
    },
    "langermann_start_3": {
        "criterion": langermann,
        "start_x": np.full(2, 2),
        "solution_x": np.array([2.00299219, 1.006096]),
        "start_criterion": 4.11663984986056,
        "solution_criterion": -5.1621261599638375,
    },
    "langermann_start_4": {
        "criterion": langermann,
        "start_x": np.zeros(2),
        "solution_x": np.array([2.00299219, 1.006096]),
        "start_criterion": 1.0271573538269159,
        "solution_criterion": -5.1621261599638375,
    },
    "matyas_good_start": {
        "criterion": matyas,
        "start_x": np.full(2, 0.01),
        "solution_x": np.zeros(2),
        "start_criterion": 4.000000000000009e-06,
        "solution_criterion": 0,
    },
    "matyas_bad_start": {
        "criterion": matyas,
        "start_x": np.full(2, 100),
        "solution_x": np.zeros(2),
        "start_criterion": 400.0,
        "solution_criterion": 0,
    },
    "mccormick_good_start": {
        "criterion": mccormick,
        "start_x": np.array([0, -1]),
        "solution_x": np.array([-0.547, -1.547]),
        "start_criterion": -1.3414709848078967,
        "solution_criterion": -1.9132228873800594,
    },
    "mccormick_bad_start": {
        "criterion": mccormick,
        "start_x": np.array([4, -3]),
        "solution_x": np.array([-0.547, -1.547]),
        "start_criterion": 37.3414709848079,
        "solution_criterion": -1.9132228873800594,
    },
    "michalewicz_good_start": {
        "criterion": michalewicz,
        "start_x": np.array([2.5, 1.5]),
        "solution_x": np.array([2.2, 1.57]),
        "start_criterion": -0.9214069505685454,
        "solution_criterion": -1.801140718473825,
    },
    "michalewicz_bad_start": {
        "criterion": michalewicz,
        "start_x": np.array([0.5, 3]),
        "solution_x": np.array([2.2, 1.57]),
        "start_criterion": -3.6755801116192943e-07,
        "solution_criterion": -1.801140718473825,
    },
    "periodic_good_start": {
        "criterion": periodic,
        "start_x": np.full(1, 0.001),
        "solution_x": np.zeros(2),
        "start_criterion": 0.9000010999996166,
        "solution_criterion": 0.9,
    },
    "periodic_bad_start": {
        "criterion": periodic,
        "start_x": np.full(10, 20),
        "solution_x": np.zeros(2),
        "start_criterion": 9.33469030826131,
        "solution_criterion": 0.9,
    },
    "permzerodbeta_good_start": {
        "criterion": permzerodbeta,
        "start_x": np.array([0.2, 0.8]),
        "solution_x": np.array([1, 1 / 2]),
        "start_criterion": 61.614399999999975,
        "solution_criterion": 0,
    },
    "permzerodbeta_bad_start": {
        "criterion": permzerodbeta,
        "start_x": np.array([10, 2]),
        "solution_x": np.array([1, 1 / 2]),
        "start_criterion": 1299645.0,
        "solution_criterion": 0,
    },
    "permdbeta_good_start": {
        "criterion": permdbeta,
        "start_x": np.array([1, 1.75]),
        "solution_x": np.array([1, 2]),
        "start_criterion": 1.21002197265625,
        "solution_criterion": 0,
    },
    "permdbeta_bad_start": {
        "criterion": permdbeta,
        "start_x": np.array([10, 3]),
        "solution_x": np.array([1, 2]),
        "start_criterion": 23972.078125,
        "solution_criterion": 0,
    },
    "powell_good_start": {
        "criterion": powell,
        "start_x": np.full(2, 0.75),
        "solution_x": np.zeros(10),
        "start_criterion": 0.984375,
        "solution_criterion": 0,
    },
    "powell_bad_start": {
        "criterion": powell,
        "start_x": np.full(10, -1),
        "solution_x": np.zeros(10),
        "start_criterion": 10,
        "solution_criterion": 0,
    },
    "qing_good_start": {
        "criterion": qing,
        "start_x": np.array([1.5, 1.5]),
        "solution_x": np.array([2, 2, 2, 2]),
        "start_criterion": 0.0625,
        "solution_criterion": 0,
    },
    "qing_bad_start": {
        "criterion": qing,
        "start_x": np.array([10, 10]),
        "solution_x": np.array([2, 2, 2, 2]),
        "start_criterion": 9604,
        "solution_criterion": 0,
    },
    "quartic_solution": {
        "criterion": quartic,
        "start_x": np.ones(100),
        "solution_x": np.zeros(100),
        "start_criterion": 5050.550797902574,
        "solution_criterion": 0.5507979025745755,
    },
    "rastrigin_good_start": {
        "criterion": rastrigin,
        "start_x": np.full(1, 0.15),
        "solution_x": np.zeros(9),
        "start_criterion": 4.144647477075267,
        "solution_criterion": 0,
    },
    "rastrigin_bad_start": {
        "criterion": rastrigin,
        "start_x": np.full(9, 3),
        "solution_x": np.zeros(9),
        "start_criterion": 81,
        "solution_criterion": 0,
    },
    "ridge_good_start": {
        "criterion": ridge,
        "start_x": np.array([-4, 0]),
        "solution_x": np.array([-1, 0]),
        "start_criterion": -4,
        "solution_criterion": -1,
    },
    "ridge_bad_start": {
        "criterion": ridge,
        "start_x": np.array([2, 2]),
        "solution_x": np.array([-1, 0]),
        "start_criterion": 4.29739670999407,
        "solution_criterion": -1,
    },
    "rosenbrock_good_start": {
        "criterion": rosenbrock,
        "start_x": np.full(3, 0.5),
        "solution_x": np.array([1, 1]),
        "start_criterion": 6.5,
        "solution_criterion": 0,
    },
    "rosenbrock_bad_start": {
        "criterion": rosenbrock,
        "start_x": np.full(10, -1.5),
        "solution_x": np.array([1, 1]),
        "start_criterion": 1412.5,
        "solution_criterion": 0,
    },
    "rotatedhyperellipsoid_good_start": {
        "criterion": rotatedhyperellipsoid,
        "start_x": np.full(2, -0.25),
        "solution_x": np.zeros(10),
        "start_criterion": 0.1875,
        "solution_criterion": 0,
    },
    "rotatedhyperellipsoid_bad_start": {
        "criterion": rotatedhyperellipsoid,
        "start_x": np.full(3, -60),
        "solution_x": np.zeros(10),
        "start_criterion": 21600,
        "solution_criterion": 0,
    },
    "salomon_good_start": {
        "criterion": salomon,
        "start_x": np.full(2, 0.71),
        "solution_x": np.zeros(10),
        "start_criterion": 0.10073960731443472,
        "solution_criterion": 0,
    },
    "salomon_bad_start": {
        "criterion": salomon,
        "start_x": np.full(10, 3),
        "solution_x": np.zeros(10),
        "start_criterion": 2.945263054935206,
        "solution_criterion": 0,
    },
    "schaffer1_good_start": {
        "criterion": schaffer1,
        "start_x": np.array([4, -4]),
        "solution_x": np.array([0, 0]),
        "start_criterion": 0.05412538364222219,
        "solution_criterion": 0,
    },
    "schaffer1_bad_start": {
        "criterion": schaffer1,
        "start_x": np.array([1, 1.5]),
        "solution_x": np.array([0, 0]),
        "start_criterion": 0.8217877372933657,
        "solution_criterion": 0,
    },
    "schaffer2_good_start": {
        "criterion": schaffer2,
        "start_x": np.array([3, -4]),
        "solution_x": np.array([0, 0]),
        "start_criterion": 0.04076572112213522,
        "solution_criterion": 0,
    },
    "schaffer2_bad_start": {
        "criterion": schaffer2,
        "start_x": np.array([3, 7]),
        "solution_x": np.array([0, 0]),
        "start_criterion": 0.933992959675674,
        "solution_criterion": 0,
    },
    "schaffer3_good_start": {
        "criterion": schaffer3,
        "start_x": np.array([-1, -4]),
        "solution_x": np.array([0, 1.253115]),
        "start_criterion": 0.08795185526199178,
        "solution_criterion": 0.0015668545260126288,
    },
    "schaffer3_bad_start": {
        "criterion": schaffer3,
        "start_x": np.array([-1, -7]),
        "solution_x": np.array([0, 1.253115]),
        "start_criterion": 0.6593946641439471,
        "solution_criterion": 0.0015668545260126288,
    },
    "schaffer4_good_start": {
        "criterion": schaffer4,
        "start_x": np.array([1, -2.66]),
        "solution_x": np.array([0, 1.253115]),
        "start_criterion": 0.3173669258600901,
        "solution_criterion": 0.2925786328424814,
    },
    "schaffer4_bad_start": {
        "criterion": schaffer4,
        "start_x": np.array([0.25, -0.25]),
        "solution_x": np.array([0, 1.253115]),
        "start_criterion": 0.9844154691523228,
        "solution_criterion": 0.2925786328424814,
    },
    "schwefel_good_start": {
        "criterion": schwefel,
        "start_x": np.array([420, 420]),
        "solution_x": np.array([420.9687, 420.9687]),
        "start_criterion": 0.2367647802917645,
        "solution_criterion": 2.545567497236334e-05,
    },
    "schwefel_bad_start": {
        "criterion": schwefel,
        "start_x": np.array([1, 1]),
        "solution_x": np.array([420.9687, 420.9687]),
        "start_criterion": 836.2828580303842,
        "solution_criterion": 2.545567497236334e-05,
    },
    "schwefel2_20_good_start": {
        "criterion": schwefel2_20,
        "start_x": np.full(2, 0.25),
        "solution_x": np.zeros(10),
        "start_criterion": 0.5,
        "solution_criterion": 0,
    },
    "schwefel2_20_bad_start": {
        "criterion": schwefel2_20,
        "start_x": np.full(10, -75),
        "solution_x": np.zeros(10),
        "start_criterion": 750,
        "solution_criterion": 0,
    },
    "schwefel2_21_good_start": {
        "criterion": schwefel2_21,
        "start_x": np.full(10, 0.001),
        "solution_x": np.zeros(2),
        "start_criterion": 0.001,
        "solution_criterion": 0,
    },
    "schwefel2_21_bad_start": {
        "criterion": schwefel2_21,
        "start_x": np.full(2, 100),
        "solution_x": np.zeros(2),
        "start_criterion": 100,
        "solution_criterion": 0,
    },
    "schwefel2_22_good_start": {
        "criterion": schwefel2_22,
        "start_x": np.full(2, 0.5),
        "solution_x": np.zeros(5),
        "start_criterion": 1.25,
        "solution_criterion": 0,
    },
    "schwefel2_22_bad_start": {
        "criterion": schwefel2_22,
        "start_x": np.full(3, 100),
        "solution_x": np.zeros(5),
        "start_criterion": 1000300,
        "solution_criterion": 0,
    },
    "schwefel2_23_good_start": {
        "criterion": schwefel2_23,
        "start_x": np.full(4, 0.5),
        "solution_x": np.zeros(5),
        "start_criterion": 0.00390625,
        "solution_criterion": 0,
    },
    "schwefel2_23_bad_start": {
        "criterion": schwefel2_23,
        "start_x": np.full(5, -7.5),
        "solution_x": np.zeros(5),
        "start_criterion": 2815675735.473633,
        "solution_criterion": 0,
    },
    "shekel_good_start": {
        "criterion": shekel,
        "start_x": np.array([1, 1, 1, 1]),
        "solution_x": np.array([4, 4, 4, 4]),
        "start_criterion": -5.128471039662403,
        "solution_criterion": -10.536283726219605,
    },
    "shekel_bad_start": {
        "criterion": shekel,
        "start_x": np.array([10, 50, 80, 40]),
        "solution_x": np.array([4, 4, 4, 4]),
        "start_criterion": -0.001113569345997643,
        "solution_criterion": -10.536283726219605,
    },
    "shubert_good_start": {
        "criterion": shubert,
        "start_x": np.array([-0.8, 5]),
        "solution_x": np.array([-1.4251, -7.0835]),
        "start_criterion": -140.07596851545017,
        "solution_criterion": -186.73090685668902,
    },
    "shubert_bad_start": {
        "criterion": shubert,
        "start_x": np.array([1, 10]),
        "solution_x": np.array([-1.4251, -7.0835]),
        "start_criterion": 5.9625568444156105,
        "solution_criterion": -186.73090685668902,
    },
    "shubert3_good_start": {
        "criterion": shubert3,
        "start_x": np.array([5, -5]),
        "solution_x": np.array([-7.4, -7.4]),
        "start_criterion": -11.310019174079681,
        "solution_criterion": -29.673336786222684,
    },
    "shubert3_bad_start": {
        "criterion": shubert3,
        "start_x": np.array([-10, 10]),
        "solution_x": np.array([-7.4, -7.4]),
        "start_criterion": 3.973720065167866,
        "solution_criterion": -29.673336786222684,
    },
    "shubert4_good_start": {
        "criterion": shubert4,
        "start_x": np.array([-5.5, -7.65]),
        "solution_x": np.array([4.85, 4.85]),
        "start_criterion": -15.962961434620379,
        "solution_criterion": -25.72096854993633,
    },
    "shubert4_bad_start": {
        "criterion": shubert4,
        "start_x": np.array([-7, 10]),
        "solution_x": np.array([4.85, 4.85]),
        "start_criterion": 10.001052922065966,
        "solution_criterion": -25.72096854993633,
    },
    "sphere_good_start": {
        "criterion": sphere,
        "start_x": np.full(2, 0.25),
        "solution_x": np.zeros(5),
        "start_criterion": 0.125,
        "solution_criterion": 0,
    },
    "sphere_bad_start": {
        "criterion": sphere,
        "start_x": np.full(2, 3),
        "solution_x": np.zeros(5),
        "start_criterion": 18,
        "solution_criterion": 0,
    },
    "styblinskitank_good_start": {
        "criterion": styblinskitank,
        "start_x": np.full(1, 3.5),
        "solution_x": np.full(1, -2.903534),
        "start_criterion": -14.21875,
        "solution_criterion": -39.1661657037714,
    },
    "styblinskitank_bad_start": {
        "criterion": styblinskitank,
        "start_x": np.full(1, 5),
        "solution_x": np.full(1, -2.903534),
        "start_criterion": 125.0,
        "solution_criterion": -39.1661657037714,
    },
    "sumsquares_good_start": {
        "criterion": sumsquares,
        "start_x": np.full(2, 0.5),
        "solution_x": np.zeros(10),
        "start_criterion": 0.75,
        "solution_criterion": 0.0,
    },
    "sumsquares_bad_start": {
        "criterion": sumsquares,
        "start_x": np.full(2, 10),
        "solution_x": np.zeros(10),
        "start_criterion": 300,
        "solution_criterion": 0.0,
    },
    "threehump_good_start": {
        "criterion": threehump,
        "start_x": np.full(2, 0.5),
        "solution_x": np.zeros(2),
        "start_criterion": 0.9369791666666667,
        "solution_criterion": 0,
    },
    "threehump_bad_start": {
        "criterion": threehump,
        "start_x": np.full(2, 10),
        "solution_x": np.zeros(2),
        "start_criterion": 156566.66666666666,
        "solution_criterion": 0,
    },
    "thevenot_good_start": {
        "criterion": thevenot,
        "start_x": np.full(200, 10),
        "solution_x": np.zeros(10),
        "start_criterion": 0.031169788108193906,
        "solution_criterion": -1,
    },
    "thevenot_bad_start": {
        "criterion": thevenot,
        "start_x": np.full(1, 10),
        "solution_x": np.zeros(10),
        "start_criterion": 0.9828079689867767,
        "solution_criterion": -1,
    },
    "trid_good_start": {
        "criterion": trid,
        "start_x": np.array([1, 1]),
        "solution_x": np.array([2, 2]),
        "start_criterion": -1,
        "solution_criterion": 0,
    },
    "trid_bad_start": {
        "criterion": trid,
        "start_x": np.array([10, 2]),
        "solution_x": np.array([2, 2]),
        "start_criterion": 80,
        "solution_criterion": 0,
    },
    "wolfe_good_start": {
        "criterion": wolfe,
        "start_x": np.full(3, 0.25),
        "solution_x": np.zeros(3),
        "start_criterion": 0.41666666666666663,
        "solution_criterion": 0,
    },
    "wolfe_bad_start": {
        "criterion": wolfe,
        "start_x": np.full(3, 10),
        "solution_x": np.zeros(3),
        "start_criterion": 52.16370213557839,
        "solution_criterion": 0,
    },
    "xinsheyang_solution": {
        "criterion": xinsheyang,
        "start_x": np.full(4, 10),
        "solution_x": np.zeros(10),
        "start_criterion": 5475.503552177131,
        "solution_criterion": 0,
    },
    "xinsheyang2_good_start": {
        "criterion": xinsheyang2,
        "start_x": np.full(1, 4),
        "solution_x": np.zeros(10),
        "start_criterion": 5.334513433011149,
        "solution_criterion": 0,
    },
    "xinsheyang2_bad_start": {
        "criterion": xinsheyang2,
        "start_x": np.full(10, 4),
        "solution_x": np.zeros(10),
        "start_criterion": 711.8823227848003,
        "solution_criterion": 0,
    },
    "xinsheyang3_good_start": {
        "criterion": xinsheyang3,
        "start_x": np.full(1, 1),
        "solution_x": np.zeros(10),
        "start_criterion": 0.7852124245010498,
        "solution_criterion": -1,
    },
    "xinsheyang3_bad_start": {
        "criterion": xinsheyang3,
        "start_x": np.full(40, 3),
        "solution_x": np.zeros(10),
        "start_criterion": 0.9999959040083886,
        "solution_criterion": -1,
    },
    "xinsheyang4_good_start": {
        "criterion": xinsheyang4,
        "start_x": np.zeros(1),
        "solution_x": np.zeros(2),
        "start_criterion": -1,
        "solution_criterion": -2,
    },
    "xinsheyang4_bad_start": {
        "criterion": xinsheyang4,
        "start_x": np.full(20, 4),
        "solution_x": np.zeros(2),
        "start_criterion": 7.538971657276083e-07,
        "solution_criterion": -2,
    },
    "zakharov_good_start": {
        "criterion": zakharov,
        "start_x": np.full(1, 3),
        "solution_x": np.zeros(10),
        "start_criterion": 16.3125,
        "solution_criterion": 0,
    },
    "zakharov_bad_start": {
        "criterion": zakharov,
        "start_x": np.full(10, 3),
        "solution_x": np.zeros(10),
        "start_criterion": 46331935.3125,
        "solution_criterion": 0,
    },
}

SCALAR_FUNCTIONS_TAGS = {
    "ackley": {
        "continuous": True,
        "convex": False,
        "separable": True,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": True,
    },
    "ackley2": {
        "continuous": False,
        "convex": True,
        "separable": False,
        "differentiable": True,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "ackley3": {
        "continuous": False,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "ackley4": {
        "continuous": False,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "adjiman": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "alpine1": {
        "continuous": False,
        "convex": False,
        "separable": True,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "alpine2": {
        "continuous": True,
        "convex": False,
        "separable": True,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "bartels": {
        "continuous": False,
        "convex": False,
        "separable": False,
        "differentiable": False,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "beale": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": True,
    },
    "bird": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": True,
    },
    "bohachevsky1": {
        "continuous": True,
        "convex": True,
        "separable": True,
        "differentiable": True,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "bohachevsky2": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "bohachevsky3": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "booth": {
        "continuous": True,
        "convex": True,
        "separable": False,
        "differentiable": True,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "branin": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "brent": {
        "continuous": True,
        "convex": True,
        "separable": False,
        "differentiable": True,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "brown": {
        "continuous": True,
        "convex": True,
        "separable": False,
        "differentiable": True,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "bukin6": {
        "continuous": True,
        "convex": True,
        "separable": False,
        "differentiable": False,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "colville": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "crossintray": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": False,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "dejong5": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": True,
    },
    "deckkersaarts": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "dixonprice": {
        "continuous": True,
        "convex": True,
        "separable": False,
        "differentiable": True,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "dropwave": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "easom": {
        "continuous": True,
        "convex": False,
        "separable": True,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "eggcrate": {
        "continuous": True,
        "convex": False,
        "separable": True,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "eggholder": {
        "continuous": False,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "exponential": {
        "continuous": True,
        "convex": True,
        "separable": True,
        "differentiable": True,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "forrester": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "goldsteinprice": {
        "continuous": False,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "gramacylee": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "griewank": {
        "continuous": True,
        "convex": False,
        "separable": True,
        "differentiable": True,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "happycat": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": True,
    },
    "himmelblau": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "holdertable": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": False,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "keane": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "langermann": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": True,
    },
    "leon": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "levy13": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "matyas": {
        "continuous": True,
        "convex": True,
        "separable": False,
        "differentiable": True,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "mccormick": {
        "continuous": True,
        "convex": True,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "michalewicz": {
        "continuous": True,
        "convex": False,
        "separable": True,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": True,
    },
    "periodic": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "permzerodbeta": {
        "continuous": True,
        "convex": True,
        "separable": False,
        "differentiable": True,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": True,
    },
    "permdbeta": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": True,
    },
    "powell": {
        "continuous": True,
        "convex": True,
        "separable": True,
        "differentiable": False,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "qing": {
        "continuous": True,
        "convex": False,
        "separable": True,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "quartic": {
        "continuous": True,
        "convex": False,
        "separable": True,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": True,
        "parametric": False,
    },
    "rastrigin": {
        "continuous": True,
        "convex": False,
        "separable": True,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "ridge": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": True,
    },
    "rosenbrock": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": True,
    },
    "rotatedhyperellipsoid": {
        "continuous": True,
        "convex": True,
        "separable": False,
        "differentiable": True,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "salomon": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "schaffer1": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "schaffer2": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "schaffer3": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "schaffer4": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "schwefel": {
        "continuous": True,
        "convex": False,
        "separable": True,
        "differentiable": False,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "schwefel2_20": {
        "continuous": True,
        "convex": True,
        "separable": True,
        "differentiable": False,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "schwefel2_21": {
        "continuous": True,
        "convex": True,
        "separable": True,
        "differentiable": False,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "schwefel2_22": {
        "continuous": True,
        "convex": True,
        "separable": True,
        "differentiable": False,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "schwefel2_23": {
        "continuous": True,
        "convex": True,
        "separable": True,
        "differentiable": False,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "shekel": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": True,
    },
    "shubert": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "shubert3": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "shubert4": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "sphere": {
        "continuous": True,
        "convex": True,
        "separable": True,
        "differentiable": False,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "styblinskitank": {
        "continuous": True,
        "convex": False,
        "separable": True,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "sumsquares": {
        "continuous": True,
        "convex": True,
        "separable": True,
        "differentiable": True,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "thevenot": {
        "continuous": True,
        "convex": True,
        "separable": True,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": True,
    },
    "threehumps": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "trid": {
        "continuous": True,
        "convex": True,
        "separable": False,
        "differentiable": True,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "wolfe": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "xinsheyang": {
        "continuous": False,
        "convex": False,
        "separable": True,
        "differentiable": False,
        "mutimodal": True,
        "randomized_term": True,
        "parametric": False,
    },
    "xinsheyang2": {
        "continuous": False,
        "convex": False,
        "separable": False,
        "differentiable": False,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "xinsheyang3": {
        "continuous": True,
        "convex": True,
        "separable": False,
        "differentiable": True,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": True,
    },
    "xinsheyang4": {
        "continuous": True,
        "convex": True,
        "separable": False,
        "differentiable": False,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "zakharov": {
        "continuous": False,
        "convex": False,
        "separable": False,
        "differentiable": False,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
}
