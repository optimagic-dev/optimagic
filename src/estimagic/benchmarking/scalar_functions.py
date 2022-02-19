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
    res = -20 * np.exp(-0.2 * np.sqrt(np.mean(x ** 2)))
    out = res - np.exp(np.mean(np.cos(2 * np.pi * x))) + 20 + np.exp(1)
    return out


def ackley2(x):
    x_1, x_2 = x
    out = -200 * np.exp(-0.2 * np.sqrt(x_1 ** 2 + x_2 ** 2))
    return out


def ackley3(x):
    x_1, x_2 = x
    res = -200 * np.exp(-0.2 * np.sqrt(x_1 ** 2 + x_2 ** 2))
    out = res + 5 * np.exp(np.cos(3 * x_1) + np.sin(3 * x_2))
    return out


def ackley4(x):
    x_1, x_2 = x
    out = np.sum(
        np.exp(-0.2) * np.sqrt(x_1 ** 2 + x_2 ** 2)
        + 3 * (np.cos(2 * x_1) + np.sin(2 * x_2))
    )
    return out


def adjiman(x):
    x_1, x_2 = x
    out = np.cos(x_1) * np.sin(x_2) - x_1 / (x_2 ** 2 + 1)
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
        np.abs(x_1 ** 2 + x_2 ** 2 + x_1 * x_2)
        + np.abs(np.sin(x_1))
        + np.abs(np.cos(x_2))
    )
    return out


def beale(x):
    x_1, x_2 = x
    out = (
        (1.5 - x_1 + x_1 * x_2) ** 2
        + (2.25 - x_1 + x_1 * x_2 ** 2) ** 2
        + (2.625 - x_1 + x_1 * x_2 ** 3) * 2
    )
    return out


def bird(x):
    x_1, x_2 = x
    res = np.sin(x_1) * np.exp((1 - np.cos(x_2)) ** 2)
    out = res + np.cos(x_2) * np.exp((1 - np.sin(x_1)) ** 2) + (x_1 - x_2) ** 2
    return out


def bohachevsky1(x):
    x_1, x_2 = x
    out = (
        x_1 ** 2
        + 2 * x_2 ** 2
        - 0.3 * np.cos(3 * np.pi * x_1)
        - 0.4 * np.cos(4 * np.pi * x_2)
        + 0.7
    )
    return out


def bohachevsky2(x):
    x_1, x_2 = x
    out = (
        x_1 ** 2
        + 2 * x_2 ** 2
        - 0.3 * np.cos(3 * np.pi * x_1) * np.cos(4 * np.pi * x_2)
        + 0.3
    )
    return out


def bohachevsky3(x):
    x_1, x_2 = x
    out = (
        x_1 ** 2
        + 2 * x_2 ** 2
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
    res = (x_2 - 5.1 / (4 * np.pi ** 2) * x_1 ** 2 + 5 / np.pi * x_1 - 6) ** 2
    out = res + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x_1) + 10
    return out


def brent(x):
    x_1, x_2 = x
    out = (x_1 + 10) ** 2 + (x_2 + 10) ** 2 + np.exp(-(x_1 ** 2) - x_2 ** 2)
    return out


def brown(x):
    x_1, x_2 = x
    out = np.sum((x_1 ** 2) ** (x_2 ** 2 + 1) + (x_2 ** 2) ** (x_1 ** 2 + 1))
    return out


def bukin6(x):
    x_1, x_2 = x
    out = 100 * np.sqrt(np.abs(x_2 - 0.01 * x_1 ** 2)) + 0.01 * np.abs(x_1 + 10)
    return out


def colville(x):
    x_1, x_2, x_3, x_4 = x
    res = 100 * (x_1 ** 2 - x_2) ** 2 + (x_1 - 1) ** 2 + (x_3 - 1) ** 2
    out = (
        res
        + 90 * (x_3 ** 2 - x_4) ** 2
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
            * np.exp(np.abs(100 - np.sqrt(x_1 ** 2 + x_2 ** 2) / np.pi))
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
        1e5 * x_1 ** 2
        + x_2 ** 2
        - (x_1 ** 2 + x_2 ** 2)
        + 1e-5 * (x_1 ** 2 + x_2 ** 2) ** 4
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
    out = -(1 + np.cos(12 * np.sqrt(x_1 ** 2 + x_2 ** 2))) / (
        0.5 * (x_1 ** 2 + x_2 ** 2) + 2
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
    out = x_1 ** 2 + x_2 ** 2 + 25 * (np.sin(x_1) ** 2 + np.sin(x_2) ** 2)
    return out


def eggholder(x):
    x_1, x_2 = x
    out = -(x_2 + 47) * np.sin(np.sqrt(np.abs(x_2 + x_1 / 2 + 47))) - x_1 * np.sin(
        np.sqrt(np.abs(x_1 - x_2 - 47))
    )
    return out


def exponential(x):
    out = -np.exp(-0.5 * np.sum(x ** 2))
    return out


def forrester(x):
    out = ((6 * x - 2) ** 2) * np.sin(12 * x - 4)
    return out


def goldsteinprice(x):
    x_1, x_2 = x
    res = 1 + (x_1 + x_2 + 1) ** 2 * (
        19 - 14 * x_1 + 3 * x_1 ** 2 - 14 * x_2 + 6 * x_1 * x_2 + 3 * x_2 ** 2
    )
    out = res * (
        30
        + (2 * x_1 - 3 * x_2) ** 2
        * (18 - 32 * x_1 + 12 * x_1 ** 2 + 48 * x_2 - 36 * x_1 * x_2 + 27 * x_2 ** 2)
    )
    return out


def gramacylee(x):
    out = np.sin(10 * np.pi * x) / 2 / x + (x - 1) ** 4
    return out


def griewank(x):
    d = x.shape[0]
    i = np.arange(1, d + 1)
    out = 1 + np.sum(x ** 2 / 4000) - np.prod(np.cos(x / np.sqrt(i)))
    return out


def happycat(x):
    d = x.shape[0]
    norm = np.sum(x ** 2)
    out = ((norm - d) ** 2) ** 0.5 + (1 / d) * (0.5 * norm + np.sum(x)) + 0.5
    return out


def himmelblau(x):
    x_1, x_2 = x
    out = (x_1 ** 2 + x_2 - 11) ** 2 + (x_1 + x_2 ** 2 - 7) ** 2
    return out


def holdertable(x):
    x_1, x_2 = x
    out = -np.abs(
        np.sin(x_1)
        * np.cos(x_2)
        * np.exp(np.abs(1 - np.sqrt(x_1 ** 2 + x_2 ** 2) / np.pi))
    )
    return out


def keane(x):
    x_1, x_2 = x
    out = -(np.sin(x_1 - x_2) ** 2 * np.sin(x_1 + x_2) ** 2) / np.sqrt(
        x_1 ** 2 + x_2 ** 2
    )
    return out


def langermann(x):
    c = np.array([1, 2, 5, 2, 3])
    a = np.array([[3, 5], [5, 2], [2, 1], [1, 4], [7, 9]])
    out = np.sum(
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
    out = 100 * (x_2 - x_1 ** 3) ** 2 + (1 - x_1) ** 2
    return out


def levy13(x):
    x_1, x_2 = x
    out = (
        np.sin(3 * np.pi * x_1) ** 2
        + (x_1 - 1) ** 2 * (1 + np.sin(3 * np.pi * x_2) ** 2)
        + (x_2 - 1) ** 2 * (1 + np.sin(2 * np.pi * x_2) ** 2)
    )
    return out


def rosenbrock(x):
    out = mw.rosenbrock(x) @ mw.rosenbrock(x)
    return out


def rotatedhyperellipsoid(x):
    d = x.shape[0]
    out = np.sum([np.sum(x[: i + 1] ** 2) for i in range(d)])
    return out


def salomon(x):
    res = 1 - np.cos(2 * np.pi * np.sqrt(np.sum(x ** 2)))
    out = res + 0.1 * np.sqrt(np.sum(x ** 2))
    return out


def schaffel1(x):
    x_1, x_2 = x
    out = (
        0.5
        + (np.sin((x_1 ** 2 + x_2 ** 2) ** 2) ** 2 - 0.5)
        / (1 + 0.001 * (x_1 ** 2 + x_2 ** 2)) ** 2
    )
    return out


def schaffel2(x):
    x_1, x_2 = x
    out = (
        0.5
        + (np.sin((x_1 ** 2 + x_2 ** 2)) ** 2 - 0.5)
        / (1 + 0.001 * (x_1 ** 2 + x_2 ** 2)) ** 2
    )
    return out


def schaffel3(x):
    x_1, x_2 = x
    out = (
        0.5
        + (np.sin(np.cos(np.abs(x_1 ** 2 + x_2 ** 2))) ** 2 - 0.5)
        / (1 + 0.001 * (x_1 ** 2 + x_2 ** 2)) ** 2
    )
    return out


def schaffel4(x):
    x_1, x_2 = x
    out = (
        0.5
        + (np.cos(np.sin(np.abs(x_1 ** 2 + x_2 ** 2))) ** 2 - 0.5)
        / (1 + 0.001 * (x_1 ** 2 + x_2 ** 2)) ** 2
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
    out = np.sum(x ** 10)
    return out


def shubert(x):
    d = x.shape[0]
    for i in range(0, d):
        out = np.prod(np.sum([i * np.cos((j + 1) * x[i] + j) for j in range(1, 5 + 1)]))
    return out


def shubert3(x):
    out = np.sum(np.sum([j * np.sin((j + 1) * x + j) for j in range(1, 5 + 1)]))
    return out


def shubert4(x):
    out = np.sum(np.sum([j * np.cos((j + 1) * x + j) for j in range(1, 5 + 1)]))
    return out


def sphere(x):
    out = np.sum(x ** 2)
    return out


def styblinskitank(x):
    out = 0.5 * np.sum(x ** 4 - 16 * x ** 2 + 5 * x)
    return out


def sumquares(x):
    d = x.shape[0]
    i = np.arange(1, d + 1)
    out = np.sum(i * x ** 2)
    return out


def threehump(x):
    x_1, x_2 = x
    out = 2 * x_1 ** 2 - 1.05 * x_1 ** 4 + x_1 ** 6 * (1 / 6) + x_1 * x_2 + x_2 ** 2
    return out


def trid(x):
    out = np.sum(x - 1) ** 2 - np.sum(x[1:] * x[:-1])
    return out


def wolfe(x):
    x_1, x_2, x_3 = x
    out = 4 / 3 * (x_1 ** 2 + x_2 ** 2 - x_1 * x_2) ** 0.75 + x_3
    return out


def xinsheyang(x):
    d = x.shape[0]
    i = np.arange(1, d + 1)
    rand = np.random.random(d)
    out = np.sum(rand * np.abs(x) ** i)
    return out


def xinsheyang2(x):
    out = np.sum(np.abs(x)) * np.exp(-np.sum(np.sin(x ** 2)))
    return out


def xinsheyang3(x, m=5, beta=15):
    res = np.exp(-np.sum((x / beta) ** (2 * m)))
    out = res - 2 * np.exp(-np.sum(x ** 2)) * np.prod(np.cos(x) ** 2)
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
    out = np.sum(x ** 2) + np.sum(0.5 * i * x) ** 2 + np.sum(0.5 * i * x) ** 4
    return out


SCALAR_FUNCTION_PROBLEMS = {
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
        # no unique solution
        "solution_x": None,
        "start_criterion": -82.57324651934985,
        "solution_criterion": -170.07756299785044,
    },
    "ackley3_bad_start": {
        "criterion": ackley3,
        "start_x": np.full(2, 25),
        # no unique solution
        "solution_x": None,
        "start_criterion": 8.358584120180984,
        "solution_criterion": -170.07756299785044,
    },
    "ackley4_good_start": {
        "criterion": ackley4,
        "start_x": np.full(2, 3),
        "solution_x": np.array([-1.51, -0.755]),
        "start_criterion": 4.5901006651507235,
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
    "rosenbrock_good_start": {
        "criterion": rosenbrock,
        "start_x": np.array([-1.2, 1]),
        "solution_x": np.ones(2),
        "start_criterion": 24.2,
        "solution_criterion": 0,
    },
    "xinsheyang_good_start": {
        "criterion": xinsheyang,
        "start_x": np.full(1, 1),
        "solution_x": np.zeros(10),
        "start_criterion": 0.7646279260733415,
        "solution_criterion": 0,
    },
    "xinsheyang_bad_start": {
        "criterion": xinsheyang,
        "start_x": np.full(20, 4),
        "solution_x": np.zeros(10),
        "start_criterion": 1255254854146.7185,
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

SCALAR_FUNCTION_TAGS = {
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
    "schaffel1": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "schaffel2": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "schaffel3": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "schaffel4": {
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
