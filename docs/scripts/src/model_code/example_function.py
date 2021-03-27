import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("white")
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from scipy.optimize import minimize
import functools
import imageio
from matplotlib.widgets import TextBox


WEIGHTS = [
    9.003014962148157,
    -3.383000146393776,
    -0.6037887934635748,
    1.6984454347036886,
    -0.9447426232680957,
    0.2669069434366247,
    -0.04446368897497234,
    0.00460781796708519,
    -0.0003000790127508276,
    1.1934114174145725e-05,
    -2.6471293419570505e-07,
    2.5090819960943964e-09,
]


def example_criterion(x):
    x = _unpack_x(x)
    exponents = np.arange(len(WEIGHTS))
    return WEIGHTS @ x ** exponents


def example_gradient(x):
    x = _unpack_x(x)
    exponents = np.arange(len(WEIGHTS))
    return (WEIGHTS * exponents) @ x ** (exponents - 1).clip(0)


def example_hessian(x):
    x = _unpack_x(x)
    exponents = np.arange(len(WEIGHTS))
    return (WEIGHTS * exponents * (exponents - 1)) @ x ** (exponents - 2).clip(0)


def _unpack_x(x):
    if hasattr(x, "__len__"):
        assert len(x) == 1

    if isinstance(x, pd.DataFrame):
        res = x["value"].to_numpy()[0]
    elif isinstance(x, pd.Series):
        res = x.to_numpy()[0]
    elif isinstance(x, (np.ndarray, list, tuple)):
        res = x[0]
    else:
        res = float(x)
    return res
