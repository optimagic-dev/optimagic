import numpy as np
import pandas as pd

from optimagic.optimizers.gradient_free_optimizers import _value2para
from optimagic.parameters.bounds import Bounds

params = {"a": 5, "b": 6, "c": pd.Series([12, 13, 14])}
bounds = Bounds(
    lower={"a": 0, "b": 1, "c": pd.Series([2, 3, 4])},
    upper={"a": 10, "b": 11, "c": pd.Series([21, 31, 41])},
)


def test_value2para():
    assert _value2para(np.array([0, 1, 2])) == {"x0": 0, "x1": 1, "x2": 2}
