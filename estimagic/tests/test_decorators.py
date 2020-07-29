import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from estimagic.decorators import catch
from estimagic.decorators import numpy_interface
from estimagic.decorators import unpack


def test_numpy_interface():
    params = pd.DataFrame()
    params["value"] = np.arange(5).astype(float)
    params["lower"] = -1
    constraints = [{"loc": np.arange(3), "type": "fixed"}]

    x = np.array([10, 11])

    @numpy_interface(params=params, constraints=constraints, numpy_output=True)
    def f(params):
        return params

    calculated = f(x)

    excepected = np.array([[0, -1], [1, -1], [2, -1], [10, -1], [11.0, -1]])
    aaae(calculated, excepected)


def test_catch_at_defaults():
    @catch
    def f():
        raise ValueError

    with pytest.warns(UserWarning):
        assert f() is None

    @catch
    def g():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        g()


def test_catch_with_reraise():
    @catch(reraise=True)
    def f():
        raise ValueError

    with pytest.raises(ValueError):
        f()


def test_unpack_decorator_none():
    @unpack(symbol=None)
    def f(x):
        return x

    assert f(3) == 3


def test_unpack_decorator_one_star():
    @unpack(symbol="*")
    def f(x, y):
        return x + y

    assert f((3, 4)) == 7


def test_unpack_decorator_two_stars():
    @unpack(symbol="**")
    def f(x, y):
        return x + y

    assert f({"x": 3, "y": 4}) == 7
