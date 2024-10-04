import pytest

from optimagic.decorators import (
    catch,
    unpack,
)


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
