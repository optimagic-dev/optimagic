import pandas as pd


def f(ser: pd.Series):
    return ser * 2


def g(ser: "pd.Series[float]"):
    return ser * 2


s = pd.Series([1.0, 2.0])
f(s)
g(s)
