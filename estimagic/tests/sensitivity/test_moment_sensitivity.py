import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from scipy import stats

from estimagic.config import TEST_DIR
from estimagic.sensitivity.moment_sensitivity import MEASURES
from estimagic.sensitivity.moment_sensitivity import moment_sensitivity

pd.set_option("precision", 6)

params_index = [["beta"], ["intersection", "x1", "x2"]]
params_index = pd.MultiIndex.from_product(params_index, names=["type", "name"])

data = pd.read_csv(TEST_DIR / "sensitivity" / "sensitivity_probit_example_data.csv")
y_data = data[["y"]]
x_data = data[["intercept", "x1", "x2"]]


def calc_moments_expectation(params, x, y, estimate_y=True):
    """This is the func1 in sensitivity module.

    Args:
        params (pd.DataFrame): see :ref:`params`
        x (pd.DataFrame)
        y (pd.DataFrame)
        estimate_y (boolean): use estimated y_star

    Return:
        moments (np.array): expectation of moments
    """

    mom_value = calc_moments_value(params, x, y, estimate_y)

    moments = mom_value.mean(axis=1)

    return moments


def calc_moments_value(params, x, y, estimate_y=True):
    """This is the func2 in sensitivity module.

    Args:
        params (pd.DataFrame): see :ref:`params`
        x (pd.DataFrame)
        y (pd.DataFrame)
        estimate_y (boolean): use estimated y_star

    Return:
        mom_value (pd.DataFrame): sample value of moments
    """

    if estimate_y is True:
        y_estimated = x.to_numpy() @ (params["value"].to_numpy())
    else:
        y_estimated = y.copy(deep=True)

    x_np = x.T.to_numpy()

    residual = y.T.to_numpy() - stats.norm.cdf(y_estimated)

    mom_value = []

    # loop through all x

    length = len(x_np)

    for i in range(length):
        for j in range(i, length):
            moment = residual * x_np[i] * x_np[j]
            mom_value.append(moment)

    mom_value = np.stack(mom_value, axis=1)[0]
    mom_value = pd.DataFrame(data=mom_value)

    return mom_value


@pytest.fixture
def sens_input():
    out = {}
    out["func1"] = calc_moments_expectation
    out["func2"] = calc_moments_value
    out["params"] = pd.DataFrame(
        data=[[0.57735], [0.57735], [0.57735]], index=params_index, columns=["value"]
    )
    out["func_kwargs"] = {"x": x_data, "y": y_data}

    return out


@pytest.fixture
def expected():
    out = {
        "sensitivity_to_bias": pd.DataFrame(
            data=[
                [4.010481, 2.068143, 2.753155, 0.495683, 1.854492, 0.641020],
                [0.605718, 6.468960, -2.235886, 1.324065, -1.916986, -0.116590],
                [2.218011, -1.517303, 7.547212, -0.972578, 1.956985, 0.255691],
            ],
            index=params_index,
        ),
        "fundamental_sensitivity_to_noise": pd.DataFrame(
            data=[
                [1.108992, 0.191341, 0.323757, 0.020377, 0.085376, 0.029528],
                [0.017262, 1.277374, 0.145700, 0.099208, 0.062248, 0.000667],
                [0.211444, 0.064198, 1.516571, 0.048900, 0.059264, 0.002929],
            ],
            index=params_index,
        ),
        "actual_sensitivity_to_noise": pd.DataFrame(
            data=[
                [1.108992, 0.191341, 0.323757, 0.020377, 0.085376, 0.029528],
                [0.017262, 1.277374, 0.145700, 0.099208, 0.062248, 0.000667],
                [0.211444, 0.064198, 1.516571, 0.048900, 0.059264, 0.002929],
            ],
            index=params_index,
        ),
        "actual_sensitivity_to_removal": pd.DataFrame(
            data=[
                [1.020791, 0.343558, 0.634299, 0.014418, 0.058827, 0.017187],
                [0.016262, 2.313441, 0.285552, 0.052574, 0.043585, 0.000306],
                [0.189769, 0.114946, 2.984443, 0.022729, 0.042140, 0.005072],
            ],
            index=params_index,
        ),
        "fundamental_sensitivity_to_removal": pd.DataFrame(
            data=[
                [0.992910, 0.340663, 0.634157, 0.009277, 0.058815, 0.013542],
                [0.015455, 2.274235, 0.285389, 0.045166, 0.042882, 0.000306],
                [0.189311, 0.114299, 2.970578, 0.022262, 0.040827, 0.001343],
            ],
            index=params_index,
        ),
        "sensitivity_to_weighting": pd.DataFrame(
            data=np.zeros((3, 6)),
            index=params_index,
        ),
    }

    return out


@pytest.mark.parametrize("measure", MEASURES)
def test_moments_value(sens_input, expected, measure):
    calculated = moment_sensitivity(
        func1=sens_input["func1"],
        func2=sens_input["func2"],
        params=sens_input["params"],
        func1_kwargs=sens_input["func_kwargs"],
        func2_kwargs=sens_input["func_kwargs"],
    )

    aaae(calculated[measure].to_numpy(), expected[measure].to_numpy())
