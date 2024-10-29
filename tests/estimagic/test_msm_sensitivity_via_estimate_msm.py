import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from scipy import stats

from estimagic.config import EXAMPLE_DIR
from estimagic.estimate_msm import estimate_msm


def simulate_aggregated_moments(params, x, y):
    """Calculate aggregated moments for example from Honore, DePaula, Jorgensen."""
    mom_value = simulate_moment_contributions(params, x, y)
    moments = mom_value.mean(axis=1)

    return moments


def simulate_moment_contributions(params, x, y):
    """Calculate moment contributions for example from Honore, DePaula, Jorgensen."""
    y_estimated = x.to_numpy() @ (params["value"].to_numpy())

    x_np = x.T.to_numpy()

    residual = y.T.to_numpy() - stats.norm.cdf(y_estimated)

    mom_value = []

    length = len(x_np)

    for i in range(length):
        for j in range(i, length):
            moment = residual * x_np[i] * x_np[j]
            mom_value.append(moment)

    mom_value = np.stack(mom_value, axis=1)[0]
    mom_value = pd.DataFrame(data=mom_value)

    return mom_value


@pytest.fixture()
def moments_cov(params, func_kwargs):
    mom_value = simulate_moment_contributions(params, **func_kwargs)
    mom_value = mom_value.to_numpy()
    s = np.cov(mom_value, ddof=0)
    return s


@pytest.fixture()
def params():
    params_index = [["beta"], ["intersection", "x1", "x2"]]
    params_index = pd.MultiIndex.from_product(params_index, names=["type", "name"])
    params = pd.DataFrame(
        data=[[0.57735], [0.57735], [0.57735]], index=params_index, columns=["value"]
    )
    return params


@pytest.fixture()
def func_kwargs():
    data = pd.read_csv(EXAMPLE_DIR / "sensitivity_probit_example_data.csv")
    y_data = data[["y"]]
    x_data = data[["intercept", "x1", "x2"]]
    func_kwargs = {"x": x_data, "y": y_data}
    return func_kwargs


@pytest.fixture()
def msm_res(params, moments_cov, func_kwargs):
    res = estimate_msm(
        simulate_moments=simulate_aggregated_moments,
        # only needed for shape since optimization is skipped
        empirical_moments=np.zeros(6),
        params=params,
        optimize_options=False,
        moments_cov=moments_cov,
        simulate_moments_kwargs=func_kwargs,
        weights="optimal",
    )
    return res


def test_sensitivity_to_bias(msm_res):
    calculated = msm_res.sensitivity(kind="bias")
    expected = np.array(
        [
            [4.010481, 2.068143, 2.753155, 0.495683, 1.854492, 0.641020],
            [0.605718, 6.468960, -2.235886, 1.324065, -1.916986, -0.116590],
            [2.218011, -1.517303, 7.547212, -0.972578, 1.956985, 0.255691],
        ]
    )

    aaae(calculated, expected)


def test_fundamental_sensitivity_to_noise(msm_res):
    calculated = msm_res.sensitivity(kind="noise_fundamental")
    expected = np.array(
        [
            [1.108992, 0.191341, 0.323757, 0.020377, 0.085376, 0.029528],
            [0.017262, 1.277374, 0.145700, 0.099208, 0.062248, 0.000667],
            [0.211444, 0.064198, 1.516571, 0.048900, 0.059264, 0.002929],
        ]
    )

    aaae(calculated, expected)


def test_actual_sensitivity_to_noise(msm_res):
    calculated = msm_res.sensitivity(kind="noise")
    expected = np.array(
        [
            [1.108992, 0.191341, 0.323757, 0.020377, 0.085376, 0.029528],
            [0.017262, 1.277374, 0.145700, 0.099208, 0.062248, 0.000667],
            [0.211444, 0.064198, 1.516571, 0.048900, 0.059264, 0.002929],
        ]
    )

    aaae(calculated, expected)


def test_actual_sensitivity_to_removal(msm_res):
    calculated = msm_res.sensitivity(kind="removal")

    expected = np.array(
        [
            [1.020791, 0.343558, 0.634299, 0.014418, 0.058827, 0.017187],
            [0.016262, 2.313441, 0.285552, 0.052574, 0.043585, 0.000306],
            [0.189769, 0.114946, 2.984443, 0.022729, 0.042140, 0.005072],
        ]
    )

    aaae(calculated, expected)


def test_fundamental_sensitivity_to_removal(msm_res):
    calculated = msm_res.sensitivity(kind="removal_fundamental")

    expected = np.array(
        [
            [0.992910, 0.340663, 0.634157, 0.009277, 0.058815, 0.013542],
            [0.015455, 2.274235, 0.285389, 0.045166, 0.042882, 0.000306],
            [0.189311, 0.114299, 2.970578, 0.022262, 0.040827, 0.001343],
        ]
    )

    aaae(calculated, expected)


def test_sensitivity_to_weighting(msm_res):
    calculated = msm_res.sensitivity(kind="weighting")

    expected = np.zeros((3, 6))

    aaae(calculated, expected)
