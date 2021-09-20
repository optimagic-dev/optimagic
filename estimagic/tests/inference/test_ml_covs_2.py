import numpy as np
import pandas as pd
import pytest

from estimagic.inference.ml_covs import cov_hessian
from estimagic.inference.ml_covs import cov_jacobian
from estimagic.inference.ml_covs import cov_robust
from estimagic.inference.se_estimation import _clustering
from estimagic.inference.se_estimation import _sandwich_step
from estimagic.inference.se_estimation import _stratification
from estimagic.inference.se_estimation import cov_cluster_robust
from estimagic.inference.se_estimation import cov_strata_robust


@pytest.fixture
def jac():
    _jac = np.array(
        [
            [0.017986, 0.089931, 0, 0.035972],
            [0.0024726, 0.014836, 0.0024726, 0.0098905],
            [0.0009111, 0.002733, 0, 0.009111],
            [-0.993307, -4.966536, 0, -3.973229],
            [0.119203, 0.238406, 0, 0.119203],
        ]
    )
    return _jac


@pytest.fixture
def hess():
    _hess = np.array(
        [
            [-0.132681, -0.349071, -0.002467, -0.185879],
            [-0.349071, -1.124730, -0.014799, -0.606078],
            [-0.002467, -0.014799, -0.002467, -0.009866],
            [-0.185879, -0.606078, -0.009866, -0.412500],
        ]
    )
    return _hess


@pytest.fixture
def design_options():
    df = pd.DataFrame(
        data=[
            [164, 88, 0.116953],
            [562, 24, 0.174999],
            [459, 71, 0.374608],
            [113, 25, 0.369494],
            [311, 63, 0.203738],
        ],
        columns=["psu", "strata", "weight"],
    )
    return df


def test_clustering(jac, design_options):
    calculated = _clustering(design_options, jac)
    expected = np.array(
        [
            [1.251498, 6.204213, 0.000008, 4.951907],
            [6.204213, 30.914541, 0.000046, 24.706263],
            [0.000008, 0.000046, 0.000008, 0.000031],
            [4.951907, 24.706263, 0.000031, 19.752791],
        ]
    )
    np.allclose(calculated, expected)


def test_stratification(jac, design_options):
    calculated = _stratification(design_options, jac)

    expected = np.array(
        [
            [1.0012, 4.963, 0.000006, 3.9615],
            [4.9634, 24.732, 0.000037, 19.765],
            [0.000006, 0.000037, 0.000006, 0.000024],
            [3.961525, 19.76501, 0.000024, 15.8022],
        ]
    )
    np.allclose(calculated, expected)


def test_sandwich_step(hess):
    calculated = _sandwich_step(hess, meat=np.ones((4, 4)))

    expected = np.array(
        [
            [5194.925, -1876.241, 36395.846, -279.962],
            [-1876.2415, 677.638707, -13145.02087, 101.11338],
            [36395.8461, -13145.0208, 254990.7081, -1961.4250],
            [-279.962055, 101.113381, -1961.425002, 15.087562],
        ]
    )
    np.allclose(calculated, expected)


def test_cov_robust(jac, hess):
    calculated = cov_robust(jac, hess)

    expected = np.array(
        [
            [911.67667, -172.809772, 2264.15098415, -534.7422541],
            [-172.809772, 32.823296, -429.142924, 101.253230],
            [2264.150984, -429.142924, 5647.129400, -1333.791658],
            [-534.742254, 101.253230, -1333.791658, 315.253633],
        ]
    )
    np.allclose(calculated, expected)


def test_cov_cluster_robust(jac, hess, design_options):
    calculated = cov_cluster_robust(
        jac,
        hess,
        design_options,
    )

    expected = np.array(
        [
            [911.411, -172.753, 2264.03, -534.648],
            [-172.753, 32.8104, -429.901, 101.228],
            [2263.03, -428.901, 5643, -1333.24],
            [-534.648, 101.228, -1333.24, 315.225],
        ]
    )

    np.allclose(calculated, expected)


def test_cov_strata_robust(jac, hess, design_options):
    calculated = cov_strata_robust(
        jac,
        hess,
        design_options,
    )

    expected = np.array(
        [
            [729.153, -138.203, 1810.42, -427.719],
            [-138.203, 26.2483, -343.121, 80.9828],
            [1810.42, -343.121, 4514.4, -1066.59],
            [-427.719, 80.9828, -1066.59, 252.18],
        ]
    )
    np.allclose(calculated, expected)


def test_cov_hessian(hess):
    calculated = cov_hessian(hess)

    expected = np.array(
        [
            [44.7392, -14.563, 41.659, 0.2407],
            [-14.56307, 9.01046, -14.14055, -6.3383],
            [41.65906, -14.14055, 487.09343, -9.645899],
            [0.240678, -6.338334, -9.645898, 11.859284],
        ]
    )
    np.allclose(calculated, expected)


def test_cov_jacobian(jac):
    calculated = cov_jacobian(jac)
    expected = np.array(
        [
            [937.03508, -780.893, 781.1802, 741.8099],
            [-780.893, 749.9739, -749.918, -742.28097],
            [781.1802, -749.918045, 164316.58829, 741.88592],
            [741.8099, -742.280970, 741.8859, 742.520006],
        ]
    )
    np.allclose(calculated, expected)
