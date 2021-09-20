import numpy as np
import pandas as pd
import pytest

from estimagic.inference.ml_covs import cov_hessian
from estimagic.inference.ml_covs import cov_jacobian
from estimagic.inference.se_estimation import _clustering
from estimagic.inference.se_estimation import _sandwich_step
from estimagic.inference.se_estimation import _stratification
from estimagic.inference.se_estimation import cov_cluster_robust
from estimagic.inference.se_estimation import cov_robust
from estimagic.inference.se_estimation import cov_strata_robust


@pytest.fixture
def setup_robust():
    out = {}
    out["jac"] = np.array(
        [
            [0.017986, 0.089931, 0, 0.035972],
            [0.0024726, 0.014836, 0.0024726, 0.0098905],
            [0.0009111, 0.002733, 0, 0.009111],
            [-0.993307, -4.966536, 0, -3.973229],
            [0.119203, 0.238406, 0, 0.119203],
        ]
    )

    out["hess"] = np.array(
        [
            [-0.132681, -0.349071, -0.002467, -0.185879],
            [-0.349071, -1.124730, -0.014799, -0.606078],
            [-0.002467, -0.014799, -0.002467, -0.009866],
            [-0.185879, -0.606078, -0.009866, -0.412500],
        ]
    )

    out["design_dict"] = {"psu": "psu", "strata": "strata", "weight": "weight"}

    out["design_options"] = pd.DataFrame(
        data=[
            [164, 88, 0.116953],
            [562, 24, 0.174999],
            [459, 71, 0.374608],
            [113, 25, 0.369494],
            [311, 63, 0.203738],
        ],
        columns=["psu", "strata", "weight"],
    )

    out["data"] = out["design_options"].copy()

    out["meat"] = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])

    out["cov_type"] = "sandwich"

    out["se"] = np.array([[30.1901], [5.7280], [75.1199], [17.7546]])

    out["var"] = np.array(
        [
            [911.411, -172.753, 2264.03, -534.648],
            [-172.753, 32.8104, -429.901, 101.228],
            [2263.03, -428.901, 5643, -1333.24],
            [-534.648, 101.228, -1333.24, 315.225],
        ]
    )
    out["params"] = pd.DataFrame(data=[0.5, 0.5, 0.5, 0.5], columns=["value"])

    out["like_kwargs"] = {
        "formulas": ["eco_friendly ~ ppltrst + male + income"],
        "data": pd.DataFrame(
            data=[
                [1, 5, 0, 2],
                [1, 6, 1, 4],
                [1, 3, 0, 10],
                [0, 5, 0, 4],
                [1, 2, 0, 1],
            ],
            columns=["eco_friendly", "ppltrst", "male", "income"],
        ),
        "model": "probit",
    }

    return out


@pytest.fixture
def expected_robust():
    out = {}

    out["robust_variance"] = np.array(
        [
            [911.67667, -172.809772, 2264.15098415, -534.7422541],
            [-172.809772, 32.823296, -429.142924, 101.253230],
            [2264.150984, -429.142924, 5647.129400, -1333.791658],
            [-534.742254, 101.253230, -1333.791658, 315.253633],
        ]
    )

    out["cluster_robust_var"] = np.array(
        [
            [911.411, -172.753, 2264.03, -534.648],
            [-172.753, 32.8104, -429.901, 101.228],
            [2263.03, -428.901, 5643, -1333.24],
            [-534.648, 101.228, -1333.24, 315.225],
        ]
    )

    out["strata_robust_var"] = np.array(
        [
            [729.153, -138.203, 1810.42, -427.719],
            [-138.203, 26.2483, -343.121, 80.9828],
            [1810.42, -343.121, 4514.4, -1066.59],
            [-427.719, 80.9828, -1066.59, 252.18],
        ]
    )

    out["sandwich_var"] = np.array(
        [
            [5194.925, -1876.241, 36395.846, -279.962],
            [-1876.2415, 677.638707, -13145.02087, 101.11338],
            [36395.8461, -13145.0208, 254990.7081, -1961.4250],
            [-279.962055, 101.113381, -1961.425002, 15.087562],
        ]
    )

    out["cluster_meat"] = np.array(
        [
            [1.251498, 6.204213, 0.000008, 4.951907],
            [6.204213, 30.914541, 0.000046, 24.706263],
            [0.000008, 0.000046, 0.000008, 0.000031],
            [4.951907, 24.706263, 0.000031, 19.752791],
        ]
    )

    out["strata_meat"] = np.array(
        [
            [1.0012, 4.963, 0.000006, 3.9615],
            [4.9634, 24.732, 0.000037, 19.765],
            [0.000006, 0.000037, 0.000006, 0.000024],
            [3.961525, 19.76501, 0.000024, 15.8022],
        ]
    )

    out["cov_hessian"] = np.array(
        [
            [44.7392, -14.563, 41.659, 0.2407],
            [-14.56307, 9.01046, -14.14055, -6.3383],
            [41.65906, -14.14055, 487.09343, -9.645899],
            [0.240678, -6.338334, -9.645898, 11.859284],
        ]
    )

    out["cov_jacobian"] = np.array(
        [
            [937.03508, -780.893, 781.1802, 741.8099],
            [-780.893, 749.9739, -749.918, -742.28097],
            [781.1802, -749.918045, 164316.58829, 741.88592],
            [741.8099, -742.280970, 741.8859, 742.520006],
        ]
    )

    return out


def test_clustering(setup_robust, expected_robust):
    cluster_meat = _clustering(setup_robust["design_options"], setup_robust["jac"])
    np.allclose(cluster_meat, expected_robust["cluster_meat"])


def test_stratification(setup_robust, expected_robust):
    strata_meat = _stratification(setup_robust["design_options"], setup_robust["jac"])
    np.allclose(strata_meat, expected_robust["strata_meat"])


def test_sandwich_estimator(setup_robust, expected_robust):
    calc_sandwich_var = _sandwich_step(setup_robust["hess"], setup_robust["meat"])
    np.allclose(calc_sandwich_var, expected_robust["sandwich_var"])


def test_robust_se(setup_robust, expected_robust):
    calc_robust_var = cov_robust(setup_robust["jac"], setup_robust["hess"])
    np.allclose(calc_robust_var, expected_robust["robust_variance"])


def test_cluster_robust_se(setup_robust, expected_robust):
    calc_robust_cvar = cov_cluster_robust(
        setup_robust["jac"],
        setup_robust["hess"],
        setup_robust["design_options"],
    )
    np.allclose(calc_robust_cvar, expected_robust["cluster_robust_var"])


def test_stratified_robust_se(setup_robust, expected_robust):
    calc_strata_var = cov_strata_robust(
        setup_robust["jac"],
        setup_robust["hess"],
        setup_robust["design_options"],
    )
    np.allclose(calc_strata_var, expected_robust["strata_robust_var"])


def test_cov_hessian(setup_robust, expected_robust):
    calculated = cov_hessian(setup_robust["hess"])
    np.allclose(calculated, expected_robust["cov_hessian"])


def test_cov_jacobian(setup_robust, expected_robust):
    calc_opg_var = cov_jacobian(setup_robust["jac"])
    np.allclose(calc_opg_var, expected_robust["cov_jacobian"])
