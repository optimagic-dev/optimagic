import numpy as np
import pandas as pd
import pytest

from estimagic.inference.se_estimation import cluster_robust_se
from estimagic.inference.se_estimation import clustering
from estimagic.inference.se_estimation import inference_table
from estimagic.inference.se_estimation import observed_information_matrix
from estimagic.inference.se_estimation import outer_product_of_gradients
from estimagic.inference.se_estimation import robust_se
from estimagic.inference.se_estimation import sandwich_step
from estimagic.inference.se_estimation import strata_robust_se
from estimagic.inference.se_estimation import stratification
from estimagic.inference.se_estimation import variance_estimator


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
    out["robust_stderror"] = np.array(
        [[30.193984], [5.729162], [75.147385], [17.755383]]
    )

    out["robust_variance"] = np.array(
        [
            [911.67667, -172.809772, 2264.15098415, -534.7422541],
            [-172.809772, 32.823296, -429.142924, 101.253230],
            [2264.150984, -429.142924, 5647.129400, -1333.791658],
            [-534.742254, 101.253230, -1333.791658, 315.253633],
        ]
    )

    out["cluster_robust_se"] = np.array([[30.1901], [5.7280], [75.1199], [17.7546]])

    out["cluster_robust_var"] = np.array(
        [
            [911.411, -172.753, 2264.03, -534.648],
            [-172.753, 32.8104, -429.901, 101.228],
            [2263.03, -428.901, 5643, -1333.24],
            [-534.648, 101.228, -1333.24, 315.225],
        ]
    )

    out["strata_robust_se"] = np.array([[27.0028], [5.1233], [67.1893], [15.8802]])

    out["strata_robust_var"] = np.array(
        [
            [729.153, -138.203, 1810.42, -427.719],
            [-138.203, 26.2483, -343.121, 80.9828],
            [1810.42, -343.121, 4514.4, -1066.59],
            [-427.719, 80.9828, -1066.59, 252.18],
        ]
    )

    out["sandwich_se"] = np.array([[72.0984], [26.0388], [505.115], [3.88746]])

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

    out["oim_var"] = np.array(
        [
            [44.7392, -14.563, 41.659, 0.2407],
            [-14.56307, 9.01046, -14.14055, -6.3383],
            [41.65906, -14.14055, 487.09343, -9.645899],
            [0.240678, -6.338334, -9.645898, 11.859284],
        ]
    )

    out["oim_se"] = np.array([6.688734, 3.001743, 22.07019, 3.44373])

    out["opg_var"] = np.array(
        [
            [937.03508, -780.893, 781.1802, 741.8099],
            [-780.893, 749.9739, -749.918, -742.28097],
            [781.1802, -749.918045, 164316.58829, 741.88592],
            [741.8099, -742.280970, 741.8859, 742.520006],
        ]
    )

    out["opg_se"] = np.array([30.611028, 27.3856, 405.3598, 27.2492])

    out["cov_type_se"] = np.array([27.0028, 5.1233, 67.1893, 15.8801])

    out["cov_type_var"] = np.array(
        [
            [729.1525, -138.2026, 1810.4228, -427.7185],
            [-138.2026, 26.2483, -343.1207, 80.9827],
            [1810.4228, -343.1207, 4514.4020, -1066.5944],
            [-427.7185, 80.9827, -1066.5944, 252.1797],
        ]
    )

    out["params_df"] = pd.DataFrame(
        data=[
            [0.5, 30.1901, -58.672596, 59.672596],
            [0.5, 5.7280, -10.726880, 11.726880],
            [0.5, 75.1199, -146.735004, 147.735004],
            [0.5, 17.7546, -34.299016, 35.299016],
        ],
        columns=["value", "sandwich_standard_errors", "ci_lower", "ci_upper"],
    )

    out["cov_df"] = pd.DataFrame(
        data=np.array(
            [
                [911.411, -172.753, 2264.03, -534.648],
                [-172.753, 32.8104, -429.901, 101.228],
                [2263.03, -428.901, 5643, -1333.24],
                [-534.648, 101.228, -1333.24, 315.225],
            ]
        )
    )

    return out


def test_clustering(setup_robust, expected_robust):
    cluster_meat = clustering(setup_robust["design_options"], setup_robust["jac"])
    np.allclose(cluster_meat, expected_robust["cluster_meat"])


def test_stratification(setup_robust, expected_robust):
    strata_meat = stratification(setup_robust["design_options"], setup_robust["jac"])
    np.allclose(strata_meat, expected_robust["strata_meat"])


def test_sandwich_estimator(setup_robust, expected_robust):
    calc_sandwich_se, calc_sandwich_var = sandwich_step(
        setup_robust["hess"], setup_robust["meat"]
    )
    np.allclose(calc_sandwich_var, expected_robust["sandwich_var"])
    np.allclose(calc_sandwich_se, expected_robust["sandwich_se"])


def test_robust_se(setup_robust, expected_robust):
    calc_robust_se, calc_robust_var = robust_se(
        setup_robust["jac"], setup_robust["hess"]
    )
    np.allclose(calc_robust_se, expected_robust["robust_stderror"])
    np.allclose(calc_robust_var, expected_robust["robust_variance"])


def test_cluster_robust_se(setup_robust, expected_robust):
    calc_robust_cstd, calc_robust_cvar = cluster_robust_se(
        setup_robust["jac"], setup_robust["hess"], setup_robust["design_options"],
    )
    np.allclose(calc_robust_cvar, expected_robust["cluster_robust_var"])
    np.allclose(calc_robust_cstd, expected_robust["cluster_robust_se"])


def test_stratified_robust_se(setup_robust, expected_robust):
    calc_strata_se, calc_strata_var = strata_robust_se(
        setup_robust["jac"], setup_robust["hess"], setup_robust["design_options"],
    )
    np.allclose(calc_strata_var, expected_robust["strata_robust_var"])
    np.allclose(calc_strata_se, expected_robust["strata_robust_se"])


def test_observed_information_matrix(setup_robust, expected_robust):
    calc_oim_se, calc_oim_var = observed_information_matrix(setup_robust["hess"])
    np.allclose(calc_oim_var, expected_robust["oim_var"])
    np.allclose(calc_oim_se, expected_robust["oim_se"])


def test_outer_product_of_gradients(setup_robust, expected_robust):
    calc_opg_se, calc_opg_var = outer_product_of_gradients(setup_robust["jac"])
    np.allclose(calc_opg_var, expected_robust["opg_var"])
    np.allclose(calc_opg_se, expected_robust["opg_se"])


def test_variance_estimator(setup_robust, expected_robust):
    calc_cov_type_se, calc_cov_type_var = variance_estimator(
        setup_robust["jac"],
        setup_robust["hess"],
        setup_robust["design_options"],
        setup_robust["cov_type"],
    )
    np.allclose(calc_cov_type_var, expected_robust["cov_type_var"])
    np.allclose(calc_cov_type_se, expected_robust["cov_type_se"])


def test_inference_table(setup_robust, expected_robust):
    calc_params_df, calc_cov_df = inference_table(
        setup_robust["params"],
        setup_robust["se"],
        setup_robust["var"],
        setup_robust["cov_type"],
    )
    np.allclose(calc_params_df, expected_robust["params_df"])
    np.allclose(calc_cov_df, expected_robust["cov_df"])
