import numpy as np
import pandas as pd
import pytest

from estimagic.inference.src.functions.se_estimation import cluster_robust_se
from estimagic.inference.src.functions.se_estimation import clustering
from estimagic.inference.src.functions.se_estimation import design_options_preprocessing
from estimagic.inference.src.functions.se_estimation import robust_se
from estimagic.inference.src.functions.se_estimation import sandwich_step
from estimagic.inference.src.functions.se_estimation import strata_robust_se
from estimagic.inference.src.functions.se_estimation import stratification


@pytest.fixture
def setup_robust():
    out = {}
    out["jacobian"] = np.array(
        [
            [0.017986, 0.089931, 0, 0.035972],
            [0.0024726, 0.014836, 0.0024726, 0.0098905],
            [0.0009111, 0.002733, 0, 0.009111],
            [-0.993307, -4.966536, 0, -3.973229],
            [0.119203, 0.238406, 0, 0.119203],
        ]
    )

    out["hessian"] = np.array(
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

    out["weighted_jacobian"] = np.array(
        [
            [0.049258, 0.246290, 0, 0.098516],
            [0.078713, 0.472276, 0.078713, 0.314851],
            [0.175482, 0.526447, 0, 1.754820],
            [-0.388211, -1.941050, 0, -1.552840],
            [0.096610, 0.193221, 0, 0.096610],
        ]
    )

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
            [-172.80977222, 32.82329601, -429.14292424, 101.2532301],
            [2264.15098415, -429.14292424, 5647.12940078, -1333.79165873],
            [-534.7422541, 101.2532301, -1333.79165873, 315.25363306],
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

    out["sandwich_se"] = np.array([[72.0984], [26.0388], [505.115], [3.88746]])

    out["sandwich_var"] = np.array(
        [
            [5198.18, -1877.35, 36418, -280.28],
            [-1877.35, 678.017, -13152.6, 101.225],
            [36418, -13152.6, 255141, -1963.61],
            [-280.28, 101.225, -1963.61, 15.1123],
        ]
    )

    out["cluster_meat"] = np.array(
        [
            [0.249321, 1.14237, 0.0077446, 1.18717],
            [1.14237, 5.45734, 0.0464676, 5.162],
            [0.0077446, 0.0464676, 0.0077446, 0.0309784],
            [1.18717, 5.162, 0.0309784, 7.01111],
        ]
    )

    out["strata_meat"] = np.array(
        [
            [0.159566, 0.731114, 0.00495654, 0.759791],
            [0.731114, 3.4927, 0.0297393, 3.30368],
            [0.00495654, 0.0297393, 0.00495654, 0.0198262],
            [0.759791, 3.30368, 0.0198262, 4.48711],
        ]
    )

    return out


def test_clustering(setup_robust, expected_robust):
    cluster_meat = clustering(
        setup_robust["design_options"], setup_robust["weighted_jacobian"]
    )
    np.allclose(cluster_meat, expected_robust["cluster_meat"])


def test_stratification(setup_robust, expected_robust):
    strata_meat = stratification(
        setup_robust["design_options"], setup_robust["weighted_jacobian"]
    )
    np.allclose(strata_meat, expected_robust["strata_meat"])


def test_sandwich_estimator(setup_robust, expected_robust):
    calc_sandwich_se, calc_sandwich_var = sandwich_step(
        setup_robust["hessian"], setup_robust["meat"]
    )
    np.allclose(calc_sandwich_var, expected_robust["sandwich_var"])
    np.allclose(calc_sandwich_se, expected_robust["sandwich_se"])


def test_design_specification(setup_robust, expected_robust):
    calc_design_options = design_options_preprocessing(
        setup_robust["data"], setup_robust["design_dict"]
    )
    np.allclose(calc_design_options, expected_robust["design_options"])


def test_robust_se(setup_robust, expected_robust):
    calc_robust_se, calc_robust_var = robust_se(
        setup_robust["jacobian"], setup_robust["hessian"]
    )
    np.allclose(calc_robust_se, expected_robust["robust_stderror"])
    np.allclose(calc_robust_var, expected_robust["robust_variance"])


def test_cluster_robust_se(setup_robust, expected_robust):
    calc_robust_cstd, calc_robust_cvar = cluster_robust_se(
        setup_robust["jacobian"],
        setup_robust["hessian"],
        setup_robust["design_options"],
    )
    np.allclose(calc_robust_cvar, expected_robust["cluster_robust_var"])
    np.allclose(calc_robust_cstd, expected_robust["cluster_robust_se"])


def test_stratified_robust_se(setup_robust, expected_robust):
    calc_strata_se, calc_strata_var = strata_robust_se(
        setup_robust["jacobian"],
        setup_robust["hessian"],
        setup_robust["design_options"],
    )
    np.allclose(calc_strata_var, expected_robust["strata_robust_var"])
    np.allclose(calc_strata_se, expected_robust["strata_robust_se"])
