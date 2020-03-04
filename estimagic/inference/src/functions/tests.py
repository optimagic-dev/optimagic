import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from src.functions.se_estimation import cluster_robust_se
from src.functions.se_estimation import clustering
from src.functions.se_estimation import design_options_preprocessing
from src.functions.se_estimation import robust_se
from src.functions.se_estimation import sandwich_step
from src.functions.se_estimation import strata_robust_se
from src.functions.se_estimation import stratification


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

    out["design_options"] = pd.DataFrame(
        data=[[164, 88], [562, 24], [459, 71], [113, 25], [311, 63]],
        columns=["psu", "strata"],
    )

    out["data"] = out["design_options"].copy()

    out["meat"] = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])

    out["design_options_all"] = pd.DataFrame(data=[[]], columns=["", ""])

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

    out["cluster_robust_se"] = np.array(
        [[30.19397978], [5.7291604], [75.14738283], [17.75538399]]
    )

    out["cluster_robust_var"] = np.array(
        [
            [911.67641484, -172.80970332, 2264.15062153, -534.74220412],
            [-172.80970332, 32.82327889, -429.14280245, 101.2532081],
            [2264.15062153, -429.14280245, 5647.12914679, -1333.79168745],
            [-534.74220412, 101.2532081, -1333.79168745, 315.25366074],
        ]
    )

    out["strata_robust_se"] = np.array([[27.0063], [5.1243], [67.2139], [15.8809]])

    out["strata_robust_var"] = np.array(
        [
            [729.341, -138.248, 1811.32, -427.794],
            [-138.248, 26.2586, -343.314, 81.0026],
            [1811.32, -343.314, 4517.7, -1067.03],
            [-427.794, 81.0026, -1067.03, 252.203],
        ]
    )

    out["design_options"] = pd.DataFrame(
        data=[[164, 88], [562, 24], [459, 71], [113, 25], [311, 63]],
        columns=["psu", "strata"],
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
        setup_robust["design_options_all"], setup_robust["weighted_jacobian"]
    )
    assert_almost_equal(cluster_meat, expected_robust["cluster_meat"], decimal=6)


def test_stratification(setup_robust, expected_robust):
    strata_meat = stratification(
        setup_robust["design_options_all"], setup_robust["weighted_jacobian"]
    )
    assert_almost_equal(strata_meat, expected_robust["strata_meat"], decimal=6)


def test_sandwich_estimator(setup_robust, expected_robust):
    calc_sandwich_se, calc_sandwich_var = sandwich_step(
        setup_robust["hessian"], setup_robust["meat"]
    )
    assert_almost_equal(calc_sandwich_var, expected_robust["sandwich_var"], decimal=6)
    assert_almost_equal(calc_sandwich_se, expected_robust["sandwich_se"], decimal=6)


def test_design_specification(setup_robust, expected_robust):
    calc_design_options = design_options_preprocessing(
        setup_robust["data"], psu="psu", strata="strata", weight=None, fpc=None
    )
    assert_almost_equal(
        calc_design_options, expected_robust["design_options"], decimal=5
    )


def test_robust_se(setup_robust, expected_robust):
    calc_robust_se, calc_robust_var = robust_se(
        setup_robust["jacobian"], setup_robust["hessian"]
    )
    assert_almost_equal(calc_robust_se, expected_robust["robust_stderror"], decimal=5)
    assert_almost_equal(calc_robust_var, expected_robust["robust_variance"], decimal=3)


def test_cluster_robust_se(setup_robust, expected_robust):
    calc_robust_cstd, calc_robust_cvar = cluster_robust_se(**setup_robust)
    assert_almost_equal(
        calc_robust_cvar, expected_robust["cluster_robust_var"], decimal=6
    )
    assert_almost_equal(
        calc_robust_cstd, expected_robust["cluster_robust_se"], decimal=6
    )


def test_stratified_robust_se(setup_robust, expected_robust):
    calc_strata_se, calc_strata_var = strata_robust_se(**setup_robust)
    assert_almost_equal(
        calc_strata_var, expected_robust["strata_robust_var"], decimal=6
    )
    assert_almost_equal(calc_strata_se, expected_robust["strata_robust_se"], decimal=6)
