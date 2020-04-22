import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from estimagic.optimization.optimize import maximize
from estimagic.optimization.optimize import minimize


# =====================================================================================
# FIXTURES THAT PASS
# =====================================================================================

scipy_algos = ["scipy_L-BFGS-B", "scipy_TNC", "scipy_SLSQP"]


algorithms = [
    "pygmo_de1220",
    "pygmo_sade",
    "pygmo_pso",
    "pygmo_pso_gen",
    "pygmo_bee_colony",
    "pygmo_cmaes",
    "pygmo_xnes",
    "scipy_L-BFGS-B",
    "pygmo_ihs",
    "pygmo_de",
    "scipy_TNC",
    "scipy_SLSQP",
    "nlopt_cobyla",
    "nlopt_bobyqa",
    "nlopt_newuoa",
    "nlopt_newuoa_bound",
    "nlopt_praxis",
    "nlopt_neldermead",
    "nlopt_sbplx",
    "nlopt_lbfgs",
    "nlopt_tnewton",
    "nlopt_tnewton_precond_restart",
    "nlopt_tnewton_precond",
    "nlopt_tnewton_restart",
    "nlopt_ccsaq",
    "nlopt_var2",
    "nlopt_var1",
]


currently_failing = [
    # gradient based nlopt
    # runs forever
    "nlopt_mma",
    "nlopt_slsqp",
    # multi objective pygmo optimizers
    "pygmo_nsga2",
    "pygmo_moead",
    # precision problems
    "pygmo_sea",
    "pygmo_sga",
    "pygmo_simulated_annealing",
]


def f(params):
    x = params["value"].to_numpy()
    return -x @ x


@pytest.mark.parametrize("algorithm", algorithms)
def test_maximize(algorithm):
    print(algorithm)
    np.random.seed(1234)
    params = pd.Series([1, -1, -1.5, 1.5], name="value").to_frame()
    params["lower"] = -2
    params["upper"] = 2

    origin, algo_name = algorithm.split("_", 1)
    if origin == "pygmo":
        if algo_name == "simulated_annealing":
            algo_options = {}
        elif algo_name in ["ihs"]:
            algo_options = {"popsize": 1, "gen": 1000}
        elif algo_name in ["sga"]:
            algo_options = {"popsize": 50, "gen": 500}
        elif algo_name in ["sea"]:
            algo_options = {"popsize": 5, "gen": 7000}
        elif algo_name == "simulated_annealing":
            np.random.seed(5471)
            algo_options = {"n_T_adj": 20, "Tf": 0.0001, "n_range_adj": 20}
        else:
            algo_options = {"popsize": 30, "gen": 150}
    else:
        algo_options = {}
    res_dict, final_params = maximize(
        f, params, algorithm, algo_options=algo_options, logging=False,
    )
    aaae(final_params["value"].to_numpy(), np.zeros(len(final_params)), decimal=2)


# Test with gradient
def sum_of_squares(params):
    return (params["value"] ** 2).sum()


def sum_of_squares_gradient(params):
    return params["value"].to_numpy() * 2


some_gradient_algos = ["nlopt_lbfgs", "scipy_L-BFGS-B", "scipy_SLSQP"]


@pytest.mark.parametrize("algorithm", some_gradient_algos)
def test_minimize_with_gradient(algorithm):
    start_params = pd.DataFrame()
    start_params["value"] = [1, 2.5, -1]
    info, params = minimize(
        criterion=sum_of_squares,
        params=start_params,
        algorithm=algorithm,
        gradient=sum_of_squares_gradient,
    )

    aaae(info["x"], [0, 0, 0])


def minus_sum_of_squares(params):
    return -(params["value"] ** 2).sum()


def minus_sum_of_squares_gradient(params):
    return params["value"].to_numpy() * -2


some_gradient_algos = ["nlopt_lbfgs", "scipy_L-BFGS-B", "scipy_SLSQP"]


@pytest.mark.parametrize("algorithm", some_gradient_algos)
def test_maximize_with_gradient(algorithm):
    start_params = pd.DataFrame()
    start_params["value"] = [1, 2.5, -1]
    info, params = maximize(
        criterion=minus_sum_of_squares,
        params=start_params,
        algorithm=algorithm,
        gradient=minus_sum_of_squares_gradient,
    )

    aaae(info["x"], [0, 0, 0])
