from functools import partial

import numpy as np
import pandas as pd
import pytest
from estimagic.config import TEST_FIXTURES_DIR
from estimagic.optimization.pounders_auxiliary import add_more_points
from estimagic.optimization.pounders_auxiliary import calc_jac_and_hess_res
from estimagic.optimization.pounders_auxiliary import find_nearby_points
from estimagic.optimization.pounders_auxiliary import get_params_quadratic_model
from estimagic.optimization.pounders_auxiliary import get_residuals
from estimagic.optimization.pounders_auxiliary import improve_model
from estimagic.optimization.pounders_auxiliary import update_center
from estimagic.optimization.pounders_auxiliary import update_fdiff_and_hess
from numpy.testing import assert_array_almost_equal as aaae


@pytest.fixture
def criterion():
    data = pd.read_csv(TEST_FIXTURES_DIR / "pounders_example_data.csv")
    endog = np.asarray(data["y"])
    exog = np.asarray(data["t"])

    def func(x: np.ndarray, exog: np.ndarray, endog: np.ndarray) -> np.ndarray:
        """User provided residual function."""
        return endog - np.exp(-x[0] * exog) / (x[1] + x[2] * exog)

    return partial(func, exog=exog, endog=endog)


@pytest.fixture(
    params=[
        "qmat_zero_i",
        "qmat_zero_ii",
        "qmat_zero_iii",
        "qmat_zero_iv",
        "qmat_nonzero_i",
        "qmat_nonzero_ii",
        "qmat_nonzero_iii",
    ]
)
def dict_find_nearby_points(request):
    return pd.read_pickle(TEST_FIXTURES_DIR / f"find_nearby_points_{request.param}.pkl")


@pytest.fixture(params=["4", "7"])
def dict_get_residuals(request):
    return pd.read_pickle(TEST_FIXTURES_DIR / f"get_residuals_iter_{request.param}.pkl")


@pytest.fixture(params=["i", "ii"])
def dict_improve_model(request):
    return pd.read_pickle(TEST_FIXTURES_DIR / f"improve_model_{request.param}.pkl")


@pytest.fixture
def dict_update_center():
    return pd.read_pickle(TEST_FIXTURES_DIR / "update_center.pkl")


@pytest.fixture
def dict_add_more_points():
    return pd.read_pickle(TEST_FIXTURES_DIR / "add_more_points.pkl")


@pytest.fixture
def dict_get_params_quadratic():
    return pd.read_pickle(TEST_FIXTURES_DIR / "get_params_quadratic.pkl")


@pytest.fixture
def dict_update_fdiff_and_hess():
    return pd.read_pickle(TEST_FIXTURES_DIR / "update_fdiff_and_hess.pkl")


@pytest.fixture
def dict_calc_jac_and_hess_res():
    return pd.read_pickle(TEST_FIXTURES_DIR / "calc_jac_and_hess_res.pkl")


def test_update_center(dict_update_center):
    (xmin_out, fmin_out, fdiff_out, _, _, jac_res_out, minindex_out,) = update_center(
        xplus=dict_update_center["xplus"],
        xmin=dict_update_center["xmin"],
        xhist=dict_update_center["xhist"],
        delta=dict_update_center["delta"],
        fmin=dict_update_center["fmin"],
        fdiff=dict_update_center["fdiff"],
        fnorm=dict_update_center["fnorm"],
        fnorm_min=dict_update_center["fnorm_min"],
        hess=dict_update_center["hess"],
        jac_res=dict_update_center["jac_res"],
        hess_res=dict_update_center["hess_res"],
        nhist=dict_update_center["nhist"],
    )
    aaae(xmin_out, dict_update_center["xmin_expected"])
    aaae(fmin_out, dict_update_center["fmin_expected"])
    aaae(fdiff_out, dict_update_center["fdiff_expected"])
    aaae(jac_res_out, dict_update_center["jac_res_expected"], decimal=5)
    assert np.allclose(minindex_out, dict_update_center["minindex_expected"])


def test_find_nearby_points(dict_find_nearby_points):
    qmat_out, model_indices_out, mpoints_out, q_is_I_out = find_nearby_points(
        xhist=dict_find_nearby_points["xhist"],
        xmin=dict_find_nearby_points["xmin"],
        qmat=dict_find_nearby_points["qmat"],
        q_is_I=dict_find_nearby_points["q_is_I"],
        delta=dict_find_nearby_points["delta"],
        c=dict_find_nearby_points["c"],
        model_indices=dict_find_nearby_points["model_indices"],
        mpoints=dict_find_nearby_points["mpoints"],
        nhist=dict_find_nearby_points["nhist"],
        theta1=1e-5,
        n=3,
    )

    aaae(qmat_out, dict_find_nearby_points["qmat_expected"])
    aaae(model_indices_out, dict_find_nearby_points["model_indices_expected"])
    assert np.allclose(mpoints_out, dict_find_nearby_points["mpoints_expected"])
    assert np.allclose(q_is_I_out, 0)


def test_improve_model(dict_improve_model, criterion):
    (
        xhist_out,
        fhist_out,
        _,
        model_indices_out,
        mpoints_out,
        nhist_out,
    ) = improve_model(
        xhist=dict_improve_model["xhist"],
        fhist=dict_improve_model["fhist"],
        fnorm=dict_improve_model["fnorm"],
        jac_res=dict_improve_model["jac_res"],
        hess_res=dict_improve_model["hess_res"],
        qmat=dict_improve_model["qmat"],
        model_indices=dict_improve_model["model_indices"],
        minindex=dict_improve_model["minindex"],
        mpoints=dict_improve_model["mpoints"],
        nhist=dict_improve_model["nhist"],
        delta=dict_improve_model["delta"],
        lower_bounds=None,
        upper_bounds=None,
        addallpoints=1,
        n=3,
        criterion=criterion,
    )

    aaae(xhist_out, dict_improve_model["xhist_expected"])
    aaae(fhist_out, dict_improve_model["fhist_expected"])
    aaae(model_indices_out, dict_improve_model["model_indices_expected"])
    assert np.allclose(mpoints_out, dict_improve_model["mpoints_expected"])
    assert np.allclose(nhist_out, dict_improve_model["nhist_expected"])


def test_add_more_points(dict_add_more_points):
    n = 3
    maxinterp = 2 * n + 1

    L_out, Z_out, N_out, M_out, mpoints_out = add_more_points(
        xhist=dict_add_more_points["xhist"],
        xmin=dict_add_more_points["xmin"],
        model_indices=dict_add_more_points["model_indices"],
        minindex=dict_add_more_points["minindex"],
        delta=dict_add_more_points["delta"],
        mpoints=dict_add_more_points["mpoints"],
        nhist=dict_add_more_points["nhist"],
        c2=10,
        theta2=1e-4,
        n=n,
        maxinterp=maxinterp,
    )

    aaae(L_out, dict_add_more_points["L_expected"])
    aaae(Z_out, dict_add_more_points["Z_expected"])
    aaae(N_out, dict_add_more_points["N_expected"])
    aaae(M_out, dict_add_more_points["M_expected"])
    assert np.allclose(mpoints_out, dict_add_more_points["mpoints_expected"])


def test_get_residuals(dict_get_residuals):
    xhist = dict_get_residuals["xhist"]
    xmin = dict_get_residuals["xmin"]
    model_indices = dict_get_residuals["model_indices"]
    mpoints = dict_get_residuals["mpoints"]
    delta_old = dict_get_residuals["delta_old"]

    n = 3
    maxinterp = 2 * n + 1
    nobs = 214

    xk = (xhist[model_indices[:mpoints]] - xmin) / delta_old

    residuals = get_residuals(
        xk=xk,
        hess=dict_get_residuals["hess"],
        fhist=dict_get_residuals["fhist"],
        fmin=dict_get_residuals["fmin"],
        fdiff=dict_get_residuals["fdiff"],
        model_indices=model_indices,
        mpoints=mpoints,
        nobs=nobs,
        maxinterp=maxinterp,
    )

    aaae(xk, dict_get_residuals["xk"])
    aaae(residuals, dict_get_residuals["residuals_expected"])


def test_get_params_quadratic(dict_get_params_quadratic):
    jac_quadratic, hess_quadratic = get_params_quadratic_model(
        L=dict_get_params_quadratic["L"],
        Z=dict_get_params_quadratic["Z"],
        N=dict_get_params_quadratic["N"],
        M=dict_get_params_quadratic["M"],
        res=dict_get_params_quadratic["residuals"],
        mpoints=dict_get_params_quadratic["mpoints"],
        n=3,
        nobs=214,
    )

    aaae(jac_quadratic, dict_get_params_quadratic["jac_quadratic_expected"])
    aaae(hess_quadratic, dict_get_params_quadratic["hess_quadratic_expected"])


def test_update_fdiff_and_hess(dict_update_fdiff_and_hess):
    fdiff_out, hess_out = update_fdiff_and_hess(
        fdiff=dict_update_fdiff_and_hess["fdiff"],
        hess=dict_update_fdiff_and_hess["hess"],
        jac_quadratic=dict_update_fdiff_and_hess["jac_quadratic"],
        hess_quadratic=dict_update_fdiff_and_hess["hess_quadratic"],
        delta=dict_update_fdiff_and_hess["delta"],
        delta_old=dict_update_fdiff_and_hess["delta_old"],
    )

    aaae(fdiff_out, dict_update_fdiff_and_hess["fdiff_expected"])
    aaae(hess_out, dict_update_fdiff_and_hess["hess_expected"])


def test_calc_jac_and_hess_res(dict_calc_jac_and_hess_res):
    jac_res, hess_res = calc_jac_and_hess_res(
        fdiff=dict_calc_jac_and_hess_res["fdiff"],
        fmin=dict_calc_jac_and_hess_res["fmin"],
        hess=dict_calc_jac_and_hess_res["hess"],
    )

    aaae(jac_res, dict_calc_jac_and_hess_res["jac_res_expected"])
    aaae(hess_res, dict_calc_jac_and_hess_res["hess_res_expected"], decimal=3)
