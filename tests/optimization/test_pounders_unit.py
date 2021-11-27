from functools import partial

import numpy as np
import pandas as pd
import pytest
from estimagic.config import TEST_FIXTURES_DIR
from estimagic.optimization.pounders_auxiliary import find_nearby_points
from estimagic.optimization.pounders_auxiliary import improve_model
from numpy.testing import assert_array_almost_equal as aaae


@pytest.fixture
def criterion():
    data = pd.read_csv(TEST_FIXTURES_DIR / "example_data.csv")
    endog = np.asarray(data["y"])
    exog = np.asarray(data["t"])

    def func(x: np.ndarray, exog: np.ndarray, endog: np.ndarray) -> np.ndarray:
        """User provided residual function."""
        return endog - np.exp(-x[0] * exog) / (x[1] + x[2] * exog)

    return partial(func, exog=exog, endog=endog)


@pytest.mark.parametrize(
    "pickle_dict",
    [
        TEST_FIXTURES_DIR / "find_nearby_points_qmat_zero_i.pkl",
        TEST_FIXTURES_DIR / "find_nearby_points_qmat_zero_ii.pkl",
        TEST_FIXTURES_DIR / "find_nearby_points_qmat_zero_iii.pkl",
        TEST_FIXTURES_DIR / "find_nearby_points_qmat_zero_iv.pkl",
        TEST_FIXTURES_DIR / "find_nearby_points_qmat_nonzero_i.pkl",
        TEST_FIXTURES_DIR / "find_nearby_points_qmat_nonzero_ii.pkl",
        TEST_FIXTURES_DIR / "find_nearby_points_qmat_nonzero_iii.pkl",
    ],
)
def test_find_nearby_points(pickle_dict):
    dict_ = pd.read_pickle(pickle_dict)

    qmat_out, model_indices_out, mpoints_out, q_is_I_out = find_nearby_points(
        xhist=dict_["xhist"],
        xmin=dict_["xmin"],
        qmat=dict_["qmat"],
        q_is_I=dict_["q_is_I"],
        delta=dict_["delta"],
        c=dict_["c"],
        model_indices=dict_["model_indices"],
        mpoints=dict_["mpoints"],
        nhist=dict_["nhist"],
        theta1=1e-5,
        n=3,
    )

    aaae(qmat_out, dict_["qmat_expected"])
    aaae(model_indices_out, dict_["model_indices_expected"])
    assert np.allclose(mpoints_out, dict_["mpoints_expected"])
    assert np.allclose(q_is_I_out, 0)


@pytest.mark.parametrize(
    "pickle_dict",
    [
        TEST_FIXTURES_DIR / "improve_model_i.pkl",
        TEST_FIXTURES_DIR / "improve_model_ii.pkl",
    ],
)
def test_improve_model(pickle_dict, criterion):
    dict_ = pd.read_pickle(pickle_dict)

    (
        xhist_out,
        fhist_out,
        fnorm_out,
        model_indices_out,
        mpoints_out,
        nhist_out,
    ) = improve_model(
        xhist=dict_["xhist"],
        fhist=dict_["fhist"],
        fnorm=dict_["fnorm"],
        jac_res=dict_["jac_res"],
        hess_res=dict_["hess_res"],
        qmat=dict_["qmat"],
        model_indices=dict_["model_indices"],
        minindex=dict_["minindex"],
        mpoints=dict_["mpoints"],
        nhist=dict_["nhist"],
        delta=dict_["delta"],
        lower_bounds=None,
        upper_bounds=None,
        addallpoints=1,
        n=3,
        criterion=criterion,
    )

    aaae(xhist_out, dict_["xhist_expected"])
    aaae(fhist_out, dict_["fhist_expected"])
    aaae(model_indices_out, dict_["model_indices_expected"])
    assert np.allclose(mpoints_out, dict_["mpoints_expected"])
    assert np.allclose(nhist_out, dict_["nhist_expected"])
