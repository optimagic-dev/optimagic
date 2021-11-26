import numpy as np
import pandas as pd
import pytest
from estimagic.config import TEST_FIXTURES_POUNDERS
from estimagic.optimization.pounders_auxiliary import find_nearby_points
from numpy.testing import assert_array_almost_equal as aaae


@pytest.mark.parametrize(
    "pickle_dict",
    [
        TEST_FIXTURES_POUNDERS / "find_nearby_points_qmat_zero_i.pkl",
        TEST_FIXTURES_POUNDERS / "find_nearby_points_qmat_zero_ii.pkl",
        TEST_FIXTURES_POUNDERS / "find_nearby_points_qmat_zero_iii.pkl",
        TEST_FIXTURES_POUNDERS / "find_nearby_points_qmat_zero_iv.pkl",
        TEST_FIXTURES_POUNDERS / "find_nearby_points_qmat_nonzero_i.pkl",
        TEST_FIXTURES_POUNDERS / "find_nearby_points_qmat_nonzero_ii.pkl",
        TEST_FIXTURES_POUNDERS / "find_nearby_points_qmat_nonzero_iii.pkl",
    ],
)
def test_find_nearby_points(pickle_dict):
    data = pd.read_pickle(pickle_dict)
    n = 3
    theta1 = 1e-5

    qmat_out, model_indices_out, mpoints_out, q_is_I_out = find_nearby_points(
        xhist=data["xhist"],
        xmin=data["xmin"],
        qmat=data["qmat"],
        q_is_I=data["q_is_I"],
        delta=data["delta"],
        c=data["c"],
        model_indices=data["model_indices"],
        mpoints=data["mpoints"],
        nhist=data["nhist"],
        theta1=theta1,
        n=n,
    )

    aaae(qmat_out, data["qmat_expected"])
    aaae(model_indices_out, data["model_indices_expected"])
    assert np.allclose(mpoints_out, data["mpoints_expected"])
    assert np.allclose(q_is_I_out, 0)
