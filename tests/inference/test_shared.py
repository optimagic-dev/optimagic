import numpy as np
import pandas as pd
import pytest

from estimagic.inference.shared import process_pandas_arguments


@pytest.fixture
def inputs():
    jac = pd.DataFrame(np.ones((5, 3)), columns=["a", "b", "c"])
    hess = pd.DataFrame(np.eye(3) / 2, columns=list("abc"), index=list("abc"))
    weights = pd.DataFrame(np.eye(5))
    moments_cov = 1 / weights
    out = {"jac": jac, "hess": hess, "weights": weights, "moments_cov": moments_cov}
    return out


def test_process_pandas_arguments_all_pd(inputs):
    *arrays, names = process_pandas_arguments(**inputs)
    for arr in arrays:
        assert isinstance(arr, np.ndarray)

    expected_names = {"moments": list(range(5)), "params": ["a", "b", "c"]}

    for key, value in expected_names.items():
        assert names[key].tolist() == value


def test_process_pandas_arguments_incompatible_names(inputs):
    inputs["jac"].columns = ["c", "d", "e"]

    with pytest.raises(ValueError):
        process_pandas_arguments(**inputs)
