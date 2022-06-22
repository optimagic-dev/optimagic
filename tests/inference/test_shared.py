from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest
from estimagic.inference.shared import process_pandas_arguments
from estimagic.inference.shared import transform_covariance
from numpy.testing import assert_array_almost_equal as aaae


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


def _from_internal(x, return_type="flat"):
    return x


class FakeConverter(NamedTuple):
    has_transforming_constraints: bool = True
    params_from_internal: callable = _from_internal


class FakeInternalParams(NamedTuple):
    values: np.ndarray = np.arange(2)
    lower_bounds: np.ndarray = np.full(2, -np.inf)
    upper_bounds: np.ndarray = np.full(2, np.inf)
    free_mask: np.ndarray = np.array([True, True])


def test_transform_covariance_no_bounds():
    internal_cov = np.eye(2)

    converter = FakeConverter()
    internal_params = FakeInternalParams()

    np.random.seed(1234)

    got = transform_covariance(
        internal_params=internal_params,
        internal_cov=internal_cov,
        converter=converter,
        n_samples=100,
        bounds_handling="ignore",
    )

    np.random.seed(1234)
    expected_sample = np.random.multivariate_normal(np.arange(2), np.eye(2), 100)
    expected = np.cov(expected_sample, rowvar=False)

    aaae(got, expected)


def test_transform_covariance_with_clipping():
    internal_cov = np.eye(2)

    converter = FakeConverter()
    internal_params = FakeInternalParams(
        lower_bounds=np.ones(2), upper_bounds=np.ones(2)
    )

    np.random.seed(1234)

    got = transform_covariance(
        internal_params=internal_params,
        internal_cov=internal_cov,
        converter=converter,
        n_samples=100,
        bounds_handling="clip",
    )

    expected = np.zeros((2, 2))

    aaae(got, expected)


def test_transform_covariance_invalid_bounds():
    internal_cov = np.eye(2)

    converter = FakeConverter()
    internal_params = FakeInternalParams(
        lower_bounds=np.ones(2), upper_bounds=np.ones(2)
    )

    np.random.seed(1234)

    with pytest.raises(ValueError):
        transform_covariance(
            internal_params=internal_params,
            internal_cov=internal_cov,
            converter=converter,
            n_samples=10,
            bounds_handling="raise",
        )
