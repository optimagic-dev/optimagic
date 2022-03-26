import numpy as np
from estimagic.optimization.tranquilo.aggregate_models import aggregator_identity
from estimagic.optimization.tranquilo.aggregate_models import aggregator_sum
from estimagic.optimization.tranquilo.models import ModelInfo
from estimagic.optimization.tranquilo.models import VectorModel
from numpy.testing import assert_array_equal


def test_aggregator_identity():
    model_info = ModelInfo()
    fvec_center = np.array([2.0])

    vector_model = VectorModel(
        intercepts=None,
        linear_terms=np.arange(3).reshape(1, 3),
        square_terms=np.arange(9).reshape(1, 3, 3),
    )

    got = tuple(aggregator_identity(vector_model, fvec_center, model_info))

    assert got[0] == 2.0
    assert_array_equal(got[1], np.arange(3))
    assert_array_equal(got[2], np.arange(9).reshape(3, 3))


def test_aggregator_sum():
    model_info = ModelInfo()
    fvec_center = np.array([1.0, 2.0])

    vector_model = VectorModel(
        intercepts=None,
        linear_terms=np.arange(6).reshape(2, 3),
        square_terms=np.arange(18).reshape(2, 3, 3),
    )

    got = tuple(aggregator_sum(vector_model, fvec_center, model_info))

    assert got[0] == 3.0
    assert_array_equal(got[1], np.array([3, 5, 7]))
    assert_array_equal(got[2], np.array([[9, 11, 13], [15, 17, 19], [21, 23, 25]]))
