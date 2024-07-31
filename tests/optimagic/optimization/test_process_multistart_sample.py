import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from optimagic.optimization.process_multistart_sample import process_multistart_sample

samples = [
    (
        pd.DataFrame(np.ones((2, 3)), columns=["a", "b", "c"]),
        pd.Series([1, 2, 3], index=["a", "b", "c"], name="value").to_frame(),
        lambda x: x["value"].to_numpy(),
    ),
    (
        np.ones((2, 3)),
        np.array([1, 2, 3]),
        lambda x: x,
    ),
]


@pytest.mark.parametrize("sample, x, to_internal", samples)
def test_process_multistart_sample(sample, x, to_internal):
    calculated = process_multistart_sample(sample, x, to_internal)
    expeceted = np.ones((2, 3))
    aaae(calculated, expeceted)
