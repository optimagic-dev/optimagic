import pytest
from estimagic.optimization.tranquilo.options import get_default_aggregator


def test_get_default_aggregator_scalar_quadratic():
    assert get_default_aggregator("scalar", "quadratic") == "identity"


def test_get_default_aggregator_error():
    with pytest.raises(
        ValueError,
        match="The requested combination of functype and model_type is not supported.",
    ):
        get_default_aggregator("scalar", "linear")
