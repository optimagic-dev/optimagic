import pytest

from optimagic import timing


def test_invalid_aggregate_batch_time():
    with pytest.raises(ValueError, match="aggregate_batch_time must be a callable"):
        timing.CostModel(
            fun=None,
            jac=None,
            fun_and_jac=None,
            label="label",
            aggregate_batch_time="Not callable",
        )
