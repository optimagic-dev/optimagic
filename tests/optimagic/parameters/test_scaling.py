from typing import get_args, get_type_hints

import pytest
from optimagic.exceptions import InvalidScalingError
from optimagic.parameters.scaling import (
    ScalingOptions,
    ScalingOptionsDict,
    pre_process_scaling,
)


def test_scaling_options_and_dict_have_same_attributes():
    """Test that ScalingOptions and ScalingOptionsDict have same values and types.

    As there is no easy way to not read the NotRequired types in 3.10, we need to
    activate include_extras=True to get the NotRequired types from the dict in Python
    3.11 and above. Once we drop support for Python 3.10, we can remove the
    include_extras=True argument and the removal of the NotRequired types.

    """
    types_from_scaling_options = get_type_hints(ScalingOptions)
    types_from_scaling_options_dict = get_type_hints(
        ScalingOptionsDict, include_extras=True
    )
    types_from_scaling_options_dict = {
        # Remove typing.NotRequired from the types
        k: get_args(v)[0]
        for k, v in types_from_scaling_options_dict.items()
    }
    assert types_from_scaling_options == types_from_scaling_options_dict


def test_pre_process_scaling_trivial_case():
    scaling = ScalingOptions(
        method="start_values",
        clipping_value=1,
        magnitude=2,
    )
    got = pre_process_scaling(scaling=scaling)
    assert got == scaling


def test_pre_process_scaling_none_case():
    assert pre_process_scaling(scaling=None) is None


def test_pre_process_scaling_false_case():
    assert pre_process_scaling(scaling=False) is None


def test_pre_process_scaling_true_case():
    got = pre_process_scaling(scaling=True)
    assert got == ScalingOptions()


def test_pre_process_scaling_dict_case():
    got = pre_process_scaling(
        scaling={"method": "start_values", "clipping_value": 1, "magnitude": 2}
    )
    assert got == ScalingOptions(method="start_values", clipping_value=1, magnitude=2)


def test_pre_process_scaling_invalid_type():
    with pytest.raises(InvalidScalingError, match="Invalid scaling options"):
        pre_process_scaling(scaling="invalid")


def test_pre_process_scaling_invalid_dict_key():
    with pytest.raises(InvalidScalingError, match="Invalid scaling options of type:"):
        pre_process_scaling(scaling={"wrong_key": "start_values"})


def test_pre_process_scaling_invalid_dict_value():
    with pytest.raises(InvalidScalingError, match="Invalid clipping value:"):
        pre_process_scaling(scaling={"clipping_value": "invalid"})


def test_scaling_options_invalid_method_value():
    with pytest.raises(InvalidScalingError, match="Invalid scaling method:"):
        ScalingOptions(method="invalid")


def test_scaling_options_invalid_clipping_value_type():
    with pytest.raises(InvalidScalingError, match="Invalid clipping value:"):
        ScalingOptions(clipping_value="invalid")


def test_scaling_options_invalid_magnitude_value_type():
    with pytest.raises(InvalidScalingError, match="Invalid scaling magnitude:"):
        ScalingOptions(magnitude="invalid")


def test_scaling_options_invalid_magnitude_value_range():
    with pytest.raises(InvalidScalingError, match="Invalid scaling magnitude:"):
        ScalingOptions(magnitude=-1)
