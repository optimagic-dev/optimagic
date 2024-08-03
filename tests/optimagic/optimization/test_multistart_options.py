from typing import get_args, get_type_hints

import pytest
from optimagic.exceptions import InvalidMultistartError
from optimagic.optimization.multistart import (
    MultistartOptions,
    MultistartOptionsDict,
    pre_process_multistart,
)


def test_multistart_options_and_dict_have_same_attributes():
    """Test that MultistartOptions and MultistartOptionsDict have same values and types.

    As there is no easy way to not read the NotRequired types in 3.10, we need to
    activate include_extras=True to get the NotRequired types from the dict in Python
    3.11 and above. Once we drop support for Python 3.10, we can remove the
    include_extras=True argument and the removal of the NotRequired types.

    """
    types_from_multistart_options = get_type_hints(MultistartOptions)
    types_from_multistart_options_dict = get_type_hints(
        MultistartOptionsDict, include_extras=True
    )
    types_from_multistart_options_dict = {
        # Remove typing.NotRequired from the types
        k: get_args(v)[0]
        for k, v in types_from_multistart_options_dict.items()
    }
    assert types_from_multistart_options == types_from_multistart_options_dict


def test_pre_process_multistart_trivial_case():
    multistart = MultistartOptions(n_samples=10, convergence_max_discoveries=55)
    got = pre_process_multistart(multistart)
    assert got == multistart


def test_pre_process_multistart_none_case():
    assert pre_process_multistart(None) is None


def test_pre_process_multistart_false_case():
    assert pre_process_multistart(False) is None


def test_pre_process_multistart_dict_case():
    got = pre_process_multistart(
        multistart={
            "n_samples": 10,
            "convergence_max_discoveries": 55,
        }
    )
    assert got == MultistartOptions(
        n_samples=10,
        convergence_max_discoveries=55,
    )


def test_pre_process_multistart_invalid_type():
    with pytest.raises(InvalidMultistartError, match="Invalid multistart options"):
        pre_process_multistart(multistart="invalid")


def test_pre_process_multistart_invalid_dict_key():
    with pytest.raises(InvalidMultistartError, match="Invalid multistart options"):
        pre_process_multistart(multistart={"invalid": "invalid"})


def test_pre_process_multistart_invalid_dict_value():
    with pytest.raises(InvalidMultistartError, match="Invalid number of samples"):
        pre_process_multistart(multistart={"n_samples": "invalid"})


@pytest.mark.parametrize("value", ["invalid", -1])
def test_multistart_options_invalid_n_samples_value(value):
    with pytest.raises(InvalidMultistartError, match="Invalid number of samples"):
        MultistartOptions(n_samples=value)


@pytest.mark.parametrize("value", ["invalid", -1])
def test_multistart_options_invalid_n_optimizations(value):
    with pytest.raises(InvalidMultistartError, match="Invalid number of optimizations"):
        MultistartOptions(n_optimizations=value)


def test_multistart_options_invalid_sampling_distribution():
    with pytest.raises(InvalidMultistartError, match="Invalid sampling distribution"):
        MultistartOptions(sampling_distribution="invalid")


def test_multistart_options_invalid_sampling_method():
    with pytest.raises(InvalidMultistartError, match="Invalid sampling method"):
        MultistartOptions(sampling_method="invalid")


def test_multistart_options_invalid_mixing_weight_method():
    with pytest.raises(InvalidMultistartError, match="Invalid mixing weight method"):
        MultistartOptions(mixing_weight_method="invalid")


@pytest.mark.parametrize("value", [("a", "b"), (1, 2, 3), {"a": 1.0, "b": 3.0}])
def test_multistart_options_invalid_mixing_weight_bounds(value):
    with pytest.raises(InvalidMultistartError, match="Invalid mixing weight bounds"):
        MultistartOptions(mixing_weight_bounds=value)


def test_multistart_options_invalid_convergence_relative_params_tolerance():
    with pytest.raises(InvalidMultistartError, match="Invalid relative params"):
        MultistartOptions(convergence_relative_params_tolerance="invalid")


@pytest.mark.parametrize("value", ["invalid", -1])
def test_multistart_options_invalid_convergence_max_discoveries(value):
    with pytest.raises(InvalidMultistartError, match="Invalid max discoveries"):
        MultistartOptions(convergence_max_discoveries=value)


@pytest.mark.parametrize("value", ["invalid", -1])
def test_multistart_options_invalid_n_cores(value):
    with pytest.raises(InvalidMultistartError, match="Invalid number of cores"):
        MultistartOptions(n_cores=value)


@pytest.mark.parametrize("value", ["invalid", -1])
def test_multistart_options_invalid_batch_size(value):
    with pytest.raises(InvalidMultistartError, match="Invalid batch size"):
        MultistartOptions(batch_size=value)


def test_multistart_options_batch_size_smaller_than_n_cores():
    with pytest.raises(InvalidMultistartError, match="Invalid batch size"):
        MultistartOptions(batch_size=1, n_cores=2)


def test_multistart_options_invalid_batch_evaluator():
    with pytest.raises(InvalidMultistartError, match="Invalid batch evaluator"):
        MultistartOptions(batch_evaluator="invalid")


def test_multistart_options_invalid_seed():
    with pytest.raises(InvalidMultistartError, match="Invalid seed"):
        MultistartOptions(seed="invalid")


def test_multistart_options_invalid_optimization_error_handling():
    with pytest.raises(InvalidMultistartError, match="Invalid optimization error"):
        MultistartOptions(optimization_error_handling="invalid")


def test_multistart_options_invalid_exploration_error_handling():
    with pytest.raises(InvalidMultistartError, match="Invalid exploration error"):
        MultistartOptions(exploration_error_handling="invalid")
