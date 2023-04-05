import pytest
from collections import namedtuple
from estimagic.optimization.tranquilo.options import (
    get_default_aggregator,
    get_default_stagnation_options,
    update_option_bundle,
)


def test_get_default_aggregator_scalar_quadratic():
    assert get_default_aggregator("scalar", "quadratic") == "identity"


def test_get_default_aggregator_error():
    with pytest.raises(
        NotImplementedError,
        match="The requested combination of functype and model_type is not supported.",
    ):
        get_default_aggregator("scalar", "linear")


@pytest.fixture
def default_options():
    options = namedtuple("default_options", "number")
    return options(number=1)


def test_update_option_bundle_fast_path():
    assert update_option_bundle("whatever", user_options=None) == "whatever"


def test_update_option_bundle_dict(default_options):
    got = update_option_bundle(default_options, user_options={"number": 2})
    assert got.number == 2


def test_update_option_bundle_namedtuple(default_options):
    user_option = default_options._replace(number=2)
    got = update_option_bundle(default_options, user_options=user_option)
    assert got.number == 2


def test_update_option_bundle_convert_type(default_options):
    got = update_option_bundle(default_options, user_options={"number": "2"})
    assert got.number == 2


def test_update_option_bundle_wrong_type(default_options):
    with pytest.raises(ValueError, match="invalid literal for int"):
        update_option_bundle(default_options, user_options={"number": "not_a_number"})


def test_update_option_bundle_invalid_field(default_options):
    with pytest.raises(
        ValueError, match="The following user options are not valid: {'not_a_field'}"
    ):
        update_option_bundle(default_options, user_options={"not_a_field": 10})


def test_get_default_stagnation_options():
    assert get_default_stagnation_options(10).sample_increment == 10
