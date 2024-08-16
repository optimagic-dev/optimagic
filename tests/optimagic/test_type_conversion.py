import numpy as np
import pytest
from optimagic.type_conversion import TYPE_CONVERTERS
from optimagic.typing import (
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)


@pytest.mark.parametrize("candidate", [1, "1", 1.0, "1.0", np.int32(1), np.array(1.0)])
def test_int_conversion(candidate):
    got = TYPE_CONVERTERS[int](candidate)
    assert isinstance(got, int)
    assert got == 1


@pytest.mark.parametrize("candidate", [1, "1", 1.0, "1.0", np.int32(1), np.array(1.0)])
def test_positive_int_conversion(candidate):
    got = TYPE_CONVERTERS[PositiveInt](candidate)
    assert isinstance(got, int)
    assert got == 1


@pytest.mark.parametrize("candidate", [1, "1", 1.0, "1.0", np.int32(1), np.array(1.0)])
def test_non_negative_int_conversion(candidate):
    got = TYPE_CONVERTERS[NonNegativeInt](candidate)
    assert isinstance(got, int)
    assert got == 1


@pytest.mark.parametrize("candidate", [-1, "-1", -1.0, 0])
def test_positive_int_conversion_fail(candidate):
    with pytest.raises(Exception):  # noqa: B017
        TYPE_CONVERTERS[PositiveInt](candidate)


@pytest.mark.parametrize("candidate", [-1, "-1", -1.0])
def test_non_negative_int_conversion_fail(candidate):
    with pytest.raises(Exception):  # noqa: B017
        TYPE_CONVERTERS[NonNegativeInt](candidate)


@pytest.mark.parametrize("candidate", [1, "1", 1.0, "1.0", np.int32(1), np.array(1.0)])
def test_float_conversion(candidate):
    got = TYPE_CONVERTERS[float](candidate)
    assert isinstance(got, float)
    assert got == 1.0


@pytest.mark.parametrize("candidate", [1, "1", 1.0, "1.0", np.int32(1), np.array(1.0)])
def test_positive_float_conversion(candidate):
    got = TYPE_CONVERTERS[PositiveFloat](candidate)
    assert isinstance(got, float)
    assert got == 1.0


@pytest.mark.parametrize("candidate", [1, "1", 1.0, "1.0", np.int32(1), np.array(1.0)])
def test_non_negative_float_conversion(candidate):
    got = TYPE_CONVERTERS[NonNegativeFloat](candidate)
    assert isinstance(got, float)
    assert got == 1.0


@pytest.mark.parametrize("candidate", [-1, "-1", -1.0, 0])
def test_positive_float_conversion_fail(candidate):
    with pytest.raises(Exception):  # noqa: B017
        TYPE_CONVERTERS[PositiveFloat](candidate)


@pytest.mark.parametrize("candidate", [-1, "-1", -1.0])
def test_non_negative_float_conversion_fail(candidate):
    with pytest.raises(Exception):  # noqa: B017
        TYPE_CONVERTERS[NonNegativeFloat](candidate)
