"""Tests for the validated_dataclass decorator and the shared pydantic configs.

These tests pin down how optimagic uses pydantic (error translation, config
choices, preserved class attributes), not pydantic's own validation logic.

"""

import pickle
import subprocess
import sys
from dataclasses import FrozenInstanceError, dataclass, replace

import cloudpickle
import pydantic
import pytest

from optimagic.typing import (
    DEFAULT_PYDANTIC_CONFIG,
    STRICT_PYDANTIC_CONFIG,
    PositiveInt,
    _LazyValidator,
    validated_dataclass,
)


class CustomError(Exception):
    pass


def _make_error(e: pydantic.ValidationError) -> Exception:
    return CustomError(str(e))


@validated_dataclass(config=DEFAULT_PYDANTIC_CONFIG, make_error=_make_error)
@dataclass(frozen=True)
class Options:
    """Docstring of Options."""

    n_points: PositiveInt = 1
    share: float = 0.5


@validated_dataclass(config=STRICT_PYDANTIC_CONFIG, make_error=_make_error)
@dataclass(frozen=True)
class StrictOptions:
    n_points: int = 1
    label: str = "a"


def test_values_are_coerced_to_annotated_types():
    options = Options(n_points="2", share=1)  # ty:ignore[invalid-argument-type]
    assert isinstance(options.n_points, int)
    assert options.n_points == 2
    assert isinstance(options.share, float)
    assert options.share == 1.0


def test_fractional_float_for_int_field_raises():
    with pytest.raises(CustomError):
        Options(n_points=2.5)  # ty:ignore[invalid-argument-type]


def test_constraint_violations_raise_the_translated_error():
    with pytest.raises(CustomError):
        Options(n_points=0)


def test_original_validation_error_is_chained():
    with pytest.raises(CustomError) as exc_info:
        Options(n_points=0)
    assert isinstance(exc_info.value.__cause__, pydantic.ValidationError)


def test_unknown_keyword_arguments_raise_the_translated_error():
    with pytest.raises(CustomError):
        Options(this_is_not_an_option=1)  # ty:ignore[unknown-argument]


def test_all_invalid_fields_are_reported_at_once():
    with pytest.raises(CustomError, match="(?s)n_points.*share"):
        Options(n_points=0, share="not a number")  # ty:ignore[invalid-argument-type]


def test_defaults_are_validated():
    @validated_dataclass(config=DEFAULT_PYDANTIC_CONFIG, make_error=_make_error)
    @dataclass(frozen=True)
    class InvalidDefault:
        n_points: PositiveInt = 0

    with pytest.raises(CustomError):
        InvalidDefault()


def test_defaults_are_coerced():
    @validated_dataclass(config=DEFAULT_PYDANTIC_CONFIG, make_error=_make_error)
    @dataclass(frozen=True)
    class CoercibleDefault:
        n_points: PositiveInt = 2.0  # ty:ignore[invalid-assignment]

    assert isinstance(CoercibleDefault().n_points, int)
    assert CoercibleDefault().n_points == 2


def test_replace_revalidates():
    options = Options()
    assert replace(options, n_points="3").n_points == 3
    with pytest.raises(CustomError):
        replace(options, n_points=0)


def test_decorated_class_is_frozen():
    options = Options()
    with pytest.raises(FrozenInstanceError):
        options.n_points = 2  # ty:ignore[invalid-assignment]


def test_docstring_and_annotations_are_preserved():
    assert Options.__doc__ == "Docstring of Options."
    assert "n_points" in Options.__annotations__
    assert "share" in Options.__annotations__


def test_fields_inherited_from_a_base_dataclass_are_validated():
    @dataclass(frozen=True)
    class Base:
        n_points: PositiveInt = 1

    @validated_dataclass(config=DEFAULT_PYDANTIC_CONFIG, make_error=_make_error)
    @dataclass(frozen=True)
    class Child(Base):
        share: float = 0.5

    assert Child(n_points="2").n_points == 2  # ty:ignore[invalid-argument-type]
    with pytest.raises(CustomError):
        Child(n_points=0)


def test_strict_config_rejects_values_that_need_conversion():
    assert StrictOptions(n_points=2, label="b").n_points == 2
    with pytest.raises(CustomError):
        StrictOptions(n_points="2")  # ty:ignore[invalid-argument-type]
    with pytest.raises(CustomError):
        StrictOptions(label=3)  # ty:ignore[invalid-argument-type]


def test_decorated_class_is_not_replaced():
    @dataclass(frozen=True)
    class Plain:
        n_points: PositiveInt = 1

    decorator = validated_dataclass(
        config=DEFAULT_PYDANTIC_CONFIG, make_error=_make_error
    )
    assert decorator(Plain) is Plain


def test_validator_is_only_built_when_it_is_needed():
    lazy_validator = _LazyValidator(Options)
    assert "validator" not in vars(lazy_validator)
    lazy_validator.validator  # noqa: B018
    assert "validator" in vars(lazy_validator)


def test_pickled_lazy_validator_contains_the_validator():
    restored = pickle.loads(pickle.dumps(_LazyValidator(Options)))
    assert "validator" in vars(restored)


@pytest.mark.parametrize("instantiate_before_pickling", [True, False])
def test_class_that_is_pickled_by_value_validates_in_other_process(
    instantiate_before_pickling,
):
    # cloudpickle pickles classes that are defined inside of functions by value, as
    # it does for classes that are defined in a script or notebook. This is what
    # happens to user defined algorithms in parallel optimizations.
    @validated_dataclass(config=DEFAULT_PYDANTIC_CONFIG, make_error=ValueError)
    @dataclass(frozen=True)
    class Local:
        n_points: PositiveInt = 1

    if instantiate_before_pickling:
        Local()

    code = """
import pickle, sys
cls = pickle.loads(sys.stdin.buffer.read())
assert cls(n_points="2").n_points == 2
try:
    cls(n_points=0)
except ValueError:
    print("validated")
"""
    out = subprocess.run(
        [sys.executable, "-c", code],
        input=cloudpickle.dumps(Local),
        capture_output=True,
        check=True,
    )
    assert out.stdout.decode().strip() == "validated"
