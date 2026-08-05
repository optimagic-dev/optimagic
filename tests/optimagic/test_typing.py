"""Tests for the validated_dataclass decorator and the shared pydantic configs.

These tests pin down how optimagic uses pydantic (error translation, config
choices, preserved class attributes), not pydantic's own validation logic.

"""

from dataclasses import FrozenInstanceError, dataclass, replace

import pydantic
import pytest

from optimagic.typing import (
    DEFAULT_PYDANTIC_CONFIG,
    STRICT_PYDANTIC_CONFIG,
    PositiveInt,
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
    options = Options(n_points="2", share=1)
    assert isinstance(options.n_points, int)
    assert options.n_points == 2
    assert isinstance(options.share, float)
    assert options.share == 1.0


def test_fractional_float_for_int_field_raises():
    with pytest.raises(CustomError):
        Options(n_points=2.5)


def test_constraint_violations_raise_the_translated_error():
    with pytest.raises(CustomError):
        Options(n_points=0)


def test_original_validation_error_is_chained():
    with pytest.raises(CustomError) as exc_info:
        Options(n_points=0)
    assert isinstance(exc_info.value.__cause__, pydantic.ValidationError)


def test_unknown_keyword_arguments_raise_the_translated_error():
    with pytest.raises(CustomError):
        Options(this_is_not_an_option=1)


def test_all_invalid_fields_are_reported_at_once():
    with pytest.raises(CustomError, match="(?s)n_points.*share"):
        Options(n_points=0, share="not a number")


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
        n_points: PositiveInt = 2.0  # type: ignore[assignment]

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
        options.n_points = 2


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

    assert Child(n_points="2").n_points == 2
    with pytest.raises(CustomError):
        Child(n_points=0)


def test_strict_config_rejects_values_that_need_conversion():
    assert StrictOptions(n_points=2, label="b").n_points == 2
    with pytest.raises(CustomError):
        StrictOptions(n_points="2")
    with pytest.raises(CustomError):
        StrictOptions(label=3)
