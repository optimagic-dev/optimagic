from typing import Any

from optimagic.typing import (
    GtOneFloat,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)


def _process_float_like(value: Any) -> float:
    """Process a value that should be converted to a float."""
    return float(value)


def _process_int_like(value: Any) -> int:
    """Process a value that should be converted to an int."""
    if isinstance(value, int):
        return value
    elif isinstance(value, str):
        return int(float(value))
    else:
        return int(value)


def _process_positive_int_like(value: Any) -> PositiveInt:
    """Process a value that should be converted to a positive int."""
    out = _process_int_like(value)
    if out <= 0:
        raise ValueError(f"Value must be positive, got {out}")
    return out


def _process_non_negative_int_like(value: Any) -> NonNegativeInt:
    """Process a value that should be converted to a non-negative int."""
    out = _process_int_like(value)
    if out < 0:
        raise ValueError(f"Value must be non-negative, got {out}")
    return out


def _process_positive_float_like(value: Any) -> PositiveFloat:
    """Process a value that should be converted to a positive float."""
    out = _process_float_like(value)
    if out <= 0:
        raise ValueError(f"Value must be positive, got {out}")
    return out


def _process_non_negative_float_like(value: Any) -> NonNegativeFloat:
    """Process a value that should be converted to a non-negative float."""
    out = _process_float_like(value)
    if out < 0:
        raise ValueError(f"Value must be non-negative, got {out}")
    return out


def _process_gt_one_float_like(value: Any) -> GtOneFloat:
    """Process a value that should be converted to a float greater than one."""
    out = _process_float_like(value)
    if out <= 1:
        raise ValueError(f"Value must be greater than one, got {out}")
    return out


def _process_bool_like(value: Any) -> bool:
    """Process a value that should be converted to a bool."""
    if isinstance(value, bool):
        return value
    elif isinstance(value, str):
        if value.lower() in {"true", "1", "yes"}:
            return True
        elif value.lower() in {"false", "0", "no"}:
            return False

    return bool(value)


TYPE_CONVERTERS = {
    float: _process_float_like,
    int: _process_int_like,
    bool: _process_bool_like,
    PositiveInt: _process_positive_int_like,
    NonNegativeInt: _process_non_negative_int_like,
    PositiveFloat: _process_positive_float_like,
    NonNegativeFloat: _process_non_negative_float_like,
    GtOneFloat: _process_gt_one_float_like,
}
