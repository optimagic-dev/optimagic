from dataclasses import dataclass
from typing import Literal, TypedDict

from optimagic.exceptions import InvalidScalingOptionsError


@dataclass(frozen=True)
class ScalingOptions:
    method: Literal["start_values", "bounds"] = "start_values"
    clipping_value: float = 0.1
    magnitude: float = 1.0


class ScalingOptionsDict(TypedDict):
    method: Literal["start_values", "bounds"]
    clipping_value: float
    magnitude: float


def pre_process_scaling(
    scaling: bool | ScalingOptions | ScalingOptionsDict | None,
) -> ScalingOptions | None:
    """Convert all valid types of scaling options to optimagic.ScalingOptions.

    This just harmonizes multiple ways of specifying scaling options into a single
    format. It performs runtime type checks, but it does not check whether scaling
    options are consistent with other option choices.

    Args:
        scaling: The user provided scaling options.

    Returns:
        The scaling options in the optimagic format.

    Raises:
        InvalidScalingOptionsError: If scaling options cannot be processed, e.g. because
            they do not have the correct type.

    """
    if isinstance(scaling, bool):
        scaling = ScalingOptions() if scaling else None
    elif isinstance(scaling, ScalingOptions) or scaling is None:
        pass
    else:
        try:
            scaling = ScalingOptions(**scaling)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            raise InvalidScalingOptionsError(
                f"Invalid scaling options of type: {type(scaling)}. Scaling options "
                "must be of type optimagic.ScalingOptions, a dictionary with a subset "
                "of the keys {'method', 'clipping_value', 'magnitude'}, None, or a "
                "boolean."
            ) from e

    if isinstance(scaling, ScalingOptions):
        if scaling.method not in ("start_values", "bounds"):
            raise InvalidScalingOptionsError(
                f"Invalid scaling method: {scaling.method}. Valid methods are "
                "'start_values' and 'bounds'."
            )

        if not isinstance(scaling.clipping_value, (int, float)):
            raise InvalidScalingOptionsError(
                f"Invalid clipping value: {scaling.clipping_value}. Clipping value "
                "must be a number."
            )

        if not isinstance(scaling.magnitude, (int, float)) or scaling.magnitude <= 0:
            raise InvalidScalingOptionsError(
                f"Invalid scaling magnitude: {scaling.magnitude}. Scaling magnitude "
                "must be a positive number."
            )

    return scaling
