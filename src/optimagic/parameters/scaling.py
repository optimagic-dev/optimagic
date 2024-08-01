from dataclasses import dataclass
from typing import Literal, TypedDict

from typing_extensions import NotRequired

from optimagic.exceptions import InvalidScalingOptionsError


@dataclass(frozen=True)
class ScalingOptions:
    method: Literal["start_values", "bound"] = "start_values"
    clipping_value: float = 0.1
    magnitude: float = 1.0


class ScalingOptionsDict(TypedDict):
    method: NotRequired[Literal["start_values", "bound"]]
    clipping_value: NotRequired[float]
    magnitude: NotRequired[float]


def pre_process_scaling(
    scaling: bool | ScalingOptions | ScalingOptionsDict | None,
) -> ScalingOptions | None:
    """Convert all valid types of specifying scaling options to
    optimagic.ScalingOptions.

    This just harmonizes multiple ways of specifying scaling options into a single
    format. It does not check that scaling options are valid.

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
    return scaling
