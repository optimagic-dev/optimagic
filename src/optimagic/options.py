from dataclasses import dataclass
from typing import Literal


@dataclass
class ScalingOptions:
    method: Literal["start_values", "bound"] = "start_values"
    clipping_value: float = 0.1
    magnitude: float = 1.0
