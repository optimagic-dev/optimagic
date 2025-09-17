from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class BenchmarkProblem(ABC):
    @abstractmethod
    def fun(self, x: NDArray[np.float64]) -> float | NDArray[np.float64]:
        pass

    @property
    @abstractmethod
    def start_x(self) -> NDArray[np.float64]:
        pass

    @property
    @abstractmethod
    def solution_x(self) -> NDArray[np.float64] | None:
        pass

    @property
    @abstractmethod
    def start_fun(self) -> float:
        pass

    @property
    @abstractmethod
    def solution_fun(self) -> float:
        pass
