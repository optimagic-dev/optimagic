"""PyGAD optimizer configuration classes and utilities.

This module provides easy access to PyGAD mutation classes and Protocols.

Example:
    >>> import optimagic as om
    >>> mutation = om.pygad.RandomMutation(probability=0.15, by_replacement=True)
    >>> result = om.minimize(
    ...     fun=lambda x: x @ x,
    ...     params=[1.0, 2.0, 3.0],
    ...     algorithm=om.algos.pygad(mutation=mutation)
    ... )

"""

from optimagic.optimizers.pygad_optimizer import (
    AdaptiveMutation as _AdaptiveMutation,
)
from optimagic.optimizers.pygad_optimizer import (
    CrossoverFunction,
    GeneConstraintFunction,
    MutationFunction,
    ParentSelectionFunction,
)
from optimagic.optimizers.pygad_optimizer import (
    InversionMutation as _InversionMutation,
)
from optimagic.optimizers.pygad_optimizer import (
    RandomMutation as _RandomMutation,
)
from optimagic.optimizers.pygad_optimizer import (
    ScrambleMutation as _ScrambleMutation,
)
from optimagic.optimizers.pygad_optimizer import (
    SwapMutation as _SwapMutation,
)

RandomMutation = _RandomMutation
AdaptiveMutation = _AdaptiveMutation
SwapMutation = _SwapMutation
InversionMutation = _InversionMutation
ScrambleMutation = _ScrambleMutation

__all__ = [
    "RandomMutation",
    "AdaptiveMutation",
    "SwapMutation",
    "InversionMutation",
    "ScrambleMutation",
    "MutationFunction",
    "CrossoverFunction",
    "ParentSelectionFunction",
    "GeneConstraintFunction",
]
