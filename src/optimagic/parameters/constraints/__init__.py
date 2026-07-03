"""Internal machinery for enforcing constraints via reparametrization.

This subpackage is being built up during the constraints refactoring. It will
eventually contain the full pipeline that turns user provided constraints into a
reparametrization of the optimization problem:

1. resolution: selectors -> positions in the flat parameter vector (resolution.py)
2. validation: checks that constraints are well specified and satisfied at the
   start params
3. normalization: rewrites of one constraint kind into another (e.g. increasing ->
   linear)
4. consolidation: cross-constraint interactions (equalities, fixes, bounds) and
   construction of the transformations
5. converter assembly: the space converter that maps between external and internal
   parameters

The user facing constraint classes live in optimagic.constraints.

"""

from optimagic.parameters.constraints.resolution import (
    resolve_constraints,
    to_legacy_dicts,
)
from optimagic.parameters.constraints.types import (
    ConstraintSource,
    ResolvedConstraint,
)

__all__ = [
    "ConstraintSource",
    "ResolvedConstraint",
    "resolve_constraints",
    "to_legacy_dicts",
]
