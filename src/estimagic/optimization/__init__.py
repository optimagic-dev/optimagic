import inspect

from estimagic.config import IS_CYIPOPT_INSTALLED
from estimagic.config import IS_DFOLS_INSTALLED
from estimagic.config import IS_FIDES_INSTALLED
from estimagic.config import IS_NLOPT_INSTALLED
from estimagic.config import IS_PETSC4PY_INSTALLED
from estimagic.config import IS_PYBOBYQA_INSTALLED
from estimagic.config import IS_PYGMO_INSTALLED
from estimagic.optimization import cyipopt_optimizers
from estimagic.optimization import fides_optimizers
from estimagic.optimization import nag_optimizers
from estimagic.optimization import nlopt_optimizers
from estimagic.optimization import pygmo_optimizers
from estimagic.optimization import scipy_optimizers
from estimagic.optimization import tao_optimizers


COLLECTED_FUNCTIONS = {
    **dict(inspect.getmembers(scipy_optimizers, inspect.isfunction)),
}

if IS_PETSC4PY_INSTALLED:
    COLLECTED_FUNCTIONS.update(
        **dict(inspect.getmembers(tao_optimizers, inspect.isfunction))
    )

if IS_NLOPT_INSTALLED:
    COLLECTED_FUNCTIONS.update(
        **dict(inspect.getmembers(nlopt_optimizers, inspect.isfunction))
    )

# drop private and helper functions
AVAILABLE_ALGORITHMS = {}
PUBLIC_HELPERS = [
    "calculate_trustregion_initial_radius",
    "get_scipy_bounds",
    "process_scipy_result",
]
for k, v in COLLECTED_FUNCTIONS.items():
    if not k.startswith("_") and k not in PUBLIC_HELPERS:
        AVAILABLE_ALGORITHMS[k] = v

if IS_PYBOBYQA_INSTALLED:
    AVAILABLE_ALGORITHMS["nag_pybobyqa"] = nag_optimizers.nag_pybobyqa

if IS_DFOLS_INSTALLED:
    AVAILABLE_ALGORITHMS["nag_dfols"] = nag_optimizers.nag_dfols

if IS_PYGMO_INSTALLED:
    _PYGMO_FUNCTIONS = dict(inspect.getmembers(pygmo_optimizers, inspect.isfunction))
    PYGMO_ALGORITHMS = {
        k: v for k, v in _PYGMO_FUNCTIONS.items() if k.startswith("pygmo_")
    }
    AVAILABLE_ALGORITHMS.update(**PYGMO_ALGORITHMS)

if IS_CYIPOPT_INSTALLED:
    AVAILABLE_ALGORITHMS["ipopt"] = cyipopt_optimizers.ipopt

if IS_FIDES_INSTALLED:
    AVAILABLE_ALGORITHMS["fides"] = fides_optimizers.fides

GLOBAL_ALGORITHMS = [
    "nlopt_direct",
    "nlopt_esch",
    "nlopt_isres",
    "nlopt_crs2_lm",
]
if IS_PYGMO_INSTALLED:
    GLOBAL_ALGORITHMS += PYGMO_ALGORITHMS.keys()
