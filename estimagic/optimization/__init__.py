import inspect

from estimagic.config import IS_CYIPOPT_INSTALLED
from estimagic.config import IS_DFOLS_INSTALLED
from estimagic.config import IS_PETSC4PY_INSTALLED
from estimagic.config import IS_PYBOBYQA_INSTALLED
from estimagic.optimization import cyipopt_optimizers
from estimagic.optimization import nag_optimizers
from estimagic.optimization import scipy_optimizers
from estimagic.optimization import tao_optimizers


SCIPY_FUNCTIONS = dict(inspect.getmembers(scipy_optimizers, inspect.isfunction))
SCIPY_ALGORITHMS = {k: v for k, v in SCIPY_FUNCTIONS.items() if k.startswith("scipy")}

if IS_PETSC4PY_INSTALLED:
    TAO_FUNCTIONS = dict(inspect.getmembers(tao_optimizers, inspect.isfunction))
    TAO_ALGORITHMS = {k: v for k, v in TAO_FUNCTIONS.items() if k.startswith("tao")}
else:
    TAO_ALGORITHMS = {}

AVAILABLE_ALGORITHMS = {**SCIPY_ALGORITHMS, **TAO_ALGORITHMS}

if IS_PYBOBYQA_INSTALLED:
    AVAILABLE_ALGORITHMS["nag_pybobyqa"] = nag_optimizers.nag_pybobyqa

if IS_DFOLS_INSTALLED:
    AVAILABLE_ALGORITHMS["nag_dfols"] = nag_optimizers.nag_dfols

if IS_CYIPOPT_INSTALLED:
    AVAILABLE_ALGORITHMS["ipopt"] = cyipopt_optimizers.ipopt
