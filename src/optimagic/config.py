import importlib.util
from pathlib import Path

import pandas as pd
import plotly.express as px
from packaging import version

DOCS_DIR = Path(__file__).parent.parent / "docs"
OPTIMAGIC_ROOT = Path(__file__).parent

PLOTLY_TEMPLATE = "simple_white"
PLOTLY_PALETTE = px.colors.qualitative.Set2

DEFAULT_N_CORES = 1

CRITERION_PENALTY_SLOPE = 0.1
CRITERION_PENALTY_CONSTANT = 100


def _is_installed(module_name: str) -> bool:
    """Return True if the given module is installed, otherwise False."""
    return importlib.util.find_spec(module_name) is not None


# ======================================================================================
# Check Available Packages
# ======================================================================================

IS_PETSC4PY_INSTALLED = _is_installed("petsc4py")
IS_NLOPT_INSTALLED = _is_installed("nlopt")
IS_PYBOBYQA_INSTALLED = _is_installed("pybobyqa")
IS_DFOLS_INSTALLED = _is_installed("dfols")
IS_PYGMO_INSTALLED = _is_installed("pygmo")
IS_CYIPOPT_INSTALLED = _is_installed("cyipopt")
IS_FIDES_INSTALLED = _is_installed("fides")
IS_JAX_INSTALLED = _is_installed("jax")
IS_TRANQUILO_INSTALLED = _is_installed("tranquilo")
IS_NUMBA_INSTALLED = _is_installed("numba")
IS_IMINUIT_INSTALLED = _is_installed("iminuit")
IS_NEVERGRAD_INSTALLED = _is_installed("nevergrad")
# despite the similar names, the bayes_opt and bayes_optim packages are
# completely unrelated. However, both of them are dependencies of nevergrad.
IS_BAYESOPTIM_INSTALLED = _is_installed("bayes-optim")
# Note: There is a dependancy conflict with nevergrad and bayesian_optimization
# installing nevergrad pins bayesian_optimization to 1.4.0,
# but "bayes_opt" requires bayesian_optimization>=2.0.0 to work.
# so if nevergrad is installed, bayes_opt will not work and vice-versa.
IS_BAYESOPT_INSTALLED_AND_VERSION_NEWER_THAN_2 = (
    _is_installed("bayes_opt")
    and importlib.metadata.version("bayesian_optimization") > "2.0.0"
)
IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED = _is_installed("gradient_free_optimizers")
IS_PYGAD_INSTALLED = _is_installed("pygad")


# ======================================================================================
# Check if pandas version is newer or equal to version 2.1.0
# ======================================================================================

IS_PANDAS_VERSION_NEWER_OR_EQUAL_TO_2_1_0 = version.parse(
    pd.__version__
) >= version.parse("2.1.0")
