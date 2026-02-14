import importlib.util
from pathlib import Path

import pandas as pd
import plotly.express as px
import scipy
from packaging import version

DOCS_DIR = Path(__file__).parent.parent / "docs"
OPTIMAGIC_ROOT = Path(__file__).parent

PLOTLY_TEMPLATE = "simple_white"
PLOTLY_PALETTE = px.colors.qualitative.Set2

# The hex strings are obtained from the Plotly D3 qualitative palette.
DEFAULT_PALETTE = [
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
    "#17BECF",
]

DEFAULT_N_CORES = 1

CRITERION_PENALTY_SLOPE = 0.1
CRITERION_PENALTY_CONSTANT = 100


def _is_installed(module_name: str) -> bool:
    """Return True if the given module is installed, otherwise False."""
    return importlib.util.find_spec(module_name) is not None


# ======================================================================================
# Check Available Optimization Packages
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
IS_PYSWARMS_INSTALLED = _is_installed("pyswarms")

# ======================================================================================
# Check Available Visualization Packages
# ======================================================================================

IS_MATPLOTLIB_INSTALLED = _is_installed("matplotlib")
IS_BOKEH_INSTALLED = _is_installed("bokeh")
IS_ALTAIR_INSTALLED = _is_installed("altair")

# ======================================================================================
# Check if pandas version is newer or equal to version 2.1.0
# ======================================================================================

IS_PANDAS_VERSION_GE_2_1 = version.parse(pd.__version__) >= version.parse("2.1.0")

# ======================================================================================
# Check SciPy Version for COBYQA support (added in scipy 1.14.0)
# ======================================================================================

IS_SCIPY_GE_1_14 = version.parse(scipy.__version__) >= version.parse("1.14.0")
