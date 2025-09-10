import importlib.util
from pathlib import Path

import pandas as pd
import plotly.express as px
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
IS_BAYESOPT_INSTALLED = _is_installed("bayes_opt")
IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED = _is_installed("gradient_free_optimizers")
IS_PYGAD_INSTALLED = _is_installed("pygad")
IS_PYSWARMS_INSTALLED = _is_installed("pyswarms")

# ======================================================================================
# Check Available Visualization Packages
# ======================================================================================

IS_MATPLOTLIB_INSTALLED = _is_installed("matplotlib")


# ======================================================================================
# Check if pandas version is newer or equal to version 2.1.0
# ======================================================================================

IS_PANDAS_VERSION_NEWER_OR_EQUAL_TO_2_1_0 = version.parse(
    pd.__version__
) >= version.parse("2.1.0")
