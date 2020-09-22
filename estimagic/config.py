from pathlib import Path

DEFAULT_DATABASE_NAME = "logging.db"
DEFAULT_SEED = 5471

TEST_DIR = Path(__file__).parent / "tests"

DOCS_DIR = Path(__file__).parent.parent / "docs"

DEFAULT_N_CORES = 1

CRITERION_PENALTY_SLOPE = 0.1
CRITERION_PENALTY_CONSTANT = 100

# =====================================================================================
# Check Available Packages
# =====================================================================================

try:
    from petsc4py import PETSc  # noqa: F401
except ImportError:
    IS_PETSC4PY_INSTALLED = False
else:
    IS_PETSC4PY_INSTALLED = True

try:
    import matplotlib  # noqa: F401
except ImportError:
    IS_MATPLOTLIB_INSTALLED = False
else:
    IS_MATPLOTLIB_INSTALLED = True

try:
    import pybobyqa  # noqa: F401
except ImportError:
    IS_PYBOBYQA_INSTALLED = False
else:
    IS_PYBOBYQA_INSTALLED = True


# =====================================================================================
# Stopping Criteria
# =====================================================================================

RELATIVE_CRITERION_TOLERANCE = 2e-9
"""float: Inspired by scipy L-BFGS-B defaults, but rounded."""

ABSOLUTE_CRITERION_TOLERANCE = 0
"""float: Disabled by default because it is very problem specific."""

ABSOLUTE_GRADIENT_TOLERANCE = 1e-5
"""float: Same as scipy."""

RELATIVE_GRADIENT_TOLERANCE = 1e-8

SCALED_GRADIENT_TOLERANCE = 1e-8

RELATIVE_PARAMS_TOLERANCE = 1e-5
"""float: Same as scipy."""

ABSOLUTE_PARAMS_TOLERANCE = 0
"""float: Disabled by default because it is very problem specific."""

MAX_CRITERION_EVALUATIONS = 1_000_000
MAX_ITERATIONS = 1_000_000

SECOND_BEST_ABSOLUTE_CRITERION_TOLERANCE = 1e-08
"""float: absolute criterion tolerance estimagic requires if no other stopping
criterion apart from max iterations etc. is available
this is taken from scipy (SLSQP's value, smaller than Nelder-Mead)"""

SECOND_BEST_ABSOLUTE_PARAMS_TOLERANCE = 0.0001
"""float: The absolute parameter tolerance estimagic requires if no other stopping
criterion apart from max iterations etc. is available. This is taken from Nelder-Mead.
"""

# =====================================================================================
# Other Common Tuning Parameters for Optimization Algorithms
# =====================================================================================

MAX_LINE_SEARCH_STEPS = 20
"""int: Inspired by scipy L-BFGS-B."""

LIMITED_MEMORY_STORAGE_LENGTH = 10
"""int: Taken from scipy L-BFGS-B."""

# -------------------------
# Trust Region Parameters
# -------------------------

INITIAL_TRUST_RADIUS = 1
"""float: recommended for scipy_trust_constr in :cite:`Conn2000`, p. 19.
It is also scipy's default for COBYLA's start Rho, which behaves similar to an
initial trust radius. pyBOBYQA's default is 0.1 times the norm of the start params
but no larger than 1."""

# ---------------------------------------------
# Numerical Algorithm Group Tuning Parameters
# ---------------------------------------------

RANDOM_INITIAL_DIRECTIONS = False
"""bool: Whether to draw the initial directions randomly or use the coordinate
directions."""

RANDOM_DIRECTIONS_ORTHOGONAL = True
"""bool: Whether to make randomly drawn initial directions orthogonal."""

CRITERION_NOISY = False
"""bool: Whether the criterion function is noisy, i.e. does not always return the
same value when evaluated at the same parameter values."""


def NR_EVALS_PER_POINT(delta, rho, iter, nrestarts):  # noqa: A002, N802
    """Evaluate the criterion function once at every point.

    This is only applicable for criterion functions with stochastic noise,
    when averaging multiple evaluations at the same point increases accuracy.

    Args:
        delta (float): the trust region radius.
        rho (float): the lower bound on the trust region radius.
        iter (int): how many iterations the algorithm has been running for.
        nrestarts (int): how many restarts have been performed

    Returns:
        nr_evals_per_point (int)

    """
    return 1


NR_INTERPOLATION_POINTS = None
"""the number of interpolation points to use. The default is to calculate it from the
problem dimension. See the algorithm's function docstring for details."""

# =================================================================================
# Dashboard Defaults
# =================================================================================

Y_RANGE_PADDING = 0.05
Y_RANGE_PADDING_UNITS = "absolute"
PLOT_WIDTH = 750
PLOT_HEIGHT = 300
MIN_BORDER_LEFT = 50
MIN_BORDER_RIGHT = 50
MIN_BORDER_TOP = 20
MIN_BORDER_BOTTOM = 50
TOOLBAR_LOCATION = None
GRID_VISIBLE = False
MINOR_TICK_LINE_COLOR = None
MAJOR_TICK_OUT = 0
MINOR_TICK_OUT = 0
MAJOR_TICK_IN = 0
OUTLINE_LINE_WIDTH = 0
LEGEND_LABEL_TEXT_FONT_SIZE = "11px"
LEGEND_SPACING = -2
