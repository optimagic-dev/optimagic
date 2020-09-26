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

try:
    import dfols  # noqa: F401
except ImportError:
    IS_DFOLS_INSTALLED = False
else:
    IS_DFOLS_INSTALLED = True

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

SECOND_BEST_ABSOLUTE_PARAMS_TOLERANCE = 1e-08
"""float: The absolute parameter tolerance estimagic requires if no other stopping
criterion apart from max iterations etc. is available. This is taken from pybobyqa.
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

THRESHOLD_FOR_SUCCESSFUL_ITERATION = 0.1
"""float: minimum share of predicted improvement that has to be realized for an
iteration to count as successful."""

THRESHOLD_FOR_VERY_SUCCESFUL_ITERATION = 0.7
"""float: share of predicted improvement that has to be surpassed for an iteration to
count as very successful.
"""

THRESHOLD_FOR_SAFETY_STEP = 0.5
"""float: Threshold for when to call the safety step,
:math:`\text{proposed step} \leq \text{threshold_for_safety_step} \cdot
\text{current_trust_region_radius}`
"""

TRUST_REGION_INCREASE_AFTER_SUCCESS = 2.0
"""float: Ratio by which to increase the trust region radius :math:`\Delta_k` in
very successful iterations (:math:`\gamma_{inc}`)."""

TRUST_REGION_INCREASE_AFTER_LARGE_SUCCESS = 4.0
"""float: Ratio of the proposed step ($\|s_k\|$) by which to increase the trust region
radius (:math:`\Delta_k`) in very successful iterations
(:math:`\overline{\gamma}_{inc}`)."""


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

INTERPOLATION_ROUNDING_ERROR = 0.1
"""float: Internally, all the NAG algorithms store interpolation points with respect
to a base point :math:`x_b`; that is, we store :math:`\{y_t-x_b\}`, which reduces the
risk of roundoff errors. We shift :math:`x_b` to :math:`x_k` when
:math:`\text{proposed step} \leq \text{interpolation_rounding_error} \cdot \|x_k-x_b\|`.
"""

CLIP_CRITERION_IF_OVERFLOWING = True
"""bool: Whether to clip the criterion if it would raise an ``OverflowError`` otherwise.
"""

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
