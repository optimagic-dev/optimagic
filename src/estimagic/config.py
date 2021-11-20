from pathlib import Path

DEFAULT_SEED = 5471

DOCS_DIR = Path(__file__).parent.parent / "docs"

EXAMPLE_DIR = Path(__file__).parent / "examples"


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
    import nlopt  # noqa: F401
except ImportError:
    IS_NLOPT_INSTALLED = False
else:
    IS_NLOPT_INSTALLED = True

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

try:
    import pygmo  # noqa: F401
except ImportError:
    IS_PYGMO_INSTALLED = False
else:
    IS_PYGMO_INSTALLED = True


try:
    import cyipopt  # noqa: F401
except ImportError:
    IS_CYIPOPT_INSTALLED = False
else:
    IS_CYIPOPT_INSTALLED = True


try:
    import fides  # noqa: F401
except ImportError:
    IS_FIDES_INSTALLED = False
else:
    IS_FIDES_INSTALLED = True


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
