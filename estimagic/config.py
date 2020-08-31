from pathlib import Path

DEFAULT_DATABASE_NAME = "logging.db"
DEFAULT_SEED = 5471
MAX_CRITERION_PENALTY = 1e200

TEST_DIR = Path(__file__).parent / "tests"

DOCS_DIR = Path(__file__).parent.parent / "docs"

DEFAULT_N_CORES = 1

try:
    from petsc4py import PETSc  # noqa: F401
except ImportError:
    IS_PETSC4PY_INSTALLED = False
else:
    IS_PETSC4PY_INSTALLED = True
