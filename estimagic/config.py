import inspect
from pathlib import Path

from estimagic.optimization import scipy_optimizers

DEFAULT_DATABASE_NAME = "logging.db"
DEFAULT_SEED = 5471
MAX_CRITERION_PENALTY = 1e200

TEST_DIR = Path(__file__).parent / "tests"

DOCS_DIR = Path(__file__).parent.parent / "docs"


DEFAULT_N_CORES = 1


AVAILABLE_ALGORITHMS = dict(inspect.getmembers(scipy_optimizers, inspect.isfunction))
AVAILABLE_ALGORITHMS = {
    key: val for key, val in AVAILABLE_ALGORITHMS.items() if not key.startswith("_")
}
