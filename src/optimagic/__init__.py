from optimagic import utilities
from optimagic.benchmarking.get_benchmark_problems import get_benchmark_problems
from optimagic.benchmarking.run_benchmark import run_benchmark
from optimagic.benchmarking.benchmark_reports import convergence_report
from optimagic.benchmarking.benchmark_reports import rank_report
from optimagic.benchmarking.benchmark_reports import traceback_report
from optimagic.differentiation.derivatives import first_derivative, second_derivative
from optimagic.logging.read_log import OptimizeLogReader
from optimagic.optimization.optimize import maximize, minimize
from optimagic.optimization.optimize_result import OptimizeResult
from optimagic.parameters.constraint_tools import check_constraints, count_free_params
from optimagic.visualization.convergence_plot import convergence_plot

from optimagic.visualization.history_plots import criterion_plot, params_plot
from optimagic.visualization.profile_plot import profile_plot
from optimagic.visualization.slice_plot import slice_plot
import warnings

try:
    from ._version import version as __version__
except ImportError:
    # broken installation, we don't even try unknown only works because we do poor mans
    # version compare
    __version__ = "unknown"


__all__ = [
    "maximize",
    "minimize",
    "utilities",
    "first_derivative",
    "second_derivative",
    "run_benchmark",
    "get_benchmark_problems",
    "profile_plot",
    "convergence_plot",
    "convergence_report",
    "rank_report",
    "traceback_report",
    "slice_plot",
    "criterion_plot",
    "params_plot",
    "count_free_params",
    "check_constraints",
    "OptimizeLogReader",
    "OptimizeResult",
    "__version__",
]

warnings.warn(
    "estimagic has been renamed to optimagic. Please uninstall estimagic and install "
    "optimagic instead. Don't worry, your estimagic imports will still work if you "
    "install optimagic, and simple warnings will help you to adjust them for future "
    "releases.\n\n"
    "To make these changes using pip, run:\n"
    "-------------------------------------\n"
    "$ pip uninstall estimagic\n"
    "$ pip install optimagic\n\n"
    "For conda users, use:\n"
    "---------------------\n"
    "$ conda remove estimagic\n"
    "$ conda install -c conda-forge optimagic\n",
    FutureWarning,
)
