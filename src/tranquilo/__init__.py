from tranquilo import utilities
from tranquilo.benchmarking.get_benchmark_problems import get_benchmark_problems
from tranquilo.benchmarking.run_benchmark import run_benchmark
from tranquilo.differentiation.derivatives import first_derivative, second_derivative
from tranquilo.estimation.estimate_ml import LikelihoodResult, estimate_ml
from tranquilo.estimation.estimate_msm import MomentsResult, estimate_msm
from tranquilo.estimation.msm_weighting import get_moments_cov
from tranquilo.inference.bootstrap import BootstrapResult, bootstrap
from tranquilo.logging.read_log import OptimizeLogReader
from tranquilo.optimization.optimize import maximize, minimize
from tranquilo.optimization.optimize_result import OptimizeResult
from tranquilo.parameters.constraint_tools import check_constraints, count_free_params
from tranquilo.visualization.convergence_plot import convergence_plot
from tranquilo.visualization.derivative_plot import derivative_plot
from tranquilo.visualization.estimation_table import (
    estimation_table,
    render_html,
    render_latex,
)
from tranquilo.visualization.history_plots import criterion_plot, params_plot
from tranquilo.visualization.lollipop_plot import lollipop_plot
from tranquilo.visualization.profile_plot import profile_plot
from tranquilo.visualization.slice_plot import slice_plot

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
    "bootstrap",
    "bootstrap_from_outcomes",
    "estimate_msm",
    "estimate_ml",
    "get_moments_cov",
    "run_benchmark",
    "get_benchmark_problems",
    "profile_plot",
    "convergence_plot",
    "lollipop_plot",
    "derivative_plot",
    "slice_plot",
    "estimation_table",
    "render_html",
    "render_latex",
    "criterion_plot",
    "params_plot",
    "count_free_params",
    "check_constraints",
    "OptimizeLogReader",
    "OptimizeResult",
    "BootstrapResult",
    "LikelihoodResult",
    "MomentsResult",
    "__version__",
]
