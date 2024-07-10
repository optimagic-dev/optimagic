from optimagic import utilities
from optimagic.benchmarking.get_benchmark_problems import get_benchmark_problems
from optimagic.benchmarking.run_benchmark import run_benchmark
from optimagic.benchmarking.benchmark_reports import convergence_report
from optimagic.benchmarking.benchmark_reports import rank_report
from optimagic.benchmarking.benchmark_reports import traceback_report
from optimagic.differentiation.derivatives import first_derivative, second_derivative
from optimagic.estimation.estimate_ml import LikelihoodResult, estimate_ml
from optimagic.estimation.estimate_msm import MomentsResult, estimate_msm
from optimagic.estimation.msm_weighting import get_moments_cov
from optimagic.inference.bootstrap import BootstrapResult, bootstrap
from optimagic.logging.read_log import OptimizeLogReader
from optimagic.optimization.optimize import maximize, minimize
from optimagic.optimization.optimize_result import OptimizeResult
from optimagic.parameters.constraint_tools import check_constraints, count_free_params
from optimagic.visualization.convergence_plot import convergence_plot
from optimagic.visualization.estimation_table import (
    estimation_table,
    render_html,
    render_latex,
)
from optimagic.visualization.history_plots import criterion_plot, params_plot
from optimagic.visualization.lollipop_plot import lollipop_plot
from optimagic.visualization.profile_plot import profile_plot
from optimagic.visualization.slice_plot import slice_plot

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
    "convergence_report",
    "rank_report",
    "traceback_report",
    "lollipop_plot",
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
