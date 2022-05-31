from estimagic import utilities
from estimagic.benchmarking.get_benchmark_problems import get_benchmark_problems
from estimagic.benchmarking.run_benchmark import run_benchmark
from estimagic.differentiation.derivatives import first_derivative
from estimagic.differentiation.derivatives import second_derivative
from estimagic.estimation.estimate_ml import estimate_ml
from estimagic.estimation.estimate_msm import estimate_msm
from estimagic.estimation.msm_weighting import get_moments_cov
from estimagic.inference.bootstrap import bootstrap
from estimagic.inference.bootstrap import bootstrap_from_outcomes
from estimagic.logging.read_log import OptimizeLogReader
from estimagic.optimization.optimize import maximize
from estimagic.optimization.optimize import minimize
from estimagic.parameters.constraint_tools import check_constraints
from estimagic.parameters.constraint_tools import count_free_params
from estimagic.visualization.convergence_plot import convergence_plot
from estimagic.visualization.derivative_plot import derivative_plot
from estimagic.visualization.estimation_table import estimation_table
from estimagic.visualization.estimation_table import render_html
from estimagic.visualization.estimation_table import render_latex
from estimagic.visualization.history_plots import criterion_plot
from estimagic.visualization.history_plots import params_plot
from estimagic.visualization.lollipop_plot import lollipop_plot
from estimagic.visualization.profile_plot import profile_plot
from estimagic.visualization.slice_plot import slice_plot

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
    "__version__",
]
