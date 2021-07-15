import functools
import warnings

import numpy as np
import scipy
import nlopt

from estimagic.optimization.algo_options import CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE
from estimagic.optimization.algo_options import CONVERGENCE_ABSOLUTE_GRADIENT_TOLERANCE
from estimagic.optimization.algo_options import CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE
from estimagic.optimization.algo_options import CONVERGENCE_RELATIVE_CRITERION_TOLERANCE
from estimagic.optimization.algo_options import CONVERGENCE_RELATIVE_GRADIENT_TOLERANCE
from estimagic.optimization.algo_options import CONVERGENCE_RELATIVE_PARAMS_TOLERANCE
from estimagic.optimization.algo_options import (
    CONVERGENCE_SECOND_BEST_ABSOLUTE_CRITERION_TOLERANCE,
)
from estimagic.optimization.algo_options import (
    CONVERGENCE_SECOND_BEST_ABSOLUTE_PARAMS_TOLERANCE,
)
from estimagic.optimization.algo_options import LIMITED_MEMORY_STORAGE_LENGTH
from estimagic.optimization.algo_options import MAX_LINE_SEARCH_STEPS
from estimagic.optimization.algo_options import STOPPING_MAX_CRITERION_EVALUATIONS
from estimagic.optimization.algo_options import STOPPING_MAX_ITERATIONS
from estimagic.optimization.utilities import calculate_trustregion_initial_radius

DEFAULT_ALGO_INFO = {
    "primary_criterion_entry": "value",
    "parallelizes": False,
    "needs_scaling": False,

}

def nlopt_bobyqa(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_relative_params_tolerance=CONVERGENCE_RELATIVE_PARAMS_TOLERANCE,
    convergence_absolute_params_tolerance=CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE,
    convergence_relative_criterion_tolerance=CONVERGENCE_RELATIVE_CRITERION_TOLERANCE,
    convergence_absolute_criterion_tolerance=CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS,
):
    """ Minimize a scalar function using nlopt algorithm derived from the
     BOBYQA subroutine of M. J. D. Powell.

     Do not call this function directly but pass its name "nlopt_bobyqa" to
     estimagic's maximize or minimize function as `algorithm` argument. Specify
     your desired arguments as a dictionary and pass them as `algo_options` to
     minimize or maximize.
     Below, only details of the optional algorithm options are listed. For the mandatory
     arguments see :ref:`internal_optimizer_interface`. For more background on those
     options, see :ref:`naming_conventions`.

     Args:
         convergence_relative_params_tolerance (float): Stop when the relative movement
             between parameter vectors is smaller than this.
         convergence_relative_criterion_tolerance (float): Stop when the relative
             improvement between two iterations is smaller than this.
             More formally, this is expressed as

             .. math::

                 \\frac{(f^k - f^{k+1})}{\\max{{|f^k|, |f^{k+1}|, 1}}} \\leq
                 \\text{relative_criterion_tolerance}

         stopping_max_criterion_evaluations (int): If the maximum number of function
             evaluation is reached, the optimization stops but we do not count this
             as convergence.
         stopping_max_iterations (int): If the maximum number of iterations is reached,
             the optimization stops, but we do not count this as convergence.

     Returns:
         dict: See :ref:`internal_optimizer_output` for details.

     """
    out = _minimize_nlopt(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    algorithm=nlopt.LN_BOBYQA,
    algorithm_name="nlopt_bobyqa",
    convergence_relative_params_tolerance=convergence_relative_params_tolerance,
    convergence_absolute_params_tolerance=convergence_absolute_params_tolerance,
    convergence_relative_criterion_tolerance=convergence_relative_criterion_tolerance,
    convergence_absolute_criterion_tolerance=convergence_absolute_criterion_tolerance,
    stopping_max_criterion_evaluations=stopping_max_criterion_evaluations,
    )

    return out


def nlopt_neldermead (
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_relative_params_tolerance=CONVERGENCE_RELATIVE_PARAMS_TOLERANCE,
    convergence_absolute_params_tolerance=CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE,
    convergence_relative_criterion_tolerance=CONVERGENCE_RELATIVE_CRITERION_TOLERANCE,
    convergence_absolute_criterion_tolerance=CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS,
):
""" Minimize a scalar function using nlopt algorithm based on the
  original Nelder-Mead simplex algorithm, as described in:

 J. A. Nelder and R. Mead, "A simplex method for function minimization,"
 The Computer Journal 7, p. 308-313 (1965)..

 Do not call this function directly but pass its name "nlopt_bobyqa" to
 estimagic's maximize or minimize function as `algorithm` argument. Specify
 your desired arguments as a dictionary and pass them as `algo_options` to
 minimize or maximize.
 Below, only details of the optional algorithm options are listed. For the mandatory
 arguments see :ref:`internal_optimizer_interface`. For more background on those
 options, see :ref:`naming_conventions`.

 Args:
     convergence_relative_params_tolerance (float): Stop when the relative movement
         between parameter vectors is smaller than this.
     convergence_relative_criterion_tolerance (float): Stop when the relative
         improvement between two iterations is smaller than this.
         More formally, this is expressed as

         .. math::

             \\frac{(f^k - f^{k+1})}{\\max{{|f^k|, |f^{k+1}|, 1}}} \\leq
             \\text{relative_criterion_tolerance}

     stopping_max_criterion_evaluations (int): If the maximum number of function
         evaluation is reached, the optimization stops but we do not count this
         as convergence.
     stopping_max_iterations (int): If the maximum number of iterations is reached,
         the optimization stops, but we do not count this as convergence.

 Returns:
     dict: See :ref:`internal_optimizer_output` for details.

 """
    if np.isfinite(lower_bounds).any():
        warnings.warn(f"With finite lower bounds are specified nlopt_neldermead"+
        " may fail if finite uper bounds are not specified")


    out = _minimize_nlopt(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    algorithm=nlopt.LN_NELDERMEAD,
    algorithm_name="nlopt_neldermead",
    convergence_relative_params_tolerance=convergence_relative_params_tolerance,
    convergence_absolute_params_tolerance=convergence_absolute_params_tolerance,
    convergence_relative_criterion_tolerance=convergence_relative_criterion_tolerance,
    convergence_absolute_criterion_tolerance=convergence_absolute_criterion_tolerance,
    stopping_max_criterion_evaluations=stopping_max_criterion_evaluations,
    )

    return out


def _minimize_nlopt(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    algorithm,
    algorithm_name,
    *,
    convergence_relative_params_tolerance=None,
    convergence_absolute_params_tolerance=None,
    convergence_relative_criterion_tolerance=None,
    convergence_absolute_criterion_tolerance=None,
    stopping_max_criterion_evaluations=None,
    stopping_max_iterations=None,
    limited_memory_storage_length=None,
    max_line_search_steps=None,
):
    """ Run actual nlopt optimization argument, set relevant
    attributes. 
    """
    algo_info = DEFAULT_ALGO_INFO.copy()
    algo_info["name"] = algorithm_name
    def func(x, grad):
        if grad.size>0:
            criterion, derivative  = criterion_and_derivative(
                x,
                task='criterion_and_derivative',
                algorithm_info=algo_info,
             )
            grad[:] = derivative
        else:
            criterion = criterion_and_derivative(
                 x,
                 task='criterion',
                 algorithm_info=algo_info,
             )
        return criterion

    opt = nlopt.opt(algorithm, x.shape[0])
    if convergence_relative_criterion_tolerance is not None:
        opt.set_ftol_rel(convergence_relative_criterion_tolerance)
    if convergence_absolute_criterion_tolerance is not None:
        opt.set_ftol_abs(convergence_absolute_criterion_tolerance)
    if convergence_relative_params_tolerance is not None:
        opt.set_xtol_rel(convergence_relative_params_tolerance)
    if convergence_absolute_params_tolerance is not None:
        opt.set_xtol_abs(convergence_absolute_params_tolerance)
    if lower_bounds is not None:
        opt.set_lower_bounds(lower_bounds)
    if upper_bounds is not None:
        opt.set_upper_bounds(upper_bounds)
    if stopping_max_criterion_evaluations is not None:
        opt.set_maxeval(stopping_max_criterion_evaluations)
    opt.set_min_objective(func)
    solution_x = opt.optimize(x)
    return _process_nlopt_results(opt, solution_x)

def _process_nlopt_results(nlopt_obj, solution_x):
    messages = {
    1: "Convergence achieved ",
    2: (
    "Optimizer stopped because maximum value of criterion function was reached"
    ),
    3: (
    "Optimizer stopped because convergence_relative_criterion_tolerance or " +
    "convergence_absolute_criterion_tolerance was reached"
    ),
    4:  (
    "Optimizer stopped because convergence_relative_params_tolerance or " +
    "convergence_absolute_params_tolerance was reached"
    ),
    5: "Optimizer stopped because max_criterion_evaluations was reached",
    6: "Optimizer stopped because max running time was reached",
    -1: "Optimizer failed",
    -2: "Invalid arguments were passed",
    -3: "Memory error",
    -4: "Halted because roundoff errors limited progress",
    -5: "Halted because of user specified forced stop"
    }
    processed = {
        "solution_x":solution_x,
        "solution_criterion": nlopt_obj.last_optimum_value(),
        "solution_derivative": None,
        "solution_hessian": None,
        "n_criterion_evaluations": nlopt_obj.get_numevals(),
        "n_derivative_evaluations": None,
        "n_iterations": None,
        "success": nlopt_obj.last_optimize_result() in [1,2,3,4],
        "message": messages[nlopt_obj.last_optimize_result()],
        "reached_convergence_criterion": None,
    }
    return processed
