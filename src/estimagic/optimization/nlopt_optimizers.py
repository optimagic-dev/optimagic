"""Implement `nlopt` algorithms.

The documentation is heavily based on (nlopt documentation)[nlopt.readthedocs.io].

"""
import numpy as np
from estimagic.config import IS_NLOPT_INSTALLED
from estimagic.optimization.algo_options import CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE
from estimagic.optimization.algo_options import CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE
from estimagic.optimization.algo_options import CONVERGENCE_RELATIVE_CRITERION_TOLERANCE
from estimagic.optimization.algo_options import CONVERGENCE_RELATIVE_PARAMS_TOLERANCE
from estimagic.optimization.algo_options import STOPPING_MAX_CRITERION_EVALUATIONS
from estimagic.optimization.algo_options import (
    STOPPING_MAX_CRITERION_EVALUATIONS_GLOBAL,
)


if IS_NLOPT_INSTALLED:
    import nlopt


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

    """Minimize a scalar function using the BOBYQA algorithm.

    For details see :ref:`list_of_nlopt_algorithms`.

    """
    out = _minimize_nlopt(
        criterion_and_derivative,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=nlopt.LN_BOBYQA,
        algorithm_name="nlopt_bobyqa",
        convergence_xtol_rel=convergence_relative_params_tolerance,
        convergence_xtol_abs=convergence_absolute_params_tolerance,
        convergence_ftol_rel=convergence_relative_criterion_tolerance,
        convergence_ftol_abs=convergence_absolute_criterion_tolerance,
        stopping_max_eval=stopping_max_criterion_evaluations,
    )

    return out


def nlopt_neldermead(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_relative_params_tolerance=CONVERGENCE_RELATIVE_PARAMS_TOLERANCE,
    convergence_absolute_params_tolerance=CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE,
    convergence_relative_criterion_tolerance=0,
    convergence_absolute_criterion_tolerance=CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS,
):
    """Minimize a scalar function using the Nelder-Mead simplex algorithm.

    For details see :ref:`list_of_nlopt_algorithms`.

    """

    out = _minimize_nlopt(
        criterion_and_derivative,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=nlopt.LN_NELDERMEAD,
        algorithm_name="nlopt_neldermead",
        convergence_xtol_rel=convergence_relative_params_tolerance,
        convergence_xtol_abs=convergence_absolute_params_tolerance,
        convergence_ftol_rel=convergence_relative_criterion_tolerance,
        convergence_ftol_abs=convergence_absolute_criterion_tolerance,
        stopping_max_eval=stopping_max_criterion_evaluations,
    )

    return out


def nlopt_praxis(
    criterion_and_derivative,
    x,
    *,
    convergence_relative_params_tolerance=CONVERGENCE_RELATIVE_PARAMS_TOLERANCE,
    convergence_absolute_params_tolerance=CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE,
    convergence_relative_criterion_tolerance=CONVERGENCE_RELATIVE_CRITERION_TOLERANCE,
    convergence_absolute_criterion_tolerance=CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS,
):
    """Minimize a scalar function using principal-axis method.

    For details see :ref:`list_of_nlopt_algorithms`.

    """
    algo_info = DEFAULT_ALGO_INFO.copy()
    algo_info.update({"needs_scaling": True})
    out = _minimize_nlopt(
        criterion_and_derivative,
        x,
        lower_bounds=None,
        upper_bounds=None,
        algorithm=nlopt.LN_PRAXIS,
        algorithm_name="nlopt_praxis",
        convergence_xtol_rel=convergence_relative_params_tolerance,
        convergence_xtol_abs=convergence_absolute_params_tolerance,
        convergence_ftol_rel=convergence_relative_criterion_tolerance,
        convergence_ftol_abs=convergence_absolute_criterion_tolerance,
        stopping_max_eval=stopping_max_criterion_evaluations,
        algo_info=algo_info,
    )

    return out


def nlopt_cobyla(
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
    """Minimize a scalar function using the cobyla method.

    For details see :ref:`list_of_nlopt_algorithms`.

    """

    out = _minimize_nlopt(
        criterion_and_derivative,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=nlopt.LN_COBYLA,
        algorithm_name="nlopt_cobyla",
        convergence_xtol_rel=convergence_relative_params_tolerance,
        convergence_xtol_abs=convergence_absolute_params_tolerance,
        convergence_ftol_rel=convergence_relative_criterion_tolerance,
        convergence_ftol_abs=convergence_absolute_criterion_tolerance,
        stopping_max_eval=stopping_max_criterion_evaluations,
    )

    return out


def nlopt_sbplx(
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
    """Minimize a scalar function using the "Subplex" algorithm.

    For details see :ref:`list_of_nlopt_algorithms`.

    """

    out = _minimize_nlopt(
        criterion_and_derivative,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=nlopt.LN_SBPLX,
        algorithm_name="nlopt_sbplx",
        convergence_xtol_rel=convergence_relative_params_tolerance,
        convergence_xtol_abs=convergence_absolute_params_tolerance,
        convergence_ftol_rel=convergence_relative_criterion_tolerance,
        convergence_ftol_abs=convergence_absolute_criterion_tolerance,
        stopping_max_eval=stopping_max_criterion_evaluations,
    )

    return out


def nlopt_newuoa(
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
    """Minimize a scalar function using the NEWUOA algorithm.

    For details see :ref:`list_of_nlopt_algorithms`.

    """
    if np.any(np.isfinite(lower_bounds)) or np.any(np.isfinite(upper_bounds)):
        algo = nlopt.LN_NEWUOA_BOUND
    else:
        algo = nlopt.LN_NEWUOA

    out = _minimize_nlopt(
        criterion_and_derivative,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=algo,
        algorithm_name="nlopt_newuoa",
        convergence_xtol_rel=convergence_relative_params_tolerance,
        convergence_xtol_abs=convergence_absolute_params_tolerance,
        convergence_ftol_rel=convergence_relative_criterion_tolerance,
        convergence_ftol_abs=convergence_absolute_criterion_tolerance,
        stopping_max_eval=stopping_max_criterion_evaluations,
    )

    return out


def nlopt_tnewton(
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
    """Minimize a scalar function using the "TNEWTON" algorithm.

    For details see :ref:`list_of_nlopt_algorithms`.

    """

    out = _minimize_nlopt(
        criterion_and_derivative,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=nlopt.LD_TNEWTON,
        algorithm_name="nlopt_tnewton",
        convergence_xtol_rel=convergence_relative_params_tolerance,
        convergence_xtol_abs=convergence_absolute_params_tolerance,
        convergence_ftol_rel=convergence_relative_criterion_tolerance,
        convergence_ftol_abs=convergence_absolute_criterion_tolerance,
        stopping_max_eval=stopping_max_criterion_evaluations,
    )

    return out


def nlopt_lbfgs(
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
    """Minimize a scalar function using the "LBFGS" algorithm.

    For details see :ref:`list_of_nlopt_algorithms`.

    """

    out = _minimize_nlopt(
        criterion_and_derivative,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=nlopt.LD_TNEWTON,
        algorithm_name="nlopt_tnewton",
        convergence_xtol_rel=convergence_relative_params_tolerance,
        convergence_xtol_abs=convergence_absolute_params_tolerance,
        convergence_ftol_rel=convergence_relative_criterion_tolerance,
        convergence_ftol_abs=convergence_absolute_criterion_tolerance,
        stopping_max_eval=stopping_max_criterion_evaluations,
    )

    return out


def nlopt_ccsaq(
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

    """Minimize a scalar function using CCSAQ algorithm.

    For details see :ref:`list_of_nlopt_algorithms`.

    """
    out = _minimize_nlopt(
        criterion_and_derivative,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=nlopt.LD_CCSAQ,
        algorithm_name="nlopt_ccsaq",
        convergence_xtol_rel=convergence_relative_params_tolerance,
        convergence_xtol_abs=convergence_absolute_params_tolerance,
        convergence_ftol_rel=convergence_relative_criterion_tolerance,
        convergence_ftol_abs=convergence_absolute_criterion_tolerance,
        stopping_max_eval=stopping_max_criterion_evaluations,
    )

    return out


def nlopt_mma(
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

    """Minimize a scalar function using the method of moving asymptotes (MMA).

    For details see :ref:`list_of_nlopt_algorithms`.

    """
    out = _minimize_nlopt(
        criterion_and_derivative,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=nlopt.LD_MMA,
        algorithm_name="nlopt_mma",
        convergence_xtol_rel=convergence_relative_params_tolerance,
        convergence_xtol_abs=convergence_absolute_params_tolerance,
        convergence_ftol_rel=convergence_relative_criterion_tolerance,
        convergence_ftol_abs=convergence_absolute_criterion_tolerance,
        stopping_max_eval=stopping_max_criterion_evaluations,
    )

    return out


def nlopt_var(
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
    rank_1_update=True,
):

    """Minimize a scalar function limited memory switching variable-metric method.

    For details see :ref:`list_of_nlopt_algorithms`.

    """
    if rank_1_update:
        algo = nlopt.LD_VAR1
    else:
        algo = nlopt.LD_VAR2
    out = _minimize_nlopt(
        criterion_and_derivative,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=algo,
        algorithm_name="nlopt_var",
        convergence_xtol_rel=convergence_relative_params_tolerance,
        convergence_xtol_abs=convergence_absolute_params_tolerance,
        convergence_ftol_rel=convergence_relative_criterion_tolerance,
        convergence_ftol_abs=convergence_absolute_criterion_tolerance,
        stopping_max_eval=stopping_max_criterion_evaluations,
    )

    return out


def nlopt_slsqp(
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
    """Optimize a scalar function based on SLSQP method.

    For details see :ref:`list_of_nlopt_algorithms`.

    """
    out = _minimize_nlopt(
        criterion_and_derivative,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=nlopt.LD_SLSQP,
        algorithm_name="nlopt_slsqp",
        convergence_xtol_rel=convergence_relative_params_tolerance,
        convergence_xtol_abs=convergence_absolute_params_tolerance,
        convergence_ftol_rel=convergence_relative_criterion_tolerance,
        convergence_ftol_abs=convergence_absolute_criterion_tolerance,
        stopping_max_eval=stopping_max_criterion_evaluations,
    )
    return out


def nlopt_direct(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_relative_params_tolerance=CONVERGENCE_RELATIVE_PARAMS_TOLERANCE,
    convergence_absolute_params_tolerance=CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE,
    convergence_relative_criterion_tolerance=CONVERGENCE_RELATIVE_CRITERION_TOLERANCE,
    convergence_absolute_criterion_tolerance=CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS_GLOBAL,
    locally_biased=False,
    random_search=False,
    unscaled_bounds=False,
):
    """Optimize a scalar function based on DIRECT method.

    For details see :ref:`list_of_nlopt_algorithms`.

    """
    if not locally_biased and not random_search and not unscaled_bounds:
        algo = nlopt.GN_DIRECT
    elif locally_biased and not random_search and not unscaled_bounds:
        algo = nlopt.GN_DIRECT_L
    elif locally_biased and not random_search and unscaled_bounds:
        algo = nlopt.GN_DIRECT_L_NOSCAL
    elif locally_biased and random_search and not unscaled_bounds:
        algo = nlopt.GN_DIRECT_L_RAND
    elif locally_biased and random_search and unscaled_bounds:
        algo = nlopt.GN_DIRECT_L_RAND_NOSCAL
    elif not locally_biased and not random_search and unscaled_bounds:
        algo = nlopt.GN_DIRECT_NOSCAL
    out = _minimize_nlopt(
        criterion_and_derivative,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=algo,
        algorithm_name="nlopt_direct",
        convergence_xtol_rel=convergence_relative_params_tolerance,
        convergence_xtol_abs=convergence_absolute_params_tolerance,
        convergence_ftol_rel=convergence_relative_criterion_tolerance,
        convergence_ftol_abs=convergence_absolute_criterion_tolerance,
        stopping_max_eval=stopping_max_criterion_evaluations,
    )

    # this is a global optimizer
    out["success"] = None
    return out


def nlopt_esch(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_relative_params_tolerance=CONVERGENCE_RELATIVE_PARAMS_TOLERANCE,
    convergence_absolute_params_tolerance=CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE,
    convergence_relative_criterion_tolerance=CONVERGENCE_RELATIVE_CRITERION_TOLERANCE,
    convergence_absolute_criterion_tolerance=CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS_GLOBAL,
):
    """Optimize a scalar function using the ESCH algorithm.

    For details see :ref:`list_of_nlopt_algorithms`.

    """
    out = _minimize_nlopt(
        criterion_and_derivative,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=nlopt.GN_ESCH,
        algorithm_name="nlopt_esch",
        convergence_xtol_rel=convergence_relative_params_tolerance,
        convergence_xtol_abs=convergence_absolute_params_tolerance,
        convergence_ftol_rel=convergence_relative_criterion_tolerance,
        convergence_ftol_abs=convergence_absolute_criterion_tolerance,
        stopping_max_eval=stopping_max_criterion_evaluations,
    )

    # this is a global optimizer
    out["success"] = None
    return out


def nlopt_isres(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_relative_params_tolerance=CONVERGENCE_RELATIVE_PARAMS_TOLERANCE,
    convergence_absolute_params_tolerance=CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE,
    convergence_relative_criterion_tolerance=CONVERGENCE_RELATIVE_CRITERION_TOLERANCE,
    convergence_absolute_criterion_tolerance=CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS_GLOBAL,
):
    """Optimize a scalar function using the ISRES algorithm.

    For details see :ref:`list_of_nlopt_algorithms`.

    """
    out = _minimize_nlopt(
        criterion_and_derivative,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=nlopt.GN_ISRES,
        algorithm_name="nlopt_isres",
        convergence_xtol_rel=convergence_relative_params_tolerance,
        convergence_xtol_abs=convergence_absolute_params_tolerance,
        convergence_ftol_rel=convergence_relative_criterion_tolerance,
        convergence_ftol_abs=convergence_absolute_criterion_tolerance,
        stopping_max_eval=stopping_max_criterion_evaluations,
    )

    # this is a global optimizer
    out["success"] = None
    return out


def nlopt_crs2_lm(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_relative_params_tolerance=CONVERGENCE_RELATIVE_PARAMS_TOLERANCE,
    convergence_absolute_params_tolerance=CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE,
    convergence_relative_criterion_tolerance=CONVERGENCE_RELATIVE_CRITERION_TOLERANCE,
    convergence_absolute_criterion_tolerance=CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS_GLOBAL,
    population_size=None,
):
    """Optimize a scalar function using the CRS2_LM algorithm.

    For details see :ref:`list_of_nlopt_algorithms`.

    """
    if population_size is None:
        population_size = 10 * (len(x) + 1)
    out = _minimize_nlopt(
        criterion_and_derivative,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=nlopt.GN_CRS2_LM,
        algorithm_name="nlopt_crs2_lm",
        convergence_xtol_rel=convergence_relative_params_tolerance,
        convergence_xtol_abs=convergence_absolute_params_tolerance,
        convergence_ftol_rel=convergence_relative_criterion_tolerance,
        convergence_ftol_abs=convergence_absolute_criterion_tolerance,
        stopping_max_eval=stopping_max_criterion_evaluations,
        population_size=population_size,
    )

    # this is a global optimizer
    out["success"] = None
    return out


def _minimize_nlopt(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    algorithm,
    algorithm_name,
    *,
    convergence_xtol_rel=None,
    convergence_xtol_abs=None,
    convergence_ftol_rel=None,
    convergence_ftol_abs=None,
    stopping_max_eval=None,
    population_size=None,
    algo_info=None,
):
    """Run actual nlopt optimization argument, set relevant attributes."""
    if algo_info is None:
        algo_info = DEFAULT_ALGO_INFO.copy()
    else:
        algo_info = algo_info.copy()
    algo_info["name"] = algorithm_name

    def func(x, grad):
        if grad.size > 0:
            criterion, derivative = criterion_and_derivative(
                x,
                task="criterion_and_derivative",
                algorithm_info=algo_info,
            )
            grad[:] = derivative
        else:
            criterion = criterion_and_derivative(
                x,
                task="criterion",
                algorithm_info=algo_info,
            )
        return criterion

    opt = nlopt.opt(algorithm, x.shape[0])
    if convergence_ftol_rel is not None:
        opt.set_ftol_rel(convergence_ftol_rel)
    if convergence_ftol_abs is not None:
        opt.set_ftol_abs(convergence_ftol_abs)
    if convergence_xtol_rel is not None:
        opt.set_xtol_rel(convergence_xtol_rel)
    if convergence_xtol_abs is not None:
        opt.set_xtol_abs(convergence_xtol_abs)
    if lower_bounds is not None:
        opt.set_lower_bounds(lower_bounds)
    if upper_bounds is not None:
        opt.set_upper_bounds(upper_bounds)
    if stopping_max_eval is not None:
        opt.set_maxeval(stopping_max_eval)
    if population_size is not None:
        opt.set_population(population_size)
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
            "Optimizer stopped because convergence_relative_criterion_tolerance or "
            + "convergence_absolute_criterion_tolerance was reached"
        ),
        4: (
            "Optimizer stopped because convergence_relative_params_tolerance or "
            + "convergence_absolute_params_tolerance was reached"
        ),
        5: "Optimizer stopped because max_criterion_evaluations was reached",
        6: "Optimizer stopped because max running time was reached",
        -1: "Optimizer failed",
        -2: "Invalid arguments were passed",
        -3: "Memory error",
        -4: "Halted because roundoff errors limited progress",
        -5: "Halted because of user specified forced stop",
    }
    processed = {
        "solution_x": solution_x,
        "solution_criterion": nlopt_obj.last_optimum_value(),
        "solution_derivative": None,
        "solution_hessian": None,
        "n_criterion_evaluations": nlopt_obj.get_numevals(),
        "n_derivative_evaluations": None,
        "n_iterations": None,
        "success": nlopt_obj.last_optimize_result() in [1, 2, 3, 4],
        "message": messages[nlopt_obj.last_optimize_result()],
        "reached_convergence_criterion": None,
    }
    return processed
