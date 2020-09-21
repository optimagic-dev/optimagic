""""Implement algorithms by the [Numerical Algorithms Group](https://www.nag.com/)."""
from functools import partial

from estimagic.config import INITIAL_TRUST_RADIUS
from estimagic.config import IS_PYBOBYQA_INSTALLED
from estimagic.config import MAX_CRITERION_EVALUATIONS
from estimagic.config import RANDOM_DIRECTIONS_ORTHOGONAL
from estimagic.config import RANDOM_INITIAL_DIRECTIONS

try:
    import pybobyqa
except ImportError:
    pass


def nag_pybobyqa(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    max_criterion_evaluations=MAX_CRITERION_EVALUATIONS,
    initial_trust_radius=INITIAL_TRUST_RADIUS,
    random_initial_directions=RANDOM_INITIAL_DIRECTIONS,
    random_directions_orthogonal=RANDOM_DIRECTIONS_ORTHOGONAL,
):
    """Minimize a function using the BOBYQA algorithm.

    BOBYQA is a derivative-free trust-region method.
    It is designed to solve nonlinear local minimization problems.

    There are two main situations when using a derivative-free algorithm like BOBYQA
    is preferable to derivative-based algorithms:

    1. The criterion function is not deterministic, i.e. if we evaluate the criterion
        function multiple times at the same parameter vector we get different results.

    2. The criterion function is very expensive to evaluate and only finite differences
        are available to calculate its derivative.

    For a detailed documentation of the algorithm go to
    https://numericalalgorithmsgroup.github.io/pybobyqa/build/html/index.html.

    Args:
        max_criterion_evaluations (int): If the maximum number of function evaluation is
            reached, the optimization stops but we do not count this as convergence.
        initial_trust_radius (float): the initial value of the trust region radius.
        random_initial_directions (bool): If True, the initial directions are drawn
            randomly (as opposed to coordinate directions).
        random_directions_orthogonal (bool): If True and random initial directions are
            drawn, the drawn directions are made orthogonal.

    Returns:
        dict: See :ref:`internal_optimizer_output` for details.

    """
    if not IS_PYBOBYQA_INSTALLED:
        raise NotImplementedError(
            "The pybobyqa package is not installed and required for 'nag_pybobyqa'. "
            "You can install it with 'pip install Py-BOBYQA'."
        )

    algo_info = {
        "name": "nag_pybobyqa",
        "primary_criterion_entry": "value",
        "parallelizes": False,
        "needs_scaling": False,
    }
    criterion = partial(
        criterion_and_derivative, task="criterion", algorithm_info=algo_info
    )

    advanced_options = {
        "init.random_initial_directions": random_initial_directions,
        "init.random_directions_make_orthogonal": random_directions_orthogonal,
    }

    res = pybobyqa.solve(
        criterion,
        x0=x,
        # always pass the bounds as bobyqa would set +/- 1e20 as defaults.
        bounds=(lower_bounds, upper_bounds),
        maxfun=max_criterion_evaluations,
        scaling_within_bounds=False,
        do_logging=False,
        print_progress=False,
        rhobeg=initial_trust_radius,
        user_params=advanced_options,
    )

    return _process_nag_result(res)


def _process_nag_result(nag_result_obj):
    processed = {
        "solution_x": nag_result_obj.x,
        "solution_criterion": nag_result_obj.f,
        "solution_derivative": nag_result_obj.gradient,
        "solution_hessian": nag_result_obj.hessian,
        "n_iterations": nag_result_obj.nf,
        "n_criterion_evaluations": nag_result_obj.nx,
        "message": nag_result_obj.msg,
        "success": nag_result_obj.flag == nag_result_obj.EXIT_SUCCESS,
    }
    return processed
