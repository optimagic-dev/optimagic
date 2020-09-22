""""Implement algorithms by the [Numerical Algorithms Group](https://www.nag.com/)."""
from functools import partial

from estimagic.config import CRITERION_NOISY
from estimagic.config import INITIAL_TRUST_RADIUS
from estimagic.config import IS_PYBOBYQA_INSTALLED
from estimagic.config import MAX_CRITERION_EVALUATIONS
from estimagic.config import NR_EVALS_PER_POINT
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
    nr_interpolation_points=None,
    criterion_noisy=CRITERION_NOISY,
    nr_evals_per_point=NR_EVALS_PER_POINT,
):
    r"""Minimize a function using the BOBYQA algorithm.

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
        nr_interpolation_points (int): the number of interpolation points to use.
            default is 2n+1 for a problem with len(x)=n if not criterion_noisy,
            otherwise it is set to (n+1)(n+2)/2). Larger values are particularly
            useful for noisy problems. Py-BOBYQA requires

            .. math::
                n + 1 \leq nr_interpolation_points \leq (n+1)(n+2)/2.

        criterion_noisy (bool): Whether the criterion function is noisy, i.e. whether
            it does not always return the same value when evaluated at the same
            parameters.
        nr_evals_per_point (func): How often to evaluate the criterion function at each
            point. The function must take `delta`, `rho`, `iter` and `nrestarts` as
            arguments and return an integer. This is only applicable for
            criterion functions with stochastic noise, when averaging multiple
            evaluations at the same point produces a more accurate value.
            The input parameters are the trust region radius (delta), the lower bound
            on the trust region radius (rho), how many iterations the algorithm has
            been running for (iter), and how many restarts have been performed
            (nrestarts). Default is no averaging (i.e.
            nr_evals_per_point(delta, rho, iter, nrestarts) = 1).

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
        bounds=(lower_bounds, upper_bounds),
        maxfun=max_criterion_evaluations,
        rhobeg=initial_trust_radius,
        user_params=advanced_options,
        scaling_within_bounds=False,
        do_logging=False,
        print_progress=False,
        objfun_has_noise=criterion_noisy,
        nsamples=nr_evals_per_point,
        npt=nr_interpolation_points,
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
