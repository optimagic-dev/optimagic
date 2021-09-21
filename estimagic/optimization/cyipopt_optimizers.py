"""Implement cyipopt's Interior Point Optimizer."""
import functools

from estimagic.config import IS_CYIPOPT_INSTALLED
from estimagic.optimization.algo_options import CONVERGENCE_RELATIVE_CRITERION_TOLERANCE
from estimagic.optimization.algo_options import STOPPING_MAX_ITERATIONS
from estimagic.optimization.scipy_optimizers import get_scipy_bounds
from estimagic.optimization.scipy_optimizers import process_scipy_result

try:
    import cyipopt
except ImportError:
    pass

DEFAULT_ALGO_INFO = {
    "primary_criterion_entry": "value",
    "parallelizes": False,
    "needs_scaling": False,
}


def ipopt(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    # convergence criteria
    convergence_relative_criterion_tolerance=CONVERGENCE_RELATIVE_CRITERION_TOLERANCE,
    dual_inf_tol=1.0,
    constr_viol_tol=0.0001,
    compl_inf_tol=0.0001,
    #
    s_max=100.0,
    mu_strategy="monotone",
    mu_target=0.0,
    # stopping criteria
    stopping_max_iterations=STOPPING_MAX_ITERATIONS,
    stopping_max_wall_time_seconds=1e20,
    stopping_max_cpu_time=1e20,
    # acceptable heuristic
    acceptable_iter=15,
    acceptable_tol=1e-6,
    acceptable_dual_inf_tol=1e-10,
    acceptable_constr_viol_tol=0.01,
    acceptable_compl_inf_tol=0.01,
    acceptable_obj_change_tol=1e20,
    #
    diverging_iterates_tol=1e20,
    nlp_lower_bound_inf=-1e19,
    nlp_upper_bound_inf=1e19,
    fixed_variable_treatment="make_parameter",
    dependency_detector="none",
    dependency_detection_with_rhs="no",
    # bounds
    kappa_d=1e-5,
    bound_relax_factor=1e-8,
    honor_original_bounds="no",
    #
    check_derivatives_for_naninf="no",
):
    """Minimize a scalar function using the Interior Point Optimizer.

    This implementation of the Interior Point Optimizer (:cite:`Waechter2005`,
    :cite:`Waechter2005a`, :cite:`Waechter2005b`, :cite:`Nocedal2009`) relies on
    `cyipopt <https://cyipopt.readthedocs.io/en/latest/index.html>`_, a Python
    wrapper for the `Ipopt optimization package
    <https://coin-or.github.io/Ipopt/index.html>`_.

    There are two levels of termination criteria. If the usual "desired"
    tolerances (see tol, dual_inf_tol etc) are satisfied at an iteration, the
    algorithm immediately terminates with a success message. On the other hand,
    if the algorithm encounters "acceptable_iter" many iterations in a row that
    are considered "acceptable", it will terminate before the desired
    convergence tolerance is met. This is useful in cases where the algorithm
    might not be able to achieve the "desired" level of accuracy.

    - convergence.relative_criterion_tolerance (float): The algorithm terminates
      successfully, if the (scaled) non linear programming error becomes smaller
      than this value.

    - mu_strategy (str): which barrier parameter update strategy is to be used.
      Can be "monotone" or "adaptive". Default is "monotone", i.e. use the
      monotone (Fiacco-McCormick) strategy.
    - mu_target: Desired value of complementarity. Usually, the barrier
      parameter is driven to zero and the termination test for complementarity
      is measured with respect to zero complementarity. However, in some cases
      it might be desired to have Ipopt solve barrier problem for strictly
      positive value of the barrier parameter. In this case, the value of
      "mu_target" specifies the final value of the barrier parameter, and the
      termination tests are then defined with respect to the barrier problem for
      this value of the barrier parameter. The valid range for this real option
      is 0 ≤ mu_target and its default value is 0.

    - s_max (float): Scaling threshold for the NLP error.

    - stopping.max_iterations (int):  If the maximum number of iterations is
      reached, the optimization stops, but we do not count this as successful
      convergence. The difference to ``max_criterion_evaluations`` is that one
      iteration might need several criterion evaluations, for example in a line
      search or to determine if the trust region radius has to be shrunk.
    - stopping.max_wall_time_seconds (float): Maximum number of walltime clock
      seconds.
    - stopping.max_cpu_time (float): Maximum number of CPU seconds. A limit on
      CPU seconds that Ipopt can use to solve one problem. If during the
      convergence check this limit is exceeded, Ipopt will terminate with a
      corresponding message. The valid range for this real option is 0 <
      max_cpu_time and its default value is 1e20.

    - dual_inf_tol (float): Desired threshold for the dual infeasibility.
      Absolute tolerance on the dual infeasibility. Successful termination
      requires that the max-norm of the (unscaled) dual infeasibility is less
      than this threshold. The valid range for this real option is 0 <
      dual_inf_tol and its default value is 1.
    - constr_viol_tol (float): Desired threshold for the constraint and variable
      bound violation. Absolute tolerance on the constraint and variable bound
      violation. Successful termination requires that the max-norm of the
      (unscaled) constraint violation is less than this threshold. If option
      bound_relax_factor is not zero 0, then Ipopt relaxes given variable
      bounds. The value of constr_viol_tol is used to restrict the absolute
      amount of this bound relaxation. The valid range for this real option is 0
      < constr_viol_tol and its default value is 0.0001.
    - compl_inf_tol (float): Desired threshold for the complementarity
      conditions. Absolute tolerance on the complementarity. Successful
      termination requires that the max-norm of the (unscaled) complementarity
      is less than this threshold. The valid range for this real option is 0 <
      compl_inf_tol and its default value is 0.0001.

    - acceptable_iter (int): Number of "acceptable" iterates before triggering
      termination. If the algorithm encounters this many successive "acceptable"
      iterates (see above on the acceptable heuristic), it terminates, assuming
      that the problem has been solved to best possible accuracy given
      round-off. If it is set to zero, this heuristic is disabled. The valid
      range for this integer option is 0 ≤ acceptable_iter.
    - acceptable_tol (float):"Acceptable" convergence tolerance (relative).
      Determines which (scaled) overall optimality error is considered to be
      "acceptable". The valid range for this real option is 0 < acceptable_tol.
    - acceptable_dual_inf_tol (float):  "Acceptance" threshold for the dual
      infeasibility. Absolute tolerance on the dual infeasibility. "Acceptable"
      termination requires that the (max-norm of the unscaled) dual
      infeasibility is less than this threshold; see also acceptable_tol. The
      valid range for this real option is 0 < acceptable_dual_inf_tol and its
      default value is 10+10.
    - acceptable_constr_viol_tol (float): "Acceptance" threshold for the
      constraint violation. Absolute tolerance on the constraint violation.
      "Acceptable" termination requires that the max-norm of the (unscaled)
      constraint violation is less than this threshold; see also acceptable_tol.
      The valid range for this real option is 0 < acceptable_constr_viol_tol and
      its default value is 0.01.
    - acceptable_compl_inf_tol (float): "Acceptance" threshold for the
      complementarity conditions. Absolute tolerance on the complementarity.
      "Acceptable" termination requires that the max-norm of the (unscaled)
      complementarity is less than this threshold; see also acceptable_tol. The
      valid range for this real option is 0 < acceptable_compl_inf_tol and its
      default value is 0.01.
    - acceptable_obj_change_tol (float): "Acceptance" stopping criterion based
      on objective function change. If the relative change of the objective
      function (scaled by Max(1,|f(x)|)) is less than this value, this part of
      the acceptable tolerance termination is satisfied; see also
      acceptable_tol. This is useful for the quasi-Newton option, which has
      trouble to bring down the dual infeasibility. The valid range for this
      real option is 0 ≤ acceptable_obj_change_tol and its default value is
      10+20.

    - diverging_iterates_tol (float): Threshold for maximal value of primal
      iterates. If any component of the primal iterates exceeded this value (in
      absolute terms), the optimization is aborted with the exit message that
      the iterates seem to be diverging. The valid range for this real option is
      0 < diverging_iterates_tol and its default value is 10+20.
    - nlp_lower_bound_inf (float): any bound less or equal this value will be
      considered -inf (i.e. not lwer bounded). The valid range for this real
      option is unrestricted and its default value is -10+19.
    - nlp_upper_bound_inf (float): any bound greater or this value will be
      considered +inf (i.e. not upper bunded). The valid range for this real
      option is unrestricted and its default value is 10+19.
    - fixed_variable_treatment (str): Determines how fixed variables should be
      handled. The main difference between those options is that the starting
      point in the "make_constraint" case still has the fixed variables at their
      given values, whereas in the case "make_parameter(_nodual)" the functions
      are always evaluated with the fixed values for those variables. Also, for
      "relax_bounds", the fixing bound constraints are relaxed (according to"
      bound_relax_factor"). For all but "make_parameter_nodual", bound
      multipliers are computed for the fixed variables. The default value for
      this string option is "make_parameter". Possible values:
      - make_parameter:Remove fixed variable from optimization variables
      - make_parameter_nodual: Remove fixed variable from optimization variables
        and do not compute bound multipliers for fixed variables
      - make_constraint: Add equality constraints fixing variables
      - relax_bounds: Relax fixing bound constraints
    - dependency_detector (str): Indicates which linear solver should be used to
        detect lnearly dependent equality constraints. This is experimental and
        does not work well. The default value for this string option is "none".
        Possible values:
        - none: don't check; no extra work at beginning
        - mumps: use MUMPS
        - wsmp: use WSMP
        - ma28: use MA28
    - dependency_detection_with_rhs (str or bool): Indicates if the right hand
      sides of the constraints should be considered in addition to gradients
      during dependency detection. The default value for this string option is
      "no". Possible values: 'yes', 'no', True, False.

    - kappa_d (float): Weight for linear damping term (to handle one-sided
      bounds). See Section 3.7 in implementation paper. The valid range for this
      real option is 0 ≤ kappa_d and its default value is 10-05.
    - bound_relax_factor (float): Factor for initial relaxation of the bounds.
      Before start of the optimization, the bounds given by the user are
      relaxed. This option sets the factor for this relaxation. Additional, the
      constraint violation tolerance constr_viol_tol is used to bound the
      relaxation by an absolute value. If it is set to zero, then then bounds
      relaxation is disabled. See Eqn.(35) in implementation paper. Note that
      the constraint violation reported by Ipopt at the end of the solution
      process does not include violations of the original (non-relaxed) variable
      bounds. See also option honor_original_bounds. The valid range for this
      real option is 0 ≤ bound_relax_factor and its default value is 10-08.
    - honor_original_bounds (str or bool): Indicates whether final points should
      be projected into original bunds. Ipopt might relax the bounds during the
      optimization (see, e.g., option "bound_relax_factor"). This option
      determines whether the final point should be projected back into the
      user-provide original bounds after the optimization. Note that violations
      of constraints and complementarity reported by Ipopt at the end of the
      solution process are for the non-projected point. The default value for
      this string option is "no". Possible values: 'yes', 'no', True, False

    - check_derivatives_for_naninf (str): whether to check for NaN / inf in the
      derivative matrices. Activating this option will cause an error if an
      invalid number is detected in the constraint Jacobians or the Lagrangian
      Hessian. If this is not activated, the test is skipped, and the algorithm
      might proceed with invalid numbers and fail. If test is activated and an
      invalid number is detected, the matrix is written to output with
      print_level corresponding to J_MORE_DETAILED; so beware of large output!
      The default value for this string option is "no".


    The following options are not supported: - num_linear_variables: since
        estimagic may reparametrize your problem and this changes the parameter
        problem, we do not support num_linear_variables. - mu_oracle

    """
    if not IS_CYIPOPT_INSTALLED:
        raise NotImplementedError(
            "The cyipopt package is not installed and required for 'ipopt'. You can "
            "install the package with: `conda install -c conda-forge cyipopt`"
        )
    if acceptable_tol <= convergence_relative_criterion_tolerance:
        raise ValueError(
            "The acceptable tolerance must be larger than the desired tolerance."
        )
    if mu_strategy not in ["monotone", "adaptive"]:
        raise ValueError(
            f"Unknown barrier strategy: {mu_strategy}. It must be 'monotone' or "
            "'adaptive'."
        )
    if nlp_upper_bound_inf < 0:
        raise ValueError("nlp_upper_bound_inf should be > 0.")
    if nlp_lower_bound_inf > 0:
        raise ValueError("nlp_lower_bound_inf should be < 0.")

    dependency_detection_with_rhs = convert_bool_to_str(
        dependency_detection_with_rhs, "dependency_detection_with_rhs"
    )
    dependency_detector = "none" if dependency_detector is None else dependency_detector
    if dependency_detector not in {"none", "mumps", "wsmp", "ma28"}:
        raise ValueError(
            "dependency_detector must be one of 'none', 'mumps', 'wsmp', 'ma28' or "
            f"None. You specified {dependency_detector}."
        )
    check_derivatives_for_naninf = convert_bool_to_str(
        check_derivatives_for_naninf, "check_derivatives_for_naninf"
    )

    algo_info = DEFAULT_ALGO_INFO.copy()
    algo_info["name"] = "ipopt"

    gradient = functools.partial(
        criterion_and_derivative, task="derivative", algorithm_info=algo_info
    )

    func = functools.partial(
        criterion_and_derivative,
        task="criterion",
        algorithm_info=algo_info,
    )

    options = {
        "s_max": float(s_max),
        "max_iter": stopping_max_iterations,
        "max_wall_time": float(stopping_max_wall_time_seconds),
        "max_cpu_time": stopping_max_cpu_time,
        "dual_inf_tol": dual_inf_tol,
        "constr_viol_tol": constr_viol_tol,
        "compl_inf_tol": compl_inf_tol,
        #
        "acceptable_iter": int(acceptable_iter),
        "acceptable_tol": acceptable_tol,
        "acceptable_dual_inf_tol": acceptable_dual_inf_tol,
        "acceptable_constr_viol_tol": acceptable_constr_viol_tol,
        "acceptable_compl_inf_tol": acceptable_compl_inf_tol,
        "acceptable_obj_change_tol": acceptable_obj_change_tol,
        #
        "diverging_iterates_tol": diverging_iterates_tol,
        "nlp_lower_bound_inf": nlp_lower_bound_inf,
        "nlp_upper_bound_inf": nlp_upper_bound_inf,
        "fixed_variable_treatment": fixed_variable_treatment,
        "dependency_detector": dependency_detector,
        "dependency_detection_with_rhs": dependency_detection_with_rhs,
        "kappa_d": kappa_d,
        "bound_relax_factor": bound_relax_factor,
        "honor_original_bounds": honor_original_bounds,
        #
        "check_derivatives_for_naninf": check_derivatives_for_naninf,
        #
        "mu_strategy": mu_strategy,
        "mu_target": float(mu_target),
        "print_level": 0,  # disable verbosity
    }

    raw_res = cyipopt.minimize_ipopt(
        fun=func,
        x0=x,
        bounds=get_scipy_bounds(lower_bounds, upper_bounds),
        jac=gradient,
        constraints=(),
        tol=convergence_relative_criterion_tolerance,
        options=options,
    )

    res = process_scipy_result(raw_res)
    return res


def convert_bool_to_str(var, name):
    """Convert input to either 'yes' or 'no' and check the output is yes or no.

    Args:
        var (str or bool): user input
        name (str): name of the variable.

    Returns:
        out (str): "yes" or "no".

    """
    if var is True:
        out = "yes"
    elif var is False:
        out = "no"
    else:
        out = var
    if out not in {"yes", "no"}:
        raise ValueError(
            f"{name} must be 'yes', 'no', True or False. You specified {var}."
        )
    return out
