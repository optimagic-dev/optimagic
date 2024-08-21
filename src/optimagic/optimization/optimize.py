"""Public functions for optimization.

This module defines the public functions `maximize` and `minimize` that will be called
by users.

Internally, `maximize` and `minimize` just call `create_optimization_problem` with
all arguments and add the `direction`. In `create_optimization_problem`, the user input
is consolidated and converted to stricter types.  The resulting `OptimizationProblem`
is then passed to `_optimize` which handles the optimization logic.

`_optimize` processes the optimization problem and performs the actual optimization.

"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence, Type, cast

from scipy.optimize import Bounds as ScipyBounds

from optimagic.batch_evaluators import process_batch_evaluator
from optimagic.constraints import Constraint
from optimagic.differentiation.numdiff_options import NumdiffOptions, NumdiffOptionsDict
from optimagic.exceptions import (
    InvalidFunctionError,
)
from optimagic.logging.logger import LogReader, LogStore
from optimagic.logging.types import ProblemInitialization
from optimagic.optimization.algorithm import Algorithm
from optimagic.optimization.create_optimization_problem import (
    OptimizationProblem,
    create_optimization_problem,
)
from optimagic.optimization.error_penalty import get_error_penalty_function
from optimagic.optimization.fun_value import FunctionValue
from optimagic.optimization.internal_optimization_problem import (
    InternalBounds,
    InternalOptimizationProblem,
)
from optimagic.optimization.multistart import (
    run_multistart_optimization,
)
from optimagic.optimization.multistart_options import (
    MultistartOptions,
    MultistartOptionsDict,
    get_internal_multistart_options_from_public,
)
from optimagic.optimization.optimization_logging import log_scheduled_steps_and_get_ids
from optimagic.optimization.optimize_result import OptimizeResult
from optimagic.optimization.process_results import (
    ExtraResultFields,
    process_multistart_result,
    process_single_result,
)
from optimagic.parameters.bounds import Bounds
from optimagic.parameters.conversion import (
    get_converter,
)
from optimagic.parameters.nonlinear_constraints import process_nonlinear_constraints
from optimagic.parameters.scaling import ScalingOptions, ScalingOptionsDict
from optimagic.typing import (
    AggregationLevel,
    Direction,
    ErrorHandling,
    ErrorHandlingLiteral,
    NonNegativeFloat,
    PyTree,
)

FunType = Callable[..., float | PyTree | FunctionValue]
AlgorithmType = str | Algorithm | Type[Algorithm]
ConstraintsType = Constraint | list[Constraint] | dict[str, Any] | list[dict[str, Any]]
JacType = Callable[..., PyTree]
FunAndJacType = Callable[..., tuple[float | PyTree | FunctionValue, PyTree]]
HessType = Callable[..., PyTree]
# TODO: refine this type
CallbackType = Callable[..., Any]

CriterionType = Callable[..., float | dict[str, Any]]
CriterionAndDerivativeType = Callable[..., tuple[float | dict[str, Any], PyTree]]


from optimagic.logging.logger import LogOptions


def maximize(
    fun: FunType | CriterionType | None = None,
    params: PyTree | None = None,
    algorithm: AlgorithmType | None = None,
    *,
    bounds: Bounds | ScipyBounds | Sequence[tuple[float, float]] | None = None,
    constraints: ConstraintsType | None = None,
    fun_kwargs: dict[str, Any] | None = None,
    algo_options: dict[str, Any] | None = None,
    jac: JacType | list[JacType] | None = None,
    jac_kwargs: dict[str, Any] | None = None,
    fun_and_jac: FunAndJacType | CriterionAndDerivativeType | None = None,
    fun_and_jac_kwargs: dict[str, Any] | None = None,
    numdiff_options: NumdiffOptions | NumdiffOptionsDict | None = None,
    # TODO: add typed-dict support?
    logging: bool | str | Path | LogOptions | dict[str, Any] | None = None,
    error_handling: ErrorHandling | ErrorHandlingLiteral = ErrorHandling.RAISE,
    error_penalty: dict[str, float] | None = None,
    scaling: bool | ScalingOptions | ScalingOptionsDict = False,
    multistart: bool | MultistartOptions | MultistartOptionsDict = False,
    collect_history: bool = True,
    skip_checks: bool = False,
    # scipy aliases
    x0: PyTree | None = None,
    method: str | None = None,
    args: tuple[Any] | None = None,
    # scipy arguments that are not yet supported
    hess: HessType | None = None,
    hessp: HessType | None = None,
    callback: CallbackType | None = None,
    # scipy arguments that will never be supported
    options: dict[str, Any] | None = None,
    tol: NonNegativeFloat | None = None,
    # deprecated arguments
    criterion: CriterionType | None = None,
    criterion_kwargs: dict[str, Any] | None = None,
    derivative: JacType | None = None,
    derivative_kwargs: dict[str, Any] | None = None,
    criterion_and_derivative: CriterionAndDerivativeType | None = None,
    criterion_and_derivative_kwargs: dict[str, Any] | None = None,
    log_options: dict[str, Any] | None = None,
    lower_bounds: PyTree | None = None,
    upper_bounds: PyTree | None = None,
    soft_lower_bounds: PyTree | None = None,
    soft_upper_bounds: PyTree | None = None,
    scaling_options: dict[str, Any] | None = None,
    multistart_options: dict[str, Any] | None = None,
) -> OptimizeResult:
    """Maximize fun using algorithm subject to constraints.

    Args:
        fun: The objective function of a scalar, least-squares or likelihood
            optimization problem. Non-scalar objective functions have to be marked
            with the `mark.likelihood` or `mark.least_squares` decorators. `fun` maps
            params and fun_kwargs to an objective value. See :ref:`how-to-fun` for
            details and examples.
        params: The start parameters for the optimization. Params can be numpy arrays,
            dictionaries, pandas.Series, pandas.DataFrames, NamedTuples, floats, lists,
            and any nested combination thereof. See :ref:`params` for details and
            examples.
        algorithm: The optimization algorithm to use. Can be a string, subclass of
            :class:`optimagic.Algorithm` or an instance of a subclass of
            :class:`optimagic.Algorithm`. For guidelines on how to choose an algorithm
            see :ref:`how-to-select-algorithms`. For examples of specifying and
            configuring algorithms see :ref:`specify-algorithm`.
        bounds: Lower and upper bounds on the parameters. The most general and preferred
            way to specify bounds is an :class:`optimagic.Bounds` object that collects
            lower, upper, soft_lower and soft_upper bounds. The soft bounds are used for
            sampling based optimizers but are not enforced during optimization. Each
            bound type mirrors the structure of params. See :ref:`how-to-bounds` for
            details and examples. If params is a flat numpy array, you can also provide
            bounds via any format that is supported by scipy.optimize.minimize.
        constraints: Constraints for the optimization problem. Constraints can be
            specified as a single :class:`optimagic.Constraint` object, a list of
            Constraint objects. For details and examples check :ref:`constraints`.
        fun_kwargs: Additional keyword arguments for the objective function.
        algo_options: Additional options for the optimization algorithm. `algo_options`
            is an alternative to configuring algorithm objects directly. See
            :ref:`list_of_algorithms` for supported options of each algorithm.
        jac: The first derivative of `fun`. Providing a closed form derivative can be
            a great way to speed up your optimization. The easiest way to get
            a derivative for your objective function are autodiff frameworks like
            JAX. For details and examples see :ref:`how-to-jac`.
        jac_kwargs: Additional keyword arguments for `jac`.
        fun_and_jac: A function that returns both the objective value and the
            derivative. This can be used do exploit synergies in the calculation of the
            function value and its derivative. For details and examples see
            :ref:`how-to-jac`.
        fun_and_jac_kwargs: Additional keyword arguments for `fun_and_jac`.
        numdiff_options: Options for numerical differentiation. Can be a dictionary
            or an instance of :class:`optimagic.NumdiffOptions`.
        logging: If None, no logging is used. If a str or pathlib.Path is provided,
            it is interpreted as path to an sqlite3 file (which typically has
            the file extension ``.db``. If the file does not exist, it will be created.
            and the optimization history will be stored in that database. For more
            customization, provide LogOptions. For details and examples see
            :ref:`how-to-logging`.
        error_handling: If "raise" or ErrorHandling.RAISE, exceptions that occur during
            the optimization are raised and the optimization is stopped. If "continue"
            or ErrorHandling.CONTINUE, exceptions are caught and the function value and
            its derivative are replaced by penalty values. The penalty values are
            constructed such that the optimizer is guided back towards the start
            parameters until a feasible region is reached and then continues the
            optimization from there. For details see  :ref:`how-to-errors`.
        error_penalty: A dictionary with the keys "slope" and "constant" that
            influences the magnitude of the penalty values. For maximization problems
            both should be negative. For details see :ref:`how-to-errors`.
        scaling: If None or False, the parameter space is not rescaled. If True,
            a heuristic is used to improve the conditioning of the optimization problem.
            To choose which heuristic is used and to customize the scaling, provide
            a dictionary or an instance of :class:`optimagic.ScalingOptions`.
            For details and examples see :ref:`scaling`.
        multistart: If None or False, no multistart approach is used. If True, the
            optimization is restarted from multiple starting points. Note that this
            requires finite bounds or soft bounds for all parameters. To customize the
            multistart approach, provide a dictionary or an instance of
            :class:`optimagic.MultistartOptions`. For details and examples see
            :ref:`how-to-multistart`.
        collect_history: If True, the optimization history is collected and returned
            in the OptimizeResult. This is required to create `criterion_plot` or
            `params_plot` from an OptimizeResult.
        skip_checks: If True, some checks are skipped to speed up the optimization.
            This is only relevant if your objective function is very fast, i.e. runs in
            a few microseconds.
        x0: Alias for params for scipy compatibility.
        method: Alternative to algorithm for scipy compatibility. With `method` you can
            select scipy optimizers via their original scipy name.
        args: Alternative to fun_kwargs for scipy compatibility.
        hess: Not yet supported.
        hessp: Not yet supported.
        callback: Not yet supported.
        options: Not yet supported.
        tol: Not yet supported.
        criterion: Deprecated. Use fun instead.
        criterion_kwargs: Deprecated. Use fun_kwargs instead.
        derivative: Deprecated. Use jac instead.
        derivative_kwargs: Deprecated. Use jac_kwargs instead.
        criterion_and_derivative: Deprecated. Use fun_and_jac instead.
        criterion_and_derivative_kwargs: Deprecated. Use fun_and_jac_kwargs instead.
        lower_bounds: Deprecated. Use bounds instead.
        upper_bounds: Deprecated. Use bounds instead.
        soft_lower_bounds: Deprecated. Use bounds instead.
        soft_upper_bounds: Deprecated. Use bounds instead.
        scaling_options: Deprecated. Use scaling instead.
        multistart_options: Deprecated. Use multistart instead.

    """
    problem = create_optimization_problem(
        direction=Direction.MAXIMIZE,
        fun=fun,
        params=params,
        bounds=bounds,
        algorithm=algorithm,
        fun_kwargs=fun_kwargs,
        constraints=constraints,
        algo_options=algo_options,
        jac=jac,
        jac_kwargs=jac_kwargs,
        fun_and_jac=fun_and_jac,
        fun_and_jac_kwargs=fun_and_jac_kwargs,
        numdiff_options=numdiff_options,
        logging=logging,
        log_options=log_options,
        error_handling=error_handling,
        error_penalty=error_penalty,
        scaling=scaling,
        multistart=multistart,
        collect_history=collect_history,
        skip_checks=skip_checks,
        # scipy aliases
        x0=x0,
        method=method,
        args=args,
        # scipy arguments that are not yet supported
        hess=hess,
        hessp=hessp,
        callback=callback,
        # scipy arguments that will never be supported
        options=options,
        tol=tol,
        # deprecated arguments
        criterion=criterion,
        criterion_kwargs=criterion_kwargs,
        derivative=derivative,
        derivative_kwargs=derivative_kwargs,
        criterion_and_derivative=criterion_and_derivative,
        criterion_and_derivative_kwargs=criterion_and_derivative_kwargs,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        soft_lower_bounds=soft_lower_bounds,
        soft_upper_bounds=soft_upper_bounds,
        scaling_options=scaling_options,
        multistart_options=multistart_options,
    )
    return _optimize(problem)


def minimize(
    fun: FunType | CriterionType | None = None,
    params: PyTree | None = None,
    algorithm: AlgorithmType | None = None,
    *,
    bounds: Bounds | ScipyBounds | Sequence[tuple[float, float]] | None = None,
    constraints: ConstraintsType | None = None,
    fun_kwargs: dict[str, Any] | None = None,
    algo_options: dict[str, Any] | None = None,
    jac: JacType | list[JacType] | None = None,
    jac_kwargs: dict[str, Any] | None = None,
    fun_and_jac: FunAndJacType | CriterionAndDerivativeType | None = None,
    fun_and_jac_kwargs: dict[str, Any] | None = None,
    numdiff_options: NumdiffOptions | NumdiffOptionsDict | None = None,
    # TODO: add typed-dict support?
    logging: bool | str | Path | LogOptions | dict[str, Any] | None = None,
    error_handling: ErrorHandling | ErrorHandlingLiteral = ErrorHandling.RAISE,
    error_penalty: dict[str, float] | None = None,
    scaling: bool | ScalingOptions | ScalingOptionsDict = False,
    multistart: bool | MultistartOptions | MultistartOptionsDict = False,
    collect_history: bool = True,
    skip_checks: bool = False,
    # scipy aliases
    x0: PyTree | None = None,
    method: str | None = None,
    args: tuple[Any] | None = None,
    # scipy arguments that are not yet supported
    hess: HessType | None = None,
    hessp: HessType | None = None,
    callback: CallbackType | None = None,
    # scipy arguments that will never be supported
    options: dict[str, Any] | None = None,
    tol: NonNegativeFloat | None = None,
    # deprecated arguments
    criterion: CriterionType | None = None,
    criterion_kwargs: dict[str, Any] | None = None,
    derivative: JacType | None = None,
    derivative_kwargs: dict[str, Any] | None = None,
    criterion_and_derivative: CriterionAndDerivativeType | None = None,
    criterion_and_derivative_kwargs: dict[str, Any] | None = None,
    log_options: dict[str, Any] | None = None,
    lower_bounds: PyTree | None = None,
    upper_bounds: PyTree | None = None,
    soft_lower_bounds: PyTree | None = None,
    soft_upper_bounds: PyTree | None = None,
    scaling_options: dict[str, Any] | None = None,
    multistart_options: dict[str, Any] | None = None,
) -> OptimizeResult:
    """Minimize criterion using algorithm subject to constraints.

    Args:
        fun: The objective function of a scalar, least-squares or likelihood
            optimization problem. Non-scalar objective functions have to be marked
            with the `mark.likelihood` or `mark.least_squares` decorators. `fun` maps
            params and fun_kwargs to an objective value. See :ref:`how-to-fun` for
            details and examples.
        params: The start parameters for the optimization. Params can be numpy arrays,
            dictionaries, pandas.Series, pandas.DataFrames, NamedTuples, floats, lists,
            and any nested combination thereof. See :ref:`params` for details and
            examples.
        algorithm: The optimization algorithm to use. Can be a string, subclass of
            :class:`optimagic.Algorithm` or an instance of a subclass of
            :class:`optimagic.Algorithm`. For guidelines on how to choose an algorithm
            see :ref:`how-to-select-algorithms`. For examples of specifying and
            configuring algorithms see :ref:`specify-algorithm`.
        bounds: Lower and upper bounds on the parameters. The most general and preferred
            way to specify bounds is an :class:`optimagic.Bounds` object that collects
            lower, upper, soft_lower and soft_upper bounds. The soft bounds are used for
            sampling based optimizers but are not enforced during optimization. Each
            bound type mirrors the structure of params. See :ref:`how-to-bounds` for
            details and examples. If params is a flat numpy array, you can also provide
            bounds via any format that is supported by scipy.optimize.minimize.
        constraints: Constraints for the optimization problem. Constraints can be
            specified as a single :class:`optimagic.Constraint` object, a list of
            Constraint objects. For details and examples check :ref:`constraints`.
        fun_kwargs: Additional keyword arguments for the objective function.
        algo_options: Additional options for the optimization algorithm. `algo_options`
            is an alternative to configuring algorithm objects directly. See
            :ref:`list_of_algorithms` for supported options of each algorithm.
        jac: The first derivative of `fun`. Providing a closed form derivative can be
            a great way to speed up your optimization. The easiest way to get
            a derivative for your objective function are autodiff frameworks like
            JAX. For details and examples see :ref:`how-to-jac`.
        jac_kwargs: Additional keyword arguments for `jac`.
        fun_and_jac: A function that returns both the objective value and the
            derivative. This can be used do exploit synergies in the calculation of the
            function value and its derivative. For details and examples see
            :ref:`how-to-jac`.
        fun_and_jac_kwargs: Additional keyword arguments for `fun_and_jac`.
        numdiff_options: Options for numerical differentiation. Can be a dictionary
            or an instance of :class:`optimagic.NumdiffOptions`.
        logging: If None, no logging is used. If a str or pathlib.Path is provided,
            it is interpreted as path to an sqlite3 file (which typically has
            the file extension ``.db``. If the file does not exist, it will be created.
            and the optimization history will be stored in that database. For more
            customization, provide LogOptions. For details and examples see
            :ref:`how-to-logging`.
        error_handling: If "raise" or ErrorHandling.RAISE, exceptions that occur during
            the optimization are raised and the optimization is stopped. If "continue"
            or ErrorHandling.CONTINUE, exceptions are caught and the function value and
            its derivative are replaced by penalty values. The penalty values are
            constructed such that the optimizer is guided back towards the start
            parameters until a feasible region is reached and then continues the
            optimization from there. For details see  :ref:`how-to-errors`.
        error_penalty: A dictionary with the keys "slope" and "constant" that
            influences the magnitude of the penalty values. For minimization problems
            both should be positive. For details see :ref:`how-to-errors`.
        scaling: If None or False, the parameter space is not rescaled. If True,
            a heuristic is used to improve the conditioning of the optimization problem.
            To choose which heuristic is used and to customize the scaling, provide
            a dictionary or an instance of :class:`optimagic.ScalingOptions`.
            For details and examples see :ref:`scaling`.
        multistart: If None or False, no multistart approach is used. If True, the
            optimization is restarted from multiple starting points. Note that this
            requires finite bounds or soft bounds for all parameters. To customize the
            multistart approach, provide a dictionary or an instance of
            :class:`optimagic.MultistartOptions`. For details and examples see
            :ref:`how-to-multistart`.
        collect_history: If True, the optimization history is collected and returned
            in the OptimizeResult. This is required to create `criterion_plot` or
            `params_plot` from an OptimizeResult.
        skip_checks: If True, some checks are skipped to speed up the optimization.
            This is only relevant if your objective function is very fast, i.e. runs in
            a few microseconds.
        x0: Alias for params for scipy compatibility.
        method: Alternative to algorithm for scipy compatibility. With `method` you can
            select scipy optimizers via their original scipy name.
        args: Alternative to fun_kwargs for scipy compatibility.
        hess: Not yet supported.
        hessp: Not yet supported.
        callback: Not yet supported.
        options: Not yet supported.
        tol: Not yet supported.
        criterion: Deprecated. Use fun instead.
        criterion_kwargs: Deprecated. Use fun_kwargs instead.
        derivative: Deprecated. Use jac instead.
        derivative_kwargs: Deprecated. Use jac_kwargs instead.
        criterion_and_derivative: Deprecated. Use fun_and_jac instead.
        criterion_and_derivative_kwargs: Deprecated. Use fun_and_jac_kwargs instead.
        lower_bounds: Deprecated. Use bounds instead.
        upper_bounds: Deprecated. Use bounds instead.
        soft_lower_bounds: Deprecated. Use bounds instead.
        soft_upper_bounds: Deprecated. Use bounds instead.
        scaling_options: Deprecated. Use scaling instead.
        multistart_options: Deprecated. Use multistart instead.

    """

    problem = create_optimization_problem(
        direction=Direction.MINIMIZE,
        fun=fun,
        params=params,
        algorithm=algorithm,
        bounds=bounds,
        fun_kwargs=fun_kwargs,
        constraints=constraints,
        algo_options=algo_options,
        jac=jac,
        jac_kwargs=jac_kwargs,
        fun_and_jac=fun_and_jac,
        fun_and_jac_kwargs=fun_and_jac_kwargs,
        numdiff_options=numdiff_options,
        logging=logging,
        error_handling=error_handling,
        error_penalty=error_penalty,
        scaling=scaling,
        multistart=multistart,
        collect_history=collect_history,
        skip_checks=skip_checks,
        # scipy aliases
        x0=x0,
        method=method,
        args=args,
        # scipy arguments that are not yet supported
        hess=hess,
        hessp=hessp,
        callback=callback,
        # scipy arguments that will never be supported
        options=options,
        tol=tol,
        # deprecated arguments
        criterion=criterion,
        criterion_kwargs=criterion_kwargs,
        derivative=derivative,
        derivative_kwargs=derivative_kwargs,
        criterion_and_derivative=criterion_and_derivative,
        criterion_and_derivative_kwargs=criterion_and_derivative_kwargs,
        lower_bounds=lower_bounds,
        log_options=log_options,
        upper_bounds=upper_bounds,
        soft_lower_bounds=soft_lower_bounds,
        soft_upper_bounds=soft_upper_bounds,
        scaling_options=scaling_options,
        multistart_options=multistart_options,
    )
    return _optimize(problem)


def _optimize(problem: OptimizationProblem) -> OptimizeResult:
    """Solve an optimization problem."""
    # ==================================================================================
    # Split constraints into nonlinear and reparametrization parts
    # ==================================================================================
    constraints = problem.constraints

    nonlinear_constraints = [c for c in constraints if c["type"] == "nonlinear"]

    if nonlinear_constraints:
        if not problem.algorithm.algo_info.supports_nonlinear_constraints:
            raise ValueError(
                f"Algorithm {problem.algorithm.name} does not support "
                "nonlinear constraints."
            )

    # the following constraints will be handled via reparametrization
    constraints = [c for c in constraints if c["type"] != "nonlinear"]

    # ==================================================================================
    # Do first evaluation of user provided functions
    # ==================================================================================
    first_crit_eval = problem.fun_eval

    # do first derivative evaluation (if given)
    if problem.jac is not None:
        try:
            first_deriv_eval = problem.jac(problem.params)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            msg = "Error while evaluating derivative at start params."
            raise InvalidFunctionError(msg) from e

    if problem.fun_and_jac is not None:
        try:
            first_crit_and_deriv_eval = problem.fun_and_jac(problem.params)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            msg = "Error while evaluating criterion_and_derivative at start params."
            raise InvalidFunctionError(msg) from e

    if problem.jac is not None:
        used_deriv = first_deriv_eval
    elif problem.fun_and_jac is not None:
        used_deriv = first_crit_and_deriv_eval[1]
    else:
        used_deriv = None

    # ==================================================================================
    # Get the converter (for tree flattening, constraints and scaling)
    # ==================================================================================
    converter, internal_params = get_converter(
        params=problem.params,
        constraints=constraints,
        bounds=problem.bounds,
        func_eval=first_crit_eval.value,
        solver_type=problem.algorithm.algo_info.solver_type,
        scaling=problem.scaling,
        derivative_eval=used_deriv,
        add_soft_bounds=problem.multistart is not None,
    )

    # ==================================================================================
    # initialize the log database
    # ==================================================================================
    logger: LogStore[Any, Any] | None

    if problem.logging:
        logger = LogStore.from_options(problem.logging)
        problem_data = ProblemInitialization(problem.direction, problem.params)
        logger.problem_store.insert(problem_data)
    else:
        logger = None

    # ==================================================================================
    # Do some things that require internal parameters or bounds
    # ==================================================================================

    if converter.has_transforming_constraints and problem.multistart is not None:
        raise NotImplementedError(
            "multistart optimizations are not yet compatible with transforming "
            "constraints."
        )

    # get error penalty function
    error_penalty_func = get_error_penalty_function(
        start_x=internal_params.values,
        start_criterion=first_crit_eval,
        error_penalty=problem.error_penalty,
        solver_type=problem.algorithm.algo_info.solver_type,
        direction=problem.direction,
    )

    # process nonlinear constraints:
    internal_nonlinear_constraints = process_nonlinear_constraints(
        nonlinear_constraints=nonlinear_constraints,
        params=problem.params,
        bounds=problem.bounds,
        converter=converter,
        numdiff_options=problem.numdiff_options,
        skip_checks=problem.skip_checks,
    )

    x = internal_params.values
    internal_bounds = InternalBounds(
        lower=internal_params.lower_bounds,
        upper=internal_params.upper_bounds,
    )
    # ==================================================================================
    # Create a batch evaluator
    # ==================================================================================
    # TODO: Make batch evaluator an argument of maximize and minimize and move this
    # to create_optimization_problem
    batch_evaluator = process_batch_evaluator("joblib")

    # ==================================================================================
    # Create the InternalOptimizationProblem
    # ==================================================================================

    internal_problem = InternalOptimizationProblem(
        fun=problem.fun,
        jac=problem.jac,
        fun_and_jac=problem.fun_and_jac,
        converter=converter,
        solver_type=problem.algorithm.algo_info.solver_type,
        direction=problem.direction,
        bounds=internal_bounds,
        numdiff_options=problem.numdiff_options,
        error_handling=problem.error_handling,
        error_penalty_func=error_penalty_func,
        batch_evaluator=batch_evaluator,
        # TODO: Actually pass through linear constraints if possible
        linear_constraints=None,
        nonlinear_constraints=internal_nonlinear_constraints,
        logger=logger,
    )

    # ==================================================================================
    # Do actual optimization
    # ==================================================================================
    if problem.multistart is None:
        steps = [{"type": "optimization", "name": "optimization"}]

        # TODO: Actually use the step ids
        step_id = log_scheduled_steps_and_get_ids(  # noqa: F841
            steps=steps,
            logger=logger,
        )[0]

        raw_res = problem.algorithm.solve_internal_problem(internal_problem, x, step_id)

    else:
        multistart_options = get_internal_multistart_options_from_public(
            options=problem.multistart,
            params=problem.params,
            params_to_internal=converter.params_to_internal,
        )

        sampling_bounds = InternalBounds(
            lower=internal_params.soft_lower_bounds,
            upper=internal_params.soft_upper_bounds,
        )

        raw_res = run_multistart_optimization(
            local_algorithm=problem.algorithm,
            internal_problem=internal_problem,
            x=x,
            sampling_bounds=sampling_bounds,
            options=multistart_options,
            logger=logger,
            error_handling=problem.error_handling,
        )

    # ==================================================================================
    # Process the result
    # ==================================================================================

    _scalar_start_criterion = cast(
        float, first_crit_eval.internal_value(AggregationLevel.SCALAR)
    )
    log_reader: LogReader[Any] | None

    extra_fields = ExtraResultFields(
        start_fun=_scalar_start_criterion,
        start_params=problem.params,
        algorithm=problem.algorithm.algo_info.name,
        direction=problem.direction,
        n_free=internal_params.free_mask.sum(),
    )

    if problem.multistart is None:
        res = process_single_result(
            raw_res=raw_res,
            converter=converter,
            solver_type=problem.algorithm.algo_info.solver_type,
            extra_fields=extra_fields,
        )
    else:
        res = process_multistart_result(
            raw_res=raw_res,
            converter=converter,
            solver_type=problem.algorithm.algo_info.solver_type,
            extra_fields=extra_fields,
        )

    if logger is not None:
        assert problem.logging is not None
        log_reader = LogReader.from_options(problem.logging)
    else:
        log_reader = None

    res.logger = log_reader

    return res
