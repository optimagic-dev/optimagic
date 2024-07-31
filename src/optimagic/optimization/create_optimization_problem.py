from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from optimagic import deprecations
from optimagic.decorators import AlgoInfo
from optimagic.deprecations import (
    replace_and_warn_about_deprecated_algo_options,
    replace_and_warn_about_deprecated_bounds,
)
from optimagic.exceptions import (
    AliasError,
    MissingInputError,
)
from optimagic.optimization.check_arguments import check_optimize_kwargs
from optimagic.optimization.get_algorithm import (
    process_user_algorithm,
)
from optimagic.optimization.scipy_aliases import (
    map_method_to_algorithm,
    split_fun_and_jac,
)
from optimagic.parameters.bounds import Bounds, pre_process_bounds
from optimagic.shared.process_user_function import (
    get_kwargs_from_args,
    process_func_of_params,
)
from optimagic.typing import PyTree


@dataclass(frozen=True)
class OptimizationProblem:
    """Collect everything that defines the optimization problem.

    The attributes are very close to the arguments of `maximize` and `minimize` but they
    are converted to stricter types. For example, the bounds argument that can be a
    sequence of tuples, a scipy.optimize.Bounds object or an optimagic.Bounds when
    calling `maximize` or `minimize` is converted to an optimagic.Bounds object.

    All deprecated arguments are removed and all scipy aliases are replaced by their
    optimagic counterparts.

    All user provided functions are partialled if corresponding `kwargs` dictionaries
    were provided.

    # TODO: Document attributes after other todos are resolved.

    """

    fun: Callable[[PyTree], float | PyTree]
    params: PyTree
    # TODO: algorithm will become an Algorithm object; algo_options and algo_info will
    # be removed and become part of Algorithm
    algorithm: Callable
    algo_options: dict[str, Any] | None
    algo_info: AlgoInfo
    bounds: Bounds
    # TODO: constraints will become list[Constraint] | None
    constraints: list[dict[str, Any]]
    jac: Callable[[PyTree], PyTree] | None
    fun_and_jac: Callable[[PyTree], tuple[float, PyTree]] | None
    # TODO: numdiff_options will become NumDiffOptions
    numdiff_options: dict[str, Any] | None
    # TODO: logging will become None | Logger and log_options will be removed
    logging: bool | Path | None
    log_options: dict[str, Any] | None
    # TODO: error_handling will become None | ErrorHandlingOptions and error_penalty
    # will be removed
    error_handling: Literal["raise", "continue"]
    error_penalty: dict[str, Any] | None
    # TODO: scaling will become None | ScalingOptions and scaling_options will be
    # removed
    scaling: bool
    scaling_options: dict[str, Any] | None
    # TODO: multistart will become None | MultistartOptions and multistart_options will
    # be removed
    multistart: bool
    multistart_options: dict[str, Any] | None
    collect_history: bool
    skip_checks: bool
    direction: Literal["minimize", "maximize"]


def create_optimization_problem(
    direction,
    fun,
    params,
    algorithm,
    *,
    bounds,
    fun_kwargs,
    constraints,
    algo_options,
    jac,
    jac_kwargs,
    fun_and_jac,
    fun_and_jac_kwargs,
    numdiff_options,
    logging,
    log_options,
    error_handling,
    error_penalty,
    scaling,
    scaling_options,
    multistart,
    multistart_options,
    collect_history,
    skip_checks,
    # scipy aliases
    x0,
    method,
    args,
    # scipy arguments that are not yet supported
    hess,
    hessp,
    callback,
    # scipy arguments that will never be supported
    options,
    tol,
    # deprecated arguments
    criterion,
    criterion_kwargs,
    derivative,
    derivative_kwargs,
    criterion_and_derivative,
    criterion_and_derivative_kwargs,
    lower_bounds,
    upper_bounds,
    soft_lower_bounds,
    soft_upper_bounds,
):
    # ==================================================================================
    # error handling needed as long as fun is an optional argument (i.e. until
    # criterion is fully removed).
    # ==================================================================================

    if fun is None and criterion is None:
        msg = (
            "Missing objective function. Please provide an objective function as the "
            "first positional argument or as the keyword argument `fun`."
        )
        raise MissingInputError(msg)

    if params is None and x0 is None:
        msg = (
            "Missing start parameters. Please provide start parameters as the second "
            "positional argument or as the keyword argument `params`."
        )
        raise MissingInputError(msg)

    if algorithm is None and method is None:
        msg = (
            "Missing algorithm. Please provide an algorithm as the third positional "
            "argument or as the keyword argument `algorithm`."
        )
        raise MissingInputError(msg)

    # ==================================================================================
    # deprecations
    # ==================================================================================

    if criterion is not None:
        deprecations.throw_criterion_future_warning()
        fun = criterion if fun is None else fun

    if criterion_kwargs is not None:
        deprecations.throw_criterion_kwargs_future_warning()
        fun_kwargs = criterion_kwargs if fun_kwargs is None else fun_kwargs

    if derivative is not None:
        deprecations.throw_derivative_future_warning()
        jac = derivative if jac is None else jac

    if derivative_kwargs is not None:
        deprecations.throw_derivative_kwargs_future_warning()
        jac_kwargs = derivative_kwargs if jac_kwargs is None else jac_kwargs

    if criterion_and_derivative is not None:
        deprecations.throw_criterion_and_derivative_future_warning()
        fun_and_jac = criterion_and_derivative if fun_and_jac is None else fun_and_jac

    if criterion_and_derivative_kwargs is not None:
        deprecations.throw_criterion_and_derivative_kwargs_future_warning()
        fun_and_jac_kwargs = (
            criterion_and_derivative_kwargs
            if fun_and_jac_kwargs is None
            else fun_and_jac_kwargs
        )

    algo_options = replace_and_warn_about_deprecated_algo_options(algo_options)

    bounds = replace_and_warn_about_deprecated_bounds(
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        bounds=bounds,
        soft_lower_bounds=soft_lower_bounds,
        soft_upper_bounds=soft_upper_bounds,
    )

    # ==================================================================================
    # handle scipy aliases
    # ==================================================================================

    if x0 is not None:
        if params is not None:
            msg = (
                "x0 is an alias for params (for better compatibility with scipy). "
                "Do not use both x0 and params."
            )
            raise AliasError(msg)
        else:
            params = x0

    if method is not None:
        if algorithm is not None:
            msg = (
                "method is an alias for algorithm to select the scipy optimizers under "
                "their original name. Do not use both method and algorithm."
            )
            raise AliasError(msg)
        else:
            algorithm = map_method_to_algorithm(method)

    if args is not None:
        if (
            fun_kwargs is not None
            or jac_kwargs is not None
            or fun_and_jac_kwargs is not None
        ):
            msg = (
                "args is an alternative to fun_kwargs, jac_kwargs and "
                "fun_and_jac_kwargs that optimagic supports for compatibility "
                "with scipy. Do not use args in conjunction with any of the other "
                "arguments."
            )
            raise AliasError(msg)
        else:
            kwargs = get_kwargs_from_args(args, fun, offset=1)
            fun_kwargs, jac_kwargs, fun_and_jac_kwargs = kwargs, kwargs, kwargs

    # jac is not an alias but we need to handle the case where `jac=True`, i.e. fun is
    # actually fun_and_jac. This is not recommended in optimagic because then optimizers
    # cannot evaluate fun in isolation but we can easily support it for compatibility.
    if jac is True:
        jac = None
        if fun_and_jac is None:
            fun_and_jac = fun
            fun = split_fun_and_jac(fun_and_jac, target="fun")

    bounds = pre_process_bounds(bounds)

    # ==================================================================================
    # Handle scipy arguments that are not yet implemented
    # ==================================================================================

    if hess is not None:
        msg = (
            "The hess argument is not yet supported in optimagic. Creat an issue on "
            "https://github.com/OpenSourceEconomics/optimagic/ if you have urgent need "
            "for this feature."
        )
        raise NotImplementedError(msg)

    if hessp is not None:
        msg = (
            "The hessp argument is not yet supported in optimagic. Creat an issue on "
            "https://github.com/OpenSourceEconomics/optimagic/ if you have urgent need "
            "for this feature."
        )
        raise NotImplementedError(msg)

    if callback is not None:
        msg = (
            "The callback argument is not yet supported in optimagic. Creat an issue "
            "on https://github.com/OpenSourceEconomics/optimagic/ if you have urgent "
            "need for this feature."
        )
        raise NotImplementedError(msg)

    # ==================================================================================
    # Handle scipy arguments that will never be supported
    # ==================================================================================

    if options is not None:
        # TODO: Add link to a how-to guide or tutorial for this
        msg = (
            "The options argument is not supported in optimagic. Please use the "
            "algo_options argument instead."
        )
        raise NotImplementedError(msg)

    if tol is not None:
        # TODO: Add link to a how-to guide or tutorial for this
        msg = (
            "The tol argument is not supported in optimagic. Please use "
            "algo_options or configured algorithms instead to set convergence criteria "
            "for your optimizer."
        )
        raise NotImplementedError(msg)

    # ==================================================================================
    # Set default values and check options
    # ==================================================================================
    fun_kwargs = {} if fun_kwargs is None else fun_kwargs
    constraints = [] if constraints is None else constraints
    algo_options = {} if algo_options is None else algo_options
    jac_kwargs = {} if jac_kwargs is None else jac_kwargs
    fun_and_jac_kwargs = {} if fun_and_jac_kwargs is None else fun_and_jac_kwargs
    numdiff_options = {} if numdiff_options is None else numdiff_options
    log_options = {} if log_options is None else log_options
    error_penalty = {} if error_penalty is None else error_penalty
    scaling_options = {} if scaling_options is None else scaling_options
    multistart_options = {} if multistart_options is None else multistart_options
    if logging:
        logging = Path(logging)

    # ==================================================================================
    # Check types of arguments
    # ==================================================================================
    # TODO: This should probably be inlined

    if not skip_checks:
        check_optimize_kwargs(
            direction=direction,
            criterion=fun,
            criterion_kwargs=fun_kwargs,
            params=params,
            algorithm=algorithm,
            constraints=constraints,
            algo_options=algo_options,
            derivative=jac,
            derivative_kwargs=jac_kwargs,
            criterion_and_derivative=fun_and_jac,
            criterion_and_derivative_kwargs=fun_and_jac_kwargs,
            numdiff_options=numdiff_options,
            logging=logging,
            log_options=log_options,
            error_handling=error_handling,
            error_penalty=error_penalty,
            scaling=scaling,
            scaling_options=scaling_options,
            multistart=multistart,
            multistart_options=multistart_options,
        )
    # ==================================================================================
    # Get the algorithm info
    # ==================================================================================
    raw_algo, algo_info = process_user_algorithm(algorithm)

    if algo_info.primary_criterion_entry == "root_contributions":
        if direction == "maximize":
            msg = (
                "Optimizers that exploit a least squares structure like {} can only be "
                "used for minimization."
            )
            raise ValueError(msg.format(algo_info.name))

    # ==================================================================================
    # partial the kwargs into corresponding functions
    # ==================================================================================
    fun = process_func_of_params(
        func=fun,
        kwargs=fun_kwargs,
        name="criterion",
        skip_checks=skip_checks,
    )
    if isinstance(jac, dict):
        jac = jac.get(algo_info.primary_criterion_entry)
    if jac is not None:
        jac = process_func_of_params(
            func=jac,
            kwargs=jac_kwargs,
            name="derivative",
            skip_checks=skip_checks,
        )
    if isinstance(fun_and_jac, dict):
        fun_and_jac = fun_and_jac.get(algo_info.primary_criterion_entry)

    if fun_and_jac is not None:
        fun_and_jac = process_func_of_params(
            func=fun_and_jac,
            kwargs=fun_and_jac_kwargs,
            name="criterion_and_derivative",
            skip_checks=skip_checks,
        )

    # ==================================================================================
    # create the problem object
    # ==================================================================================

    problem = OptimizationProblem(
        fun=fun,
        params=params,
        algorithm=raw_algo,
        algo_options=algo_options,
        algo_info=algo_info,
        bounds=bounds,
        constraints=constraints,
        jac=jac,
        fun_and_jac=fun_and_jac,
        numdiff_options=numdiff_options,
        logging=logging,
        log_options=log_options,
        error_handling=error_handling,
        error_penalty=error_penalty,
        scaling=scaling,
        scaling_options=scaling_options,
        multistart=multistart,
        multistart_options=multistart_options,
        collect_history=collect_history,
        skip_checks=skip_checks,
        direction=direction,
    )

    return problem