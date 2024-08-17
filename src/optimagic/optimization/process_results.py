from dataclasses import dataclass

import numpy as np

from optimagic.optimization.algorithm import InternalOptimizeResult
from optimagic.optimization.convergence_report import get_convergence_report
from optimagic.optimization.optimize_result import MultistartInfo, OptimizeResult
from optimagic.parameters.conversion import Converter
from optimagic.typing import AggregationLevel, Direction, PyTree
from optimagic.utilities import isscalar


@dataclass(frozen=True)
class ExtraResultFields:
    """Fields for OptimizeResult that are not part of InternalOptimizeResult."""

    start_fun: float
    start_params: PyTree
    algorithm: str
    direction: Direction
    n_free: int


def _process_one_result(res, converter, solver_type, fixed_kwargs, skip_checks):
    _params = converter.params_from_internal(res["solution_x"])

    if isscalar(res["solution_criterion"]):
        _criterion = float(res["solution_criterion"])
    elif solver_type == AggregationLevel.LIKELIHOOD:
        _criterion = float(np.sum(res["solution_criterion"]))
    elif solver_type == AggregationLevel.LEAST_SQUARES:
        _criterion = res["solution_criterion"] @ res["solution_criterion"]

    if fixed_kwargs["direction"] == "maximize":
        _criterion = -_criterion

    optional_entries = [
        "n_fun_evals",
        "n_jac_evals",
        "n_iterations",
        "success",
        "message",
        "history",
    ]

    optional_kwargs = {}
    for key in optional_entries:
        if key in res:
            optional_kwargs[key] = res[key]

    algo_output = {}
    for key in res:
        if key not in [*optional_entries, "solution_x", "solution_criterion"]:
            algo_output[key] = res[key]

    if "history" in res and not skip_checks:
        conv_report = get_convergence_report(
            history=res["history"],
            direction=fixed_kwargs["direction"],
        )

    else:
        conv_report = None

    out = OptimizeResult(
        params=_params,
        fun=_criterion,
        **fixed_kwargs,
        **optional_kwargs,
        algorithm_output=algo_output,
        convergence_report=conv_report,
    )

    return out


def process_single_result(
    res: InternalOptimizeResult,
    converter: Converter,
    solver_type: AggregationLevel,
    extra_fields: ExtraResultFields,
) -> OptimizeResult:
    """Process an internal optimizer result."""
    params = converter.params_from_internal(res.x)
    if isscalar(res.fun):
        fun = float(res.fun)
    elif solver_type == AggregationLevel.LIKELIHOOD:
        fun = float(np.sum(res.fun))
    elif solver_type == AggregationLevel.LEAST_SQUARES:
        fun = np.dot(res.fun, res.fun)

    if extra_fields.direction == Direction.MAXIMIZE:
        fun = -fun

    if res.history is not None:
        conv_report = get_convergence_report(
            history=res.history,
            direction=extra_fields.direction.value,
        )
    else:
        conv_report = None

    out = OptimizeResult(
        params=params,
        fun=fun,
        start_fun=extra_fields.start_fun,
        start_params=extra_fields.start_params,
        algorithm=extra_fields.algorithm,
        direction=extra_fields.direction.value,
        n_free=extra_fields.n_free,
        message=res.message,
        success=res.success,
        n_fun_evals=res.n_fun_evals,
        n_jac_evals=res.n_jac_evals,
        n_hess_evals=res.n_hess_evals,
        n_iterations=res.n_iterations,
        status=res.status,
        jac=res.jac,
        hess=res.hess,
        hess_inv=res.hess_inv,
        max_constraint_violation=res.max_constraint_violation,
        history=res.history,
        algorithm_output=res.info,
    )
    return out


def process_internal_optimizer_result(
    res,
    converter,
    solver_type,
    fixed_kwargs,
    skip_checks,
):
    """Process results of internal optimizers.

    Args:
        res (dict): Results dictionary of an internal optimizer or multistart optimizer.

    """
    is_multistart = "multistart_info" in res
    multistart_info = res.get("multistart_info")

    if isinstance(res, str):
        res = _dummy_result_from_traceback(res, fixed_kwargs)
    else:
        res = _process_one_result(
            res, converter, solver_type, fixed_kwargs, skip_checks
        )

        if is_multistart:
            info = _process_multistart_info(
                multistart_info,
                converter,
                solver_type,
                fixed_kwargs=fixed_kwargs,
                skip_checks=skip_checks,
            )

            crit_hist = [opt.fun for opt in info.local_optima]
            params_hist = [opt.params for opt in info.local_optima]
            time_hist = [np.nan for opt in info.local_optima]
            hist = {"criterion": crit_hist, "params": params_hist, "runtime": time_hist}

            conv_report = get_convergence_report(
                history=hist,
                direction=fixed_kwargs["direction"],
            )

            res.convergence_report = conv_report

            res.algorithm = f"multistart_{res.algorithm}"
            res.n_iterations = _sum_or_none(
                [opt.n_iterations for opt in info.local_optima]
            )

            res.n_fun_evals = _sum_or_none(
                [opt.n_fun_evals for opt in info.local_optima]
            )
            res.n_jac_evals = _sum_or_none(
                [opt.n_jac_evals for opt in info.local_optima]
            )

            res.multistart_info = info
    return res


def _process_multistart_info(info, converter, solver_type, fixed_kwargs, skip_checks):
    direction = fixed_kwargs["direction"]

    starts = [converter.params_from_internal(x) for x in info["start_parameters"]]

    optima = []
    for res, start in zip(info["local_optima"], starts, strict=False):
        kwargs = fixed_kwargs.copy()
        kwargs["start_params"] = start
        kwargs["start_fun"] = None
        processed = _process_one_result(
            res,
            converter=converter,
            solver_type=solver_type,
            fixed_kwargs=kwargs,
            skip_checks=skip_checks,
        )
        optima.append(processed)

    sample = [converter.params_from_internal(x) for x in info["exploration_sample"]]

    if direction == "minimize":
        exploration_res = info["exploration_results"]
    else:
        exploration_res = [switch_sign(res) for res in info["exploration_results"]]

    return MultistartInfo(
        start_parameters=starts,
        local_optima=optima,
        exploration_sample=sample,
        exploration_results=exploration_res,
    )


def _dummy_result_from_traceback(candidate, fixed_kwargs):  # noqa: ARG001
    out = OptimizeResult(
        params=None,
        fun=None,
        **fixed_kwargs,
    )
    return out


def switch_sign(critval):
    if isinstance(critval, dict):
        out = critval.copy()
        if "value" in critval:
            out["value"] = -critval["value"]
        if "contributions" in critval:
            out["contributions"] = -critval["contributions"]
    else:
        out = -critval
    return out


def _sum_or_none(summands):
    if any(s is None for s in summands):
        out = None
    else:
        out = int(np.sum(summands))
    return out
