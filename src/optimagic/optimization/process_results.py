from dataclasses import dataclass, replace

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
        convergence_report=conv_report,
    )
    return out


def process_multistart_result(
    res,
    converter,
    solver_type,
    extra_fields,
):
    """Process results of internal optimizers.

    Args:
        res (dict): Results dictionary of an internal optimizer or multistart optimizer.

    """
    info = res.multistart_info

    if isinstance(res, str):
        res = _dummy_result_from_traceback(res, extra_fields)
    else:
        res = process_single_result(
            res=res,
            converter=converter,
            solver_type=solver_type,
            extra_fields=extra_fields,
        )

        info = _process_multistart_info(
            info,
            converter=converter,
            solver_type=solver_type,
            extra_fields=extra_fields,
        )

        # ==============================================================================
        # create a convergence report for the multistart optimization; This is not
        # the same as the convergence report for the individual local optimizations.
        # ==============================================================================
        crit_hist = [opt.fun for opt in info.local_optima]
        params_hist = [opt.params for opt in info.local_optima]
        time_hist = [np.nan for opt in info.local_optima]
        hist = {"criterion": crit_hist, "params": params_hist, "runtime": time_hist}

        conv_report = get_convergence_report(
            history=hist,
            direction=extra_fields.direction,
        )

        res.convergence_report = conv_report

        res.algorithm = f"multistart_{res.algorithm}"
        res.n_iterations = _sum_or_none([opt.n_iterations for opt in info.local_optima])

        res.n_fun_evals = _sum_or_none([opt.n_fun_evals for opt in info.local_optima])
        res.n_jac_evals = _sum_or_none([opt.n_jac_evals for opt in info.local_optima])

        res.multistart_info = info
    return res


def _process_multistart_info(info, converter, solver_type, extra_fields):
    starts = [converter.params_from_internal(x) for x in info["start_parameters"]]

    optima = []
    for res, start in zip(info["local_optima"], starts, strict=False):
        replacements = {
            "start_params": start,
            "start_fun": None,
        }

        processed = process_single_result(
            res,
            converter=converter,
            solver_type=solver_type,
            extra_fields=replace(extra_fields, **replacements),
        )
        optima.append(processed)

    sample = [converter.params_from_internal(x) for x in info["exploration_sample"]]

    if extra_fields.direction == Direction.MINIMIZE:
        exploration_res = info["exploration_results"]
    else:
        exploration_res = [-res for res in info["exploration_results"]]

    return MultistartInfo(
        start_parameters=starts,
        local_optima=optima,
        exploration_sample=sample,
        exploration_results=exploration_res,
    )


def _dummy_result_from_traceback(candidate, extra_fields):  # noqa: ARG001
    out = OptimizeResult(
        params=None,
        fun=None,
        start_fun=extra_fields.start_fun,
        start_params=extra_fields.start_params,
        algorithm=extra_fields.algorithm,
        direction=extra_fields.direction.value,
        n_free=extra_fields.n_free,
        message=candidate,
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
