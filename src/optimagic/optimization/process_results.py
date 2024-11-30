import copy
from typing import Any

import numpy as np

from optimagic.optimization.convergence_report import get_convergence_report
from optimagic.optimization.optimize_result import MultistartInfo, OptimizeResult
from optimagic.parameters.conversion import Converter
from optimagic.typing import Direction, ExtraResultFields


def process_multistart_result(
    raw_res: OptimizeResult,
    converter: Converter,
    extra_fields: ExtraResultFields,
    multistart_info: dict[str, Any],
) -> OptimizeResult:
    """Process results of internal optimizers."""

    if isinstance(raw_res, str):
        res = _dummy_result_from_traceback(raw_res, extra_fields)
    else:
        res = raw_res
        info = _process_multistart_info(
            multistart_info,
            converter=converter,
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


def _process_multistart_info(
    info: dict[str, Any],
    converter: Converter,
    extra_fields: ExtraResultFields,
) -> MultistartInfo:
    starts = [converter.params_from_internal(x) for x in info["start_parameters"]]

    optima = []
    for res, start in zip(info["local_optima"], starts, strict=False):
        processed = copy.copy(res)
        processed.start_params = start
        processed.start_fun = None
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


def _dummy_result_from_traceback(
    candidate: str, extra_fields: ExtraResultFields
) -> OptimizeResult:
    out = OptimizeResult(
        params=extra_fields.start_params,
        fun=extra_fields.start_fun,
        start_fun=extra_fields.start_fun,
        start_params=extra_fields.start_params,
        algorithm=extra_fields.algorithm,
        direction=extra_fields.direction.value,
        n_free=extra_fields.n_free,
        message=candidate,
    )
    return out


def _sum_or_none(summands: list[int | None | float]) -> int | None:
    if any(s is None for s in summands):
        out = None
    else:
        out = int(np.array(summands).sum())
    return out
