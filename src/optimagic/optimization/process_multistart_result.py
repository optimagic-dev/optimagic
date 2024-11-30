import numpy as np
from numpy.typing import NDArray

from optimagic.optimization.convergence_report import get_convergence_report
from optimagic.optimization.optimize_result import MultistartInfo, OptimizeResult
from optimagic.typing import Direction, ExtraResultFields


def process_multistart_result(
    raw_res: OptimizeResult,
    extra_fields: ExtraResultFields,
    local_optima: list[OptimizeResult],
    exploration_sample: list[NDArray[np.float64]],
    exploration_results: list[float],
) -> OptimizeResult:
    """Process results of internal optimizers."""

    if isinstance(raw_res, str):
        res = _dummy_result_from_traceback(raw_res, extra_fields)
    else:
        res = raw_res
        if extra_fields.direction == Direction.MAXIMIZE:
            exploration_results = [-res for res in exploration_results]

        info = MultistartInfo(
            start_parameters=[opt.start_params for opt in local_optima],
            local_optima=local_optima,
            exploration_sample=exploration_sample,
            exploration_results=exploration_results,
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


def _dummy_result_from_traceback(
    candidate: str, extra_fields: ExtraResultFields
) -> OptimizeResult:
    if extra_fields.start_fun is None:
        start_fun = np.inf
    else:
        start_fun = extra_fields.start_fun

    out = OptimizeResult(
        params=extra_fields.start_params,
        fun=start_fun,
        start_fun=start_fun,
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
