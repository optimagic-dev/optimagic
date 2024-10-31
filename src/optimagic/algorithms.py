import inspect

from optimagic.optimization.algorithm import Algorithm
from optimagic.optimizers import (
    bhhh,
    fides,
    ipopt,
    nag_optimizers,
    neldermead,
    nlopt_optimizers,
    pounders,
    pygmo_optimizers,
    scipy_optimizers,
    tao_optimizers,
    tranquilo,
)

MODULES = [
    ipopt,
    fides,
    nag_optimizers,
    nlopt_optimizers,
    pygmo_optimizers,
    scipy_optimizers,
    tao_optimizers,
    bhhh,
    neldermead,
    pounders,
    tranquilo,
]

ALL_ALGORITHMS = {}
AVAILABLE_ALGORITHMS = {}
for module in MODULES:
    candidate_dict = dict(inspect.getmembers(module, inspect.isclass))
    candidate_dict = {
        k: v for k, v in candidate_dict.items() if hasattr(v, "__algo_info__")
    }
    for candidate in candidate_dict.values():
        name = candidate.__algo_info__.name
        if issubclass(candidate, Algorithm) and candidate is not Algorithm:
            ALL_ALGORITHMS[name] = candidate
            if candidate.__algo_info__.is_available:  # type: ignore[attr-defined]
                AVAILABLE_ALGORITHMS[name] = candidate


GLOBAL_ALGORITHMS = [
    name
    for name, algo in ALL_ALGORITHMS.items()
    if algo.__algo_info__.is_global  # type: ignore[attr-defined]
]
