import inspect

from estimagic.optimization import bhhh
from estimagic.optimization import cyipopt_optimizers
from estimagic.optimization import fides_optimizers
from estimagic.optimization import nag_optimizers
from estimagic.optimization import neldermead
from estimagic.optimization import nlopt_optimizers
from estimagic.optimization import pounders
from estimagic.optimization import pygmo_optimizers
from estimagic.optimization import scipy_optimizers
from estimagic.optimization import tao_optimizers


MODULES = [
    cyipopt_optimizers,
    fides_optimizers,
    nag_optimizers,
    nlopt_optimizers,
    pygmo_optimizers,
    scipy_optimizers,
    tao_optimizers,
    bhhh,
    neldermead,
    pounders,
]

ALL_ALGORITHMS = {}
AVAILABLE_ALGORITHMS = {}
for module in MODULES:
    func_dict = dict(inspect.getmembers(module, inspect.isfunction))
    for name, func in func_dict.items():
        if hasattr(func, "_algorithm_info"):
            ALL_ALGORITHMS[name] = func
            if func._algorithm_info.is_available:
                AVAILABLE_ALGORITHMS[name] = func


GLOBAL_ALGORITHMS = []
for name, func in ALL_ALGORITHMS.items():
    if func._algorithm_info.is_global:
        GLOBAL_ALGORITHMS.append(name)
