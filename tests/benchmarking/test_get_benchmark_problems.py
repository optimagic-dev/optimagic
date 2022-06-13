from itertools import product

import numpy as np
import pytest
from estimagic import get_benchmark_problems

PARMETRIZATION = []
for name in ["more_wild", "cartis_roberts", "example", "estimagic"]:
    for additive, multiplicative, scaling in product([False, True], repeat=3):
        PARMETRIZATION.append((name, additive, multiplicative, scaling))


@pytest.mark.parametrize(
    "name, additive_noise, multiplicative_noise, scaling", PARMETRIZATION
)
def test_get_problems(name, additive_noise, multiplicative_noise, scaling):
    is_noisy = any((additive_noise, multiplicative_noise))
    problems = get_benchmark_problems(
        name=name,
        additive_noise=additive_noise,
        multiplicative_noise=multiplicative_noise,
        scaling=scaling,
    )
    first_name = list(problems)[0]
    first = problems[first_name]
    func = first["inputs"]["criterion"]
    params = first["inputs"]["params"]

    np.random.seed()
    first_eval = func(params)["value"]
    second_eval = func(params)["value"]

    if is_noisy:
        assert first_eval != second_eval
    else:
        assert first_eval == second_eval
