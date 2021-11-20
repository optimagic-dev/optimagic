from itertools import product

import numpy as np
import pytest
from estimagic.examples.benchmarking import get_problems

PARMETRIZATION = []
for name in ["more_wild", "cartis_roberts"]:
    for additive, multiplicative in product([False, True], repeat=2):
        PARMETRIZATION.append((name, additive, multiplicative))


@pytest.mark.parametrize("name, additive_noise, multiplicative_noise", PARMETRIZATION)
def test_get_problems(name, additive_noise, multiplicative_noise):
    is_noisy = any((additive_noise, multiplicative_noise))
    problems = get_problems(
        name=name,
        additive_noise=additive_noise,
        multiplicative_noise=multiplicative_noise,
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
