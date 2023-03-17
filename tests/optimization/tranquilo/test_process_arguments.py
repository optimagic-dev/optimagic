"""Tests for the process_arguments function and subfunctions.

When testing process_arguments we should only test the values of outputs that somehow
depend on the inputs, not the values with static defaults.

"""

import numpy as np
from estimagic.optimization.tranquilo.process_arguments import process_arguments


def test_process_arguments_scalar_deterministic():
    process_arguments(
        functype="scalar",
        criterion=lambda x: x @ x,
        x=np.array([-3, 1, 2]),
    )
