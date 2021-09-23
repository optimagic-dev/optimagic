import itertools

import pytest
from numpy.testing import assert_array_almost_equal as aaae

from estimagic.estimation.estimate_ml import estimate_ml
from estimagic.examples.logit import logit_derivative
from estimagic.examples.logit import logit_hessian
from estimagic.examples.logit import logit_loglike
from estimagic.examples.logit import logit_loglike_and_derivative


test_cases = itertools.product(
    [None, logit_derivative], [None, logit_loglike_and_derivative]
)


@pytest.mark.parametrize("derivative, loglike_and_derivative", test_cases)
def test_estimate_ml_with_logit_no_constraints(
    logit_inputs, logit_object, derivative, loglike_and_derivative
):

    sm_res = logit_object.fit()

    calculated = estimate_ml(
        logit_loglike,
        logit_inputs["params"],
        loglike_kwargs={"y": logit_inputs["y"], "x": logit_inputs["x"]},
        minimize_options={"algorithm": "scipy_lbfgsb"},
    )

    aaae(calculated["summary"]["value"].to_numpy(), sm_res["params"].to_numpy())
