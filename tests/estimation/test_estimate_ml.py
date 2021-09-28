import itertools

import pytest
from estimagic.estimation.estimate_ml import estimate_ml
from estimagic.examples.logit import logit_derivative
from estimagic.examples.logit import logit_hessian
from estimagic.examples.logit import logit_loglike
from estimagic.examples.logit import logit_loglike_and_derivative
from numpy.testing import assert_array_almost_equal as aaae


test_cases = itertools.product(
    [None, logit_derivative], [None, logit_loglike_and_derivative]
)


@pytest.mark.parametrize("derivative, loglike_and_derivative", test_cases)
def test_estimate_ml_with_logit_no_constraints(
    logit_inputs, logit_object, derivative, loglike_and_derivative
):

    sm_res = logit_object.fit()
    kwargs = {"y": logit_inputs["y"], "x": logit_inputs["x"]}
    calculated = estimate_ml(
        logit_loglike,
        logit_inputs["params"],
        loglike_kwargs=kwargs,
        optimize_options={"algorithm": "scipy_lbfgsb"},
        hessian=logit_hessian,
        hessian_kwargs=kwargs,
    )

    calc_summary = calculated["summary_hessian"]

    aaae(calc_summary["value"].to_numpy(), sm_res.params.to_numpy(), decimal=4)

    aaae(
        calculated["cov_hessian"].to_numpy(), sm_res.cov_params().to_numpy(), decimal=3
    )

    aaae(calc_summary["standard_error"].to_numpy(), sm_res.bse.to_numpy(), decimal=4)

    aaae(
        calc_summary[["ci_lower", "ci_upper"]].to_numpy(),
        sm_res.conf_int().to_numpy(),
        decimal=3,
    )
