import itertools

import numpy as np
import pytest
from estimagic.estimation.estimate_ml import estimate_ml
from estimagic.examples.logit import logit_derivative
from estimagic.examples.logit import logit_hessian
from estimagic.examples.logit import logit_loglike
from estimagic.examples.logit import logit_loglike_and_derivative as llad


def aaae(obj1, obj2, decimal=3):
    arr1 = np.asarray(obj1)
    arr2 = np.asarray(obj2)
    np.testing.assert_array_almost_equal(arr1, arr2, decimal=decimal)


def logit_jacobian(params, y, x):
    return logit_derivative(params, y, x)["contributions"]


def logit_loglike_and_derivative(params, y, x):
    loglike, loglike_derivative = llad(params, y, x)
    return loglike, loglike_derivative["value"]


test_cases = itertools.product(
    [{"algorithm": "scipy_lbfgsb"}, "scipy_lbfgsb"],  # optimize_options
    [None, logit_loglike_and_derivative],  # loglike_and_derivative
    [None, logit_jacobian, False],  # jacobian
    [None, logit_hessian, False],  # hessian
)


@pytest.mark.parametrize(
    "optimize_options, loglike_and_derivative, jacobian, hessian", test_cases
)
def test_estimate_ml_with_logit_no_constraints(
    logit_object,
    logit_inputs,
    optimize_options,
    loglike_and_derivative,
    jacobian,
    hessian,
):

    if jacobian is False and hessian is False:
        pytest.xfail("jacobian and hessian cannot both be False.")

    # ==================================================================================
    # estimate
    # ==================================================================================

    kwargs = {"y": logit_inputs["y"], "x": logit_inputs["x"]}

    got = estimate_ml(
        loglike=logit_loglike,
        params=logit_inputs["params"],
        loglike_kwargs=kwargs,
        optimize_options=optimize_options,
        jacobian=jacobian,
        jacobian_kwargs=kwargs,
        hessian=hessian,
        hessian_kwargs=kwargs,
        loglike_and_derivative=loglike_and_derivative,
        loglike_and_derivative_kwargs=kwargs,
    )

    # ==================================================================================
    # test
    # ==================================================================================

    from statsmodels.base.model import GenericLikelihoodModel

    class GenericLogit(GenericLikelihoodModel):
        def nloglikeobs(self, params, *args, **kwargs):
            return -logit_object.loglikeobs(params, *args, **kwargs)

    generic_logit = GenericLogit(logit_object.endog, logit_object.exog)
    exp = generic_logit.fit()

    if jacobian is not False and hessian is not False:
        cases = ["jacobian", "hessian", "robust"]
    elif jacobian is not False:
        cases = ["jacobian"]
    elif hessian is not False:
        cases = ["hessian"]

    statsmodels_suffix_map = {
        "jacobian": "jac",
        "hessian": "",
        "robust": "jhj",
    }

    for case in cases:

        summary = got[f"summary_{case}"]

        # compare estimated parameters
        aaae(summary["value"], exp.params, decimal=4)

        # compare estimated standard errors
        exp_se = getattr(exp, f"bse{statsmodels_suffix_map[case]}")
        aaae(summary["standard_error"], exp_se, decimal=3)

        # compare estimated confidence interval
        if case == "hessian":
            aaae(summary[["ci_lower", "ci_upper"]], exp.conf_int(), decimal=3)

        # compare covariance (if not robust case)
        if case == "hessian":
            aaae(got[f"cov_{case}"], exp.cov_params(), decimal=3)
        elif case == "robust":
            aaae(got[f"cov_{case}"], exp.covjhj, decimal=2)
        elif case == "jacobian":
            aaae(got[f"cov_{case}"], exp.covjac, decimal=4)
