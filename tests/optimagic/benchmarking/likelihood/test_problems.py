import numpy as np
import pytest

from optimagic.benchmarking.likelihood.problems import (
    LIKELIHOOD_SETS,
    get_likelihood_problems,
)
from optimagic.benchmarking.likelihood.solutions import LIKELIHOOD_SOLUTIONS

PARAM_RANGES = {
    "likelihood_small": (3, 25),
    "likelihood_large": (25, 150),
    "likelihood_misspecified_small": (3, 25),
    "likelihood_misspecified_large": (25, 150),
}

INSTANCES = [
    (set_name, instance)
    for set_name, instances in LIKELIHOOD_SETS.items()
    for instance in instances
]


@pytest.fixture(
    params=INSTANCES, ids=[instance.name for _, instance in INSTANCES], scope="module"
)
def case(request):
    set_name, instance = request.param
    return set_name, instance, instance.build()


def _bounds(likelihood):
    k = likelihood.n_params
    lower = likelihood.lower_bounds
    upper = likelihood.upper_bounds
    lower = np.full(k, -np.inf) if lower is None else lower
    upper = np.full(k, np.inf) if upper is None else upper
    return lower, upper


def test_set_names_match_param_ranges():
    assert set(LIKELIHOOD_SETS) == set(PARAM_RANGES)


def test_instance_names_are_unique():
    names = [instance.name for _, instance in INSTANCES]
    assert len(names) == len(set(names))


def test_stored_solutions_match_instances():
    names = {instance.name for _, instance in INSTANCES}
    assert set(LIKELIHOOD_SOLUTIONS) == names


@pytest.mark.parametrize("set_name", list(LIKELIHOOD_SETS))
def test_misspecification_is_documented(set_name):
    is_misspecified_set = "misspecified" in set_name
    for instance in LIKELIHOOD_SETS[set_name]:
        if is_misspecified_set:
            assert instance.misspecification is not None
            assert instance.severity in {"tiny", "mild", "moderate", "severe"}
            assert f"_{instance.severity}_" in instance.name
        else:
            assert instance.misspecification is None
            assert instance.severity is None


def test_n_params_matches_name_and_range(case):
    set_name, instance, likelihood = case
    low, high = PARAM_RANGES[set_name]
    assert low <= likelihood.n_params <= high
    assert instance.name.endswith(f"_{likelihood.n_params}")


def test_true_params_iff_correctly_specified(case):
    _, instance, likelihood = case
    if instance.misspecification is None:
        assert likelihood.true_params is not None
    else:
        assert likelihood.true_params is None


def test_stored_solution_value_is_reproduced(case):
    _, instance, likelihood = case
    solution = LIKELIHOOD_SOLUTIONS[instance.name]
    value = likelihood.fun(np.array(solution["params"])).sum()
    np.testing.assert_allclose(value, solution["value"], rtol=1e-10)


def test_stored_solution_satisfies_optimality_conditions(case):
    _, instance, likelihood = case
    x = np.array(LIKELIHOOD_SOLUTIONS[instance.name]["params"])
    lower, upper = _bounds(likelihood)
    assert (x >= lower).all()
    assert (x <= upper).all()

    jac = likelihood.jac(x)
    gradient = jac.sum(axis=0)
    # on a bound, the gradient must not point into the feasible set
    assert (gradient[x == lower] >= -1e-6).all()
    assert (gradient[x == upper] <= 1e-6).all()

    # at free params, the gradient must be zero; measured in the OPG metric, this is
    # the squared Wald distance to the optimum
    free = (x > lower) & (x < upper)
    jac_free = jac[:, free]
    direction = np.linalg.lstsq(jac_free, np.ones(len(jac)), rcond=None)[0]
    decrement = gradient[free] @ direction
    assert decrement < 1e-10


def test_solution_is_better_than_start_values(case):
    _, instance, likelihood = case
    x = np.array(LIKELIHOOD_SOLUTIONS[instance.name]["params"])
    value = likelihood.fun(x).sum()
    assert value < likelihood.fun(likelihood.good_start).sum()
    assert value < likelihood.fun(likelihood.bad_start).sum()


def test_stored_solution_metadata_is_valid():
    for solution in LIKELIHOOD_SOLUTIONS.values():
        low, high = solution["opg_hessian_ratio"]
        assert 0 < low <= high
        assert isinstance(solution["other_local_optima"], bool)


def test_get_likelihood_problems():
    problems = get_likelihood_problems("likelihood_misspecified_small")
    instances = LIKELIHOOD_SETS["likelihood_misspecified_small"]
    assert len(problems) == 2 * len(instances)

    instance = instances[0]
    good = problems[f"{instance.name}_good_start"]
    bad = problems[f"{instance.name}_bad_start"]
    expected_keys = {
        "fun",
        "jac",
        "start_x",
        "solution_x",
        "start_criterion",
        "solution_criterion",
        "lower_bounds",
        "upper_bounds",
        "info",
    }
    assert set(good) == expected_keys
    assert good["info"]["start"] == "good"
    assert bad["info"]["start"] == "bad"
    assert good["info"]["severity"] == instance.severity
    assert isinstance(good["info"]["other_local_optima"], bool)
    assert good["solution_criterion"] == bad["solution_criterion"]
    np.testing.assert_allclose(
        good["start_criterion"], good["fun"](good["start_x"]).sum()
    )
    np.testing.assert_array_equal(good["solution_x"], bad["solution_x"])


def test_get_likelihood_problems_without_unique_solution():
    problems = get_likelihood_problems("likelihood_small")
    problem = problems["normal_mixture_5_good_start"]
    assert problem["solution_x"] is None
    assert not problem["info"]["unique_solution"]
    assert np.isfinite(problem["solution_criterion"])


def test_get_likelihood_problems_with_bounds():
    problems = get_likelihood_problems("likelihood_small")
    problem = problems["garch_3_bad_start"]
    assert problem["info"]["has_bounds"]
    np.testing.assert_array_equal(problem["lower_bounds"], [1e-6, 0, 0])
    np.testing.assert_array_equal(problem["upper_bounds"], [np.inf, 1, 1])
