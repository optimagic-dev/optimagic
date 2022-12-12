import pytest
from estimagic import get_benchmark_problems
from estimagic import minimize
from estimagic.visualization.visualize_tranquilo import visualize_tranquilo

cases = []
for problem in ["rosenbrock_good_start", "watson_6_good_start"]:
    inputs = get_benchmark_problems("more_wild")[problem]["inputs"]
    criterion = inputs["criterion"]
    start_params = inputs["params"]
    for setting in [
        {
            "tranquilo": {
                "sampler": "sphere",
                "subsolver": "gqtpar_fast",
                "sample_filter": "drop_pounders",
                "stopping.max_iterations": 10,
            },
            "nag_pybobyqa": {"stopping.max_iterations": 10},
        },
        {
            "tranquilo_ls": {
                "sampler": "sphere",
                "subsolver": "gqtpar_fast",
                "sample_filter": "drop_pounders",
                "stopping.max_iterations": 10,
            },
            "nag_dfols": {"stopping.max_iterations": 10},
        },
    ]:
        results = {}
        for s, options in setting.items():
            results[s] = minimize(
                criterion=criterion,
                params=start_params,
                algorithm=s,
                algo_options=options,
            )
        cases.append(results)


@pytest.mark.parametrize("results", cases)
def test_visualize_tranquilo(results):
    visualize_tranquilo(results, 5)
    for res in results.values():
        visualize_tranquilo(res, [1, 5])
