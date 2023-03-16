import pytest
from estimagic import get_benchmark_problems, minimize
from estimagic.visualization.visualize_tranquilo import visualize_tranquilo

cases = []
algo_options = {
    "random_hull": {
        "sampler": "random_hull",
        "subsolver": "gqtpar_fast",
        "sample_filter": "drop_pounders",
        "stopping.max_iterations": 10,
    },
    "optimal_hull": {
        "sampler": "optimal_hull",
        "subsolver": "gqtpar_fast",
        "sample_filter": "drop_pounders",
        "stopping.max_iterations": 10,
    },
}
for problem in ["rosenbrock_good_start", "watson_6_good_start"]:
    inputs = get_benchmark_problems("more_wild")[problem]["inputs"]
    criterion = inputs["criterion"]
    start_params = inputs["params"]
    for algo in ["tranquilo", "tranquilo_ls"]:
        results = {}
        for s, options in algo_options.items():
            results[s] = minimize(
                criterion=criterion,
                params=start_params,
                algorithm=algo,
                algo_options=options,
            )
        cases.append(results)


@pytest.mark.parametrize("results", cases)
def test_visualize_tranquilo(results):
    visualize_tranquilo(results, 5)
    for res in results.values():
        visualize_tranquilo(res, [1, 5])
