import estimagic as em
from estimagic.benchmarking.get_benchmark_problems import get_benchmark_problems


def test_gqtpar_lambdas():
    algo_options = {
        "disable_convergence": True,
        "stopping_max_iterations": 30,
        "sample_filter": "keep_all",
        "sampler": "random_hull",
        "subsolver_options": {"k_hard": 0.001, "k_easy": 0.001},
    }
    problem_info = get_benchmark_problems("more_wild")["freudenstein_roth_good_start"]

    em.minimize(
        criterion=problem_info["inputs"]["criterion"],
        params=problem_info["inputs"]["params"],
        algo_options=algo_options,
        algorithm="tranquilo",
    )
