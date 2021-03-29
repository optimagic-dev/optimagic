"""
Test the TikTak optimization algorithm
"""
import numpy as np
import pandas as pd

from estimagic.optimization.tiktak import TikTakOptimize


# define a criterion function for testing TikTak
def Griewank(params):
    column = params["value"]
    array = column.to_numpy()
    X = np.transpose(array)
    d = X.shape[0]
    i = np.arange(1, d + 1)
    res = 2 + np.sum(X ** 2 / 200) - np.prod(np.cos(X / np.sqrt(i)))
    return res


domain = (-100, 100)  # domain of the Griewank function
true_min = 1  # the real min. value of the Griewank function
n = 10  # the dimension of our problem

lower_bounds = [domain[0] for i in range(n)]  # lower bounds on each dimension
upper_bounds = [domain[1] for i in range(n)]  # upper bounds on each dimension
bounds_dict = {"lower_bounds": lower_bounds, "upper_bounds": upper_bounds}

## define all the arguments for our optimization
criterion = Griewank
bounds = pd.DataFrame(bounds_dict)
local_search_algorithm = "scipy_neldermead"
num_points = 1000
num_restarts = 100
shrink_after = 30
algo_options = {"convergence_absolute_criterion_tolerance": 1e-8}
logging = False

# run the algorithm
solution = TikTakOptimize(
    criterion=criterion,
    bounds=bounds,
    local_search_algorithm=local_search_algorithm,
    num_points=num_points,
    num_restarts=num_restarts,
    shrink_after=shrink_after,
    algo_options=algo_options,
)

x = solution["solution_x"]
value = solution["solution_criterion"]
n_criterion_evals = solution["n_criterion_evaluations"]

print(f"The solution is {x}")
print(f"The criterion value at the min is {value}")
print(f"Number of function evaluations was {n_criterion_evals}")
