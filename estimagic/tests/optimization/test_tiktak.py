"""
Test the TikTak optimization algorithm.

This file performs two tests.
First, it allows TikTak to automatically sample a set of Sobol points.
Second, it provides TikTak with a custom set of starting points to test
the custom_sample feature.
"""
import chaospy
import numpy as np
import pandas as pd

from estimagic.optimization.tiktak import minimize_tik_tak


# define a criterion function for testing TikTak
def griewank(params):
    column = params["value"]
    array = column.to_numpy()
    x = np.transpose(array)
    d = x.shape[0]
    i = np.arange(1, d + 1)
    res = 2 + np.sum(x ** 2 / 200) - np.prod(np.cos(x / np.sqrt(i)))
    return res


domain = (-100, 100)  # domain of the Griewank function
true_min = 1  # the real min. value of the Griewank function
n = 10  # the dimension of our problem

######################################
# First, Test of Automatic Sampling
#####################################

lower_bounds = [domain[0] for i in range(n)]  # lower bounds on each dimension
upper_bounds = [domain[1] for i in range(n)]  # upper bounds on each dimension
bounds_dict = {"lower_bounds": lower_bounds, "upper_bounds": upper_bounds}

# define all the basic arguments for our optimization
criterion = griewank
bounds = pd.DataFrame(bounds_dict)
local_search_algorithm = "scipy_neldermead"
num_points = 100
num_restarts = 10
algo_options = {
    "convergence.absolute_criterion_tolerance": 1e-3,
    "convergence.absolute_params_tolerance": 1e-3,
}
logging = False


# run the algorithm
solution = minimize_tik_tak(
    criterion=criterion,
    bounds=bounds,
    local_search_algorithm=local_search_algorithm,
    num_points=num_points,
    num_restarts=num_restarts,
    algo_options=algo_options,
)

x = solution["solution_x"]
value = solution["solution_criterion"]
n_criterion_evals = solution["n_criterion_evaluations"]

print(f"The solution is {x}")
print(f"The criterion value at the min is {value}")
print(f"Number of function evaluations was {n_criterion_evals}")


#############################################
# Second, Test of User-Provided Custom Sample
#############################################

# define a dataframe of custom starting points
# first, generate a set of random points points
distribution = chaospy.Iid(chaospy.Uniform(0, 1), n)
xstarts = distribution.sample(num_points, rule="random")

# generate array versions of the bounds
lower_array = bounds["lower_bounds"].to_numpy()
upper_array = bounds["upper_bounds"].to_numpy()

# spread out the sample within the bounds
xstarts = (
    lower_array[:, np.newaxis] + (upper_array - lower_array)[:, np.newaxis] * xstarts
)

# define the custom starting points as a dataframe
custom_sample = pd.DataFrame(data=xstarts, index=[f"x_{i}" for i in range(10)])


# run the algorithm again with custom starting points
solution = minimize_tik_tak(
    criterion=criterion,
    local_search_algorithm=local_search_algorithm,
    num_restarts=num_restarts,
    sampling="custom",
    custom_sample=custom_sample,
    algo_options=algo_options,
)

x = solution["solution_x"]
value = solution["solution_criterion"]
n_criterion_evals = solution["n_criterion_evaluations"]

print(f"The solution is {x}")
print(f"The criterion value at the min is {value}")
print(f"Number of function evaluations was {n_criterion_evals}")
