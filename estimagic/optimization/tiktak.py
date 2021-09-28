import chaospy
import numpy as np
import pandas as pd

from estimagic.optimization.optimize import minimize


def minimize_tik_tak(
    criterion,
    local_search_algorithm,
    num_restarts,
    sampling=None,
    bounds=None,
    num_points=None,
    custom_sample=None,
    mixing_weight=None,
    algo_options=None,
    logging=False,
):
    """
    Minimize a function using the TikTak algorithm.

    TikTak (`Arnoud, Guvenen, and Kleineberg
    <https://www.nber.org/system/files/working_papers/w26340/w26340.pdf>`_)
    is an algorithm for solving global optimization problems. It performs local
    searches from a set of carefully-selected points in the parameter space.

    First implemented in Python by Alisdair McKay
    (`GitHub Repository <https://github.com/amckay/TikTak>`_)

    Args:
        criterion (Callable): A function that takes a pandas DataFrame (see
            :ref:`params`) as first argument and returns a scalar floating point.
        local_search_algorithm (str or callable): Specifies the optimization algorithm.
            For supported algorithms this is a string with the name of the algorithm.
            Otherwise it can be a callable with the estimagic algorithm interface. See
            :ref:`algorithms`.
        num_restarts (int): the number of initial sample points from which to perform
            local optimization. If automatically generating sample points, this value
            must be smaller than num_points. If providing a custom set of sample points,
            this value must be smaller than the number of columns in your custom_sample
            dataframe (see below)
        sampling (str): specifies the procedure for random or quasi-random sampling.
            See chaospy documentation of  an overview of possible rules:
            https://chaospy.readthedocs.io/
            In addition to chaospy's sampling rules, we also allow users to input a
            custom set of starting points as a datfarame. In that case, set this
            argument to "custom".
        bounds (pandas.DataFrame): A DataFrame with one column called "lower_bounds" and
            another called "upper_bounds," with an entry in each column for every
            dimension of the optimization problem. Not required if the user provides a
            custom sample.
        num_points (int): the number of initial points to sample in the parameter space.
            Not required if the user provides a custom sample.
        custom_sample (pandas.DataFrame): A dataframe of custom starting points. Set
            sampling to "custom" if you plan to use this argument. Each column of the
            dataframe should be a distinct point in the sample. Each row of the
            dataframe should be a parameter of the point. So if you want to pass a
            sample of 100 10-dimensional starting points, your dataframe should have 10
            rows and 100 columns. If you do not specify this argument, you must specify
             bounds and num_points so that TikTak can automatically generate a sample.
        mixing_weight (callable): As TikTak performs local optimizations on a set
            of sample points, the algorithm computes a convex combination of each
            successive point and the "best" point sampled yet. Users may supply their
            own functions for computing this convex combination. This user-supplied
            function must take only one argument ``i`` (integer), where ``i`` is the
            number of points sampled so far out of the total ``num_restarts`` supplied
             above. The function must return a float between 0 and 1. This output
             `theta` will serve as the "weight" assigned to the best point so far,
            compared to the current point in the sampling process. By default, we
            implement the mixing weight formula described by Arnoud, Guvenen and
            Kleinenberg in the paper linked above.
        algo_options (dict): Algorithm specific configuration of the local optimization.
            See :ref:`list_of_algorithms` for supported options of each algorithm.

    """

    def df_wrapper(array):
        df = pd.DataFrame(
            data=array, columns=["value"], index=[f"x_{i}" for i in range(len(array))]
        )
        return df

    # Users must either provide a custom sample of points,
    # or all the information necessary for TikTak to sample points.

    # If the user provide custom sample points, use them instead of sampling manually
    if sampling == "custom":

        xstarts = custom_sample.to_numpy().transpose()

    # If the user provided sufficient information, perform the sampling process
    elif (bounds is not None) & (num_points is not None):
        lower_bounds = bounds["lower_bounds"]
        lower_bounds = lower_bounds.to_numpy()
        lower_bounds = lower_bounds.transpose()

        upper_bounds = bounds["upper_bounds"]
        upper_bounds = upper_bounds.to_numpy()
        upper_bounds = upper_bounds.transpose()

        nparam = len(lower_bounds)

        # the default sampling method depends on nparam
        if sampling is None:
            if nparam <= 15:
                sampling = "sobol"
            else:
                sampling = "random"

        # start the global search for initial points
        distribution = chaospy.Iid(chaospy.Uniform(0, 1), nparam)
        xstarts = distribution.sample(
            num_points, rule=sampling
        )  # generate a sample based on the sampling rule
        xstarts = np.transpose(xstarts)  # transpose the array of samples
        xstarts = (
            lower_bounds[np.newaxis, :]
            + (upper_bounds - lower_bounds)[np.newaxis, :] * xstarts
        )  # spread out the sample within the bounds

    # If the user did not provide custom starting points and did not provide
    # enough information for the function to sample, raise an error.
    else:
        print(
            "ERROR: User did not provide enough information to sample starting points"
        )

    # --- evaluate the criterion function on each starting point ----
    dfxstarts = [df_wrapper(point) for point in xstarts]

    y = np.array([criterion(point) for point in dfxstarts])
    # sort the results
    sorting_indices = np.argsort(y)
    xstarts = xstarts[sorting_indices]
    y = y[sorting_indices]

    # take the best Imax
    xstarts = list(xstarts[:num_restarts])

    # define some parameters for the local searches
    best_so_far = {"solution_x": 0, "solution_criterion": 1e10}
    result_trackers = []
    num_func_evals = 0

    for i in range(num_restarts):
        new_task = xstarts.pop()

        # compute the convex combination of this sample point and the "best" so far
        if (
            mixing_weight is None
        ):  # by default, we implement the formula supplied by Arnoud, Guvenen, and
            # Kleinenberg
            term = (i / num_restarts) ** (1 / 2)
            max_term = max([0.1, term])
            theta = min([max_term, 0.995])

        else:  # otherwise, users have supplied their own function
            theta = mixing_weight(i)

        new_task = theta * best_so_far["solution_x"] + (1 - theta) * new_task

        # params dataframe
        params = df_wrapper(new_task)

        result = minimize(
            criterion=criterion,
            params=params,
            algorithm=local_search_algorithm,
            algo_options=algo_options,
            logging=logging,
        )

        # if the new x returns the lowest function val yet, it's the best so far
        if result["solution_criterion"] < best_so_far["solution_criterion"]:
            print("best so far %f" % result["solution_criterion"])
            best_so_far = result

        result_trackers.append(result)
        num_func_evals += result["n_criterion_evaluations"]

    res = {
        "solution_x": best_so_far["solution_x"],
        "solution_criterion": best_so_far["solution_criterion"],
        "n_criterion_evaluations": num_func_evals,
    }

    return res
