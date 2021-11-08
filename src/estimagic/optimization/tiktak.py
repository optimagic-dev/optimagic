"""Functions for multi start optimization a la TikTak.

To-Do:

- Make sampling compatible with constraints
    - Write a get_approximate_internal_bounds function
    - Sample on internal parameters space
    - Do not do any bounds checking for fixed parameters

- Improve Error handling
    - Go over all error messages and provide information on the exact problem
      (e.g. section of params where bounds are missing, ...)

"""
import chaospy
import numpy as np
import pandas as pd
from chaospy.distributions import Triangle
from chaospy.distributions import Uniform
from estimagic.optimization.optimize import minimize
from estimagic.parameters.parameter_conversion import get_internal_bounds
from estimagic.parameters.parameter_conversion import get_reparametrize_functions


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


def get_exploration_sample(
    params,
    n_samples=None,
    sampling_distribution="uniform",
    sampling_method=None,
    seed=None,
    constraints=None,
):
    """Get a sample of parameter values for the first stage of the tiktak algorithm.

    The sample is created randomly or using low a low discrepancy sequence. Different
    distributions are available.

    Args:
        params (pandas.DataFrame): see :ref:`params`.
        n_samples (int, pandas.DataFrame or numpy.ndarray): Number of sampled points on
            which to do one function evaluation. Default is 10 * n_params.
            Alternatively, a DataFrame or numpy array with an existing sample.
        sampling_distribution (str): One of "uniform", "triangle". Default is
            "uniform"  as in the original tiktak algorithm.
        sampling_method (str): One of "random", "sobol", "halton",
            "hammersley", "korobov", "latin_hypercube" and "chebyshev" or a numpy array
            or DataFrame with custom points. Default is sobol for problems with up to 30
            parameters and random for problems with more than 30 parameters.
        seed (int): Random seed.
        constraints (list): See :ref:`constraints`.

    Returns:
        np.ndarray: Numpy array of shape n_samples, len(params). Each row is a vector
            of parameter values.

    """
    if n_samples is None:
        n_samples = 10 * len(params)

    if sampling_method is None:
        sampling_method = "sobol" if len(params) <= 30 else "random"

    if isinstance(n_samples, (np.ndarray, pd.DataFrame)):
        sample = _process_sample(n_samples, params, constraints)
    elif isinstance(n_samples, (int, float)):
        sample = _create_sample(
            params=params,
            n_samples=n_samples,
            sampling_distribution=sampling_distribution,
            sampling_method=sampling_method,
            seed=seed,
            constraints=constraints,
        )
    else:
        raise TypeError(f"Invalid type for n_samples: {type(n_samples)}")
    return sample


def _process_sample(raw_sample, params, constraints):
    if isinstance(raw_sample, pd.DataFrame):
        if not raw_sample.columns.equals(params.index):
            raise ValueError(
                "If you provide a custom sample as DataFrame the columns of that "
                "DataFrame and the index of params must be equal."
            )
        sample = raw_sample[params.index].to_numpy()
    elif isinstance(raw_sample, np.ndarray):
        _, n_params = raw_sample.shape
        if n_params != len(params):
            raise ValueError(
                "If you provide a custom sample as a numpy array it must have as many "
                "columns as parameters."
            )
        sample = raw_sample

    to_internal, _ = get_reparametrize_functions(params, constraints)

    sample = np.array([to_internal(x) for x in sample])

    return sample


def _create_sample(
    params,
    n_samples,
    sampling_distribution,
    sampling_method,
    seed,
    constraints,
):
    if _has_transforming_constraints(constraints):
        raise NotImplementedError(
            "Multistart optimization is not yet compatible with transforming "
            "Constraints that require a transformation of parameters such as "
            "linear, probability, covariance and sdcorr constranits."
        )

    lower, upper = _get_internal_sampling_bounds(params, constraints)

    sample = _do_actual_sampling(
        midpoint=params["value"].to_numpy(),
        lower=lower,
        upper=upper,
        size=n_samples,
        distribution=sampling_distribution,
        rule=sampling_method,
        seed=seed,
    )

    return sample


def _do_actual_sampling(midpoint, lower, upper, size, distribution, rule, seed):

    valid_rules = [
        "random",
        "sobol",
        "halton",
        "hammersley",
        "korobov",
        "latin_hypercube",
    ]

    if rule not in valid_rules:
        raise ValueError(f"Invalid rule: {rule}. Must be one of\n\n{valid_rules}\n\n")

    if distribution == "uniform":
        dist_list = [Uniform(lb, ub) for lb, ub in zip(lower, upper)]
    elif distribution == "triangle":
        dist_list = [Triangle(lb, mp, ub) for lb, mp, ub in zip(lower, midpoint, upper)]
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

    joint_distribution = chaospy.J(*dist_list)

    np.random.seed(seed)

    sample = joint_distribution.sample(
        size=size,
        rule=rule,
    ).T
    return sample


def _get_internal_sampling_bounds(params, constraints):
    params = params.copy(deep=True)
    params["lower_bound"] = _extract_external_sampling_bound(params, "lower")
    params["upper_bound"] = _extract_external_sampling_bound(params, "upper")

    problematic = params.query("lower_bound >= upper_bound")

    if len(problematic):
        raise ValueError(
            "Lower bound must be smaller than upper bound for all parameters. "
            f"This is violated for:\n\n{problematic.to_string()}\n\n"
        )

    lower, upper = get_internal_bounds(params=params, constraints=constraints)

    for b in lower, upper:
        if not np.isfinite(b).all():
            raise ValueError(
                "Sampling bounds of all free parameters must be finite to create a "
                "parameter sample for multistart optimization."
            )

    return lower, upper


def _extract_external_sampling_bound(params, bounds_type):
    soft_name = f"soft_{bounds_type}_bound"
    hard_name = f"{bounds_type}_bound"
    if soft_name in params:
        bounds = params[soft_name]
    elif hard_name in params:
        bounds = params[hard_name]
    else:
        raise ValueError(
            f"{soft_name} or {hard_name} must be in params to sample start values."
        )

    return bounds


def _has_transforming_constraints(constraints):
    constraints = [] if constraints is None else constraints
    transforming_types = {
        "linear",
        "probability",
        "covariance",
        "sdcorr",
        "increasing",
        "decreasing",
        "sum",
    }
    present_types = {constr["type"] for constr in constraints}
    return bool(transforming_types.intersection(present_types))


def run_explorations(func, params, sample, batch_evaluator, n_cores):
    """Do the function evaluations for the exploration phase.

    Args:
        func (callable): An already partialled version of
            `internal_criterion_and_derivative_template` where the following arguments
            are still free: ``x``, ``task``, ``algorithm_info``, ``error_handling``,
            ``error_penalty``.
        params (pandas.DataFrame): See :ref:`params`.
        sample (numpy.ndarray): 2d numpy array where each row is a sampled internal
            parameter vector.
        batch_evaluator (str or callable): See :ref:`batch_evaluators`.
        n_cores (int): Number of cores.

    Returns:
        dict: A dictionary with the the following entries:
            "sorted_values": 1d numpy array with sorted function values. Invalid
                function values are excluded.
            "sorted_sample": 2d numpy array with corresponding internal parameter
                vectors.
            "contributions": None or 2d numpy array with the contributions entries of
                the function evaluations.
            "root_contributions": None or 2d numpy array with the root_contributions
                entries of the function evaluations.

    """
    # partial in remaining arguments of internal criterion
    # do function evaluations
    # process outputs
    pass


def get_batched_optimization_sample(sorted_sample, n_optimizations, batch_size):
    """Create a batched sample of internal parameters for the optimization phase.

    Note that in the end the optimizations will not be started from those parameter
    vectors but from a convex combination of that parameter vector and the
    best parameter vector at the time when the optimization is started.

    Args:
        sorted_sample (np.ndarray): 2d numpy array with containing sorted internal
            parameter vectors.
        n_optimizations (int): Number of optimizations to run. If sample is shorter
            than that, optimizations are run on all entries of the sample.
        batch_size (int): Batch size.

    Returns:
        list: Nested list of parameter vectors from which an optimization is run.
            The inner lists have length ``batch_size`` or shorter.

    """
    pass


def run_local_optimizations(
    sample,
    n_cores,
    batch_evaluator,
    optimize_options,
    mixing_weight_method,
    mixing_weight_bounds,
    convergence_relative_params_tolerance,
):
    """Run the actual local optimizations until convergence.

    Args:
        sample (list): Nested list of parameter vectors from which an optimization is
            run. The inner lists have length ``batch_size`` or shorter.
        n_cores (int): Number of cores.
        batch_evaluator (str or callable): See :ref:`batch_evaluators`.
        optimize_options (dict): Keyword arguments for the optimizations that are fixed
            across all optimizations.
        mixing_weight_method (str or callable): Specifies how much weight is put on the
            currently best point when calculating a new starting point for a local
            optimization out of the currently best point and the next random starting
            point. Either "tiktak" or a callable that takes the arguments ``iteration``,
            ``n_iterations``, ``min_weight``, ``max_weight``. Default "tiktak".
        mixing_weight_bounds (tuple): A tuple consisting of a lower and upper bound on
            mixing weights. Default (0.1, 0.995).
        convergence.relative_params_tolerance (float): If the maximum relative
            difference between the results of two consecutive local optimizations is
            smaller than the multistart optimization converges. Default 0.01. Note that
            this is independent of a convergence criterion with the same name for each
            local optimization.

    Returns:
        dict: A Dictionary containing the best parameters and criterion values,
            convergence information and the history of optimization solutions.

    """
    # process batch evaluator
    # process mixing weight function and options
    # partial optimize options into optimize
    # implement loop with convergence check and parallel optimizations on the inside.
    # process results to something compatible with output of optimize but with
    # additional history entries.
