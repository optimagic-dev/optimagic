"""Functions to create, run and visualize optimization benchmarks.

TO-DO:
- Come up with a good specification for noise_options and implement adding noise.
- Add other benchmark sets:
    - medium scale problems from https://arxiv.org/pdf/1710.11005.pdf, Page 34.
    - scalar problems from https://github.com/AxelThevenot
- Think about a good way for handling seeds. Maybe this should be part of the noise
    options or only part of run_benchmark. Needs to be possible to have fixed noise
    and random noise. Maybe distinguish fixed noise by differentiable and
    non differentiable noise.
- Need to think about logging. We probably do not want to use databases for speed
    and disk cluttering reasons but we do want a full history. Maybe fast_logging?
- Instead of one plot_benchmark function we probably want one plotting function for each
    plot type. Inspiration:
    - https://arxiv.org/pdf/1710.11005.pdf
    - https://www.mcs.anl.gov/~more/dfo/

"""


def get_problems(name, noise_options=None, add_bounds=False):
    """Get a dictionary of test problems for a benchmark.

    Args:
        name (str): The name of the set of test problems. Currently "more_wild_ls"
            is the only supported one.
        noise_options (dict or None): Specficies the type of noise to add to the test
            problems. Has the entries:
            - type (str): "multiplicative" or "additive"
            - ...
        add_bounds (bool): If True, all problems get finite lower and upper bounds on
            all parameters.

    Returns:
        dict: Nested dictionary with benchmark problems of the structure:
            {"name": {"inputs": {...}, "solution": {...}, "info": {...}}}
            where "inputs" are keyword arguments for ``minimize`` such as the criterion
            function and start parameters. "solution" contains the entries "params" and
            "value" and "info" might  contain information about the test problem.

    """
    if name not in ["more_wild"]:
        raise NotImplementedError()

    if noise_options is not None:
        raise NotImplementedError()

    if add_bounds:
        raise NotImplementedError()


def run_benchmark(
    problems,
    optimize_options,
    logging_directory,
    batch_evaluator="joblib",
    n_cores=1,
    error_handling="continue",
    seed=None,
):
    """Run problems with different optimize options.

    Args:
        problems (dict): Nested dictionary with benchmark problems of the structure:
            {"name": {"inputs": {...}, "solution": {...}, "info": {...}}}
            where "inputs" are keyword arguments for ``minimize`` such as the criterion
            function and start parameters. "solution" contains the entries "params" and
            "value" and "info" might  contain information about the test problem.
        optimize_options: Nested dictionary that maps a name to a set of keyword
            arguments for ``minimize``.
            batch_evaluator (str or callable): See :ref:`batch_evaluators`.
        logging_directory (pathlib.Path): Directory in which the log databases are
            saved.
        n_cores (int): Number of optimizations that is run in parallel. Note that in
            addition to that an optimizer might parallelize.
        error_handling (str): One of "raise", "continue".

    Returns:
        dict: Nested Dictionary with information on the benchmark run. The outer keys
            are tuples where the first entry is the name of the problem and the second
            the name of the optimize options. The values are dicts with the entries:
            "walltime", "params_history", "criterion_history", "solution"

    """
    pass


def plot_benchmark(results, problems, optimize_options):
    pass
