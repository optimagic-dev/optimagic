from collections import Callable

# the _get_num_optimizations_ ... functions are replacements for the len_list
# functions. len_list .. was not a very good name.

# the is_scalar is meant to handle the case of lists of length one, but maybe
# there is a better solution

# the name argument is just there for better error messages. This will be very
# helpful for users.

# I import Callable because for isinstance(criterion, Callable). I had to google that
# so explain it here.


def process_optimization_arguments(
    criterion,
    params,
    algorithm,
    criterion_args=None,
    criterion_kwargs=None,
    constraints=None,
    general_options=None,
    algo_options=None,
    dashboard=False,
    db_options=None,
):
    """Process and validate arguments for minimize or maximize."""

    criterion_args = [] if criterion_args is None else criterion_args
    criterion_kwargs = {} if criterion_kwargs is None else criterion_kwargs
    constraints = [] if constraints is None else constraints
    algo_options = {} if algo_options is None else algo_options
    db_options = {} if db_options is None else db_options

    nested_args = []

    # ...


def _get_num_optimizations_from_list_argument(candidate, name):
    msg = f"{name} has to be a list/tuple or a nested list/tuple."
    is_scalar = True
    if not isinstance(candidate, (list, tuple)):
        raise ValueError(msg)
    elif candidate in ([], ()):
        num_opt = 1
    elif isinstance(candidate[0], (list, tuple)):
        num_opt = len(candidate)
        for c in candidate:
            assert isinstance(c, (list, tuple)), msg
        is_scalar = False
        else:
            num_opt = 1
    return num_opt, is_scalar


def _get_num_optimizations_from_nonlist_argument(candidate, scalar_type, name):
    """Return 1 if candidate is no list and the length of the list otherwise."""

    msg = f"{name} has to be a {scalar_type} or list of {scalar_type}s."
    is_scalar = False
    if isinstance(candidate, scalar_type):
        num_opt = 1
        is_scalar = True
    elif isinstance(candidate, (list, tuple)):
        num_opt = len(candidate)
    else:
        raise TypeError(msg)
    return num_opt, is_scalar


