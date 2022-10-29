"""Implement simopt optimizers."""
from estimagic.config import IS_SIMOPT_INSTALLED
from estimagic.decorators import mark_minimizer

try:
    import simopt as so  # noqa: F401
except ImportError:
    pass


@mark_minimizer(
    name="simopt_adam",
    primary_criterion_entry="value",
    needs_scaling=True,
    is_available=IS_SIMOPT_INSTALLED,
)
def simopt_adam():
    pass


@mark_minimizer(
    name="simopt_aloe",
    primary_criterion_entry="value",
    needs_scaling=True,
    is_available=IS_SIMOPT_INSTALLED,
)
def simopt_aloe():
    pass


@mark_minimizer(
    name="simopt_astrodf",
    primary_criterion_entry="value",
    needs_scaling=True,
    is_available=IS_SIMOPT_INSTALLED,
)
def simopt_astrodf():
    pass


@mark_minimizer(
    name="simopt_neldmd",
    primary_criterion_entry="value",
    needs_scaling=True,
    is_available=IS_SIMOPT_INSTALLED,
)
def simopt_neldmd():
    pass


@mark_minimizer(
    name="simopt_randomsearch",
    primary_criterion_entry="value",
    needs_scaling=True,
    is_available=IS_SIMOPT_INSTALLED,
)
def simopt_randomsearch(
    criterion,
    x,
    lower_bounds,
    upper_bounds,
    *,
    population_size=None,
    batch_evaluator=None,
    n_cores=1,
    seed=None,
    discard_start_params=False,
):
    pass


@mark_minimizer(
    name="simopt_spsa",
    primary_criterion_entry="value",
    needs_scaling=True,
    is_available=IS_SIMOPT_INSTALLED,
)
def simopt_spsa():
    pass


@mark_minimizer(
    name="simopt_strong",
    primary_criterion_entry="value",
    needs_scaling=True,
    is_available=IS_SIMOPT_INSTALLED,
)
def simopt_strong():
    pass
