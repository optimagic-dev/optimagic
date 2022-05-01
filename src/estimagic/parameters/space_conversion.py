from typing import NamedTuple


def get_space_converter(
    flat_params,
    flat_constraints,
):
    """Get functions to convert between in-/external space of params and derivatives.


    Args:
        params ()


    Returns:
        SpaceConverter
        FlatParams: NamedTuple of 1d numpy array with flat and internal params and
            bounds.

    """
    pass


class SpaceConverter(NamedTuple):
    params_to_internal: callable
    params_from_internal: callable
    derivative_to_internal: callable
    has_transforming_constraints: bool
