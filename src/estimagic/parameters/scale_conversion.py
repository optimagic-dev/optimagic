from typing import NamedTuple


def get_scale_converter(
    internal_params,
    tree_converter,
    params_converter,
    criterion,
    scaling_options,
):
    """Get parameter and derivative converter.


    Returns:
        ScaleConverter: NamedTuple with methods to convert between scaled and unscaled
            internal parameters and derivatives.
        FlatParams: NamedTuple of 1d numpy array with flat, internal and scaled
            parameters and bounds.

    """


class ScaleConverter(NamedTuple):
    params_to_internal: callable
    params_from_internal: callable
    derivative_to_internal: callable
