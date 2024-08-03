from typing import Callable, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from optimagic.exceptions import InvalidMultistartError
from optimagic.typing import PyTree


def process_multistart_sample(
    raw_sample: Sequence[PyTree] | pd.DataFrame | NDArray[np.float64],
    params: PyTree,
    params_to_internal: Callable[[PyTree], NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Process a user provided multistart sample.

    Args:
        raw_sample: A user provided sample of external start parameters.
        params: User provided start parameters.
        params_to_internal: A converter from external parameters to internal parameters.

    Returns:
        np.ndarray: 2d array where each row is an internal parameter vector.

    Raises:
        InvalidMultistartError: If the user provided multistart sample is invalid.

    """
    is_df_params = isinstance(params, pd.DataFrame) and "value" in params
    is_np_params = isinstance(params, np.ndarray) and params.ndim == 1

    if isinstance(raw_sample, pd.DataFrame):
        if not is_df_params:
            msg = (
                "User provided multistart samples can only be a DataFrame if "
                "params is a DataFrame with 'value' column."
            )
            raise InvalidMultistartError(msg)
        elif not raw_sample.columns.equals(params.index):
            msg = (
                "If you provide a custom sample as DataFrame the columns of that "
                "DataFrame and the index of params must be equal."
            )
            raise InvalidMultistartError(msg)

        list_sample = [params.assign(value=row) for _, row in raw_sample.iterrows()]

    elif isinstance(raw_sample, np.ndarray):
        if not is_np_params:
            msg = (
                "User provided multistart samples can only be a numpy array if params "
                "is a 1d numpy array."
            )
            raise InvalidMultistartError(msg)
        elif raw_sample.ndim != 2:
            msg = (
                "If user provided multistart samples are a numpy array, the array "
                "must be two dimensional."
            )
            raise InvalidMultistartError(msg)
        elif raw_sample.shape[1] != len(params):
            msg = (
                "If user provided multistart samples are a numpy array, the number of "
                "columns must be equal to the number of parameters."
            )
            raise InvalidMultistartError(msg)

        list_sample = list(raw_sample)

    elif not isinstance(raw_sample, Sequence):
        msg = (
            "User provided multistart samples must be a Sequence of PyTrees, a "
            "pandas DataFrame or a numpy array."
        )
        raise InvalidMultistartError(msg)
    else:
        list_sample = list(raw_sample)

    return np.array([params_to_internal(x) for x in list_sample])
