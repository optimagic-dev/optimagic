import numpy as np
import pandas as pd


def process_multistart_sample(raw_sample, params, params_to_internal):
    """Process a user provided multistart sample.

    Args:
        raw_sample (list, pd.DataFrame or np.ndarray): A user provided sample of
            external start parameters.
        params (pytree): User provided start parameters.
        params_to_internal (callable): A function that converts external parameters
            to internal ones.


    Returns:
        np.ndarray: 2d numpy array where each row is an internal parameter vector.


    """
    is_df_params = isinstance(params, pd.DataFrame) and "value" in params
    is_np_params = isinstance(params, np.ndarray) and params.ndim == 1

    if isinstance(raw_sample, pd.DataFrame):
        if not is_df_params:
            msg = (
                "User provided multistart samples can only be a DataFrame if "
                "params is a DataFrame with 'value' column."
            )
            raise ValueError(msg)
        elif not raw_sample.columns.equals(params.index):
            msg = (
                "If you provide a custom sample as DataFrame the columns of that "
                "DataFrame and the index of params must be equal."
            )
            raise ValueError(msg)

        list_sample = []
        for _, row in raw_sample.iterrows():
            list_sample.append(params.assign(value=row))

    elif isinstance(raw_sample, np.ndarray):
        if not is_np_params:
            msg = (
                "User provided multistart samples can only be a numpy array if params "
                "is a 1d numpy array."
            )
            raise ValueError(msg)
        elif raw_sample.ndim != 2:
            msg = (
                "If user provided multistart samples are a numpy array, the array "
                "must be two dimensional."
            )
            raise ValueError(msg)
        elif raw_sample.shape[1] != len(params):
            msg = (
                "If user provided multistart samples are a numpy array, the number of "
                "columns must be equal to the number of parameters."
            )
            raise ValueError(msg)

        list_sample = list(raw_sample)

    elif not isinstance(raw_sample, (list, tuple)):
        msg = (
            "User provided multistart samples must be a list, tuple, numpy array or "
            "DataFrame."
        )
        raise ValueError(msg)
    else:
        list_sample = list(raw_sample)

    sample = np.array([params_to_internal(x) for x in list_sample])

    return sample
