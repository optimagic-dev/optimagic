import numpy as np


def get_infinity_handler(infinity_handler):
    if isinstance(infinity_handler, str):
        built_in_handlers = {"relative": clip_relative}
        infinity_handler = built_in_handlers[infinity_handler]
    elif not callable(infinity_handler):
        raise TypeError("infinity_handler must be a string or callable.")

    return infinity_handler


def clip_relative(fvecs):
    """Clip infinities at a value that is relative to worst finite value.

    Args:
        fvecs (np.ndarray): 2d numpy array of shape n_samples, n_residuals.


    Returns:
        np.ndarray: Array of same shape as fvecs with finite values.

    """
    _mask = np.isfinite(fvecs)

    _mins = np.min(fvecs, axis=0, where=_mask, initial=1e300)
    _maxs = np.max(fvecs, axis=0, where=_mask, initial=-1e300)

    # abs is necessary because if all values are infinite, the diffs can switch sign
    # due to the initial value in the masked min and max
    _diff = _maxs - _mins

    # Due to the initial value of the masked min and max, the sign of the diff can
    # be negative if all values are infinite. In that case we want to switch the
    # signe of _diff, _mins and _maxs.
    _signs = np.sign(_diff)
    _diff *= _signs
    _maxs *= _signs
    _mins *= _signs

    _pos_penalty = _maxs + 2 * _diff + 1
    _neg_penalty = _mins - 2 * _diff - 1

    out = np.nan_to_num(
        fvecs, nan=_pos_penalty, posinf=_pos_penalty, neginf=_neg_penalty
    )

    return out
