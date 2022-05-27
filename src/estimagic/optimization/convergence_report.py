import numpy as np
from estimagic.optimization.history_tools import get_history_arrays
from scipy.spatial.distance import pdist as pairwise_distance


def get_convergence_report(history, direction, converter=None):

    history_arrs = get_history_arrays(
        history=history,
        direction=direction,
        converter=converter,
    )

    critvals = history_arrs["criterion"][history_arrs["is_accepted"]]
    params = history_arrs["params"][history_arrs["is_accepted"]]

    if len(critvals) < 2:
        out = None
    else:
        out = {}
        for name, n_entries in [("one_step", 2), ("five_steps", min(6, len(critvals)))]:
            relevant_critvals = critvals[-n_entries:]
            relevant_params = params[-n_entries:]

            max_f_abs = _get_max_f_abs(relevant_critvals)
            max_x_abs = _get_max_x_abs(relevant_params)
            max_f_rel = _get_max_f_rel(relevant_critvals, direction)
            max_x_rel = _get_max_x_rel(relevant_params, relevant_critvals, direction)

            col_dict = {
                "relative_criterion_change": max_f_rel,
                "relative_params_change": max_x_rel,
                "absolute_criterion_change": max_f_abs,
                "absolute_params_change": max_x_abs,
            }

            out[name] = col_dict

    return out


def _get_max_f_abs(critvals):
    max_change = critvals.max() - critvals.min()
    return max_change


def _get_max_f_rel(critvals, direction):
    max_val = critvals.max()
    min_val = critvals.min()

    diff = max_val - min_val

    denom = max_val if direction == "maximize" else min_val
    denom = np.clip(np.abs(denom), 0.1, np.inf)
    max_change = diff / denom
    return max_change


def _get_max_x_abs(params):
    distances = pairwise_distance(params)
    max_change = distances.max()
    return max_change


def _get_max_x_rel(params, critvals, direction):
    best_index = np.argmax(critvals) if direction == "maximize" else np.argmin(critvals)
    denom = np.clip(np.abs(params[best_index]), 0.1, np.inf)
    scaled = params / denom

    distances = pairwise_distance(scaled)
    max_change = distances.max()
    return max_change
