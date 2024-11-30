import numpy as np
from numpy.typing import NDArray

from optimagic.optimization.history import History


def get_convergence_report(history: History) -> dict[str, dict[str, float]] | None:
    is_accepted = history.is_accepted

    critvals = np.array(history.fun, dtype=np.float64)[is_accepted]
    params = np.array(history.flat_params, dtype=np.float64)[is_accepted]

    if len(critvals) < 2:
        out = None
    else:
        out = {}
        for name, n_entries in [("one_step", 2), ("five_steps", min(6, len(critvals)))]:
            relevant_critvals = critvals[-n_entries:]
            relevant_params = params[-n_entries:]

            max_f_rel, max_f_abs = _get_max_f_changes(relevant_critvals)
            max_x_rel, max_x_abs = _get_max_x_changes(relevant_params)

            col_dict = {
                "relative_criterion_change": max_f_rel,
                "relative_params_change": max_x_rel,
                "absolute_criterion_change": max_f_abs,
                "absolute_params_change": max_x_abs,
            }

            out[name] = col_dict

    return out


def _get_max_f_changes(critvals: NDArray[np.float64]) -> tuple[float, float]:
    best_val = critvals[-1]
    worst_val = critvals[0]

    max_change_abs = np.abs(best_val - worst_val)
    denom = max(np.abs(best_val), 0.1)

    max_change_rel = max_change_abs / denom

    return max_change_rel, max_change_abs


def _get_max_x_changes(params: NDArray[np.float64]) -> tuple[float, float]:
    best_x = params[-1]
    diffs = params - best_x
    denom = np.clip(np.abs(best_x), 0.1, np.inf)

    distances_abs = np.linalg.norm(diffs, axis=1)
    max_change_abs = distances_abs.max()

    scaled = diffs / denom

    distances_rel = np.linalg.norm(scaled, axis=1)
    max_change_rel = distances_rel.max()
    return max_change_rel, max_change_abs
