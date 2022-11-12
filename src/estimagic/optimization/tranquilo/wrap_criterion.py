import functools

import numpy as np
from estimagic.batch_evaluators import process_batch_evaluator


def get_wrapped_criterion(criterion, batch_evaluator, n_cores, history):
    batch_evaluator = process_batch_evaluator(batch_evaluator)

    @functools.wraps(criterion)
    def wrapper_criterion(params):
        if np.array(params).size == 0:
            return (np.array([]), np.array([]), np.array([]).astype(int))

        _is_just_one = np.array(params).ndim == 1

        _parlist = list(np.atleast_2d(params))

        _effective_n_cores = min(n_cores, len(_parlist))

        _raw_evals = batch_evaluator(
            criterion,
            arguments=_parlist,
            n_cores=_effective_n_cores,
        )

        # replace NaNs but keep infinite values. NaNs would be problematic in many
        # places, infs are only a problem in the model fitting and will thus be handled
        # there
        _clipped_evals = [
            np.nan_to_num(critval, nan=np.inf, posinf=np.inf, neginf=-np.inf)
            for critval in _raw_evals
        ]

        indices = np.arange(history.get_n_fun(), history.get_n_fun() + len(_parlist))
        history.add_entries(_parlist, _clipped_evals)
        fvecs = history.get_fvecs(indices)
        fvals = history.get_fvals(indices)

        if _is_just_one:
            out = (fvecs[0], fvals[0], indices[0])
        else:
            out = (fvecs, fvals, indices)

        return out

    return wrapper_criterion
