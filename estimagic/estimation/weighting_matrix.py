"""Construct a weighting matrix for minimum distance estimators.

Notes:

weighting_matrix should be the only public function in this module, so users only
have to learn one interface, but of course we need several other functions that
do the actual work.

The list of weight_types is very preliminary. Maybe we need some others, maybe
we don't need some that are there. Probably the optimal ones are never used,
but I don't think it's a lot of extra work to implement them if we need
functions to calculate their diagonals anyways

The interface is only a proposal for the long run. We should not be afraid of
raising NotImplementedErrors in the cases we haven't written yet. I would
only implement methods once we actually need them.

"""


def weighting_matrix(moment_func, data, estimation_principle, weight_type="diagonal"):
    """Construct a weighting matrix for minimum distance estimators.

    Args:
        moment_func (function): Calculates a vector of moments from *data*
        data (DataFrame or dict): A pandas DataFrame with simulated or empirical data
            or a dictionary containing several DataFrames.

        estimation_principle (str): One of ['gmm', 'msm', 'indirect_inference']
        weight_type (str): One of [
            'optimal', 'optimal_bootstrap', 'diagonal', 'diagonal_bootstrap']

    Returns:
        weights (numpy.ndarray): A positive semi-definite weighting matrix for minimum
            distance estimators.

    """
    pass
