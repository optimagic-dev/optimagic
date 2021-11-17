import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)


def lowest_so_far(sr):
    """For each row give the lowest value so far in the Series."""
    only_lowest = sr[sr.diff() < 0].reindex(sr.index)
    nan_filled = only_lowest.fillna(method="ffill").fillna(sr)
    return nan_filled


def calculate_share_of_improvement_missing(current, start_value, target_value):
    """Calculate the share of improvement still missing relative to the start point.

    Args:
        current (float or pandas.Series): current value (e.g. criterion value)
        start_value (float): start value (e.g. criterion value at the start parameters)
        target_value (float): target value (e.g. criterion value at the optimum)

    Returns:
        float or pandas.Series: The lower the value the closer the current value is
            to the target value. 0 means the target value has been reached. 1 means the
            current value is as far from the target value as the start value.

    """
    total_improvement = start_value - target_value
    missing_improvement = current - target_value
    share_missing = missing_improvement / total_improvement
    return share_missing
