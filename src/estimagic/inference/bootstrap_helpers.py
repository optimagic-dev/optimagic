import pandas as pd


def check_inputs(
    data=None, cluster_by=None, ci_method="percentile", ci_level=0.95, skipdata=False
):
    """Check validity of inputs.

    Args:
        data (pd.DataFrame): original dataset.
        cluster_by (str): column name of variable to cluster by.
        ci_method (str): method of choice for confidence interval computation.
        alpha (float): significance level of choice.
        skipdata (bool): Whether to skip all checks on the data argument.

    """
    ci_method_list = ["percentile", "bc", "t", "normal", "basic"]

    if not skipdata:
        if data is None:
            raise ValueError("Data cannot be None if outcome is callable.")
        elif not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas.DataFrame.")
        elif (cluster_by is not None) and (cluster_by not in data.columns.tolist()):
            raise ValueError(
                "Input 'cluster_by' must be None or a column name of 'data'."
            )

    if ci_method not in ci_method_list:
        msg = (
            "ci_method must be 'percentile', 'bc', 't', 'basic' or 'normal', "
            f"'{ci_method}' was supplied"
        )
        raise ValueError(msg)
    if ci_level > 1 or ci_level < 0:
        raise ValueError("Input 'ci_level' must be in [0,1].")
