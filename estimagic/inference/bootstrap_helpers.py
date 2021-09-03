import pandas as pd


def check_inputs(data, cluster_by=None, ci_method="percentile", alpha=0.05):
    """Check validity of inputs.
    Args:
        data (pd.DataFrame): original dataset.
        cluster_by (str): column name of variable to cluster by.
        ci_method (str): method of choice for confidence interval computation.
        alpha (float): significance level of choice.

    """

    ci_method_list = ["percentile", "bca", "bc", "t", "normal", "basic"]

    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input 'data' must be DataFrame.")

    elif (cluster_by is not None) and (cluster_by not in data.columns.tolist()):
        raise ValueError(
            "Input 'cluster_by' must be None or a column name of DataFrame."
        )

    elif ci_method not in ci_method_list:
        raise ValueError(
            "ci_method must be 'percentile', 'bc',"
            " 'bca', 't', 'basic' or 'normal', '{method}'"
            " was supplied".format(method=ci_method)
        )

    elif alpha > 1 or alpha < 0:
        raise ValueError("Input 'alpha' must be in [0,1].")
