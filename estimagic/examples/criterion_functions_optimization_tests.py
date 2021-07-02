import numpy as np
import pandas as pd

# ======================================================================================
# Define criterion functions
# ======================================================================================


def trid_scalar_criterion(params):
    """Implement Trid function.
    Function description: https://www.sfu.ca/~ssurjano/trid.html

    Args:
        params (pd.DataFrame): Must have the column "value" containing
        input values for parameters. Accepts arbitrary numbers of input values.

    Returns:
        int: Trid function output.
    """
    x = params["value"].to_numpy()
    return ((params["value"] - 1) ** 2).sum() - (params["value"][1:] * x[:-1]).sum()


def trid_gradient(params):
    """Calculate gradient of trid function.

    Args:
        params(pandas.DataFrame): Must have the column "value" containing
        input values for parameters. Accepts arbitrary numbers of input values.

    Returns:
        np.ndarray: gradient of trid function.
    """
    x = params["value"].to_numpy()
    l1 = np.insert(x, 0, 0)
    l1 = np.delete(l1, [-1])
    l2 = np.append(x, 0)
    l2 = np.delete(l2, [0])
    return 2 * (x - 1) - l1 - l2


def trid_pandas_gradient(params):
    """Calculate gradient of trid function.

    Args:
        params(pandas.DataFrame): Must have the column "value" containing
        input values for parameters. Accepts arbitrary numbers of input values.

    Returns:
        pd.Series: gradient of trid function.
    """
    x = params["value"].to_numpy()
    l1 = np.insert(x, 0, 0)
    l1 = np.delete(l1, [-1])
    l2 = np.append(x, 0)
    l2 = np.delete(l2, [0])
    return pd.Series(2 * (x - 1) - l1 - l2)


def trid_criterion_and_gradient(params):
    """Implement Trid function and calculate gradient.

    Args:
        params (pd.DataFrame): Must have the column "value" containing
        input values for parameters. Accepts arbitrary numbers of input values.

    Returns:
        int: trid function output.
        np.ndarray: gradient of trid function.
    """
    x = params["value"].to_numpy()
    l1 = np.insert(x, 0, 0)
    l1 = np.delete(l1, [-1])
    l2 = np.append(x, 0)
    l2 = np.delete(l2, [0])
    return ((params["value"] - 1) ** 2).sum() - (
        params["value"][1:] * x[:-1]
    ).sum(), 2 * (x - 1) - l1 - l2


def trid_dict_criterion(params):
    """Implement trid function.

    Args:
        params(pandas.DataFrame): Must have the column "value" containing
        input values for parameters. Accepts arbitrary numbers of input values.

    Returns:
        Dictionary with the following entries:
        "value" (a scalar float): trid function output.
    """
    out = {
        "value": trid_scalar_criterion(params),
    }
    return out


def rotated_hyper_ellipsoid_scalar_criterion(params):
    """Implement Rotated Hyper Ellipsoid function.
    Function description: https://www.sfu.ca/~ssurjano/rothyp.html.

    Args:
        params (pd.DataFrame): Must have the column "value" containing
        input values for parameters. Accepts arbitrary numbers of input values.

    Returns:
        int: Rotated Hyper Ellipsoid function output.
    """
    val = 0
    for i in range(len(params["value"])):
        val += (params["value"][: i + 1] ** 2).sum()
    return val


def rotated_hyper_ellipsoid_gradient(params):
    """Calculate gradient of rotated_hyper_ellipsoid function.

    Args:
        params(pandas.DataFrame): Must have the column "value" containing
        input values for parameters. Accepts arbitrary numbers of input values.

    Returns:
        np.ndarray: gradient of rotated hyper ellipsoid function.
    """
    x = params["value"].to_numpy()
    return np.arange(2 * len(x), 0, -2) * x


def rotated_hyper_ellipsoid_pandas_gradient(params):
    """Calculate gradient of rotated_hyper_ellipsoid function.

    Args:
        params(pandas.DataFrame): Must have the column "value" containing
        input values for parameters. Accepts arbitrary numbers of input values.

    Returns:
        pd.Series: gradient of rotated hyper ellipsoid function.
    """
    x = params["value"].to_numpy()
    return pd.Series(np.arange(2 * len(x), 0, -2) * x)


def rotated_hyper_ellipsoid_criterion_and_gradient(params):
    """Implement Rotated Hyper Ellipsoid function and calculate gradient.

    Args:
        params (pd.DataFrame): Must have the column "value" containing
        input values for parameters. Accepts arbitrary numbers of input values.

    Returns:
        int: Rotated Hyper Ellipsoid function output.
        np.ndarray: gradient of rotated hyper ellipsoid function.
    """
    val = 0
    for i in range(len(params["value"])):
        val += (params["value"][: i + 1] ** 2).sum()
    x = params["value"].to_numpy()
    return val, np.arange(2 * len(x), 0, -2) * x


def rotated_hyper_ellipsoid_contributions(params):
    """Compute contributions of Rotated Hyper Ellipsoid function.

    Args:
        params (pd.DataFrame): Must have the column "value" containing
        input values for parameters. Accepts arbitrary numbers of input values.

    Returns:
        np.ndarray: array with contributions of function output as elements.
    """
    x = params["value"].to_numpy()
    dim = len(params)
    out = np.zeros(dim)
    for i in range(dim):
        out[i] = (x[: i + 1] ** 2).sum()
    return out


def rotated_hyper_ellipsoid_dict_criterion(params):
    """Implement Rotated Hyper Ellipsoid function and compute
        contributions and root_contributions.

    Args:
        params(pandas.DataFrame): Must have the column "value" containing
        input values for parameters. Accepts arbitrary numbers of input values.

    Returns:
        Dictionary with the following entries:
        "value" (a scalar float): rotated hyper ellipsoid function output.
        "contributions" (np.ndarray): array with contributions of function output
        as elements.
        "root_contributions" (np.ndarray): array with root of contributions of
        function output as elements.
    """
    out = {
        "value": rotated_hyper_ellipsoid_scalar_criterion(params),
        "contributions": rotated_hyper_ellipsoid_contributions(params),
        "root_contributions": np.sqrt(rotated_hyper_ellipsoid_contributions(params)),
    }
    return out


def rosenbrock_scalar_criterion(params):
    """Implement Rosenbrock function.
    Function description: https://www.sfu.ca/~ssurjano/rosen.html

    Args:
        params (pd.DataFrame): Must have the column "value" containing
        input values for parameters. Accepts arbitrary numbers of input values.

    Returns:
        int: Rosenbrock function output.
    """
    x = params["value"].to_numpy()
    r1 = ((x[1:] - x[:-1] ** 2) ** 2).sum() * 100
    r2 = ((x[:-1] - 1) ** 2).sum()
    return r1 + r2


def rosenbrock_gradient(params):
    """Calculate gradient of rosenbrock function.

    Args:
        params(pandas.DataFrame): Must have the column "value" containing
        input values for parameters. Accepts arbitrary numbers of input values.

    Returns:
        np.ndarray: gradient of rosenbrock function.
    """
    x = params["value"].to_numpy()
    l1 = np.delete(x, [-1])
    l1 = np.append(l1, 0)
    l2 = np.insert(x, 0, 0)
    l2 = np.delete(l2, [1])
    l3 = np.insert(x, 0, 0)
    l3 = np.delete(l3, [-1])
    l4 = np.delete(x, [0])
    l4 = np.append(l4, 0)
    l5 = np.full((len(params["value"]) - 1), 2)
    l5 = np.append(l5, 0)
    return 100 * (4 * (l1 ** 3) + 2 * l2 - 2 * (l3 ** 2) - 4 * (l4 * x)) + 2 * l1 - l5


def rosenbrock_pandas_gradient(params):
    """Calculate gradient of rosenbrock function.

    Args:
        params(pandas.DataFrame): Must have the column "value" containing
        input values for parameters. Accepts arbitrary numbers of input values.

    Returns:
        pd.Series: gradient of rosenbrock function.
    """
    return pd.Series(rosenbrock_gradient(params))


def rosenbrock_criterion_and_gradient(params):
    """Implement rosenbrock function and calculate gradient.

    Args:
        params (pd.DataFrame): Must have the column "value" containing
        input values for parameters. Accepts arbitrary numbers of input values.

    Returns:
        int: rosenbrock function output.
        np.ndarray: gradient of rosenbrock function.
    """
    return rosenbrock_scalar_criterion(params), rosenbrock_gradient(params)


def rosenbrock_contributions(params):
    """Compute contributions of rosenbrock function.

    Args:
        params (pd.DataFrame): Must have the column "value" containing
        input values for parameters. Accepts arbitrary numbers of input values.

    Returns:
        np.ndarray: array with contributions of function output as elements.
    """
    x = params["value"].to_numpy()
    dim = len(params)
    out = np.zeros(dim)
    for i in range(dim - 1):
        out[i] = ((x[i + 1] - x[i] ** 2) ** 2) * 100 + ((x[i] - 1) ** 2)
    return out


def rosenbrock_dict_criterion(params):
    """Implement Rosenbrock function and compute
        contributions and root_contributions.

    Args:
        params(pandas.DataFrame): Must have the column "value" containing
        input values for parameters. Accepts arbitrary numbers of input values.

    Returns:
        Dictionary with the following entries:
        "value" (a scalar float): Rosenbrock function output.
        "contributions" (np.ndarray): array with contributions of function output
        as elements.
        "root_contributions" (np.ndarray): array with root of contributions of
        function output as elements.
    """
    out = {
        "value": rosenbrock_scalar_criterion(params),
        "contributions": rosenbrock_contributions(params),
        "root_contributions": np.sqrt(rosenbrock_contributions(params)),
    }
    return out
