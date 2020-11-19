import numpy as np
import pandas as pd
import scipy

from estimagic.optimization.process_constraints import process_constraints
from estimagic.optimization.reparametrize import reparametrize_from_internal
from estimagic.optimization.reparametrize import reparametrize_to_internal


def transform_covariance(
    params, internal_cov, constraints, n_samples, bounds_handling,
):
    """Transform the internal covariance matrix to an external one, given constraints.

    Args:
        params (pd.DataFrame): DataFrame where the "value" column contains estimated
            parameters of a likelihood model. See :ref:`params` for details.
        internal_cov (np.ndarray) with a covariance matrix of the internal parameter
            vector. For background information about internal and external params
            see :ref:`implementation_of_constraints`.
        constraints (list): List with constraint dictionaries.
            See .. _link: ../../docs/source/how_to_guides/how_to_use_constraints.ipynb
        n_samples (int): Number of samples used to transform the covariance matrix of
            the internal parameter vector into the covariance matrix of the external
            parameters.
        bounds_handling (str): One of "clip", "raise", "ignore". Determines how bounds
            are handled. If "clip", confidence intervals are clipped at the bounds.
            Standard errors are only adjusted if a sampling step is necessary due to
            additional constraints. If "raise" and any lower or upper bound is binding,
            we raise an error. If "ignore", boundary problems are simply ignored.

    Returns:
        pd.DataFrame: Quadratic DataFrame containing the covariance matrix of the free
            parameters. If parameters were fixed (explicitly or by other constraints),
            the index is a subset of params.index. The columns are the same as the
            index.

    """
    processed_constraints, processed_params = process_constraints(constraints, params)
    free_index = processed_params.query("_internal_free").index

    if processed_constraints:
        free = processed_params.loc[free_index]
        is_free = processed_params["_internal_free"].to_numpy()
        pre_replacements = processed_params["_pre_replacements"].to_numpy()
        post_replacements = processed_params["_post_replacements"].to_numpy()
        fixed_values = processed_params["_internal_fixed_value"].to_numpy()
        lower_bounds = free["_internal_lower"]
        upper_bounds = free["_internal_upper"]

        internal_mean = reparametrize_to_internal(
            external=params["value"].to_numpy(),
            internal_free=is_free,
            processed_constraints=processed_constraints,
        )
        sample = np.random.multivariate_normal(
            mean=internal_mean, cov=internal_cov, size=n_samples,
        )
        transformed_free = []
        for params_vec in sample:
            if bounds_handling == "clip":
                params_vec = np.clip(params_vec, a_min=lower_bounds, a_max=upper_bounds)
            elif bounds_handling == "raise":
                if (params_vec < lower_bounds).any() or (
                    params_vec > upper_bounds
                ).any():
                    raise ValueError()

            transformed = reparametrize_from_internal(
                internal=params_vec,
                fixed_values=fixed_values,
                pre_replacements=pre_replacements,
                processed_constraints=processed_constraints,
                post_replacements=post_replacements,
            )
            transformed_free.append(transformed[is_free])

        free_cov = np.cov(np.array(transformed_free), rowvar=False,)

    else:
        free_cov = internal_cov

    res = pd.DataFrame(data=free_cov, columns=free_index, index=free_index)
    return res


def calculate_inference_quantities(params, free_cov):
    """Add standard errors, pvalues and confidence intervals to params.

    Args
        params (pd.DataFrame): See :ref:`params`.
        free_cov (pd.DataFrame): Quadratic DataFrame containing the covariance matrix
        of the free parameters. If parameters were fixed (explicitly or by other
        constraints) the index is a subset of params.index. The columns are the same as
        the index.

    Returns:
        pd.DataFrame: DataFrame with same index as params, containing the columns
            "value", "standard_error", "pvalue", "ci_lower" and "ci_upper".
            Parameters that do not have a standard error (e.g. because they were fixed
            during estimation) contain NaNs in all but the "value" column. The value
            column is only reproduced for convenience.

    """
    free = pd.DataFrame(index=free_cov.index)
    free["value"] = params.loc[free.index, "value"]
    free["standard_error"] = np.sqrt(np.diag(free_cov))
    tvalues = free["value"] / free["standard_error"]
    free["p_value"] = 1.96 * scipy.stats.norm.sf(np.abs(tvalues))
    free["ci_lower"] = free["value"] - 1.96 * free["standard_error"]
    free["ci_upper"] = free["value"] + 1.96 * free["standard_error"]
    free["stars"] = pd.cut(
        free["p_value"], bins=[-1, 0.01, 0.05, 0.1, 2], labels=["***", "**", "*", ""]
    )

    res = free.reindex(params.index)
    res["value"] = params["value"]
    return res
