import numpy as np
import pandas as pd
from patsy import dmatrices


def ordered_logit_processing(formula, data):
    """Process user input for an ordered logit model."""
    # extract data arrays
    y, x = dmatrices(formula + " - 1", data, return_type="dataframe")
    y = y[y.columns[0]]

    # extract dimensions
    num_choices = len(y.unique())
    beta_names = list(x.columns)
    num_betas = len(beta_names)
    num_cutoffs = num_choices - 1

    # set-up index for params_df
    names = beta_names + list(range(num_cutoffs))
    categories = ["beta"] * num_betas + ["cutoff"] * num_cutoffs
    index = pd.MultiIndex.from_tuples(zip(categories, names), names=["type", "name"])

    # make params_df
    rng = np.random.default_rng(seed=5471)
    start_params = pd.DataFrame(index=index)
    start_params["value"] = np.hstack(
        [
            rng.uniform(low=-0.5, high=0.5, size=len(x.columns)),
            np.arange(num_cutoffs) * 2,
        ]
    )

    # make constraints
    constr = [{"loc": "cutoff", "type": "increasing"}]

    return start_params, y.to_numpy().astype(int), x.to_numpy(), constr
