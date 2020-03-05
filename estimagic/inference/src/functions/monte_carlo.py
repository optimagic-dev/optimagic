"""Monte Carlo simulations on our estimator."""
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_blobs

from estimagic.inference.src.functions.mle_unconstrained import estimate_likelihood
from estimagic.inference.src.functions.mle_unconstrained import estimate_parameters
from estimagic.inference.src.functions.se_estimation import design_options_preprocessing


def create_strata(categories_dict, sample_size, orig_data=None, random=True):
    """Adds strata column to given *orig_data* or to the random sample provided.

    Args:
        sample_size (positive int): number of observations wanted for the
            random sample
        categories_dict (dictionary): key-value pairs of category and number of
            categories. These are the columns providing the stratification
            information.
            Example: categories = {"income": "5", "sex": "2", "age": "3"}
        orig_data (pd.DataFrame): the data you wish to strata
        random (bool): tells the function to produce a random sample

    Returns:
        population (pd.DataFrame): returned for random=True
        orig_data (pd.DataFrame): original dataset returned with strata column

    """
    if orig_data:
        population = pd.DataFrame(index=range(0, len(orig_data)))
    else:
        population = pd.DataFrame(index=range(0, sample_size))
    if random:
        for category in categories.keys():
            population[category] = np.random.randint(
                0, categories[category], sample_size
            )
    else:
        for category in categories.keys():
            population[category] = orig_data[category]

    # Strata information
    population["strata"] = population[population.columns].apply(
        lambda row: "_".join(row.values.astype(str)), axis=1
    )
    population["strata"] = pd.factorize(population["strata"])[0]

    if random:
        return population
    else:
        orig_data["strata"] = population["strata"]
        return orig_data


# Example
categories = {"income": "5", "sex": "2", "age": "3"}
data = create_strata(categories, sample_size=10000, random=True)

# Additional columns. The previous columns were just used to create the strata.
data["passed_class"] = np.random.randint(0, 2, 10000)


def make_stratified_sample(stratified_sample, frac_of_sample):
    """Generates a dataframe that is a *frac_of_sample* of *stratified_sample*

    """
    proportioned_stratified_sample = []
    for strata in stratified_sample["strata"].unique():
        strata_obs = stratified_sample[stratified_sample["strata"] == strata]
        fractional_sample = strata_obs.sample(frac=frac_of_sample)
        proportioned_stratified_sample.append(fractional_sample)

    proportioned_stratified_sample = pd.concat(proportioned_stratified_sample)
    return proportioned_stratified_sample


# make 100 clusters, create data between 0 and 10. Normal distribution.
cluster_data, cluster = make_blobs(
    n_samples=10000,
    n_features=1,
    centers=100,
    cluster_std=1.0,
    center_box=(0, 10.0),
    random_state=None,
)
data["hours_spent_studying_daily"] = cluster_data


# SAMPLES
# independent observations sample
regular_sample = pd.DataFrame()
regular_sample["passed_class"] = np.random.randint(0, 2, 10000)
regular_sample["hours_spent_studying_daily"] = np.random.randint(0, 10, 10000)

# clustered sample
clustered_sample = data[["passed_class", "hours_spent_studying_daily"]]

# stratified sample
stratified_sample = data[["passed_class", "hours_spent_studying_daily", "strata"]]
proportioned_stratified_sample = make_stratified_sample(data, 0.75)[
    ["passed_class", "hours_spent_studying_daily"]
]

# Define formula to use estimate_parameters
formula = ["passed_class ~ hours_spent_studying_daily"]
design_option = design_options_preprocessing(data)
log_like_kwargs = {"formulas": formula, "data": clustered_sample, "model": "logit"}


# Setup Monte Carlo sample size, how many samples and zeroed beta matrix
def monte_carlo_sim(
    n_obs, m_samples, n_parameters, log_like_kwargs, design_option, dashboard=False,
):

    for m in range(m_samples):
        mc_sample = log_like_kwargs["data"].sample(n_obs)
        mc_kwargs = {
            "formulas": log_like_kwargs["formulas"],
            "data": mc_sample,
            "model": log_like_kwargs["model"],
        }
        info, params = estimate_parameters(
            estimate_likelihood, design_option, mc_kwargs, dashboard=dashboard
        )
        beta_matrix[:, m] = params["value"]

    return beta_matrix


beta_matrix = monte_carlo_sim(
    1000, 500, 1, log_like_kwargs, design_option, dashboard=False
)
clu_beta_matrix = monte_carlo_sim(
    1000, 500, 1, log_like_kwargs, design_option, dashboard=False
)

str_beta_matrix = np.zeros((2, 500))
for m in range(500):
    mc_sample = make_stratified_sample(stratified_sample, 0.10)
    mc_kwargs = {
        "formulas": log_like_kwargs["formulas"],
        "data": mc_sample,
        "model": log_like_kwargs["model"],
    }
    info, params = estimate_parameters(
        estimate_likelihood, design_option, mc_kwargs, dashboard=False
    )
    str_beta_matrix[:, m] = params["value"]

beta_matrix = pd.DataFrame(beta_matrix[1:])
clu_beta_matrix = pd.DataFrame(clu_beta_matrix[1:])
str_beta_matrix = pd.DataFrame(str_beta_matrix[1:])

beta_matrix.to_csv("beta_matrix.csv")
clu_beta_matrix.to_csv("clu_beta_matrix.csv")
str_beta_matrix.to_csv("str_beta_matrix.csv")

sns.distplot(beta_matrix)
sns.distplot(clu_beta_matrix)
sns.distplot(str_beta_matrix)

ind_plot = sns.distplot(beta_matrix)
clu_plot = sns.distplot(clu_beta_matrix)
str_plot = sns.distplot(str_beta_matrix)

ind_plot.figure.savefig("ind_plot.png")
clu_plot.figure.savefig("clu_plot.png")
str_plot.figure.savefig("str_plot.png")
