"""Monte Carlo simulations on our estimator."""
import numpy as np
import pandas as pd
from mle_unconstrained import estimate_parameters
from sklearn.datasets import make_blobs


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


# Test
categories = {"income": "5", "sex": "2", "age": "3"}
data = create_strata(sample_size=10000, categories=categories, random=True)

# Additional columns. The previous columns were just used to create the strata.
data["passed_class"] = np.random.randint(0, 1, 10000)


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
regular_sample["passed_class"] = np.random.randint(0, 1, 10000)
regular_sample["hours_spent_studying_daily"] = np.random.randint(0, 10, 10000)

# clustered sample
clustered_sample = data[["passed_class", "hours_spent_studying_daily"]]

# stratified sample
proportioned_stratified_sample = make_stratified_sample(data, 0.75)[
    ["passed_class", "hours_spent_studying_daily"]
]

# Define formula to use estimate_parameters
formulas = ["passed_class ~ hours_spent_studying_daily"]


# Setup Monte Carlo sample size, how many samples and zeroed beta matrix
def monte_carlo_sim(
    model,
    n_obs,
    m_samples,
    n_parameters,
    data,
    formulas,
    design_option,
    dashboard=False,
):

    beta_matrix = np.zeros((n_parameters + 1, m_samples))
    for m in range(m_samples):
        mc_sample = data.sample(n_obs)
        info, params = estimate_parameters(
            model, formulas, mc_sample, design_option, dashboard=dashboard
        )
        beta_matrix[:, m] = params["value"]

    return beta_matrix


regular_sample.to_csv("independent_obs.csv")
clustered_sample.to_csv("clustered_obs.csv")
