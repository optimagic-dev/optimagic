import pandas as pd

from estimagic.inference.src.functions.se_estimation import likelihood_inference


# Define data
orig_data = pd.read_csv(
    r"C:\Users\Jrxz12\estimagic\estimagic\inference\src\examples\data.csv"
)
orig_data["fpc"] = 0.8
formulas = ["eco_friendly ~ ppltrst + male + income"]


# Define design dictionary
design_dict = {"weight": "weight"}

# Define log likelihood keyword arguments
log_like_kwargs = {"formulas": formulas, "data": orig_data, "model": "logit"}

# =============================================================================
params_df, cov = likelihood_inference(
    log_like_kwargs, design_dict=design_dict, cov_type="sandwich"
)
