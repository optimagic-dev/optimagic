import numpy as np


def get_aggregator(aggregator, functype, model_info):
    """Get a function that aggregates a VectorModel into a ScalarModel.

    Args:
        aggregator (str or callable): Name of an aggregator or aggregator function.
            The function must take vector_model, residuals and model_info.
        functype (str): One of "scalar", "least_squares" and "likelihood".
        model_info (ModelInfo): Information that describes the functional form of
            the model.

    Returns:
        callable: The partialled aggregator that only depends on vector_model and
            residuals

    """
    # determine if aggregator is compatible with functype and model_info
    # load aggregator
    # partial stuff
    pass


def _aggregate_models_template(vector_model, residuals, aggregator, model_info):
    """Aggregate a VectorModel into a ScalarModel.

    Notes:

    - If some aggregators have special options, we need to add a user_options argument.
    - I still don't get why we need the residuals here, but it's not a problem. Just
      make sure I described them correctly.

    Args:
        vector_model (VectorModel): The VectorModel to aggregate.
        residuals (np.ndarray): The residuals on which the vector model was fit. A 1d
            array of length n_residuals.
        aggregator (callable): The function that does the actual aggregation.
        model_info (ModelInfo): Information that describes the functional form of
            the model.

    Returns:
        ScalarModel: The aggregated model

    """
    # call the aggregator with suitable arguments
    # stick the results into a ScalarModel
    pass


def aggregate_residual_models(model, residuals, options, functype):
    """Aggregate residual models to main model.

    The main problem
    ----------------

        minimize sum_i f(residual_i)

    with i = 1, ..., n.

    For the scalar case n = 1, otherwise n can be arbitrary. For the scalar and the
    likelihood case f(x) = x, for the least_squares case f(x) = x ** 2.

    The residual model
    ------------------

        for i = 1, ..., n: create surrogate model of residual_i


    The aggregation problem
    -----------------------

        given the residual models reconstruct a quadratic model for sum_i f(residual_i)


    Args:
        model (dict): Model dictionary containing the following entries:
            - "intercepts" np.ndarray of shape (n_residuals, 1)
            - "linear_terms" np.ndarray of shape (n_residuals, n_params)
            - "square_terms" np.ndarray of shape (n_residuals, n_params, n_params)
        residuals (np.ndarray): Array of shape (n_residuals, )
        options (dict): Options passed to fitting method for model creation.
        functype (str): Type of function that is being optimized. Must be in {'scalar',
                'likelihood', 'least_squares'}.

    Returns:
        dict: Aggregated model of same structure as 'model'.

    """
    linear_terms = model["linear_terms"]
    square_terms = model["square_terms"]

    # intercept and linear_terms
    if functype in ("scalar", "likelihood"):
        aggregated = {
            "intercepts": model["intercepts"].sum(),
            "linear_terms": linear_terms.sum(axis=0),
        }
    elif functype == "least_squares":
        aggregated = {
            "intercepts": None,  # update this
            "linear_terms": linear_terms.T @ residuals,
        }
    else:
        raise ValueError("functype must be in {'scalar','likelihood','least_squares'}.")

    # square_terms
    if functype == "scalar":
        aggregated["square_terms"] = square_terms.squeeze()
    elif functype in ("likelihood", "least_squares"):

        update_term = (
            square_terms.T @ residuals
            if functype == "least_squares"
            else square_terms.sum(axis=0)
        )

        # combine information on hessian using first- and second-degree coefficients
        square_terms = (linear_terms.T @ linear_terms + update_term) / 2
        # correct averaging if some square terms are zero
        diag_mask = np.eye(linear_terms.shape[1], dtype=bool)
        if not options["include_squares"]:
            square_terms[diag_mask] *= 2
        if not options["include_interaction"]:
            square_terms[~diag_mask] *= 2

        aggregated["square_terms"] = square_terms

    return aggregated
