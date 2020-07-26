import jax.numpy as jnp
from jax.ops import index
from jax.ops import index_update


def covariance_to_internal(external_values):
    """Do a cholesky reparametrization."""
    cov = cov_params_to_matrix_jax(external_values)
    chol = jnp.linalg.cholesky(cov)
    return chol[jnp.tril_indices(len(cov))]


def covariance_from_internal(internal_values):
    """Undo a cholesky reparametrization."""
    chol = chol_params_to_lower_triangular_matrix_jax(internal_values)
    cov = chol @ chol.T
    return cov[jnp.tril_indices(len(chol))]


def sdcorr_to_internal(external_values):
    """Convert sdcorr to cov and do a cholesky reparametrization."""
    cov = sdcorr_params_to_matrix_jax(external_values)
    chol = jnp.linalg.cholesky(cov)
    return chol[jnp.tril_indices(len(cov))]


def sdcorr_from_internal(internal_values):
    """Undo a cholesky reparametrization."""
    chol = chol_params_to_lower_triangular_matrix_jax(internal_values)
    cov = chol @ chol.T
    return cov_matrix_to_sdcorr_params_jax(cov)


def probability_to_internal(external_values):
    """Reparametrize probability constrained parameters to internal."""
    return external_values / external_values[-1]


def probability_from_internal(internal_values):
    """Reparametrize probability constrained parameters from internal."""
    return internal_values / internal_values.sum()


def chol_params_to_lower_triangular_matrix_jax(params):
    dim = number_of_triangular_elements_to_dimension_jax(len(params))
    mat = index_update(jnp.zeros((dim, dim)), index[jnp.tril_indices(dim)], params)
    return mat


def number_of_triangular_elements_to_dimension_jax(num):
    """Calculate the dimension of a square matrix from number of triangular elements.

    Args:
        num (int): The number of upper or lower triangular elements in the matrix.

    Examples:
        >>> number_of_triangular_elements_to_dimension_jax(6)
        3
        >>> number_of_triangular_elements_to_dimension_jax(10)
        4

    """
    return int(jnp.sqrt(8 * num + 1) / 2 - 0.5)


def cov_params_to_matrix_jax(cov_params):
    """Build covariance matrix from 1d array with its lower triangular elements.

    Args:
        cov_params (np.array): 1d array with the lower triangular elements of a
            covariance matrix (in C-order)

    Returns:
        cov (np.array): a covariance matrix

    """
    lower = chol_params_to_lower_triangular_matrix_jax(cov_params)
    cov = lower + jnp.tril(lower, k=-1).T
    return cov


def cov_matrix_to_params_jax(cov):
    return cov[jnp.tril_indices(len(cov))]


def sdcorr_params_to_sds_and_corr_jax(sdcorr_params):
    dim = number_of_triangular_elements_to_dimension_jax(len(sdcorr_params))
    sds = jnp.array(sdcorr_params[:dim])
    corr = jnp.eye(dim)
    corr = index_update(corr, index[jnp.tril_indices(dim, k=-1)], sdcorr_params[dim:])
    corr += jnp.tril(corr, k=-1).T
    return sds, corr


def sds_and_corr_to_cov_jax(sds, corr):
    diag = jnp.diag(sds)
    return diag @ corr @ diag


def cov_to_sds_and_corr_jax(cov):
    sds = jnp.sqrt(jnp.diagonal(cov))
    diag = jnp.diag(1 / sds)
    corr = diag @ cov @ diag
    return sds, corr


def sdcorr_params_to_matrix_jax(sdcorr_params):
    """Build covariance matrix out of variances and correlations.

    Args:
        sdcorr_params (np.array): 1d array with parameters. The dimensions of the
            covariance matrix are inferred automatically. The first dim parameters
            are assumed to be the variances. The remainder are the lower triangular
            elements (excluding the diagonal) of a correlation matrix.

    Returns:
        cov (np.array): a covariance matrix

    """
    return sds_and_corr_to_cov_jax(*sdcorr_params_to_sds_and_corr_jax(sdcorr_params))


def cov_matrix_to_sdcorr_params_jax(cov):
    dim = len(cov)
    sds, corr = cov_to_sds_and_corr_jax(cov)
    correlations = corr[jnp.tril_indices(dim, k=-1)]
    return jnp.hstack([sds, correlations])


def dimension_to_number_of_triangular_elements_jax(dim):
    """Calculate number of triangular elements from the dimension of a square matrix.

    Args:
        dim (int): Dimension of a square matrix.

    """
    return int(dim * (dim + 1) / 2)
