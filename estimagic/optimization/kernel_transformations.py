import numpy as np

from estimagic.optimization.utilities import chol_params_to_lower_triangular_matrix
from estimagic.optimization.utilities import cov_matrix_to_sdcorr_params
from estimagic.optimization.utilities import cov_params_to_matrix
from estimagic.optimization.utilities import robust_cholesky
from estimagic.optimization.utilities import sdcorr_params_to_matrix


def covariance_to_internal(external_values, constr):
    """Do a cholesky reparametrization."""
    cov = cov_params_to_matrix(external_values)
    chol = robust_cholesky(cov)
    return chol[np.tril_indices(len(cov))]


def covariance_from_internal(internal_values, constr):
    """Undo a cholesky reparametrization."""
    chol = chol_params_to_lower_triangular_matrix(internal_values)
    cov = chol @ chol.T
    return cov[np.tril_indices(len(chol))]


def sdcorr_to_internal(external_values, constr):
    """Convert sdcorr to cov and do a cholesky reparametrization."""
    cov = sdcorr_params_to_matrix(external_values)
    chol = robust_cholesky(cov)
    return chol[np.tril_indices(len(cov))]


def sdcorr_from_internal(internal_values, constr):
    """Undo a cholesky reparametrization."""
    chol = chol_params_to_lower_triangular_matrix(internal_values)
    cov = chol @ chol.T
    return cov_matrix_to_sdcorr_params(cov)


def probability_to_internal(external_values, constr):
    """Reparametrize probability constrained parameters to internal."""
    return external_values / external_values[-1]


def probability_from_internal(internal_values, constr):
    """Reparametrize probability constrained parameters from internal."""
    return internal_values / internal_values.sum()


def linear_to_internal(external_values, constr):
    """Reparametrize linear constraint to internal."""
    return constr["to_internal"] @ external_values


def linear_from_internal(internal_values, constr):
    """Reparametrize linear constraint from internal."""
    return constr["from_internal"] @ internal_values
