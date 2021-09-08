"""Implement local sensitivity measures for method of moments.

measures:
m1: Andrews, Gentzkow & Shapiro
(https://academic.oup.com/qje/article/132/4/1553/3861634)

epsilon 2-6: Honore, Jorgensen & de Paula
(https://papers.srn.com/abstract=3518640)

"""
import numpy as np

from estimagic.inference.msm_covs import cov_sandwich
from estimagic.utilities import robust_inverse


def calculate_sensitivity_to_bias(jac, weights):
    """calculate the sensitivity to bias.

    The sensitivity measure is calculated for each parameter wrt each moment.

    It answers the following question: How strongly would the parameter estimates be
        biased if the kth moment was misspecified, i.e not zero in expectation?

    Args:
        jac (np.ndarray): Numpy array with the jacobian of the function that
            calculates the deviation between simulated and empirical moments
            with respect to params, evaluated at the point estimates.
        weights (np.ndarray): The weighting matrix for msm estimation.

    Returns:
        np.ndarray

    """
    gwg = _sandwich(jac, weights)
    gwg_inverse = robust_inverse(gwg)
    m1 = -gwg_inverse @ jac.T @ weights

    return m1


def calculate_fundamental_sensitivity_to_noise(
    jac, weights_opt, moments_cov, params_cov_opt
):
    """calculate the fundamental sensitivity to noise.

    The sensitivity measure is calculated for each parameter wrt each moment.

    It answers the following question: How much precision would be lost if the kth
        moment was subject to a little additional noise if the optimal weighting matrix
        is used?

    Args:
        jac (np.ndarray): Numpy array with the jacobian of the function that
            calculates the deviation between simulated and empirical moments
            with respect to params, evaluated at the point estimates.
        weights_opt (np.ndarray): The asymptotically efficient weighting matrix for
            msm estimation.
        moments_cov (np.ndarray): The covariance matrix of the empirical moments.
        params_cov_opt (np.ndarray): The covariance matrix of the parameter estimates
            that would result if the asymptotically efficient weighting matrix was
            used for msm estimation.

    Returns:
        np.ndarray

    """
    m2 = []

    for k in range(len(weights_opt)):
        mask_matrix_o = np.zeros(shape=weights_opt.shape)
        mask_matrix_o[k, k] = 1

        meat = _sandwich_plus(jac, weights_opt, mask_matrix_o)

        m2k = params_cov_opt @ meat @ params_cov_opt
        m2k = np.diagonal(m2k)

        m2.append(m2k)

    m2 = np.array(m2).T

    moments_variances = np.diagonal(moments_cov)
    params_variances = np.diagonal(params_cov_opt)

    m2_scaled = m2 / params_variances.reshape(-1, 1)
    m2_scaled = m2_scaled * moments_variances

    return m2_scaled


def calculate_actual_sensitivity_to_noise(
    sensitivity_to_bias, weights, moments_cov, params_cov
):
    """calculate the actual sensitivity to noise.

    The sensitivity measure is calculated for each parameter wrt each moment.

    It answers the following question: How much precision would be lost if the kth
        moment was subjet to a little additional noise if "weights" is used as
        weighting matrix?

    Args:
        sensitivity_to_bias (np.ndarray)
        weights (np.ndarray): A square weighting matrix.
        moments_cov (np.ndarray): Covariance matrix of the empirical moments.
        params_cov (np.ndarray): Covariance matrix of the estimated parameters

    Returns:
        np.ndarray

    """

    m3 = []

    for k in range(len(weights)):
        mask_matrix_o = np.zeros(shape=weights.shape)
        mask_matrix_o[k, k] = 1

        m3k = _sandwich(sensitivity_to_bias.T, mask_matrix_o)
        m3k = np.diagonal(m3k)

        m3.append(m3k)

    m3 = np.array(m3).T

    moments_variances = np.diagonal(moments_cov)
    params_variances = np.diagonal(params_cov)

    m3_scaled = m3 / params_variances.reshape(-1, 1)
    m3_scaled = m3_scaled * moments_variances

    return m3_scaled


def calculate_actual_sensitivity_to_removal(jac, weights, moments_cov, params_cov):
    """calculate the actual sensitivity to removal.

    The sensitivity measure is calculated for each parameter wrt each moment.

    It answers the following question: How much precision would be lost if the kth
        moment was excluded from the estimation if "weights" is used as weighting
        matrix?

    Args:
        jac (np.ndarray): Numpy array with the jacobian of the function that
            calculates the deviation between simulated and empirical moments
            with respect to params, evaluated at the point estimates.
        weights_opt (np.ndarray): Square weighting matrix.
        moments_cov (np.ndarray): Covariance matrix of the empirical moments.
        params_cov (np.ndarray): Covariance matrix of the estimated parameters

    Returns:
        np.ndarray

    """
    m4 = []

    for k in range(len(weights)):
        weight_tilde_k = np.copy(weights)
        weight_tilde_k[k, :] = 0
        weight_tilde_k[:, k] = 0

        sigma_tilde_k = cov_sandwich(jac, weight_tilde_k, moments_cov)

        m4k = sigma_tilde_k - params_cov
        m4k = m4k.diagonal()

        m4.append(m4k)

    m4 = np.array(m4).T

    params_variances = np.diagonal(params_cov)
    m4_scaled = m4 / params_variances.reshape(-1, 1)

    return m4_scaled


def calculate_fundamental_sensitivity_to_removal(jac, moments_cov, params_cov_opt):
    """calculate the fundamental sensitivity to removal.

    The sensitivity measure is calculated for each parameter wrt each moment.

    It answers the following question: How much precision would be lost if the kth
        moment was excluded from the estimation with if the optimal weighting matrix is
        used?

    Args:
        jac (np.ndarray): Numpy array with the jacobian of the function that
            calculates the deviation between simulated and empirical moments
            with respect to params, evaluated at the point estimates.
        moments_cov (np.ndarray): Covariance matrix of the empirical moments.
        params_cov_opt (np.ndarray): The covariance matrix of the parameter estimates
            that would result if the asymptotically efficient weighting matrix was
            used for msm estimation.

    Returns:
        np.ndarray

    """
    m5 = []

    for k in range(len(moments_cov)):
        g_k = np.copy(jac)
        g_k = np.delete(g_k, k, axis=0)

        s_k = np.copy(moments_cov)
        s_k = np.delete(s_k, k, axis=0)
        s_k = np.delete(s_k, k, axis=1)

        sigma_k = _sandwich(g_k, np.linalg.inv(s_k))
        sigma_k = np.linalg.inv(sigma_k)

        m5k = sigma_k - params_cov_opt
        m5k = m5k.diagonal()

        m5.append(m5k)

    m5 = np.array(m5).T

    params_variances = np.diagonal(params_cov_opt)
    m5_scaled = m5 / params_variances.reshape(-1, 1)

    return m5_scaled


def calculate_sensitivity_to_weighting(jac, weights, moments_cov, params_cov):
    """calculate the sensitivity to weighting.

    The sensitivity measure is calculated for each parameter wrt each moment.

    It answers the following question: How would the precision change if the weight of
        the kth moment is increased a little?

    Args:
        jac (np.ndarray): Numpy array with the jacobian of the function that
            calculates the deviation between simulated and empirical moments
            with respect to params, evaluated at the point estimates.
        weights_opt (np.ndarray): Square weighting matrix.
        moments_cov (np.ndarray): Covariance matrix of the empirical moments.
        params_cov (np.ndarray): Covariance matrix of the estimated parameters

    Returns:
        np.ndarray

    """
    gwg_inverse = _sandwich(jac, weights)
    gwg_inverse = np.linalg.pinv(gwg_inverse)

    m6 = []

    for k in range(len(weights)):
        mask_matrix_o = np.zeros(shape=weights.shape)
        mask_matrix_o[k, k] = 1

        m6k_1 = gwg_inverse @ _sandwich(jac, mask_matrix_o) @ params_cov
        m6k_2 = (
            gwg_inverse
            @ jac.T
            @ mask_matrix_o
            @ moments_cov
            @ weights
            @ jac
            @ gwg_inverse
        )
        m6k_3 = (
            gwg_inverse
            @ jac.T
            @ weights
            @ moments_cov
            @ mask_matrix_o
            @ jac
            @ gwg_inverse
        )
        m6k_4 = params_cov @ _sandwich(jac, mask_matrix_o) @ gwg_inverse

        m6k = -m6k_1 + m6k_2 + m6k_3 - m6k_4
        m6k = m6k.diagonal()

        m6.append(m6k)

    m6 = np.array(m6).T

    weights_diagonal = np.diagonal(weights)
    params_variances = np.diagonal(params_cov)

    m6_scaled = m6 / params_variances.reshape(-1, 1)
    m6_scaled = m6_scaled * weights_diagonal

    return m6_scaled


def _sandwich(a, b):
    """calculate the sandwich product of two matrices: a.T * b * a."""
    sandwich = a.T @ b @ a
    return sandwich


def _sandwich_plus(a, b, c):
    """calculate the sandwich product of three matrices: a.T * b.T * c * b * a"""
    sandwich = a.T @ b.T @ c @ b @ a
    return sandwich
