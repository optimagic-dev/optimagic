"""Implement local sensitivity measures for method of moments.

measures:
m1: Andrews, Gentzkow & Shapiro
(https://academic.oup.com/qje/article/132/4/1553/3861634)

epsilon 2-6: Honore, Jorgensen & de Paula
(https://papers.srn.com/abstract=3518640)

"""
import numpy as np
import pandas as pd

from estimagic.differentiation.derivatives import first_derivative


EXPLANATIONS = {
    "sensitivity_to_bias": (
        "How strongly would the parameter estimates be biased if the kth moment was "
        "misspecified, i.e not zero in expectation?"
    ),
    "fundamental_sensitivity_to_noise": (
        "How much precision would be lost if the kth moment was subject to a little "
        "additional noise if the optimal weighting matrix is used?"
    ),
    "actual_sensitivity_to_noise": (
        "How much precision would be lost if the kth moment was subjet to a little "
        "additional noise if the current weighting matrix is used?"
    ),
    "actual_sensitivity_to_removal": (
        "How much precision would be lost if the kth moment was excluded from "
        "the estimation if the current weighting matrix is used."
    ),
    "fundamental_sensitivity_to_removal": (
        "How much precision would be lost if the kth moment was excluded from "
        "the estimation with if the optimal weighting matrix is used?"
    ),
    "sensitivity_to_weighting": (
        "How would the precision change if the weight of the kth moment is increased "
        "a little?"
    ),
}

MEASURES = list(EXPLANATIONS.keys())


def moment_sensitivity(
    moment_func,
    moment_contributions_func,
    params,
    func1_kwargs=None,
    func2_kwargs=None,
    weight_matrix=None,
    save_csv=False,
):
    """Return all six (6) measurements,
    and save as six (6) .csv files.

    args:
        moment_func (function): moment function (expectations)
        moment_contributions_func (function): moment function (actual values)
        params (pd.DataFrame): see :ref:`params`
        func1_kwargs (dict): additional arguments for moment_func
        func2_kwargs (dict): additional arguments for moment_contributions_func
        weight_matrix (np.array): user defined weight matrix.
                                  If not specified, use the optimal weight matrix.
        save_csv (boolean): save the sensitivity measures as csv tables.

    Returns:
        sensitivity (list of pd.DataFrame)
    """

    func1_kwargs = {} if func1_kwargs is None else func1_kwargs
    func2_kwargs = {} if func2_kwargs is None else func2_kwargs

    derivative_dict = first_derivative(
        func=moment_func,
        params=params,
        func_kwargs=func1_kwargs,
    )

    g = derivative_dict["derivative"]

    if isinstance(g, (pd.Series, pd.DataFrame)):
        g = g.to_numpy()

    s = _calc_moments_variance(moment_contributions_func, params, func2_kwargs)

    weight_opt = np.linalg.inv(s)
    sigma_opt = _calc_estimator_variance(g, s, weight_opt)

    if weight_matrix is None:
        weight_matrix = np.copy(weight_opt)
        sigma = np.copy(sigma_opt)
    else:
        sigma = _calc_estimator_variance(g, s, weight_matrix)

    m1 = _calc_sensitivity_m1(g, weight_matrix)
    m1 = pd.DataFrame(data=m1, index=params.index)

    m2 = _calc_sensitivity_m2(g, sigma_opt, weight_opt)
    epsilon2 = _calc_sensitivity_epsilon(m2, s, sigma_opt)
    epsilon2.index = params.index

    m3 = _calc_sensitivity_m3(m1, weight_matrix)
    epsilon3 = _calc_sensitivity_epsilon(m3, s, sigma)
    epsilon3.index = params.index

    m4 = _calc_sensitivity_m4(g, s, sigma, weight_matrix)
    epsilon4 = _calc_sensitivity_epsilon(m4, 1, sigma)
    epsilon4.index = params.index

    m5 = _calc_sensitivity_m5(g, s, sigma_opt)
    epsilon5 = _calc_sensitivity_epsilon(m5, 1, sigma_opt)
    epsilon5.index = params.index

    m6 = _calc_sensitivity_m6(g, s, sigma, weight_matrix)
    epsilon6 = _calc_sensitivity_epsilon(m6, weight_matrix, sigma)
    epsilon6.index = params.index

    res_list = [m1, epsilon2, epsilon3, epsilon4, epsilon5, epsilon6]
    sensitivity = dict(zip(EXPLANATIONS, res_list))

    return sensitivity


def _sandwich(a, b):
    """calculate the sandwich product of two matrices,
    a.T * b * a.

    args:
        a, b (np.ndarray or pd.DataFrame)

    Return:
        sandwich (np.array)
    """

    bread = np.copy(a)
    meat = np.copy(b)

    sandwich = bread.T @ meat @ bread

    return sandwich


def _sandwich_plus(a, b, c):
    """calculate the sandwich product of three matrices,
    a.T * b * c * b * a.

    args:
        a (np.ndarray or pd.DataFrame)
        b, c (np.ndarray or pd.DataFrame): squrare matrix

    Return:
        sandwich (np.array)
    """

    bread = np.copy(a)
    salad = np.copy(b)
    meat = np.copy(c)

    sandwich = bread.T @ salad @ meat @ salad @ bread

    return sandwich


def _calc_moments_variance(moment_contributions_func, params, func2_kwargs):
    """calculate asymptotic variance-covariance matrix of the sample moments,
    s := Var(g) = E[g'],
    which is also the inverse of the optimal weight matrix.

    args:
        moment_contributions_func (function): moment function (actual values)
        params (pd.DataFrame): see :ref:`params`
        func2_kwargs (dict): additional positional arguments for
            moment_contributions_func.

    Return:
        s (np.array)
    """

    mom_value = moment_contributions_func(params, **func2_kwargs)
    mom_value = mom_value.to_numpy()

    s = np.cov(mom_value, ddof=0)

    return s


def _calc_estimator_variance(g, s, weight_matrix):
    """calculates the covariance matrix of gm estimator.
    Defined as: sigma = (g'Wg)^(-1).g'WsWg.(g'Wg)^(-1)
    When using optimal weighting matrix, W = s^(-1),
    is equvalent to: sigma = (g'Wg)^(-1)

    args:
        weight_matrix (np.array or pd.DataFrame): user-defined weight matrix
        g (np.array or pd.DataFrame): Jacobian
        s (np.array or pd.DataFrame): covariance matrix of the sample moments

    Return:
        sigma (np.array): covariance matrix of gm estimator
    """

    bread = _sandwich(g, weight_matrix)
    bread = np.linalg.inv(bread)

    meat = _sandwich_plus(g, weight_matrix, s)

    sigma = _sandwich(bread, meat)

    return sigma


def _calc_sensitivity_epsilon(m, s, sigma):
    """calculate the epsilon for m2 through m6.
    Note that epsilon2, 3, 6 are elasticities,
    and epsilon4, 5 are the relative changes in the asymptotic
    variance compared to when all moments included.

    args:
        m (np.array): auxiliary measurement, can be m2~6.
        s: can be s (np.array), weight_matrix (np.array) or 1 (int).
        sigma (np.array): the variance-covariance matrix of gm estimator,
                          can be sigma or sigma_opt.

    Return:
        epsilon (pd.DataFrame)
    """

    epsilon = np.copy(m)

    for j in range(len(epsilon)):
        epsilon[j] = np.divide(epsilon[j], np.diagonal(sigma)[j])

        if type(s) is not int:
            for k in range(len(epsilon[j])):
                epsilon[j, k] = epsilon[j, k] * s[k, k]

    epsilon = pd.DataFrame(data=epsilon)

    return epsilon


def _calc_sensitivity_m1(g, weight_matrix):
    """calculate m1, the original local sensitivity measure
    for each parameter wrt each moment.

    args:
        g (np.array or pd.DataFrame)
        weight_matrix (np.array)

    Return:
        m1 (np.array)
    """

    gwg = _sandwich(g, weight_matrix)
    gwg_inverse = np.linalg.inv(gwg)
    m1 = -gwg_inverse @ g.T @ weight_matrix

    return m1


def _calc_sensitivity_m2(g, sigma_opt, optimal_weight_matrix):
    """calculate m2, the lost precision in sigma if
    the k-th moment is subject to additional noise,
    using optimal weight matrix.

    Defined as: m2 = d sigma_opt/d s

    args:
        g (np.array or pd.DataFrame): Jacobian
        sigma_opt (np.array)
        optimal_weight_matrix (np.array)

    Return:
        m2 (np.array)
    """

    m2 = []

    for k in range(len(optimal_weight_matrix)):
        mask_matrix_o = np.zeros(shape=optimal_weight_matrix.shape)
        mask_matrix_o[k, k] = 1

        meat = _sandwich_plus(g, optimal_weight_matrix, mask_matrix_o)

        m2k = sigma_opt @ meat @ sigma_opt
        m2k = m2k.diagonal()

        m2.append(m2k)

    m2 = np.array(m2).T

    return m2


def _calc_sensitivity_m3(m1, weight_matrix):
    """calculate m3, the lost precision in sigma if
    the k-th moment is subject to additional noise,
    using non-optimal weight matrix.

    Defined as: m3 = d sigma/d s

    args:
        m1 (pd.DataFrame)
        weight_matrix (np.array): user-defined weight matrix

    Return:
        m3 (np.array)
    """

    m3 = []

    for k in range(len(weight_matrix)):
        mask_matrix_o = np.zeros(shape=weight_matrix.shape)
        mask_matrix_o[k, k] = 1

        m3k = _sandwich(m1.T, mask_matrix_o)
        m3k = m3k.diagonal()

        m3.append(m3k)

    m3 = np.array(m3).T

    return m3


def _calc_sensitivity_m4(g, s, sigma, weight_matrix):
    """calculates the change in sigma
    if completely exclude the k-th moment.

    args:
        g (np.array): Jacobian
        s (np.array): asymptotic variance-covariance matrix of the sample moments
        sigma
        weight_matrix (np.array)

    Return:
        m4 (np.array)
    """

    m4 = []

    for k in range(len(weight_matrix)):
        weight_tilde_k = np.copy(weight_matrix)
        weight_tilde_k[k, :] = 0
        weight_tilde_k[:, k] = 0

        sigma_tilde_k = _calc_estimator_variance(g, s, weight_tilde_k)

        m4k = sigma_tilde_k - sigma
        m4k = m4k.diagonal()

        m4.append(m4k)

    m4 = np.array(m4).T

    return m4


def _calc_sensitivity_m5(g, s, sigma_opt):
    """compare the precision of gm estimator
    with or without including the k-th moment.

    args:
        g (np.array or pd.DataFrame): Jacobian
        s (np.array): covariance matrix of the sample moments
        sigma_opt (np.array)

    Return:
        m5 (np.array)
    """

    m5 = []

    for k in range(len(s)):
        g_k = np.copy(g)
        g_k = np.delete(g_k, k, axis=0)

        s_k = np.copy(s)
        s_k = np.delete(s_k, k, axis=0)
        s_k = np.delete(s_k, k, axis=1)

        sigma_k = _sandwich(g_k, np.linalg.inv(s_k))
        sigma_k = np.linalg.inv(sigma_k)

        m5k = sigma_k - sigma_opt
        m5k = m5k.diagonal()

        m5.append(m5k)

    m5 = np.array(m5).T

    return m5


def _calc_sensitivity_m6(g, s, sigma, weight_matrix):
    """calculates how far theweight matrix is to being optimal.

    args:
        g (np.array): Jacobian
        s (np.array): asymptotic variance-covariance matrix of the sample moments
        sigma
        weight_matrix (np.array)

    Return:
        m6 (np.array)
    """

    gwg_inverse = _sandwich(g, weight_matrix)
    gwg_inverse = np.linalg.pinv(gwg_inverse)

    m6 = []

    for k in range(len(weight_matrix)):
        mask_matrix_o = np.zeros(shape=weight_matrix.shape)
        mask_matrix_o[k, k] = 1

        m6k_1 = gwg_inverse @ _sandwich(g, mask_matrix_o) @ sigma
        m6k_2 = gwg_inverse @ g.T @ mask_matrix_o @ s @ weight_matrix @ g @ gwg_inverse
        m6k_3 = gwg_inverse @ g.T @ weight_matrix @ s @ mask_matrix_o @ g @ gwg_inverse
        m6k_4 = sigma @ _sandwich(g, mask_matrix_o) @ gwg_inverse

        m6k = -m6k_1 + m6k_2 + m6k_3 - m6k_4
        m6k = m6k.diagonal()

        m6.append(m6k)

    m6 = np.array(m6).T

    return m6
