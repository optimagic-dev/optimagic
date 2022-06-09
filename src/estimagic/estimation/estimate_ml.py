from dataclasses import dataclass
from typing import Any
from typing import Union

import numpy as np
import pandas as pd
from estimagic.differentiation.derivatives import first_derivative
from estimagic.differentiation.derivatives import second_derivative
from estimagic.exceptions import InvalidFunctionError
from estimagic.exceptions import NotAvailableError
from estimagic.inference.ml_covs import cov_cluster_robust
from estimagic.inference.ml_covs import cov_hessian
from estimagic.inference.ml_covs import cov_jacobian
from estimagic.inference.ml_covs import cov_robust
from estimagic.inference.ml_covs import cov_strata_robust
from estimagic.inference.shared import calculate_ci
from estimagic.inference.shared import calculate_inference_quantities
from estimagic.inference.shared import calculate_p_values
from estimagic.inference.shared import check_is_optimized_and_derivative_case
from estimagic.inference.shared import get_derivative_case
from estimagic.inference.shared import transform_covariance
from estimagic.optimization.optimize import maximize
from estimagic.optimization.optimize_result import OptimizeResult
from estimagic.parameters.block_trees import block_tree_to_matrix
from estimagic.parameters.block_trees import matrix_to_block_tree
from estimagic.parameters.conversion import Converter
from estimagic.parameters.conversion import get_converter
from estimagic.shared.check_option_dicts import check_numdiff_options
from estimagic.shared.check_option_dicts import check_optimization_options
from estimagic.utilities import to_pickle


def estimate_ml(
    loglike,
    params,
    optimize_options,
    *,
    lower_bounds=None,
    upper_bounds=None,
    constraints=None,
    logging=False,
    log_options=None,
    loglike_kwargs=None,
    numdiff_options=None,
    jacobian=None,
    jacobian_kwargs=None,
    hessian=None,
    hessian_kwargs=None,
    design_info=None,
):
    """Do a maximum likelihood (ml) estimation.

    This is a high level interface of our lower level functions for maximization,
    numerical differentiation and inference. It does the full workflow for maximum
    likelihood estimation with just one function call.

    While we have good defaults, you can still configure each aspect of each step
    via the optional arguments of this function. If you find it easier to do the
    maximization separately, you can do so and just provide the optimal parameters as
    ``params`` and set ``optimize_options=False``.

    Args:
        loglike (callable): Likelihood function that takes a params (and potentially
            other keyword arguments) and returns a dictionary that has at least the
            entries "value" (a scalar float) and "contributions" (a 1d numpy array or
            pytree) with the log likelihood contribution per individual.
        params (pytree): A pytree containing the estimated or start parameters of the
            likelihood model. If the supplied parameters are estimated parameters, set
            optimize_options to False. Pytrees can be a numpy array, a pandas Series, a
            DataFrame with "value" column, a float and any kind of (nested) dictionary
            or list containing these elements. See :ref:`params` for examples.
        optimize_options (dict, str or False): Keyword arguments that govern the
            numerical optimization. Valid entries are all arguments of
            :func:`~estimagic.optimization.optimize.minimize` except for those that are
            passed explicilty to ``estimate_ml``. If you pass False as optimize_options
            you signal that ``params`` are already the optimal parameters and no
            numerical optimization is needed. If you pass a str as optimize_options it
            is used as the ``algorithm`` option.
        lower_bounds (pytree): A pytree with the same structure as params with lower
            bounds for the parameters. Can be ``-np.inf`` for parameters with no lower
            bound.
        upper_bounds (pytree): As lower_bounds. Can be ``np.inf`` for parameters with
            no upper bound.
        constraints (list, dict): List with constraint dictionaries or single dict.
            See :ref:`constraints`.
        logging (pathlib.Path, str or False): Path to sqlite3 file (which typically has
            the file extension ``.db``. If the file does not exist, it will be created.
            The dashboard can only be used when logging is used.
        log_options (dict): Additional keyword arguments to configure the logging.
            - "fast_logging": A boolean that determines if "unsafe" settings are used
            to speed up write processes to the database. This should only be used for
            very short running criterion functions where the main purpose of the log
            is a real-time dashboard and it would not be catastrophic to get a
            corrupted database in case of a sudden system shutdown. If one evaluation
            of the criterion function (and gradient if applicable) takes more than
            100 ms, the logging overhead is negligible.
            - "if_table_exists": (str) One of "extend", "replace", "raise". What to
            do if the tables we want to write to already exist. Default "extend".
            - "if_database_exists": (str): One of "extend", "replace", "raise". What to
            do if the database we want to write to already exists. Default "extend".
        loglike_kwargs (dict): Additional keyword arguments for loglike.
        numdiff_options (dict): Keyword arguments for the calculation of numerical
            derivatives for the calculation of standard errors. See
            :ref:`first_derivative` for details.
        jacobian (callable or None): A function that takes ``params`` and potentially
            other keyword arguments and returns the jacobian of loglike["contributions"]
            with respect to the params. Note that you only need to pass a Jacobian
            function if you have a closed form Jacobian. If you pass None, a numerical
            Jacobian will be calculated.
        jacobian_kwargs (dict): Additional keyword arguments for the Jacobian function.
        hessian (callable or None or False): A function that takes ``params`` and
            potentially other keyword arguments and returns the Hessian of
            loglike["value"] with respect to the params.  If you pass None, a numerical
            Hessian will be calculated. If you pass ``False``, you signal that no
            Hessian should be calculated. Thus, no result that requires the Hessian will
            be calculated.
        hessian_kwargs (dict): Additional keyword arguments for the Hessian function.
        design_info (pandas.DataFrame): DataFrame with one row per observation that
            contains some or all of the variables "psu" (primary sampling unit),
            "strata" and "fpc" (finite population corrector). See
            :ref:`robust_likelihood_inference` for details.

    Returns:
        LikelihoodResult: A LikelihoodResult object.

    """
    # ==================================================================================
    # Check and process inputs
    # ==================================================================================

    is_optimized = optimize_options is False

    if not is_optimized:
        if isinstance(optimize_options, str):
            optimize_options = {"algorithm": optimize_options}

        check_optimization_options(
            optimize_options,
            usage="estimate_ml",
            algorithm_mandatory=True,
        )

    jac_case = get_derivative_case(jacobian)
    hess_case = get_derivative_case(hessian)

    check_is_optimized_and_derivative_case(is_optimized, jac_case)
    check_is_optimized_and_derivative_case(is_optimized, hess_case)

    check_numdiff_options(numdiff_options, "estimate_ml")
    numdiff_options = {} if numdiff_options in (None, False) else numdiff_options
    loglike_kwargs = {} if loglike_kwargs is None else loglike_kwargs
    constraints = [] if constraints is None else constraints
    jacobian_kwargs = {} if jacobian_kwargs is None else jacobian_kwargs
    hessian_kwargs = {} if hessian_kwargs is None else hessian_kwargs

    # ==================================================================================
    # Calculate estimates via maximization (if necessary)
    # ==================================================================================

    if is_optimized:
        estimates = params
        opt_res = None
    else:
        opt_res = maximize(
            criterion=loglike,
            criterion_kwargs=loglike_kwargs,
            params=params,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            constraints=constraints,
            logging=logging,
            log_options=log_options,
            **optimize_options,
        )
        estimates = opt_res.params

    # ==================================================================================
    # Do first function evaluations at estimated parameters
    # ==================================================================================

    try:
        loglike_eval = loglike(estimates, **loglike_kwargs)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        msg = "Error while evaluating loglike at estimated params."
        raise InvalidFunctionError(msg) from e

    if callable(jacobian):
        try:
            jacobian_eval = jacobian(estimates, **jacobian_kwargs)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            msg = "Error while evaluating closed form jacobian at estimated params."
            raise InvalidFunctionError(msg) from e
    else:
        jacobian_eval = None

    if callable(hessian):
        try:
            hessian_eval = hessian(estimates, **hessian_kwargs)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            msg = "Error while evaluating closed form hessian at estimated params."
            raise InvalidFunctionError(msg) from e
    else:
        hessian_eval = None

    # ==================================================================================
    # Get the converter for params and function outputs
    # ==================================================================================

    converter, internal_estimates = get_converter(
        func=loglike,
        params=estimates,
        constraints=constraints,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        func_eval=loglike_eval,
        primary_key="contributions",
        scaling=False,
        scaling_options=None,
        derivative_eval=jacobian_eval,
    )

    # ==================================================================================
    # Calculate internal jacobian
    # ==================================================================================

    if jac_case == "closed-form":
        x = converter.params_to_internal(estimates)
        int_jac = converter.derivative_to_internal(jacobian_eval, x)
    elif jac_case == "numerical":

        def func(x):
            p = converter.params_from_internal(x)
            loglike_eval = loglike(p, **loglike_kwargs)["contributions"]
            out = converter.func_to_internal(loglike_eval)
            return out

        jac_res = first_derivative(
            func=func,
            params=internal_estimates.values,
            lower_bounds=internal_estimates.lower_bounds,
            upper_bounds=internal_estimates.upper_bounds,
            **numdiff_options,
        )

        int_jac = jac_res["derivative"]
    else:
        int_jac = None

    if constraints in [None, []] and jacobian_eval is None and int_jac is not None:
        loglike_contribs = loglike_eval
        if isinstance(loglike_contribs, dict) and "contributions" in loglike_contribs:
            loglike_contribs = loglike_contribs["contributions"]

        jacobian_eval = matrix_to_block_tree(
            int_jac,
            outer_tree=loglike_contribs,
            inner_tree=estimates,
        )

    if jacobian_eval is None:
        _no_jac_reason = (
            "no closed form jacobian was provided and there are constraints"
        )
    else:
        _no_jac_reason = None
    # ==================================================================================
    # Calculate internal Hessian
    # ==================================================================================

    if hess_case == "skip":
        int_hess = None
    elif hess_case == "numerical":

        def func(x):
            p = converter.params_from_internal(x)
            loglike_eval = loglike(p, **loglike_kwargs)["value"]
            out = converter.func_to_internal(loglike_eval)
            return out

        hess_res = second_derivative(
            func=func,
            params=internal_estimates.values,
            lower_bounds=internal_estimates.lower_bounds,
            upper_bounds=internal_estimates.upper_bounds,
            **numdiff_options,
        )
        int_hess = hess_res["derivative"]
    elif hess_case == "closed-form" and constraints:
        raise NotImplementedError(
            "Closed-form Hessians are not yet compatible with constraints."
        )
    elif hess_case == "closed-form":
        int_hess = block_tree_to_matrix(
            hessian_eval,
            outer_tree=params,
            inner_tree=params,
        )
    else:
        raise ValueError()

    if constraints in [None, []] and hessian_eval is None and int_hess is not None:
        hessian_eval = matrix_to_block_tree(
            int_hess,
            outer_tree=params,
            inner_tree=params,
        )

    if hessian_eval is None:
        if hess_case == "skip":
            _no_hess_reason = "the hessian calculation was explicitly skipped."
        else:
            _no_hess_reason = (
                "no closed form hessian was provided and there are constraints"
            )
    else:
        _no_hess_reason = None

    # ==================================================================================
    # create a LikelihoodResult object
    # ==================================================================================

    res = LikelihoodResult(
        params=estimates,
        _converter=converter,
        optimize_result=opt_res,
        _jacobian=jacobian_eval,
        _no_jacobian_reason=_no_jac_reason,
        _hessian=hessian_eval,
        _no_hessian_reason=_no_hess_reason,
        _internal_jacobian=int_jac,
        _internal_hessian=int_hess,
        _design_info=design_info,
        _flat_params=internal_estimates,
        _has_constraints=constraints not in [None, []],
    )

    return res


@dataclass
class LikelihoodResult:
    params: Any
    _flat_params: Any
    _converter: Converter
    _has_constraints: bool
    optimize_result: Union[OptimizeResult, None] = None
    _jacobian: Any = None
    _no_jacobian_reason: Union[str, None] = None
    _hessian: Any = None
    _no_hessian_reason: Union[str, None] = None
    _internal_jacobian: Union[np.ndarray, None] = None
    _internal_hessian: Union[np.ndarray, None] = None
    _design_info: Union[pd.DataFrame, None] = None

    def __post_init__(self):
        if self._internal_jacobian is None and self._internal_hessian is None:
            raise ValueError(
                "At least one of _internal_jacobian or _internal_hessian must be "
                "not None."
            )

        elif self._internal_jacobian is None:
            valid_methods = ["hessian"]
        elif self._internal_hessian is None:
            valid_methods = ["jacobian"]
        else:
            valid_methods = ["jacobian", "hessian", "robust"]
            if self._design_info is not None:
                if "psu" in self._design_info:
                    valid_methods.append("cluster_robust")
                if {"strata", "psu", "fpc"}.issubset(self._design_info):
                    valid_methods.append("strata_robust")

        self._valid_methods = set(valid_methods)

    def _get_free_cov(
        self,
        method,
        n_samples,
        bounds_handling,
        seed,
    ):
        if method not in self._valid_methods:
            raise ValueError()

        internal_jac = self._internal_jacobian
        internal_hess = self._internal_hessian
        design_info = self._design_info
        converter = self._converter
        flat_params = self._flat_params

        if method == "jacobian":
            int_cov = cov_jacobian(internal_jac)
        elif method == "hessian":
            int_cov = cov_hessian(internal_hess)
        elif method == "robust":
            int_cov = cov_robust(jac=internal_jac, hess=internal_hess)
        elif method == "cluster_robust":
            int_cov = cov_cluster_robust(
                jac=internal_jac, hess=internal_hess, design_info=design_info
            )
        elif method == "strata_robust":
            int_cov = cov_strata_robust(
                jac=internal_jac, hess=internal_hess, design_info=design_info
            )
        else:
            raise ValueError(f"Invalid method: {method}")

        np.random.seed(seed)

        free_cov = transform_covariance(
            flat_params=flat_params,
            internal_cov=int_cov,
            converter=converter,
            n_samples=n_samples,
            bounds_handling=bounds_handling,
        )

        return free_cov

    @property
    def jacobian(self):
        if self._jacobian is None:
            raise NotAvailableError(
                f"No jacobian is available because {self._no_jacobian_reason}."
            )

        return self._jacobian

    @property
    def hessian(self):
        if self._hessian is None:
            raise NotAvailableError(
                f"No hessian is available because {self._no_hessian_reason}."
            )

        return self._hessian

    @property
    def _se(self):
        return self.se()

    @property
    def _cov(self):
        return self.cov()

    @property
    def _summary(self):
        return self.summary()

    @property
    def _ci(self):
        return self.ci()

    @property
    def _p_values(self):
        return self.p_values()

    def se(
        self,
        method="jacobian",
        n_samples=10_000,
        bounds_handling="clip",
        seed=None,
    ):
        """Calculate standard errors.

        Args:
            method (str): One of "jacobian", "hessian", "robust", "cluster_robust",
                "strata_robust". Default "jacobian". "cluster_robust" is only available
                if design_info containts a columns called "psu" that identifies the
                primary sampling unit. "strata_robust" is only available if the columns
                "strata", "fpc" and "psu" are in design_info.
            n_samples (int): Number of samples used to transform the covariance matrix
                of the internal parameter vector into the covariance matrix of the
                external parameters. For background information about internal and
                external params see :ref:`implementation_of_constraints`. This is only
                used if you are using constraints.
            bounds_handling (str): One of "clip", "raise", "ignore". Determines how
                bounds are handled. If "clip", confidence intervals are clipped at the
                bounds. Standard errors are only adjusted if a sampling step is
                necessary due to additional constraints. If "raise" and any lower or
                upper bound is binding, we raise an Error. If "ignore", boundary
                problems are simply ignored.
            seed (int): Seed for the random number generator. Only used if there are
                transforming constraints.

        Returns:
            Any: A pytree with the same structure as params containing standard errors
            for the parameter estimates.

        """
        free_cov = self._get_free_cov(
            method=method,
            n_samples=n_samples,
            bounds_handling=bounds_handling,
            seed=seed,
        )

        helper = np.full(len(self._flat_params.values), np.nan)
        helper[self._flat_params.free_mask] = np.sqrt(np.diagonal(free_cov))

        out = self._converter.params_from_internal(helper)

        return out

    def cov(
        self,
        method="jacobian",
        n_samples=10_000,
        bounds_handling="clip",
        return_type="pytree",
        seed=None,
    ):
        """Calculate the variance-covariance matrix of the estimated parameters.

        Args:
            method (str): One of "jacobian", "hessian", "robust", "cluster_robust",
                "strata_robust". Default "jacobian". "cluster_robust" is only available
                if design_info containts a columns called "psu" that identifies the
                primary sampling unit. "strata_robust" is only available if the columns
                "strata", "fpc" and "psu" are in design_info.
            n_samples (int): Number of samples used to transform the covariance matrix
                of the internal parameter vector into the covariance matrix of the
                external parameters. For background information about internal and
                external params see :ref:`implementation_of_constraints`. This is only
                used if you are using constraints.
            bounds_handling (str): One of "clip", "raise", "ignore". Determines how
                bounds are handled. If "clip", confidence intervals are clipped at the
                bounds. Standard errors are only adjusted if a sampling step is
                necessary due to additional constraints. If "raise" and any lower or
                upper bound is binding, we raise an Error. If "ignore", boundary
                problems are simply ignored.
            return_type (str): One of "pytree", "array" or "dataframe". Default pytree.
                If "array", a 2d numpy array with the covariance is returned. If
                "dataframe", a pandas DataFrame with parameter names in the
                index and columns are returned.
            seed (int): Seed for the random number generator. Only used if there are
                transforming constraints.

        Returns:
            Any: The covariance matrix of the estimated parameters as block-pytree or
                numpy array.

        """
        free_cov = self._get_free_cov(
            method=method,
            n_samples=n_samples,
            bounds_handling=bounds_handling,
            seed=seed,
        )
        if return_type == "array":
            out = free_cov
        elif return_type == "dataframe":
            free_index = np.array(self._flat_params.names)[self._flat_params.free_mask]
            out = pd.DataFrame(data=free_cov, columns=free_index, index=free_index)
        elif return_type == "pytree":
            if len(free_cov) != len(self._flat_params.values):
                raise NotAvailableError(
                    "Covariance matrices in block-pytree format are only available if "
                    "there are no constraints that reduce the number of free "
                    "parameters."
                )
            out = matrix_to_block_tree(free_cov, self.params, self.params)
        return out

    def summary(
        self,
        method="jacobian",
        n_samples=10_000,
        ci_level=0.95,
        bounds_handling="clip",
        seed=None,
    ):
        """Create a summary of estimation results.

        Args:
            method (str): One of "jacobian", "hessian", "robust", "cluster_robust",
                "strata_robust". Default "jacobian". "cluster_robust" is only available
                if design_info containts a columns called "psu" that identifies the
                primary sampling unit. "strata_robust" is only available if the columns
                "strata", "fpc" and "psu" are in design_info.
            ci_level (float): Confidence level for the calculation of confidence
                intervals. The default is 0.95.
            n_samples (int): Number of samples used to transform the covariance matrix
                of the internal parameter vector into the covariance matrix of the
                external parameters. For background information about internal and
                external params see :ref:`implementation_of_constraints`. This is only
                used if you are using constraints.
            bounds_handling (str): One of "clip", "raise", "ignore". Determines how
                bounds are handled. If "clip", confidence intervals are clipped at the
                bounds. Standard errors are only adjusted if a sampling step is
                necessary due to additional constraints. If "raise" and any lower or
                upper bound is binding, we raise an Error. If "ignore", boundary
                problems are simply ignored.
            seed (int): Seed for the random number generator. Only used if there are
                transforming constraints.


        Returns:
            Any: The estimation summary as pytree of DataFrames.

        """
        free_cov = self._get_free_cov(
            method=method,
            n_samples=n_samples,
            bounds_handling=bounds_handling,
            seed=seed,
        )

        summary = calculate_inference_quantities(
            estimates=self.params,
            internal_estimates=self._flat_params,
            free_cov=free_cov,
            ci_level=ci_level,
        )
        return summary

    def ci(
        self,
        method="jacobian",
        n_samples=10_000,
        ci_level=0.95,
        bounds_handling="clip",
        seed=None,
    ):
        """Calculate confidence intervals.

        Args:
            method (str): One of "jacobian", "hessian", "robust", "cluster_robust",
                "strata_robust". Default "jacobian". "cluster_robust" is only available
                if design_info containts a columns called "psu" that identifies the
                primary sampling unit. "strata_robust" is only available if the columns
                "strata", "fpc" and "psu" are in design_info.
            ci_level (float): Confidence level for the calculation of confidence
                intervals. The default is 0.95.
            n_samples (int): Number of samples used to transform the covariance matrix
                of the internal parameter vector into the covariance matrix of the
                external parameters. For background information about internal and
                external params see :ref:`implementation_of_constraints`. This is only
                used if you are using constraints.
            bounds_handling (str): One of "clip", "raise", "ignore". Determines how
                bounds are handled. If "clip", confidence intervals are clipped at the
                bounds. Standard errors are only adjusted if a sampling step is
                necessary due to additional constraints. If "raise" and any lower or
                upper bound is binding, we raise an Error. If "ignore", boundary
                problems are simply ignored.
            seed (int): Seed for the random number generator. Only used if there are
                transforming constraints.

        Returns:
            Any: Pytree with the same structure as params containing lower bounds of
                confidence intervals.
            Any: Pytree with the same structure as params containing upper bounds of
                confidence intervals.

        """
        free_cov = self._get_free_cov(
            method=method,
            n_samples=n_samples,
            bounds_handling=bounds_handling,
            seed=seed,
        )

        free_values = self._flat_params.values[self._flat_params.free_mask]
        free_se = np.sqrt(np.diagonal(free_cov))

        free_lower, free_upper = calculate_ci(free_values, free_se, ci_level)

        helper = np.full(len(self._flat_params.values), np.nan)
        helper[self._flat_params.free_mask] = free_lower
        lower = self._converter.params_from_internal(helper)

        helper = np.full(len(self._flat_params.values), np.nan)
        helper[self._flat_params.free_mask] = free_upper
        upper = self._converter.params_from_internal(helper)

        return lower, upper

    def p_values(
        self,
        method="jacobian",
        n_samples=10_000,
        bounds_handling="clip",
        seed=None,
    ):
        """Calculate p_values.

        Args:
            method (str): One of "jacobian", "hessian", "robust", "cluster_robust",
                "strata_robust". Default "jacobian". "cluster_robust" is only available
                if design_info containts a columns called "psu" that identifies the
                primary sampling unit. "strata_robust" is only available if the columns
                "strata", "fpc" and "psu" are in design_info.
            ci_level (float): Confidence level for the calculation of confidence
                intervals. The default is 0.95.
            n_samples (int): Number of samples used to transform the covariance matrix
                of the internal parameter vector into the covariance matrix of the
                external parameters. For background information about internal and
                external params see :ref:`implementation_of_constraints`. This is only
                used if you are using constraints.
            bounds_handling (str): One of "clip", "raise", "ignore". Determines how
                bounds are handled. If "clip", confidence intervals are clipped at the
                bounds. Standard errors are only adjusted if a sampling step is
                necessary due to additional constraints. If "raise" and any lower or
                upper bound is binding, we raise an Error. If "ignore", boundary
                problems are simply ignored.
            seed (int): Seed for the random number generator. Only used if there are
                transforming constraints.

        Returns:
            Any: Pytree with the same structure as params containing lower bounds of
                confidence intervals.
            Any: Pytree with the same structure as params containing upper bounds of
                confidence intervals.

        """
        free_cov = self._get_free_cov(
            method=method,
            n_samples=n_samples,
            bounds_handling=bounds_handling,
            seed=seed,
        )

        free_values = self._flat_params.values[self._flat_params.free_mask]
        free_se = np.sqrt(np.diagonal(free_cov))

        free_p_values = calculate_p_values(free_values, free_se)

        helper = np.full(len(self._flat_params.values), np.nan)
        helper[self._flat_params.free_mask] = free_p_values
        out = self._converter.params_from_internal(helper)

        return out

    def to_pickle(self, path):
        """Save the LikelihoodResult object to pickle.

        Args:
            path (str, pathlib.Path): A str or pathlib.path ending in .pkl or .pickle.

        """
        to_pickle(self, path=path)
