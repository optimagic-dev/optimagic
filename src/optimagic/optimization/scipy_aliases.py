from optimagic.utilities import propose_alternatives


def map_method_to_algorithm(method):
    implemented = {
        "Nelder-Mead": "scipy_neldermead",
        "Powell": "scipy_powell",
        "CG": "scipy_conjugate_gradient",
        "BFGS": "scipy_bfgs",
        "Newton-CG": "scipy_newton_cg",
        "L-BFGS-B": "scipy_lbfgsb",
        "TNC": "scipy_truncated_newton",
        "COBYLA": "scipy_cobyla",
        "SLSQP": "scipy_slsqp",
        "trust-constr": "scipy_trust_constr",
    }

    not_implemented = {
        "dogleg": "scipy_dogleg",
        "trust-ncg": "scipy_trust_ncg",
        "trust-exact": "scipy_trust_exact",
        "trust-krylov": "scipy_trust_krylov",
        "COBYQA": "scipy_cobyqa",
    }

    if method in implemented:
        algo = implemented[method]
    elif method in not_implemented:
        msg = (
            f"The method {method} is not yet wrapped in optimagic. Create an issue on "
            "https://github.com/OpenSourceEconomics/optimagic/ if you have urgent need "
            "for this method."
        )
        raise NotImplementedError(msg)
    else:
        alt = propose_alternatives(method, list(implemented) + list(not_implemented))
        msg = (
            "method is an alias for algorithm to select the scipy optimizers under "
            f"their original name. {method} is not a valid scipy algorithm name. "
            f"Did you mean {alt}?"
        )
        raise ValueError(msg)
    return algo
