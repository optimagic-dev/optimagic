"""Linear algebra on the outer product of gradients (OPG) of the BHHH algorithm.

The OPG B = J'J of the n x k Jacobian J of the contributions replaces the Hessian. All
operations work on the k x k matrix B, so their cost does not depend on n.

Rank decisions are made on the OPG after scaling its columns to unit length (Jacobi
scaling with the column norms of J, as in MINPACK and TOMS 717), because thresholds
on the unscaled matrix depend on the units of the parameters. Small eigenvalues of the
scaled OPG are floored instead of truncated, which avoids minimizing in a subspace and
stopping short of the optimum (:cite:`Dennis1981`).

"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

WEAK_IDENTIFICATION_THRESHOLD = 1e-10
"""Eigenvalues of the scaled OPG below this times the largest one are reported."""

LOADING_THRESHOLD = 0.1
"""Parameters with an absolute eigenvector loading above this are reported."""


@dataclass(frozen=True)
class OPGFactorization:
    gradient: NDArray[np.float64]
    """Gradient g = J'1 of the objective, shape (k,)."""

    scale: NDArray[np.float64]
    """Column norms of the Jacobian with zeros replaced by one, shape (k,)."""

    eigenvalues: NDArray[np.float64]
    """Eigenvalues of the scaled OPG in ascending order, shape (k,)."""

    eigenvectors: NDArray[np.float64]
    """Orthonormal eigenvectors of the scaled OPG as columns, shape (k, k)."""

    floor: float
    """Lower limit of the eigenvalues used in steps and decrements."""

    zero_columns: NDArray[np.bool_]
    """Whether a column of the Jacobian is zero, shape (k,)."""

    @property
    def floored_eigenvalues(self) -> NDArray[np.float64]:
        """Eigenvalues raised to the floor, shape (k,)."""
        return np.maximum(self.eigenvalues, self.floor)

    @property
    def scaled_gradient(self) -> NDArray[np.float64]:
        """Gradient in scaled parameters, shape (k,)."""
        return self.gradient / self.scale


def factorize_opg(jac: NDArray[np.float64]) -> OPGFactorization:
    """Compute the eigendecomposition of the column-scaled OPG.

    Eigenvalues are floored at k * eps times the largest eigenvalue, which corresponds
    to a condition number of the scaled Jacobian of about 1e7.

    Args:
        jac: Jacobian of the contributions, shape (n, k).

    Returns:
        The factorization of the OPG.

    """
    n_params = jac.shape[1]
    gradient = jac.sum(axis=0)
    opg = jac.T @ jac
    norms = np.sqrt(np.diag(opg))
    zero_columns = norms == 0
    scale = np.where(zero_columns, 1.0, norms)
    scaled_opg = opg / np.outer(scale, scale)

    # Zero columns are decoupled from the rest of the OPG. Decomposing the other block
    # separately keeps rounding errors from moving the corresponding parameters.
    nonzero = np.flatnonzero(~zero_columns)
    block_values, block_vectors = np.linalg.eigh(scaled_opg[np.ix_(nonzero, nonzero)])
    eigenvalues = np.concatenate([np.zeros(zero_columns.sum()), block_values])
    eigenvectors = np.zeros((n_params, n_params))
    eigenvectors[np.flatnonzero(zero_columns), np.arange(zero_columns.sum())] = 1.0
    eigenvectors[np.ix_(nonzero, np.arange(zero_columns.sum(), n_params))] = (
        block_vectors
    )
    order = np.argsort(eigenvalues, kind="stable")
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

    largest = float(eigenvalues[-1])
    floor = n_params * np.finfo(float).eps * largest if largest > 0 else 1.0
    return OPGFactorization(
        gradient=gradient,
        scale=scale,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        floor=floor,
        zero_columns=zero_columns,
    )


def unconstrained_step(
    factorization: OPGFactorization,
) -> tuple[NDArray[np.float64], float]:
    """Compute the BHHH step -B⁻¹g and the decrement g'B⁻¹g.

    B is the OPG with floored eigenvalues.

    Args:
        factorization: The factorization of the OPG.

    Returns:
        - The step, shape (k,).
        - The decrement.

    """
    v = factorization.eigenvectors
    lam = factorization.floored_eigenvalues
    projected = v.T @ factorization.scaled_gradient
    scaled_step = -v @ (projected / lam)
    step = scaled_step / factorization.scale
    decrement = float(projected @ (projected / lam))
    return step, decrement


def rank(factorization: OPGFactorization) -> int:
    """Return the numerical rank of the scaled OPG."""
    return int(np.sum(factorization.eigenvalues > factorization.floor))


def condition_number(factorization: OPGFactorization) -> float:
    """Return the condition number of the scaled OPG (inf if it is singular)."""
    smallest = factorization.eigenvalues[0]
    if smallest <= 0:
        return np.inf
    return float(factorization.eigenvalues[-1] / smallest)


def weakly_identified_directions(
    factorization: OPGFactorization,
) -> list[NDArray[np.int64]]:
    """Find the parameters involved in nearly singular directions of the scaled OPG.

    Args:
        factorization: The factorization of the OPG.

    Returns:
        One array per eigenvalue below WEAK_IDENTIFICATION_THRESHOLD times the largest
        eigenvalue. Each array contains the positions of the parameters whose absolute
        loading on the eigenvector exceeds LOADING_THRESHOLD, ordered by the absolute
        loading.

    """
    largest = max(float(factorization.eigenvalues[-1]), 0.0)
    threshold = WEAK_IDENTIFICATION_THRESHOLD * largest
    out = []
    for value, vector in zip(
        factorization.eigenvalues, factorization.eigenvectors.T, strict=True
    ):
        if value > threshold:
            break
        loadings = np.abs(vector)
        involved = np.flatnonzero(loadings > LOADING_THRESHOLD)
        out.append(involved[np.argsort(-loadings[involved])])
    return out
