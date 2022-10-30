"""Numba implementation of Lapack's potrf function."""
import numpy as np
from numba import njit
from numba import types
from numba.extending import overload
from numba.np.linalg import _check_finite_matrix
from numba.np.linalg import _check_linalg_matrix
from numba.np.linalg import _dummy_liveness_func
from numba.np.linalg import _LAPACK
from numba.np.linalg import ensure_lapack
from numba.np.linalg import get_blas_kind


def my_dpotrf(a):

    pass


@overload(my_dpotrf)
def ol_my_dpotrf(a):

    if not isinstance(a, types.Array):
        return None  # needs to be an array

    if not isinstance(a.dtype, types.Float):
        if a.dtype.bitwidth != 64:
            return None  # needs to be an array

    ensure_lapack()

    _check_linalg_matrix(a, "my_dpotrf")

    numba_xxpotrf = _LAPACK().numba_xxpotrf(a.dtype)

    kind = ord(get_blas_kind(a.dtype, "my_dpotrf"))

    def impl(a):
        n = a.shape[-1]
        if a.shape[-2] != n:
            msg = "Last 2 dimensions of the array must be square."
            raise np.linalg.LinAlgError(msg)

        _check_finite_matrix(a)

        acpy = np.ascontiguousarray(a).copy()

        r = numba_xxpotrf(
            kind,  # kind
            ord("U"),  # uplo: https://netlib.org/lapack/lug/node123.html, "U" or "P"
            n,  # n
            acpy.ctypes,  # a
            n,  # lda
        )

        # help liveness analysis
        _dummy_liveness_func([acpy.size])
        return acpy, r

    return impl


@njit
def compute_cholesky_factorization(a):
    out, info = my_dpotrf(a)
    out = out.T
    n = len(out)
    for i in range(n):
        for j in range(i):
            out[i, j] = 0.0
    return out, info


np.random.seed(0)
n = 4
x = np.random.random((n, n))
a = np.dot(x, x.T)
