import numpy as np
from numdifftools import Richardson
from numpy.testing import assert_array_equal

from estimagic.differentiation.numdiff_np import first_derivative
from estimagic.differentiation.richardson_extrapolation import richardson_extrapolation

if __name__ == "__main__":

    def f(x):
        """f:R^3 -> R^2"""
        x1, x2, x3 = x[0], x[1], x[2]
        y1, y2 = np.sin(x1) + np.cos(x2), np.exp(x3)
        return np.array([y1, y2])

    def fprime(x):
        """Jacobian(f)(x):R^3 -> R^2"""
        x1, x2, x3 = x[0], x[1], x[2]
        jac = np.array([[np.cos(x1), -np.sin(x2), 0], [0, 0, np.exp(x3)]])
        return jac

    for ns in range(2, 5):
        for nt in range(1, ns):
            sequence, steps = first_derivative(f, np.ones(3), n_steps=ns)

            r = Richardson(num_terms=nt)

            result = r(sequence, steps.pos)

            compare = richardson_extrapolation(sequence, steps, num_terms=nt)

            print(assert_array_equal(result[0], compare[0]))
            print(assert_array_equal(result[1], compare[1]))
