import numpy as np
from numdifftools import Richardson

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

    sequence, steps = first_derivative(f, np.ones(3), n_steps=4)

    steps = steps.pos[:, 0]

    r = Richardson()

    result = r(sequence, steps)

    compare = richardson_extrapolation(sequence, steps, num_terms=1)
