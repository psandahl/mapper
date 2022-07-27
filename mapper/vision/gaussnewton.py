from typing import Callable
import numpy as np

from typing import Callable


class GaussNewton():
    def __init__(self,
                 cost_function=Callable,
                 max_iter: int = 10,
                 eps_0: float = 1e-6,
                 eps_1: float = 1e-9):
        self.cost_function = cost_function
        self.max_iter = max_iter
        self.step = 1e-06,
        self.eps_0 = eps_0
        self.eps_1 = eps_1

        self.x = None
        self.y = None
        self.coefficients = None
        self.residuals = None
        self.Jacobian = None

        self.iter = 0

    def solve(self,
              x: np.ndarray,
              y: np.ndarray,
              initial_guess: np.ndarray,
              step: float = 1e-09) -> bool:
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(initial_guess, np.ndarray)

        self.x = x
        self.y = y
        self.coefficients = initial_guess
        self.step = step

        prev_err = np.inf
        for self.iter in range(self.max_iter):
            self.residuals = self._compute_residuals(self.coefficients)
            if len(self.residuals) < len(self.coefficients):
                print('Number of residuals cannot be less than number of coefficients')
                return False

            err = np.sqrt(np.sum(self.residuals ** 2))

            if err < self.eps_0:
                print('Converged! Error is less than eps_0')
                return True

            err_diff = np.abs(prev_err - err)
            if err_diff < self.eps_1:
                print('Converged! Difference between two iterations is less than eps_1')
                return True

            prev_err = err

            self.Jacobian = self._compute_jacobian(
                self.coefficients, self.residuals)

            MoorePenrose = np.linalg.inv(self.Jacobian.T @
                                         self.Jacobian) @ self.Jacobian.T
            self.coefficients -= MoorePenrose @ self.residuals

        print('Did not converge!')

        return False

    def _compute_residuals(self, coeffients: np.ndarray) -> np.ndarray:
        assert isinstance(coeffients, np.ndarray)
        assert isinstance(self.x, np.ndarray)

        y_estimate = self.cost_function(self.x, coeffients)

        norms = list()
        for index, estimate in enumerate(y_estimate):
            norms.append(np.linalg.norm(estimate - self.y[index]))

        return np.array(norms)

    def _compute_jacobian(self, c_0: np.ndarray, r_0: np.ndarray) -> np.ndarray:
        assert isinstance(c_0, np.ndarray)
        assert isinstance(r_0, np.ndarray)

        jacobian = list()
        for index, _ in enumerate(c_0):
            c_1 = c_0.copy()
            c_1[index] += self.step
            r_1 = self._compute_residuals(c_1)
            derivative = (r_1 - r_0) / self.step
            jacobian.append(derivative)

        return np.array(jacobian).T
