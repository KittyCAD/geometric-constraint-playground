"""
Newton-Raphson solver implementation.

This module contains code derived from the newton_faer Rust library
(https://github.com/alexlatif/newton-faer/tree/main), licensed under the MIT License.

Adapted and ported to Python with modifications to prove out the approach of a system
we will eventually actually write in Rust.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Optional, Protocol, Tuple, Union

import numpy as np
from scipy import sparse
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import splu


class MatrixFormat(Enum):
    SPARSE = auto()
    DENSE = auto()
    AUTO = auto()


class Control(Enum):
    CONTINUE = "continue"
    CANCEL = "cancel"


@dataclass
class NewtonConfig:
    """
    Configuration for Newton solver.
    """

    tol: float = 1e-8
    damping: float = 1.0
    max_iter: int = 25
    format: MatrixFormat = MatrixFormat.AUTO
    format_threshold: int = 100

    # Adaptive step control.
    adaptive: bool = False
    min_damping: float = 0.1
    max_damping: float = 1.0
    grow: float = 1.1
    shrink: float = 0.5
    divergence_ratio: float = 3.0
    ls_backtrack: float = 0.5
    ls_max_steps: int = 10

    @classmethod
    def sparse(cls) -> "NewtonConfig":
        # Create config for sparse matrices.
        return cls(format=MatrixFormat.SPARSE)

    @classmethod
    def dense(cls) -> "NewtonConfig":
        # Create config for dense matrices.
        return cls(format=MatrixFormat.DENSE)

    def with_adaptive(self, enabled: bool) -> "NewtonConfig":
        # Enable/disable adaptive damping.
        self.adaptive = enabled
        return self


@dataclass
class IterationStats:
    iter: int
    residual: float
    damping: float


class SolverError(Exception):
    # Exception raised when solver fails.
    def __init__(self, message="Solver operation failed."):
        self.message = message
        super().__init__(self.message)


class LUSolver(ABC):
    @abstractmethod
    def factor(self, matrix: Union[np.ndarray, sparse.csr_matrix]) -> None:
        """
        Factor the matrix for later solving.
        """
        pass

    @abstractmethod
    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """
        Solve the linear system with the previously factored matrix."""
        pass


class DenseLUSolver(LUSolver):
    """
    Dense LU solver using scipy.linalg.lu_factor/lu_solve.
    """

    def __init__(self):
        self.lu_factors = None
        self.pivots = None

    def factor(self, matrix: Union[np.ndarray, sparse.csr_matrix]) -> None:
        """
        Factor the dense matrix using LU decomposition.
        """
        if isinstance(matrix, sparse.csr_matrix):
            matrix = matrix.toarray()
        try:
            self.lu_factors, self.pivots = lu_factor(matrix)
        except np.linalg.LinAlgError as e:
            raise SolverError(f"Dense LU factorization failed: {e}")

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """
        Solve using the factored matrix.
        """
        if self.lu_factors is None or self.pivots is None:
            raise SolverError("Matrix must be factored before solving")

        try:
            return lu_solve((self.lu_factors, self.pivots), rhs)
        except np.linalg.LinAlgError as e:
            raise SolverError(f"Dense LU solve failed: {e}")


class SparseLUSolver(LUSolver):
    """
    Sparse LU solver using scipy.sparse.linalg.splu.
    """

    def __init__(self):
        self._lu_factors = None

    def factor(self, matrix: Union[np.ndarray, sparse.csr_matrix]) -> None:
        """
        Factor the sparse matrix using sparse LU decomposition.
        """
        if not isinstance(matrix, sparse.csr_matrix):
            raise TypeError("SparseLUSolver requires a sparse matrix")
        try:
            self._lu_factors = splu(matrix.tocsc())
        except Exception as e:
            raise SolverError(f"Sparse LU factorization failed: {e}")

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """
        Solve using the factored matrix.
        """
        if self._lu_factors is None:
            raise SolverError("Matrix must be factored before solving")

        try:
            return self._lu_factors.solve(rhs)
        except Exception as e:
            raise SolverError(f"Sparse LU solve failed: {e}")


class NonlinearSystem(Protocol):
    """
    Protocol for nonlinear systems that can be solved with Newton's method.
    """

    def dimension(self) -> int:
        # Return the dimension of the system.
        ...

    def residual(self, x: np.ndarray) -> np.ndarray:
        # Compute the residual f(x) for given x
        ...

    def jacobian_sparse(self, x: np.ndarray) -> sparse.csr_matrix:
        # Compute sparse Jacobian matrix at x.
        ...

    def jacobian_dense(self, x: np.ndarray) -> np.ndarray:
        # Compute dense Jacobian matrix at x.
        ...


class NewtonSolver:
    """
    Newton-Raphson solver for nonlinear systems using explicit LU decomposition.
    """

    def __init__(self, config: Optional[NewtonConfig] = None):
        self.config = config or NewtonConfig()

    def solve(
        self,
        system: NonlinearSystem,
        x0: np.ndarray,
        callback: Optional[Callable[[IterationStats], Control]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Solve the nonlinear system starting from x0.

        Args:
            system: The nonlinear system to solve.
            x0: Initial guess.
            callback: Optional callback function called each iteration.

        Returns:
            Tuple of (solution, iterations).

        Raises:
            SolverError: If solver fails to converge or encounters an error.
        """
        # No callback handling.
        if callback is None:

            def _callback(stats):
                return Control.CONTINUE

            callback = _callback

        x = x0.copy()
        n = system.dimension()

        if len(x) != n:
            raise ValueError(
                f"Initial guess dimension {len(x)} doesn't match system dimension {n}."
            )

        # Determine matrix format and create appropriate solver.
        use_dense = self.should_use_dense(n)

        if use_dense:
            lu_solver = DenseLUSolver()
            return self.solve_with_lu_solver(
                system, x, callback, lu_solver, use_dense=True
            )
        else:
            lu_solver = SparseLUSolver()
            return self.solve_with_lu_solver(
                system, x, callback, lu_solver, use_dense=False
            )

    def should_use_dense(self, n: int) -> bool:
        # Determine whether to use dense or sparse matrices.
        if self.config.format == MatrixFormat.DENSE:
            return True
        elif self.config.format == MatrixFormat.SPARSE:
            return False
        else:  # AUTO
            return n < self.config.format_threshold

    def solve_with_lu_solver(
        self,
        system: NonlinearSystem,
        x: np.ndarray,
        callback: Callable[[IterationStats], Control],
        lu_solver: LUSolver,
        use_dense: bool,
    ) -> Tuple[np.ndarray, int]:
        damping = self.config.damping
        last_res = np.inf
        n = system.dimension()

        # Buffers for line search.
        x_trial = np.zeros_like(x)
        rhs = np.zeros(n)

        for iter in range(self.config.max_iter):
            # Compute residual.
            f = system.residual(x)
            res = np.max(np.abs(f))

            # Call callback.
            stats = IterationStats(iter=iter, residual=res, damping=damping)
            if callback(stats) == Control.CANCEL:
                raise SolverError("Solve cancelled by callback")

            # Check convergence.
            if res < self.config.tol:
                return x, iter + 1

            # Compute Jacobian and factor it.
            if use_dense:
                jac = system.jacobian_dense(x)
            else:
                jac = system.jacobian_sparse(x)

            lu_solver.factor(jac)

            # Prepare RHS and solve for Newton step.
            rhs[:] = -f
            dx = lu_solver.solve(rhs)

            # Apply step with adaptive damping and line search.
            x, damping, step_applied = self._apply_step(
                system,
                x,
                f,
                dx,
                damping,
                last_res,
                x_trial,
            )

            if not step_applied and self.config.adaptive:
                raise SolverError("Divergence guard: line search failed")

            last_res = res

        raise SolverError(
            f"Newton solver did not converge after {self.config.max_iter} iterations"
        )

    def _apply_step(
        self,
        system: NonlinearSystem,
        x: np.ndarray,
        f: np.ndarray,
        dx: np.ndarray,
        damping: float,
        last_res: float,
        x_trial: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool]:
        """
        Apply Newton step with adaptive damping and line search.
        """
        res = np.max(np.abs(f))
        step_applied = False

        if self.config.adaptive:
            # Adjust damping based on progress.
            if res < last_res:
                new_damping = damping * self.config.grow
                damping = min(new_damping, self.config.max_damping)
            else:
                new_damping = damping * self.config.shrink
                damping = max(new_damping, self.config.min_damping)

            # Check for divergence and perform line search if needed.
            if np.isfinite(last_res) and res > last_res * self.config.divergence_ratio:
                alpha = max(damping * self.config.shrink, self.config.min_damping)

                for _ in range(self.config.ls_max_steps):
                    x_trial[:] = x + alpha * dx
                    f_trial = system.residual(x_trial)
                    res_trial = np.max(np.abs(f_trial))

                    if res_trial < res:
                        x[:] = x_trial
                        damping = alpha
                        step_applied = True
                        break

                    alpha *= self.config.ls_backtrack
                    if alpha < self.config.min_damping:
                        break

        # Apply regular step if line search wasn't used or failed.
        if not step_applied:
            x += damping * dx
            step_applied = True

        return x, damping, step_applied


class SimpleNonlinearSystem:
    def dimension(self) -> int:
        return 2

    def residual(self, x: np.ndarray) -> np.ndarray:
        return np.array(
            [
                x[0] ** 2 + x[1] ** 2 - 1.0,  # circle equation
                x[0] - x[1],  # line equation
            ]
        )

    def jacobian_dense(self, x: np.ndarray) -> np.ndarray:
        return np.array([[2 * x[0], 2 * x[1]], [1.0, -1.0]])

    def jacobian_sparse(self, x: np.ndarray) -> sparse.csr_matrix:
        return sparse.csr_matrix(self.jacobian_dense(x))


if __name__ == "__main__":
    # Example usage.
    system = SimpleNonlinearSystem()
    solver = NewtonSolver(NewtonConfig(tol=1e-10, adaptive=True))

    # Initial guess.
    x0 = np.array([0.5, 0.5])

    # Solve with callback.
    def progress_callback(stats: IterationStats) -> Control:
        print(
            f"Iteration {stats.iter}: residual = {stats.residual:.2e}, damping = {stats.damping:.3f}"
        )
        return Control.CONTINUE

    try:
        solution, iterations = solver.solve(system, x0, progress_callback)
        print(f"\nConverged in {iterations} iterations")
        print(f"Solution: x = {solution}")
        print(f"Residual: {system.residual(solution)}")
    except SolverError as e:
        print(f"Solver failed: {e}")
