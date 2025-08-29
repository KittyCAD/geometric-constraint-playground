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
from typing import Callable, Optional, Protocol, Tuple, Union, cast

import numpy as np
from scipy import sparse
from scipy.linalg import lu_factor, lu_solve, qr, solve_triangular
from scipy.sparse.linalg import lsqr, splu

from newton.logging_config import logger


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


class LinearSolver(ABC):
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


class DenseLUSolver(LinearSolver):
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


class SparseLUSolver(LinearSolver):
    """
    Sparse LU solver using scipy.sparse.linalg.splu.
    """

    def __init__(self):
        self.lu_factors = None

    def factor(self, matrix: Union[np.ndarray, sparse.csr_matrix]) -> None:
        """
        Factor the sparse matrix using sparse LU decomposition.
        """
        if not isinstance(matrix, sparse.csr_matrix):
            raise TypeError("SparseLUSolver requires a sparse matrix")
        try:
            self.lu_factors = splu(matrix.tocsc())
        except Exception as e:
            raise SolverError(f"Sparse LU factorization failed: {e}")

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """
        Solve using the factored matrix.
        """
        if self.lu_factors is None:
            raise SolverError("Matrix must be factored before solving")

        try:
            return self.lu_factors.solve(rhs)
        except Exception as e:
            raise SolverError(f"Sparse LU solve failed: {e}")


class QRSolver(LinearSolver):
    """
    Dense QR solver using explicit scipy.linalg.qr.
    """

    def __init__(self):
        self.q: Optional[np.ndarray] = None
        self.r: Optional[np.ndarray] = None
        self.p: Optional[np.ndarray] = None
        self.matrix_original: Optional[np.ndarray] = None

    def factor(self, matrix):
        """
        Factor the matrix using QR decomposition.
        """
        if isinstance(matrix, sparse.csr_matrix):
            matrix = matrix.toarray()
        try:
            result = qr(matrix, mode="economic", pivoting=True)
            if len(result) == 3:
                # Cast to the specific return type we expect.
                q, r, p = cast(Tuple[np.ndarray, np.ndarray, np.ndarray], result)
                self.q = q
                self.r = r
                self.p = p
                self.matrix_original = matrix

                # Build stats.
                diag = np.diag(r)  # Diagonal of R.
                diag_abs = np.abs(diag)  # Absolute values of diagonal.
                s_max = diag_abs.max() if diag_abs.size else 0.0
                s_min = diag_abs.min() if diag_abs.size else 0.0
                condition_number = (s_max / s_min) if (s_min > 0) else np.inf
                rank = np.linalg.matrix_rank(r)

                # Check reconstruction error: A[:,p] = QR
                matrix_permutation = matrix[:, p]
                recon_err = np.linalg.norm(matrix_permutation - np.matmul(q, r)) / (
                    np.linalg.norm(matrix_permutation) + 1e-16
                )

                logger.debug(f"System shape: {matrix.shape}")
                logger.debug(f"Rank estimate (np.linalg.matrix_rank(R)): {rank}")
                logger.debug(f"Frobenius norm, Q: {np.linalg.norm(q)}")
                logger.debug(f"Frobenius norm, R: {np.linalg.norm(r)}")
                logger.debug(f"Reconstruction error: {recon_err}")
                logger.debug(f"Condition number: {condition_number}")
            else:
                raise SolverError("QR factorization did not return expected results")
        except Exception as e:
            raise SolverError(f"Dense QR factorization failed: {e}")

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        if self.q is None or self.r is None or self.p is None:
            raise SolverError("Matrix must be factored before solving")
        try:
            # Solve Ax = rhs. Our caller does rhs = -f, so no sign fuss required.
            qt_rhs = self.q.transpose().dot(rhs)

            # Solve the upper triangular system R @ y = Q.T @ rhs.
            # The result, y, is the solution vector in its permuted order.
            y = solve_triangular(self.r, qt_rhs)

            # Apply the inverse permutation to get the final solution x.
            # This reorders the solution vector 'y' using the
            # permutation array 'p' such that the i-th element of y is placed
            # at the p[i]-th position in x.
            x = np.zeros_like(y)
            x[self.p] = y

            return x
        except Exception as e:
            raise SolverError(f"Dense QR solve failed: {e}")


class SparseLSQRSolver(LinearSolver):
    """
    Sparse QR solver using scipy.sparse.linalg.lsqr.
    """

    def __init__(self):
        self.matrix = None

    def factor(self, matrix: Union[np.ndarray, sparse.csr_matrix]) -> None:
        """
        Store the sparse matrix for later solving.
        """
        if not isinstance(matrix, sparse.csr_matrix):
            raise TypeError("SparseQRSolver requires a sparse matrix")
        self.matrix = matrix

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """
        Solve the least squares problem using the stored matrix.
        """
        if self.matrix is None:
            raise SolverError("Matrix must be set before solving")

        try:
            result = lsqr(self.matrix, rhs)
            return result[0]
        except Exception as e:
            raise SolverError(f"Sparse QR solve failed: {e}")


class NonlinearSystem(Protocol):
    """
    Protocol for nonlinear systems that can be solved with Newton's method.
    """

    @property
    def n_variables(self) -> int:
        # Return the dimension of the system.
        ...

    @property
    def n_residuals(self) -> int:
        # Return the dimension of the residual (number of equations).
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
    Newton-Raphson solver for nonlinear systems with QR support for overdetermined systems.
    """

    USE_QR = True

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
        n = system.n_variables  # Number of variables
        m = system.n_residuals  # Number of equations

        if len(x) != n:
            raise ValueError(
                f"Initial guess dimension {len(x)} doesn't match number of variables {n}."
            )

        # Determine matrix format and create appropriate solver.
        use_dense = self.should_use_dense(max(m, n))

        if use_dense:
            if self.USE_QR:
                solver = QRSolver()
            else:
                if m != n:
                    raise SolverError("Non-square system requires QR solver")
                solver = DenseLUSolver()

            return self.solve_iterative(system, x, callback, solver, use_dense=True)
        else:
            if self.USE_QR:
                solver = QRSolver()
            else:
                if m != n:
                    raise SolverError("Non-square system requires QR solver")
                solver = SparseLUSolver()

            return self.solve_iterative(system, x, callback, solver, use_dense=False)

    def should_use_dense(self, size: int) -> bool:
        # Determine whether to use dense or sparse matrices.
        if self.config.format == MatrixFormat.DENSE:
            return True
        elif self.config.format == MatrixFormat.SPARSE:
            return False
        else:  # AUTO
            return size < self.config.format_threshold

    def solve_iterative(
        self,
        system: NonlinearSystem,
        x: np.ndarray,
        callback: Callable[[IterationStats], Control],
        solver: LinearSolver,
        use_dense: bool,
    ) -> Tuple[np.ndarray, int]:
        damping = self.config.damping
        last_res = np.inf
        # n = system.n_variables  # Number of variables
        m = system.n_residuals  # Number of equations

        # Buffers for line search
        x_trial = np.zeros_like(x)  # Shape: (n,)
        rhs = np.zeros(m)  # Shape: (m,) - number of equations

        for iter in range(self.config.max_iter):
            # Compute residual.
            f = system.residual(x)  # Shape: (m,)

            # TODO: Switch between these between square and overdetermined systems?
            # res = np.max(np.abs(f))
            res = float(np.linalg.norm(f, ord=2))  # Use L2 norm for convergence check.

            logger.debug(f"Iter {iter}: Residual = {res:.3e}")

            # Call callback.
            stats = IterationStats(iter=iter, residual=res, damping=damping)
            if callback(stats) == Control.CANCEL:
                raise SolverError("Solve cancelled by callback")

            # Check convergence.
            if res < self.config.tol:
                return x, iter + 1

            # Compute Jacobian and factor it.
            if use_dense:
                jac = system.jacobian_dense(x)  # Shape: (m, n)
            else:
                jac = system.jacobian_sparse(x)  # Shape: (m, n)

            solver.factor(jac)

            # Prepare RHS and solve for Newton step.
            rhs[:] = -f  # Shape: (m,) = (m,)
            dx = solver.solve(rhs)  # Shape: (n,) from solving (m, n) @ (n,) = (m,)

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
    @property
    def n_variables(self) -> int:
        return 2

    @property
    def n_residuals(self) -> int:
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
