from typing import Any, Dict, List

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

import newton.backend as nb
from newton.constraints import BaseConstraint
from newton.primitives import Point
from newton.solver_base import DEBUG_LOG, SOLVE_TOLERANCE, Solver2D


class Solver2DSparse(Solver2D):
    def build_sparse_jacobian(
        self,
        subproblem: Dict[str, Any],
        var_map: Dict[str, int],
        positions: Dict[str, np.ndarray],
    ) -> lil_matrix:
        constraints: List[BaseConstraint] = subproblem["constraints"]
        n_residuals = sum(c.get_residual_dim() for c in constraints)
        n_vars = len(var_map)

        jacobian = lil_matrix((n_residuals, n_vars))
        current_row = 0

        for constraint in constraints:
            entries = constraint.get_jacobian_section(positions)
            for point_id, coord, value, residual_idx in entries:
                var_name = f"{point_id}_{coord}"
                if var_name in var_map:
                    col_idx = var_map[var_name]
                    jacobian[current_row + residual_idx, col_idx] += value
            current_row += constraint.get_residual_dim()

        if DEBUG_LOG:
            print("Jacobian:\n", jacobian.toarray())

        return jacobian.tocsc()

    def solve_subproblem(self, subproblem: Dict[str, Any]):
        # Set the backend to numpy for this solve.
        nb.set_backend(nb.Backend.NUMPY)

        free_points: List[Point] = subproblem["free_points"]
        constraints: List[BaseConstraint] = subproblem["constraints"]

        if DEBUG_LOG:
            print(f"Solving Subproblem: {[p.id for p in free_points]}")

        initial_guess = np.array([[p.x, p.y] for p in free_points]).flatten()
        var_map = {
            f"{p.id}_{c}": i * 2 + j
            for i, p in enumerate(free_points)
            for j, c in enumerate("xy")
        }

        def get_all_positions(free_vars: np.ndarray) -> Dict[str, np.ndarray]:
            positions = {p.id: np.array([p.x, p.y]) for p in self.points}
            for i, p in enumerate(free_points):
                positions[p.id] = free_vars[i * 2 : i * 2 + 2]
            return positions

        def residuals_vector(free_vars: np.ndarray) -> np.ndarray:
            positions = get_all_positions(free_vars)
            return np.concatenate([c.get_residual(positions) for c in constraints])

        def jacobian_wrapper(free_vars: np.ndarray) -> lil_matrix:
            positions = get_all_positions(free_vars)
            return self.build_sparse_jacobian(subproblem, var_map, positions)

        # Actual solve magic using the least_squares method.
        # Not sure which is most appropriate here... CC Dave Reeves: current thinking
        # Levenberg-Marquardt (lm) is good because least squares and Newton's method.
        # TRF is on the only supported method for sparse Jacobians.
        result = least_squares(
            fun=residuals_vector,
            x0=initial_guess,
            jac=jacobian_wrapper,  # type: ignore
            method="trf",
            xtol=SOLVE_TOLERANCE,
            ftol=SOLVE_TOLERANCE,
            gtol=SOLVE_TOLERANCE,
            verbose=2 if DEBUG_LOG else 0,
        )
        self.update_points_from_result(result, free_points)
