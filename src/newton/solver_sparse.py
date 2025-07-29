from typing import Any, Dict, List

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csc_matrix, lil_matrix

import newton.backend as nb
from newton.constraints import BaseConstraint, Constraint
from newton.primitives import Point
from newton.solver_base import DEBUG_LOG, SOLVER_CONVERGENCE_TOLERANCE, Solver2D


class Solver2DSparse(Solver2D):
    def __init__(self, points: List[Point], constraints: List[Constraint]):
        super().__init__(points, constraints)

        # Handle backend setup.
        nb.set_backend(nb.Backend.NUMPY)
        self.module = np

    def build_sparse_jacobian(
        self,
        subproblem: Dict[str, Any],
        var_map: Dict[str, int],
        positions: Dict[str, np.ndarray],
    ) -> csc_matrix:
        constraints: List[BaseConstraint] = subproblem["constraints"]
        n_residuals = sum(c.get_residual_dim() for c in constraints)
        n_vars = len(var_map)

        jacobian = lil_matrix((n_residuals, n_vars))
        current_row = 0

        for constraint in constraints:
            try:
                entries = constraint.get_jacobian_section(positions)
                for point_id, coord, value, residual_idx in entries:
                    var_name = f"{point_id}_{coord}"
                    if var_name in var_map:
                        col_idx = var_map[var_name]
                        jacobian[current_row + residual_idx, col_idx] += value
            except NotImplementedError as e:
                print(f"Warning: Skipping constraint {type(constraint).__name__}: {e}")
                # Skip this constraint's jacobian entries
            current_row += constraint.get_residual_dim()

        if DEBUG_LOG:
            print("Jacobian:\n", jacobian.toarray())

        return jacobian.tocsc()

    def solve_constraint_system(self, system: Dict[str, Any]):
        free_points: List[Point] = system["free_points"]
        constraints: List[BaseConstraint] = system["constraints"]

        if not free_points or not constraints:
            if DEBUG_LOG:
                print("Skipping block: No free points or no constraints.")
            return

        if DEBUG_LOG:
            print(
                f"Solving independently soluble system: {[p.id for p in free_points]}"
            )

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

        def jacobian_wrapper(free_vars: np.ndarray) -> csc_matrix:
            positions = get_all_positions(free_vars)
            return self.build_sparse_jacobian(system, var_map, positions)

        # Do rank based system state check.
        jacobian_init = jacobian_wrapper(initial_guess)

        if jacobian_init.shape is None:
            raise ValueError("Jacobian is empty or not properly initialized.")

        n_equations = jacobian_init.shape[0]
        n_variables = jacobian_init.shape[1]

        self.check_system_state(jacobian_init.todense(), n_variables, n_equations)

        # Actual solve magic using the least_squares method.
        # Not sure which is most appropriate here... CC Dave Reeves: current thinking
        # Levenberg-Marquardt (lm) is good because least squares and Newton's method.
        # TRF is on the only supported method for sparse Jacobians.
        result = least_squares(
            fun=residuals_vector,
            x0=initial_guess,
            jac=jacobian_wrapper,  # type: ignore
            method="trf",
            xtol=SOLVER_CONVERGENCE_TOLERANCE,
            ftol=SOLVER_CONVERGENCE_TOLERANCE,
            gtol=SOLVER_CONVERGENCE_TOLERANCE,
            verbose=2 if DEBUG_LOG else 0,
        )
        self.update_points_from_result(result, free_points)

        return
