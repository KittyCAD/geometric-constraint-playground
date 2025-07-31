import logging
from typing import Any, Dict, List

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csc_matrix, diags, lil_matrix, vstack

import newton.backend as nb
from newton.constraints import BaseConstraint, Constraint
from newton.logging_config import logger
from newton.primitives import Point
from newton.solver_base import SOLVER_CONVERGENCE_TOLERANCE, Solver2D

# For Tikhonov regularization
# TODO: Explore reasonable values for this.
# TODO: We should do this for dense solve too, if we go down that route.
REG_LAMBDA = 1e-9


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
                logger.warning(f"Skipping constraint {type(constraint).__name__}: {e}")
                # Skip this constraint's jacobian entries
            current_row += constraint.get_residual_dim()

        logger.debug("Jacobian:\n%s", jacobian.toarray())

        return jacobian.tocsc()

    def solve_constraint_system(self, system: Dict[str, Any]):
        free_points: List[Point] = system["free_points"]
        constraints: List[BaseConstraint] = system["constraints"]

        if not free_points or not constraints:
            logger.debug("Skipping block: No free points or no constraints.")
            return

        logger.debug(
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

            # Original constraint residuals.
            constraint_residuals = np.concatenate(
                [c.get_residual(positions) for c in constraints]
            )

            # Regularization residuals (lambda * x).
            reg_residuals = REG_LAMBDA * (free_vars - initial_guess)

            # Combine them into the new augmented residual vector.
            return np.concatenate([constraint_residuals, reg_residuals])

        def jacobian_wrapper(free_vars: np.ndarray) -> csc_matrix:
            positions = get_all_positions(free_vars)

            # Original constraint Jacobian.
            jacobian = self.build_sparse_jacobian(system, var_map, positions)

            # Regularization Jacobian (lambda * I).
            n_vars = len(free_vars)
            reg_jacobian = diags([REG_LAMBDA] * n_vars, format="csc")

            # Combine them vertically into the new augmented Jacobian.
            return vstack([jacobian, reg_jacobian], format="csc")

        # Do rank based system state check.
        jacobian_init = jacobian_wrapper(initial_guess)

        if jacobian_init.shape is None:
            raise ValueError("Jacobian is empty or not properly initialized.")

        # Because of our Tikhonov regularization, we need to adjust the number of equations.
        # We effectively vertically concatenate the actual Jacobian with another matrix of the
        # size (REG_LAMBDA * I), which adds n_variables more rows.
        n_variables = jacobian_init.shape[1]
        # n_equations = jacobian_init.shape[0]
        n_geom_equations = sum(c.get_residual_dim() for c in constraints)

        self.check_system_state(jacobian_init.todense(), n_variables, n_geom_equations)

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
            verbose=2 if logger.isEnabledFor(logging.DEBUG) else 0,
        )
        self.update_points_from_result(result, free_points)

        return
