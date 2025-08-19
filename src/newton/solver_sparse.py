import logging
from typing import Any, Dict, List, Mapping

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csc_matrix, diags, lil_matrix, vstack

import newton.backend as nb
from newton.constants import REGULARIZATION_LAMBDA
from newton.constraints import Constraint
from newton.logging_config import logger
from newton.primitives import Primitive
from newton.solver_base import SOLVER_CONVERGENCE_TOLERANCE, Solver2D


class Solver2DSparse(Solver2D):
    def __init__(self, primitives: List[Primitive], constraints: List[Constraint]):
        super().__init__(primitives, constraints)

        # Handle backend setup.
        nb.set_backend(nb.Backend.NUMPY)
        self.module = np

    def build_sparse_jacobian(
        self,
        subproblem: Dict[str, Any],
        var_map: Dict[str, int],
        variable_values: Mapping[str, float],
    ) -> csc_matrix:
        constraints: List[Constraint] = subproblem["constraints"]

        n_residuals = sum(c.n_residual_rows for c in constraints)
        n_variables = len(var_map)

        jacobian = lil_matrix((n_residuals, n_variables))

        i_row = 0

        for constraint in constraints:
            try:
                # The constraint should now have full variable IDs.
                entries = constraint.get_jacobian_row_values(variable_values)

                for var_id, value, i_row_local in entries:
                    if var_id in var_map:
                        i_col = var_map[var_id]
                        jacobian[i_row + i_row_local, i_col] += value

            except NotImplementedError as e:
                logger.warning(f"Skipping constraint {type(constraint).__name__}: {e}")

            i_row += constraint.n_residual_rows

        logger.debug("Jacobian:\n%s", jacobian.toarray())

        return jacobian.tocsc()

    def solve_constraint_system(self, system: Dict[str, Any]):
        free_primitives: List[Primitive] = system["free_primitives"]
        constraints: List[Constraint] = system["constraints"]
        substitution_map: Dict[str, str] = system["substitution_map"]

        if not free_primitives or not constraints:
            logger.debug("Skipping block: No free points or no constraints.")
            return

        logger.debug(
            f"Solving independently soluble system: {[p.id for p in free_primitives]}"
        )

        # Create an ordered list of all variables we can solve for.
        free_var_ids = [
            var_id
            for p in free_primitives
            for var_id in p.get_variable_ids()
            if var_id not in substitution_map
        ]

        # Generic variable map for the free variables.
        var_map = {var_id: i for i, var_id in enumerate(free_var_ids)}

        # Build the initial guess array based on the ordered variable list.
        initial_values: Dict[str, float] = {}
        for p in self.primitives:
            initial_values.update(p.get_initial_variable_values())

        initial_guess = np.array([initial_values[var_id] for var_id in free_var_ids])

        def build_variable_values_map(
            free_vars: np.ndarray,
            initial_values: Mapping[str, float],
        ) -> Mapping[str, float]:
            # Builds the flat map of all variable IDs to their current values.
            # This is the single source of truth passed to the constraints.

            # Start with the initial state of all variables in the system.
            variable_values = dict(initial_values)

            # Create a dictionary of the variables the solver is currently optimizing.
            solved_vars = {
                var_id: free_vars[i] for i, var_id in enumerate(free_var_ids)
            }

            # Overwrite the initial values with the current solved values.
            variable_values.update(solved_vars)

            return variable_values

        def residuals_vector(free_vars: np.ndarray) -> np.ndarray:
            variable_values = build_variable_values_map(free_vars, initial_values)
            constraint_residuals = np.concatenate(
                [c.get_residual(variable_values) for c in constraints]
            )

            # Regularization residuals (lambda * (x - x_initial)).
            reg_residuals = REGULARIZATION_LAMBDA * (free_vars - initial_guess)

            # Combine them into the new augmented residual vector.
            return np.concatenate([constraint_residuals, reg_residuals])

        def jacobian_wrapper(free_vars: np.ndarray) -> csc_matrix:
            # Use the new generic helper function.
            variable_values = build_variable_values_map(free_vars, initial_values)

            # Original constraint Jacobian.
            jacobian = self.build_sparse_jacobian(system, var_map, variable_values)

            # Regularization Jacobian (lambda * I).
            n_vars = len(free_vars)
            reg_jacobian = diags([REGULARIZATION_LAMBDA] * n_vars, format="csc")

            # Combine them vertically into the new augmented Jacobian.
            result = vstack([jacobian, reg_jacobian], format="csc")

            return csc_matrix(result)

        # Do rank based system state check.
        jacobian_init = jacobian_wrapper(initial_guess)

        if jacobian_init.shape is None:
            raise ValueError("Jacobian is empty or not properly initialized.")

        # Because of our Tikhonov regularization, we need to adjust the number of equations.
        # We effectively vertically concatenate the actual Jacobian with another matrix of the
        # size (REG_LAMBDA * I), which adds n_variables more rows.
        n_variables = jacobian_init.shape[1]
        # n_equations = jacobian_init.shape[0]
        n_geom_equations = sum(c.n_residual_rows for c in constraints)

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

        # Run checks and update.
        final_variable_values = self.compute_final_variable_values(
            result, free_primitives, substitution_map
        )
        self.update_primitives_from_map(final_variable_values)
        self.assess_solver_result(final_variable_values, constraints)
        return
