import logging
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csc_matrix, diags, lil_matrix, vstack

import newton.backend as nb
from newton.constants import REGULARIZATION_LAMBDA
from newton.constraints import Constraint
from newton.logging_config import logger
from newton.primitives import Primitive
from newton.solver_base import SOLVER_CONVERGENCE_TOLERANCE, Solver2D
from newton.symbolic_substitution import find


class Solver2DSparse(Solver2D):
    def __init__(
        self, primitives: Sequence[Primitive], constraints: Sequence[Constraint]
    ):
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
        substitution_map: Dict[str, str] = subproblem["substitution_map"]
        n_residuals = sum(c.n_residual_rows for c in constraints)
        n_variables = len(var_map)

        jacobian = lil_matrix((n_residuals, n_variables))

        i_row = 0

        for constraint in constraints:
            try:
                # The constraint should now have full variable IDs.
                entries = constraint.get_jacobian_row_values(variable_values)

                for var_id, value, i_row_local in entries:
                    root_var = find(var_id, substitution_map)
                    if root_var in var_map:
                        i_col = var_map[root_var]
                        jacobian[i_row + i_row_local, i_col] += value

            except NotImplementedError as e:
                logger.warning(f"Skipping constraint {type(constraint).__name__}: {e}")

            i_row += constraint.n_residual_rows

        # logger.debug("Jacobian:\n%s", jacobian.toarray())

        return jacobian.tocsc()

    def solve_constraint_system(self, system: Dict[str, Any]):
        """Solve using sparse matrix methods."""
        free_primitives: List[Primitive] = system["free_primitives"]
        constraints: List[Constraint] = system["constraints"]
        substitution_map: Dict[str, str] = system["substitution_map"]

        if not free_primitives or not constraints:
            logger.debug("Skipping block: No free primitives or no constraints.")
            return

        logger.debug(
            f"Solving independently soluble system: {[p.id for p in free_primitives]}"
        )

        # Determine the true independent variables for the solver.
        # These are the variables from free primitives that are _not_ substituted by another variable.
        independent_vars = self.get_independent_variables(free_primitives)

        if not independent_vars:
            logger.info(
                "System is fully constrained by substitutions, nothing to solve."
            )
            return

        var_map = {var_id: i for i, var_id in enumerate(independent_vars)}

        # Use the consolidated method from base class
        initial_values = self.get_initial_values_for_all_variables()
        initial_guess = np.array(
            [initial_values[var_id] for var_id in independent_vars]
        )

        def residuals_vector(independent_vars_values: np.ndarray) -> np.ndarray:
            variable_values = self.build_variable_values_map(
                independent_vars_values,
                independent_vars,
                initial_values,
                substitution_map,
            )

            constraint_residuals = np.concatenate(
                [c.get_residual(variable_values) for c in constraints]
            )
            reg_residuals = REGULARIZATION_LAMBDA * (
                independent_vars_values - initial_guess
            )
            return np.concatenate([constraint_residuals, reg_residuals])

        def jacobian_wrapper(independent_vars_values: np.ndarray) -> csc_matrix:
            variable_values = self.build_variable_values_map(
                independent_vars_values,
                independent_vars,
                initial_values,
                substitution_map,
            )

            # Original constraint Jacobian.
            jacobian = self.build_sparse_jacobian(system, var_map, variable_values)

            # Regularization Jacobian (lambda * I).
            n_vars = len(independent_vars)
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
        n_variables_independent = len(independent_vars)
        n_geom_equations = sum(c.n_residual_rows for c in constraints)

        self.check_system_state(
            jacobian_init.todense(), n_variables_independent, n_geom_equations
        )

        # Actual solve magic using the least_squares method.
        # Not sure which is most appropriate here... CC Dave Reeves: current thinking
        # Levenberg-Marquardt (lm) is good because least squares and Newton's method.
        # TRF is on the only supported method for sparse Jacobians
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

        # Run checks and update using the consolidated method from the base class.
        # This effectively fanss out the final variable values through the substitution map.
        final_variable_values = self.fan_out_solved_variable_values(
            result, independent_vars, substitution_map
        )
        self.update_primitives_from_map(final_variable_values)
        self.assess_solver_result(final_variable_values, constraints)
        return
