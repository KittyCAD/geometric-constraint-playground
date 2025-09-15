import logging
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import least_squares
from scipy.sparse import csc_matrix, csr_matrix, diags, lil_matrix, vstack

import newton.backend as nb
from newton.constants import (
    CONFIG_USE_NEWTON_FAER,
    CONFIG_USE_REGULARIZATION,
    REGULARIZATION_LAMBDA,
)
from newton.constraints import Constraint
from newton.logging_config import logger
from newton.ports import newton_faer
from newton.ports.newton_faer import (
    NewtonConfig,
    NewtonSolver,
    NonlinearSystem,
)
from newton.primitives import Primitive
from newton.solver_base import SOLVER_CONVERGENCE_TOLERANCE, Solver2D
from newton.symbolic_substitution import find


class ConstraintSystemAdapter(NonlinearSystem):
    """
    Adapter to make the constraint system compatible with NewtonSolver.
    """

    def __init__(
        self,
        constraints: List[Constraint],
        independent_vars: List[str],
        var_map: Dict[str, int],
        initial_values: Dict[str, float],
        substitution_map: Dict[str, str],
        solver_instance: "Solver2DSparse",
    ):
        self.constraints = constraints
        self.independent_vars = independent_vars
        self.var_map = var_map
        self.initial_values = initial_values
        self.substitution_map = substitution_map
        self.solver_instance = solver_instance

    @property
    def n_variables(self) -> int:
        return len(self.independent_vars)

    @property
    def n_residuals(self) -> int:
        # Count constraint residuals.
        n_residuals = sum(c.n_residual_rows for c in self.constraints)

        if CONFIG_USE_REGULARIZATION:
            # Add regularization terms (one per variable).
            n_regularization_residuals = len(self.independent_vars)
            n_residuals += n_regularization_residuals

        return n_residuals

    def residual(self, x: np.ndarray) -> np.ndarray:
        """
        Compute residual vector for the constraint system with Tikhonov regularization.
        """
        variable_values = self.solver_instance.build_variable_values_map(
            x,
            self.independent_vars,
            self.initial_values,
            self.substitution_map,
        )

        # Get constraint residuals
        constraint_residuals = np.concatenate(
            [c.get_residual(variable_values) for c in self.constraints]
        )

        if CONFIG_USE_REGULARIZATION:
            # Add Tikhonov regularization term: lambda * (x - x0).
            initial_guess = np.array(
                [self.initial_values[var_id] for var_id in self.independent_vars]
            )
            reg_residuals = REGULARIZATION_LAMBDA * (x - initial_guess)

            return np.concatenate([constraint_residuals, reg_residuals])

        else:
            return constraint_residuals

    def jacobian_sparse(self, x: np.ndarray) -> sparse.csr_matrix:
        variable_values = self.solver_instance.build_variable_values_map(
            x,
            self.independent_vars,
            self.initial_values,
            self.substitution_map,
        )

        # Build the sparse jacobian using existing method.
        system_dict = {
            "constraints": self.constraints,
            "substitution_map": self.substitution_map,
        }

        jacobian = self.solver_instance.build_sparse_jacobian(
            system_dict, self.var_map, variable_values
        )

        # Combine them vertically into the augmented Jacobian
        if CONFIG_USE_REGULARIZATION:
            # Add regularization Jacobian (lambda * I) to match scipy behavior.
            n_vars = len(self.independent_vars)
            reg_jacobian = diags([REGULARIZATION_LAMBDA] * n_vars, format="csc")
            augmented_jacobian = vstack([jacobian, reg_jacobian], format="csc")
            return csr_matrix(augmented_jacobian)
        else:
            return csr_matrix(jacobian)

    def jacobian_dense(self, x: np.ndarray) -> np.ndarray:
        return self.jacobian_sparse(x).toarray()


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

        return jacobian.tocsc()

    def solve_constraint_system(self, system: Dict[str, Any]):
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

        # Use the consolidated method from base class.
        initial_values = self.get_initial_values_for_all_variables()
        initial_guess = np.array(
            [initial_values[var_id] for var_id in independent_vars]
        )

        if CONFIG_USE_NEWTON_FAER:
            result = self.solve_with_newton_faer(
                constraints,
                independent_vars,
                var_map,
                initial_values,
                substitution_map,
                initial_guess,
            )
        else:
            # Use the original scipy least_squares method.
            result = self.solve_with_scipy(
                constraints,
                independent_vars,
                initial_guess,
                initial_values,
                substitution_map,
            )

        # TODO: There is some cursed LLM slop repetition of other code in here, but it works.
        # ------------------------------------------------------------------------------
        # Get the final Jacobian at the accepted solution.
        final_jacobian_sparse = None
        if CONFIG_USE_NEWTON_FAER:
            # Re-create the adapter to get the final Jacobian
            constraint_system_for_jacobian = ConstraintSystemAdapter(
                constraints=constraints,
                independent_vars=independent_vars,
                var_map=var_map,
                initial_values=initial_values,
                substitution_map=substitution_map,
                solver_instance=self,
            )
            final_jacobian_sparse = constraint_system_for_jacobian.jacobian_sparse(
                result.x
            )
        else:
            # Re-create the scipy wrapper to get the final Jacobian
            var_map_scipy = {var_id: i for i, var_id in enumerate(independent_vars)}

            def jacobian_wrapper(independent_vars_values: np.ndarray) -> csc_matrix:
                # TODO: We shouldn't be rebuilding this; just surface it from
                # where it's created for real.
                variable_values = self.build_variable_values_map(
                    independent_vars_values,
                    independent_vars,
                    initial_values,
                    substitution_map,
                )
                system_dict = {
                    "constraints": constraints,
                    "substitution_map": substitution_map,
                }
                jacobian = self.build_sparse_jacobian(
                    system_dict, var_map_scipy, variable_values
                )
                n_vars = len(independent_vars)
                reg_jacobian = diags([REGULARIZATION_LAMBDA] * n_vars, format="csc")
                return csc_matrix(vstack([jacobian, reg_jacobian], format="csc"))

            final_jacobian_sparse = jacobian_wrapper(result.x)

        final_jacobian_dense = final_jacobian_sparse.toarray()

        # Analyse our degrees of freedom based on the final Jacobian.
        constraint_status = self.analyze_degrees_of_freedom(
            final_jacobian_dense, free_primitives, var_map
        )

        # Report status.
        logger.info("Constraint status for subsystem elements:")
        for prim_id, is_constrained in sorted(constraint_status.items()):
            status_str = "Fully Constrained" if is_constrained else "Under-constrained"
            logger.info(f"  - {prim_id}: {status_str}")
        # ------------------------------------------------------------------------------

        # Run checks and update using the consolidated method from the base class.
        # This effectively fans out the final variable values through the substitution map.
        final_variable_values = self.fan_out_solved_variable_values(
            result, independent_vars, substitution_map
        )
        self.update_primitives_from_map(final_variable_values)
        self.assess_solver_result(final_variable_values, constraints)
        return

    def solve_with_newton_faer(
        self,
        constraints: List[Constraint],
        independent_vars: List[str],
        var_map: Dict[str, int],
        initial_values: Dict[str, float],
        substitution_map: Dict[str, str],
        initial_guess: np.ndarray,
    ):
        # Create the system adapter.
        constraint_system = ConstraintSystemAdapter(
            constraints=constraints,
            independent_vars=independent_vars,
            var_map=var_map,
            initial_values=initial_values,
            substitution_map=substitution_map,
            solver_instance=self,
        )

        # Configure the Newton solver and sovle.
        newton_config = NewtonConfig(
            tol=SOLVER_CONVERGENCE_TOLERANCE,
            max_iter=50,
            format=newton_faer.MatrixFormat.SPARSE,
            adaptive=True,
            damping=1.0,
        )

        newton_solver = NewtonSolver(newton_config)
        solution, iterations = newton_solver.solve(
            constraint_system,
            initial_guess,
        )

        logger.debug(f"Newton-Faer converged in {iterations} iterations")

        # Create a result-like object for compatibility with existing code.
        from scipy.optimize import OptimizeResult

        result = OptimizeResult(
            x=solution, success=True, fun=constraint_system.residual(solution)
        )
        return result

    def solve_with_scipy(
        self,
        constraints: List[Constraint],
        independent_vars: List[str],
        initial_guess: np.ndarray,
        initial_values: Dict[str, float],
        substitution_map: Dict[str, str],
    ):
        var_map = {var_id: i for i, var_id in enumerate(independent_vars)}

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
            system_dict = {
                "constraints": constraints,
                "substitution_map": substitution_map,
            }
            jacobian = self.build_sparse_jacobian(system_dict, var_map, variable_values)

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

        return result
