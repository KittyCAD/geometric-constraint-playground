# Notes:
# - This method is currently significantly slower than the sparse version; I don't think
#   this is actually because of dense vs. sparse, but because we're repeatedly
#   recompiling the JAX functions because we're feeding it things with different
#   shapes.

import logging
from typing import Any, Dict, List, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

import newton.backend as nb
from newton.constants import NONZERO_RANK_TOLERANCE, REGULARIZATION_LAMBDA
from newton.constraints import Constraint
from newton.logging_config import logger
from newton.primitives import Primitive
from newton.solver_base import SOLVER_CONVERGENCE_TOLERANCE, Solver2D


class Solver2DDense(Solver2D):
    def __init__(
        self, primitives: Sequence[Primitive], constraints: Sequence[Constraint]
    ):
        super().__init__(primitives, constraints)

        # Handle backend setup.
        # This actually creeps up on our solve tolerance, so we need 64-bit precision.
        jax.config.update("jax_enable_x64", True)
        nb.set_backend(nb.Backend.JAX)
        self.module = jnp

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

        # Use the consolidated method from the base class.
        initial_values = self.get_initial_values_for_all_variables()

        # Build the initial guess array by looking up the free variables.
        initial_guess = np.array(
            [initial_values[var_id] for var_id in independent_vars]
        )
        jnp_initial_guess = jnp.asarray(initial_guess)

        def residuals_vector(independent_vars_values: jnp.ndarray) -> jnp.ndarray:
            variable_values = self.build_variable_values_map(
                independent_vars_values,
                independent_vars,
                initial_values,
                substitution_map,
            )

            constraint_residuals = jnp.concatenate(
                [c.get_residual(variable_values) for c in constraints]
            )

            reg_residuals = REGULARIZATION_LAMBDA * (
                independent_vars_values - jnp_initial_guess
            )
            residuals = jnp.concatenate([constraint_residuals, reg_residuals])
            return residuals

        jit_residuals = jax.jit(residuals_vector)
        jit_jacobian = jax.jit(jax.jacfwd(residuals_vector))

        # Do rank based system state check.
        jacobian_init = jit_jacobian(jnp_initial_guess)

        if jacobian_init.shape is None:
            raise ValueError("Jacobian is empty or not properly initialized.")

        # The number of variables is the number of independent variables we are solving for.
        n_variables_independent = len(independent_vars)
        n_geom_equations = sum(c.n_residual_rows for c in constraints)

        self.check_system_state(
            jacobian_init, n_variables_independent, n_geom_equations
        )

        # The per-iteration rank check is still useful for diagnostics.
        # Define a function that can be used for debugging if needed
        def get_jacobian_with_rank(independent_vars_values: jnp.ndarray) -> jnp.ndarray:
            jacobian = jit_jacobian(independent_vars_values)
            s_values = jnp.linalg.svd(jacobian, compute_uv=False)
            current_rank = jnp.sum(s_values > NONZERO_RANK_TOLERANCE)
            is_deficient = current_rank < jacobian.shape[1]
            jax.debug.print("Jacobian Rank: {rank}", rank=current_rank)
            jax.lax.cond(
                is_deficient,
                lambda: jax.debug.print("WARNING: System is locally ill-posed."),
                lambda: None,
            )
            return jacobian

        # If we want to check rank at each iteration, we can use this:
        # jit_jacobian_with_rank = jax.jit(get_jacobian_with_rank)

        result = least_squares(
            fun=jit_residuals,
            x0=initial_guess,
            jac=jit_jacobian,  # type: ignore
            method="trf",
            xtol=SOLVER_CONVERGENCE_TOLERANCE,
            ftol=SOLVER_CONVERGENCE_TOLERANCE,
            gtol=SOLVER_CONVERGENCE_TOLERANCE,
            verbose=2 if logger.isEnabledFor(logging.DEBUG) else 0,
        )

        # Run checks and update using the consolidated method from the base class.
        final_variable_values = self.fan_out_solved_variable_values(
            result, independent_vars, substitution_map
        )
        self.update_primitives_from_map(final_variable_values)
        self.assess_solver_result(final_variable_values, constraints)

        return
