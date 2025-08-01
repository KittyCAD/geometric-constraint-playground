# Notes:
# - This method is currently significantly slower than the sparse version; I don't think
#   this is actually because of dense vs. sparse, but because we're repeatedly
#   recompiling the JAX functions because we're feeding it things with different
#   shapes.

import logging
from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

import newton.backend as nb
from newton.constants import NONZERO_RANK_TOLERANCE, REGULARIZATION_LAMBDA
from newton.constraints import Constraint
from newton.exceptions import UnsupportedPrimitiveError
from newton.logging_config import logger
from newton.primitives import Point, Primitive
from newton.solver_base import SOLVER_CONVERGENCE_TOLERANCE, Solver2D


class Solver2DDense(Solver2D):
    def __init__(self, primitives: List[Primitive], constraints: List[Constraint]):
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

        # Create an ordered list of all variables we can solve for.
        free_var_ids = [
            var_id
            for p in free_primitives
            for var_id in p.get_variable_ids()
            if var_id not in substitution_map
        ]

        # Build the initial guess array based on this ordered list.
        initial_values = []
        prim_map = {p.id: p for p in self.primitives}
        for var_id in free_var_ids:
            prim_id, var_type = var_id.split("_")
            p = prim_map[prim_id]

            # TODO: Handle other primitive types.
            if isinstance(p, Point):
                if var_type == "x":
                    initial_values.append(p.x)
                elif var_type == "y":
                    initial_values.append(p.y)
            else:
                raise UnsupportedPrimitiveError(type(p).__name__)

        initial_guess = np.array(initial_values)
        jnp_initial_guess = jnp.asarray(initial_guess)

        def build_inputs_for_constraints_jax(
            free_vars: jnp.ndarray,
        ) -> Dict[str, jnp.ndarray]:
            # This function provides the `positions` dict that get_residual expects.

            # Start with the initial state of all primitives.
            # TODO: Handle other primitive types.
            positions = {
                p.id: jnp.array([p.x, p.y])
                for p in self.primitives
                if isinstance(p, Point)
            }

            if any(not isinstance(p, Point) for p in free_primitives):
                raise NotImplementedError(
                    "Only Point primitives are currently supported."
                )

            # Create a dictionary of the solved variable values.
            solved_vars = {
                var_id: free_vars[i] for i, var_id in enumerate(free_var_ids)
            }

            # Overwrite the positions of free points with the new solved values.
            # We group variables by primitive ID.
            free_point_ids = {p.id for p in free_primitives if isinstance(p, Point)}
            for pid in free_point_ids:
                new_x = solved_vars.get(f"{pid}_x", positions[pid][0])
                new_y = solved_vars.get(f"{pid}_y", positions[pid][1])
                positions[pid] = jnp.array([new_x, new_y])

            return positions

        def residuals_vector(free_vars: jnp.ndarray) -> jnp.ndarray:
            positions = build_inputs_for_constraints_jax(free_vars)

            constraint_residuals = jnp.concatenate(
                [c.get_residual(positions) for c in constraints]
            )

            # Regularization residuals (lambda * (x - x_initial)).
            reg_residuals = REGULARIZATION_LAMBDA * (free_vars - jnp_initial_guess)

            # Combine into the new augmented residual vector.
            residuals = jnp.concatenate([constraint_residuals, reg_residuals])

            return residuals

        jit_residuals = jax.jit(residuals_vector)
        jit_jacobian = jax.jit(jax.jacfwd(residuals_vector))

        # Do rank based system state check.
        jacobian_init = jit_jacobian(jnp_initial_guess)

        if jacobian_init.shape is None:
            raise ValueError("Jacobian is empty or not properly initialized.")

        # Because of our Tikhonov regularization, we need to adjust the number of equations.
        # We effectively vertically concatenate the actual Jacobian with another matrix of the
        # size (REG_LAMBDA * I), which adds n_variables more rows.
        n_variables = jacobian_init.shape[1]
        # n_equations = jacobian_init.shape[0]
        n_geom_equations = sum(c.get_residual_dim() for c in constraints)

        self.check_system_state(jacobian_init, n_variables, n_geom_equations)

        # The per-iteration rank check is still useful for diagnostics.
        # Define a function that can be used for debugging if needed
        def get_jacobian_with_rank(free_vars: jnp.ndarray) -> jnp.ndarray:
            jacobian = jit_jacobian(free_vars)
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

        # Run checks and update.
        final_variable_values = self.compute_final_variable_values(
            result, free_primitives, substitution_map
        )
        self.update_primitives_from_map(final_variable_values)
        self.assess_solver_result(final_variable_values, constraints)

        return
