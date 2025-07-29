"""
Notes:
- This method is currently significantly slower than the sparse version; I don't think
  this is actually because of dense vs. sparse, but because we're repeatedly
  recompiling the JAX functions because we're feeding it things with different
  shapes.
"""

import logging
from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

import newton.backend as nb
from newton.constants import NONZERO_RANK_TOLERANCE
from newton.constraints import BaseConstraint, Constraint
from newton.logging_config import logger
from newton.primitives import Point
from newton.solver_base import SOLVER_CONVERGENCE_TOLERANCE, Solver2D


class Solver2DDense(Solver2D):
    def __init__(self, points: List[Point], constraints: List[Constraint]):
        super().__init__(points, constraints)

        # Handle backend setup.
        nb.set_backend(nb.Backend.JAX)
        self.module = jnp

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

        def get_all_positions_jax(free_vars: jnp.ndarray) -> Dict[str, jnp.ndarray]:
            positions = {p.id: jnp.array([p.x, p.y]) for p in self.points}
            for i, p in enumerate(free_points):
                positions[p.id] = free_vars[i * 2 : i * 2 + 2]
            return positions

        def residuals_vector(free_vars: jnp.ndarray) -> jnp.ndarray:
            positions = get_all_positions_jax(free_vars)
            parts = [c.get_residual(positions) for c in constraints]
            return jnp.concatenate(parts)

        jit_residuals = jax.jit(residuals_vector)
        jit_jacobian = jax.jit(jax.jacfwd(residuals_vector))

        # Do rank based system state check.
        jacobian_init = jit_jacobian(initial_guess)
        n_equations = jacobian_init.shape[0]
        n_variables = jacobian_init.shape[1]

        self.check_system_state(jacobian_init, n_variables, n_equations)

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

        jit_jacobian_with_rank = jax.jit(get_jacobian_with_rank)

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
        self.update_points_from_result(result, free_points)

        return
